/*
 *  Performance test for SpMV vs GEMV
 *
 *  Copyright 2021 by Konstantin Isupov.
 *
 *  This file is part of the MPRES-BLAS library.
 *
 *  MPRES-BLAS is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  MPRES-BLAS is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with MPRES-BLAS.  If not, see <https://www.gnu.org/licenses/>.
 */

#include "mpfr.h"
#include "logger.cuh"
#include "sparse/utils/mtx_reader.cuh"
#include "sparse/utils/csr_utils.cuh"
#include "sparse/utils/jad_utils.cuh"
#include "sparse/utils/ell_utils.cuh"
#include "sparse/csr/test_double_spmv_csr.cuh"
#include "sparse/csr/test_mpfr_spmv_csr.cuh"
#include "sparse/csr/test_mpres_spmv_csr.cuh"
#include "sparse/csr/test_mpres_spmv_csrv.cuh"
#include "sparse/jad/test_mpres_spmv_jad.cuh"
#include "sparse/ell/test_mpres_spmv_ell.cuh"

int INP_BITS; //in bits
int INP_DIGITS; //in decimal digits

void setPrecisions() {
    INP_BITS = (int) (MP_PRECISION / 4);
    INP_DIGITS = (int) (INP_BITS / 3.32 + 1);
}

void initialize() {
    cudaDeviceReset();
    rns_const_init();
    mp_const_init();
    setPrecisions();
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();
}

void finalize() {
}

// Regular multiple-precision GEMV, y = Ax
__global__ static void gemv_kernel(int m, int n, double * A, mp_float_ptr x, mp_float_ptr y) {
    unsigned int threadId = threadIdx.x + blockIdx.x * blockDim.x;
    if (threadId < m) {
        mp_float_t prod; //ПОСМОТРЕТЬ КАК БУДЕТ БЫСТРЕЕ - может в smem
        mp_float_t dot = cuda::MP_ZERO;
        for (int colId = 0; colId < n; colId++) {
            cuda::mp_mul_d(&prod, x[colId], A[colId * m + threadId]);
            cuda::mp_add(&dot, dot, prod);
        }
        cuda::mp_set(&y[threadId], dot);
    }
}

void test_gemv_kernel(const int m, const int n, const int nnz, double * A, const mpfr_t * x){
    InitCudaTimer();
    Logger::printDash();
    PrintTimerName("[GPU] MPRES-BLAS GEMV");

    //Execution configuration
    int threads = 32;
    int blocks = m / threads + 1;
    printf("\tExec. config: blocks = %i, threads = %i\n", blocks, threads);

    // Host data
    auto hx = new mp_float_t[n];
    auto hy = new mp_float_t[m];

    // GPU vectors
    mp_float_ptr dx;
    mp_float_ptr dy;
    cudaMalloc(&dx, sizeof(mp_float_t) * n);
    cudaMalloc(&dy, sizeof(mp_float_t) * m);
    convert_vector(hx, x, n);
    cudaMemcpy(dx, hx, n * sizeof(mp_float_t), cudaMemcpyHostToDevice);

    //GPU matrix
    double * dA;
    cudaMalloc(&dA, sizeof(double ) * nnz);
    cudaMemcpy(dA, A, nnz * sizeof(double), cudaMemcpyHostToDevice);

    //Launch
    StartCudaTimer();
    gemv_kernel<<<blocks, threads>>>(m, n, dA, dx, dy);
    EndCudaTimer();
    PrintCudaTimer("took");
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    //Copying to the host
    cudaMemcpy(hy, dy, m * sizeof(mp_float_t), cudaMemcpyDeviceToHost);
    print_mp_sum(hy, m);

    //Cleanup
    delete [] hx;
    delete [] hy;
    cudaFree(dx);
    cudaFree(dy);
    cudaFree(dA);
}

void test(const char * MATRIX_PATH, const int M, const int N, const int LINES, const int NNZ, const int MAXNZR, const bool SYMM, const string DATATYPE) {
    //Input vector
    mpfr_t *vectorX = create_random_array(N, INP_BITS);

    //Matrix storage structures
    csr_t CSR;
    jad_t JAD;
    ell_t ELL;
    double * dense = new double [NNZ];


    /*
     * Init matrix structures
     */
    csr_init(CSR,M,NNZ);
    jad_init(JAD, M, MAXNZR, NNZ);
    ell_init(ELL, M, MAXNZR);
    build_csr(MATRIX_PATH, M, NNZ, LINES, SYMM, CSR);
    build_jad(MATRIX_PATH, M, MAXNZR, NNZ, LINES, SYMM, JAD);
    build_ell(MATRIX_PATH, M, MAXNZR, LINES, SYMM, ELL);
    build_dense(MATRIX_PATH, M, N, LINES, SYMM, dense);

    /*
     * Double and MPFR CSR tests
     */
    test_double_spmv_csr(M, N, NNZ, CSR, vectorX);
    test_mpfr_spmv_csr(M, N, NNZ, CSR, vectorX, MP_PRECISION);

    /*
     * MPRES-BLAS tests
     */

    test_mpres_spmv_csr(M, N, NNZ, CSR, vectorX);
    test_mpres_spmv_csrv<4>(M, N, NNZ, CSR, vectorX);
    test_mpres_spmv_csrv<16>(M, N, NNZ, CSR, vectorX);
    test_mpres_spmv_csrv<32>(M, N, NNZ, CSR, vectorX);
    test_mpres_spmv_jad(M, N, MAXNZR, NNZ, JAD, vectorX);
    test_mpres_spmv_ell(M, N, MAXNZR, ELL, vectorX);
    test_gemv_kernel(M, N, NNZ, dense, vectorX);

    //Cleanup
    for(int i = 0; i < N; i++){
        mpfr_clear(vectorX[i]);
    }
    delete[] vectorX;
    csr_clear(CSR);
    jad_clear(JAD);
    ell_clear(ELL);
    delete [] dense;

}

int main(int argc, char *argv[]) {

    //The operation parameters. Read from an input file that contains a sparse matrix
    int M = 0; //number of rows
    int N = 0; //number of columns
    int NNZ = 0; //number of nonzeros in matrix
    int MAXNZR = 0; //maximum number of nonzeros per row in the matrix A
    int NZMD = 0; //number of nonzeros in the main diagonal of the matrix
    int LINES = 0; //number of lines in the input matrix file
    bool SYMM = false; //true if the input matrix is to be treated as symmetrical; otherwise false
    string DATATYPE; //defines type of data in MatrixMarket: real, integer, binary

    initialize();

    //Start logging
    Logger::beginTestDescription(Logger::SPMV_VS_GEMV_TEST);
    if(argc< 2) {
        printf("Matrix is not specified in command line arguments.");
        Logger::printSpace();
        Logger::endTestDescription();
        exit(1);
    }
    const char * MATRIX_PATH = argv[1];
    Logger::beginSection("Matrix properties:");
    Logger::printParam("Path", MATRIX_PATH);
    collect_mtx_stats(MATRIX_PATH, M, N, LINES, MAXNZR, NZMD, SYMM, DATATYPE);
    NNZ = SYMM ? ( (LINES - NZMD) * 2 + NZMD) : LINES;
    Logger::printParam("Number of rows, M", M);
    Logger::printParam("Number of column, N", N);
    Logger::printParam("Number of nonzeros, NNZ", NNZ);
    Logger::printParam("Maximum number of nonzeros per row, MAXNZR", MAXNZR);
    Logger::printParam("Average number of nonzeros per row, AVGNZR", (double)NNZ/M);
    Logger::printParam("Symmetry", SYMM);
    Logger::printParam("Data type", DATATYPE);
    Logger::printDash();
    Logger::beginSection("Additional info:");
    Logger::printParam("RNS_MODULI_SIZE", RNS_MODULI_SIZE);
    Logger::printParam("MP_PRECISION", MP_PRECISION);
    Logger::endSection(true);

    test(MATRIX_PATH, M, N, LINES, NNZ, MAXNZR, SYMM, DATATYPE);

    //Finalize
    finalize();

    //End logging
    Logger::endTestDescription();

    return 0;
}