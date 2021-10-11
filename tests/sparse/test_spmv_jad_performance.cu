/*
 *  Performance test for SpMV routines using the JAD (JDS) matrix format (double precision matrix)
 *  Path to the matrix must be given as a command line argument, e.g., ../../tests/sparse/matrices/t3dl.mtx

 *  Copyright 2020 by Konstantin Isupov and Ivan Babeshko.
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

#include "logger.cuh"
#include "tsthelper.cuh"
#include "sparse/utils/mtx_reader.cuh"
#include "sparse/utils/jad_utils.cuh"
#include "sparse/jad/test_double_spmv_jad.cuh"
#include "sparse/jad/test_mpres_spmv_jad.cuh"
#include "sparse/jad/test_mpres_spmv_jadv.cuh"
#include "sparse/jad/test_campary_spmv_jad.cuh"
#include "sparse/jad/test_cump_spmv_jad.cuh"
#include "sparse/csr/test_taco_spmv_csr.cuh"

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

/*
 * Multiple-precision JAD kernel without using shared memory
 */
__global__ void mp_spmv_jad_smem_free(const int m, const int maxnzr, const jad_t jad, mp_float_ptr x, mp_float_ptr y) {
    auto row = threadIdx.x + blockIdx.x * blockDim.x;
    mp_float_t sum;
    mp_float_t prod;
    while (row < m) {
        auto j = 0;
        auto index = row;
        sum = cuda::MP_ZERO;
        while (j < maxnzr && index < jad.jcp[j + 1]) {
            cuda::mp_mul_d(&prod, x[jad.ja[index]], jad.as[index]);
            cuda::mp_add(&sum, sum, prod);
            index = row + jad.jcp[++j];
        }
        cuda::mp_set(&y[jad.perm[row]], sum);
        row +=  gridDim.x * blockDim.x;
    }
}

/*
 * MPRES-BLAS test, no shared memory
 */
void test_mpres_spmv_jad_smem_free(const int m, const int n, const int maxnzr, const int nnz, const jad_t &jad, const mpfr_t *x) {
    InitCudaTimer();
    Logger::printDash();
    PrintTimerName("[GPU] MPRES-BLAS JAD (w/o shared memory)");

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
    jad_t djad;
    cuda::jad_init(djad, m, maxnzr, nnz);
    cuda::jad_host2device(djad, jad, m, maxnzr, nnz);

    //Launch
    StartCudaTimer();
    mp_spmv_jad_smem_free<<<blocks, threads>>>(m, maxnzr, djad, dx, dy);
    EndCudaTimer();
    PrintCudaTimer("took");
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    //Copying to the host
    cudaMemcpy(hy, dy, m * sizeof(mp_float_t), cudaMemcpyDeviceToHost);
    print_mp_sum(hy, m);

    //Cleanup
    delete[] hx;
    delete[] hy;
    cudaFree(dx);
    cudaFree(dy);
    cuda::jad_clear(djad);

}

void test(const char * MATRIX_PATH, const int M, const int N, const int LINES, const int MAXNZR, const int NNZ, const bool SYMM, const string DATATYPE) {

    //Input arrays
    mpfr_t *vectorX = create_random_array(N, INP_BITS);
    jad_t JAD;
    jad_init(JAD, M, MAXNZR, NNZ);
    //Convert a sparse matrix to the double-precision JAD (JDS) format
    build_jad(MATRIX_PATH, M, MAXNZR, NNZ, LINES, SYMM, JAD);

    //Launch tests
    test_double_spmv_jad(M, N, MAXNZR, NNZ, JAD, vectorX);
    //test_taco_spmv_csr(MATRIX_PATH, vectorX, DATATYPE);
    test_mpres_spmv_jad(M, N, MAXNZR, NNZ, JAD, vectorX);
    test_mpres_spmv_jad_smem_free(M, N, MAXNZR, NNZ, JAD, vectorX);
    test_mpres_spmv_jadv<2>(M, N, MAXNZR, NNZ, JAD, vectorX);
    test_mpres_spmv_jadv<4>(M, N, MAXNZR, NNZ, JAD, vectorX);
    test_mpres_spmv_jadv<8>(M, N, MAXNZR, NNZ, JAD, vectorX);
    test_mpres_spmv_jadv<16>(M, N, MAXNZR, NNZ, JAD, vectorX);
    test_mpres_spmv_jadv<32>(M, N, MAXNZR, NNZ, JAD, vectorX);
    test_campary_spmv_jad<CAMPARY_PRECISION>(M, N, MAXNZR, NNZ, JAD, vectorX, INP_DIGITS);
    test_cump_spmv_jad(M, N, MAXNZR, NNZ, JAD, vectorX, MP_PRECISION, INP_DIGITS);
    checkDeviceHasErrors(cudaDeviceSynchronize());
    // cudaCheckErrors(); //CUMP gives failure

    //Cleanup
    for(int i = 0; i < N; i++){
        mpfr_clear(vectorX[i]);
    }
    delete[] vectorX;
    jad_clear(JAD);
    cudaDeviceReset();
}

int main(int argc, char *argv[]) {

    //The operation parameters. Read from an input file that contains a sparse matrix
    int M = 0; //number of rows
    int N = 0; //number of columns
    int NNZ = 0; //number of nonzeros in matrix
    int MAXNZR = 0; //Maximum number of nonzeros per row in the matrix A
    int NZMD = 0; //number of nonzeros in main diagonal of the matrix
    int LINES = 0; //number of lines in the input matrix file
    bool SYMM = false; //true if the input matrix is to be treated as symmetrical; otherwise false
    string DATATYPE; //defines type of data in MatrixMarket: real, integer, binary

    initialize();

    //Start logging
    Logger::beginTestDescription(Logger::SPMV_JAD_TEST);
    if(argc<=1) {
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
    Logger::printParam("Symmetry, SYMM", SYMM);
    Logger::printParam("Data type, DATATYPE", DATATYPE);
    Logger::printDash();
    Logger::beginSection("Additional info:");
    Logger::printParam("RNS_MODULI_SIZE", RNS_MODULI_SIZE);
    Logger::printParam("MP_PRECISION", MP_PRECISION);
    Logger::printParam("CAMPARY_PRECISION (n-double)", CAMPARY_PRECISION);
    Logger::endSection(true);

    //Run the test
    test(MATRIX_PATH, M, N, LINES, MAXNZR, NNZ, SYMM, DATATYPE);

    //Finalize
    finalize();

    //End logging
    Logger::endTestDescription();

    return 0;
}