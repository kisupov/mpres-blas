/*
 *  Performance test for SpMV routines using the CSR matrix format
 *  Path to the matrix must be given as a command line argument, e.g., ../../tests/sparse/matrices/t3dl.mtx
 *
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

#include "omp.h"
#include "../../logger.cuh"
#include "../../timers.cuh"
#include "../../tsthelper.cuh"
#include "../../../src/mparray.cuh"
#include "../../../src/mpcollection.cuh"
#include "../../../src/sparse/mpspmvcsr1.cuh"
#include "../../../src/sparse/mpspmvcsr2.cuh"
#include "../../../src/sparse/matrix_converter.cuh"
#include "3rdparty.cuh"

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

void convert_vector(double * dest, const mpfr_t *source, int width){
    #pragma omp parallel for
    for( int i = 0; i < width; i++ ){
        dest[i] = mpfr_get_d(source[i], MPFR_RNDN);
    }
}

void convert_vector(mp_float_ptr dest, const mpfr_t *source, int width){
    #pragma omp parallel for
    for( int i = 0; i < width; i++ ){
        mp_set_mpfr(&dest[i], source[i]);
    }
}

void convert_vector(mp_float_ptr dest, const double *source, int width){
    #pragma omp parallel for
    for( int i = 0; i < width; i++ ){
        mp_set_d(&dest[i], source[i]);
    }
}

/********************* SpMV CSR implementations and benchmarks *********************/

/////////
// Double precision
/////////
__global__ static void double_spmv_csr_kernel(const int m, const int *irp, const int *ja, const double *as, const double *x, double *y) {
    unsigned int row = threadIdx.x + blockIdx.x * blockDim.x;
    if(row < m){
        double dot = 0;
        int row_start = irp[row];
        int row_end = irp[row+1];
        for (int i = row_start; i < row_end; i++) {
            dot += as[i] * x[ja[i]];
        }
        y[row] = dot;
    }
}

void double_test(const int m, const int n, const int nnz, const int *irp, const int *ja, const double *as, const mpfr_t *x) {
    InitCudaTimer();
    Logger::printDash();
    PrintTimerName("[GPU] double SpMV CSR");

    //Execution configuration
    int threads = 32;
    int blocks = m / threads + 1;
    printf("(exec. config: blocks = %i, threads = %i)\n", blocks, threads);

    //host data
    auto *hx = new double[n];
    auto *hy = new double[m];

    //GPU data
    auto *das = new double[nnz];
    auto *dirp = new int[(m + 1)];
    auto *dja = new int[nnz];
    auto *dx = new double[n];
    auto *dy = new double[m];

    cudaMalloc(&das, sizeof(double) * nnz);
    cudaMalloc(&dirp, sizeof(int) * (m +1));
    cudaMalloc(&dja, sizeof(int) * nnz);
    cudaMalloc(&dx, sizeof(double) * n);
    cudaMalloc(&dy, sizeof(double) * m);

    // Convert from MPFR
    convert_vector(hx, x, n);

    //Copying data to the GPU
    cudaMemcpy(das, as, sizeof(double) * nnz, cudaMemcpyHostToDevice);
    cudaMemcpy(dja, ja, sizeof(int) * nnz, cudaMemcpyHostToDevice);
    cudaMemcpy(dirp, irp, sizeof(int) * (m +1), cudaMemcpyHostToDevice);
    cudaMemcpy(dx, hx, sizeof(double) * n, cudaMemcpyHostToDevice);

    //Launch
    StartCudaTimer();
    double_spmv_csr_kernel<<<blocks, threads>>>(m, dirp, dja, das, dx, dy);
    EndCudaTimer();
    PrintCudaTimer("took");
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    //Copying to the host
    cudaMemcpy(hy, dy, sizeof(double) * m , cudaMemcpyDeviceToHost);
    print_double_sum(hy, m);

    delete [] hx;
    delete [] hy;
    cudaFree(das);
    cudaFree(dja);
    cudaFree(dirp);
    cudaFree(dx);
    cudaFree(dy);
}

/////////
// TACO
/////////
void taco_test(const char * matrix_path, const mpfr_t * vectorX){
    using namespace taco;
    InitCpuTimer();
    Logger::printDash();
    PrintTimerName("[CPU] TACO spmv");

    Format csr({Dense,Sparse});
    Format  dv({Dense});

    Tensor<double> A = read(matrix_path, csr, false);

    Tensor<double> x({A.getDimension(1)}, dv);
    for (int i = 0; i < x.getDimension(0); ++i) {
        x.insert({i}, mpfr_get_d(vectorX[i], MPFR_RNDN));
    }
    x.pack();
    Tensor<double> result({A.getDimension(0)}, dv);

    IndexVar i, j;
    result(i) = (A(i,j) * x(j));

    StartCpuTimer();
    result.compile();
    result.assemble();
    result.compute();
    EndCpuTimer();

    double sum = 0.0;
    for (int k = 0; k < result.getDimension(0); k++) {
        sum += result(k);
    }
    PrintCpuTimer("took");
    printf("result: %.70f\n", sum);
}

/////////
//  First SpMV CSR implementation
/////////
void mpres_test_1(const int m, const int n, const int nnz, const int *irp, const int *ja, const double *as,  const mpfr_t * x){
    InitCudaTimer();
    Logger::printDash();
    PrintTimerName("[GPU] MPRES-BLAS mpspmv_csr1");

    //Execution configuration
    int threads = 32;
    int blocks = m / threads + 1;
    printf("(exec. config: blocks = %i, threads = %i)\n", blocks, threads);
    printf("Matrix size (MB): %lf\n", double(sizeof(mp_float_t)) * nnz /  double(1024 * 1024));

    // Host data
    auto hx = new mp_float_t[n];
    auto hy = new mp_float_t[m];
    auto has = new mp_float_t[nnz];

    // GPU data
    mp_float_ptr dx;
    mp_float_ptr dy;
    mp_float_ptr das;
    int *dirp;
    int *dja;

    //Init data
    cudaMalloc(&dx, sizeof(mp_float_t) * n);
    cudaMalloc(&dy, sizeof(mp_float_t) * m);
    cudaMalloc(&das, sizeof(mp_float_t) * nnz);
    cudaMalloc(&dirp, sizeof(int) * (m + 1));
    cudaMalloc(&dja, sizeof(int) * nnz);

    // Convert from MPFR
    convert_vector(hx, x, n);
    convert_vector(has, as, nnz);

    //Copying to the GPU
    cudaMemcpy(dx, hx, n * sizeof(mp_float_t), cudaMemcpyHostToDevice);
    cudaMemcpy(das, has, nnz * sizeof(mp_float_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dirp, irp, (m + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dja, ja, nnz * sizeof(int), cudaMemcpyHostToDevice);

    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    //Launch
    StartCudaTimer();
    cuda::mpspmv_csr1<<<blocks, threads>>>(m, dirp, dja, das, dx, dy);
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
    delete [] has;
    cudaFree(dx);
    cudaFree(dy);
    cudaFree(das);
    cudaFree(dirp);
    cudaFree(dja);
}

/////////
// MPRES-BLAS Second SpMV CSR implementation
/////////
void mpres_test_2(const int m, const int n, const int nnz, const int *irp, const int *ja, const double *as,  const mpfr_t * x){
    Logger::printDash();
    InitCudaTimer();
    PrintTimerName("[GPU] MPRES-BLAS mpspmv_csr2");

    //Execution configuration
    const int gridDim1 = 512;   //blocks for fields
    const int blockDim1 = 128;  //threads for fields
    const int gridDim2 = 32768; //blocks for residues
    const int blockDim3 = 64; //threads for reduce
    printf("(exec. config: gridDim1 = %i, blockDim1 = %i, gridDim2 = %i, blockDim3 = %i)\n", gridDim1, blockDim1, gridDim2, blockDim3);
    printf("Matrix size (MB): %lf\n", double(sizeof(mp_float_t)) * nnz /  double(1024 * 1024));

    //Host data
    auto hx = new mp_float_t[n];
    auto hy = new mp_float_t[m];
    auto has = new mp_float_t[nnz];

    //GPU data
    mp_array_t dx;
    mp_array_t dy;
    mp_collection_t das;
    mp_collection_t dbuf;
    int *dirp;
    int *dja;

    //Init data
    cuda::mp_array_init(dx, n);
    cuda::mp_array_init(dy, m);
    cuda::mp_collection_init(das, nnz);
    cuda::mp_collection_init(dbuf, nnz);
    cudaMalloc(&dirp, sizeof(int) * (m + 1));
    cudaMalloc(&dja, sizeof(int) * nnz);

    // Convert from MPFR and double
    convert_vector(hx, x, n);
    convert_vector(has, as, nnz);

    //Copying to the GPU
    cuda::mp_array_host2device(dx, hx, n);
    cuda::mp_collection_host2device(das, has, nnz);
    cudaMemcpy(dirp, irp, sizeof(int) * (m + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(dja, ja, sizeof(int) * nnz, cudaMemcpyHostToDevice);
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    //Launch
    StartCudaTimer();
    cuda::mpspmv_csr2<gridDim1, blockDim1, gridDim2, blockDim3> (m, n, nnz, dirp, dja, das, dx, dy, dbuf);
    EndCudaTimer();
    PrintCudaTimer("took");
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    //Copying to the host
    cuda::mp_array_device2host(hy, dy, m);
    print_mp_sum(hy, m);

    //Cleanup
    delete [] hx;
    delete [] hy;
    delete [] has;
    cuda::mp_array_clear(dx);
    cuda::mp_array_clear(dy);
    cuda::mp_collection_clear(das);
    cuda::mp_collection_clear(dbuf);
    cudaFree(dirp);
    cudaFree(dja);
}

/********************* Main test *********************/

void test(const char * MATRIX_PATH, const int M, const int N, const int LINES, const int NNZ, const bool SYMM, const string DATATYPE) {

    //Inputs

    mpfr_t *vectorX = create_random_array(N, INP_BITS);
    auto *AS = new double [NNZ]();
    auto *JA = new int[NNZ]();
    auto *IRP = new int[M + 1]();

    //Convert a sparse matrix to the double-precision CSR format
    convert_to_csr(MATRIX_PATH, M, NNZ, LINES, SYMM, AS, IRP, JA);

    //Launch tests
    double_test(M, N, NNZ, IRP, JA, AS, vectorX);
    if (DATATYPE == "real") {
        taco_test(MATRIX_PATH, vectorX);
    }
    mpres_test_1(M, N, NNZ, IRP, JA, AS, vectorX);
    mpres_test_2(M, N, NNZ, IRP, JA, AS, vectorX);
    campary_spmv_csr_test<CAMPARY_PRECISION>(M, N, NNZ, IRP, JA, AS, vectorX, INP_DIGITS);
    cump_spmv_csr_test(M, N, NNZ, IRP, JA, AS, vectorX, MP_PRECISION, INP_DIGITS);
    checkDeviceHasErrors(cudaDeviceSynchronize());
    // cudaCheckErrors(); //CUMP gives failure

    //Cleanup
    for(int i = 0; i < N; i++){
        mpfr_clear(vectorX[i]);
    }
    delete[] vectorX;
    delete[] AS;
    delete[] JA;
    delete[] IRP;
    cudaDeviceReset();
}

int main(int argc, char *argv[]) {

    //The operation parameters. Read from an input file that contains a sparse matrix
    int M = 0; //number of rows
    int N = 0; //number of columns
    int NNZ = 0; //number of nonzeros in matrix
    int NZR = 0; //number of nonzeros per row array (maximum number of nonzeros per row in the matrix A)
    int LINES = 0; //number of lines in the input matrix file
    bool SYMM = false; //true if the input matrix is to be treated as symmetrical; otherwise false
    string DATATYPE; //defines type of data in MatrixMarket: real, integer, binary

    initialize();

    //Start logging
    Logger::beginTestDescription(Logger::BLAS_SPMV_CSR_PERFORMANCE_TEST);
    if(argc<=1) {
        printf("Matrix is not specified in command line arguments.");
        Logger::printSpace();
        Logger::endTestDescription();
        exit(1);
    }
    const char * MATRIX_PATH = argv[1];

    Logger::beginSection("Operation info:");
    Logger::printParam("Matrix path", MATRIX_PATH);
    read_matrix_properties(MATRIX_PATH, M, N, LINES, NZR, SYMM, DATATYPE);
    NNZ = SYMM ? ( (LINES - M) * 2 + M) : LINES;
    Logger::printParam("Number of rows in matrix, M", M);
    Logger::printParam("Number of column in matrix, N", N);
    Logger::printParam("Number of nonzeros in matrix, NNZ", NNZ);
    Logger::printParam("Number of nonzeros per row array, NZR", NZR);
    Logger::printParam("Symmetry of matrix, SYMM", SYMM);
    Logger::printParam("Data type, DATATYPE", DATATYPE);
    Logger::printDash();
    Logger::beginSection("Additional info:");
    Logger::printParam("RNS_MODULI_SIZE", RNS_MODULI_SIZE);
    Logger::printParam("MP_PRECISION", MP_PRECISION);
    Logger::printParam("CAMPARY_PRECISION (n-double)", CAMPARY_PRECISION);
    Logger::endSection(true);

    //Run the test
    test(MATRIX_PATH, M, N, LINES, NNZ, SYMM, DATATYPE);

    //Finalize
    finalize();

    //End logging
    Logger::endTestDescription();

    return 0;
}