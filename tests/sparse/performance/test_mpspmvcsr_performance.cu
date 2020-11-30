/*
 *  Performance test for SpMV routines using the CSR matrix format
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
#include "../../../src/sparse/mpspmvcsr.cuh"
#include "../../../src/sparse/matrix_converter.cuh"
#include "../../sparse/performance/3rdparty.cuh"

//Execution configuration for mpspmvcsr
#define MPRES_CUDA_BLOCKS_FIELDS 512
#define MPRES_CUDA_THREADS_FIELDS 128
#define MPRES_CUDA_BLOCKS_RESIDUES 32768
#define MPRES_CUDA_THREADS_REDUCE 64

#define MATRIX_PATH "../../tests/sparse/matrices/t3dl.mtx"

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
// double precision
/////////
__global__ static void double_spmv_csr_kernel(const int num_rows, const double * data, const int * cols, const int * offsets,  const double * x, double * y) {
    unsigned int row = threadIdx.x + blockIdx.x * blockDim.x;
    if(row < num_rows){
        double dot = 0;
        int row_start = offsets[row];
        int row_end = offsets[row+1];
        for (int i = row_start; i < row_end; i++) {
            dot += data[i] * x[cols[i]];
        }
        y[row] = dot;
    }
}

void double_test(const int num_rows, const int num_cols, const int nnz, const double * data, const int * cols, const int * offsets, const mpfr_t * x) {
    InitCudaTimer();
    Logger::printDash();
    PrintTimerName("[GPU] double SpMV CSR");

    //Execution configuration
    int threads = 32;
    int blocks = num_rows / threads + 1;

    //host data
    auto *hx = new double[num_cols];
    auto *hy = new double[num_rows];

    //GPU data
    auto *ddata = new double[nnz];
    auto *dcols = new int[nnz];
    auto *doffsets = new int[(num_rows + 1)];
    auto *dx = new double[num_cols];
    auto *dy = new double[num_rows];

    cudaMalloc(&ddata, sizeof(double) * nnz);
    cudaMalloc(&dcols, sizeof(int) * nnz);
    cudaMalloc(&doffsets, sizeof(int) * (num_rows +1));
    cudaMalloc(&dx, sizeof(double) * num_cols);
    cudaMalloc(&dy, sizeof(double) * num_rows);

    // Convert from MPFR
    convert_vector(hx, x, num_cols);

    //Copying data to the GPU
    cudaMemcpy(ddata, data, sizeof(double) * nnz, cudaMemcpyHostToDevice);
    cudaMemcpy(dcols, cols, sizeof(int) * nnz, cudaMemcpyHostToDevice);
    cudaMemcpy(doffsets, offsets, sizeof(int) * (num_rows +1), cudaMemcpyHostToDevice);
    cudaMemcpy(dx, hx, sizeof(double) * num_cols, cudaMemcpyHostToDevice);

    //Launch
    StartCudaTimer();
    double_spmv_csr_kernel<<<blocks, threads>>>(num_rows, ddata, dcols, doffsets, dx, dy);
    EndCudaTimer();
    PrintCudaTimer("took");
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    //Copying to the host
    cudaMemcpy(hy, dy, sizeof(double) * num_rows , cudaMemcpyDeviceToHost);
    print_double_sum(hy, num_rows);

    delete [] hx;
    delete [] hy;
    cudaFree(ddata);
    cudaFree(dcols);
    cudaFree(doffsets);
    cudaFree(dx);
    cudaFree(dy);
}

/////////
// TACO
/////////
void taco_test(const mpfr_t * vectorX){
    using namespace taco;
    InitCpuTimer();
    Logger::printDash();
    PrintTimerName("[CPU] TACO spmv");

    Format csr({Dense,Sparse});
    Format  dv({Dense});

    Tensor<double> A = read(MATRIX_PATH, csr, false);

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
// MPRES-BLAS (structure of arrays)
/////////
void mpres_test(const int num_rows, const int num_cols, const int nnz, const double * data, const int * cols, const int * offsets, const mpfr_t * x, const mpfr_t * y) {
    Logger::printDash();
    InitCudaTimer();
    PrintTimerName("[GPU] MPRES-BLAS mpspmvcsr");

    //Host data
    auto hx = new mp_float_t[num_cols];
    auto hy = new mp_float_t[num_rows];
    auto hdata = new mp_float_t[nnz];

    //GPU data
    mp_array_t dx;
    mp_array_t dy;
    mp_collection_t ddata;
    mp_collection_t dbuf;
    int *dcols;
    int *doffsets;

    //Init data
    cuda::mp_array_init(dx, num_cols);
    cuda::mp_array_init(dy, num_rows);
    cuda::mp_collection_init(ddata, nnz);
    cuda::mp_collection_init(dbuf, nnz);
    cudaMalloc(&dcols, sizeof(int) * nnz);
    cudaMalloc(&doffsets, sizeof(int) * (num_rows + 1));

    // Convert from MPFR and double
    convert_vector(hx, x, num_cols);
    convert_vector(hdata, data, nnz);

    //Copying to the GPU
    cuda::mp_array_host2device(dx, hx, num_cols);
    cuda::mp_collection_host2device(ddata, hdata, nnz);
    cudaMemcpy(dcols, cols, sizeof(int) * nnz, cudaMemcpyHostToDevice);
    cudaMemcpy(doffsets, offsets, sizeof(int) * (num_rows + 1), cudaMemcpyHostToDevice);
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    //Launch
    StartCudaTimer();
    cuda::mpspmvcsr<
            MPRES_CUDA_BLOCKS_FIELDS,
            MPRES_CUDA_THREADS_FIELDS,
            MPRES_CUDA_BLOCKS_RESIDUES,
            MPRES_CUDA_THREADS_REDUCE>
            (num_rows, num_cols, nnz, dcols, doffsets, ddata, dx, dy, dbuf);
    EndCudaTimer();
    PrintCudaTimer("took");
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    //Copying to the host
    cuda::mp_array_device2host(hy, dy, num_rows);
    print_mp_sum(hy, num_rows);

    //Cleanup
    delete [] hx;
    delete [] hy;
    delete [] hdata;
    cuda::mp_array_clear(dx);
    cuda::mp_array_clear(dy);
    cuda::mp_collection_clear(ddata);
    cuda::mp_collection_clear(dbuf);
    cudaFree(dcols);
    cudaFree(doffsets);
}

/////////
// MPRES-BLAS straightforward (array of structures)
// Each multiple-precision operation is performed by a single thread
/////////
__global__ static void mpspmvcsr_basic_kernel(const int num_rows, mp_float_ptr data, const int * cols, const int * offsets,  mp_float_ptr x, mp_float_ptr y) {
    unsigned int row = threadIdx.x + blockIdx.x * blockDim.x;
    if (row < num_rows) {
        mp_float_t prod;
        mp_float_t dot = cuda::MP_ZERO;
        int row_start = offsets[row];
        int row_end = offsets[row+1];
        for (int i = row_start; i < row_end; i++) {
            cuda::mp_mul(&prod, &x[cols[i]], &data[i]);
            cuda::mp_add(&dot, &dot, &prod);
        }
        cuda::mp_set(&y[row], &dot);
    }
}

void mpres_test_naive(const int num_rows, const int num_cols, const int nnz, const double * data, const int * cols, const int * offsets, const mpfr_t * x, const mpfr_t * y){
    InitCudaTimer();
    Logger::printDash();
    PrintTimerName("[GPU] MPRES-BLAS mpspmvcsr (basic)");

    //Execution configuration
    int threads = 32;
    int blocks = num_rows / threads + 1;

    // Host data
    auto hx = new mp_float_t[num_cols];
    auto hy = new mp_float_t[num_rows];
    auto hdata = new mp_float_t[nnz];

    // GPU data
    mp_float_ptr dx;
    mp_float_ptr dy;
    mp_float_ptr ddata;
    int *dcols;
    int *doffsets;

    //Init data
    cudaMalloc(&dx, sizeof(mp_float_t) * num_cols);
    cudaMalloc(&dy, sizeof(mp_float_t) * num_rows);
    cudaMalloc(&ddata, sizeof(mp_float_t) * nnz);
    cudaMalloc(&dcols, sizeof(int) * nnz);
    cudaMalloc(&doffsets, sizeof(int) * (num_rows + 1));

    // Convert from MPFR
    convert_vector(hx, x, num_cols);
    convert_vector(hdata, data, nnz);

    //Copying to the GPU
    cudaMemcpy(dx, hx, num_cols * sizeof(mp_float_t), cudaMemcpyHostToDevice);
    cudaMemcpy(ddata, hdata, nnz * sizeof(mp_float_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dcols, cols, nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(doffsets, offsets, (num_rows + 1) * sizeof(int), cudaMemcpyHostToDevice);

    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    //Launch
    StartCudaTimer();
    mpspmvcsr_basic_kernel<<<blocks, threads>>>(num_rows, ddata, dcols, doffsets, dx, dy);
    EndCudaTimer();
    PrintCudaTimer("took");
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    //Copying to the host
    cudaMemcpy(hy, dy, num_rows * sizeof(mp_float_t), cudaMemcpyDeviceToHost);
    print_mp_sum(hy, num_rows);

    //Cleanup
    delete [] hx;
    delete [] hy;
    delete [] hdata;
    cudaFree(dx);
    cudaFree(dy);
    cudaFree(ddata);
    cudaFree(dcols);
    cudaFree(doffsets);
}

/********************* Main test *********************/

void test( int NUM_ROWS, int NUM_COLS, int NUM_LINES, bool MATRIX_SYMMETRIC, string DATATYPE) {

    int nnz = 0;
    if (MATRIX_SYMMETRIC){
        nnz = (NUM_LINES - NUM_ROWS) * 2 + NUM_ROWS;
    } else {
        nnz = NUM_LINES;
    }

    //Inputs
    mpfr_t *vectorX = create_random_array(NUM_COLS, INP_BITS);
    mpfr_t *vectorY = create_random_array(NUM_ROWS, INP_BITS);
    auto * data = new double [nnz]();
    auto * cols = new int[nnz]();
    auto * offsets = new int[NUM_ROWS + 1]();

    //Convert a sparse matrix to the double-precision CSR format
    convert_to_csr(MATRIX_PATH, NUM_ROWS, NUM_LINES, data, offsets, cols, MATRIX_SYMMETRIC);

    //Launch tests
    double_test(NUM_ROWS, NUM_COLS, nnz, data, cols, offsets, vectorX);
    if (DATATYPE == "real") {
        taco_test(vectorX);
    }
    //TODO main mpres test
    mpres_test(NUM_ROWS, NUM_COLS, nnz, data, cols, offsets, vectorX, vectorY);
    mpres_test_naive(NUM_ROWS, NUM_COLS, nnz, data, cols, offsets, vectorX, vectorY);
    campary_spmv_csr_test<CAMPARY_PRECISION>(NUM_ROWS, NUM_COLS, nnz, data, cols, offsets, vectorX, INP_DIGITS);
    cump_spmv_csr_test(NUM_ROWS, NUM_COLS, nnz, data, cols, offsets, vectorX, MP_PRECISION, INP_DIGITS);
    checkDeviceHasErrors(cudaDeviceSynchronize());
    // cudaCheckErrors(); //CUMP gives failure

    //Cleanup
    for(int i = 0; i < NUM_COLS; i++){
        mpfr_clear(vectorX[i]);
    }
    for(int i = 0; i < NUM_ROWS; i++){
        mpfr_clear(vectorY[i]);
    }
    delete[] vectorX;
    delete[] vectorY;
    delete[] data;
    delete[] cols;
    delete[] offsets;
    cudaDeviceReset();
}

int main() {
    //The operation parameters. Read from an input file that contains a sparse matrix
    int NUM_ROWS = 0; //number of rows
    int NUM_COLS = 0; //number of columns
    int NUM_LINES = 0; //number of lines in the input matrix file
    int COLS_PER_ROW = 0; //maximum number of nonzeros per row
    bool MATRIX_SYMMETRIC = false; //true if the input matrix is to be treated as symmetrical; otherwise false
    string DATATYPE; //defines type of data in MatrixMarket: real, integer, binary

    initialize();

    //Start logging
    Logger::beginTestDescription(Logger::BLAS_SPMV_ELL_PERFORMANCE_TEST);
    Logger::beginSection("Operation info:");
    Logger::printParam("Matrix path", MATRIX_PATH);
    read_matrix_properties(MATRIX_PATH, NUM_ROWS, NUM_COLS, NUM_LINES, COLS_PER_ROW, MATRIX_SYMMETRIC, DATATYPE);
    Logger::printParam("Matrix rows, NUM_ROWS", NUM_ROWS);
    Logger::printParam("Matrix columns, NUM_COLUMNS", NUM_COLS);
    Logger::printParam("Maximum nonzeros per row, COLS_PER_ROW", COLS_PER_ROW);
    Logger::printParam("Symmetry, MATRIX_SYMMETRIC", MATRIX_SYMMETRIC);
    Logger::printParam("Data type, DATATYPE", DATATYPE);
    Logger::printDash();
    Logger::beginSection("Additional info:");
    Logger::printParam("MP_PRECISION", MP_PRECISION);
    Logger::printParam("RNS_MODULI_SIZE", RNS_MODULI_SIZE);
    Logger::printParam("MPRES_CUDA_BLOCKS_FIELDS", MPRES_CUDA_BLOCKS_FIELDS);
    Logger::printParam("MPRES_CUDA_THREADS_FIELDS", MPRES_CUDA_THREADS_FIELDS);
    Logger::printParam("MPRES_CUDA_BLOCKS_RESIDUES", MPRES_CUDA_BLOCKS_RESIDUES);
    Logger::printParam("MPRES_CUDA_THREADS_REDUCE", MPRES_CUDA_THREADS_REDUCE);
    Logger::printParam("CAMPARY_PRECISION (n-double)", CAMPARY_PRECISION);
    Logger::endSection(true);

    //Run the test
    test(NUM_ROWS, NUM_COLS, NUM_LINES, MATRIX_SYMMETRIC, DATATYPE);

    //Finalize
    finalize();

    //End logging
    Logger::endTestDescription();

    return 0;
}