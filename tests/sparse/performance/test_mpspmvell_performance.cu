/*
 *  Performance test for SpMV routines using the ELLPACK matrix format
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
#include "../../../src/sparse/mpspmvell.cuh"
#include "../../../src/sparse/matrix_converter.cuh"
#include "../../sparse/performance/3rdparty.cuh"

//Execution configuration for mpspmvell
#define MPRES_CUDA_THREADS_SCALAR_KERNELS 32
#define MPRES_CUDA_BLOCKS_RESIDUES 256

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

/********************* SpMV ELLPACK implementations and benchmarks *********************/

/////////
// double precision
/////////
__global__ static void double_spmv_ell_kernel(const int num_rows, const int cols_per_row, const int * indices, const double * data, const double * x, double * y) {
    unsigned int row = threadIdx.x + blockIdx.x * blockDim.x;
    if(row < num_rows){
        double dot = 0;
        for (int col = 0; col < cols_per_row; col++) {
            int index = indices[col * num_rows + row];
            if(index != -1){
                dot += data[col * num_rows + row] * x[index];
            }
        }
        y[row] += dot;
    }
}

void double_test(const int num_rows, const int num_cols, const int cols_per_row, const double * data, const int * indices, const mpfr_t * x, const mpfr_t * y) {
    InitCudaTimer();
    Logger::printDash();
    PrintTimerName("[GPU] double SpMV ELLPACK");

    //Execution configuration
    int threads = 32;
    int blocks = num_rows / threads + 1;

    //host data
    auto *hx = new double[num_cols];
    auto *hy = new double[num_rows];

    //GPU data
    auto *ddata = new double[num_rows * cols_per_row];
    auto *dindices = new int[num_rows * cols_per_row];
    auto *dx = new double[num_cols];
    auto *dy = new double[num_rows];

    cudaMalloc(&ddata, sizeof(double) * num_rows * cols_per_row);
    cudaMalloc(&dindices, sizeof(int) * num_rows * cols_per_row);
    cudaMalloc(&dx, sizeof(double) * num_cols);
    cudaMalloc(&dy, sizeof(double) * num_rows);

    // Convert from MPFR
    convert_vector(hx, x, num_cols);
    convert_vector(hy, y, num_rows);

    //Copying data to the GPU
    cudaMemcpy(ddata, data, sizeof(double) * num_rows * cols_per_row, cudaMemcpyHostToDevice);
    cudaMemcpy(dindices, indices, sizeof(int) * num_rows * cols_per_row, cudaMemcpyHostToDevice);
    cudaMemcpy(dx, hx, sizeof(double) * num_cols, cudaMemcpyHostToDevice);
    cudaMemcpy(dy, hy, sizeof(double) * num_rows, cudaMemcpyHostToDevice);

    //Launch
    StartCudaTimer();
    double_spmv_ell_kernel<<<blocks, threads>>>(num_rows, cols_per_row, dindices, ddata, dx, dy);
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
    cudaFree(dindices);
    cudaFree(dx);
    cudaFree(dy);
}

/////////
// MPRES-BLAS (structure of arrays)
/////////
void mpres_test(const int num_rows, const int num_cols, const int cols_per_row, const double * data, const int * indices, const mpfr_t * x,  const mpfr_t * y) {
    Logger::printDash();
    InitCudaTimer();
    PrintTimerName("[GPU] MPRES-BLAS mpspmvell");

    size_t matrix_len = num_rows * cols_per_row;
    //Host data
    auto hx = new mp_float_t[num_cols];
    auto hy = new mp_float_t[num_rows];
    auto hdata = new mp_float_t[matrix_len];

    //GPU data
    mp_array_t dx;
    mp_array_t dy;
    mp_array_t ddata;
    mp_array_t dbuf;
    int *dindices;

    //Init data
    cuda::mp_array_init(dx, num_cols);
    cuda::mp_array_init(dy, num_rows);
    cuda::mp_array_init(ddata, matrix_len);
    cuda::mp_array_init(dbuf, matrix_len);
    cudaMalloc(&dindices, sizeof(int) * matrix_len);

    // Convert from MPFR and double
    convert_vector(hx, x, num_cols);
    convert_vector(hy, y, num_rows);
    convert_vector(hdata, data, matrix_len);

    //Copying to the GPU
    cuda::mp_array_host2device(dx, hx, num_cols);
    cuda::mp_array_host2device(dy, hy, num_rows);
    cuda::mp_array_host2device(ddata, hdata, matrix_len);
    cudaMemcpy(dindices, indices, sizeof(int) * matrix_len, cudaMemcpyHostToDevice);
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    //Launch
    StartCudaTimer();
    cuda::mpspmvell<
            MPRES_CUDA_THREADS_SCALAR_KERNELS,
            MPRES_CUDA_BLOCKS_RESIDUES>
            (num_rows, cols_per_row, dindices, ddata, dx, dy, dbuf);
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
    cuda::mp_array_clear(ddata);
    cuda::mp_array_clear(dbuf);
    cudaFree(dindices);
}

/////////
// MPRES-BLAS straightforward (array of structures)
// Each multiple-precision operation is performed by a single thread
/////////
__global__ static void mpspmvell_naive_kernel(const int num_rows, const int cols_per_row, const int * indices, mp_float_ptr data, mp_float_ptr x, mp_float_ptr y) {
    unsigned int row = threadIdx.x + blockIdx.x * blockDim.x;
    if (row < num_rows) {
        mp_float_t prod;
        mp_float_t dot = cuda::MP_ZERO;
        for (int col = 0; col < cols_per_row; col++) {
            int index = indices[col * num_rows + row];
            if(index >= 0){
                cuda::mp_mul(&prod, &x[index], &data[col * num_rows + row]);
                cuda::mp_add(&dot, &dot, &prod);
            }
        }
        cuda::mp_add(&y[row], &y[row], &dot);
    }
}

void mpres_test_naive(const int num_rows, const int num_cols, const int cols_per_row, const double * data, int * indices, const mpfr_t * x, const mpfr_t * y){
    InitCudaTimer();
    Logger::printDash();
    PrintTimerName("[GPU] MPRES-BLAS mpspmvell (naive)");

    size_t matrix_len = num_rows * cols_per_row;

    //Execution configuration
    int threads = 32;
    int blocks = num_rows / threads + 1;

    // Host data
    auto hx = new mp_float_t[num_cols];
    auto hy = new mp_float_t[num_rows];
    auto hdata = new mp_float_t[matrix_len];

    // GPU data
    mp_float_ptr dx;
    mp_float_ptr dy;
    mp_float_ptr ddata;
    int *dindices;

    //Init data
    cudaMalloc(&dx, sizeof(mp_float_t) * num_cols);
    cudaMalloc(&dy, sizeof(mp_float_t) * num_rows);
    cudaMalloc(&ddata, sizeof(mp_float_t) * matrix_len);
    cudaMalloc(&dindices, sizeof(int) * matrix_len);

    // Convert from MPFR
    convert_vector(hx, x, num_cols);
    convert_vector(hy, y, num_rows);
    convert_vector(hdata, data, matrix_len);

    //Copying to the GPU
    cudaMemcpy(dx, hx, num_cols * sizeof(mp_float_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dy, hy, num_rows * sizeof(mp_float_t), cudaMemcpyHostToDevice);
    cudaMemcpy(ddata, hdata, matrix_len * sizeof(mp_float_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dindices, indices, matrix_len * sizeof(int), cudaMemcpyHostToDevice);

    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    //Launch
    StartCudaTimer();
    mpspmvell_naive_kernel<<<blocks, threads>>>(num_rows, cols_per_row, dindices, ddata, dx, dy);
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
    cudaFree(dindices);
}

/////////
// TACO
/////////
void taco_test(const mpfr_t * vectorX, const mpfr_t * vectorY){
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

    Tensor<double> y({A.getDimension(0)}, dv);
    for (int i = 0; i < y.getDimension(0); ++i) {
        y.insert({i}, mpfr_get_d(vectorY[i], MPFR_RNDN));
    }
    y.pack();

    Tensor<double> result({A.getDimension(0)}, dv);

    IndexVar i, j;
    result(i) = (A(i,j) * x(j) + y(i));

    StartCpuTimer();
    result.compile();
    result.assemble();
    result.compute();
    EndCpuTimer();

    double sum = 0.0;
    for (int i = 0; i < result.getDimension(0); i++) {
        sum += result(i);
    }
    PrintCpuTimer("took");
    printf("result: %.70f\n", sum);
}

/********************* Main test *********************/

void test( int NUM_ROWS, int NUM_COLS, int NUM_LINES, int COLS_PER_ROW, bool MATRIX_SYMMETRIC, bool IS_REAL_DATA_TYPE) {

    //Inputs
    mpfr_t *vectorX = create_random_array(NUM_COLS, INP_BITS);
    mpfr_t *vectorY = create_random_array(NUM_ROWS, INP_BITS);
    auto * data = new double [NUM_ROWS * COLS_PER_ROW]();
    auto * indices = new int[NUM_ROWS * COLS_PER_ROW]();

    //Convert a sparse matrix to the double-precision ELLPACK format
    convert_to_ellpack(MATRIX_PATH, NUM_ROWS, COLS_PER_ROW, NUM_LINES, data, indices, MATRIX_SYMMETRIC);

    //TODO: Delete after debugging
 /*
    for (int i = 0; i < NUM_COLS; ++i) {
        mpfr_set_si(vectorX[i], (i+1), MPFR_RNDN);
    }

    print_ellpack(NUM_ROWS,COLS_PER_ROW,data,indices);
    int count = 0;
    for (int i = 0; i < NUM_ROWS * COLS_PER_ROW; ++i) {
        if (data[i] != 0) {
            count++;
        }
    }
    std::cout<<std::endl<<"NonZeros: "<<count<<std::endl;
*/
    //Launch tests
    double_test(NUM_ROWS, NUM_COLS, COLS_PER_ROW, data, indices, vectorX, vectorY);
    mpres_test(NUM_ROWS, NUM_COLS, COLS_PER_ROW, data, indices, vectorX, vectorY);
    mpres_test_naive(NUM_ROWS, NUM_COLS, COLS_PER_ROW, data, indices, vectorX, vectorY);
    campary_spmv_ell_test<CAMPARY_PRECISION>(NUM_ROWS, NUM_COLS, COLS_PER_ROW, data, indices, vectorX, vectorY, INP_DIGITS);
    cump_spmv_ell_test(NUM_ROWS, NUM_COLS, COLS_PER_ROW, data, indices, vectorX, vectorY, MP_PRECISION, INP_DIGITS);
    if (IS_REAL_DATA_TYPE) {
        taco_test(vectorX, vectorY);
    }
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
    delete[] indices;
    cudaDeviceReset();
}

int main() {
    //The operation parameters. Read from an input file that contains a sparse matrix
    int NUM_ROWS = 0; //number of rows
    int NUM_COLS = 0; //number of columns
    int NUM_LINES = 0; //number of lines in the input matrix file
    int COLS_PER_ROW = 0; //maximum number of nonzeros per row
    bool MATRIX_SYMMETRIC = false; //true if the input matrix is to be treated as symmetrical; otherwise false
    bool IS_REAL_DATA_TYPE = false; //defines type of data in MatrixMarket, used only for taco test

    initialize();

    //Start logging
    Logger::beginTestDescription(Logger::BLAS_SPMV_ELL_PERFORMANCE_TEST);
    Logger::beginSection("Operation info:");
    Logger::printParam("Matrix path", MATRIX_PATH);
    read_matrix_properties(MATRIX_PATH, NUM_ROWS, NUM_COLS, NUM_LINES, COLS_PER_ROW, MATRIX_SYMMETRIC, IS_REAL_DATA_TYPE);
    Logger::printParam("Matrix rows, NUM_ROWS", NUM_ROWS);
    Logger::printParam("Matrix columns, NUM_COLUMNS", NUM_COLS);
    Logger::printParam("Symmetry, MATRIX_SYMMETRIC", MATRIX_SYMMETRIC);
    Logger::printParam("Real data type, IS_REAL_DATA_TYPE", IS_REAL_DATA_TYPE);
    Logger::printParam("Maximum nonzeros per row, COLS_PER_ROW", COLS_PER_ROW);
    Logger::printDash();
    Logger::beginSection("Additional info:");
    Logger::printParam("MP_PRECISION", MP_PRECISION);
    Logger::printParam("RNS_MODULI_SIZE", RNS_MODULI_SIZE);
    Logger::printParam("MPRES_CUDA_BLOCKS_FIELDS_ROUND", MPRES_CUDA_THREADS_SCALAR_KERNELS);
    Logger::printParam("MPRES_CUDA_THREADS_FIELDS_ROUND", MPRES_CUDA_BLOCKS_RESIDUES);
    Logger::printParam("CAMPARY_PRECISION (n-double)", CAMPARY_PRECISION);
    Logger::endSection(true);

    //Run the test
    test(NUM_ROWS, NUM_COLS, NUM_LINES, COLS_PER_ROW, MATRIX_SYMMETRIC, IS_REAL_DATA_TYPE);

    //Finalize
    finalize();

    //End logging
    Logger::endTestDescription();

    return 0;
}