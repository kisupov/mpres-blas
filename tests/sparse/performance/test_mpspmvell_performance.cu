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
#define MPRES_CUDA_THREADS_SCALAR_KERNELS 64
#define MPRES_CUDA_BLOCKS_RESIDUES 256

#define MATRIX_PATH "../../tests/sparse/matrices/ex40.mtx"
#define MATRIX_SYMMETRIC false

int INP_BITS; //in bits
int INP_DIGITS; //in decimal digits

void setPrecisions() {
    INP_BITS = (int) (MP_PRECISION / 2);
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

void print_float(float * data, int size){
    for (int i = 0; i < size; ++i) {
        std::cout << data[i] << " ";
    }
    std::cout<<std::endl;
}

void print_int(int * data, int size){
    for (int i = 0; i < size; ++i) {
        std::cout << data[i] << " ";
    }
    std::cout<<std::endl;
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
    int blocks = num_rows / (threads) + (num_rows % (threads) ? 1 : 0);

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
// cuSPARSE
/////////
void cusparse_test_coo(const int num_rows, const int num_cols, const int num_lines, mpfr_t *x){
    InitCudaTimer();
    Logger::printDash();
    PrintTimerName("[GPU] cuSPARSE COO");

    int A_num_nnz = num_lines;
    if (IS_SYMMETRIC) {
        A_num_nnz = (num_lines - num_cols) * 2 + num_cols;
    }

/*    int *hA_rows = new int[A_num_nnz];
    int *hA_columns = new int[A_num_nnz];
    float *hA_values = new float[A_num_nnz];*/

    float hA_values[] = { 3.0f, 1.0f, 1.0f, 1.0f,
                          1.0f, 5.0f, 1.0f, 1.0f,
                          1.0f, 1.0f, 7.0f, 1.0f,
                          1.0f, 1.0f, 11.0f, 1.0f,
                          1.0f, 1.0f, 1.0f, 13.0f};
    int hA_rows[] = { 1, 1, 1, 1,
                      2, 2, 2, 2,
                      3, 3, 3, 3,
                      4, 4, 4, 4,
                      5, 5, 5, 5};

    int hA_columns[] ={ 1, 2, 4, 5,
                        1, 2, 3, 5,
                        1, 2, 3, 4,
                        2, 3, 4, 5,
                        1, 3, 4, 5};
    float *hX = new float[num_rows];
    float *hY = new float[num_rows]();


    // Convert from MPFR
    convert_vector(hX, x, num_cols);
    //Convert a sparse matrix to the double-precision ELLPACK format
    convert_to_coo(MATRIX_PATH, num_rows, num_lines, hA_values, hA_rows, hA_columns, IS_SYMMETRIC);

    print_float(hA_values, A_num_nnz);
    print_int(hA_rows, A_num_nnz);
    print_int(hA_columns, A_num_nnz);

    int *dA_rows;
    int *dA_columns;
    float *dA_values;
    float *dX;
    float *dY;
    float alpha = 1.0f;
    float beta = 0.0f;

    cudaMalloc(&dA_rows, A_num_nnz * sizeof(int));
    cudaMalloc(&dA_columns, A_num_nnz * sizeof(int));
    cudaMalloc(&dA_values, A_num_nnz * sizeof(float));
    cudaMalloc(&dX, num_cols * sizeof(float));
    cudaMalloc(&dY, num_rows * sizeof(float));

    cudaMemcpy(dA_rows, hA_rows, A_num_nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dA_columns, hA_columns, A_num_nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dA_values, hA_values, A_num_nnz * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dX, hX, num_cols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dY, hY, num_rows * sizeof(float), cudaMemcpyHostToDevice);

    cusparseHandle_t handle = NULL;
    cusparseStatus_t stat;
    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecX;
    cusparseDnVecDescr_t vecY;
    void *dBuffer = NULL;
    size_t bufferSize = 0;

    stat = cusparseCreate(&handle);
    if (stat != CUSPARSE_STATUS_SUCCESS) {
        printf ("CUSPARSE initialization failed\n");
        return;
    }

    // Create sparse matrix A in COO format
    cusparseCreateCoo(&matA, num_rows, num_cols, A_num_nnz, dA_rows, dA_columns, dA_values, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ONE, CUDA_R_32F);
    // Create dense vector X
    cusparseCreateDnVec(&vecX, num_cols, dX, CUDA_R_32F);
    // Create dense vector y
    cusparseCreateDnVec(&vecY, num_rows, dY, CUDA_R_32F);
    // allocate an external buffer if needed
    cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecX, &beta, vecY, CUDA_R_32F, CUSPARSE_MV_ALG_DEFAULT, &bufferSize);
    cudaMalloc(&dBuffer, bufferSize);

    // execute SpMV
    cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecX, &beta, vecY, CUDA_R_32F, CUSPARSE_MV_ALG_DEFAULT, dBuffer);

    // destroy matrix/vector descriptors
    cusparseDestroySpMat(matA);
    cusparseDestroyDnVec(vecX);
    cusparseDestroyDnVec(vecY);
    cusparseDestroy(handle);

    cudaMemcpy(hY, dY, num_rows * sizeof(float), cudaMemcpyDeviceToHost);
    float sum = 0;
    for (int i = 0; i < num_rows; i++) {
        sum += hY[i];
    }
    printf("result: %.70f\n", sum);

/*    delete [] hA_rows;
    delete [] hA_columns;
    delete [] hA_values;*/
    delete [] hX;
    delete [] hY;
    cudaFree(dBuffer);
    cudaFree(dA_rows);
    cudaFree(dA_columns);
    cudaFree(dA_values);
    cudaFree(dX);
    cudaFree(dY);
}

void cusparse_test_csr(const int num_rows, const int num_cols, const int num_lines, mpfr_t *x) {
    InitCudaTimer();
    Logger::printDash();
    PrintTimerName("[GPU] cuSPARSE CSR");

    int A_num_nnz = num_lines;
    /*int *hA_csrOffsets = new int[num_rows + 1]();
    int *hA_columns = new int[A_num_nnz] ;
    float *hA_values = new float[A_num_nnz];*/

    float hA_values[] = { 3.0f, 1.0f, 1.0f, 1.0f,
                          1.0f, 5.0f, 1.0f, 1.0f,
                          1.0f, 1.0f, 7.0f, 1.0f,
                          1.0f, 1.0f, 11.0f, 1.0f,
                          1.0f, 1.0f, 1.0f, 13.0f};
    int hA_csrOffsets[] = {0, 4, 8, 12, 16, 20};
    int hA_columns[] ={ 1, 2, 4, 5,
                        1, 2, 3, 5,
                        1, 2, 3, 4,
                        2, 3, 4, 5,
                        1, 3, 4, 5};
    float *hX = new float[num_cols];
    float *hY = new float[num_rows]();

    convert_vector(hX, x, num_cols);
    //Convert a sparse matrix to the double-precision ELLPACK format
    //convert_to_csr(MATRIX_PATH, num_rows, num_lines, hA_values, hA_csrOffsets, hA_columns, IS_SYMMETRIC);

    print_float(hA_values, A_num_nnz);
    print_int(hA_csrOffsets, num_rows+1);
    print_int(hA_columns, A_num_nnz);

    int *dA_csrOffsets;
    int *dA_columns;
    float *dA_values;
    float *dX;
    float *dY;
    float alpha = 1.0f;
    float beta = 0.0f;

    cudaMalloc((void**) &dA_csrOffsets, (num_rows + 1) * sizeof(int));
    cudaMalloc((void**) &dA_columns, A_num_nnz * sizeof(int));
    cudaMalloc((void**) &dA_values, A_num_nnz * sizeof(float));
    cudaMalloc((void**) &dX, num_cols * sizeof(float));
    cudaMalloc((void**) &dY, num_rows * sizeof(float));

    cudaMemcpy(dA_csrOffsets, hA_csrOffsets, (num_rows + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dA_columns, hA_columns, A_num_nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dA_values, hA_values, A_num_nnz * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dX, hX, num_cols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dY, hY, num_rows * sizeof(float), cudaMemcpyHostToDevice);

    //--------------------------------------------------------------------------
    // CUSPARSE APIs
    cusparseHandle_t handle = NULL;
    cusparseStatus_t stat;
    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecX;
    cusparseDnVecDescr_t vecY;
    void* dBuffer = NULL;
    size_t bufferSize = 0;

    cusparseCreate(&handle);
    // Create sparse matrix A in CSR format
    cusparseCreateCsr(&matA, num_rows, num_cols, A_num_nnz, dA_csrOffsets, dA_columns, dA_values, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ONE, CUDA_R_32F);
    // Create dense vector X
    cusparseCreateDnVec(&vecX, num_cols, dX, CUDA_R_32F);
    // Create dense vector y
    cusparseCreateDnVec(&vecY, num_rows, dY, CUDA_R_32F);
    // allocate an external buffer if needed
    cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecX, &beta, vecY, CUDA_R_32F, CUSPARSE_MV_ALG_DEFAULT, &bufferSize);
    cudaMalloc(&dBuffer, bufferSize);

    // execute SpMV
    cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecX, &beta, vecY, CUDA_R_32F, CUSPARSE_MV_ALG_DEFAULT, dBuffer);

    // destroy matrix/vector descriptors
    cusparseDestroySpMat(matA);
    cusparseDestroyDnVec(vecX);
    cusparseDestroyDnVec(vecY);
    cusparseDestroy(handle);

    //--------------------------------------------------------------------------
    // device result check

    cudaMemcpy(hY, dY, num_rows * sizeof(float), cudaMemcpyDeviceToHost);
    float sum = 0;
    for (int i = 0; i < num_rows; i++) {
        sum += hY[i];
    }
    printf("result: %.70f\n", sum);

    //--------------------------------------------------------------------------
    // device memory deallocation
/*    delete [] hA_csrOffsets;
    delete [] hA_columns;
    delete [] hA_values;*/
    delete [] hX;
    delete [] hY;
    cudaFree(dBuffer);
    cudaFree(dA_csrOffsets);
    cudaFree(dA_columns);
    cudaFree(dA_values);
    cudaFree(dX);
    cudaFree(dY);
}

/********************* Main test *********************/

void test( int NUM_ROWS, int NUM_COLS, int NUM_LINES, int COLS_PER_ROW) {

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

    initialize();

    //Start logging
    Logger::beginTestDescription(Logger::BLAS_SPMV_ELL_PERFORMANCE_TEST);
    Logger::beginSection("Operation info:");
    Logger::printParam("Matrix path", MATRIX_PATH);
    read_matrix_properties(MATRIX_PATH, NUM_ROWS, NUM_COLS, NUM_LINES, COLS_PER_ROW, MATRIX_SYMMETRIC);
    Logger::printParam("Matrix rows, NUM_ROWS", NUM_ROWS);
    Logger::printParam("Matrix columns, NUM_COLUMNS", NUM_COLS);
    Logger::printParam("Maximum nonzeros per row, COLS_PER_ROW", COLS_PER_ROW);
    Logger::printDash();
    Logger::beginSection("Additional info:");
    Logger::printParam("RNS_MODULI_SIZE", RNS_MODULI_SIZE);
    Logger::printParam("MPRES_CUDA_BLOCKS_FIELDS_ROUND", MPRES_CUDA_THREADS_SCALAR_KERNELS);
    Logger::printParam("MPRES_CUDA_THREADS_FIELDS_ROUND", MPRES_CUDA_BLOCKS_RESIDUES);
    Logger::printParam("CAMPARY_PRECISION (n-double)", CAMPARY_PRECISION);
    Logger::endSection(true);

    //Run the test
    test(NUM_ROWS, NUM_COLS, NUM_LINES, COLS_PER_ROW);

    //Finalize
    finalize();

    //End logging
    Logger::endTestDescription();

    return 0;
}