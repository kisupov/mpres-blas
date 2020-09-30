/*
 *  Performance test for BLAS GEMV routines
 *
 *  Copyright 2020 by Konstantin Isupov.
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
#include "../../../src/sparse/matrix_convertor.cuh"
#include "../../sparse/performance/3rdparty.cuh"


#define REPEAT_TEST 5 //Number of repeats

//Execution configuration for mpgemv
#define MPRES_CUDA_BLOCKS_FIELDS_ROUND 256
#define MPRES_CUDA_THREADS_FIELDS_ROUND 128
#define MPRES_CUDA_BLOCKS_RESIDUES 256
#define MPRES_CUDA_THREADS_REDUCE 32

#define OPENBLAS_THREADS 4

#define MATRIX_PATH "../../tests/sparse/matrices/Trefethen_20b.mtx"

int MP_PRECISION_DEC; //in decimal digits
int INP_BITS; //in bits
int INP_DIGITS; //in decimal digits

void setPrecisions() {
    MP_PRECISION_DEC = (int) (MP_PRECISION / 3.32 + 1);
    INP_BITS = (int) (MP_PRECISION / 4);
    INP_DIGITS = (int) (INP_BITS / 3.32 + 1);
}

void initialize() {
    cudaDeviceReset();
    rns_const_init();
    mp_const_init();
    setPrecisions();
    mp_real::mp_init(MP_PRECISION_DEC);
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();
}

void finalize() {
    mp_real::mp_finalize();
}

void convert_vector(double *&dest, mp_float_t *source, int width) {
    for (int i = 0; i < width; i++) {
        dest[i] = mp_get_d(&source[i]);
    }
}


//TODO придумать как убрать сложение с Y просчитать сразу строку
__global__ static void double_ellpack_spmv(double *data, double *x, double *y, int *indices, int num_rows, int num_cols_per_row) {
    double sum = 0;
    int indice = 0;
    for (int colId = 0; colId < num_cols_per_row; colId++) {
        int index = (threadIdx.x + blockIdx.x * blockDim.x);
        while( index < num_rows ){
            indice = indices[colId * num_rows + index];
            sum = data[colId * num_rows + index] * x[indice];
            y[index] = y[index] + sum;
            index += gridDim.x * blockDim.x;
        }
        __syncthreads();
    }
}


/********************* SPMV implementations and benchmarks *********************/

/////////
// SpMV-ELLPACK double(structure of arrays)
/////////
void spmv_ellpack_double_test(int num_rows, int num_cols, int num_cols_per_row, mp_float_t *data, int *indices,
                                mp_float_t *x, mp_float_t *y){
    InitCudaTimer();
    Logger::printDash();
    PrintTimerName("[GPU] ELLPACK SpMV (straightforward)");

    //Execution configuration
    int threads = 32;
    int blocks = num_rows / (threads) + (num_rows % (threads) ? 1 : 0);

    //host data
    double *hdata = new double[num_rows * num_cols_per_row];
    double *hx = new double[num_cols];
    double *hy = new double[num_rows];

    //GPU data
    double *ddata = new double[num_rows * num_cols_per_row];
    int *dindices = new int[num_rows * num_cols_per_row];
    double *dx = new double[num_cols];
    double *dy = new double[num_rows];

    convert_vector(hdata, data, num_rows * num_cols_per_row);
    convert_vector(hx, x, num_cols);
    convert_vector(hy, y, num_rows);

    cudaMalloc(&ddata, sizeof(double) * num_rows * num_cols_per_row);
    cudaMalloc(&dindices, sizeof(int) * num_rows * num_cols_per_row);
    cudaMalloc(&dx, sizeof(double) * num_cols);
    cudaMalloc(&dy, sizeof(double) * num_rows);

    cudaMemcpy(ddata, hdata, sizeof(double) * num_rows * num_cols_per_row, cudaMemcpyHostToDevice);
    cudaMemcpy(dindices, indices, sizeof(int) * num_rows * num_cols_per_row, cudaMemcpyHostToDevice);
    cudaMemcpy(dx, hx, sizeof(double) * num_cols, cudaMemcpyHostToDevice);

    for(int i = 0; i < REPEAT_TEST; i ++) {
        cudaMemcpy(dy, hy, sizeof(double) * num_rows, cudaMemcpyHostToDevice);
        StartCudaTimer();
        double_ellpack_spmv<<<blocks, threads>>>(ddata, dx, dy, dindices, num_rows, num_cols_per_row);
        EndCudaTimer();
    }

    cudaMemcpy(hy, dy, sizeof(double) * num_rows , cudaMemcpyDeviceToHost);

    PrintCudaTimer("took");
    print_double_sum(hy, num_rows);

    delete [] hdata;
    delete [] hx;
    delete [] hy;
    cudaFree(ddata);
    cudaFree(dindices);
    cudaFree(dx);
    cudaFree(dy);
}

/////////
// SpMV-ELLPACK (structure of arrays)
/////////
void spmv_ellpack_test(int num_rows, int num_cols, int num_cols_per_row, mp_float_ptr data, int *indices, mp_float_ptr x, mp_float_ptr y) {
    Logger::printDash();
    InitCudaTimer();
    PrintTimerName("[GPU] ELLPACK SpMV");

    //Host data
    mp_float_ptr hy = new mp_float_t[num_rows];

    //GPU data
    mp_array_t dx;
    mp_array_t dy;
    mp_array_t ddata;
    mp_array_t dbuf1;
    int *dindices;

    //Init data
    cuda::mp_array_init(dx, num_cols);
    cuda::mp_array_init(dy, num_rows);
    cuda::mp_array_init(ddata, num_rows * num_cols_per_row);
    cuda::mp_array_init(dbuf1, num_rows * num_cols_per_row);
    cudaMalloc(&dindices, sizeof(int) * num_rows * num_cols_per_row);

    //Copying to the GPU
    cuda::mp_array_host2device(dx, x, num_cols);
    cuda::mp_array_host2device(ddata, data, num_rows * num_cols_per_row);
    cudaMemcpy(dindices, indices, sizeof(int) * num_rows * num_cols_per_row, cudaMemcpyHostToDevice);
    cuda::mp_array_host2device(dy, y, num_rows);

    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    //Launch
    for (int i = 0; i < REPEAT_TEST; i++) {
        StartCudaTimer();
        cuda::mpspmvell<
                MPRES_CUDA_BLOCKS_FIELDS_ROUND,
                MPRES_CUDA_THREADS_FIELDS_ROUND,
                MPRES_CUDA_BLOCKS_RESIDUES,
                MPRES_CUDA_THREADS_REDUCE>
                (num_rows, num_cols_per_row, ddata, dindices, dx, dy, dbuf1);
        EndCudaTimer();
    }
    PrintCudaTimer("took");
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    //Copying to the host
    cuda::mp_array_device2host(hy, dy, num_rows);
    print_mp_sum(hy, num_rows);

    //Cleanup
    delete [] hy;
    cuda::mp_array_clear(dx);
    cuda::mp_array_clear(dy);
    cuda::mp_array_clear(ddata);
    cuda::mp_array_clear(dbuf1);
    cudaFree(dindices);
}


/********************* Main test *********************/

/*
 * Test for non-transposed matrix
 * x is of size num_cols
 * y is of size num_rows
 * a is of size num_rows * num_cols
 */
void test() {
    //Actual length of the vectors

    int num_rows = 0, num_cols = 0, num_cols_per_row = 0;
    mp_float_t *data;
    int *indices;

    create_ellpack_matrices(MATRIX_PATH, data, indices, num_rows, num_cols, num_cols_per_row);

    //Inputs
    mp_float_t *vectorX = new mp_float_t[num_cols];
    mp_float_t *vectorY = new mp_float_t[num_rows];

    //инициализируем вектор X
    for (int i = 0; i < num_cols; ++i) {
        mp_set_d(&vectorX[i], (i + 1));
    }

    //инициализируем вектор Y
    for (int i = 0; i < num_rows; ++i) {
        mp_set_d(&vectorY[i], 0);
    }

    //Launch tests
    spmv_ellpack_test(num_rows, num_cols, num_cols_per_row, data, indices, vectorX, vectorY);
    campary_spmv_ellpack_test<CAMPARY_PRECISION>(num_rows, num_cols, num_cols_per_row, data, indices, vectorX, vectorY, INP_DIGITS, REPEAT_TEST);
    spmv_ellpack_double_test(num_rows, num_cols, num_cols_per_row, data, indices, vectorX, vectorY);
    cump_spmv_ellpack_test(num_rows, num_cols, num_cols_per_row, data, indices, vectorX, vectorY, MP_PRECISION, INP_DIGITS, REPEAT_TEST);

    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    //Cleanup
    delete[] vectorX;
    delete[] vectorY;
    delete[] data;
    delete[] indices;
    cudaDeviceReset();
}

int main() {

    initialize();

    //Start logging
    Logger::beginTestDescription(Logger::BLAS_SPMV_PERFORMANCE_TEST);
    Logger::beginSection("Operation info:");
    Logger::printParam("Matrix path", MATRIX_PATH);
    Logger::printDash();
    Logger::beginSection("Additional info:");
    Logger::printParam("RNS_MODULI_SIZE", RNS_MODULI_SIZE);
    Logger::printParam("MPRES_CUDA_BLOCKS_FIELDS_ROUND", MPRES_CUDA_BLOCKS_FIELDS_ROUND);
    Logger::printParam("MPRES_CUDA_THREADS_FIELDS_ROUND", MPRES_CUDA_THREADS_FIELDS_ROUND);
    Logger::printParam("MPRES_CUDA_BLOCKS_RESIDUES", MPRES_CUDA_BLOCKS_RESIDUES);
    Logger::printParam("MPRES_CUDA_THREADS_REDUCE", MPRES_CUDA_THREADS_REDUCE);
    Logger::printParam("CAMPARY_PRECISION (n-double)", CAMPARY_PRECISION);
    Logger::endSection(true);

    //Run the test
    test();

    //Finalize
    finalize();

    //End logging
    Logger::endTestDescription();

    return 0;
}