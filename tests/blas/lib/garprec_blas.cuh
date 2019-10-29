/*
 *  Multiple-precision kernels using GARPREC as well as corresponding performance benchmarks.
 *
 *  Copyright 2018, 2019 by Konstantin Isupov and Alexander Kuvaev.
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

#ifndef MPRES_TEST_GARPREC_BLAS_CUH
#define MPRES_TEST_GARPREC_BLAS_CUH

#include <stdio.h>
#include "mpfr.h"
#include "../../tsthelper.cuh"
#include "../../logger.cuh"
#include "../../timers.cuh"
//garprec
#include "../../3rdparty/garprec/inc/garprec.cu"
#include "../../3rdparty/garprec/inc/gmp_array.h"
#include "../../3rdparty/garprec/inc/gmp_array_func.h"


int initializeGarprec(int prec){
    const int maxPrecWords = garprecInit(prec, CUDA_DEVICE);
    return maxPrecWords;
}

/********************* Computational kernels *********************/

/*
 * Computes the sum of the elements of vector x, Kernel #1
 */
__global__ void garprec_sum_kernel1(int n, double *result, int int_result, double *x, int int_x, double *temp, int int_temp, int prec_words){
    // parameters
    unsigned int tid = threadIdx.x;
    unsigned int bid = blockIdx.x;
    unsigned int bsize = blockDim.x;
    unsigned int globalIdx = bid * bsize + tid;
    unsigned int i = bid * bsize * 2 + tid;
    unsigned int k = 2 * gridDim.x * bsize;
    double d[MAX_D_SIZE];

    while (i < n) {
        gmpadd(x + i, int_x, temp + globalIdx, int_temp, temp + globalIdx, int_temp, prec_words, d);
        if (i + bsize < n){
            gmpadd(x + (i + bsize), int_x, temp + globalIdx, int_temp, temp + globalIdx, int_temp, prec_words, d);
        }
        i += k;
    }
    __syncthreads();

    i = bsize;
    while(i >= 2){
        unsigned int half = i >> 1;
        if ((bsize >= i) && (tid < half) && (globalIdx + half < n)) {
            gmpadd(temp + globalIdx + half, int_temp, temp + globalIdx, int_temp, temp + globalIdx, int_temp, prec_words, d);
        }
        i = i >> 1;
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) {
        gmpeq(temp + globalIdx, int_temp, result + bid, int_result, prec_words);
    }
    __syncthreads();
}

/*
 * Computes the sum of the elements of vector x, Kernel #2 (optimized)
 */
__global__ void garprec_sum_kernel2(int n, double *result, int interval_result, double *x, int interval_x, int prec_words){
    unsigned int tid = threadIdx.x;
    double d[MAX_PREC_WORDS];
    // do reduction in global mem
    gmpeq(x + tid, interval_x, result + tid, interval_result, prec_words); //здесь x - это результат работы блоков с предыдущего запуска
    __syncthreads();
    for(unsigned int s = blockDim.x / 2; s > 0; s >>= 1){
        if(tid < s && tid + s < n){
            gmpadd(result + tid + s, interval_result, result + tid, interval_result, result + tid, interval_result, prec_words, d);
        }
        __syncthreads();
    }
    __syncthreads();
}

/*
 * Computes the componentwise vector-vector product
 */
__global__ void garprec_vec_mul_kernel(int n, double *x, int interval_x, double *y, int interval_y, double *r, int interval_r, int prec_words){
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int delta = blockDim.x * gridDim.x;
    double d[MAX_D_SIZE];
    for(int i = index; i < n; i+= delta){
        gmpmul(x + i, interval_x, y + i, interval_y, r + i, interval_r, prec_words, d);
    }
}

/*
 * Multiplies a scalar by a vector
 */
__global__ void garprec_scal_kernel(int n, double *alpha, int interval_alpha, double *x, int interval_x, int prec_words){
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int delta = blockDim.x * gridDim.x;
    double d[MAX_D_SIZE];
    for(int i = index; i < n; i+= delta){
        gmpmul(alpha, interval_alpha, x + i, interval_x, x + i, interval_x, prec_words, d);
    }
}

/*
 * Constant times a vector plus a vector
 */
__global__ void garprec_axpy_kernel(int n, double *alpha, int interval_alpha, double *x, int interval_x, double *y, int interval_y,
                             double *tmp, int interval_tmp, int prec_words){
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int delta = blockDim.x * gridDim.x;
    double d[MAX_D_SIZE];
    for(int i = index; i < n; i+= delta){
        gmpmul(alpha, interval_alpha, x + i, interval_x, tmp + i, interval_tmp, prec_words, d);
        gmpadd(y + i, interval_y, tmp + i, interval_tmp, y + i, interval_y, prec_words, d);
    }
}

/*
 * Reset array
 */
__global__ void garprec_reset_array(int n, double *temp, int int_temp, int prec_words) {
    int numberIdx =  blockDim.x * blockIdx.x + threadIdx.x;
    double d[MAX_D_SIZE];
    while (numberIdx < n) {
        gmpsub(temp + numberIdx, int_temp, temp + numberIdx, int_temp, temp + numberIdx, int_temp, prec_words, d); // set to zero
        numberIdx +=  gridDim.x * blockDim.x;
    }
}


/********************* Benchmarks *********************/


/*
 * SUM test
 */
void garprec_sum_test(int n, mpfr_t *x, int prec, int input_prec_dec, int repeat){
    Logger::printDash();
    InitCudaTimer();
    PrintTimerName("[GPU] GARPREC sum");

    //Init and set precision
    int maxPrecWords = initializeGarprec(prec);
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    //Execution configuration
    int threads = 64;
    int blocks = 1024;

    //Host data
    mp_real *hx = new mp_real[n];

    //GPU data
    gmp_array *dx = new gmp_array(maxPrecWords, n, true);
    gmp_array *dtemp = new gmp_array(maxPrecWords, n, true);
    gmp_array *dblock_result = new gmp_array(maxPrecWords, n, true);

    //Convert from MPFR
    #pragma omp parallel for
    for(int i = 0; i < n; i++){
        hx[i].read(convert_to_string_sci(x[i], input_prec_dec));
    }

    //Launch
    for(int j = 0; j < repeat; j ++){
        dx->toGPU(hx, n);
        garprec_reset_array<<<blocks, threads>>>(n,  dtemp->d_mpr, dtemp->interval, maxPrecWords);
        StartCudaTimer();
        garprec_sum_kernel1<<<blocks, threads>>>(
                n,
                        dblock_result->d_mpr,
                        dblock_result->interval,
                        dx->d_mpr,
                        dx->interval,
                        dtemp->d_mpr,
                        dtemp->interval,
                        maxPrecWords);

        garprec_sum_kernel2<<<1, blocks>>>(
                blocks,
                        dx->d_mpr,
                        dx->interval,
                        dblock_result->d_mpr,
                        dblock_result->interval,
                        maxPrecWords);
        EndCudaTimer();
    }
    PrintCudaTimer("took");

    //Copying to the host
    dx->fromGPU(hx, n);
    printf("result: %.83s \n", hx[0].to_string().c_str());

    //Cleanup
    delete [] hx;
    dx->release();
    dblock_result->release();
    dtemp->release();
    garprecFinalize();
}


/*
 * DOT test
 */
void garprec_dot_test(int n, mpfr_t *x, mpfr_t *y, int prec, int input_prec_dec, int repeat){
    Logger::printDash();
    InitCudaTimer();
    PrintTimerName("[GPU] GARPREC dot");

    //Init and set precision
    int maxPrecWords = initializeGarprec(prec);
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    //Execution configuration
    int threads = 64;
    int blocks_mul = n / threads + (n % threads ? 1 : 0);
    int blocks_red = 1024;

    //Host data
    mp_real *hx = new mp_real[n];
    mp_real *hy = new mp_real[n];

    //GPU data
    gmp_array *dx = new gmp_array(maxPrecWords, n, true);
    gmp_array *dy = new gmp_array(maxPrecWords, n, true);
    gmp_array *dtemp = new gmp_array(maxPrecWords, n, true);

    //Convert from MPFR
    #pragma omp parallel for
    for(int i = 0; i < n; i++){
        hx[i].read(convert_to_string_sci(x[i], input_prec_dec));
        hy[i].read(convert_to_string_sci(y[i], input_prec_dec));
    }

    //Launch
    for(int j = 0; j < repeat; j ++){
        dx->toGPU(hx, n);
        dy->toGPU(hy, n);
        garprec_reset_array<<<blocks_mul, threads>>>(n,  dtemp->d_mpr, dtemp->interval, maxPrecWords);

        StartCudaTimer();
        garprec_vec_mul_kernel<<<blocks_mul, threads>>>(n, dx->d_mpr, dx->interval, dy->d_mpr, dy->interval, dx->d_mpr, dx->interval, maxPrecWords);

        garprec_sum_kernel1<<<blocks_red, threads>>>(
                n,
                        dy->d_mpr,
                        dy->interval,
                        dx->d_mpr,
                        dx->interval,
                        dtemp->d_mpr,
                        dtemp->interval,
                        maxPrecWords);

        garprec_sum_kernel2<<<1, blocks_red>>>(
                blocks_red,
                        dx->d_mpr,
                        dx->interval,
                        dy->d_mpr,
                        dy->interval,
                        maxPrecWords);
        EndCudaTimer();
    }
    PrintCudaTimer("took");

    //Copying to the host
    dx->fromGPU(hx, n);
    printf("result: %.83s \n", hx[0].to_string().c_str());

    //Cleanup
    delete [] hx;
    delete [] hy;
    dx->release();
    dy->release();
    dtemp->release();
    garprecFinalize();
}


/*
 * SCAL test
 */
void garprec_scal_test(int n, mpfr_t alpha, mpfr_t *x, int prec, int input_prec_dec, int repeat){
    Logger::printDash();
    InitCudaTimer();
    PrintTimerName("[GPU] GARPREC scal");

    //Init and set precision
    int maxPrecWords = initializeGarprec(prec);
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    //Execution configuration
    int threads = 64;
    int blocks = n / threads + (n % threads ? 1 : 0);

    //Host data
    mp_real *lx = new mp_real[n];
    mp_real *lalpha = new mp_real[1];

    //GPU data
    gmp_array *gx = new gmp_array(maxPrecWords, n, true);
    gmp_array *galpha = new gmp_array(maxPrecWords, 1, true);

    //Convert from MPFR
    #pragma omp parallel for
    for(int i = 0; i < n; i++){
        lx[i].read(convert_to_string_sci(x[i], input_prec_dec));
    }
    lalpha[0].read(convert_to_string_sci(alpha, input_prec_dec));

    //Copying to the GPU
    //gx->toGPU(lx, n);
    galpha->toGPU(lalpha, 1);

    //Launch
    for(int j = 0; j < repeat; j ++){
        gx->toGPU(lx, n);
        StartCudaTimer();
        garprec_scal_kernel<<<blocks, threads>>>(n, galpha->d_mpr, galpha->interval, gx->d_mpr, gx->interval, maxPrecWords);
        EndCudaTimer();
    }
    PrintCudaTimer("took");

    //Copying to the host
    gx->fromGPU(lx, n);
    for (int i = 1; i < n; i++) {
        lx[0] += lx[i];
    }
    printf("result: %.83s \n", lx[0].to_string().c_str());

    //Cleanup
    delete [] lx;
    galpha->release();
    gx->release();
    garprecFinalize();
}

/*
 * AXPY test
 */
void garprec_axpy_test(int n, mpfr_t alpha, mpfr_t *x, mpfr_t *y, int prec, int input_prec_dec, int repeat){
    Logger::printDash();
    InitCudaTimer();
    PrintTimerName("[GPU] GARPREC axpy");

    //Init and set precision
    int maxPrecWords = initializeGarprec(prec);
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    //Execution configuration
    int threads = 64;
    int blocks = n / threads + (n % threads ? 1 : 0);

    //Host data
    mp_real *lx = new mp_real[n];
    mp_real *ly = new mp_real[n];
    mp_real *lalpha = new mp_real[1];

    //GPU data
    gmp_array *gx = new gmp_array(maxPrecWords, n, true);
    gmp_array *gy = new gmp_array(maxPrecWords, n, true);
    gmp_array *gtmp = new gmp_array(maxPrecWords, n, true);
    gmp_array *galpha = new gmp_array(maxPrecWords, 1, true);

    //Convert from MPFR
    #pragma omp parallel for
    for(int i = 0; i < n; i++){
        lx[i].read(convert_to_string_sci(x[i], input_prec_dec));
        ly[i].read(convert_to_string_sci(y[i], input_prec_dec));
    }
    lalpha[0].read(convert_to_string_sci(alpha, input_prec_dec));

    //Copying to the GPU
    gx->toGPU(lx, n);
    //gy->toGPU(ly, n);
    galpha->toGPU(lalpha, 1);

    //Launch
    for(int j = 0; j < repeat; j ++){
        gy->toGPU(ly, n);
        StartCudaTimer();
        garprec_axpy_kernel<<<blocks, threads>>>(n, galpha->d_mpr, galpha->interval, gx->d_mpr, gx->interval, gy->d_mpr, gy->interval, gtmp->d_mpr, gtmp->interval, maxPrecWords);
        EndCudaTimer();
    }
    PrintCudaTimer("took");

    //Copying to the host
    gy->fromGPU(ly, n);
    for (int i = 1; i < n; i++) {
        ly[0] += ly[i];
    }
    printf("result: %.83s \n", ly[0].to_string().c_str());

    //Cleanup
    delete [] lx;
    galpha->release();
    gx->release();
    gy->release();
    garprecFinalize();
}

#endif //MPRES_TEST_GARPREC_BLAS_CUH