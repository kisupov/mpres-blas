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

#ifndef EXCLUDE_GARPREC
#ifndef MPRES_TEST_GARPREC_BLAS_CUH
#define MPRES_TEST_GARPREC_BLAS_CUH

#include <stdio.h>
#include "mpfr.h"
#include "../../tsthelper.cuh"
#include "../../logger.cuh"
#include "../../timers.cuh"
#include "lib/garprec/inc/garprec.cu"
#include "lib/garprec/inc/gmp_array.h"
#include "lib/garprec/inc/gmp_array_func.h"


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
 * Computes the sum of the elements of vector x, Kernel #2 (optimized kernel)
 */
__global__ void garprec_sum_kernel2(int n, double *result, int interval_result, double *x, int interval_x, int prec_words){
    unsigned int tid = threadIdx.x;
    double d[MAX_D_SIZE];
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
 * Computes the element-wise vector-vector product
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
 * Performs rotation of points in the plane
 */
__global__ void garprec_rot_kernel(int n, double *x, int interval_x, double *y, int interval_y, double *c, int interval_c,
                                   double *s, int interval_s, double *tmp, int interval_tmp, double *tmp2, int interval_tmp2, int prec_words){
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int delta = blockDim.x * gridDim.x;
    double d[MAX_D_SIZE];
    for(int i = index; i < n; i+= delta){
        gmpmul(x + i, interval_x, c + 0,  interval_c, tmp + i, interval_tmp, prec_words, d);
        gmpmul(y + i, interval_y, s + 0,  interval_s, tmp2 + i, interval_tmp2, prec_words, d);
        gmpmul(y + i, interval_y, c + 0, interval_c, y + i, interval_y, prec_words, d);
        gmpmul(x + i, interval_x, s + 0, interval_s, x + i, interval_x, prec_words, d);
        gmpsub(y + i, interval_y, x + i, interval_x, y + i, interval_y, prec_words, d);
        gmpadd(tmp + i, interval_tmp, tmp2 + i, interval_tmp2, x + i, interval_x, prec_words, d);
    }
    __syncthreads();
}

/*
 * Set the elements of an array to zero
 */
__global__ void garprec_reset_array(int n, double *temp, int int_temp, int prec_words) {
    int numberIdx =  blockDim.x * blockIdx.x + threadIdx.x;
    double d[MAX_D_SIZE];
    while (numberIdx < n) {
        gmpsub(temp + numberIdx, int_temp, temp + numberIdx, int_temp, temp + numberIdx, int_temp, prec_words, d); // set to zero
        numberIdx +=  gridDim.x * blockDim.x;
    }
}

/*
 * Performs the matrix-vector operation  y := A*x + beta*y,
 * where beta is a scalar, x and y are vectors and A is an m by n matrix
 */
__global__ void garprec_gemv_kernel(int m, int n, double *A, int interval_A,
                                    int lda, double *x, int interval_x, double *beta, int interval_beta, double *y, int interval_y, double *tmp1, int interval_tmp1, int prec_words) {
    const unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
    double d[MAX_D_SIZE];
    //Scaling y by beta
    if (index < m){
        gmpmul(y + index, interval_y, beta + 0, interval_beta, y + index, interval_y, prec_words, d);
    }
    //Matrix-vector multiplication
    for (int j = 0; j < n; j++) {
        if( index < m ){
            gmpmul(x + j, interval_x, A + (index + j * lda), interval_A, tmp1 + index, interval_tmp1, prec_words, d);
            gmpadd(tmp1 + index, interval_tmp1, y + index, interval_y, y + index, interval_y, prec_words, d);
        }
    }
}

/********************* Benchmarks *********************/


/*
 * SUM test
 * Note that the sum is calculated instead of the sum of absolute values
 */
void garprec_sum_test(int n, mpfr_t *x, int prec, int convert_digits, int repeat){
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
        hx[i].read(convert_to_string_sci(x[i], convert_digits));
    }

    //Launch
    for(int j = 0; j < repeat; j++){
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
    PrintAndResetCudaTimer("took");

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
void garprec_dot_test(int n, mpfr_t *x, mpfr_t *y, int prec, int convert_digits, int repeat){
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
        hx[i].read(convert_to_string_sci(x[i], convert_digits));
        hy[i].read(convert_to_string_sci(y[i], convert_digits));
    }

    //Launch
    for(int j = 0; j < repeat; j++){
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
    PrintAndResetCudaTimer("took");

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
void garprec_scal_test(int n, mpfr_t alpha, mpfr_t *x, int prec, int convert_digits, int repeat){
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
        lx[i].read(convert_to_string_sci(x[i], convert_digits));
    }
    lalpha[0].read(convert_to_string_sci(alpha, convert_digits));

    //Copying to the GPU
    galpha->toGPU(lalpha, 1);

    //Launch
    for(int j = 0; j < repeat; j++){
        gx->toGPU(lx, n);
        StartCudaTimer();
        garprec_scal_kernel<<<blocks, threads>>>(n, galpha->d_mpr, galpha->interval, gx->d_mpr, gx->interval, maxPrecWords);
        EndCudaTimer();
    }
    PrintAndResetCudaTimer("took");

    //Copying to the host
    gx->fromGPU(lx, n);
    for (int i = 1; i < n; i++) {
        lx[0] += lx[i];
    }
    printf("result: %.83s \n", lx[0].to_string().c_str());

    //Cleanup
    delete [] lx;
    delete [] lalpha;
    gx->release();
    galpha->release();
    garprecFinalize();
}

/*
 * AXPY test
 */
void garprec_axpy_test(int n, mpfr_t alpha, mpfr_t *x, mpfr_t *y, int prec, int convert_digits, int repeat){
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
        lx[i].read(convert_to_string_sci(x[i], convert_digits));
        ly[i].read(convert_to_string_sci(y[i], convert_digits));
    }
    lalpha[0].read(convert_to_string_sci(alpha, convert_digits));

    //Copying to the GPU
    gx->toGPU(lx, n);
    galpha->toGPU(lalpha, 1);

    //Launch
    for(int j = 0; j < repeat; j++){
        gy->toGPU(ly, n);
        StartCudaTimer();
        garprec_axpy_kernel<<<blocks, threads>>>(n, galpha->d_mpr, galpha->interval, gx->d_mpr, gx->interval, gy->d_mpr, gy->interval, gtmp->d_mpr, gtmp->interval, maxPrecWords);
        EndCudaTimer();
    }
    PrintAndResetCudaTimer("took");

    //Copying to the host
    gy->fromGPU(ly, n);
    for (int i = 1; i < n; i++) {
        ly[0] += ly[i];
    }
    printf("result: %.83s \n", ly[0].to_string().c_str());

    //Cleanup
    delete [] lx;
    delete [] ly;
    delete [] lalpha;
    gx->release();
    gy->release();
    gtmp->release();
    galpha->release();
    garprecFinalize();
}

/*
 * ROT test
 */
void garprec_rot_test(int n, mpfr_t *x, mpfr_t *y, mpfr_t c, mpfr_t s, int prec, int convert_digits, int repeat){
    Logger::printDash();
    InitCudaTimer();
    PrintTimerName("[GPU] GARPREC rot");

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
    mp_real *lc = new mp_real[1];
    mp_real *ls = new mp_real[1];

    //GPU data
    gmp_array *gx = new gmp_array(maxPrecWords, n, true);
    gmp_array *gy = new gmp_array(maxPrecWords, n, true);
    gmp_array *gs = new gmp_array(maxPrecWords, 1, true);
    gmp_array *gc = new gmp_array(maxPrecWords, 1, true);
    gmp_array *gtmp = new gmp_array(maxPrecWords, n, true);
    gmp_array *gtmp2 = new gmp_array(maxPrecWords, n, true);

    //Convert from MPFR
    #pragma omp parallel for
    for(int i = 0; i < n; i++){
        lx[i].read(convert_to_string_sci(x[i], convert_digits));
        ly[i].read(convert_to_string_sci(y[i], convert_digits));
    }
    lc[0].read(convert_to_string_sci(c, convert_digits));
    ls[0].read(convert_to_string_sci(s, convert_digits));

    //Copying to the GPU
    gc->toGPU(lc, 1);
    gs->toGPU(ls, 1);

    //Launch
    for(int j = 0; j < repeat; j++){
        gx->toGPU(lx, n);
        gy->toGPU(ly, n);
        StartCudaTimer();
        garprec_rot_kernel<<<blocks, threads>>>(
                n,
                gx->d_mpr, gx->interval,
                gy->d_mpr, gy->interval,
                gc->d_mpr, gc->interval,
                gs->d_mpr, gs->interval,
                gtmp->d_mpr, gtmp->interval,
                gtmp2->d_mpr, gtmp2->interval,
                maxPrecWords);
        EndCudaTimer();
    }
    PrintAndResetCudaTimer("took");

    //Copying to the host
    gy->fromGPU(ly, n);
    gx->fromGPU(lx, n);
    for (int i = 1; i < n; i++) {
        lx[0] += lx[i];
        ly[0] += ly[i];
    }
    printf("result x: %.83s \n", lx[0].to_string().c_str());
    printf("result y: %.83s \n", ly[0].to_string().c_str());

    //Cleanup
    delete [] lx;
    delete [] ly;
    delete [] lc;
    delete [] ls;
    gx->release();
    gy->release();
    gc->release();
    gs->release();
    gtmp->release();
    gtmp2->release();
    garprecFinalize();
}

/*
 * AXPY_DOT test
 */
void garprec_axpy_dot_test(int n, mpfr_t alpha, mpfr_t *w, mpfr_t *v, mpfr_t *u, int prec, int convert_digits, int repeat){
    Logger::printDash();
    InitCudaTimer();
    PrintTimerName("[GPU] GARPREC axpy_dot");

    //Init and set precision
    int maxPrecWords = initializeGarprec(prec);
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    //Execution configuration
    int threads = 64;
    int blocks_mul = n / threads + (n % threads ? 1 : 0);
    int blocks_red = 1024;

    //Host data
    mp_real *lalpha = new mp_real[1];
    mp_real *lv = new mp_real[n];
    mp_real *lu = new mp_real[n];
    mp_real *lw = new mp_real[n];

    //GPU data
    gmp_array *galpha = new gmp_array(maxPrecWords, 1, true);
    gmp_array *gv = new gmp_array(maxPrecWords, n, true);
    gmp_array *gu = new gmp_array(maxPrecWords, n, true);
    gmp_array *gw = new gmp_array(maxPrecWords, n, true);
    gmp_array *gtmp = new gmp_array(maxPrecWords, n, true);

    //Convert from MPFR
    #pragma omp parallel for
    for(int i = 0; i < n; i++){
        lv[i].read(convert_to_string_sci(v[i], convert_digits));
        lu[i].read(convert_to_string_sci(u[i], convert_digits));
        lw[i].read(convert_to_string_sci(w[i], convert_digits));
    }
    lalpha[0].read(convert_to_string_sci(alpha, convert_digits));
    lalpha[0] = lalpha[0] * -1.0;

    //Copying unchangeable data to the GPU
    galpha->toGPU(lalpha, 1);

    //Launch
    for(int j = 0; j < repeat; j ++){
        gv->toGPU(lv, n);
        gu->toGPU(lu, n);
        gw->toGPU(lw, n);
        garprec_reset_array<<<blocks_mul, threads>>>(n,  gtmp->d_mpr, gtmp->interval, maxPrecWords);
        StartCudaTimer();
        garprec_axpy_kernel<<<blocks_mul, threads>>>(n,
                galpha->d_mpr, galpha->interval,
                gv->d_mpr, gv->interval,
                gw->d_mpr, gw->interval,
                gv->d_mpr, gv->interval,
                maxPrecWords);
        garprec_vec_mul_kernel<<<blocks_mul, threads>>>(n, gu->d_mpr, gu->interval, gw->d_mpr, gw->interval, gu->d_mpr, gu->interval, maxPrecWords);
        garprec_sum_kernel1<<<blocks_red, threads>>>(
                n,
                gv->d_mpr,
                gv->interval,
                gu->d_mpr,
                gu->interval,
                gtmp->d_mpr,
                gtmp->interval,
                maxPrecWords);
        garprec_sum_kernel2<<<1, blocks_red>>>(
                blocks_red,
                gu->d_mpr,
                gu->interval,
                gv->d_mpr,
                gv->interval,
                maxPrecWords);
        EndCudaTimer();
    }
    PrintAndResetCudaTimer("took");

    //Copying to the host
    gu->fromGPU(lu, n);
    gw->fromGPU(lw, n);
    for(int i = 1; i < n; i ++){
        lw[0] += lw[i];
    }
    printf("result w: %.83s \n", lw[0].to_string().c_str());
    printf("result r: %.83s \n", lu[0].to_string().c_str());

    //Cleanup
    delete [] lv;
    delete [] lu;
    delete [] lw;
    delete [] lalpha;
    galpha->release();
    gv->release();
    gu->release();
    gw->release();
    gtmp->release();
    garprecFinalize();
}

/*
 * GEMV test
 */
void garprec_gemv_test(int m, int n, mpfr_t alpha, mpfr_t *A, int lda, mpfr_t *x, mpfr_t beta, mpfr_t *y, int prec, int convert_digits, int repeat){
    Logger::printDash();
    InitCudaTimer();
    PrintTimerName("[GPU] GARPREC gemv");

    //Init and set precision
    int maxPrecWords = initializeGarprec(prec);
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    //Execution configuration
    int threads = 64;
    int blocks_scal = n / (threads) + (n % (threads) ? 1 : 0);
    int blocks_gemv = m / (threads) + (m % (threads) ? 1 : 0);

    //Host data
    mp_real *hx = new mp_real[n];
    mp_real *hy = new mp_real[m];
    mp_real *hA = new mp_real[lda * n];
    mp_real *halpha = new mp_real[1];
    mp_real *hbeta = new mp_real[1];

    //GPU data
    gmp_array *dx = new gmp_array(maxPrecWords, n, true);
    gmp_array *dy = new gmp_array(maxPrecWords, m, true);
    gmp_array *dA = new gmp_array(maxPrecWords, lda * n, true);
    gmp_array *dalpha = new gmp_array(maxPrecWords, 1, true);
    gmp_array *dbeta = new gmp_array(maxPrecWords, 1, true);
    gmp_array *dtemp = new gmp_array(maxPrecWords, m, true);

    //Convert from MPFR
    #pragma omp parallel for
    for(int i = 0; i < lda * n; i++){
        hA[i].read(convert_to_string_sci(A[i], convert_digits));
        if(i < n){
            hx[i].read(convert_to_string_sci(x[i], convert_digits));
        }
        if (i < m){
            hy[i].read(convert_to_string_sci(y[i], convert_digits));
        }
    }
    halpha[0].read(convert_to_string_sci(alpha, convert_digits));
    hbeta[0].read(convert_to_string_sci(beta, convert_digits));

    //Copying to the GPU
    dA->toGPU(hA, lda * n);
    dalpha->toGPU(halpha, 1);
    dbeta->toGPU(hbeta, 1);

    //Launch
    for(int i = 0; i < repeat; i ++){
        dx->toGPU(hx, n);
        dy->toGPU(hy, m);
        garprec_reset_array<<<blocks_gemv, threads>>>(m, dtemp->d_mpr, dtemp->interval, maxPrecWords);
        StartCudaTimer();
        garprec_scal_kernel<<<blocks_scal, threads>>>(n, dalpha->d_mpr, dalpha->interval, dx->d_mpr, dx->interval, maxPrecWords);
        garprec_gemv_kernel<<<blocks_gemv, threads>>>(m, n,
                dA->d_mpr, dA->interval,
                lda,
                dx->d_mpr, dx->interval,
                dbeta->d_mpr, dbeta->interval,
                dy->d_mpr, dy->interval,
                dtemp->d_mpr, dtemp->interval,
                maxPrecWords);
        EndCudaTimer();
    }
    PrintAndResetCudaTimer("took");
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    //Copying to the host
    dy->fromGPU(hy, m);
    for(int i = 1; i < m; i++){
        hy[0] += hy[i];
    }
    printf("result: %.83s \n", hy[0].to_string().c_str());

    //Cleanup
    delete [] hx;
    delete [] hy;
    delete [] hA;
    delete [] halpha;
    delete [] hbeta;
    dx->release();
    dy->release();
    dA->release();
    dalpha->release();
    dbeta->release();
    dtemp->release();
    garprecFinalize();
}

#endif //MPRES_TEST_GARPREC_BLAS_CUH
#endif //EXCLUDE_GARPREC