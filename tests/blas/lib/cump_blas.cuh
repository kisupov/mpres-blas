/*
 *  Multiple-precision BLAS routines using CUMP as well as corresponding performance benchmarks.
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

#ifndef MPRES_TEST_CUMP_BLAS_CUH
#define MPRES_TEST_CUMP_BLAS_CUH

#include <stdio.h>
#include "mpfr.h"
#include "../../tsthelper.cuh"
#include "../../logger.cuh"
#include "../../timers.cuh"
#include "cump/cump.cuh"

using cump::mpf_array_t;

/********************* Computational kernels *********************/


/*
 * Computes the sum of the elements of vector x
 */
__global__ void cump_sum_kernel1(int n, mpf_array_t result, mpf_array_t x, mpf_array_t temp){
    using namespace cump;
    // parameters
    unsigned int tid = threadIdx.x;
    unsigned int bid = blockIdx.x;
    unsigned int bsize = blockDim.x;
    unsigned int globalIdx = bid * bsize + tid;
    unsigned int i = bid * bsize * 2 + tid;
    unsigned int k = 2 * gridDim.x * bsize;

    while (i < n) {
        mpf_add(temp[globalIdx], temp[globalIdx], x[i]);
        if (i + bsize < n){
            mpf_add(temp[globalIdx], temp[globalIdx], x[i + bsize]);
        }
        i += k;
    }
    __syncthreads();
    i = bsize;
    while(i >= 2){
        unsigned int half = i >> 1;
        if ((bsize >= i) && (tid < half) && (globalIdx + half < n)) {
            mpf_add(temp[globalIdx], temp[globalIdx], temp[globalIdx + half]);
        }
        i = i >> 1;
        __syncthreads();
    }
    // write result for this block to global mem
    if (tid == 0) {
        mpf_set(result[bid], temp[globalIdx]);
    };
    __syncthreads();
}

/*
 * Computes the sum of the elements of vector x (optimized kernel)
 */
__global__ void cump_sum_kernel2(mpf_array_t x, mpf_array_t result){
    using namespace cump;
    unsigned int tid = threadIdx.x;
    for(unsigned int s = blockDim.x / 2; s > 0; s >>= 1){
        if(tid < s){
            mpf_add(x[tid], x[tid], x[tid + s]);
        }
        __syncthreads();
    }
    // write result for this block to global mem
    if (tid == 0) {
        mpf_set(result[0], x[tid]);
    }
}

/*
 * Computes the componentwise vector-vector product
 */
__global__ void cump_vec_mul_kernel(int n, mpf_array_t result, mpf_array_t x, mpf_array_t y) {
    using namespace cump;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    while (idx < n) {
        mpf_mul(result[idx], y[idx], x[idx]);
        idx += gridDim.x * blockDim.x;
    }
}

/*
 * Multiplies a scalar by a vector
 */
__global__ void cump_scal_kernel(int n, mpf_array_t alpha, mpf_array_t x) {
    using namespace cump;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    while (idx < n) {
        mpf_mul(x[idx], alpha[0], x[idx]);
        idx += gridDim.x * blockDim.x;
    }
}

/*
 * Constant times a vector plus a vector
 */
__global__  void cump_axpy_kernel(int n, mpf_array_t a, mpf_array_t X, mpf_array_t Y) {
    using namespace cump;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    while (idx < n) {
        mpf_mul(X[idx], a[0], X[idx]);
        mpf_add(Y[idx], X[idx], Y[idx]);
        idx += gridDim.x * blockDim.x;
    }
}

/*
 * Performs rotation of points in the plane
 */
__global__  void cump_rot_kernel(int n, mpf_array_t x, mpf_array_t y, mpf_array_t c, mpf_array_t s, mpf_array_t buffer1, mpf_array_t buffer2) {
    using namespace cump;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    while (idx < n) {
        //perform c * x
        mpf_mul(buffer1[idx], c[0], x[idx]);
        //perform s * y
        mpf_mul(buffer2[idx], s[0], y[idx]);
        //perform y = c * y - s * x
        mpf_mul(x[idx], x[idx], s[0]);
        mpf_mul(y[idx], y[idx], c[0]);
        mpf_sub(y[idx], y[idx], x[idx]);
        //perform x = c * x + s * y
        mpf_add(x[idx], buffer1[idx], buffer2[idx]);
        idx += gridDim.x * blockDim.x;
    }
}


/*
 * Set the elements of an array to zero
 */
__global__ void cump_reset_array(int n, mpf_array_t temp) {
    using namespace cump;
    int numberIdx =  blockDim.x * blockIdx.x + threadIdx.x;
    while (numberIdx < n) {
        mpf_sub(temp[numberIdx], temp[numberIdx], temp[numberIdx]); // set to zero
        numberIdx +=  gridDim.x * blockDim.x;
    }
}


/********************* Benchmarks *********************/

/*
 * SUM test
 * Note that the sum is calculated instead of the sum of absolute values
 */
void cump_sum_test(int n, mpfr_t *x, int prec, int convert_digits, int repeat){
    InitCudaTimer();
    Logger::printDash();
    PrintTimerName("[GPU] CUMP sum");

    //Set precision
    mpf_set_default_prec(prec);
    cumpf_set_default_prec(prec);

    //Execution configuration
    int threads = 64;
    int blocks = 1024;

    //Host data
    mpf_t *hx = new mpf_t[n];
    mpf_t hresult;

    //GPU data
    cumpf_array_t dx;
    cumpf_array_t dresult;
    cumpf_array_t dtemp;
    cumpf_array_t dblock_result;

    cumpf_array_init2(dx, n, prec);
    cumpf_array_init2(dresult, 1, prec);
    cumpf_array_init2(dtemp, n, prec);
    cumpf_array_init2(dblock_result, blocks, prec);

    //Convert from MPFR
    for(int i = 0; i < n; i ++){
        mpf_init2(hx[i], prec);
        mpf_set_str(hx[i], convert_to_string_sci(x[i], convert_digits).c_str(), 10);
    }
    mpf_init2(hresult, prec);
    mpf_set_d(hresult, 0);

    //Copying to the GPU
    cumpf_array_set_mpf(dx, hx, n);

    //Launch
    for(int i = 0; i < repeat; i ++){
        cump_reset_array<<<blocks, threads>>>(n, dtemp);
        StartCudaTimer();
        cump_sum_kernel1<<<blocks, threads>>>(n, dblock_result, dx, dtemp);
        cump_sum_kernel2<<<1, blocks>>>(dblock_result, dresult);
        EndCudaTimer();
    }
    PrintCudaTimer("took");

    //Copying to the host
    mpf_array_set_cumpf(&hresult, dresult, 1);
    gmp_printf ("result: %.70Ff \n", hresult);

    //Cleanup
    mpf_clear(hresult);
    for(int i = 0; i < n; i ++){
        mpf_clear(hx[i]);
    }
    delete[] hx;
    cumpf_array_clear(dx);
    cumpf_array_clear(dresult);
    cumpf_array_clear(dblock_result);
    cumpf_array_clear(dtemp);
}


/*
 * DOT test
 */
void cump_dot_test(int n, mpfr_t *x, mpfr_t *y, int prec, int convert_digits, int repeat){
    InitCudaTimer();
    Logger::printDash();
    PrintTimerName("[GPU] CUMP dot");

    //Set precision
    mpf_set_default_prec(prec);
    cumpf_set_default_prec(prec);

    //Execution configuration
    int threads = 64;
    int blocks_mul = n / threads + (n % threads ? 1 : 0);
    int blocks_red = 1024;

    //Host data
    mpf_t *hx = new mpf_t[n];
    mpf_t *hy = new mpf_t[n];
    mpf_t hresult;

    //GPU data
    cumpf_array_t dx;
    cumpf_array_t dy;
    cumpf_array_t dresult;
    cumpf_array_t dvecprod;
    cumpf_array_t dtemp;
    cumpf_array_t dblock_result;

    cumpf_array_init2(dx, n, prec);
    cumpf_array_init2(dy, n, prec);
    cumpf_array_init2(dresult, 1, prec);
    cumpf_array_init2(dvecprod, n, prec);
    cumpf_array_init2(dtemp, n, prec);
    cumpf_array_init2(dblock_result, blocks_red, prec);

    //Convert from MPFR
    for(int i = 0; i < n; i ++){
        mpf_init2(hx[i], prec);
        mpf_init2(hy[i], prec);
        mpf_set_str(hx[i], convert_to_string_sci(x[i], convert_digits).c_str(), 10);
        mpf_set_str(hy[i], convert_to_string_sci(y[i], convert_digits).c_str(), 10);
    }
    mpf_init2(hresult, prec);
    mpf_set_d(hresult, 0);

    //Copying to the GPU
    cumpf_array_set_mpf(dx, hx, n);
    cumpf_array_set_mpf(dy, hy, n);

    //Launch
    for(int i = 0; i < repeat; i ++){
        cump_reset_array<<<blocks_mul, threads>>>(n, dtemp);
        StartCudaTimer();
        cump_vec_mul_kernel<<<blocks_mul, threads>>>(n, dvecprod, dx, dy);
        cump_sum_kernel1<<<blocks_red, threads>>>(n, dblock_result, dvecprod, dtemp);
        cump_sum_kernel2<<<1, blocks_red>>>(dblock_result, dresult);
        EndCudaTimer();
    }
    PrintCudaTimer("took");

    //Copying to the host
    mpf_array_set_cumpf(&hresult, dresult, 1);
    gmp_printf ("result: %.70Ff \n", hresult);

    //Cleanup
    mpf_clear(hresult);
    for(int i = 0; i < n; i ++){
        mpf_clear(hx[i]);
        mpf_clear(hy[i]);
    }
    delete [] hx;
    delete [] hy;
    cumpf_array_clear(dx);
    cumpf_array_clear(dy);
    cumpf_array_clear(dresult);
    cumpf_array_clear(dvecprod);
    cumpf_array_clear(dblock_result);
    cumpf_array_clear(dtemp);
}

/*
 * SCAL test
 */
void cump_scal_test(int n, mpfr_t alpha, mpfr_t *x, int prec, int convert_digits, int repeat){
    Logger::printDash();
    InitCudaTimer();
    PrintTimerName("[GPU] CUMP scal");

    //Set precision
    mpf_set_default_prec(prec);
    cumpf_set_default_prec(prec);

    //Execution configuration
    int threads = 64;
    int blocks = n / threads + (n % threads ? 1 : 0);

    //Host data
    mpf_t *hx = new mpf_t[n];
    mpf_t halpha;

    //GPU data
    cumpf_array_t dx;
    cumpf_array_t dalpha;

    cumpf_array_init2(dx, n, prec);
    cumpf_array_init2(dalpha, 1, prec);

    //Convert from MPFR
    for(int i = 0; i < n; i ++){
        mpf_init2(hx[i], prec);
        mpf_set_str(hx[i], convert_to_string_sci(x[i], convert_digits).c_str(), 10);
    }
    mpf_init2(halpha, prec);
    mpf_set_str(halpha, convert_to_string_sci(alpha, convert_digits).c_str(), 10);

    //Copying alpha to the GPU
    cumpf_array_set_mpf(dalpha, &halpha, 1);

    //Launch
    for(int i = 0; i < repeat; i ++){
        cumpf_array_set_mpf(dx, hx, n);
        cudaDeviceSynchronize();
        StartCudaTimer();
        cump_scal_kernel<<<blocks, threads>>>(n, dalpha, dx);
        EndCudaTimer();
    }
    PrintCudaTimer("took");

    //Copying to the host
    mpf_array_set_cumpf(hx, dx, n);
    for(int i = 1; i < n; i ++){
        mpf_add(hx[0], hx[i], hx[0]);
    }
    gmp_printf ("result: %.70Ff \n", hx[0]);

    //Cleanup
    mpf_clear(halpha);
    for(int i = 0; i < n; i ++){
        mpf_clear(hx[i]);
    }
    delete [] hx;
    cumpf_array_clear(dalpha);
    cumpf_array_clear(dx);
}

/*
 * AXPY test
 */
void cump_axpy_test(int n, mpfr_t alpha, mpfr_t *x, mpfr_t *y, int prec, int convert_digits, int repeat){
    Logger::printDash();
    InitCudaTimer();
    PrintTimerName("[GPU] CUMP axpy");

    //Set precision
    mpf_set_default_prec(prec);
    cumpf_set_default_prec(prec);

    //Execution configuration
    int threads = 64;
    int blocks = n / threads + (n % threads ? 1 : 0);

    //Host data
    mpf_t *hx = new mpf_t[n];
    mpf_t *hy = new mpf_t[n];
    mpf_t halpha;

    //GPU data
    cumpf_array_t dx;
    cumpf_array_t dy;
    cumpf_array_t dalpha;

    cumpf_array_init2(dx, n, prec);
    cumpf_array_init2(dy, n, prec);
    cumpf_array_init2(dalpha, 1, prec);

    //Convert from MPFR
    for(int i = 0; i < n; i ++){
        mpf_init2(hx[i], prec);
        mpf_init2(hy[i], prec);
        mpf_set_str(hx[i], convert_to_string_sci(x[i], convert_digits).c_str(), 10);
        mpf_set_str(hy[i], convert_to_string_sci(y[i], convert_digits).c_str(), 10);
    }
    mpf_init2(halpha, prec);
    mpf_set_str(halpha, convert_to_string_sci(alpha, convert_digits).c_str(), 10);

    //Copying alpha to the GPU
    cumpf_array_set_mpf(dalpha, &halpha, 1);

    //Launch
    for(int i = 0; i < repeat; i ++){
        cumpf_array_set_mpf(dx, hx, n);
        cumpf_array_set_mpf(dy, hy, n);
        cudaDeviceSynchronize();
        StartCudaTimer();
        cump_axpy_kernel<<<blocks, threads>>>(n, dalpha, dx, dy);
        EndCudaTimer();
    }
    PrintCudaTimer("took");

    //Copying to the host
    mpf_array_set_cumpf(hy, dy, n);
    for(int i = 1; i < n; i ++){
        mpf_add(hy[0], hy[i], hy[0]);
    }
    gmp_printf ("result: %.70Ff \n", hy[0]);

    //Cleanup
    mpf_clear(halpha);
    for(int i = 0; i < n; i ++){
        mpf_clear(hx[i]);
        mpf_clear(hy[i]);
    }
    delete [] hx;
    delete [] hy;
    cumpf_array_clear(dx);
    cumpf_array_clear(dy);
    cumpf_array_clear(dalpha);
}

/*
 * ROT test
 */
void cump_rot_test(int n, mpfr_t * x, mpfr_t * y, mpfr_t c, mpfr_t s, int prec, int convert_digits, int repeat) {
    Logger::printDash();
    InitCudaTimer();
    PrintTimerName("[GPU] CUMP rot");

    //Set precision
    mpf_set_default_prec(prec);
    cumpf_set_default_prec(prec);

    //Execution configuration
    int threads = 64;
    int blocks = n / threads + (n % threads ? 1 : 0);

    //Host data
    mpf_t *hx = new mpf_t[n];
    mpf_t *hy = new mpf_t[n];
    mpf_t hs;
    mpf_t hc;

    //GPU data
    cumpf_array_t dx;
    cumpf_array_t dy;
    cumpf_array_t ds;
    cumpf_array_t dc;
    cumpf_array_t dbuffer1;
    cumpf_array_t dbuffer2;

    cumpf_array_init2(dx, n, prec);
    cumpf_array_init2(dy, n, prec);
    cumpf_array_init2(dbuffer1, n, prec);
    cumpf_array_init2(dbuffer2, n, prec);
    cumpf_array_init2(ds, 1, prec);
    cumpf_array_init2(dc, 1, prec);

    //Convert from MPFR
    for(int i = 0; i < n; i++){
        mpf_init2(hx[i], prec);
        mpf_init2(hy[i], prec);
        mpf_set_str(hx[i], convert_to_string_sci(x[i], convert_digits).c_str(), 10);
        mpf_set_str(hy[i], convert_to_string_sci(y[i], convert_digits).c_str(), 10);
    }
    mpf_init2(hs, prec);
    mpf_set_str(hs, convert_to_string_sci(s, convert_digits).c_str(), 10);
    mpf_init2(hc, prec);
    mpf_set_str(hc, convert_to_string_sci(c, convert_digits).c_str(), 10);

    //Copying alpha to the GPU
    cumpf_array_set_mpf(ds, &hs, 1);
    cumpf_array_set_mpf(dc, &hc, 1);

    //Launch
    for(int i = 0; i < repeat; i++){
        cumpf_array_set_mpf(dx, hx, n);
        cumpf_array_set_mpf(dy, hy, n);
        cudaDeviceSynchronize();
        StartCudaTimer();
        cump_rot_kernel<<<blocks, threads>>>(n, dx, dy, dc, ds, dbuffer1, dbuffer2);
        EndCudaTimer();
    }
    PrintCudaTimer("took");

    //Copying to the host
    mpf_array_set_cumpf(hx, dx, n);
    mpf_array_set_cumpf(hy, dy, n);
    for(int i = 1; i < n; i++){
        mpf_add(hx[0], hx[i], hx[0]);
        mpf_add(hy[0], hy[i], hy[0]);
    }
    gmp_printf ("result x: %.70Ff \n", hx[0]);
    gmp_printf ("result y: %.70Ff \n", hy[0]);

    //Cleanup
    mpf_clear(hc);
    mpf_clear(hs);
    for(int i = 0; i < n; i++){
        mpf_clear(hx[i]);
        mpf_clear(hy[i]);
    }
    delete [] hx;
    delete [] hy;
    cumpf_array_clear(dx);
    cumpf_array_clear(dy);
    cumpf_array_clear(ds);
    cumpf_array_clear(dc);
    cumpf_array_clear(dbuffer1);
    cumpf_array_clear(dbuffer2);
}


#endif //MPRES_TEST_CUMP_BLAS_CUH