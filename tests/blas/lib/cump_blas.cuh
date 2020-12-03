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

#ifndef EXCLUDE_CUMP
#ifndef MPRES_TEST_CUMP_BLAS_CUH
#define MPRES_TEST_CUMP_BLAS_CUH

#include "blas/mblas_enum.cuh"
#include "../../tsthelper.cuh"
#include "../../logger.cuh"
#include "../../timers.cuh"
#include "3rdparty/cump_common.cuh"

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
__global__ void
__launch_bounds__(CUMP_MAX_THREADS_PER_BLOCK)
cump_sum_kernel2(mpf_array_t x, mpf_array_t result){
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
 * Computes the element-wise vector-vector product
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
 * Performs the matrix-vector operation  y := A*x + beta*y,
 * where beta is a scalar, x and y are vectors and A is an m by n matrix
 */
__global__ void cump_gemv_kernel(int m, int n, mpf_array_t alpha, mpf_array_t A, int lda, mpf_array_t x, mpf_array_t beta, mpf_array_t y, mpf_array_t tmp1)  {
    using namespace cump;
    unsigned int threadId = threadIdx.x + blockIdx.x * blockDim.x;
    if (threadId < m) {
        mpf_mul(y[threadId], beta[0], y[threadId]);
        for (int colId = 0; colId < n; colId++) {
            mpf_mul(tmp1[threadId], x[colId], A[colId * lda + threadId]);
            mpf_add(y[threadId], y[threadId], tmp1[threadId]);
        }
    }
}

/*
* Computes a matrix-matrix product with general matrices.
* C = alpha * A * B + beta * C
* where alpha and beta are scalars, A, B, and C are matrices.
* All the matrices should be stored in column-major order.
 */
__global__ void cump_gemm_kernel(int m, int n, int k, mpf_array_t alpha, mpf_array_t A, int lda, mpf_array_t B, int ldb, mpf_array_t beta, mpf_array_t C, int ldc, mpf_array_t buf1, mpf_array_t buf2) {
    using namespace cump;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int indexC = row + col * ldc;
    if(col < n && row < m){
        for(int i = 0; i < k; i++){
            mpf_mul(buf1[indexC], alpha[0], A[lda * i + row]);
            mpf_mul(buf1[indexC], B[col * ldb + i], buf1[indexC]);
            mpf_add(buf2[indexC], buf1[indexC], buf2[indexC]);
        }
        mpf_mul(C[indexC], beta[0], C[indexC]);
        mpf_add(C[indexC], buf2[indexC], C[indexC]);
    }
}

/*
 * Performs the matrix-vector operation  A := x*y^T + A,
 * x and y are vectors and A is an lda by n matrix
 */
__global__ void cump_ger_kernel(int m, int n, mpf_array_t A, int lda, mpf_array_t x, mpf_array_t y, mpf_array_t tmp1) {
    using namespace cump;
    int j = blockIdx.y; // The column index
    while (j < n){
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        if( i < m ){
            mpf_mul(tmp1[i + j * m], x[i], y[j]);
            mpf_add(A[i + j * lda], A[i + j * lda], tmp1[i + j * m]);
            i += gridDim.x * blockDim.x;
        }
        __syncthreads();
        j += gridDim.y;
    }
}

/*
* Scales two matrices A and B and stores their sum in a matrix C
* C = alpha * A + beta * B
* where alpha and beta are scalars, and A, B, C are m by n matrices.
* All the matrices should be stored in column-major order.
 */
__global__ void cump_ge_add_kernel(int m, int n, mpf_array_t alpha, mpf_array_t A, int lda, mpf_array_t beta, mpf_array_t B, int ldb, mpf_array_t C, int ldc, mpf_array_t buf) {
    using namespace cump;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < n && row < m) {
        mpf_mul(C[row + col * ldc], beta[0], B[row + col * ldb]);
        mpf_mul(buf[row + col * m], alpha[0], A[row + col * lda]);
        mpf_add(C[row + col * ldc], buf[row + col * m], C[row + col * ldc]);
    }
}

/*
* Scales a matrix A and scales a matrix B and accumulates the result in the matrix B
* B = alpha * A + beta * B
* where alpha and beta are scalars, and A and B are matrices.
* All the matrices should be stored in column-major order.
 */
__global__ void cump_ge_acc_kernel(int m, int n, mpf_array_t alpha, mpf_array_t A, int lda, mpf_array_t beta, mpf_array_t B, int ldb, mpf_array_t buf) {
    using namespace cump;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < n && row < m) {
        mpf_mul(B[row + col * ldb], beta[0], B[row + col * ldb]);
        mpf_mul(buf[row + col * m], alpha[0], A[row + col * lda]);
        mpf_add(B[row + col * ldb], buf[row + col * m], B[row + col * ldb]);
    }
}


/*
* Scales a general matrix A on the right side or by a diagonal matrix D: A = AD
 */
__global__ void cump_ge_diag_scale_r_kernel(int m, int n, mpf_array_t D, int incd, mpf_array_t A, int lda) {
    using namespace cump;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int indexA = row + col * lda;
    int indexD = incd > 0 ? col * incd : (-n + col + 1)*incd;
    if (col < n && row < m) {
        mpf_mul(A[indexA], A[indexA], D[indexD]);
    }
}

/*
* Scales a general matrix A on the left side or by a diagonal matrix D: A = DA
 */
__global__ void cump_ge_diag_scale_l_kernel(int m, int n, mpf_array_t D, int incd, mpf_array_t A, int lda) {
    using namespace cump;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int indexA = row + col * lda;
    int indexD = incd > 0 ? row * incd : (-m + row + 1)*incd;
    if (col < n && row < m) {
        mpf_mul(A[indexA], A[indexA], D[indexD]);
    }
}

/*
* Scales a general matrix A on the left side by a diagonal matrix DL and on the right side by a diagonal matrix DR: A = DL * A * DR
 */
__global__ void cump_ge_lrscale_kernel(int m, int n, mpf_array_t DL, int incdl, mpf_array_t DR, int incdr, mpf_array_t A, int lda) {
    using namespace cump;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int indexA = row + col * lda;
    int indexDL = incdl > 0 ? row * incdl : (-m + row + 1)*incdl;
    int indexDR = incdr > 0 ? col * incdr : (-n + col + 1)*incdr;
    if (col < n && row < m) {
        mpf_mul(A[indexA], A[indexA], DL[indexDL]);
        mpf_mul(A[indexA], A[indexA], DR[indexDR]);
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

/*
 * AXPY_DOT test
 */
void cump_axpy_dot_test(int n, mpfr_t alpha, mpfr_t *w, mpfr_t *v, mpfr_t *u, int prec, int convert_digits, int repeat){
    Logger::printDash();
    InitCudaTimer();
    PrintTimerName("[GPU] CUMP axpy_dot");

    //Set precision
    mpf_set_default_prec(prec);
    cumpf_set_default_prec(prec);

    //Execution configuration
    int threads = 32;
    int blocks_mul = n / threads + (n % threads ? 1 : 0);
    int blocks_red = 1024;

    //Host data
    mpf_t *hv = new mpf_t[n];
    mpf_t *hu = new mpf_t[n];
    mpf_t *hw = new mpf_t[n];
    mpf_t halpha;
    mpf_t hr;

    //GPU data
    cumpf_array_t dv;
    cumpf_array_t du;
    cumpf_array_t dw;
    cumpf_array_t dr;
    cumpf_array_t dalpha;
    cumpf_array_t dvecprod;
    cumpf_array_t dtemp;
    cumpf_array_t dblock_result;

    cumpf_array_init2(dv, n, prec);
    cumpf_array_init2(du, n, prec);
    cumpf_array_init2(dw, n, prec);
    cumpf_array_init2(dr, 1, prec);
    cumpf_array_init2(dalpha, 1, prec);
    cumpf_array_init2(dvecprod, n, prec);
    cumpf_array_init2(dtemp, n, prec);
    cumpf_array_init2(dblock_result, blocks_red, prec);

    //Convert from MPFR
    for(int i = 0; i < n; i ++){
        mpf_init2(hv[i], prec);
        mpf_init2(hu[i], prec);
        mpf_init2(hw[i], prec);
        mpf_set_str(hv[i], convert_to_string_sci(v[i], convert_digits).c_str(), 10);
        mpf_set_str(hu[i], convert_to_string_sci(u[i], convert_digits).c_str(), 10);
        mpf_set_str(hw[i], convert_to_string_sci(w[i], convert_digits).c_str(), 10);
    }
    mpf_init2(halpha, prec);
    mpf_init2(hr, prec);
    mpf_set_str(halpha, convert_to_string_sci(alpha, convert_digits).c_str(), 10);
    mpf_set_d(hr, 0);

    //Multiplication alpha by minus 1
    mpf_t minus1;
    mpf_init2(minus1, prec);
    mpf_set_d(minus1, -1.0);
    mpf_mul(halpha, minus1, halpha);

    //Copying alpha to the GPU
    cumpf_array_set_mpf(dalpha, &halpha, 1);
    cumpf_array_set_mpf(du, hu, n);

    //Launch
    for(int i = 0; i < repeat; i ++){
        cumpf_array_set_mpf(dv, hv, n);
        cumpf_array_set_mpf(dw, hw, n);
        cump_reset_array<<<blocks_mul, threads>>>(n, dtemp);
        cudaDeviceSynchronize();
        StartCudaTimer();
        cump_axpy_kernel<<<blocks_mul, threads>>>(n, dalpha, dv, dw);
        cump_vec_mul_kernel<<<blocks_mul, threads>>>(n, dvecprod, du, dw);
        cump_sum_kernel1<<<blocks_red, threads>>>(n, dblock_result, dvecprod, dtemp);
        cump_sum_kernel2<<<1, blocks_red>>>(dblock_result, dr);
        EndCudaTimer();
    }
    PrintCudaTimer("took");

    //Copying to the host
    mpf_array_set_cumpf(&hr, dr, 1);
    mpf_array_set_cumpf(hw, dw, n);

    for(int i = 1; i < n; i ++){
        mpf_add(hw[0], hw[0], hw[i]);
    }
    gmp_printf ("result w: %.70Ff \n", hw[0]);
    gmp_printf ("result r: %.70Ff \n", hr);

    //Cleanup
    mpf_clear(halpha);
    for(int i = 0; i < n; i ++){
        mpf_clear(hv[i]);
        mpf_clear(hu[i]);
        mpf_clear(hw[i]);
    }
    delete [] hv;
    delete [] hu;
    delete [] hw;
    cumpf_array_clear(dv);
    cumpf_array_clear(du);
    cumpf_array_clear(dw);
    cumpf_array_clear(dr);
    cumpf_array_clear(dalpha);
    cumpf_array_clear(dvecprod);
    cumpf_array_clear(dtemp);
    cumpf_array_clear(dblock_result);
}

/*
 * GEMV test
 */
void cump_gemv_test(int m, int n, mpfr_t alpha, mpfr_t *A, int lda, mpfr_t *x, mpfr_t beta, mpfr_t *y, int prec, int convert_digits, int repeats){
    Logger::printDash();
    InitCudaTimer();
    PrintTimerName("[GPU] CUMP gemv");

    //Set precision
    mpf_set_default_prec(prec);
    cumpf_set_default_prec(prec);

    //Execution configuration
    int threads = 64;
    int blocks_scal = n / (threads) + (n % (threads) ? 1 : 0);
    int blocks_gemv = m / (threads) + (m % (threads) ? 1 : 0);

    //Host data
    mpf_t halpha;
    mpf_t hbeta;
    mpf_t *hA = new mpf_t[lda * n];
    mpf_t *hx = new mpf_t[n];
    mpf_t *hy = new mpf_t[m];

    //GPU data
    cumpf_array_t dalpha;
    cumpf_array_t dbeta;
    cumpf_array_t dA;
    cumpf_array_t dx;
    cumpf_array_t dy;
    cumpf_array_t dtemp;

    cumpf_array_init2(dalpha, 1, prec);
    cumpf_array_init2(dbeta, 1, prec);
    cumpf_array_init2(dA, lda * n, prec);
    cumpf_array_init2(dx, n, prec);
    cumpf_array_init2(dy, m, prec);
    cumpf_array_init2(dtemp, m, prec);

    //Convert from MPFR
    for(int i = 0; i < lda * n; i++){
        mpf_init2(hA[i], prec);
        mpf_set_str(hA[i], convert_to_string_sci(A[i], convert_digits).c_str(), 10);
    }
    for(int i = 0; i < n; i++){
        mpf_init2(hx[i], prec);
        mpf_set_str(hx[i], convert_to_string_sci(x[i], convert_digits).c_str(), 10);
    }
    for(int i = 0; i < m; i++){
        mpf_init2(hy[i], prec);
        mpf_set_str(hy[i], convert_to_string_sci(y[i], convert_digits).c_str(), 10);
    }
    mpf_init2(halpha, prec);
    mpf_init2(hbeta, prec);
    mpf_set_str(halpha, convert_to_string_sci(alpha, convert_digits).c_str(), 10);
    mpf_set_str(hbeta, convert_to_string_sci(beta, convert_digits).c_str(), 10);

    //Copying to the GPU
    cumpf_array_set_mpf(dalpha, &halpha, 1);
    cumpf_array_set_mpf(dbeta, &hbeta, 1);
    cumpf_array_set_mpf(dA, hA, lda * n);

    //Launch
    for(int i = 0; i < repeats; i++){
        cumpf_array_set_mpf(dy, hy, m);
        cumpf_array_set_mpf(dx, hx, n);
        cudaDeviceSynchronize();
        StartCudaTimer();
        cump_scal_kernel<<<blocks_scal, threads>>>(n, dalpha, dx);
        cump_gemv_kernel<<<blocks_gemv, threads>>>(m, n, dalpha, dA, lda, dx, dbeta, dy, dtemp);
        EndCudaTimer();
    }
    PrintCudaTimer("took");

    //Copying to the host
    mpf_array_set_cumpf(hy, dy, m);
    for(int i = 1; i < m; i++){
        mpf_add(hy[0], hy[i], hy[0]);
    }
    gmp_printf ("result: %.70Ff \n", hy[0]);

    //Cleanup
    mpf_clear(halpha);
    mpf_clear(hbeta);
    for(int i = 0; i < lda * n; i++){
        mpf_clear(hA[i]);
    }
    for(int i = 0; i < n; i++){
        mpf_clear(hx[i]);
    }
    for(int i = 0; i < m; i++){
        mpf_clear(hy[i]);
    }
    delete [] hA;
    delete [] hx;
    delete [] hy;
    cumpf_array_clear(dalpha);
    cumpf_array_clear(dbeta);
    cumpf_array_clear(dA);
    cumpf_array_clear(dx);
    cumpf_array_clear(dy);
    cumpf_array_clear(dtemp);
}

/*
 * GER test
 */
void cump_ger_test(int m, int n, mpfr_t alpha, mpfr_t *A, int lda, mpfr_t *x, mpfr_t *y, int prec, int convert_digits, int repeats) {
    Logger::printDash();InitCudaTimer();
    PrintTimerName("[GPU] CUMP ger");

    //Set precision
    mpf_set_default_prec(prec);
    cumpf_set_default_prec(prec);

    //Execution configuration
    int threads = 64;
    int blocks_scal = n / (threads) + (n % (threads) ? 1 : 0);
    dim3 blocks_ger(m / (threads) + (m % (threads) ? 1 : 0), n, 1);

    //Host data
    mpf_t halpha;
    mpf_t *hA = new mpf_t[lda * n];
    mpf_t *hx = new mpf_t[m];
    mpf_t *hy = new mpf_t[n];

    //GPU data
    cumpf_array_t dalpha;
    cumpf_array_t dA;
    cumpf_array_t dx;
    cumpf_array_t dy;
    cumpf_array_t dtemp;

    cumpf_array_init2(dalpha, 1, prec);
    cumpf_array_init2(dA, lda * n, prec);
    cumpf_array_init2(dx, m, prec);
    cumpf_array_init2(dy, n, prec);
    cumpf_array_init2(dtemp, m * n, prec);

    //Convert from MPFR
    for (int i = 0; i < lda * n; i++) {
        mpf_init2(hA[i], prec);
        mpf_set_str(hA[i], convert_to_string_sci(A[i], convert_digits).c_str(), 10);
    }
    for (int i = 0; i < m; i++) {
        mpf_init2(hx[i], prec);
        mpf_set_str(hx[i], convert_to_string_sci(x[i], convert_digits).c_str(), 10);
    }
    for (int i = 0; i < n; i++) {
        mpf_init2(hy[i], prec);
        mpf_set_str(hy[i], convert_to_string_sci(y[i], convert_digits).c_str(), 10);
    }
    mpf_init2(halpha, prec);
    mpf_set_str(halpha, convert_to_string_sci(alpha, convert_digits).c_str(), 10);

    //Copying to the GPU
    cumpf_array_set_mpf(dalpha, &halpha, 1);
    cumpf_array_set_mpf(dx, hx, m);

    //Launch
    for (int i = 0; i < repeats; i++) {
        cumpf_array_set_mpf(dy, hy, n);
        cumpf_array_set_mpf(dA, hA, lda * n);
        cudaDeviceSynchronize();
        StartCudaTimer();
        cump_scal_kernel <<<blocks_scal, threads>>> (n, dalpha, dy);
        cump_ger_kernel <<<blocks_ger, threads>>> (m, n, dA, lda, dx, dy, dtemp);
        EndCudaTimer();
    }
    PrintCudaTimer("took");

    //Copying to the host
    mpf_array_set_cumpf(hA, dA, lda * n);
    for (int i = 1; i < lda * n; i++) {
        mpf_add(hA[0], hA[i], hA[0]);
    }
    gmp_printf("result: %.70Ff \n", hA[0]);

    //Cleanup
    mpf_clear(halpha);
    for (int i = 0; i < lda * n; i++) {
        mpf_clear(hA[i]);
    }
    for (int i = 0; i < m; i++) {
        mpf_clear(hx[i]);
    }
    for (int i = 0; i < n; i++) {
        mpf_clear(hy[i]);
    }
    delete[] hA;
    delete[] hx;
    delete[] hy;
    cumpf_array_clear(dalpha);
    cumpf_array_clear(dA);
    cumpf_array_clear(dx);
    cumpf_array_clear(dy);
    cumpf_array_clear(dtemp);
}


/*
 * GEMM test
 */
void cump_gemm_test(int m, int n, int k, mpfr_t alpha, mpfr_t *A, int lda, mpfr_t *B, int ldb, mpfr_t beta, mpfr_t *C, int ldc, int prec, int convert_prec, int repeats) {
    Logger::printDash();
    InitCudaTimer();
    PrintTimerName("[GPU] CUMP gemm");

    //Set precision
    mpf_set_default_prec(prec);
    cumpf_set_default_prec(prec);

    //Execution configuration
    dim3 dimBlock(32, 16);
    dim3 dimGrid((n + dimBlock.x - 1) / dimBlock.x, (m + dimBlock.y - 1) / dimBlock.y);

    //Host data
    mpf_t halpha;
    mpf_t hbeta;
    mpf_t *hA = new mpf_t[lda * k];
    mpf_t *hB = new mpf_t[ldb * n];
    mpf_t *hC = new mpf_t[ldc * n];

    //GPU data
    cumpf_array_t dalpha;
    cumpf_array_t dbeta;
    cumpf_array_t dA;
    cumpf_array_t dB;
    cumpf_array_t dC;
    cumpf_array_t dbuf1;
    cumpf_array_t dbuf2;

    cumpf_array_init2(dalpha, 1, prec);
    cumpf_array_init2(dbeta, 1, prec);
    cumpf_array_init2(dA, lda * k, prec);
    cumpf_array_init2(dB, ldb * n, prec);
    cumpf_array_init2(dC, ldc * n, prec);
    cumpf_array_init2(dbuf1, m * n, prec);
    cumpf_array_init2(dbuf2, m * n, prec);

    //Convert from MPFR
    #pragma omp parallel for
    for (int i = 0; i < lda * k; i++) {
        mpf_init2(hA[i], prec);
        mpf_set_str(hA[i], convert_to_string_sci(A[i], convert_prec).c_str(), 10);
    }
    #pragma omp parallel for
    for (int i = 0; i < ldb * n; i++) {
        mpf_init2(hB[i], prec);
        mpf_set_str(hB[i], convert_to_string_sci(B[i], convert_prec).c_str(), 10);
    }
    #pragma omp parallel for
    for (int i = 0; i < ldc * n; i++) {
        mpf_init2(hC[i], prec);
        mpf_set_str(hC[i], convert_to_string_sci(C[i], convert_prec).c_str(), 10);
    }

    mpf_init2(halpha, prec);
    mpf_set_str(halpha, convert_to_string_sci(alpha, convert_prec).c_str(), 10);
    mpf_init2(hbeta, prec);
    mpf_set_str(hbeta, convert_to_string_sci(beta, convert_prec).c_str(), 10);

    //Copying to the GPU
    cumpf_array_set_mpf(dalpha, &halpha, 1);
    cumpf_array_set_mpf(dbeta, &hbeta, 1);
    cumpf_array_set_mpf(dA, hA, lda * k);
    cumpf_array_set_mpf(dB, hB, ldb * n);

    //Launch
    for (int i = 0; i < repeats; i++) {
        cumpf_array_set_mpf(dC, hC, ldc * n);
        cump_reset_array<<<1024, 64>>>(m *n, dbuf2);
        StartCudaTimer();
        cump_gemm_kernel <<<dimGrid, dimBlock>>> (m, n, k, dalpha, dA, lda, dB, ldb, dbeta, dC, ldc, dbuf1, dbuf2);
        EndCudaTimer();
    }
    PrintCudaTimer("took");

    //Copying to the host
    mpf_array_set_cumpf(hC, dC, ldc * n);
    for (int i = 1; i < ldc * n; i++) {
        mpf_add(hC[0], hC[i], hC[0]);
    }
    gmp_printf("result: %.70Ff \n", hC[0]);

    //Cleanup
    mpf_clear(halpha);
    mpf_clear(hbeta);
    for (int i = 0; i < lda * k; i++) {
        mpf_clear(hA[i]);
    }
    for (int i = 0; i < ldb * n; i++) {
        mpf_clear(hB[i]);
    }
    for (int i = 0; i < ldc * n; i++) {
        mpf_clear(hC[i]);
    }

    delete[] hA;
    delete[] hB;
    delete[] hC;
    cumpf_array_clear(dalpha);
    cumpf_array_clear(dbeta);
    cumpf_array_clear(dA);
    cumpf_array_clear(dB);
    cumpf_array_clear(dC);
    cumpf_array_clear(dbuf1);
    cumpf_array_clear(dbuf2);
}

/*
 * GE_ADD test
 */
void cump_ge_add_test(int m, int n, mpfr_t alpha, mpfr_t *A, int lda, mpfr_t beta, mpfr_t *B, int ldb, mpfr_t *C, int ldc, int prec, int convert_prec, int repeats) {
    Logger::printDash();
    InitCudaTimer();
    PrintTimerName("[GPU] CUMP ge_add");

    //Set precision
    mpf_set_default_prec(prec);
    cumpf_set_default_prec(prec);

    //Execution configuration
    dim3 dimBlock(32, 16);
    dim3 dimGrid((n + dimBlock.x - 1) / dimBlock.x, (m + dimBlock.y - 1) / dimBlock.y);

    //Host data
    mpf_t halpha;
    mpf_t hbeta;
    mpf_t *hA = new mpf_t[lda * n];
    mpf_t *hB = new mpf_t[ldb * n];
    mpf_t *hC = new mpf_t[ldc * n];

    //GPU data
    cumpf_array_t dalpha;
    cumpf_array_t dbeta;
    cumpf_array_t dA;
    cumpf_array_t dB;
    cumpf_array_t dC;
    cumpf_array_t dbuf;

    cumpf_array_init2(dalpha, 1, prec);
    cumpf_array_init2(dbeta, 1, prec);
    cumpf_array_init2(dA, lda * n, prec);
    cumpf_array_init2(dB, ldb * n, prec);
    cumpf_array_init2(dC, ldc * n, prec);
    cumpf_array_init2(dbuf, m * n, prec);

    //Convert from MPFR
    for (int i = 0; i < lda * n; i++) {
        mpf_init2(hA[i], prec);
        mpf_set_str(hA[i], convert_to_string_sci(A[i], convert_prec).c_str(), 10);
    }
    for (int i = 0; i < ldb * n; i++) {
        mpf_init2(hB[i], prec);
        mpf_set_str(hB[i], convert_to_string_sci(B[i], convert_prec).c_str(), 10);
    }
    for (int i = 0; i < ldc * n; i++) {
        mpf_init2(hC[i], prec);
        mpf_set_str(hC[i], convert_to_string_sci(C[i], convert_prec).c_str(), 10);
    }
    mpf_init2(halpha, prec);
    mpf_set_str(halpha, convert_to_string_sci(alpha, convert_prec).c_str(), 10);
    mpf_init2(hbeta, prec);
    mpf_set_str(hbeta, convert_to_string_sci(beta, convert_prec).c_str(), 10);

    //Copying to the GPU
    cumpf_array_set_mpf(dalpha, &halpha, 1);
    cumpf_array_set_mpf(dbeta, &hbeta, 1);
    cumpf_array_set_mpf(dA, hA, lda * n);
    cumpf_array_set_mpf(dB, hB, ldb * n);
    cumpf_array_set_mpf(dC, hC, ldc * n);

    //Launch
    for (int i = 0; i < repeats; i++) {
        StartCudaTimer();
        cump_ge_add_kernel <<<dimGrid, dimBlock>>> (m, n, dalpha, dA, lda, dbeta, dB, ldb, dC, ldc, dbuf);
        EndCudaTimer();
    }
    PrintCudaTimer("took");

    //Copying to the host
    mpf_array_set_cumpf(hC, dC, ldc * n);
    for (int i = 1; i < ldc * n; i++) {
        mpf_add(hC[0], hC[i], hC[0]);
    }
    gmp_printf("result: %.70Ff \n", hC[0]);

    //Cleanup
    mpf_clear(halpha);
    mpf_clear(hbeta);
    for (int i = 0; i < lda * n; i++) {
        mpf_clear(hA[i]);
    }
    for (int i = 0; i < ldb * n; i++) {
        mpf_clear(hB[i]);
    }
    for (int i = 0; i < ldc * n; i++) {
        mpf_clear(hC[i]);
    }
    delete[] hA;
    delete[] hB;
    delete[] hC;
    cumpf_array_clear(dalpha);
    cumpf_array_clear(dbeta);
    cumpf_array_clear(dA);
    cumpf_array_clear(dB);
    cumpf_array_clear(dC);
    cumpf_array_clear(dbuf);
}

/*
 * GE_ACC test
 */
void cump_ge_acc_test(int m, int n, mpfr_t alpha, mpfr_t *A, int lda, mpfr_t beta, mpfr_t *B, int ldb, int prec, int convert_prec, int repeats) {
    Logger::printDash();
    InitCudaTimer();
    PrintTimerName("[GPU] CUMP ge_acc");

    //Set precision
    mpf_set_default_prec(prec);
    cumpf_set_default_prec(prec);

    //Execution configuration
    dim3 dimBlock(32, 16);
    dim3 dimGrid((n + dimBlock.x - 1) / dimBlock.x, (m + dimBlock.y - 1) / dimBlock.y);

    //Host data
    mpf_t halpha;
    mpf_t hbeta;
    mpf_t *hA = new mpf_t[lda * n];
    mpf_t *hB = new mpf_t[ldb * n];

    //GPU data
    cumpf_array_t dalpha;
    cumpf_array_t dbeta;
    cumpf_array_t dA;
    cumpf_array_t dB;
    cumpf_array_t dbuf;

    cumpf_array_init2(dalpha, 1, prec);
    cumpf_array_init2(dbeta, 1, prec);
    cumpf_array_init2(dA, lda * n, prec);
    cumpf_array_init2(dB, ldb * n, prec);
    cumpf_array_init2(dbuf, m * n, prec);

    //Convert from MPFR
    for (int i = 0; i < lda * n; i++) {
        mpf_init2(hA[i], prec);
        mpf_set_str(hA[i], convert_to_string_sci(A[i], convert_prec).c_str(), 10);
    }
    for (int i = 0; i < ldb * n; i++) {
        mpf_init2(hB[i], prec);
        mpf_set_str(hB[i], convert_to_string_sci(B[i], convert_prec).c_str(), 10);
    }
    mpf_init2(halpha, prec);
    mpf_set_str(halpha, convert_to_string_sci(alpha, convert_prec).c_str(), 10);
    mpf_init2(hbeta, prec);
    mpf_set_str(hbeta, convert_to_string_sci(beta, convert_prec).c_str(), 10);

    //Copying to the GPU
    cumpf_array_set_mpf(dalpha, &halpha, 1);
    cumpf_array_set_mpf(dbeta, &hbeta, 1);
    cumpf_array_set_mpf(dA, hA, lda * n);

    //Launch
    for (int i = 0; i < repeats; i++) {
        cumpf_array_set_mpf(dB, hB, ldb * n);
        StartCudaTimer();
        cump_ge_acc_kernel <<<dimGrid, dimBlock>>> (m, n, dalpha, dA, lda, dbeta, dB, ldb, dbuf);
        EndCudaTimer();
    }
    PrintCudaTimer("took");

    //Copying to the host
    mpf_array_set_cumpf(hB, dB, ldb * n);
    for (int i = 1; i < ldb * n; i++) {
        mpf_add(hB[0], hB[i], hB[0]);
    }
    gmp_printf("result: %.70Ff \n", hB[0]);

    //Cleanup
    mpf_clear(halpha);
    mpf_clear(hbeta);
    for (int i = 0; i < lda * n; i++) {
        mpf_clear(hA[i]);
    }
    for (int i = 0; i < ldb * n; i++) {
        mpf_clear(hB[i]);
    }
    delete[] hA;
    delete[] hB;
    cumpf_array_clear(dalpha);
    cumpf_array_clear(dbeta);
    cumpf_array_clear(dA);
    cumpf_array_clear(dB);
    cumpf_array_clear(dbuf);
}


/*
 * GE_DIAG_SCALE test
 */
void cump_ge_diag_scale_test(enum  mblas_side_type side, int m, int n, int lend, mpfr_t *D, int incd, mpfr_t *A, int lda, int prec, int convert_prec, int repeats) {
    Logger::printDash();
    InitCudaTimer();
    PrintTimerName("[GPU] CUMP ge_diag_scale");

    //Set precision
    mpf_set_default_prec(prec);
    cumpf_set_default_prec(prec);

    //Execution configuration
    dim3 dimBlock(64, 1);
    dim3 dimGrid((n + dimBlock.x - 1) / dimBlock.x, (m + dimBlock.y - 1) / dimBlock.y);

    //Host data
    mpf_t *hA = new mpf_t[lda * n];
    mpf_t *hD = new mpf_t[lend];

    //GPU data
    cumpf_array_t dA;
    cumpf_array_t dD;

    cumpf_array_init2(dA, lda * n, prec);
    cumpf_array_init2(dD, lend, prec);

    //Convert from MPFR
    for (int i = 0; i < lda * n; i++) {
        mpf_init2(hA[i], prec);
        mpf_set_str(hA[i], convert_to_string_sci(A[i], convert_prec).c_str(), 10);
    }
    for (int i = 0; i < lend; i++) {
        mpf_init2(hD[i], prec);
        mpf_set_str(hD[i], convert_to_string_sci(D[i], convert_prec).c_str(), 10);
    }
    //Copying to the GPU
    cumpf_array_set_mpf(dD, hD, lend);
    //Launch
    if (side == mblas_right_side){
        for (int i = 0; i < repeats; i++) {
            cumpf_array_set_mpf(dA, hA, lda * n);
            StartCudaTimer();
            cump_ge_diag_scale_r_kernel <<<dimGrid, dimBlock>>> (m, n, dD, incd, dA, lda);
            EndCudaTimer();
        }
    } else {
        for (int i = 0; i < repeats; i++) {
            cumpf_array_set_mpf(dA, hA, lda * n);StartCudaTimer();
            cump_ge_diag_scale_l_kernel <<< dimGrid, dimBlock >>> (m, n, dD, incd, dA, lda);EndCudaTimer();
        }
    }
    PrintCudaTimer("took");

    //Copying to the host
    mpf_array_set_cumpf(hA, dA, lda * n);
    for (int i = 1; i < lda * n; i++) {
        mpf_add(hA[0], hA[i], hA[0]);
    }
    gmp_printf("result: %.70Ff \n", hA[0]);

    //Cleanup
    for (int i = 0; i < lda * n; i++) {
        mpf_clear(hA[i]);
    }
    for (int i = 0; i < lend; i++) {
        mpf_clear(hD[i]);
    }
    delete[] hA;
    delete[] hD;
    cumpf_array_clear(dA);
    cumpf_array_clear(dD);
}

/*
 * GE_LRSCALE test
 */
void cump_ge_lrscale_test(int m, int n, mpfr_t *DL, int incdl, mpfr_t *DR, int incdr, mpfr_t *A, int lda, int prec, int convert_prec, int repeats) {
    Logger::printDash();
    InitCudaTimer();
    PrintTimerName("[GPU] CUMP ge_lrscale");

    int lendl = (1 + (m - 1) * abs(incdl));
    int lendr = (1 + (n - 1) * abs(incdr));

    //Set precision
    mpf_set_default_prec(prec);
    cumpf_set_default_prec(prec);

    //Execution configuration
    dim3 dimBlock(64, 1);
    dim3 dimGrid((n + dimBlock.x - 1) / dimBlock.x, (m + dimBlock.y - 1) / dimBlock.y);

    //Host data
    mpf_t *hA = new mpf_t[lda * n];
    mpf_t *hDL = new mpf_t[lendl];
    mpf_t *hDR = new mpf_t[lendr];

    //GPU data
    cumpf_array_t dA;
    cumpf_array_t dDL;
    cumpf_array_t dDR;

    cumpf_array_init2(dA, lda * n, prec);
    cumpf_array_init2(dDL, lendl, prec);
    cumpf_array_init2(dDR, lendr, prec);

    //Convert from MPFR
    for (int i = 0; i < lda * n; i++) {
        mpf_init2(hA[i], prec);
        mpf_set_str(hA[i], convert_to_string_sci(A[i], convert_prec).c_str(), 10);
    }
    for (int i = 0; i < lendl; i++) {
        mpf_init2(hDL[i], prec);
        mpf_set_str(hDL[i], convert_to_string_sci(DL[i], convert_prec).c_str(), 10);
    }
    for (int i = 0; i < lendr; i++) {
        mpf_init2(hDR[i], prec);
        mpf_set_str(hDR[i], convert_to_string_sci(DR[i], convert_prec).c_str(), 10);
    }
    //Copying to the GPU
    cumpf_array_set_mpf(dDL, hDL, lendl);
    cumpf_array_set_mpf(dDR, hDR, lendr);
    //Launch
    for (int i = 0; i < repeats; i++) {
        cumpf_array_set_mpf(dA, hA, lda * n);
        StartCudaTimer();
        cump_ge_lrscale_kernel <<<dimGrid, dimBlock>>> (m, n, dDL, incdl, dDR, incdr, dA, lda);
        EndCudaTimer();
    }
    PrintCudaTimer("took");

    //Copying to the host
    mpf_array_set_cumpf(hA, dA, lda * n);
    for (int i = 1; i < lda * n; i++) {
        mpf_add(hA[0], hA[i], hA[0]);
    }
    gmp_printf("result: %.70Ff \n", hA[0]);

    //Cleanup
    for (int i = 0; i < lda * n; i++) {
        mpf_clear(hA[i]);
    }
    for (int i = 0; i < lendl; i++) {
        mpf_clear(hDL[i]);
    }
    for (int i = 0; i < lendr; i++) {
        mpf_clear(hDR[i]);
    }
    delete[] hA;
    delete[] hDL;
    delete[] hDR;
    cumpf_array_clear(dA);
    cumpf_array_clear(dDL);
    cumpf_array_clear(dDR);
}

#endif //MPRES_TEST_CUMP_BLAS_CUH
#endif //EXCLUDE_CUMP