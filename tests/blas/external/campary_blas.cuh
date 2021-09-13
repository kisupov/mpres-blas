/*
 *  Multiple-precision BLAS routines using CAMPARY as well as corresponding performance benchmarks.
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

#ifndef EXCLUDE_CAMPARY
#ifndef MPRES_TEST_CAMPARY_BLAS_CUH
#define MPRES_TEST_CAMPARY_BLAS_CUH

#include "blas/mblas_enum.cuh"
#include "tsthelper.cuh"
#include "logger.cuh"
#include "timers.cuh"
#include "lib/campary_common.cuh"

/********************* Computational kernels *********************/

/*
 * Computes the sum of absolute values of the elements of vector x
 */
template<int prec>
__global__ void campary_asum_kernel(int n, multi_prec<prec> *x, multi_prec<prec> *result, const unsigned int nextPow2) {

    __shared__ multi_prec<prec> sdata[CAMPARY_REDUCTION_THREADS];

    const unsigned int tid = threadIdx.x;
    const unsigned int bid = blockIdx.x;
    const unsigned int bsize = blockDim.x;
    const unsigned int k = gridDim.x * bsize;
    unsigned int i = bid * bsize + tid;

    // do reduction in global mem
    sdata[tid] = 0.0;
    while (i < n) {
        sdata[tid] += abs(x[i]);
        i += k;
    }
    __syncthreads();
    // do reduction in shared mem
    i = nextPow2 >> 1; // half of nextPow2
    while(i >= 1){
        if ((tid < i) && (tid + i < bsize)) {
            sdata[tid] += sdata[tid + i];
        }
        i = i >> 1;
        __syncthreads();
    }
    // write result for this block to global mem
    if (tid == 0) {
        result[bid] = sdata[tid];
    };
    __syncthreads();
}

/*
 * Computes the sum of the elements of vector x
 */
template<int prec>
__global__ void campary_sum_kernel(int n, multi_prec<prec> *x, multi_prec<prec> *result, const unsigned int nextPow2) {

    __shared__ multi_prec<prec> sdata[CAMPARY_REDUCTION_THREADS];

    const unsigned int tid = threadIdx.x;
    const unsigned int bid = blockIdx.x;
    const unsigned int bsize = blockDim.x;
    const unsigned int k = gridDim.x * bsize;
    unsigned int i = bid * bsize + tid;

    // do reduction in global mem
    sdata[tid] = 0.0;
    while (i < n) {
        sdata[tid] += x[i];
        i += k;
    }
    __syncthreads();
    // do reduction in shared mem
    i = nextPow2 >> 1; // half of nextPow2
    while(i >= 1){
        if ((tid < i) && (tid + i < bsize)) {
            sdata[tid] += sdata[tid + i];
        }
        i = i >> 1;
        __syncthreads();
    }
    // write result for this block to global mem
    if (tid == 0) {
        result[bid] = sdata[tid];
    };
    __syncthreads();
}

/*
 * Computes the inner product of two vectors
 * For final reduction, campary_sum_kernel should be used
 */
template<int prec>
__global__ void campary_dot_kernel(int n, multi_prec<prec> *x, multi_prec<prec> *y, multi_prec<prec> *result, const unsigned int nextPow2) {

    __shared__ multi_prec<prec> sdata[CAMPARY_REDUCTION_THREADS];
    multi_prec<prec> mul_res;

    const unsigned int tid = threadIdx.x;
    const unsigned int bid = blockIdx.x;
    const unsigned int bsize = blockDim.x;
    const unsigned int k = gridDim.x * bsize;
    unsigned int i = bid * bsize + tid;

    // do reduction in global mem
    sdata[tid] = 0.0;
    while (i < n) {
        mul_res = x[i] * y[i];
        sdata[tid] += mul_res;
        i += k;
    }
    __syncthreads();
    // do reduction in shared mem
    i = nextPow2 >> 1; // half of nextPow2
    while(i >= 1){
        if ((tid < i) && (tid + i < bsize)) {
            sdata[tid] += sdata[tid + i];
        }
        i = i >> 1;
        __syncthreads();
    }
    // write result for this block to global mem
    if (tid == 0) {
        result[bid] = sdata[tid];
    };
    __syncthreads();
}

/*
 * Multiplies a scalar by a vector
 */
template<int prec>
__global__ void campary_scal_kernel(multi_prec<prec> *alpha, multi_prec<prec> *x, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if( index < n){
        x[index] *= alpha[0];
    }
}

/*
 * Constant times a vector plus a vector
 */
template<int prec>
__global__ void campary_axpy_kernel(multi_prec<prec> *alpha, multi_prec<prec> *x, multi_prec<prec> *y, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    multi_prec<prec> mul;
    if( index < n){
        mul = x[index] * alpha[0];
        y[index] += mul;
    }
}

/*
 * Performs rotation of points in the plane
 */
template<int prec>
__global__ void campary_rot_kernel(multi_prec<prec> *x, multi_prec<prec> *y, multi_prec<prec> *c, multi_prec<prec> *s, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    multi_prec<prec> temp;
    if( index < n){
        temp = c[0] * x[index] + s[0] * y[index];
        y[index] = c[0] * y[index] - s[0] * x[index];
        x[index] = temp;
    }
}

/*
 * Combines an axpy and a dot product and stores the results of each block in the result array
 * For final reduction, campary_sum_kernel should be used
 */
template<int prec>
__global__ void campary_axpy_dot_kernel(int n, multi_prec<prec> *alpha, multi_prec<prec> *w, multi_prec<prec> *v,  multi_prec<prec> *u, multi_prec<prec> *result,  const unsigned int nextPow2) {

    __shared__ multi_prec<prec> sdata[CAMPARY_REDUCTION_THREADS];
    multi_prec<prec> tmp;

    const unsigned int tid = threadIdx.x;
    const unsigned int bid = blockIdx.x;
    const unsigned int bsize = blockDim.x;
    const unsigned int k = gridDim.x * bsize;
    unsigned int i = bid * bsize + tid;

    sdata[tid] = 0.0;
    while (i < n) {
        tmp = w[i] - alpha[0] * v[i];
        w[i] = tmp;
        sdata[tid] += u[i] * tmp;
        i += k;
    }
    __syncthreads();
    // do reduction in shared mem
    i = nextPow2 >> 1; // half of nextPow2
    while(i >= 1){
        if ((tid < i) && (tid + i < bsize)) {
            sdata[tid] += sdata[tid + i];
        }
        i = i >> 1;
        __syncthreads();
    }
    // write result for this block to global mem
    if (tid == 0) {
        result[bid] = sdata[tid];
    };
    __syncthreads();

}

/*
 * Performs the matrix-vector operation  y := A*x + beta*y,
 * where beta is a scalar, x and y are vectors and A is an m by n matrix
 */
template<int prec>
__global__ void campary_gemv_kernel(int m, int n, multi_prec<prec> *A, int lda, multi_prec<prec> *x, multi_prec<prec> *beta, multi_prec<prec> *y) {
    unsigned int threadId = threadIdx.x + blockIdx.x * blockDim.x;
    if (threadId < m) {
        multi_prec<prec> dot = 0.0;
        for (int colId = 0; colId < n; colId++) {
            dot = dot + A[colId * lda + threadId] * x[colId];
        }
        y[threadId] = beta[0] * y[threadId];
        y[threadId] = y[threadId] + dot;
    }
}

/*
* Computes a matrix-matrix product with general matrices.
* C = alpha * A * B + beta * C
* where alpha and beta are scalars, A, B, and C are matrices.
* All the matrices should be stored in column-major order.
 */
template<int prec>
__global__ void campary_gemm_kernel(int m, int n, int k, multi_prec<prec> * alpha, multi_prec<prec> * A, const int lda, multi_prec<prec> * B, const int ldb, multi_prec<prec> * beta, multi_prec<prec> * C, const int ldc) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int indexC = row + col * ldc;
    if(col < n && row < m){
        multi_prec<prec> sum;
        sum = 0.0;
        for(int i = 0; i < k; i++){
            sum = sum + (alpha * A[lda * i + row] * B[col * ldb + i]);
        }
        C[indexC] = beta * C[indexC] + sum;
    }
}

/*
* Scales two matrices A and B and stores their sum in a matrix C
* C = alpha * A + beta * B
* where alpha and beta are scalars, and A, B, C are m by n matrices.
* All the matrices should be stored in column-major order.
 */
template<int prec>
__global__ void campary_ge_add_kernel(int m, int n, multi_prec<prec> * alpha, multi_prec<prec> * A, int lda, multi_prec<prec> * beta, multi_prec<prec> * B, int ldb, multi_prec<prec> * C, int ldc) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int indexA = row + col * lda;
    int indexB = row + col * ldb;
    int indexC = row + col * ldc;
    if (col < n && row < m) {
        C[indexC] = alpha * A[indexA] + beta * B[indexB];
    }
}

/*
* Scales a matrix A and scales a matrix B and accumulates the result in the matrix B
* B = alpha * A + beta * B
* where alpha and beta are scalars, and A and B are matrices.
* All the matrices should be stored in column-major order.
 */
template<int prec>
__global__ void campary_ge_acc_kernel(int m, int n, multi_prec<prec> * alpha, multi_prec<prec> * A, int lda, multi_prec<prec> * beta, multi_prec<prec> * B, int ldb) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int indexA = row + col * lda;
    int indexB = row + col * ldb;
    if (col < n && row < m) {
        B[indexB] = alpha * A[indexA] + beta * B[indexB];
    }
}

/*
* Scales a general matrix A on the right side or by a diagonal matrix D: A = AD
 */
template<int prec>
__global__ void campary_ge_diag_scale_r_kernel(int m, int n, multi_prec<prec> * D, int incd, multi_prec<prec> * A, int lda) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int indexA = row + col * lda;
    int indexD = incd > 0 ? col * incd : (-n + col + 1)*incd;
    if (col < n && row < m) {
        A[indexA] = D[indexD] * A[indexA];
    }
}

/*
* Scales a general matrix A on the left side or by a diagonal matrix D: A = DA
 */
template<int prec>
__global__ void campary_ge_diag_scale_l_kernel(int m, int n, multi_prec<prec> * D, int incd, multi_prec<prec> * A, int lda) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int indexA = row + col * lda;
    int indexD = incd > 0 ? row * incd : (-m + row + 1)*incd;
    if (col < n && row < m) {
        A[indexA] = D[indexD] * A[indexA];
    }
}

/*
* Scales a general matrix A on the left side by a diagonal matrix DL and on the right side by a diagonal matrix DR: A = DL * A * DR
 */
template<int prec>
__global__ void campary_ge_lrscale_kernel(int m, int n, multi_prec<prec> * DL, int incdl, multi_prec<prec> * DR, int incdr, multi_prec<prec> * A, int lda) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int indexA = row + col * lda;
    int indexDL = incdl > 0 ? row * incdl : (-m + row + 1)*incdl;
    int indexDR = incdr > 0 ? col * incdr : (-n + col + 1)*incdr;
    if (col < n && row < m) {
        A[indexA] = DL[indexDL] * A[indexA] * DR[indexDR];
    }
}

/********************* BLAS functions *********************/

/*
 * ASUM
 */
template <int prec>
void campary_asum(int n, multi_prec<prec> *x, multi_prec<prec> *result) {
    multi_prec<prec> *d_buf; // device buffer

    // Allocate memory buffers for the device results
    cudaMalloc((void **) &d_buf, sizeof(multi_prec<prec>) * CAMPARY_REDUCTION_BLOCKS);

    // Power of two that is greater that or equals to CAMPARY_REDUCTION_THREADS
    const unsigned int POW = nextPow2(CAMPARY_REDUCTION_THREADS);

    // Kernel memory configurations. We prefer shared memory
    cudaFuncSetCacheConfig(campary_asum_kernel <prec>, cudaFuncCachePreferShared);

    //Launch the 1st CUDA kernel
    campary_asum_kernel <prec> <<< CAMPARY_REDUCTION_BLOCKS, CAMPARY_REDUCTION_THREADS >>> (n, x, d_buf, POW);

    //Launch the 2nd CUDA kernel to perform summation of the results of parallel blocks on the GPU
    campary_asum_kernel <prec> <<< 1, CAMPARY_REDUCTION_THREADS >>> (CAMPARY_REDUCTION_BLOCKS, d_buf, result, POW);

    // Cleanup
    cudaFree(d_buf);
}

/*
 * DOT
 */
template <int prec>
void campary_dot(int n, multi_prec<prec> *x, multi_prec<prec> *y, multi_prec<prec> *result) {
    multi_prec<prec> *d_buf; // device buffer

    // Allocate memory buffers for the device results
    cudaMalloc((void **) &d_buf, sizeof(multi_prec<prec>) * CAMPARY_REDUCTION_BLOCKS);

    // Power of two that is greater that or equals to CAMPARY_REDUCTION_THREADS
    const unsigned int POW = nextPow2(CAMPARY_REDUCTION_THREADS);

    // Kernel memory configurations. We prefer shared memory
    cudaFuncSetCacheConfig(campary_asum_kernel <prec>, cudaFuncCachePreferShared);
    cudaFuncSetCacheConfig(campary_dot_kernel <prec>, cudaFuncCachePreferShared);

    //Launch the 1st CUDA kernel
    campary_dot_kernel <prec> <<< CAMPARY_REDUCTION_BLOCKS, CAMPARY_REDUCTION_THREADS >>> (n, x, y, d_buf, POW);

    //Launch the 2nd CUDA kernel to perform summation of the results of parallel blocks on the GPU
    campary_sum_kernel <prec> <<< 1, CAMPARY_REDUCTION_THREADS >>> (CAMPARY_REDUCTION_BLOCKS, d_buf, result, POW);

    // Cleanup
    cudaFree(d_buf);
}

/*
 * SCAL
 */
template <int prec>
void campary_scal(int n, multi_prec<prec> *alpha, multi_prec<prec> *x){
    int BLOCKS = n / CAMPARY_VECTOR_MULTIPLY_THREADS + 1;
    campary_scal_kernel <prec> <<<BLOCKS, CAMPARY_VECTOR_MULTIPLY_THREADS>>>(alpha, x, n);
}

/*
 * AXPY
 */
template <int prec>
void campary_axpy(int n, multi_prec<prec> *alpha, multi_prec<prec> *x, multi_prec<prec> *y){
    int BLOCKS = n / CAMPARY_VECTOR_MULTIPLY_THREADS + 1;
    campary_axpy_kernel <prec> <<<BLOCKS, CAMPARY_VECTOR_MULTIPLY_THREADS>>>(alpha, x, y, n);
}

/*
 * ROT
 */
template <int prec>
void campary_rot(int n, multi_prec<prec> *x, multi_prec<prec> *y, multi_prec<prec> *c, multi_prec<prec> *s){
    int BLOCKS = n / CAMPARY_VECTOR_MULTIPLY_THREADS + 1;
    campary_rot_kernel <prec> <<<BLOCKS, CAMPARY_VECTOR_MULTIPLY_THREADS>>>(x, y, c, s, n);
}

/*
 * AXPY_DOT
 */
template <int prec>
void campary_axpy_dot(int n, multi_prec<prec> *alpha, multi_prec<prec> *w, multi_prec<prec> *v,  multi_prec<prec> *u, multi_prec<prec> *r) {
    multi_prec<prec> *d_buf; // device buffer

    // Allocate memory buffers for the device results
    cudaMalloc((void **) &d_buf, sizeof(multi_prec<prec>) * CAMPARY_REDUCTION_BLOCKS);

    // Power of two that is greater that or equals to CAMPARY_REDUCTION_THREADS
    const unsigned int POW = nextPow2(CAMPARY_REDUCTION_THREADS);

    // Kernel memory configurations. We prefer shared memory
    cudaFuncSetCacheConfig(campary_asum_kernel <prec>, cudaFuncCachePreferShared);
    cudaFuncSetCacheConfig(campary_dot_kernel <prec>, cudaFuncCachePreferShared);

    //Launch the 1st CUDA kernel
    campary_axpy_dot_kernel <prec> <<< CAMPARY_REDUCTION_BLOCKS, CAMPARY_REDUCTION_THREADS >>> (n, alpha, w, v, u, d_buf, POW);

    //Launch the 2nd CUDA kernel to perform summation of the results of parallel blocks on the GPU
    campary_sum_kernel <prec> <<< 1, CAMPARY_REDUCTION_THREADS >>> (CAMPARY_REDUCTION_BLOCKS, d_buf, r, POW);

    // Cleanup
    cudaFree(d_buf);
}

/*
 * GEMM
 */
template <int prec>
void campary_gemm(const unsigned int m, const unsigned int n, const unsigned int k,
                  multi_prec<prec> * alpha, multi_prec<prec> * A, const int lda, multi_prec<prec> * B, const int ldb,
                  multi_prec<prec> * beta, multi_prec<prec> * C, const int ldc){
    dim3 dimBlock(CAMPARY_MATRIX_THREADS_X, CAMPARY_MATRIX_THREADS_Y);
    dim3 dimGrid((n + dimBlock.x - 1) / dimBlock.x, (m + dimBlock.y - 1) / dimBlock.y);
    campary_gemm_kernel <prec> <<<dimGrid, dimBlock>>>(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

/*
 * GE_ADD
 */
template <int prec>
void campary_ge_add(int m, int n, multi_prec<prec> * alpha, multi_prec<prec> * A, int lda, multi_prec<prec> * beta, multi_prec<prec> * B, int ldb, multi_prec<prec> * C, int ldc){
    dim3 dimBlock(CAMPARY_MATRIX_THREADS_X, CAMPARY_MATRIX_THREADS_Y);
    dim3 dimGrid((n + dimBlock.x - 1) / dimBlock.x, (m + dimBlock.y - 1) / dimBlock.y);
    campary_ge_add_kernel <prec> <<<dimGrid, dimBlock>>>(m, n, alpha, A, lda, beta, B, ldb, C, ldc);
}

/*
 * GE_ACC
 */
template <int prec>
void campary_ge_acc(int m, int n, multi_prec<prec> * alpha, multi_prec<prec> * A, int lda, multi_prec<prec> * beta, multi_prec<prec> * B, int ldb){
    dim3 dimBlock(CAMPARY_MATRIX_THREADS_X, CAMPARY_MATRIX_THREADS_Y);
    dim3 dimGrid((n + dimBlock.x - 1) / dimBlock.x, (m + dimBlock.y - 1) / dimBlock.y);
    campary_ge_acc_kernel <prec> <<<dimGrid, dimBlock>>>(m, n, alpha, A, lda, beta, B, ldb);
}

/*
 * GE_DIAG_SCALE
 */
template <int prec>
void campary_ge_diag_scale(enum mblas_side_type side, int m, int n, multi_prec<prec> * D, int incd, multi_prec<prec> * A, int lda){
    dim3 dimBlock(CAMPARY_MATRIX_THREADS_X, CAMPARY_MATRIX_THREADS_Y);
    dim3 dimGrid((n + dimBlock.x - 1) / dimBlock.x, (m + dimBlock.y - 1) / dimBlock.y);
    if(side == mblas_right_side){
        campary_ge_diag_scale_r_kernel <prec> <<<dimGrid, dimBlock>>>(m, n, D, incd, A, lda);
    } else{
        campary_ge_diag_scale_l_kernel <prec> <<<dimGrid, dimBlock>>>(m, n, D, incd, A, lda);
    }
}

/*
 * GE_LRSCALE
 */
template <int prec>
void campary_ge_lrscale(int m, int n, multi_prec<prec> * DL, int incdl, multi_prec<prec> * DR, int incdr, multi_prec<prec> * A, int lda){
    dim3 dimBlock(CAMPARY_MATRIX_THREADS_X, CAMPARY_MATRIX_THREADS_Y);
    dim3 dimGrid((n + dimBlock.x - 1) / dimBlock.x, (m + dimBlock.y - 1) / dimBlock.y);
    campary_ge_lrscale_kernel <prec> <<<dimGrid, dimBlock>>>(m, n, DL, incdl, DR, incdr, A, lda);
}

/********************* Benchmarks *********************/

/*
 * ASUM test
 */
template <int prec>
void campary_asum_test(int n, mpfr_t *x, int convert_digits, int repeats) {
    Logger::printDash();
    InitCudaTimer();
    PrintTimerName("[GPU] CAMPARY asum");

    //Host data
    multi_prec<prec> *hx = new multi_prec<prec>[n];
    multi_prec<prec> *hres = new multi_prec<prec>[1];

    //GPU data
    multi_prec<prec> *dx;
    multi_prec<prec> *dres;

    cudaMalloc(&dx, sizeof(multi_prec<prec>) * n);
    cudaMalloc(&dres, sizeof(multi_prec<prec>));

    //Convert from MPFR
    #pragma omp parallel for
    for(int i = 0; i < n; i ++){
        hx[i] = convert_to_string_sci(x[i], convert_digits).c_str();
    }

    //Copying data to the GPU
    cudaMemcpy(dx, hx, sizeof(multi_prec<prec>) * n, cudaMemcpyHostToDevice);

    //Launch
    StartCudaTimer();
    for(int i = 0; i < repeats; i ++){
        campary_asum<prec>(n, dx, dres);
    }
    EndCudaTimer();
    PrintCudaTimer("took");
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    //Copying to the host
    cudaMemcpy(hres, dres, sizeof(multi_prec<prec>), cudaMemcpyDeviceToHost);
    printResult<prec>(hres[0]);

    //Cleanup
    delete [] hx;
    delete [] hres;
    cudaFree(dres);
    cudaFree(dx);
}


/*
 * DOT test
 */
template <int prec>
void campary_dot_test(int n, mpfr_t *x, mpfr_t *y, int convert_digits, int repeats){
    Logger::printDash();
    InitCudaTimer();
    PrintTimerName("[GPU] CAMPARY dot");

    //Host data
    multi_prec<prec> *hx = new multi_prec<prec>[n];
    multi_prec<prec> *hy = new multi_prec<prec>[n];
    multi_prec<prec> *hres = new multi_prec<prec>[1];

    //GPU data
    multi_prec<prec> *dx;
    multi_prec<prec> *dy;
    multi_prec<prec> *dres;

    cudaMalloc(&dx, sizeof(multi_prec<prec>) * n);
    cudaMalloc(&dy, sizeof(multi_prec<prec>) * n);
    cudaMalloc(&dres, sizeof(multi_prec<prec>));

    //Convert from MPFR
    #pragma omp parallel for
    for(int i = 0; i < n; i ++){
        hx[i] = convert_to_string_sci(x[i], convert_digits).c_str();
        hy[i] = convert_to_string_sci(y[i], convert_digits).c_str();
    }

    //Copying data to the GPU
    cudaMemcpy(dx, hx, sizeof(multi_prec<prec>) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(dy, hy, sizeof(multi_prec<prec>) * n, cudaMemcpyHostToDevice);
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    //Launch
    StartCudaTimer();
    for(int i = 0; i < repeats; i ++){
        campary_dot<prec>(n, dx, dy, dres);
    }
    EndCudaTimer();
    PrintCudaTimer("took");

    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    //Copying to the host
    cudaMemcpy(hres, dres, sizeof(multi_prec<prec>), cudaMemcpyDeviceToHost);
    printResult<prec>(hres[0]);

    //Cleanup
    delete [] hx;
    delete [] hy;
    delete [] hres;

    cudaFree(dx);
    cudaFree(dy);
    cudaFree(dres);
}


/*
 * SCAL test
 */
template<int prec>
void campary_scal_test(int n, mpfr_t alpha, mpfr_t *x, int convert_digits, int repeats){
    Logger::printDash();
    InitCudaTimer();
    PrintTimerName("[GPU] CAMPARY scal");

    //Host data
    multi_prec<prec> halpha;
    multi_prec<prec> *hx = new multi_prec<prec>[n];

    //GPU data
    multi_prec<prec> *dalpha;
    multi_prec<prec> *dx;

    cudaMalloc(&dalpha, sizeof(multi_prec<prec>));
    cudaMalloc(&dx, sizeof(multi_prec<prec>) * n);

    //Convert from MPFR
    #pragma omp parallel for
    for(int i = 0; i < n; i ++){
        hx[i] = convert_to_string_sci(x[i], convert_digits).c_str();
    }
    halpha = convert_to_string_sci(alpha, convert_digits).c_str();

    //Copying alpha to the GPU
    cudaMemcpy(dalpha, &halpha, sizeof(multi_prec<prec>), cudaMemcpyHostToDevice);
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    //Launch
    for(int i = 0; i < repeats; i ++){
        cudaMemcpy(dx, hx, sizeof(multi_prec<prec>) * n, cudaMemcpyHostToDevice);
        StartCudaTimer();
        campary_scal<prec>(n, dalpha, dx);
        EndCudaTimer();
    }
    PrintCudaTimer("took");
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    //Copying to the host
    cudaMemcpy(hx, dx, sizeof(multi_prec<prec>) * n, cudaMemcpyDeviceToHost);
    for(int i = 1; i < n; i ++){
        hx[0] += hx[i];
    }
    printResult<prec>(hx[0]);

    //Cleanup
    delete [] hx;
    cudaFree(dalpha);
    cudaFree(dx);
}


/*
 * AXPY test
 */
template<int prec>
void campary_axpy_test(int n, mpfr_t alpha, mpfr_t *x, mpfr_t *y, int convert_digits, int repeats){
    Logger::printDash();
    InitCudaTimer();
    PrintTimerName("[GPU] CAMPARY axpy");

    //Host data
    multi_prec<prec> halpha;
    multi_prec<prec> *hx = new multi_prec<prec>[n];
    multi_prec<prec> *hy = new multi_prec<prec>[n];

    //GPU data
    multi_prec<prec> *dalpha;
    multi_prec<prec> *dx;
    multi_prec<prec> *dy;

    cudaMalloc(&dalpha, sizeof(multi_prec<prec>));
    cudaMalloc(&dx, sizeof(multi_prec<prec>) * n);
    cudaMalloc(&dy, sizeof(multi_prec<prec>) * n);

    //Convert from MPFR
    #pragma omp parallel for
    for(int i = 0; i < n; i ++){
        hx[i] = convert_to_string_sci(x[i], convert_digits).c_str();
        hy[i] = convert_to_string_sci(y[i], convert_digits).c_str();
    }
    halpha = convert_to_string_sci(alpha, convert_digits).c_str();

    //Copying data to the GPU
    cudaMemcpy(dalpha, &halpha, sizeof(multi_prec<prec>), cudaMemcpyHostToDevice);
    cudaMemcpy(dx, hx, sizeof(multi_prec<prec>) * n, cudaMemcpyHostToDevice);
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    //Launch
    for(int i = 0; i < repeats; i ++){
        cudaMemcpy(dy, hy, sizeof(multi_prec<prec>) * n, cudaMemcpyHostToDevice);
        StartCudaTimer();
        campary_axpy<prec>(n, dalpha, dx, dy);
        EndCudaTimer();
    }
    PrintCudaTimer("took");
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    //Copying to the host
    cudaMemcpy(hy, dy, sizeof(multi_prec<prec>) * n, cudaMemcpyDeviceToHost);
    for(int i = 1; i < n; i ++){
        hy[0] += hy[i];
    }
    printResult<prec>(hy[0]);

    //Cleanup
    delete [] hx;
    delete [] hy;
    cudaFree(dalpha);
    cudaFree(dx);
    cudaFree(dy);
}

/*
 * ROT test
 */
template<int prec>
void campary_rot_test(int n, mpfr_t *x, mpfr_t *y, mpfr_t c, mpfr_t s, int convert_digits, int repeats){
    Logger::printDash();
    InitCudaTimer();
    PrintTimerName("[GPU] CAMPARY rot");

    //Host data
    multi_prec<prec> hc;
    multi_prec<prec> hs;
    multi_prec<prec> *hx = new multi_prec<prec>[n];
    multi_prec<prec> *hy = new multi_prec<prec>[n];

    //GPU data
    multi_prec<prec> *dx;
    multi_prec<prec> *dy;
    multi_prec<prec> *dc;
    multi_prec<prec> *ds;

    cudaMalloc(&dx, sizeof(multi_prec<prec>) * n);
    cudaMalloc(&dy, sizeof(multi_prec<prec>) * n);
    cudaMalloc(&dc, sizeof(multi_prec<prec>));
    cudaMalloc(&ds, sizeof(multi_prec<prec>));

    //Convert from MPFR
    #pragma omp parallel for
    for(int i = 0; i < n; i++){
        hx[i] = convert_to_string_sci(x[i], convert_digits).c_str();
        hy[i] = convert_to_string_sci(y[i], convert_digits).c_str();
    }
    hc = convert_to_string_sci(c, convert_digits).c_str();
    hs = convert_to_string_sci(s, convert_digits).c_str();

    //Copying scalars to the GPU
    cudaMemcpy(dc, &hc, sizeof(multi_prec<prec>), cudaMemcpyHostToDevice);
    cudaMemcpy(ds, &hs, sizeof(multi_prec<prec>), cudaMemcpyHostToDevice);
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    //Launch
    for(int i = 0; i < repeats; i++){
        cudaMemcpy(dx, hx, sizeof(multi_prec<prec>) * n, cudaMemcpyHostToDevice);
        cudaMemcpy(dy, hy, sizeof(multi_prec<prec>) * n, cudaMemcpyHostToDevice);
        StartCudaTimer();
        campary_rot<prec>(n, dx, dy, dc, ds);
        EndCudaTimer();
    }
    PrintCudaTimer("took");
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    //Copying to the host
    cudaMemcpy(hy, dy, sizeof(multi_prec<prec>) * n, cudaMemcpyDeviceToHost);
    cudaMemcpy(hx, dx, sizeof(multi_prec<prec>) * n, cudaMemcpyDeviceToHost);

    for(int i = 1; i < n; i++){
        hx[0] += hx[i];
        hy[0] += hy[i];
    }
    printResult<prec>(hx[0]);
    printResult<prec>(hy[0]);

    //Cleanup
    delete [] hx;
    delete [] hy;
    cudaFree(dx);
    cudaFree(dy);
    cudaFree(dc);
    cudaFree(ds);
}

/*
 * AXPY_DOT test
 */
template<int prec>
void campary_axpy_dot_test(int n, mpfr_t alpha, mpfr_t *w, mpfr_t *v, mpfr_t *u, int convert_digits, int repeats){
    Logger::printDash();
    InitCudaTimer();
    PrintTimerName("[GPU] CAMPARY axpy_dot");

    //Host data
    multi_prec<prec> halpha;
    multi_prec<prec> *hv = new multi_prec<prec>[n];
    multi_prec<prec> *hu = new multi_prec<prec>[n];
    multi_prec<prec> *hw = new multi_prec<prec>[n];
    multi_prec<prec> *hr = new multi_prec<prec>[1];

    //GPU data
    multi_prec<prec> *dalpha;
    multi_prec<prec> *dv;
    multi_prec<prec> *du;
    multi_prec<prec> *dw;
    multi_prec<prec> *dr;

    cudaMalloc(&dalpha, sizeof(multi_prec<prec>));
    cudaMalloc(&dv, sizeof(multi_prec<prec>) * n);
    cudaMalloc(&du, sizeof(multi_prec<prec>) * n);
    cudaMalloc(&dw, sizeof(multi_prec<prec>) * n);
    cudaMalloc(&dr, sizeof(multi_prec<prec>));

    //Convert from MPFR
    #pragma omp parallel for
    for(int i = 0; i < n; i ++){
        hv[i] = convert_to_string_sci(v[i], convert_digits).c_str();
        hu[i] = convert_to_string_sci(u[i], convert_digits).c_str();
        hw[i] = convert_to_string_sci(w[i], convert_digits).c_str();
    }
    halpha = convert_to_string_sci(alpha, convert_digits).c_str();

    //Copying data to the GPU
    cudaMemcpy(dalpha, &halpha, sizeof(multi_prec<prec>), cudaMemcpyHostToDevice);
    cudaMemcpy(dv, hv, sizeof(multi_prec<prec>) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(du, hu, sizeof(multi_prec<prec>) * n, cudaMemcpyHostToDevice);
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    //Launch
    for(int i = 0; i < repeats; i ++){
        cudaMemcpy(dw, hw, sizeof(multi_prec<prec>) * n, cudaMemcpyHostToDevice);
        StartCudaTimer();
        campary_axpy_dot<prec>(n, dalpha, dw, dv, du, dr);
        EndCudaTimer();
    }
    PrintCudaTimer("took");
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    //Copying to the host
    cudaMemcpy(hr, dr, sizeof(multi_prec<prec>), cudaMemcpyDeviceToHost);
    cudaMemcpy(hw, dw, sizeof(multi_prec<prec>) * n, cudaMemcpyDeviceToHost);
    for(int i = 1; i < n; i++){
        hw[0] += hw[i];
    }
    printResult<prec>(hw[0]);
    printResult<prec>(hr[0]);

    //Cleanup
    delete [] hv;
    delete [] hu;
    delete [] hw;
    delete [] hr;
    cudaFree(dalpha);
    cudaFree(dv);
    cudaFree(du);
    cudaFree(dw);
    cudaFree(dr);
}

/*
 * GEMV test
 */
template<int prec>
void campary_gemv_test(int m, int n, mpfr_t alpha, mpfr_t *A, int lda, mpfr_t *x, mpfr_t beta, mpfr_t *y, int convert_prec, int repeats){
    Logger::printDash();
    InitCudaTimer();
    PrintTimerName("[GPU] CAMPARY gemv");

    //Execution configuration
    int BLOCKS = m / CAMPARY_VECTOR_MULTIPLY_THREADS + 1;

    //Host data
    multi_prec<prec> halpha;
    multi_prec<prec> hbeta;
    multi_prec<prec> *hx = new multi_prec<prec>[n];
    multi_prec<prec> *hy = new multi_prec<prec>[m];
    multi_prec<prec> *hA = new multi_prec<prec>[lda * n];

    //GPU data
    multi_prec<prec> *dalpha;
    multi_prec<prec> *dbeta;
    multi_prec<prec> *dx;
    multi_prec<prec> *dy;
    multi_prec<prec> *dA;

    cudaMalloc(&dalpha, sizeof(multi_prec<prec>));
    cudaMalloc(&dbeta, sizeof(multi_prec<prec>));
    cudaMalloc(&dx, sizeof(multi_prec<prec>) * n);
    cudaMalloc(&dy, sizeof(multi_prec<prec>) * m);
    cudaMalloc(&dA, sizeof(multi_prec<prec>) * lda * n);

    //Convert from MPFR
    #pragma omp parallel for
    for(int i = 0; i < n; i ++){
        hx[i] = convert_to_string_sci(x[i], convert_prec).c_str();
    }
    #pragma omp parallel for
    for(int i = 0; i < m; i ++){
        hy[i] = convert_to_string_sci(y[i], convert_prec).c_str();
    }
    #pragma omp parallel for
    for(int i = 0; i < lda * n; i ++){
        hA[i] = convert_to_string_sci(A[i], convert_prec).c_str();
    }
    halpha = convert_to_string_sci(alpha, convert_prec).c_str();
    hbeta = convert_to_string_sci(beta, convert_prec).c_str();

    //Copying to the GPU
    cudaMemcpy(dalpha, &halpha, sizeof(multi_prec<prec>), cudaMemcpyHostToDevice);
    cudaMemcpy(dbeta, &hbeta, sizeof(multi_prec<prec>), cudaMemcpyHostToDevice);
    cudaMemcpy(dA, hA, sizeof(multi_prec<prec>) * lda * n, cudaMemcpyHostToDevice);
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    //Launch
    for(int i = 0; i < repeats; i ++){
        cudaMemcpy(dx, hx, sizeof(multi_prec<prec>) * n, cudaMemcpyHostToDevice);
        cudaMemcpy(dy, hy, sizeof(multi_prec<prec>) * m, cudaMemcpyHostToDevice);
        StartCudaTimer();
        campary_scal(n, dalpha, dx);
        campary_gemv_kernel<prec><<<BLOCKS, CAMPARY_VECTOR_MULTIPLY_THREADS>>>(m, n, dA, lda, dx, dbeta, dy);
        EndCudaTimer();
    }
    PrintCudaTimer("took");
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    //Copying to the host
    cudaMemcpy(hy, dy, sizeof(multi_prec<prec>) * m, cudaMemcpyDeviceToHost);
    for(int i = 1; i < m; i ++){
        hy[0] += hy[i];
    }
    printResult<prec>(hy[0]);

    //Cleanup
    delete [] hx;
    delete [] hy;
    delete [] hA;
    cudaFree(dalpha);
    cudaFree(dbeta);
    cudaFree(dx);
    cudaFree(dy);
    cudaFree(dA);
}

/*
 * GEMM test
 */
template<int prec>
void campary_gemm_test(int m, int n, int k, mpfr_t alpha, mpfr_t *A, int lda, mpfr_t *B, int ldb, mpfr_t beta, mpfr_t *C, int ldc, int convert_prec, int repeats){
    Logger::printDash();
    InitCudaTimer();
    PrintTimerName("[GPU] CAMPARY gemm");

    //Host data
    multi_prec<prec> halpha;
    multi_prec<prec> hbeta;
    multi_prec<prec> *hA = new multi_prec<prec>[lda * k];
    multi_prec<prec> *hB = new multi_prec<prec>[ldb * n];
    multi_prec<prec> *hC = new multi_prec<prec>[ldc * n];

    //GPU data
    multi_prec<prec> *dalpha;
    multi_prec<prec> *dbeta;
    multi_prec<prec> *dA;
    multi_prec<prec> *dB;
    multi_prec<prec> *dC;

    cudaMalloc(&dalpha, sizeof(multi_prec<prec>));
    cudaMalloc(&dbeta, sizeof(multi_prec<prec>));
    cudaMalloc(&dA, sizeof(multi_prec<prec>) * lda * k);
    cudaMalloc(&dB, sizeof(multi_prec<prec>) * ldb * n);
    cudaMalloc(&dC, sizeof(multi_prec<prec>) * ldc * n);

    //Convert from MPFR
    halpha = convert_to_string_sci(alpha, convert_prec).c_str();
    hbeta = convert_to_string_sci(beta, convert_prec).c_str();
    #pragma omp parallel for
    for(int i = 0; i < lda * k; i++){
        hA[i] = convert_to_string_sci(A[i], convert_prec).c_str();
    }
    #pragma omp parallel for
    for(int i = 0; i < ldb * n; i++){
        hB[i] = convert_to_string_sci(B[i], convert_prec).c_str();
    }
    #pragma omp parallel for
    for(int i = 0; i < ldc * n; i++){
        hC[i] = convert_to_string_sci(C[i], convert_prec).c_str();
    }

    //Copying to the GPU
    cudaMemcpy(dalpha, &halpha, sizeof(multi_prec<prec>), cudaMemcpyHostToDevice);
    cudaMemcpy(dbeta, &hbeta, sizeof(multi_prec<prec>), cudaMemcpyHostToDevice);
    cudaMemcpy(dA, hA, sizeof(multi_prec<prec>) * lda * k, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, sizeof(multi_prec<prec>) * ldb * n, cudaMemcpyHostToDevice);

    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    //Launch
    for(int i = 0; i < repeats; i ++){
        cudaMemcpy(dC, hC, sizeof(multi_prec<prec>) * ldc * n, cudaMemcpyHostToDevice);
        StartCudaTimer();
        campary_gemm<prec>(m, n, k, dalpha, dA, lda, dB, ldb, dbeta, dC, ldc)
        EndCudaTimer();
    }
    PrintCudaTimer("took");
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    //Copying to the host
    cudaMemcpy(hC, dC, sizeof(multi_prec<prec>) * ldc * n, cudaMemcpyDeviceToHost);
    for(int i = 1; i < ldc * n; i++){
        hC[0] += hC[i];
    }
    printResult<prec>(hC[0]);

    //Cleanup
    delete [] hA;
    delete [] hB;
    delete [] hC;
    cudaFree(dalpha);
    cudaFree(dbeta);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
}

/*
 * GE_ADD test
 */
template<int prec>
void campary_ge_add_test(int m, int n, mpfr_t alpha, mpfr_t *A, int lda, mpfr_t beta, mpfr_t *B, int ldb, mpfr_t *C, int ldc, int convert_prec, int repeats){
    Logger::printDash();
    InitCudaTimer();
    PrintTimerName("[GPU] CAMPARY ge_add");

    //Host data
    multi_prec<prec> halpha;
    multi_prec<prec> hbeta;
    multi_prec<prec> *hA = new multi_prec<prec>[lda * n];
    multi_prec<prec> *hB = new multi_prec<prec>[ldb * n];
    multi_prec<prec> *hC = new multi_prec<prec>[ldc * n];

    //GPU data
    multi_prec<prec> *dalpha;
    multi_prec<prec> *dbeta;
    multi_prec<prec> *dA;
    multi_prec<prec> *dB;
    multi_prec<prec> *dC;

    cudaMalloc(&dalpha, sizeof(multi_prec<prec>));
    cudaMalloc(&dbeta, sizeof(multi_prec<prec>));
    cudaMalloc(&dA, sizeof(multi_prec<prec>) * lda * n);
    cudaMalloc(&dB, sizeof(multi_prec<prec>) * ldb * n);
    cudaMalloc(&dC, sizeof(multi_prec<prec>) * ldc * n);

    //Convert from MPFR
    halpha = convert_to_string_sci(alpha, convert_prec).c_str();
    hbeta = convert_to_string_sci(beta, convert_prec).c_str();
    #pragma omp parallel for
    for(int i = 0; i < lda * n; i++){
        hA[i] = convert_to_string_sci(A[i], convert_prec).c_str();
    }
    #pragma omp parallel for
    for(int i = 0; i < ldb * n; i++){
        hB[i] = convert_to_string_sci(B[i], convert_prec).c_str();
    }
    #pragma omp parallel for
    for(int i = 0; i < ldc * n; i++){
        hC[i] = convert_to_string_sci(C[i], convert_prec).c_str();
    }
    //Copying to the GPU
    cudaMemcpy(dalpha, &halpha, sizeof(multi_prec<prec>), cudaMemcpyHostToDevice);
    cudaMemcpy(dbeta, &hbeta, sizeof(multi_prec<prec>), cudaMemcpyHostToDevice);
    cudaMemcpy(dA, hA, sizeof(multi_prec<prec>) * lda * n, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, sizeof(multi_prec<prec>) * ldb * n, cudaMemcpyHostToDevice);
    cudaMemcpy(dC, hC, sizeof(multi_prec<prec>) * ldc * n, cudaMemcpyHostToDevice);

    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    //Launch
    for(int i = 0; i < repeats; i ++){
        StartCudaTimer();
        campary_ge_add<prec>(m, n, dalpha, dA, lda, dbeta, dB, ldb, dC, ldc)
        EndCudaTimer();
    }
    PrintCudaTimer("took");
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    //Copying to the host
    cudaMemcpy(hC, dC, sizeof(multi_prec<prec>) * ldc * n, cudaMemcpyDeviceToHost);
    for(int i = 1; i < ldc * n; i++){
        hC[0] += hC[i];
    }
    printResult<prec>(hC[0]);

    //Cleanup
    delete [] hA;
    delete [] hB;
    delete [] hC;
    cudaFree(dalpha);
    cudaFree(dbeta);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
}

/*
 * GE_ACC test
 */
template<int prec>
void campary_ge_acc_test(int m, int n, mpfr_t alpha, mpfr_t *A, int lda, mpfr_t beta, mpfr_t *B, int ldb, int convert_prec, int repeats){
    Logger::printDash();
    InitCudaTimer();
    PrintTimerName("[GPU] CAMPARY ge_acc");

    //Host data
    multi_prec<prec> halpha;
    multi_prec<prec> hbeta;
    multi_prec<prec> *hA = new multi_prec<prec>[lda * n];
    multi_prec<prec> *hB = new multi_prec<prec>[ldb * n];

    //GPU data
    multi_prec<prec> *dalpha;
    multi_prec<prec> *dbeta;
    multi_prec<prec> *dA;
    multi_prec<prec> *dB;

    cudaMalloc(&dalpha, sizeof(multi_prec<prec>));
    cudaMalloc(&dbeta, sizeof(multi_prec<prec>));
    cudaMalloc(&dA, sizeof(multi_prec<prec>) * lda * n);
    cudaMalloc(&dB, sizeof(multi_prec<prec>) * ldb * n);

    //Convert from MPFR
    halpha = convert_to_string_sci(alpha, convert_prec).c_str();
    hbeta = convert_to_string_sci(beta, convert_prec).c_str();
    #pragma omp parallel for
    for(int i = 0; i < lda * n; i++){
        hA[i] = convert_to_string_sci(A[i], convert_prec).c_str();
    }
    #pragma omp parallel for
    for(int i = 0; i < ldb * n; i++){
        hB[i] = convert_to_string_sci(B[i], convert_prec).c_str();
    }

    //Copying to the GPU
    cudaMemcpy(dalpha, &halpha, sizeof(multi_prec<prec>), cudaMemcpyHostToDevice);
    cudaMemcpy(dbeta, &hbeta, sizeof(multi_prec<prec>), cudaMemcpyHostToDevice);
    cudaMemcpy(dA, hA, sizeof(multi_prec<prec>) * lda * n, cudaMemcpyHostToDevice);
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    //Launch
    for(int i = 0; i < repeats; i ++){
        cudaMemcpy(dB, hB, sizeof(multi_prec<prec>) * ldb * n, cudaMemcpyHostToDevice);
        StartCudaTimer();
        campary_ge_acc<prec>(m, n, dalpha, dA, lda, dbeta, dB, ldb)
        EndCudaTimer();
    }
    PrintCudaTimer("took");
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    //Copying to the host
    cudaMemcpy(hB, dB, sizeof(multi_prec<prec>) * ldb * n, cudaMemcpyDeviceToHost);
    for(int i = 1; i < ldb * n; i++){
        hB[0] += hB[i];
    }
    printResult<prec>(hB[0]);

    //Cleanup
    delete [] hA;
    delete [] hB;
    cudaFree(dalpha);
    cudaFree(dbeta);
    cudaFree(dA);
    cudaFree(dB);
}

/*
 * GE_DIAG_SCALE test
 */
template<int prec>
void campary_ge_diag_scale_test(enum  mblas_side_type side, int m, int n, int lend, mpfr_t *D, int incd, mpfr_t *A, int lda, int convert_prec, int repeats){
    Logger::printDash();
    InitCudaTimer();
    PrintTimerName("[GPU] CAMPARY ge_diag_scale");

    //Host data
    multi_prec<prec> *hA = new multi_prec<prec>[lda * n];
    multi_prec<prec> *hD = new multi_prec<prec>[lend];

    //GPU data
    multi_prec<prec> *dA;
    multi_prec<prec> *dD;

    cudaMalloc(&dA, sizeof(multi_prec<prec>) * lda * n);
    cudaMalloc(&dD, sizeof(multi_prec<prec>) * lend);

    //Convert from MPFR
    #pragma omp parallel for
    for(int i = 0; i < lda * n; i++){
        hA[i] = convert_to_string_sci(A[i], convert_prec).c_str();
    }
    #pragma omp parallel for
    for(int i = 0; i < lend; i++){
        hD[i] = convert_to_string_sci(D[i], convert_prec).c_str();
    }
    //Copying to the GPU
    cudaMemcpy(dD, hD, sizeof(multi_prec<prec>) * lend, cudaMemcpyHostToDevice);

    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    //Launch
    for(int i = 0; i < repeats; i ++){
        cudaMemcpy(dA, hA, sizeof(multi_prec<prec>) * lda * n, cudaMemcpyHostToDevice);
        StartCudaTimer();
        campary_ge_diag_scale<prec>(side, m, n, dD, incd, dA, lda)
        EndCudaTimer();
    }
    PrintCudaTimer("took");
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    //Copying to the host
    cudaMemcpy(hA, dA, sizeof(multi_prec<prec>) * lda * n, cudaMemcpyDeviceToHost);
    for(int i = 1; i < lda * n; i++){
        hA[0] += hA[i];
    }
    printResult<prec>(hA[0]);

    //Cleanup
    delete [] hA;
    delete [] hD;
    cudaFree(dA);
    cudaFree(dD);
}

/*
 * GE_LRSCALE test
 */
template<int prec>
void campary_ge_lrscale_test(int m, int n,  mpfr_t *DL, int incdl, mpfr_t *DR, int incdr, mpfr_t *A, int lda, int convert_prec, int repeats){
    Logger::printDash();
    InitCudaTimer();
    PrintTimerName("[GPU] CAMPARY ge_lrscale");

    int lendl = (1 + (m - 1) * abs(incdl));
    int lendr = (1 + (n - 1) * abs(incdr));

    //Host data
    multi_prec<prec> *hA = new multi_prec<prec>[lda * n];
    multi_prec<prec> *hDL = new multi_prec<prec>[lendl];
    multi_prec<prec> *hDR = new multi_prec<prec>[lendr];

    //GPU data
    multi_prec<prec> *dA;
    multi_prec<prec> *dDL;
    multi_prec<prec> *dDR;

    cudaMalloc(&dA, sizeof(multi_prec<prec>) * lda * n);
    cudaMalloc(&dDL, sizeof(multi_prec<prec>) * lendl);
    cudaMalloc(&dDR, sizeof(multi_prec<prec>) * lendr);

    //Convert from MPFR
    #pragma omp parallel for
    for(int i = 0; i < lda * n; i++){
        hA[i] = convert_to_string_sci(A[i], convert_prec).c_str();
    }
    #pragma omp parallel for
    for(int i = 0; i < lendl; i++){
        hDL[i] = convert_to_string_sci(DL[i], convert_prec).c_str();
    }
    #pragma omp parallel for
    for(int i = 0; i < lendr; i++){
        hDR[i] = convert_to_string_sci(DR[i], convert_prec).c_str();
    }
    //Copying to the GPU
    cudaMemcpy(dDL, hDL, sizeof(multi_prec<prec>) * lendl, cudaMemcpyHostToDevice);
    cudaMemcpy(dDR, hDR, sizeof(multi_prec<prec>) * lendr, cudaMemcpyHostToDevice);

    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    //Launch
    for(int i = 0; i < repeats; i ++){
        cudaMemcpy(dA, hA, sizeof(multi_prec<prec>) * lda * n, cudaMemcpyHostToDevice);
        StartCudaTimer();
        campary_ge_lrscale<prec>(m, n, dDL, incdl, dDR, incdr, dA, lda)
        EndCudaTimer();
    }
    PrintCudaTimer("took");
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    //Copying to the host
    cudaMemcpy(hA, dA, sizeof(multi_prec<prec>) * lda * n, cudaMemcpyDeviceToHost);
    for(int i = 1; i < lda * n; i++){
        hA[0] += hA[i];
    }
    printResult<prec>(hA[0]);

    //Cleanup
    delete [] hA;
    delete [] hDL;
    delete [] hDR;
    cudaFree(dA);
    cudaFree(dDL);
    cudaFree(dDR);
}

#endif //MPRES_TEST_CAMPARY_BLAS_CUH
#endif //EXCLUDE_CAMPARY