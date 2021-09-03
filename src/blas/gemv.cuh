/*
 *  Multiple-precision GEMV function for GPU (BLAS Level-2)
 *  Computes a matrix-vector product using a general matrix.
 *
 *  Copyright 2019-2020 by Konstantin Isupov.
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

#ifndef MPRES_GEMV_CUH
#define MPRES_GEMV_CUH

#include "../arith/add.cuh"
#include "../mpvector.cuh"
#include "../mpmatrix.cuh"
#include "../kernel_config.cuh"
#include "mblas_enum.cuh"

namespace cuda
{
    /*!
     * Kernel that calculates the sum of all the elements in each row of an m-by-n multiple-precision matrix
     * The result (a vector of size m) is then added to the vector y
     * @note Each block is associated with its own element of y, so this kernel must be run on an one-dimensional grid of m one-dimensional blocks
     * @note Shared memory of size sizeof(mp_float_t) * nThreads must be allocated, where nThreads is the number of threads per block
     * @param A - matrix of m rows and n columns
     * @param y - vector of size m
     * @param incy - storage spacing between elements of y
     * @param nextPow2 - least power of two greater than or equal to blockDim.x
     */
    __global__ static void matrix_row_sum_kernel(const unsigned int m, const unsigned int n, mp_array_t A, mp_array_t y, const int incy, const unsigned int nextPow2) {
        extern __shared__ mp_float_t sdata[];

        // parameters
        const unsigned int tid = threadIdx.x;
        const unsigned int bid = blockIdx.x;
        const unsigned int bsize = blockDim.x;
        unsigned int i = threadIdx.x;

        // do reduction in global mem
        sdata[tid] = cuda::MP_ZERO;
        while (i < n) {
            cuda::mp_add(&sdata[tid], &sdata[tid], A, i * m + bid);
            i += bsize;
        }
        __syncthreads();

        // do reduction in shared mem
        i = nextPow2 >> 1; // half of nextPow2
        while(i >= 1){
            if ((tid < i) && (tid + i < bsize)) {
                cuda::mp_add(&sdata[tid], &sdata[tid], &sdata[tid + i]);
            }
            i = i >> 1;
            __syncthreads();
        }

        // write result for this block to global mem
        if (tid == 0) {
            int iy = incy > 0 ? bid * incy : (-m + bid + 1)*incy;
            cuda::mp_add(y, iy, y, iy, &sdata[tid]);
        }
    }

    /*!
     * Kernel that calculates the sum of all the elements in each column of an m-by-n multiple-precision matrix
     * The result (a vector of size n) is then added to the vector y
     * @note Each block is associated with its own element of y, so this kernel must be run on an one-dimensional grid of n one-dimensional blocks
     * @note Shared memory of size sizeof(mp_float_t) * nThreads must be allocated, where nThreads is the number of threads per block
     * @param A - matrix of m rows and n columns
     * @param y - vector of size n
     * @param incy - storage spacing between elements of y
     * @param nextPow2 - least power of two greater than or equal to blockDim.x
     */
    __global__ static void matrix_col_sum_kernel(const unsigned int m, const unsigned int n, mp_array_t A, mp_array_t y, const int incy, const unsigned int nextPow2) {
        extern __shared__ mp_float_t sdata[];

        // parameters
        const unsigned int tid = threadIdx.x;
        const unsigned int bid = blockIdx.x;
        const unsigned int bsize = blockDim.x;
        unsigned int i = tid;

        // do reduction in global mem
        sdata[tid] = cuda::MP_ZERO;
        while (i < m) {
            cuda::mp_add(&sdata[tid], &sdata[tid], A, bid * m + i);
            i += bsize;
        }
        __syncthreads();

        // do reduction in shared mem
        i = nextPow2 >> 1; // half of nextPow2
        while(i >= 1){
            if ((tid < i) && (tid + i < bsize)) {
                cuda::mp_add(&sdata[tid], &sdata[tid], &sdata[tid + i]);
            }
            i = i >> 1;
            __syncthreads();
        }

        // write result for this block to global mem
        if (tid == 0) {
            int iy = incy > 0 ? bid * incy : (-n + bid + 1)*incy;
            cuda::mp_add(y, iy, y, iy, &sdata[tid]);
        }
    }

    /*!
     * Performs one of the matrix-vector operations
     * y = alpha*A*x + beta*y  or
     * y = alpha*A**T*x + beta*y,
     * where alpha and beta are scalars, x and y are vectors and A is an m-by-n matrix.
     * The matrix should be stored in column-major order.

     * @tparam gridDim1 - number of blocks used to compute the signs, exponents, interval evaluations in element-wise scalar-vector and matrix-vector operations. A 2D grid of gridDim1 x gridDim1 blocks will be launched
     * @tparam blockDim1 - number of threads per block used to compute the signs, exponents, interval evaluations, and also to round the result in element-wise scalar-vector and matrix-vector operations
     * @tparam gridDim2 - number of blocks  used to compute the digits of multiple-precision significands in element-wise scalar-vector and matrix-vector operations.  A 2D grid of gridDim2 x gridDim2 blocks will be launched
     * @tparam blockDim3 - number of threads per block for parallel summation (the number of blocks is equal to the size of y)
     *
     * @param trans - specifies the operation:
     * if trans = 'N' or 'n', then y := alpha*A*x + beta*y.
     * if trans = 'T' or 't' or 'C' or 'c' then y = alpha*A**T*x + beta*y (transposed matrix).
     * @param m - specifies the number of rows of the matrix A. The value of m must be greater than zero.
     * @param n - specifies the number of columns of the matrix A. The value of n must be greater than zero.
     * @param alpha - pointer to the scalar in the global GPU memory
     * @param A - pointer to the array, size lda * n, in the global GPU memory. Before entry, the leading m-by-n part of the array must contain the matrix A.
     * @param lda - specifies the leading dimension of A as declared in the calling (sub)program. The value of lda must be at least max(1, m).
     * @param x - pointer to the vector in the global GPU memory, size at least (1+(n-1)*abs(incx)) for non-transposed matrix and at least (1+(m-1)*abs(incx)) otherwise.
     * @param incx - storage spacing between elements of x. The value of incx must not be zero.
     * @param beta - pointer to the scalar in the global GPU memory
     * @param y - pointer to the vector in the global GPU memory, size at least (1+(m-1)*abs(incy)) for non-transposed matrix and at least (1+(n-1)*abs(incy)) otherwise.
     * @param incy - storage spacing between elements of y. The value of incy must not be zero.
     * @param buffer1 - auxiliary array in the global GPU memory, size at least n for non-transposed matrix and at least m otherwise.
     * @param buffer2 - auxiliary array, size m * n, in the global GPU memory for storing the intermediate matrix
     */
    template<int gridDim1, int blockDim1, int gridDim2, int blockDim3>
    void mp_gemv(enum mblas_trans_type trans, const int m, const int n, mp_array_t &alpha, mp_array_t &A, const int lda,
            mp_array_t &x, const int incx, mp_array_t &beta, mp_array_t &y, const int incy, mp_array_t &buffer1, mp_array_t &buffer2){

        //Quick return if possible
        if( (m <= 0) || (n <= 0) ){
            return;
        }
        //Test the input parameters
        if( (incx == 0) || (incy == 0) || (lda < MAX(1, m)) ){
            return;
        }

        //Execution configuration
        //  To compute the signs, exponents, and interval evaluations
        dim3 grid1(gridDim1, gridDim1);
        //  To compute the digits in RNS
        dim3 grid2(gridDim2, gridDim2);
        //  To compute digits (residues) in the vector operations
        int numThreadsX = (incx == 1) ? BLOCK_SIZE_FOR_RESIDUES : RNS_MODULI_SIZE;
        int numThreadsY = (incy == 1) ? BLOCK_SIZE_FOR_RESIDUES : RNS_MODULI_SIZE;

        if(trans == mblas_no_trans){

            //Multiplication buffer1 = alpha * x - Computing the signs, exponents, and interval evaluations
            mp_vec2scal_mul_esi_kernel<<< gridDim1, blockDim1 >>> (buffer1, 1, x, incx, alpha, n);

            //Multiplication buffer1 = alpha * x - Multiplying the digits in the RNS
            mp_vec2scal_mul_digits_kernel<<< gridDim2, numThreadsX >>> (buffer1, 1, x, incx, alpha, n);

            //Rounding the intermediate result (buffer1)
            mp_vector_round_kernel<<< gridDim1, blockDim1 >>> (buffer1, 1, n);

            //Multiplication y = beta * y - Computing the signs, exponents, and interval evaluations
            mp_vec2scal_mul_esi_kernel<<< gridDim1, blockDim1 >>> (y, incy, y, incy, beta, m);

            //Multiplication y = beta * y - Multiplying the digits in the RNS
            mp_vec2scal_mul_digits_kernel<<< gridDim2, numThreadsY >>> (y, incy, y, incy, beta, m);

            //Rounding y
            mp_vector_round_kernel<<< gridDim1, blockDim1 >>> (y, incy, m);

            //We consider the vector buffer1 (contains alpha * x)  as a diagonal n-by-n matrix and perform the right diagonal scaling, buffer2 = A * buffer1
            //Each column of the matrix is multiplied by one element of the vector.
            //We run a 2D grid of 1D blocks.
            //Each line in the grid (i.e., all blocks with the same y coordinate) is associated with its own column of the matrix.
            //The result is written to the intermediate m-by-n buffer2.

            //Multiplication buffer2 = A * buffer1 - Computing the signs, exponents, and interval evaluations
            mp_mat2vec_right_scal_esi_kernel<<<grid1, blockDim1>>> (buffer2, m, A, lda, buffer1, 1, m, n);

            //Multiplication buffer2 = A * buffer1 - Multiplying the digits in the RNS
            mp_mat2vec_right_scal_digits_kernel<<<grid2, BLOCK_SIZE_FOR_RESIDUES>>> (buffer2, m, A, lda, buffer1, 1, m, n);

            //Rounding the intermediate result (buffer2)
            mp_vector_round_kernel<<<gridDim1, blockDim1>>>(buffer2, 1, m * n);

            //The following is tne reduction of the intermediate matrix (buffer 2).
            //Here, the sum of the elements in each row is calculated, and then y is added to the calculated sum
            //The result is a vector of size m

            // Kernel memory configurations. We prefer shared memory
            //cudaFuncSetCacheConfig(matrix_row_sum_kernel, cudaFuncCachePreferShared);

            // Power of two that is greater that or equals to blockDim3
            const unsigned int POW = nextPow2(blockDim3);

            // Compute row sums
            matrix_row_sum_kernel<<<m, blockDim3, sizeof(mp_float_t) * blockDim3>>>(m, n, buffer2, y, incy, POW);

        } else {

            //Multiplication buffer1 = alpha * x - Computing the signs, exponents, and interval evaluations
            mp_vec2scal_mul_esi_kernel<<< gridDim1, blockDim1 >>> (buffer1, 1, x, incx, alpha, m);

            //Multiplication buffer1 = alpha * x - Multiplying the digits in the RNS
            mp_vec2scal_mul_digits_kernel<<< gridDim2, numThreadsX >>> (buffer1, 1, x, incx, alpha, m);

            //Rounding the intermediate result (buffer1)
            mp_vector_round_kernel<<< gridDim1, blockDim1 >>> (buffer1, 1, m);

            //Multiplication y = beta * y - Computing the signs, exponents, and interval evaluations
            mp_vec2scal_mul_esi_kernel<<< gridDim1, blockDim1 >>> (y, incy, y, incy, beta, n);

            //Multiplication y = beta * y - Multiplying the digits in the RNS
            mp_vec2scal_mul_digits_kernel<<< gridDim2, numThreadsY >>> (y, incy, y, incy, beta, n);

            //Rounding y
            mp_vector_round_kernel<<< gridDim1, blockDim1 >>> (y, incy, n);

            //We consider the vector buffer1 (contains alpha * x)  as a diagonal m-by-m matrix and perform the left diagonal scaling, buffer2 = buffer1 * A
            //Each column of the matrix is multiplied by the vector.
            //We run a 2D grid of 1D blocks.
            //Each line in the grid (i.e., all blocks with the same y coordinate) is associated with its own column of the matrix.
            //The result is written to the intermediate m-by-n buffer2.

            //Multiplication buffer2 = A^T * buffer1 - Computing the signs, exponents, and interval evaluations
            mp_mat2vec_left_scal_esi_kernel<<<grid1, blockDim1>>> (buffer2, m, A, lda, buffer1, 1, m, n);

            //Multiplication buffer2 = A^T * buffer1 - Multiplying the digits in the RNS
            mp_mat2vec_left_scal_digits_kernel<<<grid2, BLOCK_SIZE_FOR_RESIDUES>>> (buffer2, m, A, lda, buffer1, 1, m, n);

            //Rounding the intermediate result (buffer2)
            mp_vector_round_kernel<<<gridDim1, blockDim1>>>(buffer2, 1, m * n);

            //The following is tne reduction of the intermediate matrix (buffer 2).
            //Here, the sum of the elements in each column is calculated, and then y is added to the calculated sum
            //The result is a vector of size n

            // Kernel memory configurations. We prefer shared memory
            //cudaFuncSetCacheConfig(matrix_col_sum_kernel, cudaFuncCachePreferShared);

            // Power of two that is greater that or equals to blockDim3
            const unsigned int POW = nextPow2(blockDim3);

            // Compute column sums
            matrix_col_sum_kernel<<<n, blockDim3, sizeof(mp_float_t) * blockDim3>>>(m, n, buffer2, y, incy, POW);
        }
    }

} // namespace cuda

#endif //MPRES_GEMV_CUH
