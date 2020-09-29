/*
 *  CUDA kernels for per-row or per-column reduction of a multiple-precision matrix
 *  (computing the sum of all the elements in each row or in each column of a multiple-precision matrix)
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


#ifndef MPRES_MPREDUCT_CUH
#define MPRES_MPREDUCT_CUH

#include "mparray.cuh"

namespace cuda {

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
    __global__ static void matrix_row_sum_add_kernel(const unsigned int m, const unsigned int n, mp_array_t A, mp_array_t y, int incy, const unsigned int nextPow2) {
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
    __global__ static void matrix_col_sum_add_kernel(const unsigned int m, const unsigned int n, mp_array_t A, mp_array_t y, int incy, const unsigned int nextPow2) {
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
     * Kernel that calculates the sum of all the elements in each row of an m-by-n multiple-precision matrix and stores it in y (without adding y)
     * @note Each block is associated with its own element of y, so this kernel must be run on an one-dimensional grid of m one-dimensional blocks
     * @note Shared memory of size sizeof(mp_float_t) * nThreads must be allocated, where nThreads is the number of threads per block
     * @param A - matrix of m rows and n columns
     * @param y - vector of size m
     * @param incy - storage spacing between elements of y
     * @param nextPow2 - least power of two greater than or equal to blockDim.x
     */
    __global__ static void matrix_row_sum_kernel(const unsigned int m, const unsigned int n, mp_array_t A, mp_array_t y, int incy, const unsigned int nextPow2) {
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
            cuda::mp_set(y, iy, &sdata[tid]);
        }
    }

    /*!
     * Kernel that calculates the sum of all the elements in each column of an m-by-n multiple-precision matrix and stores it in y (without adding y)
     * @note Each block is associated with its own element of y, so this kernel must be run on an one-dimensional grid of n one-dimensional blocks
     * @note Shared memory of size sizeof(mp_float_t) * nThreads must be allocated, where nThreads is the number of threads per block
     * @param A - matrix of m rows and n columns
     * @param y - vector of size n
     * @param incy - storage spacing between elements of y
     * @param nextPow2 - least power of two greater than or equal to blockDim.x
     */
    __global__ static void matrix_col_sum_kernel(const unsigned int m, const unsigned int n, mp_array_t A, mp_array_t y, int incy, const unsigned int nextPow2) {
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
            cuda::mp_set(y, iy, &sdata[tid]);
        }
    }

} //end of namespace


#endif //MPRES_MPREDUCT_CUH
