/*
 *  Multiple-precision DOT function for GPU (BLAS Level-1)
 *  Dot product of two vectors
 *
 *  Copyright 2021 by Konstantin Isupov.
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


#ifndef MPRES_DOT_V2_CUH
#define MPRES_DOT_V2_CUH

#include "arith/add.cuh"
#include "arith/mul.cuh"

namespace cuda {

    /*
     * Computes the inner product of two vectors and stores result in an array of size gridDim.x
     * For final reduction, mp_sum_kernel should be used
     */
    template<int threads>
    __global__ void mp_dot_kernel1(const unsigned int n, const unsigned int nextPow2, mp_float_ptr x, mp_float_ptr y, mp_float_ptr r) {
        __shared__ mp_float_t sdata[threads];
        __shared__ mp_float_t prods[threads];

        // parameters
        const unsigned int tid = threadIdx.x;
        const unsigned int bid = blockIdx.x;
        const unsigned int bsize = blockDim.x;
        const unsigned int k = gridDim.x * bsize;
        unsigned int i = bid * bsize + tid;

        // do reduction in global mem
        // we reduce multiple elements per thread. The number is determined by the
        // number of active thread blocks (via gridDim).  More blocks will result
        // in a larger gridSize and therefore fewer elements per thread
        sdata[tid] = cuda::MP_ZERO;
        while (i < n) {
            cuda::mp_mul(&prods[threadIdx.x],&x[i], &y[i]);
            cuda::mp_add(&sdata[tid], &sdata[tid], &prods[threadIdx.x]);
            i += k;
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
            r[bid] = sdata[tid];
        };
        __syncthreads();
    }

    /*!
     * Multiple-precision summation kernel: computes the sum of the elements of vector x
     * @param nextPow2 - least power of two greater than or equal to blockDim.x
      */
    template<int threads>
    __global__ void mp_dot_kernel2(const unsigned int n, const unsigned int nextPow2, mp_float_ptr x, mp_float_ptr r) {
        __shared__ mp_float_t sdata[threads];

        const unsigned int tid = threadIdx.x;
        const unsigned int bid = blockIdx.x;
        const unsigned int bsize = blockDim.x;
        const unsigned int k = gridDim.x * bsize;
        unsigned int i = bid * bsize + tid;

        sdata[tid] = cuda::MP_ZERO;
        while (i < n) {
            cuda::mp_add(&sdata[tid], &sdata[tid], &x[i]);
            i += k;
        }
        __syncthreads();

        i = nextPow2 >> 1;
        while(i >= 1){
            if ((tid < i) && (tid + i < bsize)) {
                cuda::mp_add(&sdata[tid], &sdata[tid], &sdata[tid + i]);
            }
            i = i >> 1;
            __syncthreads();
        }
        if (tid == 0) {
            r[bid] = sdata[tid];
        };
        __syncthreads();
    }

    /*
     * Computes the dot product of two vectors, r[0] = x0*y0 + x1*y1 + ... + xn*yn
     * @tparam blocks - number of blocks
     * @tparam threads - thread block size
     * @param n - operation size (must be positive)
     * @param x - multiple-precision vector in the GPU memory
     * @param y - multiple-precision vector in the GPU memory
     * @param r - pointer to the inner product (vector of length one) in the GPU memory. Overwritten by the result
     */
    template <int blocks, int threads>
    void mp_dot(const int n, mp_float_ptr x, mp_float_ptr y, mp_float_ptr r) {
        mp_float_ptr dbuf; // Device buffer for partial results
        cudaMalloc((void **) &dbuf, sizeof(mp_float_t) * blocks);
        const unsigned int POW = nextPow2(threads); // Power of 2 that is greater that or equals to threads
        mp_dot_kernel1 <threads><<< blocks, threads >>> (n, POW, x, y, dbuf); //Launch the 1st CUDA kernel to perform parallel summation on the GPU
        mp_dot_kernel2 <threads><<< 1, threads >>> (blocks, POW, dbuf, r); //Launch the 2nd CUDA kernel to perform summation of the results of parallel blocks on the GPU
        cudaFree(dbuf);
    }

} //end of namespace

#endif //MPRES_DOT_V2_CUH

