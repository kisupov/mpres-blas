/*
 *  Multiple-precision summation routines for GPU
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


#ifndef MPRES_MPREDUCT_CUH
#define MPRES_MPREDUCT_CUH

#include "mpfloat.cuh"

namespace cuda {

    /*
     * Multiple-precision summation kernel
     * @param nextPow2 - least power of two greater than or equal to blockDim.x
     */
    __global__ static void mp_array_reduce_kernel1(const unsigned int n, mp_array_t input, mp_float_ptr result, const unsigned int nextPow2) {
        extern __shared__ mp_float_t sdata[];

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
            cuda::mp_add(&sdata[tid], &sdata[tid], input, i);
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
            result[bid] = sdata[tid];
        };
        __syncthreads();
    }

    /*
     * Multiple-precision summation kernel.  This kernel is exactly the same as the previous one,
     * but takes mp_float_ptr instead of mp_array_t for the input vector and mp_array_t instead of mp_float_ptr for the result
     */
    __global__ static void mp_array_reduce_kernel2(const unsigned int n, mp_float_ptr input, mp_array_t result, const unsigned int nextPow2) {
        extern __shared__ mp_float_t sdata[];
        const unsigned int tid = threadIdx.x;
        const unsigned int bid = blockIdx.x;
        const unsigned int bsize = blockDim.x;
        const unsigned int k = gridDim.x * bsize;
        unsigned int i = bid * bsize + tid;
        sdata[tid] = cuda::MP_ZERO;
        while (i < n) {
            cuda::mp_add(&sdata[tid], &sdata[tid], &input[i]);
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
            result.sign[bid] = 0;
            result.exp[bid] = sdata[tid].exp;
            result.eval[bid] = sdata[tid].eval[0];
            result.eval[bid + result.len[0]] = sdata[tid].eval[1];
            for(int j = 0; j < RNS_MODULI_SIZE; j++){
                result.digits[RNS_MODULI_SIZE * bid + j] = sdata[tid].digits[j];
            }
        }
        //__syncthreads();
    }

    /*!
     * Calculates the sum of the elements of a multiple-precision vector (two-pass summation)
     * @tparam gridDim1 - number of blocks used to launch the kernel
     * @tparam blockDim1 - number of threads per block
     * @param n - size of the vector
     * @param x - multiple-precision vector in the GPU memory
     * @param result - pointer to the sum (vector of length one) in the GPU memory
     */
    template <int gridDim1, int blockDim1>
    void mp_array_reduce(int n, mp_array_t x, mp_array_t result) {
        mp_float_ptr d_buf; // device buffer

        //Allocate memory buffers for the device results
        cudaMalloc((void **) &d_buf, sizeof(mp_float_t) * gridDim1);

        // Compute the size of shared memory allocated per block
        size_t smemsize = blockDim1 * sizeof(mp_float_t);

        // Kernel memory configurations. We prefer shared memory
        cudaFuncSetCacheConfig(mp_array_reduce_kernel1, cudaFuncCachePreferShared);
        cudaFuncSetCacheConfig(mp_array_reduce_kernel2, cudaFuncCachePreferShared);

        // Power of two that is greater that or equals to blockDim1
        const unsigned int POW = nextPow2(blockDim1);

        //Launch the 1st CUDA kernel to perform parallel summation on the GPU
        mp_array_reduce_kernel1 <<< gridDim1, blockDim1, smemsize >>> (n, x, d_buf, POW);

        //Launch the 2nd CUDA kernel to perform summation of the results of parallel blocks on the GPU
        mp_array_reduce_kernel2 <<< 1, blockDim1, smemsize >>> (gridDim1, d_buf, result, POW);

        // Cleanup
        cudaFree(d_buf);
    }

} //end of namespace


#endif //MPRES_MPREDUCT_CUH
