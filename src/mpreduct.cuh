/*
 *  Multiple-precision reduction CUDA kernels (sum, sum of absolute values, max value)
 *
 *  Copyright 2019, 2020 by Konstantin Isupov.
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

#include "arith/add.cuh"
#include "arith/addabs.cuh"
#include "arith/cmpabs.cuh"

namespace cuda {

    /********************* Computing the sum of the elements of a multiple-precision vector *********************/

    /*!
     * Multiple-precision summation kernel
     * @param nextPow2 - least power of two greater than or equal to blockDim.x
     */
    __global__ void mp_array_reduce_sum_kernel1(const unsigned int n, mp_array_t input, mp_float_ptr result, const unsigned int nextPow2) {
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
            cuda::mp_add(&sdata[tid], sdata[tid], input, i);
            i += k;
        }
        __syncthreads();

        // do reduction in shared mem
        i = nextPow2 >> 1; // half of nextPow2
        while(i >= 1){
            if ((tid < i) && (tid + i < bsize)) {
                cuda::mp_add(&sdata[tid], sdata[tid], sdata[tid + i]);
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

    /*!
     * Multiple-precision summation kernel.  This kernel is exactly the same as the previous one,
     * but takes mp_float_ptr instead of mp_array_t for the input vector and mp_array_t instead of mp_float_ptr for the result
     * @param nextPow2 - least power of two greater than or equal to blockDim.x
     */
    __global__ void mp_array_reduce_sum_kernel2(const unsigned int n, mp_float_ptr input, mp_array_t result, const unsigned int nextPow2) {
        extern __shared__ mp_float_t sdata[];
        const unsigned int tid = threadIdx.x;
        const unsigned int bid = blockIdx.x;
        const unsigned int bsize = blockDim.x;
        const unsigned int k = gridDim.x * bsize;
        unsigned int i = bid * bsize + tid;
        sdata[tid] = cuda::MP_ZERO;
        while (i < n) {
            cuda::mp_add(&sdata[tid], sdata[tid], input[i]);
            i += k;
        }
        __syncthreads();
        i = nextPow2 >> 1;
        while(i >= 1){
            if ((tid < i) && (tid + i < bsize)) {
                cuda::mp_add(&sdata[tid], sdata[tid], sdata[tid + i]);
            }
            i = i >> 1;
            __syncthreads();
        }
        if (tid == 0) {
            result.sign[bid] = sdata[tid].sign;
            result.exp[bid] = sdata[tid].exp;
            result.eval[bid] = sdata[tid].eval[0];
            result.eval[bid + result.len[0]] = sdata[tid].eval[1];
            for(int j = 0; j < RNS_MODULI_SIZE; j++){
                result.digits[RNS_MODULI_SIZE * bid + j] = sdata[tid].digits[j];
            }
        }
        //__syncthreads();
    }

    /********************* Computing the sum of the absolute values of the elements of a multiple-precision vector *********************/

    /*!
     * Kernel that calculates the sum of magnitudes of the vector elements
     * @param nextPow2 - least power of two greater than or equal to blockDim.x
     */
    __global__ void mp_array_reduce_sum_abs_kernel1(const unsigned int n, mp_array_t input, int incx, mp_float_ptr result, const unsigned int nextPow2) {
        extern __shared__ mp_float_t sdata[];

        // parameters
        const unsigned int tid = threadIdx.x;
        const unsigned int bid = blockIdx.x;
        const unsigned int bsize = blockDim.x;
        const unsigned int k = gridDim.x * bsize;
        unsigned int i = bid * bsize + tid;

        // do reduction in global mem
        sdata[tid] = cuda::MP_ZERO;
        while (i < n) {
            cuda::mp_add_abs(&sdata[tid], sdata[tid], input, i * incx);
            i += k;
        }
        __syncthreads();

        // do reduction in shared mem
        i = nextPow2 >> 1; // half of nextPow2
        while(i >= 1){
            if ((tid < i) && (tid + i < bsize)) {
                cuda::mp_add_abs(&sdata[tid], sdata[tid], sdata[tid + i]);
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

    /*!
     * Kernel that calculates the sum of magnitudes of the vector elements.
     * This kernel is exactly the same as the previous one, but takes mp_float_ptr instead of mp_array_t for the input vector
     * and mp_array_t instead of mp_float_ptr for the result
     * @param nextPow2 - least power of two greater than or equal to blockDim.x
     */
    __global__ void mp_array_reduce_sum_abs_kernel2(const unsigned int n, mp_float_ptr input, int incx, mp_array_t result, const unsigned int nextPow2) {
        extern __shared__ mp_float_t sdata[];
        const unsigned int tid = threadIdx.x;
        const unsigned int bid = blockIdx.x;
        const unsigned int bsize = blockDim.x;
        const unsigned int k = gridDim.x * bsize;
        unsigned int i = bid * bsize + tid;
        sdata[tid] = cuda::MP_ZERO;
        while (i < n) {
            cuda::mp_add_abs(&sdata[tid], sdata[tid], input[i*incx]);
            i += k;
        }
        __syncthreads();
        i = nextPow2 >> 1;
        while(i >= 1){
            if ((tid < i) && (tid + i < bsize)) {
                cuda::mp_add_abs(&sdata[tid], sdata[tid], sdata[tid + i]);
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
    }

    /********************* Computing the maximum of the absolute values of the elements of a multiple-precision vector *********************/

    /*!
      * Kernel that calculates the maximum absolute value of the elements of a multiple-precision vector
      * @param nextPow2 - least power of two greater than or equal to blockDim.x
      */
    __global__ void mp_array_reduce_max_abs_kernel1(const unsigned int n, mp_array_t input, int incx, mp_float_ptr result, const unsigned int nextPow2) {
        extern __shared__ mp_float_t sdata[];

        // parameters
        const unsigned int tid = threadIdx.x;
        const unsigned int bid = blockIdx.x;
        const unsigned int bsize = blockDim.x;
        const unsigned int k = gridDim.x * bsize;
        unsigned int i = bid * bsize + tid;
        unsigned int ix = i * incx;

        sdata[tid] = cuda::MP_ZERO; //since we attempt to find the maximum absolute value
        while (i < n) {
            if(cuda::mp_cmp_abs(input, ix, sdata[tid]) == 1){
                //sdata[tid] = input[ix]
                sdata[tid].exp = input.exp[ix];
                sdata[tid].sign = input.sign[ix];
                sdata[tid].eval[0] = input.eval[ix];
                sdata[tid].eval[1] = input.eval[ix + input.len[0]];
                for(int j = 0; j < RNS_MODULI_SIZE; j++){
                    sdata[tid].digits[j] = input.digits[ix * RNS_MODULI_SIZE + j];
                }
            }
            i += k;
            ix += k * incx;
        }
        __syncthreads();

        // do reduction in shared mem
        i = nextPow2 >> 1; // half of nextPow2
        while(i >= 1){
            if ( (tid < i) && (tid + i < bsize) && (cuda::mp_cmp_abs(sdata[tid + i], sdata[tid]) == 1) ) {
                sdata[tid] = sdata[tid + i];
            }
            i = i >> 1;
            __syncthreads();
        }

        // write the absolute value of the result for this block to global mem
        if (tid == 0) {
            sdata[tid].sign = 0;
            result[bid] = sdata[tid];
        };
        __syncthreads();
    }

    /*!
    * Kernel that calculates the maximum absolute value of the elements of a multiple-precision vector.
    * This kernel is exactly the same as the previous one, but takes mp_float_ptr instead of mp_array_t for the input vector
    * and mp_array_t instead of mp_float_ptr for the result
    * @param nextPow2 - least power of two greater than or equal to blockDim.x
    */
    __global__ void mp_array_reduce_max_abs_kernel2(const unsigned int n, mp_float_ptr input, int incx, mp_array_t result, const unsigned int nextPow2) {
        extern __shared__ mp_float_t sdata[];
        const unsigned int tid = threadIdx.x;
        const unsigned int bid = blockIdx.x;
        const unsigned int bsize = blockDim.x;
        const unsigned int k = gridDim.x * bsize;
        unsigned int i = bid * bsize + tid;
        unsigned int ix = i * incx;

        sdata[tid] = cuda::MP_ZERO;
        while (i < n) {
            if(cuda::mp_cmp_abs(input[ix], sdata[tid]) == 1){
                sdata[tid] = input[ix];
            }
            i += k;
            ix += k * incx;
        }
        __syncthreads();
        i = nextPow2 >> 1;
        while(i >= 1){
            if ((tid < i) && (tid + i < bsize) && (cuda::mp_cmp_abs(sdata[tid + i], sdata[tid]) == 1)) {
                sdata[tid] = sdata[tid + i];
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
    }

} //end of namespace


#endif //MPRES_MPREDUCT_CUH
