/*
 *  Multiple-precision NORM function for GPU (BLAS Level-1)
 *  Euclidean norm of a vector
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


#ifndef MPRES_NORM2_V2_CUH
#define MPRES_NORM2_V2_CUH

#include "arith/mul.cuh"
#include "arith/addabs.cuh"
#include "arith/sqrt.cuh"

namespace cuda {

    /*
    * Computes the sum of squares of the vector elements and stores result in an array of size gridDim.x
    * For final reduction, mp_sumsq_kernel2 should be used
    */
    template<int threads>
    __global__ void mp_sumsq_kernel1(const unsigned int n, const unsigned int nextPow2, mp_float_ptr x, mp_float_ptr r) {
        __shared__ mp_float_t sdata[threads];
        __shared__ mp_float_t prods[threads];
        const unsigned int tid = threadIdx.x;
        const unsigned int bid = blockIdx.x;
        const unsigned int bsize = blockDim.x;
        const unsigned int k = gridDim.x * bsize;
        unsigned int i = bid * bsize + tid;
        sdata[tid] = cuda::MP_ZERO;
        while (i < n) {
            cuda::mp_mul(&prods[threadIdx.x],&x[i], &x[i]);
            cuda::mp_add_abs(&sdata[tid], &sdata[tid], &prods[threadIdx.x]);
            i += k;
        }
        __syncthreads();
        i = nextPow2 >> 1;
        while(i >= 1){
            if ((tid < i) && (tid + i < bsize)) {
                cuda::mp_add_abs(&sdata[tid], &sdata[tid], &sdata[tid + i]);
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
     * Multiple-precision summation kernel: computes the sum of the absolute values of elements of vector x
     * @param nextPow2 - least power of two greater than or equal to blockDim.x
      */
    template<int threads>
    __global__ void mp_sumsq_kernel2(const unsigned int n, const unsigned int nextPow2, mp_float_ptr x, mp_float_ptr r) {
        __shared__ mp_float_t sdata[threads];
        const unsigned int tid = threadIdx.x;
        const unsigned int bid = blockIdx.x;
        const unsigned int bsize = blockDim.x;
        const unsigned int k = gridDim.x * bsize;
        unsigned int i = bid * bsize + tid;
        sdata[tid] = cuda::MP_ZERO;
        while (i < n) {
            cuda::mp_add_abs(&sdata[tid], &sdata[tid], &x[i]);
            i += k;
        }
        __syncthreads();
        i = nextPow2 >> 1;
        while(i >= 1){
            if ((tid < i) && (tid + i < bsize)) {
                cuda::mp_add_abs(&sdata[tid], &sdata[tid], &sdata[tid + i]);
            }
            i = i >> 1;
            __syncthreads();
        }
        if (tid == 0) {
            r[bid] = sdata[tid];
        };
        __syncthreads();
    }

    /*!
     * Computes the Euclidean norm of a vector x and returns double precision result
     * @tparam blocks - number of thread blocks for parallel reduction
     * @tparam threads - number of threads per block for parallel reduction
     * @param n - operation size (must be positive)
     * @param x - pointer to the vector in the global GPU memory
     */
    template <int blocks, int threads>
    double mp_norm2(const int n, mp_float_ptr x) {
        //Compute the sum of squares of elements of vectors, r[0] = x0*x0 + x1*x1 + ... + xn*xn
        mp_float_ptr dbuf; // Device buffer for partial results
        cudaMalloc((void **) &dbuf, sizeof(mp_float_t) * blocks);
        const unsigned int POW = nextPow2(threads); // Power of 2 that is greater that or equals to threads
        mp_sumsq_kernel1<threads><<< blocks, threads >>> (n, POW, x, dbuf);
        mp_sumsq_kernel2<threads><<< 1, threads >>> (blocks, POW, dbuf, dbuf); //dbuf[0] = sumsq(x)
        double norm2 = cuda::mp_dsqrt(dbuf);
        cudaFree(dbuf);
        return norm2;
    }

} //end of namespace

#endif //MPRES_NORM2_V2_CUH

