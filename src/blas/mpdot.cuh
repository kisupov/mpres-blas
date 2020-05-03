/*
 *  Multiple-precision DOT function for GPU (BLAS Level-1)
 *  Dot product of two vectors
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


#ifndef MPDOT_CUH
#define MPDOT_CUH

#include "../mpreduct.cuh"
#include "../mpvector.cuh"
#include "../kernel_config.cuh"

namespace cuda {

    /*
     * Calculates the sum of the elements of a multiple-precision vector (two-pass summation)
     * @tparam gridDim1 - number of blocks used to launch the kernel
     * @tparam blockDim1 - number of threads per block
     * @param n - size of the vector
     * @param x - multiple-precision vector in the GPU memory
     * @param result - pointer to the sum (vector of length one) in the GPU memory
     */
    template <int gridDim1, int blockDim1>
    static void mp_array_reduce_sum(int n, mp_array_t x, mp_array_t result) {
        mp_float_ptr d_buf; // device buffer

        //Allocate memory buffers for the device results
        cudaMalloc((void **) &d_buf, sizeof(mp_float_t) * gridDim1);

        // Compute the size of shared memory allocated per block
        size_t smemsize = blockDim1 * sizeof(mp_float_t);

        // Kernel memory configurations. We prefer shared memory
        cudaFuncSetCacheConfig(mp_array_reduce_sum_kernel1, cudaFuncCachePreferShared);
        cudaFuncSetCacheConfig(mp_array_reduce_sum_kernel2, cudaFuncCachePreferShared);

        // Power of two that is greater that or equals to blockDim1
        const unsigned int POW = nextPow2(blockDim1);

        //Launch the 1st CUDA kernel to perform parallel summation on the GPU
        mp_array_reduce_sum_kernel1 <<< gridDim1, blockDim1, smemsize >>> (n, x, d_buf, POW);

        //Launch the 2nd CUDA kernel to perform summation of the results of parallel blocks on the GPU
        mp_array_reduce_sum_kernel2 <<< 1, blockDim1, smemsize >>> (gridDim1, d_buf, result, POW);

        // Cleanup
        cudaFree(d_buf);
    }


    /*!
     * Computes a vector-vector dot product, r[0] = x0*y0 + x1*y1 + ... + xn*yn
     * @tparam gridDim1 - number of thread blocks used to compute the signs, exponents, interval evaluations, and also to round the result in vector-vector multiplication
     * @tparam blockDim1 - number of threads per block used to compute the signs, exponents, interval evaluations, and also to round the result in vector-vector multiplication
     * @tparam gridDim2 - number of thread blocks used to compute the digits of multiple-precision significands in vector-vector multiplication
     * @tparam gridDim3 - number of thread blocks for parallel summation
     * @tparam blockDim3 - number of threads per block for parallel summation
     * @param n - operation size (must be positive)
     * @param x - pointer to the first vector in the global GPU memory
     * @param incx - storage spacing between elements of x (must be non-zero)
     * @param y - pointer to the second vector in the global GPU memory
     * @param incy - storage spacing between elements of y (must be non-zero)
     * @param r - pointer to the inner product (vector of length one) in the GPU memory
     * @param buffer - array of size n in the global GPU memory
     */
    template <int gridDim1, int blockDim1, int gridDim2, int gridDim3, int blockDim3>
    void mpdot(const int n, mp_array_t &x, const int incx, mp_array_t &y, const int incy, mp_array_t &r, mp_array_t &buffer) {

        // Only positive operation size is permitted for DOT
        if(n <= 0){
            return;
        }

        // Block size for computing residues. If either incx or incy is not equal to 1, then numThreadsXY must be equal to RNS_MODULI_SIZE
        int numThreadsXY = (incx == 1 && incy == 1) ? BLOCK_SIZE_FOR_RESIDUES : RNS_MODULI_SIZE;

        //Vector-vector multiplication - Computing the signs, exponents, and interval evaluations
        mp_vector_mul_esi_kernel <<< gridDim1, blockDim1 >>> (buffer, 1, x, incx, y, incy, n);

        //Multiplication - Multiplying the digits in the RNS
        mp_vector_mul_digits_kernel <<< gridDim2, numThreadsXY >>> (buffer, 1, x, incx, y, incy, n);

        //Multiplication - Rounding the intermediate result
        mp_vector_round <<< gridDim1, blockDim1>>> (buffer, 1, n);

        //Two-pass summation
        mp_array_reduce_sum< gridDim3, blockDim3 >(n, buffer, r);

    }

} //end of namespace

#endif //MPDOT_CUH

