/*
 *  Multiple-precision NORM function for GPU (BLAS Level-1)
 *  Norm of a vector
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


#ifndef MPRES_NORM_CUH
#define MPRES_NORM_CUH

#include "asum.cuh"
#include "mblas_enum.cuh"

namespace cuda {

    /*!
     * Computes the norm of a vector x depending on the value passed as the norm operator argument.
     * @tparam gridDim1 - number of thread blocks for parallel reduction
     * @tparam blockDim1 - number of threads per block for parallel reduction
     * @param norm - type of norm to be computed
     * @param n - operation size (must be positive)
     * @param x - pointer to the vector in the global GPU memory
     * @param incx - storage spacing between elements of x (must be positive)
     * @param r - pointer to the computed norm (result) --- a vector of length one in the GPU memory
     */
    template <int gridDim1, int blockDim1>
    void mp_norm(enum mblas_norm_type norm, const int n, mp_array_t &x, const int incx, mp_array_t &r) {

        if(norm == mblas_one_norm){ // one-norm
            mp_asum<gridDim1, blockDim1>(n, x, incx, r);
            return;
        }
        else{ // infinity-norm

            // Only positive operation size and vector stride are permitted for NORM
            if(n <= 0 || incx <= 0){
                return;
            }
            mp_float_ptr d_buf; // device buffer

            // Allocate memory buffers for the device results
            cudaMalloc((void **) &d_buf, sizeof(mp_float_t) * gridDim1);

            // Compute the size of shared memory allocated per block
            size_t smemsize = blockDim1 * sizeof(mp_float_t);

            // Kernel memory configurations. We prefer shared memory
            cudaFuncSetCacheConfig(mp_array_reduce_max_abs_kernel1, cudaFuncCachePreferShared);
            cudaFuncSetCacheConfig(mp_array_reduce_max_abs_kernel2, cudaFuncCachePreferShared);

            // Power of two that is greater that or equals to blockDim1
            const unsigned int POW = nextPow2(blockDim1);

            //Launch the 1st CUDA kernel to perform parallel summation on the GPU
            mp_array_reduce_max_abs_kernel1 <<< gridDim1, blockDim1, smemsize >>> (n, x, incx, d_buf, POW);

            //Launch the 2nd CUDA kernel to perform summation of the results of parallel blocks on the GPU
            mp_array_reduce_max_abs_kernel2 <<< 1, blockDim1, smemsize >>> (gridDim1, d_buf, 1, r, POW);

            // Cleanup
            cudaFree(d_buf);
        }
    }

} //end of namespace

#endif //MPRES_NORM_CUH
