/*
 *  Multiple-precision SCAL function for GPU (BLAS Level-2)
 *  Computes a vector-scalar product.
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

#ifndef MPRES_SCAL_V2_CUH
#define MPRES_SCAL_V2_CUH

#include "arith/mul.cuh"

namespace cuda
{
    /*!
     * Computes the product of a vector by a scalar, x = alpha * x
     * @param n - operation size (must be positive)
     * @param alpha - specifies the scalar alpha
     * @param x - multiple-precision input vector in the GPU memory.
     * @param r - multiple-precision result vector in the GPU memory (if x needs to be updated, just pass x instead of r)
     */
    __global__ void mp_scal(const int n, mp_float_ptr alpha, mp_float_ptr x, mp_float_ptr r) {
        __shared__ mp_float_t salpha;
        auto i = threadIdx.x + blockIdx.x * blockDim.x;
        if(threadIdx.x == 0){
            salpha = alpha[0];
        }
        __syncthreads();
        while(i < n){
            cuda::mp_mul(&r[i], &salpha, &x[i]);
            i += gridDim.x * blockDim.x;
        }
    }

} // namespace cuda

#endif //MPRES_SYMV_V2_CUH
