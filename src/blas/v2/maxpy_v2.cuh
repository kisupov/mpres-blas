/*
 *  Multiple-precision AXPY function for GPU with inverted sign of alpha (BLAS Level-1)
 *  Minus constant times a vector plus a vector
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

#ifndef MPRES_MAXPY_V2_CUH
#define MPRES_MAXPY_V2_CUH

#include "arith/mul.cuh"
#include "arith/add.cuh"

namespace cuda
{
    /*!
     * Computes a vector-scalar product and subtracts the result from a vector, r = -alpha * x + y
     * @param n - operation size (must be positive)
     * @param alpha - specifies the scalar alpha
     * @param x - multiple-precision input vector in the GPU memory.
     * @param y - multiple-precision input vector in the GPU memory.
     * @param r - multiple-precision result vector in the GPU memory (if y needs to be updated, just pass y instead of r)
     */
    __global__ void mp_maxpy(const int n, mp_float_ptr alpha, mp_float_ptr x, mp_float_ptr y, mp_float_ptr r) {
        __shared__ mp_float_t a;
        mp_float_t ax;
        auto i = threadIdx.x + blockIdx.x * blockDim.x;
        if(threadIdx.x == 0){
            a = alpha[0];
            a.sign = a.sign ^ 1; // a = -alpha
        }
        __syncthreads();
        while(i < n){
            cuda::mp_mul(&ax, a, x[i]);
            cuda::mp_add(&r[i], ax, y[i]);
            i += gridDim.x * blockDim.x;
        }
    }

} // namespace cuda

#endif //MPRES_MAXPY_V2_CUH
