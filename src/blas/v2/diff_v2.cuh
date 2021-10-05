/*
 *  Multiple-precision component-wise vector difference for GPU
 *  Constant times a vector plus a vector
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

#ifndef MPRES_DIFF_V2_CUH
#define MPRES_DIFF_V2_CUH

#include "arith/sub.cuh"

namespace cuda
{
    /*!
     * Computes a component-wise vector subtraction, r = x - y
     * @param n - operation size (must be positive)
     * @param x - multiple-precision input vector in the GPU memory.
     * @param y - multiple-precision input vector in the GPU memory.
     * @param r - multiple-precision result vector in the GPU memory (if y needs to be updated, just pass y instead of r)
     */
    __global__ void mp_diff(const int n, mp_float_ptr x, mp_float_ptr y, mp_float_ptr r) {
        auto i = threadIdx.x + blockIdx.x * blockDim.x;
        while(i < n){
            cuda::mp_sub(&r[i], &x[i], &y[i]);
            i += gridDim.x * blockDim.x;
        }
    }

} // namespace cuda

#endif //MPRES_DIFF_V2_CUH
