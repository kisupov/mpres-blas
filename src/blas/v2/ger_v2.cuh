/*
 *  Multiple-precision GER function for GPU (BLAS Level-2)
 *  Performs a rank-1 update of a general matrix.
 *
 *  Copyright 2022 by Konstantin Isupov.
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

#ifndef MPRES_GER_V2_CUH
#define MPRES_GER_V2_CUH

#include "arith/add.cuh"
#include "arith/mul.cuh"
#include "blas/mblas_enum.cuh"

namespace cuda {
    /*!
     * Performs the general rank-1 update operation
     * A = A + alpha * x * y^T
     * where alpha is a scalar, x is an m element vector, y is an n element vector and A is an m by n matrix.
     * The matrix should be stored in column-major order.
     *
     * @note Each operation using multiple precision is performed as a single thread
     * @note No global memory buffer is required
     *
     * @param m - specifies the number of rows of the matrix A.
     * @param n - specifies the number of columns of the matrix A.
     * @param alpha - buffer holding the input scalar alpha in the GPU memory.
     * @param x - buffer holding the input vector x in the GPU memory, size at least (1 + (m - 1)*abs(incx)).
     * @param incx - specifies the increment for the elements of x. The value of incx must not be zero.
     * @param y - buffer holding the input vector y in the GPU memory, size at least (1 + (n - 1)*abs(incy)).
     * @param incy - specifies the increment for the elements of y. The value of incy must not be zero.
     * @param A - buffer holding the updated matrix A in the GPU memory, size at least lda * n.
     * @param lda - specifies the leading dimension of A as declared in the calling (sub)program. Must be positive and at least m as column major layout is used.
     */
    __global__ void mp_ger(const int m, const int n, mp_float_ptr alpha, mp_float_ptr x, const int incx, mp_float_ptr y, const int incy, mp_float_ptr A, const int lda) {
        __shared__ mp_float_t a;
        mp_float_t axy;
        auto row = blockIdx.x * blockDim.x + threadIdx.x;
        auto col = blockIdx.y * blockDim.y + threadIdx.y;
        if (threadIdx.x == 0 && threadIdx.y == 0) {
            a = alpha[0];
        }
        __syncthreads();
        while (row < m && col < n) {
            auto indexA = row + col * lda;
            auto ix = incx > 0 ? row * incx : (-m + row + 1) * incx;
            auto iy = incy > 0 ? col * incy : (-n + col + 1) * incy;
            cuda::mp_mul(&axy, a, x[ix]);
            cuda::mp_mul(&axy, axy, y[iy]);
            cuda::mp_add(&A[indexA], A[indexA], axy);
            row += gridDim.x * blockDim.x;
            col += gridDim.y * blockDim.y;
        }
    }
} // namespace cuda

#endif //MPRES_GER_V2_CUH
