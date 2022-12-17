/*
 *  Multiple-precision SYR function for GPU (BLAS Level-2)
 *  Performs a rank-1 update of a symmetric matrix.
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

#ifndef MPRES_SYR_V2_CUH
#define MPRES_SYR_V2_CUH

#include "arith/add.cuh"
#include "arith/mul.cuh"
#include "blas/mblas_enum.cuh"

namespace cuda {
    /*!
     * Performs the symmetric rank-1 update operation
     * A = A + alpha * x * x^T
     * where alpha is a scalar, x is a vector of length n, and A is an n by n symmetric matrix.
     * Only the upper or lower triangle of the matrix is updated
     * The matrix should be stored in column-major order.
     *
     * @note Each operation using multiple precision is performed as a single thread
     * @note No global memory buffer is required
     *
     * @param uplo - specifies whether the upper or lower triangular part of the matrix A is used.
     * @param n - number of columns of A, must be at least zero.
     * @param alpha - scaling factor for the rank-1 update.
     * @param x - the input vector x in the GPU memory, size at least (1 + (n - 1)*abs(incx)).
     * @param incx - the increment for the elements of x. The value of incx must not be zero.
     * @param A - the input/output matrix A in the GPU memory, size at least lda * n.
     * @param lda - the leading dimension of A. It must be positive and at least n.
     */
    __global__ void mp_syr(enum mblas_uplo_type uplo, const int n, mp_float_ptr alpha, mp_float_ptr x, const int incx, mp_float_ptr A, const int lda) {
        __shared__ mp_float_t a;
        mp_float_t axx;
        auto row = blockIdx.x * blockDim.x + threadIdx.x;
        auto col = blockIdx.y * blockDim.y + threadIdx.y;
        if (threadIdx.x == 0 && threadIdx.y == 0) {
            a = alpha[0];
        }
        __syncthreads();
        if (uplo == mblas_upper) { //Access the upper part of the matrix
            while (col < n && row <= col) {
                auto ir = incx > 0 ? row * incx : (-n + row + 1) * incx;
                auto ic = incx > 0 ? col * incx : (-n + col + 1) * incx;
                cuda::mp_mul(&axx, x[ir], x[ic]);
                cuda::mp_mul(&axx, axx, a);
                cuda::mp_add(&A[row + col * lda], A[row + col * lda], axx);
                row += gridDim.x * blockDim.x;
                col += gridDim.y * blockDim.y;
            }
        } else { //Access the lower part of the matrix
            while (row < n && col <= row) {
                auto ir = incx > 0 ? row * incx : (-n + row + 1) * incx;
                auto ic = incx > 0 ? col * incx : (-n + col + 1) * incx;
                cuda::mp_mul(&axx, x[ir], x[ic]);
                cuda::mp_mul(&axx, axx, a);
                cuda::mp_add(&A[row + col * lda], A[row + col * lda], axx);
                row += gridDim.x * blockDim.x;
                col += gridDim.y * blockDim.y;
            }
        }


    }
} // namespace cuda

#endif //MPRES_SYR_V2_CUH
