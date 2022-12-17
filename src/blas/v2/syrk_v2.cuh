/*
 *  Multiple-precision SYRk function for GPU (BLAS Level-3)
 *  Performs a rank-k update of a symmetric matrix C by a general matrix A.
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

#ifndef MPRES_SYRK_V2_CUH
#define MPRES_SYRK_V2_CUH

#include "arith/add.cuh"
#include "arith/mul.cuh"
#include "blas/mblas_enum.cuh"

namespace cuda {
    /**
     * Checks whether iterations should be continued depending on the uplo parameter
     * @return true if iterations over the rows/columns of the matrix are not completed
     */
    DEVICE_CUDA_FORCEINLINE bool continues(enum mblas_uplo_type uplo, const unsigned int n, const unsigned int row, const unsigned int col) {
        return (uplo == mblas_upper && col < n && row <= col) || (uplo == mblas_lower && row < n && col <= row);
    }

    /*!
     * Performs the symmetric rank-1 update operation
     * C = alpha * op(A) * op(A^T) + beta * C
     * where op(X) is one of op(X) = X or op(X) = X^T, alpha and beta are scalars,
     * C is a symmetric matrix and A is a general matrix.
     * Only the upper or lower triangle of the matrix is updated
     * The matrix should be stored in column-major order.
     *
     * @note Each operation using multiple precision is performed as a single thread
     * @note No global memory buffer is required
     *
     * @param uplo - specifies whether the upper or lower triangular part of the matrix C is used.
     * @param transa - specifies the form of op(A), the transposition operation applied to A.
     * @param n - number of rows and columns in C, must be at least zero.
     * @param k - number of columns in op(A), must be at least zero.
     * @param alpha - scaling factor for the rank-k update.
     * @param A - the input matrix A, size: (at least lda * k for non-transposed matrix) or (at least lda * n for transposed matrix).
     * @param lda - the leading dimension of A. It must be positive and: (at least n for non-transposed matrix) or (at least k for transposed matrix).
     * @param beta - scaling factor for matrix C.
     * @param C - the input/output matrix C, size at least ldc * n.
     * @param ldc - specifies the leading dimension of C. It must be positive and at least n.
     */
    __global__ void mp_syrk(enum mblas_uplo_type uplo, enum mblas_trans_type transa, const int n, const int k,
            mp_float_ptr alpha, mp_float_ptr A, const int lda, mp_float_ptr beta, mp_float_ptr C, const int ldc) {
        auto row = blockIdx.x * blockDim.x + threadIdx.x;
        auto col = blockIdx.y * blockDim.y + threadIdx.y;
        mp_float_t mul;
        mp_float_t sum;
        while (continues(uplo, n, row, col)) {
            sum = cuda::MP_ZERO;
            for (int i = 0; i < k; i++) {
                unsigned int indexA = row + lda * i;
                unsigned int indexAT = i + lda * col;
                if (transa == mblas_trans) {
                    indexA = i + lda * row;
                    indexAT = col + lda * i;
                }
                cuda::mp_mul(&mul, A[indexA], A[indexAT]);
                cuda::mp_add(&sum, sum, mul);
            }
            cuda::mp_mul(&sum, sum, alpha[0]);
            cuda::mp_mul(&mul, C[row + col * ldc], beta[0]);
            cuda::mp_add(&C[row + col * ldc], sum, mul);
            row += gridDim.x * blockDim.x;
            col += gridDim.y * blockDim.y;
        }
    }
} // namespace cuda

#endif //MPRES_SYRK_V2_CUH
