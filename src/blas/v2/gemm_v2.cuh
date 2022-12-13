/*
 *  Multiple-precision GEMM function for GPU (BLAS Level-3)
 *  Computes a matrix-matrix product with general matrices.
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

#ifndef MPRES_GEMM_V2_CUH
#define MPRES_GEMM_V2_CUH

#include "arith/add.cuh"
#include "arith/mul.cuh"
#include "blas/mblas_enum.cuh"

namespace cuda {
    /*!
     * Performs one of the matrix-matrix operations:
     * C = alpha * op(A) * op(B) + beta * C
     * where op(X) is one of op(X) = X   or   op(X) = X^T,
     * alpha and beta are scalars, A, B, C are matrices, op(A) is an m-by-k matrix, op(B) is a k-by-n matrix, and C is an m-by-n matrix.
     * The matrices should be stored in column-major order.
     *
     * @note Each operation using multiple precision is performed as a single thread
     * @note No global memory buffer is required
     *
     * @param transa - specifies the form of op(A), the transposition operation applied to A.
     * @param transb - specifies the form of op(B), the transposition operation applied to B.
     * @param m - specifies the number of rows of the matrix A and of the matrix C. The value of m must be at least zero.
     * @param n - specifies the number of columns of the matrix B and the number of columns of the matrix C. The value of n must be at least zero.
     * @param k - specifies the number of columns of the matrix A and the number of rows of the matrix B. The value of k must be at least zero.
     * @param alpha - scaling factor for the matrix-matrix product.
     * @param A - the input matrix A in the GPU memory, size: (at least lda * k for non-transposed matrix) or (at least lda * m for transposed matrix).
     * @param lda - specifies the leading dimension of A. It must be positive and: (at least m for non-transposed matrix) or (at least k for transposed matrix).
     * @param B - the input matrix B in the GPU memory, size: (at least ldb * n for non-transposed matrix) or (at least ldb * k for transposed matrix).
     * @param ldb - specifies the leading dimension of B. It must be positive and: (at least k for non-transposed matrix) or (at least n for transposed matrix).
     * @param beta - scaling factor for matrix C.
     * @param C - the input/output matrix C in the GPU memory, size at least ldc * n.
     * @param ldc - specifies the leading dimension of C. It must be positive and at least m.
     */
    __global__ void mp_gemm(enum mblas_trans_type transa, enum mblas_trans_type transb, const int m, const int n, const int k, mp_float_ptr alpha, mp_float_ptr A, const int lda, mp_float_ptr B, const int ldb,
                            mp_float_ptr beta, mp_float_ptr C, const int ldc) {
        auto row = blockIdx.x * blockDim.x + threadIdx.x;
        auto col = blockIdx.y * blockDim.y + threadIdx.y;
        mp_float_t mul;
        mp_float_t sum;
        while (row < m && col < n) {
            sum = cuda::MP_ZERO;
            for (int i = 0; i < k; i++) {
                unsigned int indexA = row + lda * i;
                unsigned int indexB = i + ldb * col;
                if (transa == mblas_trans) {
                    indexA = i + lda * row;
                }
                if (transb == mblas_trans) {
                    indexB = col + ldb * i;
                }
                cuda::mp_mul(&mul, A[indexA], B[indexB]);
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

#endif //MPRES_GEMM_V2_CUH
