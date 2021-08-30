/*
 *  Multiple-precision GEMM function for GPU (BLAS Level-3)
 *  Computes a matrix-matrix product with general matrices.
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

#ifndef MPRES_GEMM_CUH
#define MPRES_GEMM_CUH


#include "../arith/add.cuh"
#include "../arith/mul.cuh"
#include "../mpmatrix.cuh"
#include "../kernel_config.cuh"
#include "mblas_enum.cuh"

namespace cuda
{

    /*
     * Kernel that performs ordinary multiplication of non-transposed matrices, C = A * B
     */
    __global__ void matrix_multiply_notrans_kernel(const unsigned int m, const unsigned int n, const unsigned int k, mp_array_t alpha, mp_array_t A, const int lda, mp_array_t B, const int ldb, mp_array_t C, const int ldc) {
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int indexC = row + col * ldc;
        if(col < n && row < m){
            mp_float_t mul = cuda::MP_ZERO;
            mp_float_t sum = cuda::MP_ZERO;
            for(int i = 0; i < k; i++){
                cuda::mp_mul(&mul, A, (lda * i + row), B, (col * ldb + i) );
                cuda::mp_add(&sum, &sum, &mul);
            }
            C.exp[indexC] = sum.exp;
            C.sign[indexC] = sum.sign;
            C.eval[indexC] = sum.eval[0];
            C.eval[indexC + C.len[0]] = sum.eval[1];
            for(int i = 0; i < RNS_MODULI_SIZE; i++){
                C.digits[indexC * RNS_MODULI_SIZE + i] = sum.digits[i];
            }
        }
    }


    /*!
     * Performs one of the matrix-matrix operations
     * C = alpha * op(A) * op(B) + beta * C
     * where op(X) is one of op(X) = X   or   op(X) = X^T,
     * alpha and beta are scalars, and A, B and C are matrices, with
     * op(A) an m by k matrix,  op(B)  a  k by n matrix and  C an m by n matrix.
     * All the matrices should be stored in column-major order.
     */
    template<int blockDim1x, int blockDim1y, int gridDim2x, int gridDim2y, int blockDim3>
    void mpgemm(enum mblas_trans_type transa, enum mblas_trans_type transb, const int m, const int n, const int k, mp_array_t &alpha, mp_array_t &A, const int lda, mp_array_t &B, const int ldb, mp_array_t &beta, mp_array_t &C, const int ldc, mp_array_t &buffer){

        /*
         * Quick return if possible
         */
        if( (m <= 0) || (n <= 0) || (k <= 0) ){
            return;
        }

        /*
         * Test the input parameters
         */
        if( (transa == mblas_no_trans) && (lda < MAX(1, m)) ){
            return;
        }
        if( (transa == mblas_trans) && (lda < MAX(1, k)) ){
            return;
        }
        if( (transb == mblas_no_trans) && (ldb < MAX(1, k)) ){
            return;
        }
        if( (transb == mblas_trans) && (ldb < MAX(1, n)) ){
            return;
        }
        if( ldc < MAX(1, m)){
            return;
        }

        /*
         * We perform multiplication in two stages:
         * In the first step, ordinary matrix multiplication is performed, buffer = A * B
         * In the second step, digit-parallel matrix accumulation and scale is performed, C = alpha * buffer + beta * C
         */

        /*
         * Matrix multiplication, buffer =  A * B
         */

        if( (transa == mblas_no_trans) && (transb == mblas_no_trans) ){
            //Execution configuration
            dim3 block(blockDim3, blockDim3);
            dim3 grid((n + blockDim3 - 1) / blockDim3, (m + blockDim3 - 1) / blockDim3);
            matrix_multiply_notrans_kernel<<< grid, block >>> (m, n, k, alpha, A, lda, B, ldb, buffer, m);
        }
        if( (transa == mblas_trans) && (transb == mblas_no_trans) ){
            printf("\nOnly non-transposed matrices are currently supported.");
            return;
        }
        if( (transa == mblas_no_trans) && (transb == mblas_trans) ){
            printf("\nOnly non-transposed matrices are currently supported.");
            return;
        }
        if( (transa == mblas_trans) && (transb == mblas_trans) ){
            printf("\nOnly non-transposed matrices are currently supported.");
            return;
        }

        /*
         *  Matrix accumulation and scale, C = alpha * buffer + beta * C
         */

        //Execution configuration
        //  To compute the signs, exponents, and interval evaluations
        dim3 block1(blockDim1x, blockDim1y);
        dim3 grid1((m + block1.x - 1) / block1.x, (n + block1.y - 1) / block1.y);
        //  To compute the digits in RNS
        dim3 grid2(gridDim2x, gridDim2y);
        //  To rounding the result (we do not currently parameterize rounding)
        dim3 block3(16, 16);
        dim3 grid3((m + block3.x - 1) / block3.x, (n + block3.y - 1) / block3.y);

        //Multiplication buffer = alpha * buffer - Computing the signs, exponents, and interval evaluations
        mp_mat2scal_mul_esi_kernel<<< grid1, block1 >>> (buffer, m, buffer, m, alpha, m, n);

        //Multiplication buffer = alpha * buffer - Multiplying the digits in the RNS
        mp_mat2scal_mul_digits_kernel<<< grid2, BLOCK_SIZE_FOR_RESIDUES >>> (buffer, m, buffer, m, alpha, m, n);

        //Multiplication buffer = alpha * buffer  - Rounding the result
        mp_matrix_round_kernel<<< grid3, block3 >>> (buffer, m, m, n);

        //Multiplication C = beta * C - Computing the signs, exponents, and interval evaluations
        mp_mat2scal_mul_esi_kernel<<< grid1, block1 >>> (C, ldc, C, ldc, beta, m, n);

        //Multiplication C = beta * C - Multiplying the digits in the RNS
        mp_mat2scal_mul_digits_kernel<<< grid2, BLOCK_SIZE_FOR_RESIDUES >>> (C, ldc, C, ldc, beta, m, n);

        //Multiplication C = beta * C  - Rounding the result
        mp_matrix_round_kernel<<< grid3, block3 >>> (C, ldc, m, n);

        //Addition of two matrices: C = C + buffer - Computing the signs, exponents, and interval evaluations
        mp_matrix_add_esi_kernel<<< grid1, block1 >>> (C, ldc, C, ldc, buffer, m, m, n);

        //Addition of two matrices: C = C + buffer - Adding the digits in the RNS
        mp_matrix_add_digits_kernel<<< grid2, BLOCK_SIZE_FOR_RESIDUES >>> (C, ldc, C, ldc, buffer, m, m, n);

        //Final rounding
        mp_matrix_round_kernel<<< grid3, block3 >>> (C, ldc, m, n);
    }

} // namespace cuda

#endif //MPRES_GEMM_CUH
