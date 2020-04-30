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

#ifndef MPGEMM_CUH
#define MPGEMM_CUH


#include "../mpvector.cuh"
#include "../mpmatrix.cuh"
#include "../kernel_config.cuh"
#include "../mblas_enum.cuh"

namespace cuda
{

    /*
     * Kernel that computes a matrix-matrix product with general non-transposed matrices
     */
    __global__ void gemm_notrans_kernel(const unsigned int m, const unsigned int n, const unsigned int k, mp_array_t alpha, mp_array_t A, const int lda, mp_array_t B, const int ldb, mp_array_t C, const int ldc) {
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int indexC = row + col * ldc;
        if(col < n && row < m){
            mp_float_t mul = cuda::MP_ZERO;
            mp_float_t sum = cuda::MP_ZERO;
            for(int i = 0; i < k; i++){
              // if(col == 0 && row == 0)
               // cuda::mp_mul(&mul, B, (col * ldb + i), B, (col * ldb + i));
                cuda::mp_mul(&mul, A, (lda * i + row), B, (col * ldb + i) );
                cuda::mp_mul(&mul, alpha, 0, &mul);
                cuda::mp_add(&sum, &sum, &mul);
            }
           cuda::mp_add(C, indexC, C, indexC, &sum);
        }
    }

    /*!
     * Performs one of the matrix-matrix operations
     * C = alpha*op( A )*op( B ) + beta*C
     * where  op( X ) is one of op( X ) = X   or   op( X ) = X**T,
     * alpha and beta are scalars, and A, B and C are matrices, with op( A )
     * an m by k matrix,  op( B )  a  k by n matrix and  C an m by n matrix.
     * The matrices should be stored in column-major order.
     */
    template<int blockDim1x, int blockDim1y, int gridDim2x, int gridDim2y, int blockDim3>
    void mpgemm(enum mblas_trans_type transa, enum mblas_trans_type transb, const int m, const int n, const int k, mp_array_t &alpha, mp_array_t &A, const int lda, mp_array_t &B, const int ldb, mp_array_t &beta, mp_array_t &C, const int ldc){

        //Quick return if possible
        if( (m <= 0) || (n <= 0) || (k <= 0) ){
            return;
        }

        //Test the input parameters
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
         * Multiplication buffer = alpha * A
         */
/*
        if(transa == mblas_no_trans){
            //Execution configuration.
            //  To compute the signs, exponents, and interval evaluations
            dim3 block1A(blockDim1x, blockDim1y);
            dim3 grid1A((m + block1A.x - 1) / block1A.x, (k + block1A.y - 1) / block1A.y);
            //  To compute the digits in RNS
            dim3 grid2A(gridDim2x, gridDim2y);
            //  To rounding the result (we do not currently parameterize rounding)
            dim3 block3A(16, 16);
            dim3 grid3A((m + block3A.x - 1) / block3A.x, (k + block3A.y - 1) / block3A.y);

            //Multiplication buffer = alpha * A - Computing the signs, exponents, and interval evaluations
            mp_mat2scal_mul_esi_kernel<<< grid1A, block1A >>> (buffer, m, A, lda, alpha, m, k);

            //Multiplication buffer = alpha * A - Multiplying the digits in the RNS
            mp_mat2scal_mul_digits_kernel<<< grid2A, BLOCK_SIZE_FOR_RESIDUES >>> (buffer, m, A, lda, alpha, m, k);

            //Multiplication buffer = alpha * A  - Rounding the result
            mp_matrix_round<<< grid3A, block3A >>> (buffer, m, m, k);
        } else{
            //Execution configuration.
            //  To compute the signs, exponents, and interval evaluations
            dim3 block1A(blockDim1x, blockDim1y);
            dim3 grid1A((k + block1A.x - 1) / block1A.x, (m + block1A.y - 1) / block1A.y);
            //  To compute the digits in RNS
            dim3 grid2A(gridDim2x, gridDim2y);
            //  To rounding the result (we do not currently parameterize rounding)
            dim3 block3A(16, 16);
            dim3 grid3A((k + block3A.x - 1) / block3A.x, (m + block3A.y - 1) / block3A.y);

            //Multiplication buffer = alpha * A - Computing the signs, exponents, and interval evaluations
            mp_mat2scal_mul_esi_kernel<<< grid1A, block1A >>> (buffer, k, A, lda, alpha, k, m);

            //Multiplication buffer = alpha * A - Multiplying the digits in the RNS
            mp_mat2scal_mul_digits_kernel<<< grid2A, BLOCK_SIZE_FOR_RESIDUES >>> (buffer, k, A, lda, alpha, k, m);

            //Multiplication buffer = alpha * A  - Rounding the result
            mp_matrix_round<<< grid3A, block3A >>> (buffer, k, k, m);
        }
*/

        /*
         * Multiplication C = beta * C
         */

        //Execution configuration
        //  To compute the signs, exponents, and interval evaluations
        //Execution configuration. We run 2D grids of 2D blocks.
        //  To compute the signs, exponents, and interval evaluations
        dim3 block1(blockDim1x, blockDim1y);
        dim3 grid1((m + block1.x - 1) / block1.x, (n + block1.y - 1) / block1.y);
        //  To compute the digits in RNS
        dim3 grid2(gridDim2x, gridDim2y);
        //  To rounding the result (we do not currently parameterize rounding)
        dim3 block3(16, 16);
        dim3 grid3((m + block3.x - 1) / block3.x, (n + block3.y - 1) / block3.y);

        //Multiplication C = beta * C - Computing the signs, exponents, and interval evaluations
        mp_mat2scal_mul_esi_kernel<<< grid1, block1 >>> (C, ldc, C, ldc, beta, m, n);

        //Multiplication C = beta * C - Multiplying the digits in the RNS
        mp_mat2scal_mul_digits_kernel<<< grid2, BLOCK_SIZE_FOR_RESIDUES >>> (C, ldc, C, ldc, beta, m, n);

        //Multiplication C = beta * C  - Rounding the result
        mp_matrix_round<<< grid3, block3 >>> (C, ldc, m, n);

        /*
         * General matrix multiplication with addition C = alpha * A * B + C
         */

        if( (transa == mblas_no_trans) && (transb == mblas_no_trans) ){
            //Execution configuration
            dim3 block(blockDim3, blockDim3);
            dim3 grid((n + blockDim3 - 1) / blockDim3, (m + blockDim3 - 1) / blockDim3);
            gemm_notrans_kernel<<< grid, block >>> (m, n, k, alpha, A, lda, B, ldb, C, ldc);
        }

        //TODO: Other variants will be here
    }

} // namespace cuda

#endif //MPGEMM_CUH
