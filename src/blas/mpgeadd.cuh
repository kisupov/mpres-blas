/*
 *  Multiple-precision GE_ADD function for GPU (BLAS Level-3)
 *  Matrix add and scale.
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

#ifndef MPGEADD_CUH
#define MPGEADD_CUH


#include "../mpvector.cuh"
#include "../mpmatrix.cuh"
#include "../kernel_config.cuh"

namespace cuda
{

    /*!
     * Scales two matrices A and B and stores their sum in a matrix C
     * C = alpha * A + beta * B
     * where alpha and beta are scalars, and A, B, C are m by n matrices.
     * All the  matrices should be stored in column-major order.
     *
     * @tparam blockDim1x - number of threads per block (x dimension) used to compute the signs, exponents, interval evaluations
     * @tparam blockDim1y - number of threads per block (y dimension) used to compute the signs, exponents, interval evaluations
     * @tparam gridDim2x - number of blocks (x dimension) used to compute the digits of multiple-precision significands in element-wise operations
     * @tparam gridDim2y - number of blocks (y dimension) used to compute the digits of multiple-precision significands in element-wise operations
     *
     * @param m - specifies the number of rows of the matrices. The value of m must be greater than zero.
     * @param n - specifies the number of columns of the matrices. The value of n must be greater than zero.
     * @param alpha - pointer to the scalar in the global GPU memory
     * @param A - pointer to the array, size lda * n, in the global GPU memory. Before entry, the leading m-by-n part of the array must contain the matrix A.
     * @param lda - specifies the leading dimension of A as declared in the calling (sub)program. The value of lda must be at least max(1, m).
     * @param beta - pointer to the scalar in the global GPU memory
     * @param B - pointer to the array, size ldb * n, in the global GPU memory. Before entry, the leading m-by-n part of the array must contain the matrix B.
     * @param ldb - specifies the leading dimension of B as declared in the calling (sub)program. The value of ldb must be at least max(1, m).
     * @param C - pointer to the array, size ldb * n, in the global GPU memory. After calculations, the leading m-by-n part of the array contains the matrix C
     * @param ldc - specifies the leading dimension of C as declared in the calling (sub)program. The value of ldc must be at least max(1, m).
     * @param buffer - auxiliary array in the global GPU memory, size at least m * n.
     */
    template<int blockDim1x, int blockDim1y, int gridDim2x, int gridDim2y>
    void mpgeadd(const int m, const int n, mp_array_t &alpha, mp_array_t &A, const int lda,  mp_array_t &beta, mp_array_t &B, const int ldb, mp_array_t &C, const int ldc, mp_array_t &buffer){

        //Quick return if possible
        if( (m <= 0) || (n <= 0) ){
            return;
        }
        //Test the input parameters
        if( (lda < MAX(1, m)) || (ldb < MAX(1, m)) || (ldc < MAX(1, m)) ){
            return;
        }

        //Execution configuration. We run 2D grids of 2D blocks.
        //  To compute the signs, exponents, and interval evaluations
        dim3 block1(blockDim1x, blockDim1y);
        dim3 grid1((m + block1.x - 1) / block1.x, (n + block1.y - 1) / block1.y);
        //  To compute the digits in RNS
        dim3 grid2(gridDim2x, gridDim2y);
        //  To rounding the result (we do not currently parameterize rounding)
        dim3 block3(16, 16);
        dim3 grid3((m + block3.x - 1) / block3.x, (n + block3.y - 1) / block3.y);

        /*
         * Multiplication by scalars
         */

        //Multiplication buffer = alpha * A - Computing the signs, exponents, and interval evaluations
        mp_mat2scal_mul_esi_kernel<<< grid1, block1 >>> (buffer, m, A, lda, alpha, m, n);

        //Multiplication buffer = alpha * A - Multiplying the digits in the RNS
        mp_mat2scal_mul_digits_kernel<<< grid2, BLOCK_SIZE_FOR_RESIDUES >>> (buffer, m, A, lda, alpha, m, n);

        //Multiplication buffer = alpha * A  - Rounding the result
        mp_matrix_round_kernel<<< grid3, block3 >>> (buffer, m, m, n);

        //Multiplication С =  beta * B - Computing the signs, exponents, and interval evaluations
        mp_mat2scal_mul_esi_kernel<<< grid1, block1 >>> (C, ldc, B, ldb, beta, m, n);

        //Multiplication С =  beta * B - Multiplying the digits in the RNS
        mp_mat2scal_mul_digits_kernel<<< grid2, BLOCK_SIZE_FOR_RESIDUES >>> (C, ldc, B, ldb, beta, m, n);

        //Multiplication С =  beta * B - Rounding the result
        mp_matrix_round_kernel<<< grid3, block3 >>> (C, ldc, m, n);

        /*
         * Addition of two matrices
         */

        //Addition of two matrices: C = C + buffer - Computing the signs, exponents, and interval evaluations
        mp_matrix_add_esi_kernel<<< grid1, block1 >>> (C, ldc, C, ldc, buffer, m, m, n);

        //Addition of two matrices: C = C + buffer - Adding the digits in the RNS
        mp_matrix_add_digits_kernel<<< grid2, BLOCK_SIZE_FOR_RESIDUES >>> (C, ldc, C, ldc, buffer, m, m, n);

        //Final rounding
        mp_matrix_round_kernel<<< grid3, block3 >>> (C, ldc, m, n);

    }

} // namespace cuda

#endif //MPGEADD_CUH
