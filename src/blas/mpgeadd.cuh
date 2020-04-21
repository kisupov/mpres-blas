/*
 *  Multiple-precision GER function for GPU (BLAS Level-2)
 *  Performs a rank-1 update of a general matrix.
 *
 *  Copyright 2020 by Konstantin Isupov and Ivan Babeshko.
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
     * C = alpha*A + beta * B
     * where alpha and beta are scalars, and A, B, C are m by n matrix.
     * The matrix should be stored in column-major order.

     * @tparam gridDim1 - number of blocks (x dimension) used to compute the signs, exponents, interval evaluations, and also to round the result in element-wise operations
     * @tparam blockDim1 - number of threads per block used to compute the signs, exponents, interval evaluations, and also to round the result in element-wise operations
     * @tparam gridDim2 - number of blocks (x dimension) used to compute the digits of multiple-precision significands in element-wise operations
     *
     * @param m - specifies the number of rows of the matrix A. The value of m must be at least zero.
     * @param n - specifies the number of columns of the matrix A. The value of n must be at least zero.
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
    template<int gridDim1, int blockDim1, int gridDim2>
    void mpgeadd(int m, int n, mp_array_t &alpha, mp_array_t &A, int lda,  mp_array_t &beta, mp_array_t &B, int ldb, mp_array_t &C, int ldc, mp_array_t &buffer){

        //Test the input parameters
        if( (m <= 0) || (n <= 0) || (lda < MAX(1, m)) || (ldb < MAX(1, m)) || (ldc < MAX(1, m)) ){
            return;
        }

        //We run 2D grids of 1D blocks.
        //Each line in the grid (i.e., all blocks with the same y coordinate) is associated with its own element of the vector y or column of the matrix.
        dim3 blocks1(gridDim1, n, 1); //Number of blocks for computing the signs, exponents, interval evaluations, and also for rounding the result
        dim3 blocks2(gridDim2, n, 1); //Number of blocks for computing residues

        /*
         * Multiplication by scalars
         */

        //Optimized case: matrix A is treated as a vector
        if(lda == 1){
            //Multiplication buffer = alpha * A - Computing the signs, exponents, and interval evaluations
            mp_vec2scal_mul_esi_kernel<<< gridDim1, blockDim1 >>> (buffer, 1, A, 1, alpha, m * n);

            //Multiplication buffer = alpha * A - Multiplying the digits in the RNS
            mp_vec2scal_mul_digits_kernel<<< gridDim2, BLOCK_SIZE_FOR_RESIDUES >>> (buffer, 1, A, 1, alpha, m * n);

            //Multiplication buffer = alpha * A  - Rounding the result
            mp_vector_round<<< gridDim1, blockDim1 >>> (buffer, 1, m * n);
        } else{
            //Multiplication buffer = alpha * A - Computing the signs, exponents, and interval evaluations
            mp_mat2scal_mul_esi_kernel<<< blocks1, blockDim1 >>> (buffer, m, A, lda, alpha, m, n);

            //Multiplication buffer = alpha * A - Multiplying the digits in the RNS
            mp_mat2scal_mul_digits_kernel<<< blocks2, BLOCK_SIZE_FOR_RESIDUES >>> (buffer, m, A, lda, alpha, m, n);

            //Multiplication buffer = alpha * A  - Rounding the result
            mp_vector_round<<< gridDim1, blockDim1 >>> (buffer, 1, m * n);
        }

        //Optimized case: matrices B and C are treated as vectors
        if(ldb == 1 && ldc == 1){
            //Multiplication С =  beta * B - Computing the signs, exponents, and interval evaluations
            mp_vec2scal_mul_esi_kernel<<< gridDim1, blockDim1 >>> (C, 1, B, 1, beta, m * n);

            //Multiplication С =  beta * B - Multiplying the digits in the RNS
            mp_vec2scal_mul_digits_kernel<<< gridDim2, BLOCK_SIZE_FOR_RESIDUES >>> (C, 1, B, 1, beta, m * n);

            //Multiplication С =  beta * B - Rounding the result
            mp_vector_round<<< gridDim1, blockDim1 >>> (C, 1, m * n);
        } else{
            //Multiplication С =  beta * B - Computing the signs, exponents, and interval evaluations
            mp_mat2scal_mul_esi_kernel<<< blocks1, blockDim1 >>> (C, ldc, B, ldb, beta, m, n);

            //Multiplication С =  beta * B - Multiplying the digits in the RNS
            mp_mat2scal_mul_digits_kernel<<< blocks2, BLOCK_SIZE_FOR_RESIDUES >>> (C, ldc, B, ldb, beta, m, n);

            //Multiplication С =  beta * B - Rounding the result
            mp_matrix_round<<< blocks1, blockDim1 >>> (C, ldc, m, n);
        }

        /*
         * Addition of two matrices
         */

        //Optimized case: matrix С is treated as a vector
        if(ldc == 1){
            //Addition of two vectors: C = C + buffer - Computing the signs, exponents, and interval evaluations
            mp_vector_add_esi_kernel<<< gridDim1, blockDim1 >>> (C, 1, C, 1, buffer, 1, m * n);

            //Addition of two vectors: C = C + buffer - Adding the digits in the RNS
            mp_vector_add_digits_kernel<<< gridDim2, BLOCK_SIZE_FOR_RESIDUES >>> (C, 1, C, 1, buffer, 1, m * n);

            //Final rounding
            mp_vector_round<<< gridDim1, blockDim1 >>> (C, 1, m * n);
        } else{
            //Addition of two matrices: C = C + buffer - Computing the signs, exponents, and interval evaluations
            mp_matrix_add_esi_kernel<<< blocks1, blockDim1 >>> (C, ldc, C, ldc, buffer, m, m, n);

            //Addition of two matrices: C = C + buffer - Adding the digits in the RNS
            mp_matrix_add_digits_kernel<<< blocks2, BLOCK_SIZE_FOR_RESIDUES >>> (C, ldc, C, ldc, buffer, m, m, n);

            //Final rounding
            mp_matrix_round<<< blocks1, blockDim1 >>> (C, ldc, m, n);
        }
    }

} // namespace cuda

#endif //MPGEADD_CUH
