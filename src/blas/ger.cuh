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

#ifndef MPRES_GER_CUH
#define MPRES_GER_CUH


#include "../mpvector.cuh"
#include "../mpmatrix.cuh"
#include "../kernel_config.cuh"

namespace cuda
{

    /*
     * Calculating the product of a vector-column of size m by a vector-row of size n: x * y^T
     * The result is a matrix of size m * n (m rows, n columns)
     * Computing the signs, exponents, and interval evaluations
     */
    __global__ static void mp_cvec2rvec_mul_esi_kernel(mp_array_t result, mp_array_t x, const int incx, mp_array_t y, const int m, const int n) {
        int lenx = x.len[0]; //actual length of x, equals to m when incx = 1
        int leny = y.len[0]; //actual length of y, it should be equal to n since it is assumed that incy = 1
        int colId = blockIdx.y; // The column index

         // code for incx equal to 1
         if(incx == 1) {
             //Iterate over the elements of y
             while (colId < n){
                 int y_sign = y.sign[colId];
                 int y_exp = y.exp[colId];
                 er_float_t y_eval0 = y.eval[colId];
                 er_float_t y_eval1 = y.eval[colId + leny];
                 er_float_ptr y_eval0ptr = &y_eval0;
                 er_float_ptr y_eval1ptr = &y_eval1;
                 int numberIdx = blockDim.x * blockIdx.x + threadIdx.x; //Index of the element of x. Must be less than m
                 //We process in the stride loop all the elements of x
                 while (numberIdx < m) {
                     result.sign[colId * m + numberIdx] = x.sign[numberIdx] ^ y_sign;
                     result.exp[colId * m + numberIdx] = x.exp[numberIdx] + y_exp;
                     cuda::er_md_rd(&result.eval[colId * m + numberIdx], &x.eval[numberIdx], y_eval0ptr, &cuda::RNS_EVAL_UNIT.upp);
                     cuda::er_md_ru(&result.eval[m * n + colId * m + numberIdx], &x.eval[lenx + numberIdx], y_eval1ptr, &cuda::RNS_EVAL_UNIT.low);
                     numberIdx += gridDim.x * blockDim.x;
                 }
                 //Go to the next column
                 colId += gridDim.y;
             }
         }
         // code for incx not equal to 1
         else {
             while (colId < n){
                 int y_sign = y.sign[colId];
                 int y_exp = y.exp[colId];
                 er_float_t y_eval0 = y.eval[colId];
                 er_float_t y_eval1 = y.eval[colId + leny];
                 er_float_ptr y_eval0ptr = &y_eval0;
                 er_float_ptr y_eval1ptr = &y_eval1;
                 int numberIdx = blockDim.x * blockIdx.x + threadIdx.x;
                 int ix = incx > 0 ? numberIdx * incx : (-m + numberIdx + 1)*incx;
                 while (numberIdx < m) {
                     result.sign[colId * m + numberIdx] = x.sign[ix] ^ y_sign;
                     result.exp[colId * m + numberIdx] = x.exp[ix] + y_exp;
                     cuda::er_md_rd(&result.eval[colId * m + numberIdx], &x.eval[ix], y_eval0ptr, &cuda::RNS_EVAL_UNIT.upp);
                     cuda::er_md_ru(&result.eval[m * n + colId * m + numberIdx], &x.eval[lenx + ix], y_eval1ptr, &cuda::RNS_EVAL_UNIT.low);
                     numberIdx += gridDim.x * blockDim.x;
                     ix += gridDim.x * blockDim.x * incx;
                 }
                 colId += gridDim.y;
             }
         }
     }

    /*
     * Calculating the product of a vector-column of size m by a vector-row of size n: x * y^T
     * The result is a matrix of size m * n (m rows, n columns)
     * Computing the digits in the RNS
     */
    __global__ static void mp_cvec2rvec_mul_digits_kernel(mp_array_t result, mp_array_t x, const int incx, mp_array_t y, const int m, const int n) {
        int lmodul = cuda::RNS_MODULI[threadIdx.x % RNS_MODULI_SIZE];
        int colId = blockIdx.y; // The column index

        // code for incx equal to 1
        if(incx == 1) {
            //Iterate over the elements of y
            while (colId < n){
                int ly = y.digits[RNS_MODULI_SIZE * colId + threadIdx.x % RNS_MODULI_SIZE];
                int index = blockIdx.x * blockDim.x + threadIdx.x; //Index of the element of x. Must be less than m * RNS_MODULI_SIZE
                //We process in the stride loop all the elements of x
                while (index < m * RNS_MODULI_SIZE) {
                    result.digits[m * RNS_MODULI_SIZE * colId + index] = cuda::mod_mul(x.digits[index], ly, lmodul);
                    index += gridDim.x * blockDim.x;
                }
                //Go to the next column
                colId += gridDim.y;
            }
        }
        // code for incx not equal to 1
        else{
            while (colId < n){
                int ly = y.digits[RNS_MODULI_SIZE * colId + threadIdx.x % RNS_MODULI_SIZE];
                int index = blockIdx.x * blockDim.x + threadIdx.x;
                int ix = incx > 0 ? (blockIdx.x * blockDim.x * incx + threadIdx.x) : ((-m + blockIdx.x + 1) * blockDim.x * incx + threadIdx.x);
                while (index < m * RNS_MODULI_SIZE) {
                    result.digits[m * RNS_MODULI_SIZE * colId + index] = cuda::mod_mul(x.digits[ix], ly, lmodul);
                    index += gridDim.x * blockDim.x;
                    ix += gridDim.x * blockDim.x * incx;
                }
                colId += gridDim.y;
            }
        }
    }



    /*!
     * Performs the rank-1 update operation
     * A = alpha*x*y^T + A
     * where alpha is a scalar, x is an m element vector, y is an n element vector and A is an m by n matrix.
     * The matrix should be stored in column-major order.

     * @tparam blockDim1x - number of threads per block (x dimension) used to compute the signs, exponents, interval evaluations
     * @tparam blockDim1y - number of threads per block (y dimension) used to compute the signs, exponents, interval evaluations
     * @tparam gridDim2x - number of blocks (x dimension) used to compute the digits of multiple-precision significands in element-wise operations
     * @tparam gridDim2y - number of blocks (y dimension) used to compute the digits of multiple-precision significands in element-wise operations
     *
     * @param m - specifies the number of rows of the matrix A. The value of m must be at least zero.
     * @param n - specifies the number of columns of the matrix A. The value of n must be at least zero.
     * @param alpha - pointer to the scalar in the global GPU memory
     * @param x - pointer to the vector in the global GPU memory, size at least (1+(m-1)*abs(incx)).
     * @param incx - storage spacing between elements of x. The value of incx must not be zero.
     * @param beta - pointer to the scalar in the global GPU memory
     * @param y - pointer to the vector in the global GPU memory, size at least (1+(n-1)*abs(incy)).
     * @param incy - storage spacing between elements of y. The value of incx must not be zero.
     * @param A - pointer to the array, size lda * n, in the global GPU memory. Before entry, the leading m-by-n part of the array must contain the matrix A.
     * @param lda - specifies the leading dimension of A as declared in the calling (sub)program. The value of lda must be at least max(1, m).

     * @param buffer1 - auxiliary array in the global GPU memory, size at least n.
     * @param buffer2 - auxiliary array of size m-by-n in the global GPU memory for storing the intermediate Cartesian product (alpha*x*y^T)
     */
    template<int blockDim1x, int blockDim1y, int gridDim2x, int gridDim2y>
    void mp_ger(const int m, const int n, mp_array_t &alpha, mp_array_t &x, const int incx, mp_array_t &y, const int incy, mp_array_t &A, const int lda, mp_array_t &buffer1, mp_array_t &buffer2){

        //Test the input parameters
        if( (m < 0) || (n < 0) || (lda < MAX(1, m)) ){
            return;
        }
        if( (incx == 0) || (incy == 0) ){
            return;
        }
        //Quick return if possible
        if( (m == 0) || (n == 0) ){
            return;
        }

        //Execution configuration. We run 2D grids of 2D blocks.
        //  To compute the signs, exponents, and interval evaluations
        dim3 block1(blockDim1x, blockDim1y);
        dim3 grid1((m + block1.x - 1) / block1.x, (n + block1.y - 1) / block1.y);
        //  To compute the digits in RNS
        dim3 grid2(gridDim2x, gridDim2y);
        int numThreadsX = (incx == 1) ? BLOCK_SIZE_FOR_RESIDUES : RNS_MODULI_SIZE;
        int numThreadsY = (incy == 1) ? BLOCK_SIZE_FOR_RESIDUES : RNS_MODULI_SIZE;
        //  To rounding the result (we do not currently parameterize rounding)
        dim3 block3(16, 16);
        dim3 grid3((m + block3.x - 1) / block3.x, (n + block3.y - 1) / block3.y);

        //Multiplication buffer1 = alpha * y - Computing the signs, exponents, and interval evaluations
        mp_vec2scal_mul_esi_kernel<<<  (n + blockDim1x - 1) / blockDim1x, blockDim1x >>> (buffer1, 1, y, incy, alpha, n);

        //Multiplication buffer1 = alpha * y - Multiplying the digits in the RNS
        mp_vec2scal_mul_digits_kernel<<< gridDim2x, numThreadsY >>> (buffer1, 1, y, incy, alpha, n);

        //Rounding the intermediate result (buffer1)
        mp_vector_round_kernel<<< (n + 32 - 1) / 32, 32 >>> (buffer1, 1, n);

        //Calculation buffer2 = x * buffer1^T - Computing the signs, exponents, and interval evaluations
        //The result is written to the intermediate buffer2, size m * n
        mp_cvec2rvec_mul_esi_kernel<<<grid1, blockDim1x * blockDim1y>>> (buffer2, x, incx, buffer1, m, n);

        //Calculation buffer2 = x * buffer1^T - Multiplying the digits in the RNS
        //The result is written to the intermediate buffer2, size m * n
        mp_cvec2rvec_mul_digits_kernel<<<grid2, numThreadsX>>> (buffer2, x, incx, buffer1, m, n);

        //Rounding the intermediate result (buffer2)
        mp_vector_round_kernel<<< (n + 32 - 1) / 32, 32 >>>(buffer2, 1, m * n);

        /*
         * Addition of two matrices
         */

        //Addition of two matrices: A = A + buffer2 - Computing the signs, exponents, and interval evaluations
        mp_matrix_add_esi_kernel<<< grid1, block1 >>> (A, lda, A, lda, buffer2, m, m, n);

        //Addition of two matrices: A = A + buffer2 - Adding the digits in the RNS
        mp_matrix_add_digits_kernel<<< grid2, BLOCK_SIZE_FOR_RESIDUES >>> (A, lda, A, lda, buffer2, m, m, n);

        //Final rounding
        mp_matrix_round_kernel<<< grid3, block3 >>> (A, lda, m, n);

    }

} // namespace cuda

#endif //MPRES_GER_CUH
