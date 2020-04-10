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

#ifndef MPGER_CUH
#define MPGER_CUH


#include "../mpvector.cuh"
#include "../kernel_config.cuh"

namespace cuda
{

    /*
     * Calculating the Cartesian product of two vectors
     * The result is a matrix of size m * n (m rows, n columns)
     * Computing the signs, exponents, and interval evaluations
     */
    __global__ static void cartesian_product_esi_kernel(mp_array_t result, mp_array_t x, const int incx, mp_array_t y, const int m, const int n) {
        int lenx = x.len[0]; //actual length of x, it is equal to m when incx = 1
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
     * Calculating the Cartesian product of two vectors
     * The result is a matrix of size m * n (m rows, n columns)
     * Computing the digits in the RNS
     */
    __global__ static void cartesian_product_digits_kernel(mp_array_t result, mp_array_t x, const int incx, mp_array_t y, const int m, const int n) {
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

    /*
     * Element-wise addition of two matrices: R = A + B
     * A - pointer to the array, size lda * n. Before entry, the leading m-by-n part of the array must contain the matrix A.
     * lda specifies the leading dimension of A as declared in the calling (sub)program. The value of lda must be at least max(1, m).
     * B - pointer to the array, size ldb * n. Before entry, the leading m-by-n part of the array must contain the matrix B.
     * ldb specifies the leading dimension of B as declared in the calling (sub)program. The value of lda must be at least max(1, m).
     * R - pointer to the result array, size ldr * n. After calculations, the leading m-by-n part of the array contains the matrix R.
     * ldr specifies the leading dimension of R as declared in the calling (sub)program. The value of ldr must be at least max(1, m).
     * Computing the signs, exponents, and interval evaluations
     */
    __global__ void mat_add_esi_kernel(mp_array_t R, const int ldr, mp_array_t A, const int lda, mp_array_t B, const int ldb, const int m, const int n) {
        // Actual matrix lengths (may differ from the operation size, n)
        int lenA = A.len[0];
        int lenB = B.len[0];
        int lenR = R.len[0];
        int colId = blockIdx.y; // The column index

        er_float_t evalA[2];
        er_float_t evalB[2];
        int expA;
        int expB;
        int signA;
        int signB;

        //Iterate over the matrix columns
        while (colId < n){
            int numberIdx = blockDim.x * blockIdx.x + threadIdx.x;
            int ida = colId * lda;
            int idb = colId * ldb;
            int idr = colId * ldr;
            //We process in the stride loop all the elements of colId-th column
            while (numberIdx < m) {
                evalA[0] = A.eval[ida + numberIdx];
                evalA[1] = A.eval[ida + numberIdx + lenA];
                evalB[0] = B.eval[idb + numberIdx];
                evalB[1] = B.eval[idb + numberIdx + lenB];

                expA = A.exp[ida + numberIdx];
                expB = B.exp[idb + numberIdx];
                signA = A.sign[ida + numberIdx];
                signB = B.sign[idb + numberIdx];

                //Exponent alignment
                int dexp = expA - expB;
                int gamma =  dexp * (dexp > 0);
                int theta = -dexp * (dexp < 0);
                int nzA = ((evalB[1].frac == 0) || (theta + evalB[1].exp) < cuda::MP_J);
                int nzB = ((evalA[1].frac == 0) || (gamma + evalA[1].exp) < cuda::MP_J);
                gamma = gamma * nzB;
                theta = theta * nzA;
                //Correction of the exponents
                expA = (expA - gamma) * nzA;
                expB = (expB - theta) * nzB;
                //Correction of the signs
                signA *= nzA;
                signB *= nzB;
                int factorA = (1 - 2 * signA) * nzA;
                int factorB = (1 - 2 * signB) * nzB;
                //Correction of the interval evaluations (multiplication by 2^gamma or 2^theta)
                evalA[0].exp += gamma;
                evalA[1].exp += gamma;
                evalB[0].exp += theta;
                evalB[1].exp += theta;
                //Change the signs of the interval evaluation bounds when the number is negative
                //The signs will not change when the number is positive
                //If the number needs to be reset, then the bounds will also be reset
                evalA[0].frac *=  factorA;
                evalA[1].frac *=  factorA;
                evalB[0].frac *=  factorB;
                evalB[1].frac *=  factorB;
                //Interval addition
                cuda::er_add_rd(&R.eval[idr + numberIdx], &evalA[signA], &evalB[signB]);
                cuda::er_add_ru(&R.eval[idr + numberIdx + lenR], &evalA[1 - signA], &evalB[1 - signB]);
                //Calculation of the exponent and preliminary calculation of the sign (the sign will be changed if restoring is required)
                R.sign[idr + numberIdx] = 0;
                R.exp[idr + numberIdx] = (expA == 0) ? expB : expA;
                //Restoring the negative result
                int minus = R.eval[idr + numberIdx].frac < 0 && R.eval[idr + numberIdx + lenR].frac < 0;
                if(minus){
                    R.sign[idr + numberIdx] = 1;
                    er_float_t tmp = R.eval[idr + numberIdx];
                    R.eval[idr + numberIdx].frac = -R.eval[idr + numberIdx + lenR].frac;
                    R.eval[idr + numberIdx].exp  = R.eval[idr + numberIdx + lenR].exp;
                    R.eval[idr + numberIdx + lenR].frac = -tmp.frac;
                    R.eval[idr + numberIdx + lenR].exp  = tmp.exp;
                }
                //Storing data for Kernel #2 in the buffer
                int4 intBuf;
                intBuf.x = gamma;
                intBuf.y = theta;
                intBuf.z = factorA;
                intBuf.w = factorB;
                R.buf[idr + numberIdx] = intBuf;
                //Go to the next iteration
                numberIdx +=  gridDim.x * blockDim.x;
            }
            colId += gridDim.y;
        }
    }

    /*
     * Element-wise addition of two matrices: R = A + B
     * A - pointer to the array, size lda * n. Before entry, the leading m-by-n part of the array must contain the matrix A.
     * lda specifies the leading dimension of A as declared in the calling (sub)program. The value of lda must be at least max(1, m).
     * B - pointer to the array, size ldb * n. Before entry, the leading m-by-n part of the array must contain the matrix B.
     * ldb specifies the leading dimension of B as declared in the calling (sub)program. The value of lda must be at least max(1, m).
     * R - pointer to the result array, size ldr * n. After calculations, the leading m-by-n part of the array contains the matrix R.
     * ldr specifies the leading dimension of R as declared in the calling (sub)program. The value of ldr must be at least max(1, m).
     * Adding the digits in the RNS
     */
    __global__ static void mat_add_digits_kernel(mp_array_t R, int ldr, mp_array_t A, int lda, mp_array_t B, const int ldb, const int m, const int n) {
        int lmodul = cuda::RNS_MODULI[threadIdx.x % RNS_MODULI_SIZE];
        int colId = blockIdx.y; // The column index
        //Iterate over the matrix columns
        while (colId < n){
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            int numberIdx = (blockIdx.x * blockDim.x + threadIdx.x) / RNS_MODULI_SIZE;
            //We process in the stride loop all the elements of i-th column
            while (index < m * RNS_MODULI_SIZE) {
                int4 intBuf = R.buf[ldr * colId + numberIdx];
                int residue = cuda::mod_axby(
                        intBuf.z * A.digits[lda * RNS_MODULI_SIZE * colId + index],
                        cuda::RNS_POW2[intBuf.x][threadIdx.x % RNS_MODULI_SIZE],
                        intBuf.w * B.digits[ldb * RNS_MODULI_SIZE * colId + index],
                        cuda::RNS_POW2[intBuf.y][threadIdx.x % RNS_MODULI_SIZE],
                        lmodul,
                        cuda::RNS_MODULI_RECIPROCAL[threadIdx.x % RNS_MODULI_SIZE]);
                //Restoring the negative result
                if (R.sign[ldr * colId + numberIdx] == 1) {
                    residue = cuda::mod_sub(lmodul, residue, lmodul);
                }
                R.digits[ldr * RNS_MODULI_SIZE * colId + index] = residue < 0 ? residue + lmodul : residue;
                //Go to the next iteration
                index += gridDim.x * blockDim.x;
                numberIdx += gridDim.x * blockDim.x / RNS_MODULI_SIZE;
            }
            colId += gridDim.y;
        }
    }

    /*
     * Rounding a multiple-precision matrix
     */
    __global__ void mp_matrix_round(mp_array_t A, const int lda, const int m, const int n) {
        int lena = A.len[0]; // Actual matrix length (may differ from the operation size, m * n)
        int colId = blockIdx.y; // The column index
        while (colId < n){
            int numberIdx =  blockDim.x * blockIdx.x + threadIdx.x;
            while (numberIdx < m) {
                #if defined(DEBUG) || defined(_DEBUG)
                if( A.eval[lda * colId + lena + numberIdx].exp != A.eval[lda * colId + numberIdx].exp ){
                    printf("\n [CUDA WARNING] Possible loss of accuracy");
                }
                #endif
                int bits = (A.eval[lda * colId + lena + numberIdx].exp - cuda::MP_H + 1) * (A.eval[lda * colId + lena + numberIdx].frac != 0);
                while (bits > 0) {
                    A.exp[lda * colId + numberIdx] += bits;
                    cuda::rns_scale2pow(&A.digits[(lda * colId + numberIdx) * RNS_MODULI_SIZE], &A.digits[(lda * colId + numberIdx) * RNS_MODULI_SIZE], bits);
                    cuda::rns_eval_compute_fast(&A.eval[lda * colId + numberIdx], &A.eval[lda * colId + lena + numberIdx], &A.digits[(lda * colId + numberIdx) * RNS_MODULI_SIZE]);
                    bits = -1;
                }
                //Go to the next iteration
                numberIdx += gridDim.x * blockDim.x;
            }
            colId += gridDim.y;
        }
    }

    /*!
     * Performs the rank 1 operation
     * A = alpha*x*y^T + A
     * where alpha is a scalar, x is an m element vector, y is an n element vector and A is an m by n matrix.
     * The matrix should be stored in column-major order.

     * @tparam gridDim1 - number of blocks (x dimension) used to compute the signs, exponents, interval evaluations, and also to round the result in element-wise operations
     * @tparam blockDim1 - number of threads per block used to compute the signs, exponents, interval evaluations, and also to round the result in element-wise operations
     * @tparam gridDim2 - number of blocks (x dimension) used to compute the digits of multiple-precision significands in element-wise operations
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
    template<int gridDim1, int blockDim1, int gridDim2>
    void mpger(int m, int n, mp_array_t &alpha, mp_array_t &x, int incx, mp_array_t &y, int incy, mp_array_t &A, int lda, mp_array_t &buffer1, mp_array_t &buffer2){

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

        //We run 2D grids of 1D blocks.
        //Each line in the grid (i.e., all blocks with the same y coordinate) is associated with its own element of the vector y or column of the matrix.
        dim3 blocks1(gridDim1, n, 1); //Number of blocks for computing the signs, exponents, interval evaluations, and also for rounding the result
        dim3 blocks2(gridDim2, n, 1); //Number of blocks for computing residues

        //Setting the number of threads per block for computing residues
        int numThreadsX = (incx == 1) ? BLOCK_SIZE_FOR_RESIDUES : RNS_MODULI_SIZE;
        int numThreadsY = (incy == 1) ? BLOCK_SIZE_FOR_RESIDUES : RNS_MODULI_SIZE;

        //Multiplication buffer1 = alpha * y - Computing the signs, exponents, and interval evaluations
        mp_vec2scal_mul_esi_kernel<<< gridDim1, blockDim1 >>> (buffer1, 1, y, incy, alpha, n);

        //Multiplication buffer1 = alpha * y - Multiplying the digits in the RNS
        mp_vec2scal_mul_digits_kernel<<< gridDim2, numThreadsY >>> (buffer1, 1, y, incy, alpha, n);

        //Rounding the intermediate result (buffer1)
        mp_vector_round<<< gridDim1, blockDim1 >>> (buffer1, 1, n);

        //Calculation of the Cartesian product: buffer2 = x * buffer1^T - Computing the signs, exponents, and interval evaluations
        //The result is written to the intermediate buffer2, size m * n
        cartesian_product_esi_kernel<<<blocks1, blockDim1>>> (buffer2, x, incx, buffer1, m, n);

        //Calculation of the Cartesian product: buffer2 = x * buffer1^T - Multiplying the digits in the RNS
        //The result is written to the intermediate buffer2, size m * n
        cartesian_product_digits_kernel<<<blocks2, numThreadsX>>> (buffer2, x, incx, buffer1, m, n);

        //Rounding the intermediate result (buffer2)
        mp_vector_round<<<gridDim1, blockDim1>>>(buffer2, 1, m * n);

        /*
         * Addition of two matrices
         */

        //Optimized case: matrices are treated as vectors
        if(lda == m){

            //Addition of two vectors: A = A + buffer2 - Computing the signs, exponents, and interval evaluations
            mp_vector_add_esi_kernel<<< gridDim1, blockDim1 >>> (A, 1, A, 1, buffer2, 1, m*n);

            //Addition of two vectors: A = A + buffer2 - Adding the digits in the RNS
            mp_vector_add_digits_kernel<<< gridDim2, BLOCK_SIZE_FOR_RESIDUES >>> (A, 1, A, 1, buffer2, 1, m*n);

            //Final rounding
            mp_vector_round<<< gridDim1, blockDim1 >>> (A, 1, m * n);
        }
        //Common case
        else{
            //Addition of two matrices: A = A + buffer2 - Computing the signs, exponents, and interval evaluations
            cuda::mat_add_esi_kernel<<< blocks1, blockDim1 >>> (A, lda, A, lda, buffer2, m, m, n);

            //Addition of two matrices: A = A + buffer2 - Adding the digits in the RNS
            cuda::mat_add_digits_kernel<<< blocks2, BLOCK_SIZE_FOR_RESIDUES >>> (A, lda, A, lda, buffer2, m, m, n);

            //Final rounding
            cuda::mp_matrix_round<<< blocks1, blockDim1 >>> (A, lda, m, n);
        }

    }

} // namespace cuda

#endif //MPGER_CUH
