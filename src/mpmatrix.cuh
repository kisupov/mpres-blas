/*
 *  Operations with matrix of multiple-precision numbers on the GPU
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

#ifndef MPRES_MPMATRIX_CUH
#define MPRES_MPMATRIX_CUH

#include "arith/mpcommon.cuh"

namespace cuda {

    /********************* Matrix addition kernels *********************/

    /*!
    * Addition of two matrices: R = A + B
    * Kernel #1 --- Computing the exponents, signs, and interval evaluations (e-s-i)
    * @note All matrices are assumed to be stored in the column major order, that is, [column 1] [column 2] ... [column n]
    * @note For this kernel, the execution configuration should be as follows:
       * dim3 block(blockSizeX, blockSizeY);
       * dim3 grid((m + block.x - 1) / block.x, (n + block.y - 1) / block.y);
    * @param R - pointer to the result array, size ldr * n. After calculations, the leading m-by-n part of the array contains the matrix R
    * @param ldr - specifies the leading dimension of R as declared in the calling (sub)program. The value of ldr must be at least max(1, m)
    * @param A  - pointer to the array, size lda * n. Before entry, the leading m-by-n part of the array must contain the matrix A.
    * @param lda - specifies the leading dimension of A as declared in the calling (sub)program. The value of lda must be at least max(1, m).
    * @param B - pointer to the array, size ldb * n. Before entry, the leading m-by-n part of the array must contain the matrix B.
    * @param ldb - specifies the leading dimension of B as declared in the calling (sub)program. The value of lda must be at least max(1, m).
    * @param m - specifies the number of rows of the matrices
    * @param n - specifies the number of columns of the matrices
    */
    __global__ void mp_matrix_add_esi_kernel(mp_array_t R, const int ldr, mp_array_t A, const int lda, mp_array_t B, const int ldb, const int m, const int n) {
        // Actual matrix lengths (may differ from the operation size, n)
        unsigned int lenA = A.len[0];
        unsigned int lenB = B.len[0];
        unsigned int lenR = R.len[0];
        er_float_t evalA[2];
        er_float_t evalB[2];
        int expA;
        int expB;
        int signA;
        int signB;

        unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;
        unsigned int ida = iy * lda + ix;
        unsigned int idb = iy * ldb + ix;
        unsigned int idr = iy * ldr + ix;

        if (ix < m && iy < n){
                evalA[0] = A.eval[ida];
                evalA[1] = A.eval[ida + lenA];
                evalB[0] = B.eval[idb];
                evalB[1] = B.eval[idb + lenB];

                expA = A.exp[ida];
                expB = B.exp[idb];
                signA = A.sign[ida];
                signB = B.sign[idb];

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
                cuda::er_add_rd(&R.eval[idr], &evalA[signA], &evalB[signB]);
                cuda::er_add_ru(&R.eval[idr + lenR], &evalA[1 - signA], &evalB[1 - signB]);
                //Calculation of the exponent
                R.exp[idr] = (expA == 0) ? expB : expA;
                //Restoring the negative result
                int minus = R.eval[idr].frac < 0 && R.eval[idr + lenR].frac < 0;
                R.sign[idr] = minus;
            if(minus){
                    er_float_t tmp = R.eval[idr];
                    R.eval[idr].frac = -R.eval[idr + lenR].frac;
                    R.eval[idr].exp  = R.eval[idr + lenR].exp;
                    R.eval[idr + lenR].frac = -tmp.frac;
                    R.eval[idr + lenR].exp  = tmp.exp;
                }
                //Storing data for Kernel #2 in the buffer
                int4 intBuf;
                intBuf.x = gamma;
                intBuf.y = theta;
                intBuf.z = factorA;
                intBuf.w = factorB;
                R.buf[idr] = intBuf;
        }
    }

    /*!
     * Addition of two matrices: R = A + B
     * Kernel #2 --- Computing the significands in the RNS (digits)
     * @note All matrices are assumed to be stored in the column major order, that is, [column 1] [column 2] ... [column n]
     * @note This kernel can be run on a 2D grid of 1D blocks.
     * @param R - pointer to the result array, size ldr * n. After calculations, the leading m-by-n part of the array contains the matrix R
     * @param ldr - specifies the leading dimension of R as declared in the calling (sub)program. The value of ldr must be at least max(1, m)
     * @param A  - pointer to the array, size lda * n. Before entry, the leading m-by-n part of the array must contain the matrix A.
     * @param lda - specifies the leading dimension of A as declared in the calling (sub)program. The value of lda must be at least max(1, m).
     * @param B - pointer to the array, size ldb * n. Before entry, the leading m-by-n part of the array must contain the matrix B.
     * @param ldb - specifies the leading dimension of B as declared in the calling (sub)program. The value of lda must be at least max(1, m).
     * @param m - specifies the number of rows of the matrices
     * @param n - specifies the number of columns of the matrices
     */
    __global__ static void mp_matrix_add_digits_kernel(mp_array_t R, const int ldr, mp_array_t A, const int lda, mp_array_t B, const int ldb, const int m, const int n) {
        int lmodul = cuda::RNS_MODULI[threadIdx.x % RNS_MODULI_SIZE];
        //Iterate over the matrix columns
        int colId = blockIdx.y; // The column index
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
            //Go to the next column
            colId += gridDim.y;
        }
    }

    /********************* Matrix-scalar multiplication kernels *********************/

    /*!
     * Parallel element-wise multiplication of a matrix by a scalar (R = alpha * A)
     * Kernel #1 --- Computing the exponents, signs, and interval evaluations (e-s-i)
     * @note All matrices are assumed to be stored in the column major order, that is, [column 1] [column 2] ... [column n]
     * @note For this kernel, the execution configuration should be as follows:
        * dim3 block(blockSizeX, blockSizeY);
        * dim3 grid((m + block.x - 1) / block.x, (n + block.y - 1) / block.y);
     * @param R - pointer to the result array, size ldr * n. After calculations, the leading m-by-n part of the array contains the matrix R
     * @param ldr - specifies the leading dimension of R as declared in the calling (sub)program. The value of ldr must be at least max(1, m)
     * @param A  - pointer to the array, size lda * n. Before entry, the leading m-by-n part of the array must contain the matrix A.
     * @param lda - specifies the leading dimension of A as declared in the calling (sub)program. The value of lda must be at least max(1, m).
     * @param alpha - pointer to the scalar (vector of length one) in the GPU memory
     * @param m - specifies the number of rows of the matrices
     * @param n - specifies the number of columns of the matrices
     */
    __global__ void mp_mat2scal_mul_esi_kernel(mp_array_t R, const int ldr, mp_array_t A, const int lda, mp_array_t alpha, const int m, const int n) {
        unsigned int lena = A.len[0]; //actual length of A
        unsigned int lenr = R.len[0]; //actual length of R
        int alpha_sign = alpha.sign[0];
        int alpha_exp = alpha.exp[0];
        er_float_t alpha_ev0 = alpha.eval[0];
        er_float_t alpha_ev1 = alpha.eval[1];

        unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;
        unsigned int ida = iy * lda + ix;
        unsigned int idr = iy * ldr + ix;

        if (ix < m && iy < n){
            R.sign[idr] = A.sign[ida] ^ alpha_sign;
            R.exp[idr] = A.exp[ida] + alpha_exp;
            cuda::er_md_rd(&R.eval[idr], &A.eval[ida], &alpha_ev0, &cuda::RNS_EVAL_UNIT.upp);
            cuda::er_md_ru(&R.eval[lenr + idr], &A.eval[lena + ida], &alpha_ev1, &cuda::RNS_EVAL_UNIT.low);
        }
    }

    /*!
     * Parallel element-wise multiplication of a matrix by a scalar (R = alpha * A)
     * Kernel #2 --- Computing the significands in the RNS (digits)
     * @note All matrices are assumed to be stored in the column major order, that is, [column 1] [column 2] ... [column n]
     * @note This kernel can be run on a 2D grid of 1D blocks.
     * @param R - pointer to the result array, size ldr * n. After calculations, the leading m-by-n part of the array contains the matrix R
     * @param ldr - specifies the leading dimension of R as declared in the calling (sub)program. The value of ldr must be at least max(1, m)
     * @param A  - pointer to the array, size lda * n. Before entry, the leading m-by-n part of the array must contain the matrix A.
     * @param lda - specifies the leading dimension of A as declared in the calling (sub)program. The value of lda must be at least max(1, m).
     * @param alpha - pointer to the scalar (vector of length one) in the GPU memory
     * @param m - specifies the number of rows of the matrices
     * @param n - specifies the number of columns of the matrices
     */
    __global__ void mp_mat2scal_mul_digits_kernel(mp_array_t R, const int ldr, mp_array_t A, const int lda, mp_array_t alpha, const int m, const int n) {
        int lmodul = cuda::RNS_MODULI[threadIdx.x % RNS_MODULI_SIZE];
        int lalpha = alpha.digits[threadIdx.x % RNS_MODULI_SIZE];
        //Iterate over matrix columns / vector elements
        int colId = blockIdx.y; // The column index
        while (colId < n){
            int index = blockIdx.x * blockDim.x + threadIdx.x; //Index of the element of A in the i-th column. Must be less than m * RNS_MODULI_SIZE
            //We process in the stride loop all the elements of the i-th column of A
            while (index < m * RNS_MODULI_SIZE) {
                R.digits[colId * ldr * RNS_MODULI_SIZE + index] = cuda::mod_mul(A.digits[colId * lda * RNS_MODULI_SIZE + index], lalpha, lmodul);
                index += gridDim.x * blockDim.x;
            }
            //Go to the next column
            colId += gridDim.y;
        }
    }

    /********************* Matrix diagonal scaling kernels (multiplying a general matrix by a diagonal matrix) *********************/

    /*!
     * Multiplication of a general m-by-n matrix A by a diagonal n-by-n matrix on the right which is stored as a vector x of size n
     * Each i-th column of the matrix A is multiplied by i-th element of the vector x
     * The result is written into the matrix R of size m by n
     * Kernel #1 --- Computing the exponents, signs, and interval evaluations (e-s-i)
     * @note All matrices are assumed to be stored in the column major order, that is, [column 1] [column 2] ... [column n]
     * @note This kernel can be run on a 2D grid of 1D blocks. Each line in the grid (i.e., all blocks with the same y coordinate) is associated with its own column of the matrix.
     * @param R - pointer to the result array, size ldr * n. After calculations, the leading m-by-n part of the array contains the matrix R
     * @param ldr - specifies the leading dimension of R as declared in the calling (sub)program. The value of ldr must be at least max(1, m)
     * @param A - pointer to the array, size lda * n. Before entry, the leading m-by-n part of the array must contain the matrix A.
     * @param lda - specifies the leading dimension of A as declared in the calling (sub)program. The value of lda must be at least max(1, m).
     * @param x - pointer to the vector in the global GPU memory, size at least (1+(n-1)*abs(incx)).
     * @param incx - storage spacing between elements of x. The value of incx must not be zero.
     * @param m -  specifies the number of rows of the matrices
     * @param n - specifies the number of columns of the matrices
     */
    __global__ void mp_mat2vec_right_scal_esi_kernel(mp_array_t R, const int ldr, mp_array_t A, const int lda, mp_array_t x, const int incx, const int m, const int n) {
        unsigned int lenx = x.len[0];
        unsigned int lena = A.len[0];
        unsigned int lenr = R.len[0];
        unsigned int colId = blockIdx.y; // The column index
        //Iterate over matrix columns / vector elements
        while (colId < n){
            int ida = colId * lda; // The firs element of the corresponding column in the matrix A
            int idr = colId * ldr; // The firs element of the corresponding column in the matrix R
            int idx = incx > 0 ? colId * incx : (-n + colId + 1)*incx;
            //Load the corresponding vector element into the registers if possible
            int x_sign = x.sign[idx];
            int x_exp = x.exp[idx];
            er_float_t x_ev0 = x.eval[idx];
            er_float_t x_ev1 = x.eval[idx + lenx];
            //We process in the stride loop all the elements of the i-th column of A
            int ide = blockDim.x * blockIdx.x + threadIdx.x; //Index of the element of A in the colId-th column. Must be less than m
            while (ide < m) {
                R.sign[idr + ide] = A.sign[ida + ide] ^ x_sign;
                R.exp[idr + ide] = A.exp[ida + ide] + x_exp;
                cuda::er_md_rd(&R.eval[idr + ide], &A.eval[ida + ide],  &x_ev0, &cuda::RNS_EVAL_UNIT.upp);
                cuda::er_md_ru(&R.eval[lenr + idr + ide], &A.eval[lena + ida + ide], &x_ev1, &cuda::RNS_EVAL_UNIT.low);
                //Go to the next iteration
                ide += gridDim.x * blockDim.x;
            }
            //Go to the next column
            colId += gridDim.y;
        }
    }

    /*!
     * Multiplication of a general m-by-n matrix A by a diagonal n-by-n matrix on the right which is stored as a vector x of size n
     * Each i-th column of the matrix A is multiplied by i-th element of the vector x
     * The result is written into the matrix R of size m by n
     * Kernel #2 --- Computing the significands in the RNS (digits)
     * @note All matrices are assumed to be stored in the column major order, that is, [column 1] [column 2] ... [column n]
     * @note This kernel can be run on a 2D grid of 1D blocks. Each line in the grid (i.e., all blocks with the same y coordinate) is associated with its own column of the matrix.
     * @param R - pointer to the result array, size ldr * n. After calculations, the leading m-by-n part of the array contains the matrix R
     * @param ldr - specifies the leading dimension of R as declared in the calling (sub)program. The value of ldr must be at least max(1, m)
     * @param A - pointer to the array, size lda * n. Before entry, the leading m-by-n part of the array must contain the matrix A.
     * @param lda - specifies the leading dimension of A as declared in the calling (sub)program. The value of lda must be at least max(1, m).
     * @param x - pointer to the vector in the global GPU memory, size at least (1+(n-1)*abs(incx)).
     * @param incx - storage spacing between elements of x. The value of incx must not be zero.
     * @param m -  specifies the number of rows of the matrices
     * @param n - specifies the number of columns of the matrices
     */
    __global__ static void mp_mat2vec_right_scal_digits_kernel(mp_array_t R, int ldr, mp_array_t A, int lda, mp_array_t x, const int incx, const int m, const int n) {
        int lmodul = cuda::RNS_MODULI[threadIdx.x % RNS_MODULI_SIZE];
        int colId = blockIdx.y; // The column index
        while (colId < n) {
            int ix = incx > 0 ? (colId * RNS_MODULI_SIZE * incx + threadIdx.x % RNS_MODULI_SIZE) : ((-n + colId + 1) * RNS_MODULI_SIZE * incx + threadIdx.x % RNS_MODULI_SIZE);
            int lx = x.digits[ix];
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            while (index < m * RNS_MODULI_SIZE) {
                R.digits[colId * ldr * RNS_MODULI_SIZE + index] = cuda::mod_mul(A.digits[colId * lda * RNS_MODULI_SIZE + index], lx, lmodul);
                index += gridDim.x * blockDim.x;
            }
            colId += gridDim.y;
        }
    }
    
    /*!
     * Multiplication of a general m-by-n matrix A by a diagonal m-by-m matrix on the left which is stored as a vector x of size m
     * Each i-th column of the matrix A is multiplied by the vector x
     * The result is written into the matrix R of size m by n
     * Kernel #1 --- Computing the exponents, signs, and interval evaluations (e-s-i)
     * @note All matrices are assumed to be stored in the column major order, that is, [column 1] [column 2] ... [column n]
     * @note This kernel can be run on a 2D grid of 1D blocks. Each line in the grid (i.e., all blocks with the same y coordinate) is associated with its own column of the matrix.
     * @param R - pointer to the result array, size ldr * n. After calculations, the leading m-by-n part of the array contains the matrix R
     * @param ldr - specifies the leading dimension of R as declared in the calling (sub)program. The value of ldr must be at least max(1, m)
     * @param A - pointer to the array, size lda * n. Before entry, the leading m-by-n part of the array must contain the matrix A.
     * @param lda - specifies the leading dimension of A as declared in the calling (sub)program. The value of lda must be at least max(1, m).
     * @param x - pointer to the vector in the global GPU memory, size at least (1+(m-1)*abs(incx)).
     * @param incx - storage spacing between elements of x. The value of incx must not be zero.
     * @param m -  specifies the number of rows of the matrices
     * @param n - specifies the number of columns of the matrices
     */
    __global__ static void mp_mat2vec_left_scal_esi_kernel(mp_array_t R, int ldr, mp_array_t A, int lda, mp_array_t x, const int incx, const int m, const int n) {
        int lenx = x.len[0];
        int lena = A.len[0];
        int lenr = R.len[0];
        int colId = blockIdx.y;
        //Iterate over matrix columns
        while (colId < n){
            int ida = colId * lda; // The firs element of the corresponding column in the matrix A
            int idr = colId * ldr; // The firs element of the corresponding column in the matrix R
            int numberIdx = blockDim.x * blockIdx.x + threadIdx.x; //Index of the element of A in the colId-th column. Must be less than m
            int ix = incx > 0 ? numberIdx * incx : (-m + numberIdx + 1)*incx;
            //We process in the stride loop all the elements of the i-th column of A
            while (numberIdx < m) {
                R.sign[idr + numberIdx] = A.sign[ida + numberIdx] ^ x.sign[ix];
                R.exp[idr + numberIdx] = A.exp[ida + numberIdx] + x.exp[ix];
                cuda::er_md_rd(&R.eval[idr + numberIdx], &A.eval[ida + numberIdx], &x.eval[ix], &cuda::RNS_EVAL_UNIT.upp);
                cuda::er_md_ru(&R.eval[lenr + idr + numberIdx], &A.eval[lena + ida + numberIdx], &x.eval[lenx + ix], &cuda::RNS_EVAL_UNIT.low);
                numberIdx += gridDim.x * blockDim.x;
                ix += gridDim.x * blockDim.x * incx;
            }
            //Go to the next column
            colId += gridDim.y;
        }
    }

    /*!
     * Multiplication of a general m-by-n matrix A by a diagonal m-by-m matrix on the left which is stored as a vector x of size m
     * Each i-th column of the matrix A is multiplied by the vector x
     * The result is written into the matrix R of size m by n
     * Kernel #2 --- Computing the significands in the RNS (digits)
     * @note All matrices are assumed to be stored in the column major order, that is, [column 1] [column 2] ... [column n]
     * @note This kernel can be run on a 2D grid of 1D blocks. Each line in the grid (i.e., all blocks with the same y coordinate)
     * is associated with its own column of the matrix.
     * @note For this kernel, the block size is specified by either BLOCK_SIZE_FOR_RESIDUES (see kernel_config.cuh)
     * or RNS_MODULI_SIZE as declared in the calling subprogram
     * @param R - pointer to the result array, size ldr * n. After calculations, the leading m-by-n part of the array contains the matrix R
     * @param ldr - specifies the leading dimension of R as declared in the calling (sub)program. The value of ldr must be at least max(1, m)
     * @param A - pointer to the array, size lda * n. Before entry, the leading m-by-n part of the array must contain the matrix A.
     * @param lda - specifies the leading dimension of A as declared in the calling (sub)program. The value of lda must be at least max(1, m).
     * @param x - pointer to the vector in the global GPU memory, size at least (1+(m-1)*abs(incx)).
     * @param incx - storage spacing between elements of x. The value of incx must not be zero.
     * @param m -  specifies the number of rows of the matrices
     * @param n - specifies the number of columns of the matrices
     */
    __global__ static void mp_mat2vec_left_scal_digits_kernel(mp_array_t R, int ldr, mp_array_t A, int lda, mp_array_t x, const int incx, const int m, const int n) {
        int lmodul = cuda::RNS_MODULI[threadIdx.x % RNS_MODULI_SIZE];
        int colId = blockIdx.y;
        while (colId < n){
            int index = blockIdx.x * blockDim.x + threadIdx.x;  //Index of the element of A in the i-th column. Must be less than m * RNS_MODULI_SIZE
            int ix = incx > 0 ? (blockIdx.x * blockDim.x * incx + threadIdx.x) : ((-m + blockIdx.x + 1) * blockDim.x * incx + threadIdx.x);
            //We process in the stride loop all the elements of the i-th column of A
            while (index < m * RNS_MODULI_SIZE) {
                R.digits[colId * ldr * RNS_MODULI_SIZE + index] = cuda::mod_mul(A.digits[colId * lda * RNS_MODULI_SIZE + index], x.digits[ix], lmodul);
                //Go to the next iteration
                index += gridDim.x * blockDim.x;
                ix += gridDim.x * blockDim.x * incx;
            }
            //Go to the next column
            colId += gridDim.y;
        }
    }


    /*!
     * Rounding a multiple-precision matrix
     * @note The matrix A is assumed to be stored in the column major order, that is, [column 1] [column 2] ... [column n]
     * @note For this kernel, the execution configuration should be as follows:
        * dim3 block(blockSizeX, blockSizeY);
        * dim3 grid((m + block.x - 1) / block.x, (n + block.y - 1) / block.y);
     * @param A - pointer to the array, size lda * n. Before entry, the leading m-by-n part of the array must contain the matrix A to be rounded
     * @param lda - specifies the leading dimension of A as declared in the calling (sub)program. The value of lda must be at least max(1, m)
     * @param m - specifies the number of rows of the matrix A
     * @param n - specifies the number of columns of the matrix A
     */
    __global__ void mp_matrix_round_kernel(mp_array_t A, const int lda, const int m, const int n) {
        unsigned int lena = A.len[0]; // Actual matrix length
        unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;
        unsigned int ida = iy * lda + ix;
        if (ix < m && iy < n){
            #if defined(DEBUG) || defined(_DEBUG)
            if( A.eval[ida + lena].exp != A.eval[ida].exp ){
                    printf("\n [CUDA WARNING] Possible loss of accuracy");
                }
            #endif
            int bits = (A.eval[ida + lena].exp - cuda::MP_H + 1) * (A.eval[ida + lena].frac != 0);
            while (bits > 0) {
                A.exp[ida] += bits;
                cuda::rns_scale2pow(&A.digits[ida * RNS_MODULI_SIZE], &A.digits[ida * RNS_MODULI_SIZE], bits);
                cuda::rns_eval_compute_fast(&A.eval[ida], &A.eval[ida + lena], &A.digits[ida * RNS_MODULI_SIZE]);
                bits = -1;
            }
        }
    }

} //end of namespace

#endif //MPRES_MPMATRIX_CUH
