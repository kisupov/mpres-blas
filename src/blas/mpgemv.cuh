/*
 *  Multiple-precision GEMV function for GPU (BLAS Level-2)
 *  Computes a matrix-vector product using a general matrix.
 *
 *  Copyright 2019-2020 by Konstantin Isupov.
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

#ifndef MPGEMV_CUH
#define MPGEMV_CUH


#include "../mpvector.cuh"
#include "../kernel_config.cuh"
#include "../mblas_enum.cuh"

namespace cuda
{

   /*
    * Below are the kernels for element-wise multiplication of a matrix of size m * n by a vector x of size n
    * Each column of the matrix is scaled by one element of the vector
    * The result (R) is an m-by-n matrix in the column-major order
    * Both the input matrix and the result are stored in column-major, i.e. [column 1] [column 2] ... [column n]
    * The lda value specifies the leading dimension of A.
    * The ldr value specifies the leading dimension of R.
    */

     //Computing the signs, exponents, and interval evaluations
     //m is the size of a matrix column
     //n is the size of x and the number of matrix columns
    __global__ static void matvec_product_esi_kernel(mp_array_t R, int ldr, mp_array_t A, int lda, mp_array_t x, const int m, const int n) {
        int lenx = x.len[0]; //actual length of x
        int lena = A.len[0]; //actual length of A
        int lenr = R.len[0]; //actual length of R
        int colId = blockIdx.y; // The column index
         //Iterate over matrix columns / vector elements
         while (colId < n){
            int ida = colId * lda; // The firs element of the corresponding column in the matrix A
            int idr = colId * ldr; // The firs element of the corresponding column in the matrix R
            int numberIdx = blockDim.x * blockIdx.x + threadIdx.x; //Index of the element of A in the colId-th column. Must be less than m
            //Load the corresponding vector element into the registers
            int x_sign = x.sign[colId];
            int x_exp = x.exp[colId];
            er_float_t x_eval0 = x.eval[colId];
            er_float_t x_eval1 = x.eval[colId + lenx];
            er_float_ptr x_eval0ptr = &x_eval0;
            er_float_ptr x_eval1ptr = &x_eval1;
            //We process in the stride loop all the elements of the i-th column of A
            while (numberIdx < m) {
                R.sign[idr + numberIdx] = A.sign[ida + numberIdx] ^ x_sign;
                R.exp[idr + numberIdx] = A.exp[ida + numberIdx] + x_exp;
                cuda::er_md_rd(&R.eval[idr + numberIdx], &A.eval[ida + numberIdx], x_eval0ptr, &cuda::RNS_EVAL_UNIT.upp);
                cuda::er_md_ru(&R.eval[lenr + idr + numberIdx], &A.eval[lena + ida + numberIdx], x_eval1ptr, &cuda::RNS_EVAL_UNIT.low);
                //Go to the next iteration
                numberIdx += gridDim.x * blockDim.x;
            }
            //Go to the next column
            colId += gridDim.y;
        }
    }


    //Multiplying the digits in the RNS
    //m is the size of a matrix column
    //n is the size of x and the number of matrix columns
    __global__ static void matvec_product_digits_kernel(mp_array_t R, int ldr, mp_array_t A, int lda, mp_array_t x, const int m, const int n) {
        int lmodul = cuda::RNS_MODULI[threadIdx.x % RNS_MODULI_SIZE];
        int colId = blockIdx.y; // The column index
        while (colId < n){
            //Load the digit of the current vector element into the registers
            int lx = x.digits[colId * RNS_MODULI_SIZE + threadIdx.x % RNS_MODULI_SIZE];
            int index = blockIdx.x * blockDim.x + threadIdx.x; //Index of the element of A in the i-th column. Must be less than m * RNS_MODULI_SIZE
            //We process in the stride loop all the elements of the i-th column of A
            while (index < m * RNS_MODULI_SIZE) {
                R.digits[colId * ldr * RNS_MODULI_SIZE + index] = cuda::mod_mul(A.digits[colId * lda * RNS_MODULI_SIZE + index], lx, lmodul);
                //Go to the next iteration
                index += gridDim.x * blockDim.x;
            }
            //Go to the next column
            colId += gridDim.y;
        }
    }

    /*
     * Below are the kernels for element-wise multiplication of a TRANSPOSED matrix of size m * n by a vector x of size m
     * Each column of the matrix (of size m) is multiplied by the vector x
     * The result (R) is an m-by-n matrix in the column-major order
     * Both the input matrix and the result are stored in column-major, i.e. [column 1] [column 2] ... [column n]
     * The lda value specifies the leading dimension of A.
     * The ldr value specifies the leading dimension of R.
     */

    //Computing the signs, exponents, and interval evaluations
    //m is the size of a matrix column and the vector x
    //n is the number of matrix columns
    __global__ static void trans_matvec_product_esi_kernel(mp_array_t R, int ldr, mp_array_t A, int lda, mp_array_t x, const int m, const int n) {
        int lenx = x.len[0]; //actual length of x
        int lena = A.len[0]; //actual length of A
        int lenr = R.len[0]; //actual length of R
        int colId = blockIdx.y; // The column index
        //Iterate over matrix columns
        while (colId < n){
            int ida = colId * lda; // The firs element of the corresponding column in the matrix A
            int idr = colId * ldr; // The firs element of the corresponding column in the matrix R
            int numberIdx = blockDim.x * blockIdx.x + threadIdx.x; //Index of the element of A in the colId-th column. Must be less than m
            //We process in the stride loop all the elements of the i-th column of A
            while (numberIdx < m) {
                R.sign[idr + numberIdx] = A.sign[ida + numberIdx] ^ x.sign[numberIdx];
                R.exp[idr + numberIdx] = A.exp[ida + numberIdx] + x.exp[numberIdx];
                cuda::er_md_rd(&R.eval[idr + numberIdx], &A.eval[ida + numberIdx], &x.eval[numberIdx], &cuda::RNS_EVAL_UNIT.upp);
                cuda::er_md_ru(&R.eval[lenr + idr + numberIdx], &A.eval[lena + ida + numberIdx], &x.eval[lenx + numberIdx], &cuda::RNS_EVAL_UNIT.low);
                //Go to the next iteration
                numberIdx += gridDim.x * blockDim.x;
            }
            //Go to the next column
            colId += gridDim.y;
        }
    }

    //Multiplying the digits in the RNS
    //m is the size of a matrix column and the vector x
    //n is the number of matrix columns
    __global__ static void trans_matvec_product_digits_kernel(mp_array_t R, int ldr, mp_array_t A, int lda, mp_array_t x, const int m, const int n) {
        int lmodul = cuda::RNS_MODULI[threadIdx.x % RNS_MODULI_SIZE];
        int colId = blockIdx.y; // The column index
        while (colId < n){
            int index = blockIdx.x * blockDim.x + threadIdx.x;  //Index of the element of A in the i-th column. Must be less than m * RNS_MODULI_SIZE
            //We process in the stride loop all the elements of the i-th column of A
            while (index < m * RNS_MODULI_SIZE) {
                R.digits[colId * ldr * RNS_MODULI_SIZE + index] = cuda::mod_mul(A.digits[colId * lda * RNS_MODULI_SIZE + index], x.digits[index], lmodul);
                //Go to the next iteration
                index += gridDim.x * blockDim.x;
            }
            //Go to the next column
            colId += gridDim.y;
        }
    }


    /*
     * Kernel that calculates the sum of all the elements in each row of an m-by-n multiple-precision matrix
     * The result (a vector of size m) is then added to the vector y
     * @param A - matrix of m rows and n columns
     * @param y - vector of size m
     * @param incy - storage spacing between elements of y
     * @param nextPow2 - least power of two greater than or equal to blockDim.x
     */
    __global__ void matrix_reduction_kernel(const unsigned int m, const unsigned int n, mp_array_t A, mp_array_t y, int incy, const unsigned int nextPow2) {
        extern __shared__ mp_float_t sdata[];

        // parameters
        const unsigned int tid = threadIdx.x;
        const unsigned int bid = blockIdx.x;
        const unsigned int bsize = blockDim.x;
        unsigned int i = threadIdx.x;

        // do reduction in global mem
        sdata[tid] = cuda::MP_ZERO;
        while (i < n) {
            cuda::mp_add(&sdata[tid], &sdata[tid], A, i * m + bid);
            i += bsize;
        }
        __syncthreads();

        // do reduction in shared mem
        i = nextPow2 >> 1; // half of nextPow2
        while(i >= 1){
            if ((tid < i) && (tid + i < bsize)) {
                cuda::mp_add(&sdata[tid], &sdata[tid], &sdata[tid + i]);
            }
            i = i >> 1;
            __syncthreads();
        }

        // write result for this block to global mem
        if (tid == 0) {
            int iy = incy > 0 ? bid * incy : (-m + bid + 1)*incy;
            cuda::mp_add(y, iy, y, iy, &sdata[tid]);
        }
        __syncthreads();
    }

    /*
     * Kernel that calculates the sum of all the elements in each columnt of an m-by-n multiple-precision matrix
     * The result (a vector of size n) is then added to the vector y
     * @param A - matrix of m rows and n columns
     * @param y - vector of size n
     * @param incy - storage spacing between elements of y
     * @param nextPow2 - least power of two greater than or equal to blockDim.x
     */
    __global__ static void trans_matrix_reduction_kernel(const unsigned int m, const unsigned int n, mp_array_t input, mp_array_t y, int incy, const unsigned int nextPow2) {
        extern __shared__ mp_float_t sdata[];

        // parameters
        const unsigned int tid = threadIdx.x;
        const unsigned int bid = blockIdx.x;
        const unsigned int bsize = blockDim.x;
        unsigned int i = tid;

        // do reduction in global mem
        sdata[tid] = cuda::MP_ZERO;
        while (i < m) {
            cuda::mp_add(&sdata[tid], &sdata[tid], input, bid * m + i);
            i += bsize;
        }
        __syncthreads();

        // do reduction in shared mem
        i = nextPow2 >> 1; // half of nextPow2
        while(i >= 1){
            if ((tid < i) && (tid + i < bsize)) {
                cuda::mp_add(&sdata[tid], &sdata[tid], &sdata[tid + i]);
            }
            i = i >> 1;
            __syncthreads();
        }

        // write result for this block to global mem
        if (tid == 0) {
            int iy = incy > 0 ? bid * incy : (-n + bid + 1)*incy;
            cuda::mp_add(y, iy, y, iy, &sdata[tid]);
        }
        __syncthreads();
    }

    /*!
     * Performs one of the matrix-vector operations
     * y = alpha*A*x + beta*y  or
     * y = alpha*A**T*x + beta*y,
     * where alpha and beta are scalars, x and y are vectors and A is an m-by-n matrix.
     * The matrix should be stored in column-major order.

     * @tparam gridDim1 - number of blocks (x dimension) used to compute the signs, exponents, interval evaluations, and also to round the result in element-wise scalar-vector and matrix-vector operations
     * @tparam blockDim1 - number of threads per block used to compute the signs, exponents, interval evaluations, and also to round the result in element-wise scalar-vector and matrix-vector operations
     * @tparam gridDim2 - number of blocks (x dimension) used to compute the digits of multiple-precision significands in element-wise scalar-vector and matrix-vector operations
     * @tparam blockDim3 - number of threads per block for parallel summation (the number of blocks is equal to the size of y)
     *
     * @param trans - specifies the operation:
     * if trans = 'N' or 'n', then y := alpha*A*x + beta*y.
     * if trans = 'T' or 't' or 'C' or 'c' then y = alpha*A**T*x + beta*y (transposed matrix).
     * @param m - specifies the number of rows of the matrix A. The value of m must be at least zero.
     * @param n - specifies the number of columns of the matrix A. The value of n must be at least zero.
     * @param alpha - pointer to the scalar in the global GPU memory
     * @param A - pointer to the array, size lda * n, in the global GPU memory. Before entry, the leading m-by-n part of the array must contain the matrix A.
     * @param lda - specifies the leading dimension of A as declared in the calling (sub)program. The value of lda must be at least max(1, m).
     * @param x - pointer to the vector in the global GPU memory, size at least (1+(n-1)*abs(incx)) for non-transposed matrix and at least (1+(m-1)*abs(incx)) otherwise.
     * @param incx - storage spacing between elements of x. The value of incx must not be zero.
     * @param beta - pointer to the scalar in the global GPU memory
     * @param y - pointer to the vector in the global GPU memory, size at least (1+(m-1)*abs(incy)) for non-transposed matrix and at least (1+(n-1)*abs(incy)) otherwise.
     * @param incy - storage spacing between elements of y. The value of incy must not be zero.
     * @param buffer1 - auxiliary array in the global GPU memory, size at least n for non-transposed matrix and at least m otherwise.
     * @param buffer2 - auxiliary array, size m * n, in the global GPU memory for storing the intermediate matrix
     */
    template<int gridDim1, int blockDim1, int gridDim2, int blockDim3>
    void mpgemv(enum mblas_trans_type trans, const int m, const int n, mp_array_t &alpha, mp_array_t &A, const int lda,
            mp_array_t &x, const int incx, mp_array_t &beta, mp_array_t &y, const int incy, mp_array_t &buffer1, mp_array_t &buffer2){

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

        // Setting the number of threads per block for computing residues
        int numThreadsX = (incx == 1) ? BLOCK_SIZE_FOR_RESIDUES : RNS_MODULI_SIZE;
        int numThreadsY = (incy == 1) ? BLOCK_SIZE_FOR_RESIDUES : RNS_MODULI_SIZE;

        if(trans == mblas_no_trans){

            //Multiplication buffer1 = alpha * x - Computing the signs, exponents, and interval evaluations
            mp_vec2scal_mul_esi_kernel<<< gridDim1, blockDim1 >>> (buffer1, 1, x, incx, alpha, n);

            //Multiplication buffer1 = alpha * x - Multiplying the digits in the RNS
            mp_vec2scal_mul_digits_kernel<<< gridDim2, numThreadsX >>> (buffer1, 1, x, incx, alpha, n);

            //Rounding the intermediate result (buffer1)
            mp_vector_round<<< gridDim1, blockDim1 >>> (buffer1, 1, n);

            //Multiplication y = beta * y - Computing the signs, exponents, and interval evaluations
            mp_vec2scal_mul_esi_kernel<<< gridDim1, blockDim1 >>> (y, incy, y, incy, beta, m);

            //Multiplication y = beta * y - Multiplying the digits in the RNS
            mp_vec2scal_mul_digits_kernel<<< gridDim2, numThreadsY >>> (y, incy, y, incy, beta, m);

            //Rounding y
            mp_vector_round<<< gridDim1, blockDim1 >>> (y, incy, m);

            //The following is tne element-wise multiplication of an m-by-n matrix A by a vector buffer1 (contains alpha * x) of size n.
            //Each column of the matrix is multiplied by one element of the vector (without reduction).
            //We run a 2D grid of 1D blocks.
            //Each line in the grid (i.e., all blocks with the same y coordinate) is associated with its own column of the matrix.
            //The result is written to the intermediate m-by-n buffer2.

            //Multiplication buffer2 = A * buffer1 - Computing the signs, exponents, and interval evaluations
            dim3 blocks1(gridDim1, n, 1);
            cuda::matvec_product_esi_kernel<<<blocks1, blockDim1>>> (buffer2, m, A, lda, buffer1, m, n);

            //Multiplication buffer2 = A * buffer1 - Multiplying the digits in the RNS
            //cuda::matvec_product_digits_kernel<<<gridDim2, BLOCK_SIZE_FOR_RESIDUES>>> (buffer2, A, lda, buffer1, m, n);
            dim3 blocks2(gridDim2, n, 1);
            cuda::matvec_product_digits_kernel<<<blocks2, BLOCK_SIZE_FOR_RESIDUES>>> (buffer2, m, A, lda, buffer1, m, n);

            //Rounding the intermediate result (buffer2)
            mp_vector_round<<<gridDim1, blockDim1>>>(buffer2, 1, m * n);

            //The following is tne reduction of the intermediate matrix (buffer 2).
            //Here, the sum of the elements in each row is calculated, and then y is added to the calculated sum
            //The result is a vector of size m

            // Kernel memory configurations. We prefer shared memory
            //cudaFuncSetCacheConfig(matrix_reduction_kernel, cudaFuncCachePreferShared);

            // Power of two that is greater that or equals to blockDim3
            const unsigned int POW = nextPow2(blockDim3);

            // Call to the internal reduction kernel
            cuda::matrix_reduction_kernel<<<m, blockDim3, sizeof(mp_float_t) * blockDim3>>>(m, n, buffer2, y, incy, POW);

        } else {

            //Multiplication buffer1 = alpha * x - Computing the signs, exponents, and interval evaluations
            mp_vec2scal_mul_esi_kernel<<< gridDim1, blockDim1 >>> (buffer1, 1, x, incx, alpha, m);

            //Multiplication buffer1 = alpha * x - Multiplying the digits in the RNS
            mp_vec2scal_mul_digits_kernel<<< gridDim2, numThreadsX >>> (buffer1, 1, x, incx, alpha, m);

            //Rounding the intermediate result (buffer1)
            mp_vector_round<<< gridDim1, blockDim1 >>> (buffer1, 1, m);

            //Multiplication y = beta * y - Computing the signs, exponents, and interval evaluations
            mp_vec2scal_mul_esi_kernel<<< gridDim1, blockDim1 >>> (y, incy, y, incy, beta, n);

            //Multiplication y = beta * y - Multiplying the digits in the RNS
            mp_vec2scal_mul_digits_kernel<<< gridDim2, numThreadsY >>> (y, incy, y, incy, beta, n);

            //Rounding y
            mp_vector_round<<< gridDim1, blockDim1 >>> (y, incy, n);

            //The following is tne element-wise multiplication of an m-by-n transposed matrix A by a vector buffer1 (contains alpha * x) of size m.
            //Each column of the matrix is multiplied by the vector (without reduction).
            //We run a 2D grid of 1D blocks.
            //Each line in the grid (i.e., all blocks with the same y coordinate) is associated with its own column of the matrix.
            //The result is written to the intermediate m-by-n buffer2.

            //Multiplication buffer2 = A^T * buffer1 - Computing the signs, exponents, and interval evaluations
            dim3 blocks1(gridDim1, n, 1);
            cuda::trans_matvec_product_esi_kernel<<<blocks1, blockDim1>>> (buffer2, m, A, lda, buffer1, m, n);

            //Multiplication buffer2 = A^T * buffer1 - Multiplying the digits in the RNS
            dim3 blocks2(gridDim2, n, 1);
            cuda::trans_matvec_product_digits_kernel<<<blocks2, BLOCK_SIZE_FOR_RESIDUES>>> (buffer2, m, A, lda, buffer1, m, n);

            //Rounding the intermediate result (buffer2)
            mp_vector_round<<<gridDim1, blockDim1>>>(buffer2, 1, m * n);

            //The following is tne reduction of the intermediate matrix (buffer 2).
            //Here, the sum of the elements in each column is calculated, and then y is added to the calculated sum
            //The result is a vector of size n

            // Kernel memory configurations. We prefer shared memory
            //cudaFuncSetCacheConfig(trans_matrix_reduction_kernel, cudaFuncCachePreferShared);

            // Power of two that is greater that or equals to blockDim3
            const unsigned int POW = nextPow2(blockDim3);

            // Call to the internal reduction kernel
            cuda::trans_matrix_reduction_kernel<<<n, blockDim3, sizeof(mp_float_t) * blockDim3>>>(m, n, buffer2, y, incy, POW);
        }
    }

} // namespace cuda

#endif //MPGEMV_CUH
