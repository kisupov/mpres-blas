/*
 *  Multiple-precision GE_NORM function for GPU (BLAS Level-3)
 *  Norm of a matrix
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


#ifndef MPRES_GENORM_CUH
#define MPRES_GENORM_CUH

#include "norm.cuh"
#include "mblas_enum.cuh"

namespace cuda {

    /*
     * Kernel that calculates the sum of the absolute values of all the elements in each row of a general  m-by-n  matrix
     * The result is a vector 'result' of size m
     * @param A - matrix of m rows and n columns
     * @param lda - specifies the leading dimension of A as declared in the calling (sub)program. The value of lda must be at least max(1, m).
     * @param nextPow2 - least power of two greater than or equal to blockDim.x
     */
    __global__ static void matrix_row_sum_abs_kernel(const unsigned int m, const unsigned int n, mp_array_t A, const int lda, mp_array_t result, const unsigned int nextPow2) {
        extern __shared__ mp_float_t sdata[];

        // parameters
        const unsigned int tid = threadIdx.x;
        const unsigned int bid = blockIdx.x;
        const unsigned int bsize = blockDim.x;
        unsigned int i = threadIdx.x;

        // do reduction in global mem
        sdata[tid] = cuda::MP_ZERO;
        while (i < n) {
            cuda::mp_add_abs(&sdata[tid], &sdata[tid], A, i * lda + bid);
            i += bsize;
        }
        __syncthreads();

        // do reduction in shared mem
        i = nextPow2 >> 1; // half of nextPow2
        while(i >= 1){
            if ((tid < i) && (tid + i < bsize)) {
                cuda::mp_add_abs(&sdata[tid], &sdata[tid], &sdata[tid + i]);
            }
            i = i >> 1;
            __syncthreads();
        }

        // write result for this block to global mem
        if (tid == 0) {
            result.sign[bid] = 0;
            result.exp[bid] = sdata[tid].exp;
            result.eval[bid] = sdata[tid].eval[0];
            result.eval[bid + result.len[0]] = sdata[tid].eval[1];
            for(int j = 0; j < RNS_MODULI_SIZE; j++){
                result.digits[RNS_MODULI_SIZE * bid + j] = sdata[tid].digits[j];
            }
        }
    }

    /*
     * Kernel that calculates the sum of the absolute values of all the elements in each column of a general  m-by-n  matrix
     * The result is a vector 'result' of size n
     * @param A - matrix of m rows and n columns
     * @param lda - specifies the leading dimension of A as declared in the calling (sub)program. The value of lda must be at least max(1, m).
     * @param nextPow2 - least power of two greater than or equal to blockDim.x
     */
    __global__ static void matrix_col_sum_abs_kernel(const unsigned int m, const unsigned int n, mp_array_t A, const int lda, mp_array_t result, const unsigned int nextPow2) {
        extern __shared__ mp_float_t sdata[];

        // parameters
        const unsigned int tid = threadIdx.x;
        const unsigned int bid = blockIdx.x;
        const unsigned int bsize = blockDim.x;
        unsigned int i = tid;

        // do reduction in global mem
        sdata[tid] = cuda::MP_ZERO;
        while (i < m) {
            cuda::mp_add_abs(&sdata[tid], &sdata[tid], A, bid * lda + i);
            i += bsize;
        }
        __syncthreads();

        // do reduction in shared mem
        i = nextPow2 >> 1; // half of nextPow2
        while(i >= 1){
            if ((tid < i) && (tid + i < bsize)) {
                cuda::mp_add_abs(&sdata[tid], &sdata[tid], &sdata[tid + i]);
            }
            i = i >> 1;
            __syncthreads();
        }

        // write result for this block to global mem
        if (tid == 0) {
            result.sign[bid] = 0;
            result.exp[bid] = sdata[tid].exp;
            result.eval[bid] = sdata[tid].eval[0];
            result.eval[bid + result.len[0]] = sdata[tid].eval[1];
            for(int j = 0; j < RNS_MODULI_SIZE; j++){
                result.digits[RNS_MODULI_SIZE * bid + j] = sdata[tid].digits[j];
            }
        }
    }



    /*!
     * Computes the norm of a general matrix A depending on the value passed as the norm operator argument.
     * The matrix should be stored in column-major order.
     *
     * @tparam gridDim1 - number of thread blocks for parallel reduction of a vector
     * @tparam blockDim1 - number of threads per block for parallel reduction of a vector and matrix
     *
     * @param norm - type of norm to be computed
     * @param m - specifies the number of rows of the matrix A. The value of m must be greater than zero.
     * @param n - specifies the number of columns of the matrix A. The value of n must be greater than zero.
     * @param A - pointer to the array, size lda * n, in the global GPU memory. Before entry, the leading m-by-n part of the array must contain the matrix A.
     * @param lda - specifies the leading dimension of A as declared in the calling (sub)program. The value of lda must be at least max(1, m).
     * @param r - pointer to the computed norm (result) --- a vector of length one in the GPU memory
     * @param buffer - auxiliary array in the global GPU memory, size at least n for one-norm and at least m for infinity-norm.
     */
    template <int gridDim1, int blockDim1>
    void mp_ge_norm(enum mblas_norm_type norm, const int m, const int n, mp_array_t &A, const int lda, mp_array_t &r, mp_array_t &buffer) {

        //Quick return if possible
        if( (m <= 0) || (n <= 0) ){
            return;
        }
        //Test the input parameters
        if( (lda < MAX(1, m)) ){
            return;
        }

        // Power of two that is greater that or equals to blockDim3
        const unsigned int POW = nextPow2(blockDim1);

        if(norm == mblas_one_norm){ // one-norm (max column sum)

            // Compute column sums
            matrix_col_sum_abs_kernel<<<n, blockDim1, sizeof(mp_float_t) * blockDim1>>>(m, n, A, lda, buffer, POW);

            // Call mp_norm to compute the maximum value of the buffer's elements
            mp_norm<gridDim1, blockDim1> (mblas_inf_norm, n, buffer, 1, r);

        }
        else { // infinity-norm (max row sum)

            // Compute row sums
            matrix_row_sum_abs_kernel<<<m, blockDim1, sizeof(mp_float_t) * blockDim1>>>(m, n, A, lda, buffer, POW);

            // Call mp_norm to compute the maximum value of the buffer's elements
            mp_norm<gridDim1, blockDim1> (mblas_inf_norm, m, buffer, 1, r);
        }
    }

} //end of namespace

#endif //MPRES_GENORM_CUH
