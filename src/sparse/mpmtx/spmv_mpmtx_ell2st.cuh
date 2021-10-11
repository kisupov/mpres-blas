/*
 *  Multiple-precision SpMV (Sparse matrix-vector multiplication) on GPU using the ELLPACK sparse matrix format (mutiple precision matrix, multiple precision vectors)
 *  Two-stage ELLPACK implementation - first, the array of component-wise products is calculated and then its segment reduction is performed.

 *  Copyright 2020 by Konstantin Isupov and Ivan Babeshko
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

#ifndef MPRES_SPMV_MPMTX_ELL2ST_CUH
#define MPRES_SPMV_MPMTX_ELL2ST_CUH

#include "../../arith/add.cuh"
#include "../../arith/assign.cuh"
#include "../../mpvector.cuh"
#include "../../kernel_config.cuh"

namespace cuda {

    /*!
     * Multiplication of a sparse matrix A stored in the ELLPACK format by a diagonal matrix on the right which is stored as a vector x
     * Kernel #1 --- Computing the exponents, signs, and interval evaluations (e-s-i)
     * The component wise products are written into the result array of size m * maxnzr
     * @note The 'as' and 'result' arrays are treated as vectors (one-dimensional arrays)
     */
    __global__ void ell_scal_esi_kernel(const int m, const int n, const int maxnzr, const int *ja, mp_collection_t as, mp_array_t x, mp_collection_t result) {
        auto numberIdx =  blockDim.x * blockIdx.x + threadIdx.x;
        auto len = maxnzr * m;
        while (numberIdx < len) {
            auto index = ja[numberIdx];
            if(index >= 0) {
                result.sign[numberIdx] = as.sign[numberIdx] ^ x.sign[index];
                result.exp[numberIdx] = as.exp[numberIdx] + x.exp[index];
                result.eval[numberIdx] = cuda::er_md_rd(as.eval[numberIdx], x.eval[index], cuda::RNS_EVAL_UNIT.upp);
                result.eval[len + numberIdx] = cuda::er_md_ru(as.eval[len + numberIdx], x.eval[n + index], cuda::RNS_EVAL_UNIT.low);
            }
            numberIdx +=  gridDim.x * blockDim.x;
        }
    }

    /*!
     * Multiplication of a sparse matrix A stored in the ELLPACK format by a diagonal matrix on the right which is stored as a vector x
     * Kernel #2 --- Computing the significands in the RNS (digits) in parallel
     * The component wise products are written into the result array of size m * maxnzr
     * @note The 'as' and 'result' arrays are treated as vectors (one-dimensional arrays)
     */
    __global__ void ell_scal_digits_kernel(const int m, const int maxnzr, const int *ja, mp_collection_t as, mp_array_t x, mp_collection_t result){
        auto lmodul = cuda::RNS_MODULI[threadIdx.x  % RNS_MODULI_SIZE];
        auto digitId = blockIdx.x * blockDim.x + threadIdx.x; //Index of the current digit
        auto numberId = (blockIdx.x * blockDim.x + threadIdx.x) / RNS_MODULI_SIZE; //Index of the current matrix element
        while (digitId < m * maxnzr * RNS_MODULI_SIZE) {
            auto index = ja[numberId];
            if(index >= 0){
                result.digits[digitId] = cuda::mod_mul(as.digits[digitId], x.digits[index * RNS_MODULI_SIZE + threadIdx.x % RNS_MODULI_SIZE], lmodul);
            }
            //Go to the next iteration
            digitId += gridDim.x * blockDim.x;
            numberId += gridDim.x * blockDim.x / RNS_MODULI_SIZE;
        }
    }

    /*!
     * Kernel that computes the sum of all the elements in each row of a multiple-precision matrix and and stores it in the corresponding element of the vector y of m elements
     * One thread calculates the sum of the elements in one row of the matrix, i.e. one element of the vector y
     * @note Shared memory of size = sizeof(mp_float_t) * nThreads must be allocated, where nThreads is the number of threads per block
     * @note The 'as' array is assumed to be stored in the column major order, that is, [column 1] [column 2] ... [column n]
     */
    __global__ static void ell_matrix_row_sum_kernel(const int m, const int maxnzr, const int *ja, mp_collection_t as, mp_array_t y){
        extern __shared__ mp_float_t sum[];
        auto row = threadIdx.x + blockIdx.x * blockDim.x;
        if (row < m) {
            sum[threadIdx.x] = cuda::MP_ZERO;
            for (auto col = 0; col < maxnzr; col++) {
                auto index = ja[col * m + row];
                if(index >= 0){
                    cuda::mp_add(&sum[threadIdx.x], sum[threadIdx.x], as, col * m + row, m * maxnzr);
                }
            }
            cuda::mp_set(y, row, sum[threadIdx.x]);
        }
    }

    /*!
     * Performs the matrix-vector operation y = A * x, where x and y are dense vectors and A is a sparse matrix.
     * The matrix should be stored in the ELLPACK format: entries are stored in a dense array 'as' in column major order and explicit zeros are stored if necessary (zero padding)
     *
     * @note The matrix is represented in multiple precision
     * @note Multiple-precision floating-point multiplication is split into three separate kernels and for each multiple-precision number, all the digits (residues) are calculated in parallel
     * @note Requires global memory buffer of size m * maxnzr
     *
     * @tparam gridDim1 - number of blocks used to compute the signs, exponents, interval evaluations in an element-wise matrix-vector operation (diagonal scaling)
     * @tparam blockDim1 - number of threads per block used to compute the signs, exponents, interval evaluations in an element-wise matrix-vector operation (diagonal scaling)
     * @tparam gridDim2 - number of blocks used to compute the digits of multiple-precision significands in an element-wise matrix-vector operation (diagonal scaling)
     * @tparam blockDim3 - number of threads per block for parallel summation

     * @param m - number of rows in matrix
     * @param n - number of columns in matrix
     * @param maxnzr - number of nonzeros per row array (maximum number of nonzeros per row in the matrix A)
     * @param ja - column indices array to access the corresponding elements of the vector x, size = m * maxnzr (the same as for A)
     * @param as - multiple-precision coefficients array (entries of the matrix A in the ELLPACK format), size = m * maxnzr
     * @param x - input vector, size at least max(ja) + 1, where max(ja) is the maximum element from the ja array
     * @param y - output vector, size at least m
     * @param buffer - auxiliary global memory array for storing the intermediate matrix, size = m * maxnzr
     */
    template<int gridDim1, int blockDim1, int gridDim2, int blockDim3>
    void mp_spmv_mpmtx_ell2st(const int m, const int n, const int maxnzr, const int *ja, mp_collection_t &as, mp_array_t &x, mp_array_t &y, mp_collection_t &buffer) {

        //We consider the vector x as a diagonal matrix and perform the right diagonal scaling, buffer = A * x.
        //The result is written to the intermediate buffer of size m * maxnzr.

        //Multiplication buffer = A * x - Computing the signs, exponents, and interval evaluations
        ell_scal_esi_kernel<<<gridDim1, blockDim1>>>(m, n, maxnzr, ja, as, x, buffer);

        //Multiplication buffer = A * x - Multiplying the digits in the RNS
        ell_scal_digits_kernel<<< gridDim2, BLOCK_SIZE_FOR_RESIDUES >>> (m, maxnzr, ja, as, x, buffer);

        //Rounding the intermediate result (buffer)
        mp_vector_round_kernel<<< 64, 64 >>> (buffer, m * maxnzr);

        //The following is tne reduction of the intermediate matrix (buffer).
        //Here, the sum of the elements in each row is calculated, and stored in the corresponding element of y
        ell_matrix_row_sum_kernel<<<m / blockDim3 + 1, blockDim3, sizeof(mp_float_t) * blockDim3 >>>(m, maxnzr, ja, buffer, y);
    }

} // namespace cuda

#endif //MPRES_SPMV_MPMTX_ELL2ST_CUH
