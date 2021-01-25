/*
 *  Multiple-precision sparse matrix-vector multiplication (SpMV) on GPU using the CSR sparse matrix format
 *  Fourth SpMV CSR implementation (two-stage approach) - first, the array of component-wise products is calculated and then its segment reduction is performed.
 *
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

#ifndef MPSPMVCSR4_CUH
#define MPSPMVCSR4_CUH

#include "../arith/mpadd.cuh"
#include "../arith/mpassign.cuh"
#include "../mpvector.cuh"
#include "../kernel_config.cuh"

namespace cuda {

    /*!
     * Multiplication of a sparse matrix A stored in the CSR format by a diagonal matrix on the right which is stored as a vector x
     * Kernel #1 --- Computing the exponents, signs, and interval evaluations (e-s-i)
     * The component wise products are written into the result array of size nnz
     * @note The 'as' and 'result' arrays are treated as vectors (one-dimensional arrays)
     */
    __global__ void csr_scal_esi_kernel(const int n, const int nnz, const int *ja, mp_collection_t as, mp_array_t x, mp_collection_t result) {
        auto numberIdx =  blockDim.x * blockIdx.x + threadIdx.x;
        while (numberIdx < nnz) {
            auto index = ja[numberIdx];
            result.sign[numberIdx] = as.sign[numberIdx] ^ x.sign[index];
            result.exp[numberIdx] = as.exp[numberIdx] + x.exp[index];
            cuda::er_md_rd(&result.eval[numberIdx], &as.eval[numberIdx], &x.eval[index], &cuda::RNS_EVAL_UNIT.upp);
            cuda::er_md_ru(&result.eval[nnz + numberIdx], &as.eval[nnz + numberIdx], &x.eval[n + index], &cuda::RNS_EVAL_UNIT.low);
            numberIdx +=  gridDim.x * blockDim.x;
        }
    }

    /*!
     * Multiplication of a sparse matrix A stored in the CSR format by a diagonal matrix on the right which is stored as a vector x
     * Kernel #2 --- Computing the significands in the RNS (digits) in parallel
     * The component wise products are written into the result array of size nnz
     * @note The 'as' and 'result' arrays are treated as vectors (one-dimensional arrays)
     */
    __global__ void csr_scal_digits_kernel(const int nnz, const int *ja, mp_collection_t as, mp_array_t x, mp_collection_t result){
        auto lmodul = cuda::RNS_MODULI[threadIdx.x  % RNS_MODULI_SIZE];
        auto digitId = blockIdx.x * blockDim.x + threadIdx.x; //Index of the current digit
        auto numberId = (blockIdx.x * blockDim.x + threadIdx.x) / RNS_MODULI_SIZE; //Index of the current matrix element
        while (digitId < nnz * RNS_MODULI_SIZE) {
            result.digits[digitId] = cuda::mod_mul(as.digits[digitId], x.digits[ja[numberId] * RNS_MODULI_SIZE + threadIdx.x % RNS_MODULI_SIZE], lmodul);
            digitId += gridDim.x * blockDim.x;
            numberId += gridDim.x * blockDim.x / RNS_MODULI_SIZE;
        }
    }

    /*!
     * Kernel that computes the sum of all the elements in each row of a multiple-precision matrix and and stores it in the corresponding element of the vector y of m elements
     * One thread calculates the sum of the elements in one row of the matrix, i.e. one element of the vector y
     * @note Shared memory of size = sizeof(mp_float_t) * nThreads must be allocated, where nThreads is the number of threads per block
     * @note The 'as' array is assumed to be stored in the row major order, that is, [column 1] [column 2] ... [column n]
     */
    __global__ static void csr_matrix_row_sum_kernel(const int m, const int nnz, const int *irp, mp_collection_t as, mp_array_t y){
        extern __shared__ mp_float_t sum[];
        auto row = threadIdx.x + blockIdx.x * blockDim.x;
        if (row < m) {
            sum[threadIdx.x] = cuda::MP_ZERO;
            for (auto index = irp[row]; index < irp[row + 1]; index++) {
                cuda::mp_add(&sum[threadIdx.x], &sum[threadIdx.x], as, index, nnz);
            }
            cuda::mp_set(y, row, &sum[threadIdx.x]);
        }
    }

    /*!
     * Performs the matrix-vector operation y = A * x, where x and y are dense vectors and A is a sparse matrix.
     * The matrix should be stored in the CSR format: entries are stored in a dense array of nonzeros in row major order
     *
     * @tparam gridDim1 - number of blocks used to compute the signs, exponents, interval evaluations in an element-wise matrix-vector operation (diagonal scaling)
     * @tparam blockDim1 - number of threads per block used to compute the signs, exponents, interval evaluations in an element-wise matrix-vector operation (diagonal scaling)
     * @tparam gridDim2 - number of blocks used to compute the digits of multiple-precision significands in an element-wise matrix-vector operation (diagonal scaling)
     * @tparam blockDim3 - number of threads per block for parallel summation

     * @param m - number of rows in matrix
     * @param n - number of columns in matrix
     * @param nnz - number of nonzeros in matrix
     * @param irp - row start pointers array of size m + 1, last element of irp equals to nnz (number of nonzeros in matrix)
     * @param ja - column indices array to access the corresponding elements of the vector x, size = nnz
     * @param as - multiple-precision coefficients array (entries of the matrix A in the CSR format), size = nnz
     * @param x - input vector, size at least max(ja) + 1, where max(ja) is the maximum element from the ja array
     * @param y - output vector, size at least m
     * @param buffer - auxiliary array in the global GPU memory for storing the intermediate matrix, size = nnz
     */
    template<int gridDim1, int blockDim1, int gridDim2, int blockDim3>
    void mpspmv_csr4(const int m, const int n, const int nnz, const int *irp, const int *ja,  mp_collection_t &as, mp_array_t &x, mp_array_t &y, mp_collection_t &buffer) {

        //We consider the vector x as a diagonal matrix and perform the right diagonal scaling, buffer = A * x.
        //The result is written to the intermediate buffer of size = nnz.

        //Multiplication buffer = A * x - Computing the signs, exponents, and interval evaluations
        csr_scal_esi_kernel<<<gridDim1, blockDim1>>>(n, nnz, ja, as, x, buffer);

        //Multiplication buffer = A * x - Multiplying the digits in the RNS
        csr_scal_digits_kernel<<< gridDim2, BLOCK_SIZE_FOR_RESIDUES >>> (nnz, ja, as, x, buffer);

        //Rounding the intermediate result (buffer)
        mp_vector_round_kernel<<< 64, 64 >>> (buffer, nnz);

        //The following is tne reduction of the intermediate matrix (buffer).
        //Here, the sum of the elements in each row is calculated, and stored in the corresponding element of y
        csr_matrix_row_sum_kernel<<<m / blockDim3 + 1, blockDim3, sizeof(mp_float_t) * blockDim3 >>>(m, nnz, irp, buffer, y);
    }

} // namespace cuda

#endif //MPSPMVCSR4_CUH
