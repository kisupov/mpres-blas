/*
 *  Multiple-precision SpMV (Sparse matrix-vector multiplication) function for GPU using the ELLPACK sparse matrix format
 *  Computes the product of a sparse matrix and a dense vector.
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

#ifndef MPSPMVELL_CUH
#define MPSPMVELL_CUH

#include "../mpvector.cuh"
#include "../kernel_config.cuh"

namespace cuda {

    /*!
     * Multiplication of a sparse matrix A stored in the ELLPACK format (data) by a diagonal matrix on the right which is stored as a vector x
     * The ELLPACK data and indices are treated as vectors (one-dimensional arrays)
     * The component wise products are written into the result array of size num_rows * cols_per_row
     * Kernel #1 --- Computing the exponents, signs, and interval evaluations (e-s-i)
     * @note The data and result arrays are assumed to be stored in the column major order, that is, [column 1] [column 2] ... [column n]
     * @param result - pointer to the multiple-precision result array, size = num_rows * cols_per_row
     * @param data - pointer to the multiple-precision input array that contains matrix A in the ELLPACK format, size = num_rows * cols_per_row
     * @param indices - pointer to the array of column indices that are used to access the corresponding elements of the vector x, size = num_rows * cols_per_row
     * @param x - pointer to the dense vector, size = num_cols
     * @param num_rows - specifies the number of rows in the matrix A and, accordingly, in the data array
     * @param num_cols - specifies the number of columns in the matrix A and length of the vector x
     * @param cols_per_row - specifies the number of columns in the data and result arrays (maximum number of nonzeros per row in A)
     */
    __global__ void ell_scal_esi_kernel(mp_collection_t result, mp_collection_t data, const int * indices, mp_array_t x, const int num_rows, const int num_cols,  const int cols_per_row) {
        auto numberIdx =  blockDim.x * blockIdx.x + threadIdx.x;
        auto len = cols_per_row * num_rows;
        while (numberIdx < len) {
            auto index = indices[numberIdx];
            if(index >= 0) {
                result.sign[numberIdx] = data.sign[numberIdx] ^ x.sign[index];
                result.exp[numberIdx] = data.exp[numberIdx] + x.exp[index];
                cuda::er_md_rd(&result.eval[numberIdx], &data.eval[numberIdx], &x.eval[index], &cuda::RNS_EVAL_UNIT.upp);
                cuda::er_md_ru(&result.eval[len + numberIdx], &data.eval[len + numberIdx], &x.eval[num_cols + index], &cuda::RNS_EVAL_UNIT.low);
            }
            numberIdx +=  gridDim.x * blockDim.x;
        }
    }

    /*!
     * Multiplication of a sparse matrix A stored in the ELLPACK format (data) by a diagonal matrix on the right which is stored as a vector x
     * The ELLPACK data and indices are treated as vectors (one-dimensional arrays)
     * The component wise products are written into the result array of size num_rows * cols_per_row
     * Kernel #2 --- Computing the significands in the RNS (digits)
     * @note The data and result arrays are assumed to be stored in the column major order, that is, [column 1] [column 2] ... [column n]
     * @param result - pointer to the multiple-precision result array, size = num_rows * cols_per_row
     * @param data - pointer to the multiple-precision input array that contains matrix A in the ELLPACK format, size = num_rows * cols_per_row
     * @param indices - pointer to the array of column indices that are used to access the corresponding elements of the vector x, size = num_rows * cols_per_row
     * @param x - pointer to the dense vector, size = num_cols
     * @param num_rows - specifies the number of rows in the matrix A and, accordingly, in the data array
     * @param num_cols - specifies the number of columns in the matrix A and length of the vector x
     * @param cols_per_row - specifies the number of columns in the data and result arrays (maximum number of nonzeros per row in A)
     */
    __global__ void ell_scal_digits_kernel(mp_collection_t result, mp_collection_t data, const int * indices, mp_array_t x, const int num_rows, const int cols_per_row){
        auto lmodul = cuda::RNS_MODULI[threadIdx.x  % RNS_MODULI_SIZE];
        auto digitId = blockIdx.x * blockDim.x + threadIdx.x; //Index of the current digit
        auto numberId = (blockIdx.x * blockDim.x + threadIdx.x) / RNS_MODULI_SIZE; //Index of the current matrix element
        while (digitId < num_rows * cols_per_row * RNS_MODULI_SIZE) {
            auto index = indices[numberId];
            if(index >= 0){
                result.digits[digitId] = cuda::mod_mul(data.digits[digitId], x.digits[index * RNS_MODULI_SIZE + threadIdx.x % RNS_MODULI_SIZE], lmodul);
            }
            //Go to the next iteration
            digitId += gridDim.x * blockDim.x;
            numberId += gridDim.x * blockDim.x / RNS_MODULI_SIZE;
        }
    }

    /*!
     * Kernel that computes the sum of all the elements in each row of a multiple-precision matrix and and stores it in the corresponding element of the vector y
     * One thread calculates the sum of the elements in one row of the matrix, i.e. one element of the vector y
     * @note Shared memory of size = sizeof(mp_float_t) * nThreads must be allocated, where nThreads is the number of threads per block
     * @param y - pointer to the result vector of num_rows elements
     * @param indices - pointer to the array of ELLPACK column indices, size = num_rows * cols_per_row. Addition is performed only if the corresponding index is positive.
     * @param data - pointer to the multiple-precision input array to be summed, size = num_rows * cols_per_row
     * @param num_rows - specifies the number of rows in the data array and the number of elements in the vector y
     * @param cols_per_row - specifies the number of columns in the data array (maximum number of nonzeros per row in A)
     */
    __global__ static void ell_matrix_row_sum_kernel(mp_array_t y, const int * indices, mp_collection_t data, const int num_rows, const int cols_per_row){
        extern __shared__ mp_float_t sum[];
        auto row = threadIdx.x + blockIdx.x * blockDim.x;
        if (row < num_rows) {
            sum[threadIdx.x] = cuda::MP_ZERO;
            for (auto col = 0; col < cols_per_row; col++) {
                auto index = indices[col * num_rows + row];
                if(index >= 0){
                    cuda::mp_add(&sum[threadIdx.x], &sum[threadIdx.x], data, col * num_rows + row, num_rows * cols_per_row);
                }
            }
            cuda::mp_set(y, row, &sum[threadIdx.x]);
        }
    }

    /*!
     * Performs the matrix-vector operation y = A * x, where x and y are dense vectors and A is a sparse matrix.
     * The matrix should be stored in the ELLPACK format: entries are stored in a dense array in column major order and explicit zeros are stored if necessary (zero padding)
     *
     * @tparam gridDim1 - number of blocks used to compute the signs, exponents, interval evaluations in an element-wise matrix-vector operation (diagonal scaling)
     * @tparam blockDim1 - number of threads per block used to compute the signs, exponents, interval evaluations in an element-wise matrix-vector operation (diagonal scaling)
     * @tparam gridDim2 - number of blocks used to compute the digits of multiple-precision significands in an element-wise matrix-vector operation (diagonal scaling)
     * @tparam blockDim3 - number of threads per block for parallel summation

     * @param num_rows - specifies the number of rows of the matrix A. The value of num_rows must be greater than zero.
     * @param cols_per_row - specifies the maximum number of nonzeros per row. The value of cols_per_row must be greater than zero.
     * @param data - pointer to the multiple-precision array, size num_rows * cols_per_row, in the global GPU memory that contains matrix A in the ELLPACK format
     * @param indices - pointer to the array of column indices that are used to access the corresponding elements of the vector x, size num_rows * cols_per_row (the same as for data)
     * @param x - pointer to the dense vector in the global GPU memory, size at least max(indices) + 1, where max(indices) is the maximum element from the indices array
     * @param y - pointer to the result vector in the global GPU memory, size at least num_rows
     * @param buffer - auxiliary array, size num_rows * cols_per_row, in the global GPU memory for storing the intermediate matrix
     */
    template<int gridDim1, int blockDim1, int gridDim2, int blockDim3>
    void mpspmvell(const int num_rows, const int num_cols, const int cols_per_row, const int * indices, mp_collection_t &data, mp_array_t &x, mp_array_t &y, mp_collection_t &buffer) {

        //We consider the vector x as a diagonal matrix and perform the right diagonal scaling, buffer = A * x.
        //The result is written to the intermediate buffer of size num_rows * cols_per_row.

        //Multiplication buffer = A * x - Computing the signs, exponents, and interval evaluations
        ell_scal_esi_kernel<<<gridDim1, blockDim1>>>(buffer, data, indices, x, num_rows, num_cols, cols_per_row);

        //Multiplication buffer = A * x - Multiplying the digits in the RNS
        ell_scal_digits_kernel<<< gridDim2, BLOCK_SIZE_FOR_RESIDUES >>> (buffer, data, indices, x, num_rows, cols_per_row);

        //Rounding the intermediate result (buffer)
        mp_vector_round_kernel<<< 64, 64 >>> (buffer, num_rows * cols_per_row);

        //The following is tne reduction of the intermediate matrix (buffer).
        //Here, the sum of the elements in each row is calculated, and stored in the corresponding element of y
       ell_matrix_row_sum_kernel<<<num_rows / blockDim3 + 1, blockDim3, sizeof(mp_float_t) * blockDim3 >>>(y, indices, buffer, num_rows, cols_per_row);
    }

} // namespace cuda

#endif //MPSPMVELL_CUH
