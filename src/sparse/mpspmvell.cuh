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
#include "../mpreduct_rowcol.cuh"
#include "../kernel_config.cuh"

namespace cuda {

    /*!
     * Multiplication of a sparse matrix A stored in the ELLPACK format (data) by a diagonal matrix on the right which is stored as a vector x
     * The result is written into the matrix in the ELLPACK format (result) of size num_rows by num_cols_per_row
     * Kernel #1 --- Computing the exponents, signs, and interval evaluations (e-s-i)
     * @note The data and result arrays are assumed to be stored in the column major order, that is, [column 1] [column 2] ... [column n]
     * @note This kernel can be run on a 2D grid of 1D blocks. Each line in the grid (i.e., all blocks with the same y coordinate) is associated with its own column of the data array.
     * @param result - pointer to the multiple-precision result array, size num_rows * num_cols_per_row
     * @param data - pointer to the multiple-precision input array that contains matrix A in the ELLPACK format, size num_rows * num_cols_per_row
     * @param indices - pointer to the array of column indices that are used to access the corresponding elements of the vector x, size num_rows * num_cols_per_row (the same as for data)
     * @param x - pointer to the dense vector, size max(indices) + 1, where max(indices) is the maximum element from the indices array
     * @param num_rows - specifies the number of rows in the data array
     * @param num_cols_per_row - specifies the number of columns in the data array
     */
    __global__ static void ellpack_scal_esi_kernel(mp_array_t result, mp_array_t data, const int *indices, mp_array_t x, const int num_rows, const int num_cols_per_row) {
        unsigned int lenx = x.len[0];
        unsigned int lena = data.len[0];
        unsigned int lenr = result.len[0];
        unsigned int colId = blockIdx.y; // The column index
        //Iterate over matrix columns / vector elements
        while (colId < num_cols_per_row) {
            //We process in the stride loop all the elements of the i-th column of data
            //Index of the element of data in the colId-th column. Must be less than num_rows
            int index = blockDim.x * blockIdx.x + threadIdx.x;
            while (index < num_rows) {
                int idx = indices[colId * num_rows + index];
                //Load the corresponding vector element into the registers if possible
                int x_sign = x.sign[idx];
                int x_exp = x.exp[idx];
                er_float_t x_ev0 = x.eval[idx];
                er_float_t x_ev1 = x.eval[idx + lenx];

                result.sign[colId * num_rows + index] = data.sign[colId * num_rows + index] ^ x_sign;
                result.exp[colId * num_rows + index] = data.exp[colId * num_rows + index] + x_exp;
                cuda::er_md_rd(&result.eval[colId * num_rows + index], &data.eval[colId * num_rows + index], &x_ev0,
                               &cuda::RNS_EVAL_UNIT.upp);
                cuda::er_md_ru(&result.eval[lenr + colId * num_rows + index],
                               &data.eval[lena + colId * num_rows + index], &x_ev1,
                               &cuda::RNS_EVAL_UNIT.low);
                //Go to the next iteration
                index += gridDim.x * blockDim.x;
            }
            //Go to the next column
            colId += gridDim.y;
        }
    }

    /*!
     * Multiplication of a sparse matrix A stored in the ELLPACK format (data) by a diagonal matrix on the right which is stored as a vector x
     * The result is written into the matrix in the ELLPACK format (result) of size num_rows by num_cols_per_row
     * Kernel #2 --- Computing the significands in the RNS (digits)
     * @note The data and result arrays are assumed to be stored in the column major order, that is, [column 1] [column 2] ... [column n]
     * @note This kernel can be run on a 2D grid of 1D blocks. Each line in the grid (i.e., all blocks with the same y coordinate) is associated with its own column of the data array.
     * @param result - pointer to the multiple-precision result array, size num_rows * num_cols_per_row
     * @param data - pointer to the multiple-precision input array that contains matrix A in the ELLPACK format, size num_rows * num_cols_per_row
     * @param indices - pointer to the array of column indices that are used to access the corresponding elements of the vector x, size num_rows * num_cols_per_row (the same as for data)
     * @param x - pointer to the dense vector, size max(indices) + 1, where max(indices) is the maximum element from the indices array
     * @param num_rows - specifies the number of rows in the data array
     * @param num_cols_per_row - specifies the number of columns in the data array
     */
    __global__ static void ellpack_scal_digits_kernel(mp_array_t result, mp_array_t data, const int *indices, mp_array_t x, const int num_rows, const int num_cols_per_row) {
        int lmodul = cuda::RNS_MODULI[threadIdx.x % RNS_MODULI_SIZE];
        int colId = blockIdx.y; // The column index
        while (colId < num_cols_per_row) {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            int index_M = blockIdx.x * blockDim.x / RNS_MODULI_SIZE + threadIdx.x / RNS_MODULI_SIZE;
            while (index < num_rows * RNS_MODULI_SIZE) {
                int ix = (indices[colId * num_rows + index_M] * RNS_MODULI_SIZE) + threadIdx.x % RNS_MODULI_SIZE;
                int lx = x.digits[ix];
                result.digits[colId * num_rows * RNS_MODULI_SIZE + index] = cuda::mod_mul(
                        data.digits[colId * num_rows * RNS_MODULI_SIZE + index], lx, lmodul);
                index += gridDim.x * blockDim.x;
                index_M += gridDim.x * blockDim.x / RNS_MODULI_SIZE;
            }
            colId += gridDim.y;
        }
    }

    /*!
     * Performs the matrix-vector operation y = A * x
     * where x and y are dense vectors and A is a sparse matrix.
     * The matrix should be stored in the ELLPACK format: entries are stored in a dense array in column major order and explicit zeros are stored if necessary (zero padding)

     * @tparam gridDim1 - number of blocks used to compute the signs, exponents, interval evaluations in an element-wise matrix-vector operation (diagonal scaling). A 2D grid of gridDim1 x gridDim1 blocks will be launched
     * @tparam blockDim1 - number of threads per block used to compute the signs, exponents, interval evaluations, and also to round the result in an element-wise matrix-vector operation (diagonal scaling)
     * @tparam gridDim2 - number of blocks  used to compute the digits of multiple-precision significands in an element-wise matrix-vector operation (diagonal scaling).  A 2D grid of gridDim2 x gridDim2 blocks will be launched
     * @tparam blockDim3 - number of threads per block for parallel summation (the number of blocks is equal to the size of y)
     *
     * @param num_rows - specifies the number of rows of the matrix A. The value of num_rows must be greater than zero.
     * @param num_cols_per_row - specifies the maximum number of nonzeros per row. The value of num_cols_per_row must be greater than zero.
     * @param data - pointer to the multiple-precision array, size num_rows * num_cols_per_row, in the global GPU memory that contains matrix A in the ELLPACK format
     * @param indices - pointer to the array of column indices that are used to access the corresponding elements of the vector x, size num_rows * num_cols_per_row (the same as for data)
     * @param x - pointer to the dense vector in the global GPU memory, size at least max(indices) + 1, where max(indices) is the maximum element from the indices array
     * @param y - pointer to the result vector in the global GPU memory, size at least num_rows
     * @param buffer - auxiliary array, size num_rows * num_cols_per_row, in the global GPU memory for storing the intermediate matrix
     */
    template<int gridDim1, int blockDim1, int gridDim2, int blockDim3>
    void mpspmvell(const int num_rows, const int num_cols_per_row, const int *indices, mp_array_t &data, mp_array_t &x, mp_array_t &y, mp_array_t &buffer) {

        //Execution configuration
        //  To compute the signs, exponents, and interval evaluations
        dim3 grid1(gridDim1, gridDim1);
        //  To compute the digits in RNS
        dim3 grid2(gridDim2, gridDim2);

        //We consider the vector x as a diagonal matrix and perform the right diagonal scaling, buffer = A * x
        //We run a 2D grid of 1D blocks.
        //Each line in the grid (i.e., all blocks with the same y coordinate) is associated with its own column of the ELLPACK data.
        //The result is written to the intermediate num_rows by num_cols_per_row buffer.

        //Multiplication buffer = A * x - Computing the signs, exponents, and interval evaluations
        ellpack_scal_esi_kernel<<< grid1, blockDim1 >>> (buffer, data, indices, x, num_rows, num_cols_per_row);

        //Multiplication buffer = A * x - Multiplying the digits in the RNS
        ellpack_scal_digits_kernel<<< grid2, BLOCK_SIZE_FOR_RESIDUES >>> (buffer, data, indices, x, num_rows, num_cols_per_row);

        //Rounding the intermediate result (buffer)
        mp_vector_round_kernel<<< gridDim1, blockDim1 >>> (buffer, 1, num_rows * num_cols_per_row);

        //The following is tne reduction of the intermediate matrix (buffer).
        //Here, the sum of the elements in each row is calculated, and stored in the corresponding element of y

        // Kernel memory configurations. We prefer shared memory
        //cudaFuncSetCacheConfig(matrix_row_sum_kernel, cudaFuncCachePreferShared);

        // Power of two that is greater that or equals to blockDim3
        const unsigned int POW = nextPow2(blockDim3);

        matrix_row_sum_kernel<<< num_rows, blockDim3, sizeof(mp_float_t) * blockDim3 >>> (num_rows, num_cols_per_row, buffer, y, 1, POW);
    }

} // namespace cuda

#endif //MPSPMVELL_CUH
