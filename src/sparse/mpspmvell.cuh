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
     * The result is written into the matrix in the ELLPACK format (result) of size num_rows by num_cols_per_row
     * Kernel #1 --- Computing the exponents, signs, and interval evaluations (e-s-i)
     * @note The data and result arrays are assumed to be stored in the column major order, that is, [column 1] [column 2] ... [column n]
     * @param result - pointer to the multiple-precision result array, size num_rows * num_cols_per_row
     * @param data - pointer to the multiple-precision input array that contains matrix A in the ELLPACK format, size num_rows * num_cols_per_row
     * @param indices - pointer to the array of column indices that are used to access the corresponding elements of the vector x, size num_rows * num_cols_per_row (the same as for data)
     * @param x - pointer to the dense vector, size max(indices) + 1, where max(indices) is the maximum element from the indices array
     * @param num_rows - specifies the number of rows in the data array
     * @param num_cols_per_row - specifies the number of columns in the data array
     */
    __global__ static void ell_scal_esi_kernel(mp_array_t result, mp_array_t data, const int *indices, mp_array_t x, const int num_rows, const int num_cols_per_row) {
        unsigned int lenx = x.len[0];
        unsigned int lena = data.len[0];
        unsigned int lenr = result.len[0];
        unsigned int threadId = threadIdx.x + blockIdx.x * blockDim.x;
        if( threadId < num_rows ) {
            for (int colId = 0; colId < num_cols_per_row; colId++) {
                int index = indices[colId * num_rows + threadId];
                if(index >= 0){
                    //Load the corresponding vector element into the registers if possible
                    int x_sign = x.sign[index];
                    int x_exp = x.exp[index];
                    er_float_t x_ev0 = x.eval[index];
                    er_float_t x_ev1 = x.eval[index + lenx];
                    result.sign[colId * num_rows + threadId] = data.sign[colId * num_rows + threadId] ^ x_sign;
                    result.exp[colId * num_rows + threadId] = data.exp[colId * num_rows + threadId] + x_exp;
                    cuda::er_md_rd(&result.eval[colId * num_rows + threadId],&data.eval[colId * num_rows + threadId], &x_ev0, &cuda::RNS_EVAL_UNIT.upp);
                    cuda::er_md_ru(&result.eval[lenr + colId * num_rows + threadId],&data.eval[lena + colId * num_rows + threadId], &x_ev1,  &cuda::RNS_EVAL_UNIT.low);
                }
            }
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
    __global__ static void ell_scal_digits_kernel(mp_array_t result, mp_array_t data, const int *indices, mp_array_t x, const int num_rows, const int num_cols_per_row) {
        int lmodul = cuda::RNS_MODULI[threadIdx.x % RNS_MODULI_SIZE];
        unsigned int colId = blockIdx.y; // Index of the column
        while (colId < num_cols_per_row) {
            unsigned int digitId = blockIdx.x * blockDim.x + threadIdx.x; //Each thread is associated with its own digit
            unsigned int elementId = blockIdx.x * blockDim.x / RNS_MODULI_SIZE + threadIdx.x / RNS_MODULI_SIZE; //Index of the current matrix element
            while (digitId < num_rows * RNS_MODULI_SIZE) {
                //Index of the vector element
                int index = indices[colId * num_rows + elementId];
                if(index >= 0){
                    int idx = (index * RNS_MODULI_SIZE) + threadIdx.x % RNS_MODULI_SIZE;
                    int lx = x.digits[idx];
                    result.digits[colId * num_rows * RNS_MODULI_SIZE + digitId] = cuda::mod_mul(data.digits[colId * num_rows * RNS_MODULI_SIZE + digitId], lx, lmodul);
                }
                //Go to the next iteration
                digitId += gridDim.x * blockDim.x;
                elementId += gridDim.x * blockDim.x / RNS_MODULI_SIZE;
            }
            //Go to the next column
            colId += gridDim.y;
        }
    }

    /*!
     * Kernel that calculates the sum of all the elements in each row of multiple-precision matrix and stores it in the result vector
     * @note Shared memory of size sizeof(mp_float_t) * nThreads must be allocated, where nThreads is the number of threads per block
     * @param result - pointer to the result vector of num_rows elements
     * @param indices - pointer to the array of ELLPACK column indices. Addition is performed only if the corresponding index is positive, size num_rows * num_cols_per_row
     * @param data - pointer to the multiple-precision input array to be summed contains, size num_rows * num_cols_per_row
     * @param num_rows - matrix of m rows and n columns
     * @param num_cols_per_row - matrix of m rows and n columns
     */
    __global__ static void ell_matrix_row_sum_kernel(mp_array_t result, const int *indices, mp_array_t data, const int num_rows, const int num_cols_per_row){
        extern __shared__ mp_float_t sum[];
        unsigned int threadId = threadIdx.x + blockIdx.x * blockDim.x;
        if (threadId < num_rows) {
             sum[threadIdx.x] = cuda::MP_ZERO;
            for (int colId = 0; colId < num_cols_per_row; colId++) {
                int index = indices[colId * num_rows + threadId];
                if(index >= 0){
                    cuda::mp_add(&sum[threadIdx.x], &sum[threadIdx.x], data, colId * num_rows + threadId);
                }
            }
            cuda::mp_set(result, threadId, &sum[threadIdx.x]);
        }
    }

    /*!
     * Performs the matrix-vector operation y = A * x, where x and y are dense vectors and A is a sparse matrix.
     * The matrix should be stored in the ELLPACK format: entries are stored in a dense array in column major order and explicit zeros are stored if necessary (zero padding)

     * @tparam blockDim1 - number of threads per block used to compute the signs, exponents, interval evaluations in an element-wise matrix-vector operation (diagonal scaling), and for parallel summation
     * @tparam gridDim2 - number of blocks used to compute the digits of multiple-precision significands in an element-wise matrix-vector operation (diagonal scaling).  A 2D grid of num_cols_per_row x gridDim2 blocks will be launched
     *
     * @param num_rows - specifies the number of rows of the matrix A. The value of num_rows must be greater than zero.
     * @param num_cols_per_row - specifies the maximum number of nonzeros per row. The value of num_cols_per_row must be greater than zero.
     * @param data - pointer to the multiple-precision array, size num_rows * num_cols_per_row, in the global GPU memory that contains matrix A in the ELLPACK format
     * @param indices - pointer to the array of column indices that are used to access the corresponding elements of the vector x, size num_rows * num_cols_per_row (the same as for data)
     * @param x - pointer to the dense vector in the global GPU memory, size at least max(indices) + 1, where max(indices) is the maximum element from the indices array
     * @param y - pointer to the result vector in the global GPU memory, size at least num_rows
     * @param buffer - auxiliary array, size num_rows * num_cols_per_row, in the global GPU memory for storing the intermediate matrix
     */
    template<int blockDim1, int gridDim2>
    void mpspmvell(const int num_rows, const int num_cols_per_row, const int *indices, mp_array_t &data, mp_array_t &x, mp_array_t &y, mp_array_t &buffer) {

        //Execution configuration
        // To compute the signs, exponents, and interval evaluations
        dim3 grid1(num_rows / blockDim1 + 1, 1);
        // To compute the digits in RNS
        dim3 grid2(num_cols_per_row, gridDim2);
        //  To rounding the result (we do not currently parameterize rounding)
        int blocks_round = 64;
        int threads_round = 64;

        //We consider the vector x as a diagonal matrix and perform the right diagonal scaling, buffer = A * x
        //For exponents, sings, and interval evaluations, we run a regular kernel with a 1D grid of 1D thread blocks
        //For digits, we run a 2D grid of 1D blocks, and each line in the grid (i.e., all blocks with the same y coordinate)
        // is associated with its own column of the ELLPACK arrays (data and indices).
        //The result is written to the intermediate buffer of size num_rows * num_cols_per_row.

        //Multiplication buffer = A * x - Computing the signs, exponents, and interval evaluations
        ell_scal_esi_kernel<<< grid1, blockDim1 >>> (buffer, data, indices, x, num_rows, num_cols_per_row);

        //Multiplication buffer = A * x - Multiplying the digits in the RNS
        ell_scal_digits_kernel<<< grid2, BLOCK_SIZE_FOR_RESIDUES >>> (buffer, data, indices, x, num_rows, num_cols_per_row);

        //Rounding the intermediate result (buffer)
        mp_vector_round_kernel<<< blocks_round, threads_round >>> (buffer, 1, num_rows * num_cols_per_row);

        //The following is tne reduction of the intermediate matrix (buffer).
        //Here, the sum of the elements in each row is calculated, and stored in the corresponding element of y
        //Kernel memory configurations. We prefer shared memory
        //cudaFuncSetCacheConfig(matrix_row_sum_kernel, cudaFuncCachePreferShared);
        ell_matrix_row_sum_kernel<<<grid1, blockDim1, sizeof(mp_float_t) * blockDim1 >>>(y, indices, buffer, num_rows, num_cols_per_row);
    }

} // namespace cuda

#endif //MPSPMVELL_CUH
