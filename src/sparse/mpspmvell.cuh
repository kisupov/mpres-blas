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

#ifndef MPSPMVELL_CUH
#define MPSPMVELL_CUH


#include "../mpvector.cuh"
#include "../mpmatrix.cuh"
#include "../kernel_config.cuh"
#include "../mblas_enum.cuh"

namespace cuda {

    __global__ void mp_mat2vec_ellpack_scal_esi_kernel(mp_array_t result, mp_array_t data, int *indices, mp_array_t x,
                                                       const int num_rows, const int num_cols_per_row) {
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


    __global__ static void
    mp_mat2vec_ellpack_scal_digits_kernel(mp_array_t result, mp_array_t data, int *indices, mp_array_t x,
                                          const int num_rows,
                                          const int num_cols_per_row) {
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

    /*
     * Kernel that calculates the sum of all the elements in each row of an m-by-n multiple-precision matrix
     * The result (a vector of size m) is then added to the vector y
     * @param data - matrix of m rows and n columns
     * @param y - vector of size m
     * @param nextPow2 - least power of two greater than or equal to blockDim.x
     */
    __global__ static void
    matrix_row_sum_kernel(const unsigned int m, const unsigned int n, mp_array_t data, mp_array_t y,
                          const unsigned int nextPow2) {
        extern __shared__ mp_float_t
        sdata[];

        // parameters
        const unsigned int tid = threadIdx.x;
        const unsigned int bid = blockIdx.x;
        const unsigned int bsize = blockDim.x;
        unsigned int i = threadIdx.x;

        // do reduction in global mem
        sdata[tid] = cuda::MP_ZERO;
        while (i < n) {
            cuda::mp_add(&sdata[tid], &sdata[tid], data, i * m + bid);
            i += bsize;
        }
        __syncthreads();

        // do reduction in shared mem
        i = nextPow2 >> 1; // half of nextPow2
        while (i >= 1) {
            if ((tid < i) && (tid + i < bsize)) {
                cuda::mp_add(&sdata[tid], &sdata[tid], &sdata[tid + i]);
            }
            i = i >> 1;
            __syncthreads();
        }

        // write result for this block to global mem
        if (tid == 0) {
            cuda::mp_set(y, bid, &sdata[tid]);
        }
        //__syncthreads();
    }

    template<int gridDim1, int blockDim1, int gridDim2, int blockDim3>
    void mpspmvell(const int num_rows, const int num_cols_per_row, mp_array_t &data,
                   int *indices, mp_array_t &x, mp_array_t &y, mp_array_t &buffer1) {

        //Execution configuration
        //  To compute the signs, exponents, and interval evaluations
        dim3 grid1(gridDim1, gridDim1);
        //  To compute the digits in RNS
        dim3 grid2(gridDim2, gridDim2);
        //  To compute digits (residues) in the vector operations


        //Multiplication buffer1 = data * x - Computing the signs, exponents, and interval evaluations
        mp_mat2vec_ellpack_scal_esi_kernel << < grid1, blockDim1 >> >
                                                       (buffer1, data, indices, x, num_rows, num_cols_per_row);

        //Multiplication buffer1 = data * x - Multiplying the digits in the RNS
        mp_mat2vec_ellpack_scal_digits_kernel << < grid2, BLOCK_SIZE_FOR_RESIDUES >> >
                                                          (buffer1, data, indices, x, num_rows, num_cols_per_row);

        //Rounding the intermediate result (buffer1)
        mp_vector_round_kernel << < gridDim1, blockDim1 >> > (buffer1, 1, num_rows * num_cols_per_row);

        //The following is tne reduction of the intermediate matrix (buffer 1).
        //Here, the sum of the elements in each row is calculated, and then y is added to the calculated sum
        //The result is a vector of size m

        // Kernel memory configurations. We prefer shared memory
        //cudaFuncSetCacheConfig(matrix_row_sum_kernel, cudaFuncCachePreferShared);

        // Power of two that is greater that or equals to blockDim3
        const unsigned int POW = nextPow2(blockDim3);

        matrix_row_sum_kernel << < num_rows, blockDim3, sizeof(mp_float_t) * blockDim3 >> >
                                                        (num_rows, num_cols_per_row, buffer1, y, POW);
    }

} // namespace cuda

#endif //MPSPMVELL_CUH
