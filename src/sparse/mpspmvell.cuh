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

#ifndef SPMV_CUH
#define SPMV_CUH


#include "../mpvector.cuh"
#include "../mpmatrix.cuh"
#include "../kernel_config.cuh"
#include "../mblas_enum.cuh"

namespace cuda {

    __global__ void mp_mat2vec_ellpack_right_scal_esi_kernel(mp_array_t R, mp_array_t A, int *indices, mp_array_t x,
            const int m, const int n, const int maxNonZeros) {
        unsigned int lenx = x.len[0];
        unsigned int lena = A.len[0];
        unsigned int lenr = R.len[0];
        unsigned int colId = blockIdx.y; // The column index
        //Iterate over matrix columns / vector elements
        while (colId < maxNonZeros) {
            //We process in the stride loop all the elements of the i-th column of A
            int index = blockDim.x * blockIdx.x + threadIdx.x; //Index of the element of A in the colId-th column. Must be less than m
            while (index < m) {
                int idx = indices[colId * m + index];
                //Load the corresponding vector element into the registers if possible
                int x_sign = x.sign[idx];
                int x_exp = x.exp[idx];
                er_float_t x_ev0 = x.eval[idx];
                er_float_t x_ev1 = x.eval[idx + lenx];

                R.sign[colId * m + index] = A.sign[colId * m + index] ^ x_sign;
                R.exp[colId * m + index] = A.exp[colId * m + index] + x_exp;
                cuda::er_md_rd(&R.eval[colId * m + index], &A.eval[colId * m + index], &x_ev0, &cuda::RNS_EVAL_UNIT.upp);
                cuda::er_md_ru(&R.eval[lenr + colId * m + index], &A.eval[lena + colId * m + index], &x_ev1, &cuda::RNS_EVAL_UNIT.low);
                //Go to the next iteration
                index += gridDim.x * blockDim.x;
            }
            //Go to the next column
            colId += gridDim.y;
        }
    }


    __global__ static void
    mp_mat2vec_ellpack_right_scal_digits_kernel(mp_array_t R, mp_array_t A, int *indices, mp_array_t x, const int m,
            const int n, const int maxNonZeros) {
        int lmodul = cuda::RNS_MODULI[threadIdx.x % RNS_MODULI_SIZE];
        int colId = blockIdx.y; // The column index
        while (colId < maxNonZeros) {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            int index_M = blockIdx.x * blockDim.x / RNS_MODULI_SIZE + threadIdx.x / RNS_MODULI_SIZE;
            while (index < m * RNS_MODULI_SIZE) {
                int ix = (indices[colId * m + index_M] * RNS_MODULI_SIZE) + threadIdx.x % RNS_MODULI_SIZE;
                int lx = x.digits[ix];
                R.digits[colId * m * RNS_MODULI_SIZE + index] = cuda::mod_mul(A.digits[colId * m * RNS_MODULI_SIZE + index], lx, lmodul);
                index += gridDim.x * blockDim.x;
                index_M += gridDim.x * blockDim.x / RNS_MODULI_SIZE;
            }
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
    __global__ static void matrix_row_sum_kernel(const unsigned int m, const unsigned int n, mp_array_t A, mp_array_t y, int incy, const unsigned int nextPow2) {
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
        while (i >= 1) {
            if ((tid < i) && (tid + i < bsize)) {
                cuda::mp_add(&sdata[tid], &sdata[tid], &sdata[tid + i]);
            }
            i = i >> 1;
            __syncthreads();
        }

        // write result for this block to global mem
        if (tid == 0) {
            int iy = incy > 0 ? bid * incy : (-m + bid + 1) * incy;
            cuda::mp_set(y, iy, &sdata[tid]);
        }
        //__syncthreads();
    }

    void show_matrix(mp_array_t data, int m, int maxNonZeros) {
        mp_float_ptr hdata = new mp_float_t[m * maxNonZeros];
        cuda::mp_array_device2host(hdata, data, m * maxNonZeros);
        std::cout << "data" << std::endl;
/*        for (int j = 0; j < m; ++j) {
            for (int i = 0; i < (maxNonZeros); ++i) {
                std::cout << mp_get_d(&hdata[j + m * i]) << " ";
            }
            std::cout << std::endl;
        }*/
        for (int i = 0; i < m * maxNonZeros; ++i) {
            std::cout << i << " = " << mp_get_d(&hdata[i]) << std::endl;
        }
    }


    template<int gridDim1, int blockDim1, int gridDim2, int blockDim3>
    void spmv(enum mblas_trans_type trans, const int m, const int n, const int maxNonZeros, mp_array_t &A,
              int *indices, mp_array_t &x, mp_array_t &y, mp_array_t &buffer1) {

        //Execution configuration
        //  To compute the signs, exponents, and interval evaluations
        dim3 grid1(gridDim1, gridDim1);
        //  To compute the digits in RNS
        dim3 grid2(gridDim2, gridDim2);
        //  To compute digits (residues) in the vector operations


        //Multiplication buffer1 = A * x - Computing the signs, exponents, and interval evaluations
        mp_mat2vec_ellpack_right_scal_esi_kernel << < grid1, blockDim1 >> >
                                                             (buffer1, A, indices, x, m, n, maxNonZeros);

        //Multiplication buffer1 = A * x - Multiplying the digits in the RNS
        mp_mat2vec_ellpack_right_scal_digits_kernel << < grid2, BLOCK_SIZE_FOR_RESIDUES >> >
                                                                (buffer1, A, indices, x, m, n, maxNonZeros);

        //Rounding the intermediate result (buffer1)
        mp_vector_round_kernel << < gridDim1, blockDim1 >> > (buffer1, 1, m * maxNonZeros);

        //show_matrix(buffer1, m, maxNonZeros);

        //The following is tne reduction of the intermediate matrix (buffer 1).
        //Here, the sum of the elements in each row is calculated, and then y is added to the calculated sum
        //The result is a vector of size m

        // Kernel memory configurations. We prefer shared memory
        //cudaFuncSetCacheConfig(matrix_row_sum_kernel, cudaFuncCachePreferShared);

        // Power of two that is greater that or equals to blockDim3
        const unsigned int POW = nextPow2(blockDim3);

        //TODO переделать метод так, чтобы он не суммировал к y, а менял его значение
        matrix_row_sum_kernel << < m, blockDim3, sizeof(mp_float_t) * blockDim3 >> >
                                                 (m, maxNonZeros, buffer1, y, 1, POW);
    }

} // namespace cuda

#endif //SPMV_CUH
