/*
 *  Multiple-precision sparse matrix-vector multiplication (SpMV) on GPU using the CSR sparse matrix format (double precision matrix, multiple precision vectors)
 *  Computes the product of a sparse matrix and a dense vector
 *  Scalar CSR kernel - one thread is assigned to each row of the matrix
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

#ifndef MPRES_SPMV_CSR_CUH
#define MPRES_SPMV_CSR_CUH

#include "../../arith/add.cuh"
#include "../../arith/muld.cuh"
#include "../../arith/assign.cuh"

namespace cuda {

    /*!
     * Performs the matrix-vector operation y = A * x, where x and y are dense vectors and A is a sparse matrix.
     * The matrix should be stored in the CSR format: entries are stored in a dense array of nonzeros in row major order.
     *
     * @note The matrix is represented in double precision
     * @note Each operation using multiple precision is performed as a single thread
     * @note Scalar kernel - one thread is assigned to each row of the matrix, i.e. one element of the vector y.
     * @note No global memory buffer is required
     *
     * @tparam threads - thread block size
     * @param m - number of rows in matrix
     * @param csr - sparse double-precision matrix in the CSR storage format
     * @param x - input vector, size at least max(ja) + 1, where max(ja) is the maximum element from the ja array
     * @param y - output vector, size at least m
     */
    template<int threads>
    __global__ void mp_spmv_csr(const int m, const csr_t csr, mp_float_ptr x, mp_float_ptr y) {
        auto row = threadIdx.x + blockIdx.x * blockDim.x;
        __shared__ mp_float_t sums[threads];
        __shared__ mp_float_t prods[threads];
        while (row < m) {
            sums[threadIdx.x] = cuda::MP_ZERO;
            int row_start = csr.irp[row];
            int row_end = csr.irp[row+1];
            for (int i = row_start; i < row_end; i++) {
                cuda::mp_mul_d(&prods[threadIdx.x], x[csr.ja[i]], csr.as[i]);
                cuda::mp_add(&sums[threadIdx.x], sums[threadIdx.x], prods[threadIdx.x]);
            }
            cuda::mp_set(&y[row], sums[threadIdx.x]);
            row +=  gridDim.x * blockDim.x;
        }
    }

} // namespace cuda

#endif //MPRES_SPMV_CSR_CUH
