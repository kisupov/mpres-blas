/*
 *  Multiple-precision sparse matrix-vector multiplication (SpMV) on GPU using the CSR sparse matrix format (mutiple precision matrix, multiple precision vectors)
 *  Vector CSR kernel - multiple threads (up to 32) assigned to each row of the matrix
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

#ifndef MPRES_SPMV_MPMTX_CSRV_CUH
#define MPRES_SPMV_MPMTX_CSRV_CUH

#include "../../arith/add.cuh"
#include "../../arith/mul.cuh"
#include "../../arith/assign.cuh"

namespace cuda {

    /*!
     * Performs the matrix-vector operation y = A * x, where x and y are dense vectors and A is a sparse matrix.
     * Vector kernel - a group of threads (up to 32 threads) are assigned to each row of the matrix, i.e. one element of the vector y.
     * The matrix should be stored in the CSR format: entries are stored in a dense array of nonzeros in row major order.
     *
     * @note The matrix and vectors are in multiple precision
     * @note Each operation using multiple precision is performed as a single thread
     * @note No global memory buffer is required
     * @note Shared memory of size sizeof(mp_float_t) * blockDim.x must be allocated
     *
     * @tparam threadsPerRow - number of threads assigned to compute one row of the matrix (must be a power of two from 1 to 32, i.e., 1, 2, 4, 8, 16, or 32)
     * @param m - number of rows in matrix
     * @param irp - row start pointers array of size m + 1, last element of irp equals to nnz (number of nonzeros in matrix)
     * @param ja - column indices array to access the corresponding elements of the vector x, size = nnz
     * @param as - multiple-precision coefficients array (entries of the matrix A in the CSR format), size = nnz
     * @param x - input vector, size at least max(ja) + 1, where max(ja) is the maximum element from the ja array
     * @param y - output vector, size at least m
     */
    template<int threadsPerRow>
    __global__ void mp_spmv_mpmtx_csrv(const int m, const int *irp, const int *ja, mp_float_ptr as, mp_float_ptr x, mp_float_ptr y) {
        extern __shared__ mp_float_t vals[];

        auto threadId = threadIdx.x + blockIdx.x * blockDim.x; // global thread index
        auto groupId = threadId / threadsPerRow; // global thread group index
        auto lane = threadId & (threadsPerRow - 1); // thread index within the group
        auto row = groupId; // one group per row

        while (row < m) {
            mp_float_t prod;
            int row_start = irp[row];
            int row_end = irp[row + 1];
            // compute running sum per thread
            vals[threadIdx.x] = cuda::MP_ZERO;
            for (auto i = row_start + lane; i < row_end; i += threadsPerRow) {
                cuda::mp_mul(&prod, as[i], x[ja[i]]);
                cuda::mp_add(&vals[threadIdx.x], vals[threadIdx.x], prod);
            }
            // parallel reduction in shared memory
            if (threadsPerRow >= 32 && lane < 16) {
                cuda::mp_add(&vals[threadIdx.x], vals[threadIdx.x], vals[threadIdx.x + 16]);
            }
            if (threadsPerRow >= 16 && lane < 8) {
                cuda::mp_add(&vals[threadIdx.x], vals[threadIdx.x], vals[threadIdx.x + 8]);
            }
            if (threadsPerRow >= 8 && lane < 4) {
                cuda::mp_add(&vals[threadIdx.x], vals[threadIdx.x], vals[threadIdx.x + 4]);
            }
            if (threadsPerRow >= 4 && lane < 2) {
                cuda::mp_add(&vals[threadIdx.x], vals[threadIdx.x], vals[threadIdx.x + 2]);
            }
            if (threadsPerRow >= 2 && lane < 1) {
                cuda::mp_add(&vals[threadIdx.x], vals[threadIdx.x], vals[threadIdx.x + 1]);
            }
            // first thread writes the result
            if (lane == 0) {
                cuda::mp_set(&y[row], vals[threadIdx.x]);
            }
            row +=  gridDim.x * blockDim.x / threadsPerRow;
        }
    }

} // namespace cuda

#endif //MPRES_SPMV_MPMTX_CSRV_CUH
