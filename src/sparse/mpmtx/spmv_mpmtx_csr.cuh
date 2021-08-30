/*
 *  Multiple-precision sparse matrix-vector multiplication (SpMV) on GPU using the CSR sparse matrix format (mutiple precision matrix, multiple precision vectors)
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

#ifndef MPRES_SPMV_MPMTX_CSR_CUH
#define MPRES_SPMV_MPMTX_CSR_CUH

#include "../../arith/add.cuh"
#include "../../arith/mul.cuh"
#include "../../arith/assign.cuh"

namespace cuda {

    /*!
     * Performs the matrix-vector operation y = A * x, where x and y are dense vectors and A is a sparse matrix.
     * Scalar kernel - one thread is assigned to each row of the matrix, i.e. one element of the vector y.
     * The matrix should be stored in the CSR format: entries are stored in a dense array of nonzeros in row major order.
     *
     * @note The matrix is represented in multiple precision
     * @note Each operation using multiple precision is performed as a single thread
     * @note No global memory buffer is required
     *
     * @param m - number of rows in matrix
     * @param irp - row start pointers array of size m + 1, last element of irp equals to nnz (number of nonzeros in matrix)
     * @param ja - column indices array to access the corresponding elements of the vector x, size = nnz
     * @param as - multiple-precision coefficients array (entries of the matrix A in the CSR format), size = nnz
     * @param x - input vector, size at least max(ja) + 1, where max(ja) is the maximum element from the ja array
     * @param y - output vector, size at least m
     */
    __global__ void mpspmv_mpmtx_csr_scalar(const int m, const int *irp, const int *ja, mp_float_ptr as, mp_float_ptr x, mp_float_ptr y) {
        auto row = threadIdx.x + blockIdx.x * blockDim.x;
        while (row < m) {
            mp_float_t prod;
            mp_float_t dot = cuda::MP_ZERO;
            int row_start = irp[row];
            int row_end = irp[row+1];
            for (int i = row_start; i < row_end; i++) {
                cuda::mp_mul(&prod, &x[ja[i]], &as[i]);
                cuda::mp_add(&dot, &dot, &prod);
            }
            cuda::mp_set(&y[row], &dot);
            row +=  gridDim.x * blockDim.x;
        }
    }

} // namespace cuda

#endif //MPRES_SPMV_MPMTX_CSR_CUH
