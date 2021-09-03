/*
 *  Multiple-precision SpMV (Sparse matrix-vector multiplication) on GPU using the DIA sparse matrix format (multiple precision matrix)
 *  Computes the product of a sparse matrix and a dense vector
 *  SpMV DIA implementation
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

#ifndef MPRES_SPMV_MPMTX_DIA_CUH
#define MPRES_SPMV_MPMTX_DIA_CUH

#include "../../arith/add.cuh"
#include "../../arith/mul.cuh"
#include "../../arith/assign.cuh"

namespace cuda {

    /*!
     * Performs the matrix-vector operation y = A * x, where x and y are dense vectors and A is a sparse matrix.
     * The matrix should be stored in the DIA format: entries are stored in a dense array 'as' in column major order and explicit zeros are stored if necessary (zero padding)
     *
     * @note The matrix is represented in multiple precision
     * @note Each operation using multiple precision is performed as a single thread
     * @note One thread is assigned to compute one dot product, i.e. one element of the vector n
     * @note No global memory buffer is required
     *
     * @param m - number of rows in matrix
     * @param n - number of columns in matrix
     * @param ndiag - number of nonzero diagonals
     * @param offset - offset for diagonals
     * @param as - multiple-precision coefficients array (entries of the matrix A in the DIA format), size m * ndiag
     * @param x - input vector, size at least max(ja) + 1, where max(ja) is the maximum element from the ja array
     * @param y - output vector, size at least m
     */
    __global__ void mp_spmv_mpmtx_dia(const int m, const int n, const int ndiag, const int *offset, mp_float_ptr as, mp_float_ptr x, mp_float_ptr y) {
        auto row = threadIdx.x + blockIdx.x * blockDim.x;
        while (row < m) {
            mp_float_t prod;
            mp_float_t dot = cuda::MP_ZERO;
            for (int i = 0; i < ndiag; i++) {
                int col = row + offset[i];
                if(col  >= 0 && col < n) {
                    cuda::mp_mul(&prod, &x[col], &as[m * i + row]);
                    cuda::mp_add(&dot, &dot, &prod);
                }
            }
            cuda::mp_set(&y[row], &dot);
            row +=  gridDim.x * blockDim.x;
        }
    }

} // namespace cuda

#endif //MPRES_SPMV_MPMTX_DIA_CUH
