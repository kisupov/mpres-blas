/*
 *  Multiple-precision SpMV (Sparse matrix-vector multiplication) on GPU using the ELLPACK sparse matrix format (double precision matrix, multiple precision vectors)
 *  Scalar ELLPACK kernel - one thread is assigned to each row of the matrix
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

#ifndef MPDSPMV_ELL_SCALAR_CUH
#define MPDSPMV_ELL_SCALAR_CUH

#include "arith/mpadd.cuh"
#include "arith/mpmuld.cuh"
#include "arith/mpassign.cuh"

namespace cuda {

    /*!
     * Performs the matrix-vector operation y = A * x, where x and y are dense vectors and A is a sparse matrix.
     * The matrix should be stored in the ELLPACK format: entries are stored in a dense array 'as' in column major order and explicit zeros are stored if necessary (zero padding)
     *
     * @note The matrix is represented in double precision
     * @note Each operation using multiple precision is performed as a single thread
     * @note One thread is assigned to compute one dot product, i.e. one element of the vector n
     * @note No global memory buffer is required
     *
     * @param m - number of rows in matrix
     * @param nzr - number of nonzeros per row array (maximum number of nonzeros per row in the matrix A)
     * @param ja - column indices array to access the corresponding elements of the vector x, size m * nzr (the same as for A)
     * @param as - double-precision coefficients array (entries of the matrix A in the ELLPACK format), size m * nzr
     * @param x - input vector, size at least max(ja) + 1, where max(ja) is the maximum element from the ja array
     * @param y - output vector, size at least m
     */
    __global__ void mpdspmv_ell_scalar(const int m, const int nzr, const int *ja, const double *as, mp_float_ptr x, mp_float_ptr y) {
        auto row = threadIdx.x + blockIdx.x * blockDim.x;
        while (row < m) {
            mp_float_t prod;
            mp_float_t dot = cuda::MP_ZERO;
            for (int col = 0; col < nzr; col++) {
                int index = ja[col * m + row];
                if(index >= 0){
                    cuda::mp_mul_d(&prod, &x[index], as[col * m + row]);
                    cuda::mp_add(&dot, &dot, &prod);
                }
            }
            cuda::mp_set(&y[row], &dot);
            row +=  gridDim.x * blockDim.x;
        }
    }

} // namespace cuda

#endif //MPDSPMV_ELL_SCALAR_CUH
