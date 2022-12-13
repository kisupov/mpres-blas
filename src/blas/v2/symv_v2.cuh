/*
 *  Multiple-precision SYMV function for GPU (BLAS Level-2)
 *  Computes a matrix-vector product with a symmetric matrix.
 *
 *  Copyright 2021 by Konstantin Isupov.
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

#ifndef MPRES_SYMV_V2_CUH
#define MPRES_SYMV_V2_CUH

#include "arith/add.cuh"
#include "arith/mul.cuh"
#include "blas/mblas_enum.cuh"

namespace cuda
{
    /*!
     * Performs the matrix-vector operation y = alpha*A*x + beta*y for a symmetric matrix
     * (only one triangular part of the matrix, upper or lower, is used in the calculation)
     *
     * @note Each operation using multiple precision is performed as a single thread
     * @note Each thread works with its own row, i.e. goes through the columns
     * @note No global memory buffer is required
     *
     * @tparam threads - thread block size
     * @param uplo - specifies whether the upper or lower triangular part of the array a is used.
     * @param n - the order of the matrix A. The value of n must be at least zero.
     * @param alpha - the input scalar alpha.
     * @param A - array, size (lda, n). Before entry with uplo = mblas_upper, the leading n-by-n upper triangular part of the array A
     *            must contain the upper triangular part of the symmetric matrix A and the strictly lower triangular part of A is not referenced.
     *            Before entry with uplo != mblas_upper, the leading n-by-n lower triangular part of the array A must contain the lower triangular
     *            part of the symmetric matrix A and the strictly upper triangular part of a is not referenced.
     * @param lda - the leading dimension of a. It must be at least max(1, n).
     * @param x - array, size at least (1 + (n - 1)*abs(incx)). Before entry, the incremented array x must contain the n-element vector x.
     * @param incx - the increment for the elements of x. The value of incx must not be zero.
     * @param beta - the input scalar beta. When beta is supplied as zero, then y need not be set on input.
     * @param y - array, size at least (1 + (n - 1)*abs(incy)). Overwritten by the updated vector y. Before entry, the incremented array y must contain the n-element vector y.
     * @param incy - the increment for the elements of y. The value of incy must not be zero.
     *
     */
    template<int threads>
    __global__ void mp_symv(enum mblas_uplo_type uplo, int n, mp_float_ptr alpha, mp_float_ptr A, int lda, mp_float_ptr x, const int incx, mp_float_ptr beta, mp_float_ptr y, const int incy) {
        auto row = threadIdx.x + blockIdx.x * blockDim.x;
        auto iy = incy > 0 ? row * incy : (-n + row + 1)*incy;
        __shared__ mp_float_t sums[threads];
        __shared__ mp_float_t prods[threads];
        while (row < n) {
            sums[threadIdx.x] = cuda::MP_ZERO;
            if (uplo == mblas_upper) { //Use the upper part of the matrix
                for (int colId = 0; colId < n; colId++) {
                    auto ix = incx > 0 ? colId * incx : (-n + colId + 1) * incx;
                    if(row <= colId){
                        cuda::mp_mul(&prods[threadIdx.x], x[ix], A[row + colId * lda]);
                        cuda::mp_add(&sums[threadIdx.x], sums[threadIdx.x], prods[threadIdx.x]);
                    } else{
                        cuda::mp_mul(&prods[threadIdx.x], x[ix], A[colId + row * lda]);
                        cuda::mp_add(&sums[threadIdx.x], sums[threadIdx.x], prods[threadIdx.x]);
                    }
                }
            }
            else{ //Use the lower part of the matrix
                for (int colId = 0; colId < n; colId++) {
                    auto ix = incx > 0 ? colId * incx : (-n + colId + 1) * incx;
                    if(row <= colId){
                        cuda::mp_mul(&prods[threadIdx.x], x[ix], A[colId + row * lda]);
                        cuda::mp_add(&sums[threadIdx.x], sums[threadIdx.x], prods[threadIdx.x]);
                    } else{
                        cuda::mp_mul(&prods[threadIdx.x], x[ix], A[row + colId * lda]);
                        cuda::mp_add(&sums[threadIdx.x], sums[threadIdx.x], prods[threadIdx.x]);
                    }
                }

            }
            cuda::mp_mul(&y[iy], beta[0], y[iy]);
            cuda::mp_mul(&sums[threadIdx.x], alpha[0], sums[threadIdx.x]);
            cuda::mp_add(&y[iy], y[iy], sums[threadIdx.x]);
            row +=  gridDim.x * blockDim.x;
            iy += gridDim.x * blockDim.x * incy;
        }
    }
} // namespace cuda

#endif //MPRES_SYMV_V2_CUH
