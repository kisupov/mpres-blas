/*
 *  Multiple-precision SpMV (Sparse matrix-vector multiplication) on GPU using the JAD (JDS) sparse matrix format (double precision matrix)
 *  Computes the product of a sparse matrix and a dense vector
 *  Vector kernel - multiple threads (up to 32) assigned to each row of the matrix
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

#ifndef MPRES_SPMV_JADV_CUH
#define MPRES_SPMV_JADV_CUH

#include "../../arith/add.cuh"
#include "../../arith/muld.cuh"
#include "../../arith/assign.cuh"
#include "../utils/jad_utils.cuh"

namespace cuda {

    /*!
     * Performs the matrix-vector operation y = A * x, where x and y are dense vectors and A is a sparse matrix.
     * The matrix should be stored in the JAD (JDS) format: entries are stored in a dense array 'as' in column major order and explicit zeros are stored if necessary (zero padding)
     *
     * @note The matrix is represented in double precision
     * @note Each operation using multiple precision is performed as a single thread
     * @note Vector kernel - a group of threads (up to 32 threads) are assigned to each row of the matrix, i.e. one element of the vector y.
     * @note No global memory buffer is required
     *
     * @tparam threads - thread block size
     * @tparam threadsPerRow - number of threads assigned to compute one row of the matrix (must be a power of two from 1 to 32, i.e., 1, 2, 4, 8, 16, or 32)
     * @param m - number of rows in matrix
     * @param maxnzr - maximum number of nonzeros per row in the matrix A
     * @param jad -sparse double-precision matrix in the JAD storage format
     * @param x - input vector, size at least max(ja) + 1, where max(ja) is the maximum element from the ja array
     * @param y - output vector, size at least m
     */
    template<int threads, int threadsPerRow>
    __global__ void mp_spmv_jadv(const int m, const int maxnzr, const jad_t jad, mp_float_ptr x, mp_float_ptr y) {
        __shared__ mp_float_t sums[threads];
        __shared__ mp_float_t prods[threads];
        auto threadId = threadIdx.x + blockIdx.x * blockDim.x;  // global thread index
        auto groupId = threadId / threadsPerRow;                // global thread group index
        auto lane = threadId & (threadsPerRow - 1);             // thread index within the group
        auto row = groupId;                                     // one group per row
        while (row < m) {
            sums[threadIdx.x] = cuda::MP_ZERO;
            for (auto j = lane; j < maxnzr; j += threadsPerRow) {
                auto index = row + jad.jcp[j];
                if (index < jad.jcp[j + 1]) {
                    cuda::mp_mul_d(&prods[threadIdx.x], x[jad.ja[index]], jad.as[index]);
                    cuda::mp_add(&sums[threadIdx.x], sums[threadIdx.x], prods[threadIdx.x]);
                }
            }
            // parallel reduction in shared memory
            if (threadsPerRow >= 32 && lane < 16) {
                cuda::mp_add(&sums[threadIdx.x], sums[threadIdx.x], sums[threadIdx.x + 16]);
            }
            if (threadsPerRow >= 16 && lane < 8) {
                cuda::mp_add(&sums[threadIdx.x], sums[threadIdx.x], sums[threadIdx.x + 8]);
            }
            if (threadsPerRow >= 8 && lane < 4) {
                cuda::mp_add(&sums[threadIdx.x], sums[threadIdx.x], sums[threadIdx.x + 4]);
            }
            if (threadsPerRow >= 4 && lane < 2) {
                cuda::mp_add(&sums[threadIdx.x], sums[threadIdx.x], sums[threadIdx.x + 2]);
            }
            if (threadsPerRow >= 2 && lane < 1) {
                cuda::mp_add(&sums[threadIdx.x], sums[threadIdx.x], sums[threadIdx.x + 1]);
            }
            // first thread writes the result
            if (lane == 0) {
                cuda::mp_set(&y[jad.perm[row]], sums[threadIdx.x]);
            }
            row += gridDim.x * blockDim.x / threadsPerRow;
        }
    }

} // namespace cuda

#endif //MPRES_SPMV_JADV_CUH
