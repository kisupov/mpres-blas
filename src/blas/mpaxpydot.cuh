/*
 *  Multiple-precision AXPY_DOT function for GPU (BLAS Level-1)
 *  Combined AXPY and DOT
 *  Details: http://www.netlib.org/blas/blast-forum/
 *
 *  Copyright 2019 by Konstantin Isupov and Alexander Kuvaev.
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


#ifndef MPAXPYDOT_CUH
#define MPAXPYDOT_CUH

#include "mpaxpy.cuh"
#include "mpdot.cuh"

namespace cuda {

    /*!
     * The routine combines an axpy and a dot product. w is decremented by a multiple of v.
     * A dot product is then computed with the decremented w:
     * w = w - alpha * v
     * r = w^T * u
     * @tparam gridDim1 - number of thread blocks used to compute the signs, exponents, interval evaluations, and also to round the result in scalar-vector and vector-vector operations
     * @tparam blockDim1 - number of threads per block used to compute the signs, exponents, interval evaluations, and also to round the result in scalar-vector and vector-vector operations
     * @tparam gridDim2 - number of thread blocks used to compute the digits of multiple-precision significands in scalar-vector and vector-vector operations
     * @tparam gridDim3 - number of thread blocks for parallel summation
     * @tparam blockDim3 - number of threads per block for parallel summation
     * @param n - operation size (must be positive)
     * @param alpha - pointer to the scalar (vector of length one) in the global GPU memory
     * @param w - pointer to the vector in the global GPU memory
     * @param incw - storage spacing between elements of w (must be non-zero)
     * @param v - pointer to the vector in the global GPU memory
     * @param incv - storage spacing between elements of v (must be non-zero)
     * @param u - pointer to the vector in the global GPU memory
     * @param incu - storage spacing between elements of u (must be non-zero)
     * @param r - pointer to the inner product in the GPU memory
     * @param buffer - array of size n in the global GPU memory
     *
     * @warning If either incx or incy is not equal to 1, then BLOCK_SIZE_FOR_RESIDUES must be equal to RNS_MODULI_SIZE
     */
    template <int gridDim1, int blockDim1, int gridDim2, int gridDim3, int blockDim3>
    void mp_array_axpy_dot(int n, mp_array_t &alpha, mp_array_t &w, int incw, mp_array_t &v, int incv, mp_array_t &u, int incu, mp_float_ptr r, mp_array_t &buffer) {

        // Only positive operation size is permitted for AXPY_DOT
        if(n <= 0){
            return;
        }

        cuda::mp_array_axpy< gridDim1, blockDim1, gridDim2 >(n, alpha, v, incv, w, incw, buffer);
        cuda::mp_array_dot< gridDim1, blockDim1, gridDim2, gridDim3, blockDim3 >(n, u, incu, w, incw, r, buffer);
    }

} //end of namespace

#endif //MPAXPYDOT_CUH