/*
 *  Multiple-precision DOT function for GPU (BLAS Level-1)
 *  Dot product of two vectors
 *
 *  Copyright 2018, 2019 by Konstantin Isupov and Alexander Kuvaev.
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


#ifndef MPDOT_CUH
#define MPDOT_CUH

#include "../mparray.cuh"
#include "../mpreduct.cuh"

namespace cuda {

    /*!
     * Computes a vector-vector dot product, result = x0*y0 + x1*y1 + ... + xn*yn
     * @tparam gridDim1 - number of thread blocks used to compute the signs, exponents, interval evaluations, and also to round the result in vector-vector multiplication
     * @tparam blockDim1 - number of threads per block used to compute the signs, exponents, interval evaluations, and also to round the result in vector-vector multiplication
     * @tparam gridDim2 - number of thread blocks used to compute the digits of multiple-precision significands in vector-vector multiplication
     * @tparam gridDim3 - number of thread blocks for parallel summation
     * @tparam blockDim3 - number of threads per block for parallel summation
     * @param n - operation size (must be positive)
     * @param x - pointer to the first vector in the global GPU memory
     * @param incx - storage spacing between elements of x (must be non-zero)
     * @param y - pointer to the second vector in the global GPU memory
     * @param incy - storage spacing between elements of y (must be non-zero)
     * @param result - pointer to the inner product in the GPU memory
     * @param buffer - array of size n in the global GPU memory
     *
     * @warning If either incx or incy is not equal to 1, then BLOCK_SIZE_FOR_RESIDUES must be equal to RNS_MODULI_SIZE
     */
    template <int gridDim1, int blockDim1, int gridDim2, int gridDim3, int blockDim3>
    void mp_array_dot(int n, mp_array_t &x, int incx, mp_array_t &y, int incy, mp_float_ptr result, mp_array_t &buffer) {

        // Only positive operation size is permitted for DOT
        if(n <= 0){
            return;
        }
        //Vector-vector multiplication - Computing the signs, exponents, and interval evaluations
        cuda::mp_array_mul_esi_vv <<< gridDim1, blockDim1 >>> (buffer, 1, x, incx, y, incy, n);

        //Multiplication - Multiplying the digits in the RNS
        cuda::mp_array_mul_digits_vv <<< gridDim2, BLOCK_SIZE_FOR_RESIDUES >>> (buffer, 1, x, incx, y, incy, n);

        //Multiplication - Rounding the intermediate result
        cuda::mp_array_round <<< gridDim1, blockDim1>>> (buffer, 1, n);

        //Two-pass summation
        cuda::mp_array_reduce< gridDim3, blockDim3 >(n, buffer, result);

    }

} //end of namespace

#endif //MPDOT_CUH

