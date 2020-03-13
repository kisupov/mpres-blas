/*
 *  Multiple-precision SCAL function for GPU  (BLAS Level-1)
 *  Vector-scalar product
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


#ifndef MPSCAL_CUH
#define MPSCAL_CUH

#include "../mparray.cuh"

namespace cuda {

    /*!
     * Computes the product of a vector by a scalar, x = alpha * x
     * @tparam gridDim1 - number of thread blocks used to compute the signs, exponents, interval evaluations, and also to round the result
     * @tparam blockDim1 - number of threads per block used to compute the signs, exponents, interval evaluations, and also to round the result
     * @tparam gridDim2 - number of thread blocks used to compute the digits of multiple-precision significands
     * @param n - operation size (must be positive)
     * @param alpha - pointer to the scalar (vector of length one) in the global GPU memory
     * @param x - pointer to the vector in the global GPU memory
     * @param incx - storage spacing between elements of x (must be positive)
     */
    template <int gridDim1, int blockDim1, int gridDim2>
    void mpscal(int n, mp_array_t &alpha, mp_array_t &x, int incx) {

        //Only positive operation size and vector stride are permitted for SCAL
        if(n <= 0 || incx <= 0){
            return;
        }

        //Block size for computing residues. If incx is not equal to 1, then the block size must be equal to RNS_MODULI_SIZE
        int numThreadsX = (incx == 1) ? BLOCK_SIZE_FOR_RESIDUES : RNS_MODULI_SIZE;

        //Computing the signs, exponents, and interval evaluations
        mp_array_mul_esi_vs<<< gridDim1, blockDim1 >>> (x, incx, x, incx, alpha, n);

        //Multiplying the digits in the RNS
        mp_array_mul_digits_vs<<< gridDim2, numThreadsX >>> (x, incx, x, incx, alpha, n);

        //Rounding the result
        mp_array_round<<< gridDim1, blockDim1 >>> (x, incx, n);
    }

} //end of namespace

#endif //MPSCAL_CUH