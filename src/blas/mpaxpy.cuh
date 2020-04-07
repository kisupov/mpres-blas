/*
 *  Multiple-precision AXPY function for GPU (BLAS Level-1)
 *  Constant times a vector plus a vector
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


#ifndef MPAXPY_CUH
#define MPAXPY_CUH

#include "../mpvector.cuh"
#include "../kernel_config.cuh"

namespace cuda {

    /*!
     * Computes a vector-scalar product and adds the result to a vector, y = alpha * x + y
     * @tparam gridDim1 - number of thread blocks used to compute the signs, exponents, interval evaluations, and also to round the result
     * @tparam blockDim1 - number of threads per block used to compute the signs, exponents, interval evaluations, and also to round the result
     * @tparam gridDim2 - number of thread blocks used to compute the digits of multiple-precision significands
     * @param n - operation size (must be positive)
     * @param alpha - pointer to the scalar (vector of length one) in the global GPU memory
     * @param x - pointer to the vector in the global GPU memory
     * @param incx - storage spacing between elements of x (must be non-zero)
     * @param y - pointer to the vector in the global GPU memory
     * @param incy - storage spacing between elements of y (must be non-zero)
     * @param buffer - array of size n in the global GPU memory
     */
    template <int gridDim1, int blockDim1, int gridDim2>
    void mpaxpy(int n, mp_array_t &alpha, mp_array_t &x, int incx, mp_array_t &y, int incy, mp_array_t &buffer) {

        // Only positive operation size is permitted for AXPY
        if(n <= 0){
            return;
        }

        //Block size for computing residues. If the corresponding vector stride is not equal to 1, then the block size must be equal to RNS_MODULI_SIZE
        int numThreadsX = (incx == 1) ? BLOCK_SIZE_FOR_RESIDUES : RNS_MODULI_SIZE;
        int numThreadsY = (incy == 1) ? BLOCK_SIZE_FOR_RESIDUES : RNS_MODULI_SIZE;

        //Multiplication - Computing the signs, exponents, and interval evaluations
        mp_array_mul_esi_vs<<< gridDim1, blockDim1 >>> (buffer, 1, x, incx, alpha, n);

        //Multiplication - Multiplying the digits in the RNS
        mp_array_mul_digits_vs<<< gridDim2, numThreadsX >>> (buffer, 1, x, incx, alpha, n);

        //Multiplication - Rounding the intermediate result
        mp_array_round<<< gridDim1, blockDim1 >>> (buffer, 1, n);

        //Addition - Computing the signs, exponents, and interval evaluations
        mp_array_add_esi_vv<<< gridDim1, blockDim1 >>> (y, incy, buffer, 1, y, incy, n);

        //Addition - Adding the digits in the RNS
        mp_array_add_digits_vv<<< gridDim2, numThreadsY >>> (y, incy, buffer, 1, y, incy, n);

        //Final rounding
        mp_array_round<<< gridDim1, blockDim1 >>> (y, incy, n);
    }

} //end of namespace

#endif //MPAXPY_CUH