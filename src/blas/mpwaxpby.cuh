/*
 *  Multiple-precision WAXPBY function for GPU (updated set of BLAS Level-1)
 *  Scaled vector addition: w = alpha * x + beta * y
 *  Details: http://www.netlib.org/blas/blast-forum/
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


#ifndef MPWAXPBY_CUH
#define MPWAXPBY_CUH

#include "../mparray.cuh"

namespace cuda {

    /*!
     * Scales the vector x by alpha and the vector y by beta, add these two vectors to one another
     * and stores the result in the vector w, w = alpha * x + beta * y
     * @tparam gridDim1 - number of thread blocks used to compute the signs, exponents, interval evaluations, and also to round the result
     * @tparam blockDim1 - number of threads per block used to compute the signs, exponents, interval evaluations, and also to round the result
     * @tparam gridDim2 - number of thread blocks used to compute the digits of multiple-precision significands
     * @param n - operation size (must be positive)
     * @param alpha - pointer to the scalar (vector of length one) in the global GPU memory
     * @param x - pointer to the vector in the global GPU memory
     * @param incx - storage spacing between elements of x (must be non-zero)
     * @param beta - pointer to the scalar (vector of length one) in the global GPU memory
     * @param y - pointer to the vector in the global GPU memory
     * @param incy - storage spacing between elements of y (must be non-zero)
     * @param w - pointer to the result vector in the global GPU memory
     * @param incw - storage spacing between elements of w (must be non-zero)
     * @param buffer - array of size n in the global GPU memory
     */
    template <int gridDim1, int blockDim1, int gridDim2>
    void mp_array_waxpby(int n, mp_array_t &alpha, mp_array_t &x, int incx, mp_array_t &beta, mp_array_t &y, int incy, mp_array_t &w, int incw, mp_array_t &buffer) {

        // Only positive operation size is permitted for AXPY
        if(n <= 0){
            return;
        }

        //Block size for computing residues. If the corresponding vector stride is not equal to 1, then the block size must be equal to RNS_MODULI_SIZE
        int numThreadsX = (incx == 1) ? BLOCK_SIZE_FOR_RESIDUES : RNS_MODULI_SIZE;
        int numThreadsW = (incw == 1) ? BLOCK_SIZE_FOR_RESIDUES : RNS_MODULI_SIZE;
        int numThreadsYW = (incy == 1 && incw == 1) ? BLOCK_SIZE_FOR_RESIDUES : RNS_MODULI_SIZE;

        // Multiplication buffer = alpha *x - Computing the signs, exponents, and interval evaluations
        mp_array_mul_esi_vs<<< gridDim1, blockDim1 >>> (buffer, 1, x, incx, alpha, n);

       // Multiplication buffer = alpha *x - Multiplying the digits in the RNS
        mp_array_mul_digits_vs<<< gridDim2, numThreadsX >>> (buffer, 1, x, incx, alpha, n);

        // Multiplication buffer = alpha *x - Rounding the intermediate result
        mp_array_round<<< gridDim1, blockDim1 >>> (buffer, 1, n);

        // Multiplication w = beta * y - Computing the signs, exponents, and interval evaluations
        mp_array_mul_esi_vs<<< gridDim1, blockDim1 >>> (w, incw, y, incy, beta, n);

        // Multiplication w = beta * y - Multiplying the digits in the RNS
        mp_array_mul_digits_vs<<< gridDim2, numThreadsYW >>> (w, incw, y, incy, beta, n);

        // Multiplication w = beta * y - Rounding the intermediate result
        mp_array_round<<< gridDim1, blockDim1 >>> (w, incw, n);

        // Addition w = w + buffer - Computing the signs, exponents, and interval evaluations
        mp_array_add_esi_vv<<< gridDim1, blockDim1 >>> (w, incw, w, incw, buffer, 1, n);

        // Addition w = w + buffer - Adding the digits in the RNS
        mp_array_add_digits_vv<<< gridDim2, numThreadsW >>> (w, incw, w, incw, buffer, 1, n);

        // Final rounding
        mp_array_round<<< gridDim1, blockDim1 >>> (w, incw, n);
    }

} //end of namespace

#endif //MPWAXPBY_CUH