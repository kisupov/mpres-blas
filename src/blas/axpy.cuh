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


#ifndef MPRES_AXPY_CUH
#define MPRES_AXPY_CUH

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
    void mp_axpy(const int n, mp_array_t &alpha, mp_array_t &x, const int incx, mp_array_t &y, const int incy, mp_array_t &buffer) {

        // Only positive operation size is permitted for AXPY
        if(n <= 0){
            return;
        }

        //Block size for computing residues. If the corresponding vector stride is not equal to 1, then the block size must be equal to RNS_MODULI_SIZE
        int numThreadsX = (incx == 1) ? BLOCK_SIZE_FOR_RESIDUES : RNS_MODULI_SIZE;
        int numThreadsY = (incy == 1) ? BLOCK_SIZE_FOR_RESIDUES : RNS_MODULI_SIZE;

        //Multiplication - Computing the signs, exponents, and interval evaluations
        mp_vec2scal_mul_esi_kernel<<< gridDim1, blockDim1 >>> (buffer, 1, x, incx, alpha, n);

        //Multiplication - Multiplying the digits in the RNS
        mp_vec2scal_mul_digits_kernel<<< gridDim2, numThreadsX >>> (buffer, 1, x, incx, alpha, n);

        //Multiplication - Rounding the intermediate result
        mp_vector_round_kernel<<< gridDim1, blockDim1 >>> (buffer, 1, n);

        //Addition - Computing the signs, exponents, and interval evaluations
        mp_vector_add_esi_kernel<<< gridDim1, blockDim1 >>> (y, incy, buffer, 1, y, incy, n);

        //Addition - Adding the digits in the RNS
        mp_vector_add_digits_kernel<<< gridDim2, numThreadsY >>> (y, incy, buffer, 1, y, incy, n);

        //Final rounding
        mp_vector_round_kernel<<< gridDim1, blockDim1 >>> (y, incy, n);
    }

} //end of namespace

#endif //MPRES_AXPY_CUH