/*
 *  Multiple-precision ROT function for GPU (BLAS Level-1)
 *  Apply a plane rotation to vectors
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


#ifndef MPROT_CUH
#define MPROT_CUH

#include "../mparray.cuh"
#include "mpscal.cuh"

namespace cuda {

    /*!
     * Given two vectors x and y, scalars c and s, each element of the vectors is replaced as follows:
     * xi = c*xi + s*yi
     * yi = c*yi - s*xi
     * @tparam gridDim1 - number of thread blocks used to compute the signs, exponents, interval evaluations, and also to round the result
     * @tparam blockDim1 - number of threads per block used to compute the signs, exponents, interval evaluations, and also to round the result
     * @tparam gridDim2 - number of thread blocks used to compute the digits of multiple-precision significands
     * @param n - operation size (must be positive)
     * @param x - pointer to the vector in the global GPU memory
     * @param incx - storage spacing between elements of x (must be non-zero)
     * @param y - pointer to the vector in the global GPU memory
     * @param incy - storage spacing between elements of y (must be non-zero)
     * @param c - pointer to the scalar (vector of length one) in the global GPU memory
     * @param s - pointer to the scalar (vector of length one) in the global GPU memory
     * @param buffer1 - auxiliary array of size n in the global GPU memory
     * @param buffer2 - auxiliary array of size n in the global GPU memory
     */
    template <int gridDim1, int blockDim1, int gridDim2>
    void mp_array_rot(int n, mp_array_t &x, int incx, mp_array_t &y, int incy, mp_array_t &c, mp_array_t &s, mp_array_t &buffer1, mp_array_t &buffer2) {

        // Only positive operation size is permitted for ROT
        if(n <= 0){
            return;
        }

        // Setting the number of threads per block for computing residues
        // If either incx or incy is not equal to 1, then numThreads must be equal to RNS_MODULI_SIZE
        int numThreads = (incx == 1 && incy == 1) ? BLOCK_SIZE_FOR_RESIDUES : RNS_MODULI_SIZE;

        //Multiplication buffer1 = s * x - Computing the signs, exponents, and interval evaluations
        cuda::mp_array_mul_esi_vs<<< gridDim1, blockDim1 >>> (buffer1, 1, x, incx, s, n);

        //Multiplication buffer1 = s * x - Multiplying the digits in the RNS
        cuda::mp_array_mul_digits_vs<<< gridDim2, numThreads >>> (buffer1, 1, x, incx, s, n);

        // Multiplication buffer1 = s * x - Rounding the intermediate result
        mp_array_round<<< gridDim1, blockDim1 >>> (buffer1, 1, n);

        //Multiplication buffer2 = s * y - Computing the signs, exponents, and interval evaluations
        cuda::mp_array_mul_esi_vs<<< gridDim1, blockDim1 >>> (buffer2, 1, y, incy, s, n);

        //Multiplication buffer2 = s * y - Multiplying the digits in the RNS
        cuda::mp_array_mul_digits_vs<<< gridDim2, numThreads >>> (buffer2, 1, y, incy, s, n);

        // Multiplication buffer2 = s * y - Rounding the intermediate result
        mp_array_round<<< gridDim1, blockDim1 >>> (buffer2, 1, n);

        // Computing x = c * x
        cuda::mp_array_scal< gridDim1, blockDim1,gridDim2 >(n, c, x, 1);

        // Computing y = c * y
        cuda::mp_array_scal< gridDim1, blockDim1,gridDim2 >(n, c, y, 1);

        //Addition x = x + buffer2 (in fact, we have x = c*x + s*y) - Computing the signs, exponents, and interval evaluations
        mp_array_add_esi_vv<<< gridDim1, blockDim1 >>> (x, incx, x, incx, buffer2, 1, n);

        //Addition x = x + buffer2 (in fact, we have x = c*x + s*y) - Adding the digits in the RNS
        mp_array_add_digits_vv<<< gridDim2, numThreads >>> (x, incx, x, incx, buffer2, 1, n);

        //Subtraction y = y - buffer1 (in fact, we have y = c*y - s*x) - Computing the signs, exponents, and interval evaluations
        mp_array_sub_esi_vv<<< gridDim1, blockDim1 >>> (y, incy, y, incy, buffer1, 1, n);

        //Subtraction y = y - buffer1 (in fact, we have y = c*y - s*x) - Adding the digits in the RNS
        //We call mp_array_add_digits_vv since the sign has been changed in mp_array_sub_esi_vv
        mp_array_add_digits_vv<<< gridDim2, numThreads >>> (y, incy, y, incy, buffer1, 1, n);

        //Final rounding
        mp_array_round<<< gridDim1, blockDim1 >>> (x, incx, n);
        mp_array_round<<< gridDim1, blockDim1 >>> (y, incy, n);
    }

} //end of namespace

#endif //MPROT_CUH