/*
 *  Multiple-precision floating-point inverse square root.
 *  In the current version, we rely on MPFR to compute the square root.
 *  In the future, we will try to implement native algorithms using Newton's iteration.
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

#ifndef MPRES_INVSQRT_CUH
#define MPRES_INVSQRT_CUH

#include "assign.cuh"

/*!
 * Inverse square root of a multiple-precision number
 * result = 1 / sqrt(x)
 */
GCC_FORCEINLINE void mp_inv_sqrt(mp_float_ptr result, mp_float_ptr x) {
    mpfr_t op;
    mpfr_init2(op, MP_PRECISION);
    mp_get_mpfr(op, x);
    mpfr_rec_sqrt(op, op, MPFR_RNDN);
    mp_set_mpfr(result, op);
    mpfr_clear(op);
}

/*
* GPU functions
*/
namespace cuda {
    /*!
     * Inverse square root of a multiple-precision number
     * result = 1 / sqrt(x)
     * Note that this is a HOST procedure. Loading data from the GPU to RAM, dividing on the CPU using MPFR, and loading the results back to the GPU is performed.
     */
    GCC_FORCEINLINE void mp_inv_sqrt(mp_float_ptr result, mp_float_ptr x) {
        auto op = new mp_float_t[1];
        cudaMemcpy(op, x, sizeof(mp_float_t), cudaMemcpyDeviceToHost);
        ::mp_inv_sqrt(op, op);
        cudaMemcpy(result, op, sizeof(mp_float_t), cudaMemcpyHostToDevice);
        delete[] op;
    }
}

#endif //MPRES_INVSQRT_CUH
