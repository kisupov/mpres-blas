/*
 *  Multiple-precision floating-point subtraction using Residue number system
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

#ifndef MPRES_SUB_CUH
#define MPRES_SUB_CUH

#include "add.cuh"

/*!
 * Subtraction of two multiple-precision numbers
 * result = x - y
 */
GCC_FORCEINLINE void mp_sub(mp_float_ptr result, mp_float_t x, mp_float_t y) {
   y.sign = y.sign^1;
   mp_add(result, x, y);
}

/*
* GPU functions
*/
namespace cuda {
    /*!
     * Subtraction of two multiple-precision numbers
     * result = x - y
     */
    DEVICE_CUDA_FORCEINLINE void mp_sub(mp_float_ptr result, mp_float_t x, mp_float_t y) {
        er_float_t evalx[2] = { x.eval[0], x.eval[1] };
        er_float_t evaly[2] = { y.eval[0], y.eval[1] };
        er_float_ptr evalr[2] = { &result->eval[0], &result->eval[1] };
        mp_add_common(&result->sign, &result->exp, evalr, result->digits,
                      x.sign, x.exp, evalx, x.digits,
                      y.sign^1, y.exp, evaly, y.digits); //Sign inversion is performed
        if (result->eval[1].frac != 0 && result->eval[1].exp >= cuda::MP_H) {
            cuda::mp_round(result, cuda::mp_get_rnd_bits(result));
        }
    }
}

#endif //MPRES_SUB_CUH
