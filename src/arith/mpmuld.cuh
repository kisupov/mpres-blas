/*
 *  Multiplication of a multiple precision number and a double precision number
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

#ifndef MPRES_MPMULD_CUH
#define MPRES_MPMULD_CUH

#include "mpcommon.cuh"

/*!
 * Multiplication of a multiple-precision variable x by a double-precision variable y
 * result = x * y
 */
GCC_FORCEINLINE void mp_mul_d(mp_float_ptr result, mp_float_ptr x, const double y) {
    long significand = dbl_get_significand(y);
    result->exp = x->exp + dbl_get_exp(y);
    result->sign = x->sign ^ dbl_get_sign(y);
    result->eval[0].exp = x->eval[0].exp;
    result->eval[1].exp = x->eval[1].exp;
    result->eval[0].frac = dmul_rd(x->eval[0].frac, (double) significand);
    result->eval[1].frac = dmul_ru(x->eval[1].frac, (double) significand);
    er_adjust(&result->eval[0]);
    er_adjust(&result->eval[1]);
    for(int i = 0; i < RNS_MODULI_SIZE; i ++){
        result->digits[i] = mod_mul(x->digits[i], (significand % RNS_MODULI[i]), RNS_MODULI[i]);
    }
    if (result->eval[1].frac != 0 && result->eval[1].exp >= MP_H) {
        mp_round(result, mp_get_rnd_bits(result));
    }
}


/*
 * GPU functions
 */
namespace cuda {

    /*!
     * Multiplication of a multiple-precision variable x by a double-precision variable y
     * result = x * y
     */
    DEVICE_CUDA_FORCEINLINE void mp_mul_d(mp_float_ptr result, mp_float_ptr x, const double y) {
        long significand = cuda::dbl_get_significand(y);
        result->exp = x->exp + cuda::dbl_get_exp(y);
        result->sign = x->sign ^ cuda::dbl_get_sign(y);
        result->eval[0].exp = x->eval[0].exp;
        result->eval[1].exp = x->eval[1].exp;
        result->eval[0].frac = __dmul_rd(x->eval[0].frac, (double) significand);
        result->eval[1].frac = __dmul_ru(x->eval[1].frac, (double) significand);
        cuda::er_adjust(&result->eval[0]);
        cuda::er_adjust(&result->eval[1]);
        cuda::rns_mul_l(result->digits, x->digits, significand);
        if (result->eval[1].frac != 0 && result->eval[1].exp >= cuda::MP_H) {
            cuda::mp_round(result, cuda::mp_get_rnd_bits(result));
        }
    }


} //namespace cuda

#endif //MPRES_MPMULD_CUH
