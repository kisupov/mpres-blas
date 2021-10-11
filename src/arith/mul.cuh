/*
 *  Multiple-precision floating-point multiplication using Residue number system
 *
 *  Copyright 2019, 2020 by Konstantin Isupov and Ivan Babeshko.
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

#ifndef MPRES_MUL_CUH
#define MPRES_MUL_CUH

#include "arith_utils.cuh"

/*!
 * Multiplication of two multiple-precision numbers
 * result = x * y
 */
GCC_FORCEINLINE void mp_mul(mp_float_ptr result, mp_float_t x, mp_float_t y) {
    result->exp = x.exp + y.exp;
    result->sign = x.sign ^ y.sign;
    result->eval[0] = er_md_rd(x.eval[0], y.eval[0], RNS_EVAL_UNIT.upp);
    result->eval[1] = er_md_ru(x.eval[1], y.eval[1], RNS_EVAL_UNIT.low);
    for(int i = 0; i < RNS_MODULI_SIZE; i ++){
        result->digits[i] = mod_mul(x.digits[i], y.digits[i], RNS_MODULI[i]);
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
     * General routine for multiply multiple-precision numbers (result = x * y)
     * The routines below call this procedure
     */
    DEVICE_CUDA_FORCEINLINE void mp_mul_common(int * sr, int * er, er_float_ptr * evlr, int * digr,
                                               int sx, int ex, er_float_t * evlx, const int * digx,
                                               int sy, int ey, er_float_t * evly, const int * digy){
        *er = ex + ey;
        *sr = sx ^ sy;
        *evlr[0] = cuda::er_md_rd(evlx[0], evly[0], cuda::RNS_EVAL_UNIT.upp);
        *evlr[1] = cuda::er_md_ru(evlx[1], evly[1], cuda::RNS_EVAL_UNIT.low);
        cuda::rns_mul(digr, digx, digy);
    }

    /*!
     * Multiplication of two multiple-precision numbers
     * result = x * y
     */
    DEVICE_CUDA_FORCEINLINE void mp_mul(mp_float_ptr result, mp_float_t x, mp_float_t y) {
        er_float_t evalx[2] = { x.eval[0], x.eval[1] };
        er_float_t evaly[2] = { y.eval[0], y.eval[1] };
        er_float_ptr evalr[2] = { &result->eval[0], &result->eval[1] };
        mp_mul_common(&result->sign, &result->exp, evalr, result->digits, x.sign, x.exp, evalx, x.digits, y.sign, y.exp, evaly, y.digits);
        if (result->eval[1].frac != 0 && result->eval[1].exp >= cuda::MP_H) {
            cuda::mp_round(result, cuda::mp_get_rnd_bits(result));
        }
    }

    /*!
     * Multiplication of two multiple-precision numbers using the mp_array_t type for the second argument
     * result = x * y[idy]
     */
    DEVICE_CUDA_FORCEINLINE void mp_mul(mp_float_ptr result, mp_float_t x, mp_array_t y, int idy) {
        er_float_t evalx[2] = { x.eval[0], x.eval[1] };
        er_float_t evaly[2] = { y.eval[idy], y.eval[idy + y.len[0]] };
        er_float_ptr evalr[2] = { &result->eval[0], &result->eval[1] };

        mp_mul_common(&result->sign, &result->exp, evalr, result->digits,
                      x.sign, x.exp, evalx, x.digits,
                      y.sign[idy], y.exp[idy], evaly,&y.digits[RNS_MODULI_SIZE * idy]);

        if (result->eval[1].frac != 0 && result->eval[1].exp >= cuda::MP_H) {
            cuda::mp_round(result, cuda::mp_get_rnd_bits(result));
        }
    }

    /*!
     * Multiplication of two multiple-precision numbers using the mp_array_t type for the arguments
     * result = x[idx] * y[idy]
     */
    DEVICE_CUDA_FORCEINLINE void mp_mul(mp_float_ptr result, mp_array_t x, int idx, mp_array_t y, int idy) {
        er_float_t evalx[2] = { x.eval[idx], x.eval[idx + x.len[0]] };
        er_float_t evaly[2] = { y.eval[idy], y.eval[idy + y.len[0]] };
        er_float_ptr evalr[2] = { &result->eval[0], &result->eval[1] };

        mp_mul_common(&result->sign, &result->exp, evalr, result->digits,
                      x.sign[idx], x.exp[idx], evalx, &x.digits[RNS_MODULI_SIZE * idx],
                      y.sign[idy], y.exp[idy], evaly, &y.digits[RNS_MODULI_SIZE * idy]);

        if (result->eval[1].frac != 0 && result->eval[1].exp >= cuda::MP_H) {
            cuda::mp_round(result, cuda::mp_get_rnd_bits(result));
        }
    }

} //namespace cuda

#endif //MPRES_MUL_CUH
