/*
 *  Multiple-precision floating-point addition of absolute values using Residue number system
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

#ifndef MPRES_ADDABS_CUH
#define MPRES_ADDABS_CUH

#include "arith_utils.cuh"

/*!
 * Addition of the absolute values of two multiple-precision numbers
 * result = | x | + | y |
 */
GCC_FORCEINLINE void mp_add_abs(mp_float_ptr result, mp_float_t x, mp_float_t y) {
    int ex = x.exp;
    int ey = y.exp;
    int dexp = ex - ey;
    int gamma =  dexp  * (dexp > 0);
    int theta = -dexp * (dexp < 0);

    unsigned char nzx = ((y.eval[1].frac == 0) || (theta + y.eval[1].exp) < MP_J);
    unsigned char nzy = ((x.eval[1].frac == 0) || (gamma + x.eval[1].exp) < MP_J);

    gamma = gamma * nzy;
    theta = theta * nzx;

    x.eval[0].exp += gamma;
    x.eval[1].exp += gamma;
    y.eval[0].exp += theta;
    y.eval[1].exp += theta;

    x.eval[0].frac *= nzx;
    x.eval[1].frac *= nzx;
    y.eval[0].frac *= nzy;
    y.eval[1].frac *= nzy;

    ex = (ex - gamma) * nzx;
    ey = (ey - theta) * nzy;

    result->eval[0] = er_add_rd(x.eval[0], y.eval[0]);
    result->eval[1] = er_add_ru(x.eval[1], y.eval[1]);

    result->sign = 0;
    result->exp = (ex == 0) ? ey : ex;

    for (int i = 0; i < RNS_MODULI_SIZE; i++) {
        result->digits[i] = mod_axby(x.digits[i], RNS_POW2[gamma][i] * nzx, y.digits[i], RNS_POW2[theta][i] * nzy, RNS_MODULI[i]);
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
     * General routine for adding the absolute values of multiple-precision numbers (result = x + y)
     * result = | x | + | y |
     */
    DEVICE_CUDA_FORCEINLINE void mp_add_abs_common(int * er, er_float_ptr * evlr, int * digr,
                                               int ex, er_float_t * evalx, const int * digx,
                                               int ey, er_float_t * evaly, const int * digy)
    {
        int dexp = ex - ey;
        int gamma =  dexp  * (dexp > 0);
        int theta = -dexp * (dexp < 0);

        const int nzx = ((evaly[1].frac == 0) || (theta + evaly[1].exp) < cuda::MP_J);
        const int nzy = ((evalx[1].frac == 0) || (gamma + evalx[1].exp) < cuda::MP_J);

        gamma = gamma * nzy;
        theta = theta * nzx;

        ex = (ex - gamma) * nzx;
        ey = (ey - theta) * nzy;
        *er = (ex == 0) ? ey : ex;

        evalx[0].exp += gamma;
        evalx[1].exp += gamma;
        evaly[0].exp += theta;
        evaly[1].exp += theta;

        evalx[0].frac *= nzx;
        evalx[1].frac *= nzx;
        evaly[0].frac *= nzy;
        evaly[1].frac *= nzy;

        *evlr[0] = cuda::er_add_rd(evalx[0], evaly[0]);
        *evlr[1] = cuda::er_add_ru(evalx[1], evaly[1]);
        cuda::rns_axby_cd(digr, cuda::RNS_POW2[gamma], digx, nzx, cuda::RNS_POW2[theta], digy, nzy);
    }

    /*!
     * Addition of the absolute values of two multiple-precision numbers
     * result = | x | + | y |
     */
    DEVICE_CUDA_FORCEINLINE void mp_add_abs(mp_float_ptr result, mp_float_t x, mp_float_t y) {
        er_float_t evalx[2] = { x.eval[0], x.eval[1]}; //Array of pointers to interval evaluations
        er_float_t evaly[2] = { y.eval[0], y.eval[1]};
        er_float_ptr evalr[2] = {&result->eval[0], &result->eval[1]};
        result->sign = 0;
        mp_add_abs_common(&result->exp, evalr, result->digits, x.exp, evalx, x.digits, y.exp, evaly, y.digits);
        if (result->eval[1].frac != 0 && result->eval[1].exp >= cuda::MP_H) {
            cuda::mp_round(result, cuda::mp_get_rnd_bits(result));
        }
    }

    /*!
     * Addition of the absolute values of two multiple-precision numbers using mp_array_t type for the second argument
     * @param idy - index of the desired element in the vector y
     * @param result - pointer to the computed sum, result = | x | + | y[idy] |
     */
    DEVICE_CUDA_FORCEINLINE void mp_add_abs(mp_float_ptr result, mp_float_t x, mp_array_t y, int idy) {
        er_float_t evalx[2] = { x.eval[0], x.eval[1] };
        er_float_t evaly[2] = { y.eval[idy], y.eval[idy + y.len[0]] };
        er_float_ptr evalr[2] = { &result->eval[0], &result->eval[1] };
        result->sign = 0;
        mp_add_abs_common(&result->exp, evalr, result->digits,
                      x.exp, evalx, x.digits,
                      y.exp[idy], evaly,&y.digits[RNS_MODULI_SIZE * idy]);
        if (result->eval[1].frac != 0 && result->eval[1].exp >= cuda::MP_H) {
            cuda::mp_round(result, cuda::mp_get_rnd_bits(result));
        }
    }

} //namespace cuda

#endif //MPRES_ADDABS_CUH
