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

#ifndef MPRES_MPADDABS_CUH
#define MPRES_MPADDABS_CUH

#include "mpcommon.cuh"

/*!
 * Addition of the absolute values of two multiple-precision numbers
 * result = | x | + | y |
 */
GCC_FORCEINLINE void mp_add_abs(mp_float_ptr result, mp_float_ptr x, mp_float_ptr y) {
    er_float_t eval_x[2];
    er_float_t eval_y[2];
    eval_x[0] = x->eval[0];
    eval_x[1] = x->eval[1];
    eval_y[0] = y->eval[0];
    eval_y[1] = y->eval[1];

    int exp_x = x->exp;
    int exp_y = y->exp;

    int dexp = exp_x - exp_y;
    int gamma =  dexp  * (dexp > 0);
    int theta = -dexp * (dexp < 0);

    int nzx = ((eval_y[1].frac == 0) || (theta + eval_y[1].exp) < MP_J);
    int nzy = ((eval_x[1].frac == 0) || (gamma + eval_x[1].exp) < MP_J);

    gamma = gamma * nzy;
    theta = theta * nzx;

    eval_x[0].exp += gamma;
    eval_x[1].exp += gamma;
    eval_y[0].exp += theta;
    eval_y[1].exp += theta;

    eval_x[0].frac *= nzx;
    eval_x[1].frac *= nzx;
    eval_y[0].frac *= nzy;
    eval_y[1].frac *= nzy;

    exp_x = (exp_x - gamma) * nzx;
    exp_y = (exp_y - theta) * nzy;

    er_add_rd(&result->eval[0], &eval_x[0], &eval_y[0]);
    er_add_ru(&result->eval[1], &eval_x[1], &eval_y[1]);

    result->sign = 0;
    result->exp = (exp_x == 0) ? exp_y : exp_x;

    for (int i = 0; i < RNS_MODULI_SIZE; i++) {
        result->digits[i] = mod_axby(x->digits[i], RNS_POW2[gamma][i] * nzx, y->digits[i], RNS_POW2[theta][i] * nzy, RNS_MODULI[i]);
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
                                               int ex, er_float_ptr * evlx, const int * digx,
                                               int ey, er_float_ptr * evly, const int * digy)
    {
        er_float_t evalx[2];
        er_float_t evaly[2];
        evalx[0] = *evlx[0];
        evalx[1] = *evlx[1];
        evaly[0] = *evly[0];
        evaly[1] = *evly[1];

        int dexp = ex - ey;
        int gamma =  dexp  * (dexp > 0);
        int theta = -dexp * (dexp < 0);

        unsigned char  nzx = ((evaly[1].frac == 0) || (theta + evaly[1].exp) < cuda::MP_J);
        unsigned char  nzy = ((evalx[1].frac == 0) || (gamma + evalx[1].exp) < cuda::MP_J);

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

        cuda::er_add_rd(evlr[0], &evalx[0], &evaly[0]);
        cuda::er_add_ru(evlr[1], &evalx[1], &evaly[1]);

        for (int i = 0; i < RNS_MODULI_SIZE; i++) {
            digr[i] = cuda::mod_axby(
                    digx[i] * nzx, cuda::RNS_POW2[gamma][i],
                    digy[i] * nzy, cuda::RNS_POW2[theta][i],
                    cuda::RNS_MODULI[i],
                    cuda::RNS_MODULI_RECIPROCAL[i]);
        }
    }

    /*!
     * Addition of the absolute values of two multiple-precision numbers
     * result = | x | + | y |
     */
    DEVICE_CUDA_FORCEINLINE void mp_add_abs(mp_float_ptr result, mp_float_ptr x, mp_float_ptr y) {
        er_float_ptr evalx[2] = {&x->eval[0], &x->eval[1]}; //Array of pointers to interval evaluations
        er_float_ptr evaly[2] = {&y->eval[0], &y->eval[1]};
        er_float_ptr evalr[2] = {&result->eval[0], &result->eval[1]};
        result->sign = 0;

        mp_add_abs_common(&result->exp, evalr, result->digits,
                      x->exp, evalx, x->digits,
                      y->exp, evaly, y->digits);

        if (result->eval[1].frac != 0 && result->eval[1].exp >= cuda::MP_H) {
            cuda::mp_round(result, cuda::mp_get_rnd_bits(result));
        }
    }

    /*!
     * Addition of the absolute values of two multiple-precision numbers using mp_array_t type for the second argument
     * @param idy - index of the desired element in the vector y
     * @param result - pointer to the computed sum, result = | x | + | y[idy] |
     */
    DEVICE_CUDA_FORCEINLINE void mp_add_abs(mp_float_ptr result, mp_float_ptr x, mp_array_t y, int idy) {
        er_float_ptr evalx[2] = { &x->eval[0], &x->eval[1] };
        er_float_ptr evaly[2] = { &y.eval[idy], &y.eval[idy + y.len[0]] };
        er_float_ptr evalr[2] = { &result->eval[0], &result->eval[1] };
        result->sign = 0;

        mp_add_abs_common(&result->exp, evalr, result->digits,
                      x->exp, evalx, x->digits,
                      y.exp[idy], evaly,&y.digits[RNS_MODULI_SIZE * idy]);

        if (result->eval[1].frac != 0 && result->eval[1].exp >= cuda::MP_H) {
            cuda::mp_round(result, cuda::mp_get_rnd_bits(result));
        }
    }

} //namespace cuda

#endif //MPRES_MPADDABS_CUH
