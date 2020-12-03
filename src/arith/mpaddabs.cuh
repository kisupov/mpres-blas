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
     * Addition of the absolute values of two multiple-precision numbers
     * result = | x | + | y |
     */
    DEVICE_CUDA_FORCEINLINE void mp_add_abs(mp_float_ptr result, mp_float_ptr x, mp_float_ptr y) {
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

        int nzx = ((eval_y[1].frac == 0) || (theta + eval_y[1].exp) < cuda::MP_J);
        int nzy = ((eval_x[1].frac == 0) || (gamma + eval_x[1].exp) < cuda::MP_J);

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

        cuda::er_add_rd(&result->eval[0], &eval_x[0], &eval_y[0]);
        cuda::er_add_ru(&result->eval[1], &eval_x[1], &eval_y[1]);

        result->sign = 0;
        result->exp = (exp_x == 0) ? exp_y : exp_x;

        for (int i = 0; i < RNS_MODULI_SIZE; i++) {
            result->digits[i] = cuda::mod_axby(
                    x->digits[i] * nzx,
                    cuda::RNS_POW2[gamma][i],
                    y->digits[i] * nzy,
                    cuda::RNS_POW2[theta][i],
                    cuda::RNS_MODULI[i],
                    cuda::RNS_MODULI_RECIPROCAL[i]);
        }
        if (result->eval[1].frac != 0 && result->eval[1].exp >= cuda::MP_H) {
            cuda::mp_round(result, cuda::mp_get_rnd_bits(result));
        }
    }

    /*!
     * Addition of the absolute values of two multiple-precision numbers using mp_array_t type for the second argument
     * @param idy - index of the desired element in the vector y
     * @param result - pointer to the computed sum, result = | x | + | y[idy] |
     */
    DEVICE_CUDA_FORCEINLINE void mp_add_abs(mp_float_ptr result, mp_float_ptr x, mp_array_t y, int idy){
        er_float_t eval_x[2];
        er_float_t eval_y[2];
        eval_x[0] = x->eval[0];
        eval_x[1] = x->eval[1];
        eval_y[0] = y.eval[idy];
        eval_y[1] = y.eval[idy + y.len[0]];

        int exp_x = x->exp;
        int exp_y = y.exp[idy];

        int dexp = exp_x - exp_y;
        int gamma =  dexp  * (dexp > 0);
        int theta = -dexp * (dexp < 0);

        int nzx = ((eval_y[1].frac == 0) || (theta + eval_y[1].exp) < cuda::MP_J);
        int nzy = ((eval_x[1].frac == 0) || (gamma + eval_x[1].exp) < cuda::MP_J);

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

        cuda::er_add_rd(&result->eval[0], &eval_x[0], &eval_y[0]);
        cuda::er_add_ru(&result->eval[1], &eval_x[1], &eval_y[1]);

        result->sign = 0;
        result->exp = (exp_x == 0) ? exp_y : exp_x;

        for (int i = 0; i < RNS_MODULI_SIZE; i++) {
            result->digits[i] = cuda::mod_axby(
                    x->digits[i] * nzx,
                    cuda::RNS_POW2[gamma][i],
                    y.digits[RNS_MODULI_SIZE * idy + i] * nzy,
                    cuda::RNS_POW2[theta][i],
                    cuda::RNS_MODULI[i],
                    cuda::RNS_MODULI_RECIPROCAL[i]);
        }
        if (result->eval[1].frac != 0 && result->eval[1].exp >= cuda::MP_H) {
            cuda::mp_round(result, cuda::mp_get_rnd_bits(result));
        }
    }

    /*!
     * Addition of the absolute values of two multiple-precision numbers using the mp_array_t type for the first argument and result
     * @param idx - index of the desired element in the vector x
     * @param idr - index in the result vector to write the computed sum
     * @param result - pointer to the computed sum, result[idr] = | x[idx] | + | y |
     */
    DEVICE_CUDA_FORCEINLINE void mp_add_abs(mp_array_t result, int idr, mp_array_t x, int idx, mp_float_ptr y) {
        int lenr = result.len[0]; //Actual length of the result vector
        er_float_t eval_x[2];
        er_float_t eval_y[2];
        eval_x[0] = x.eval[idx];
        eval_x[1] = x.eval[idx + x.len[0]];
        eval_y[0] = y->eval[0];
        eval_y[1] = y->eval[1];

        int exp_x = x.exp[idx];
        int exp_y = y->exp;

        int dexp = exp_x - exp_y;
        int gamma =  dexp  * (dexp > 0);
        int theta = -dexp * (dexp < 0);

        int nzx = ((eval_y[1].frac == 0) || (theta + eval_y[1].exp) < cuda::MP_J);
        int nzy = ((eval_x[1].frac == 0) || (gamma + eval_x[1].exp) < cuda::MP_J);

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

        cuda::er_add_rd(&result.eval[idr], &eval_x[0], &eval_y[0]);
        cuda::er_add_ru(&result.eval[idr + lenr], &eval_x[1], &eval_y[1]);

        result.sign[idr] = 0;
        result.exp[idr] = (exp_x == 0) ? exp_y : exp_x;

        for (int i = 0; i < RNS_MODULI_SIZE; i++) {
            result.digits[RNS_MODULI_SIZE * idr + i] = cuda::mod_axby(
                    x.digits[RNS_MODULI_SIZE * idx + i] * nzx,
                    cuda::RNS_POW2[gamma][i],
                    y->digits[i] * nzy,
                    cuda::RNS_POW2[theta][i],
                    cuda::RNS_MODULI[i],
                    cuda::RNS_MODULI_RECIPROCAL[i]);
        }
        if (result.eval[idr + lenr].frac != 0 && result.eval[idr + lenr].exp >= cuda::MP_H) {
            #if defined(DEBUG) || defined(_DEBUG)
            if( result.eval[idr + lenr].exp != result.eval[idr].exp ){
                    printf("\n [CUDA WARNING] Possible loss of accuracy");
                }
            #endif
            int bits = result.eval[idr + lenr].exp - cuda::MP_H + 1;
            while (bits > 0) {
                result.exp[idr] += bits;
                cuda::rns_scale2pow(&result.digits[idr * RNS_MODULI_SIZE], &result.digits[idr * RNS_MODULI_SIZE], bits);
                cuda::rns_eval_compute_fast(&result.eval[idr], &result.eval[idr + lenr], &result.digits[idr * RNS_MODULI_SIZE]);
                bits = -1;
            }
        }
    }

} //namespace cuda

#endif //MPRES_MPADDABS_CUH
