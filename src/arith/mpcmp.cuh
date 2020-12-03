/*
 *  Multiple-precision floating-point comparison using Residue number system
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

#ifndef MPRES_MPCMP_CUH
#define MPRES_MPCMP_CUH

#include "mpcommon.cuh"

/*!
 * Comparison of x and y
 * Returns 1 if x > y, -1 if x < y, and 0 otherwise
 */
GCC_FORCEINLINE int mp_cmp(mp_float_ptr x, mp_float_ptr y) {
    int sign_x = x->sign;
    int sign_y = y->sign;
    int digits_x[RNS_MODULI_SIZE];
    int digits_y[RNS_MODULI_SIZE];
    er_float_t eval_x[2];
    er_float_t eval_y[2];
    eval_x[0] = x->eval[0];
    eval_x[1] = x->eval[1];
    eval_y[0] = y->eval[0];
    eval_y[1] = y->eval[1];

    //Exponent alignment
    int dexp = x->exp - y->exp;
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

    for (int i = 0; i < RNS_MODULI_SIZE; i++) {
        digits_x[i] = mod_mul(x->digits[i], RNS_POW2[gamma][i] * nzx, RNS_MODULI[i]);
        digits_y[i] = mod_mul(y->digits[i], RNS_POW2[theta][i] * nzy, RNS_MODULI[i]);
    }
    //RNS magnitude comparison
    int cmp = rns_cmp(digits_x, &eval_x[0], &eval_x[1], digits_y, &eval_y[0], & eval_y[1]);
    int greater = (sign_x == 0 && sign_y == 1) || (sign_x == 0 && sign_y == 0 && cmp == 1) || (sign_x == 1 && sign_y == 1 && cmp == -1); // x > y
    int less = (sign_x == 1 && sign_y == 0) || (sign_x == 0 && sign_y == 0 && cmp == -1) || (sign_x == 1 && sign_y== 1 && cmp == 1); // x < y
    return greater ? 1 : less ? -1 : 0;
}

/*
 * GPU functions
 */
namespace cuda {

    /*!
     * Comparison of x and y
     * Returns 1 if x > y, -1 if x < y, and 0 otherwise
     */
    DEVICE_CUDA_FORCEINLINE int mp_cmp(mp_float_ptr x, mp_float_ptr y) {
        int sign_x = x->sign;
        int sign_y = y->sign;
        int digits_x[RNS_MODULI_SIZE];
        int digits_y[RNS_MODULI_SIZE];
        er_float_t eval_x[2];
        er_float_t eval_y[2];
        eval_x[0] = x->eval[0];
        eval_x[1] = x->eval[1];
        eval_y[0] = y->eval[0];
        eval_y[1] = y->eval[1];

        //Exponent alignment
        int dexp = x->exp - y->exp;
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

        for (int i = 0; i < RNS_MODULI_SIZE; i++) {
            digits_x[i] = cuda::mod_mul(x->digits[i], cuda::RNS_POW2[gamma][i] * nzx, cuda::RNS_MODULI[i]);
            digits_y[i] = cuda::mod_mul(y->digits[i], cuda::RNS_POW2[theta][i] * nzy, cuda::RNS_MODULI[i]);
        }
        //RNS magnitude comparison
        int cmp = cuda::rns_cmp(digits_x, &eval_x[0], &eval_x[1], digits_y, &eval_y[0], & eval_y[1]);
        int greater = (sign_x == 0 && sign_y == 1) || (sign_x == 0 && sign_y == 0 && cmp == 1) || (sign_x == 1 && sign_y == 1 && cmp == -1); // x > y
        int less = (sign_x == 1 && sign_y == 0) || (sign_x == 0 && sign_y == 0 && cmp == -1) || (sign_x == 1 && sign_y== 1 && cmp == 1); // x < y
        return greater ? 1 : less ? -1 : 0;
    }

    /*!
     * Comparison of x and y using the mp_array_t type for the first argument
     * Returns 1 if x[idx] > y, -1 if x[idx] < y, and 0 otherwise
     */
    DEVICE_CUDA_FORCEINLINE int mp_cmp(mp_array_t x, int idx, mp_float_ptr y) {
        int sign_x = x.sign[idx];
        int sign_y = y->sign;
        int digits_x[RNS_MODULI_SIZE];
        int digits_y[RNS_MODULI_SIZE];
        er_float_t eval_x[2];
        er_float_t eval_y[2];
        eval_x[0] = x.eval[idx];
        eval_x[1] = x.eval[idx + x.len[0]];
        eval_y[0] = y->eval[0];
        eval_y[1] = y->eval[1];

        //Exponent alignment
        int dexp = x.exp[idx] - y->exp;
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

        for (int i = 0; i < RNS_MODULI_SIZE; i++) {
            digits_x[i] = cuda::mod_mul(x.digits[RNS_MODULI_SIZE * idx + i], cuda::RNS_POW2[gamma][i] * nzx, cuda::RNS_MODULI[i]);
            digits_y[i] = cuda::mod_mul(y->digits[i], cuda::RNS_POW2[theta][i] * nzy, cuda::RNS_MODULI[i]);
        }
        //RNS magnitude comparison
        int cmp = cuda::rns_cmp(digits_x, &eval_x[0], &eval_x[1], digits_y, &eval_y[0], & eval_y[1]);
        int greater = (sign_x == 0 && sign_y == 1) || (sign_x == 0 && sign_y == 0 && cmp == 1) || (sign_x == 1 && sign_y == 1 && cmp == -1); // x > y
        int less = (sign_x == 1 && sign_y == 0) || (sign_x == 0 && sign_y == 0 && cmp == -1) || (sign_x == 1 && sign_y== 1 && cmp == 1); // x < y
        return greater ? 1 : less ? -1 : 0;
    }

} //namespace cuda

#endif //MPRES_MPCMP_CUH
