/*
 *  Multiple-precision floating-point comparison of absolute values using Residue number system
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

#ifndef MPRES_CMPABS_CUH
#define MPRES_CMPABS_CUH

#include "arith_utils.cuh"

/*!
 * Comparison of the absolute values of x and y
 * Returns 1 if |x| > |y|, -1 if |x| < |y|, and 0 otherwise
 */
GCC_FORCEINLINE int mp_cmp_abs(mp_float_t x, mp_float_t y) {
    int digitx[RNS_MODULI_SIZE];
    int digity[RNS_MODULI_SIZE];

    //Exponent alignment
    int dexp = x.exp - y.exp;
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

    for (int i = 0; i < RNS_MODULI_SIZE; i++) {
        digitx[i] = mod_mul(x.digits[i], RNS_POW2[gamma][i] * nzx, RNS_MODULI[i]);
        digity[i] = mod_mul(y.digits[i], RNS_POW2[theta][i] * nzy, RNS_MODULI[i]);
    }
    //RNS magnitude comparison
    return rns_cmp(digitx, x.eval[0], x.eval[1], digity, y.eval[0], y.eval[1]);
}

/*
 * GPU functions
 */
namespace cuda {

    /*!
     * General routine for comparing multiple-precision numbers
     * The routines below call this procedure
     */
    DEVICE_CUDA_FORCEINLINE int mp_cmp_abs_common(int ex, er_float_t * evalx, const int * digx,
                                                  int ey, er_float_t * evaly, const int * digy)
    {
        int digitx[RNS_MODULI_SIZE];
        int digity[RNS_MODULI_SIZE];

        //Exponent alignment
        int dexp = ex - ey;
        int gamma =  dexp  * (dexp > 0);
        int theta = -dexp * (dexp < 0);
        unsigned char nzx = ((evaly[1].frac == 0) || (theta + evaly[1].exp) < cuda::MP_J);
        unsigned char nzy = ((evalx[1].frac == 0) || (gamma + evalx[1].exp) < cuda::MP_J);

        gamma = gamma * nzy;
        theta = theta * nzx;

        evalx[0].exp += gamma;
        evalx[1].exp += gamma;
        evaly[0].exp += theta;
        evaly[1].exp += theta;

        evalx[0].frac *= nzx;
        evalx[1].frac *= nzx;
        evaly[0].frac *= nzy;
        evaly[1].frac *= nzy;

        cuda::rns_mul_c(digitx, digx, cuda::RNS_POW2[gamma], nzx);
        cuda::rns_mul_c(digity, digy, cuda::RNS_POW2[theta], nzy);
        /*for (int i = 0; i < RNS_MODULI_SIZE; i++) {
            digitx[i] = cuda::mod_mul(digx[i], cuda::RNS_POW2[gamma][i] * nzx, moduli[i]);
            digity[i] = cuda::mod_mul(digy[i], cuda::RNS_POW2[theta][i] * nzy, moduli[i]);
        }*/
        //RNS magnitude comparison
        return cuda::rns_cmp(digitx, evalx[0], evalx[1], digity, evaly[0], evaly[1]);
    }

    /*!
     * Comparison of the absolute values of x and y
     * Returns 1 if |x| > |y|, -1 if |x| < |y|, and 0 otherwise
     */
    DEVICE_CUDA_FORCEINLINE int mp_cmp_abs(mp_float_t x, mp_float_t y) {
        er_float_t evalx[2] = { x.eval[0], x.eval[1] };
        er_float_t evaly[2] = { y.eval[0], y.eval[1] };
        return mp_cmp_abs_common(x.exp, evalx, x.digits, y.exp, evaly, y.digits);
    }

    /*!
     * Comparison of the absolute values of x and y using the mp_array_t type for the first argument
     * Returns 1 if |x[idx]| > |y|, -1 if |x[idx]| < |y|, and 0 otherwise
     */
    DEVICE_CUDA_FORCEINLINE int mp_cmp_abs(mp_array_t x, int idx, mp_float_t y) {
        er_float_t evalx[2] = { x.eval[idx], x.eval[idx + x.len[0]] };
        er_float_t evaly[2] = { y.eval[0], y.eval[1] };
        return mp_cmp_abs_common(x.exp[idx], evalx, &x.digits[RNS_MODULI_SIZE * idx], y.exp, evaly, y.digits);
    }

} //namespace cuda

#endif //MPRES_CMPABS_CUH
