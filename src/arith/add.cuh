/*
 *  Multiple-precision floating-point addition using Residue number system
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

#ifndef MPRES_ADD_CUH
#define MPRES_ADD_CUH

#include "arith_utils.cuh"

/*!
 * Addition of two multiple-precision numbers
 * result = x + y
 */
GCC_FORCEINLINE void mp_add(mp_float_ptr result, mp_float_ptr x, mp_float_ptr y) {

    //The local copies of the terms are required since the exponent alignment can be performed
    er_float_t evalx[2];
    er_float_t evaly[2];
    evalx[0] = x->eval[0];
    evalx[1] = x->eval[1];
    evaly[0] = y->eval[0];
    evaly[1] = y->eval[1];

    int ex = x->exp;
    int ey = y->exp;
    int sx = x->sign;
    int sy = y->sign;

    //Exponent alignment
    int dexp = ex - ey;
    int gamma =  dexp * (dexp > 0); //if dexp > 0, then gamma =  dexp; otherwise gamma = 0
    int theta = -dexp * (dexp < 0); //if dexp < 0, then theta = -dexp; otherwise theta = 0

    unsigned char nzx = ((evaly[1].frac == 0) || (theta + evaly[1].exp) < MP_J); //nzx (u) = 1 if x not need be zeroed; otherwise nzx = 0
    unsigned char nzy = ((evalx[1].frac == 0) || (gamma + evalx[1].exp) < MP_J); //nzy (v) = 1 if y not need be zeroed; otherwise nzy = 0

    gamma = gamma * nzy; //if nzy = 0 (y needs to be zeroed), then gamma = 0, i.e. we will multiply x by 2^0 without actually changing the value of x
    theta = theta * nzx; //if nzx = 0 (x needs to be zeroed), then theta = 0, i.e. we will multiply y by 2^0 without actually changing the value of y

    //Correction of the exponents
    ex = (ex - gamma) * nzx; //if x needs to be zeroed, ex will be equal to 0
    ey = (ey - theta) * nzy; //if y needs to be zeroed, ey will be equal to 0

    //Correction of the signs
    sx *= nzx;
    sy *= nzy;

    int factor_x = (1 - 2 * sx) * nzx; //-1 if  x is negative, 1 if x is positive, 0 if x needs to be zeroed (the exponent of x is too small)
    int factor_y = (1 - 2 * sy) * nzy; //-1 if  y is negative, 1 if y is positive, 0 if y needs to be zeroed (the exponent of y is too small)

    //Correction of the interval evaluations (multiplication by 2^gamma or 2^theta)
    evalx[0].exp += gamma;
    evalx[1].exp += gamma;
    evaly[0].exp += theta;
    evaly[1].exp += theta;

    //Change the signs of the interval evaluation bounds when the number is negative
    //The signs will not change when the number is positive
    //If the number needs to be reset, then the bounds will also be reset
    evalx[0].frac *=  factor_x;
    evalx[1].frac *=  factor_x;
    evaly[0].frac *=  factor_y;
    evaly[1].frac *=  factor_y;

    //Interval addition
    er_add_rd(&result->eval[0], &evalx[sx], &evaly[sy]);
    er_add_ru(&result->eval[1], &evalx[1 - sx], &evaly[1 - sy]);

    //Sign identification
    unsigned char sr;
    if(result->eval[0].frac * result->eval[1].frac >= 0){
        sr = (result->eval[0].frac < 0);
    } else{
        //Ambiguous case, use MRC, see http://dx.doi.org/10.14569/IJACSA.2020.0110901
        sr = sign_estimate(x->digits, y->digits, sx, sy, gamma, theta, nzx, nzy);
        result->eval[sr].frac = RNS_EVAL_UNIT.low.frac * (1 - 2 * sr);
        result->eval[sr].exp = RNS_EVAL_UNIT.low.exp;
    }
    result->sign = sr;

    //Calculation of the exponent
    result->exp = (ex == 0) ? ey : ex;

    //Addition of the RNS significands with multiplication by a power of two
    for (int i = 0; i < RNS_MODULI_SIZE; i++) {
        int residue = mod_axby(x->digits[i], RNS_POW2[gamma][i] * factor_x, y->digits[i], RNS_POW2[theta][i] * factor_y, RNS_MODULI[i]);
        result->digits[i] = residue < 0 ? residue + RNS_MODULI[i] : residue;
    }

    //Restoring the negative result
    if(sr == 1){
        for (int i = 0; i < RNS_MODULI_SIZE; i++) {
            result->digits[i] = (RNS_MODULI[i] - result->digits[i]) % RNS_MODULI[i];
        }
        er_float_t tmp = result->eval[0];
        result->eval[0].frac = -1 * result->eval[1].frac;
        result->eval[0].exp  = result->eval[1].exp;
        result->eval[1].frac = -1 * tmp.frac;
        result->eval[1].exp  = tmp.exp;
    }

    //Rounding
    if (result->eval[1].frac != 0 && result->eval[1].exp >= MP_H) {
        mp_round(result, mp_get_rnd_bits(result));
    }
}


/*
 * GPU functions
 */
namespace cuda {

    /*!
     * General routine for adding multiple-precision numbers (result = x + y)
     * The routines below call this procedure
     */
    DEVICE_CUDA_FORCEINLINE void mp_add_common(int * sr, int * er, er_float_ptr * evlr, int * digr,
                                               int sx, int ex, er_float_ptr * evlx, const int * digx,
                                               int sy, int ey, er_float_ptr * evly, const int * digy)
    {
        constexpr int moduli[ RNS_MODULI_SIZE ] = RNS_MODULI_VALUES;
        er_float_t evalx[2];
        er_float_t evaly[2];
        evalx[0] = *evlx[0];
        evalx[1] = *evlx[1];
        evaly[0] = *evly[0];
        evaly[1] = *evly[1];

        int dexp = ex - ey;
        int gamma =  dexp * (dexp > 0);
        int theta = -dexp * (dexp < 0);

        const int nzx = ((evaly[1].frac == 0) || (theta + evaly[1].exp) < cuda::MP_J);
        const int nzy = ((evalx[1].frac == 0) || (gamma + evalx[1].exp) < cuda::MP_J);

        gamma = gamma * nzy;
        theta = theta * nzx;

        ex = (ex - gamma) * nzx;
        ey = (ey - theta) * nzy;

        sx *= nzx;
        sy *= nzy;

        int factor_x = (1 - 2 * sx) * nzx;
        int factor_y = (1 - 2 * sy) * nzy;

        evalx[0].exp += gamma;
        evalx[1].exp += gamma;
        evaly[0].exp += theta;
        evaly[1].exp += theta;

        evalx[0].frac *=  factor_x;
        evalx[1].frac *=  factor_x;
        evaly[0].frac *=  factor_y;
        evaly[1].frac *=  factor_y;

        cuda::er_add_rd(evlr[0], &evalx[sx], &evaly[sy]);
        cuda::er_add_ru(evlr[1], &evalx[1 - sx], &evaly[1 - sy]);

        //Sign identification
        unsigned char sign = evlr[0]->frac < 0;
        if(evlr[0]->frac * evlr[1]->frac < 0){
            sign = cuda::sign_estimate(digx, digy, sx, sy, gamma, theta, nzx, nzy);
            evlr[sign]->frac = cuda::RNS_EVAL_UNIT.low.frac * (1 - 2 * sign);
            evlr[sign]->exp =  cuda::RNS_EVAL_UNIT.low.exp;
        }
        *sr = sign;
        *er = (ex == 0) ? ey : ex;
        cuda::rns_axby_cd(digr, cuda::RNS_POW2[gamma], digx, factor_x, cuda::RNS_POW2[theta], digy, factor_y);
        if(sign == 1){
            for (int i = 0; i < RNS_MODULI_SIZE; i++) {
                digr[i] = (digr[i] != 0) * (moduli[i] - digr[i]);
            }
            er_float_t tmp = *evlr[0];
            evlr[0]->frac = -evlr[1]->frac;
            evlr[0]->exp  = evlr[1]->exp;
            evlr[1]->frac = -1 * tmp.frac;
            evlr[1]->exp  = tmp.exp;
        }
    }

    /*!
     * Addition of two multiple-precision numbers
     * result = x + y
     */
    DEVICE_CUDA_FORCEINLINE void mp_add(mp_float_ptr result, mp_float_ptr x, mp_float_ptr y) {
        er_float_ptr evalx[2] = { &x->eval[0], &x->eval[1] }; //Array of pointers to interval evaluations
        er_float_ptr evaly[2] = { &y->eval[0], &y->eval[1] };
        er_float_ptr evalr[2] = { &result->eval[0], &result->eval[1] };

        mp_add_common(&result->sign, &result->exp, evalr, result->digits,
                      x->sign, x->exp, evalx, x->digits,
                      y->sign, y->exp, evaly, y->digits);

        if (result->eval[1].frac != 0 && result->eval[1].exp >= cuda::MP_H) {
            cuda::mp_round(result, cuda::mp_get_rnd_bits(result));
        }
    }

    /*!
     * Addition of two multiple-precision numbers using the mp_array_t type for the second argument
     * @param idy - index of the desired element in the vector y
     * @param result - pointer to the computed sum, result = x + y[idy]
     */
    DEVICE_CUDA_FORCEINLINE void mp_add(mp_float_ptr result, mp_float_ptr x, mp_array_t y, int idy) {
        er_float_ptr evalx[2] = { &x->eval[0], &x->eval[1] };
        er_float_ptr evaly[2] = { &y.eval[idy], &y.eval[idy + y.len[0]] };
        er_float_ptr evalr[2] = { &result->eval[0], &result->eval[1] };

        mp_add_common(&result->sign, &result->exp, evalr, result->digits,
                      x->sign, x->exp, evalx, x->digits,
                      y.sign[idy], y.exp[idy], evaly,&y.digits[RNS_MODULI_SIZE * idy]);

        if (result->eval[1].frac != 0 && result->eval[1].exp >= cuda::MP_H) {
            cuda::mp_round(result, cuda::mp_get_rnd_bits(result));
        }
    }

    /*!
     * Addition of two multiple-precision numbers using the mp_array_t type for the first argument and result
     * @param idx - index of the desired element in the vector x
     * @param idr - index in the result vector to write the computed sum
     * @param result - pointer to the computed sum, result[idr] = x[idx] + y
     */
    DEVICE_CUDA_FORCEINLINE void mp_add(mp_array_t result, int idr, mp_array_t x, int idx, mp_float_ptr y) {
        int lenr = result.len[0]; //Actual length of the result vector

        er_float_ptr evalx[2] = { &x.eval[idx], &x.eval[idx + x.len[0]] };
        er_float_ptr evaly[2] = { &y->eval[0], &y->eval[1] };
        er_float_ptr evalr[2] = { &result.eval[idr], &result.eval[idr + lenr] };

        mp_add_common(&result.sign[idr], &result.exp[idr], evalr, &result.digits[RNS_MODULI_SIZE * idr],
                      x.sign[idx], x.exp[idx], evalx, &x.digits[RNS_MODULI_SIZE * idx],
                      y->sign, y->exp, evaly, y->digits);
        //Rounding
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

    /*!
     * Addition of two multiple-precision numbers using the mp_collection_t type for the second argument
     * @param idy - index of the desired element in the vector y
     * @param leny - length of y
     * @param result - pointer to the computed sum, result = x + y[idy]
     */
    DEVICE_CUDA_FORCEINLINE void mp_add(mp_float_ptr result, mp_float_ptr x, mp_collection_t y, int idy, int leny) {
        er_float_ptr evalx[2] = { &x->eval[0], &x->eval[1] };
        er_float_ptr evaly[2] = { &y.eval[idy], &y.eval[idy + leny] };
        er_float_ptr evalr[2] = { &result->eval[0], &result->eval[1] };

        mp_add_common(&result->sign, &result->exp, evalr, result->digits,
                      x->sign, x->exp, evalx, x->digits,
                      y.sign[idy], y.exp[idy], evaly, &y.digits[RNS_MODULI_SIZE * idy]);
        if (result->eval[1].frac != 0 && result->eval[1].exp >= cuda::MP_H) {
            cuda::mp_round(result, cuda::mp_get_rnd_bits(result));
        }
    }


} //namespace cuda

#endif //MPRES_ADD_CUH
