/*
 *  Multiple-precision (MP) floating-point arithmetic using Residue number system
 *
 *  Copyright 2018, 2019 by Konstantin Isupov and Alexander Kuvaev.
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

#ifndef MPRES_MPFLOAT_CUH
#define MPRES_MPFLOAT_CUH

#include <iostream>
#include "rns.cuh"

/********************* Global precomputed constants *********************/

int MP_PRECISION; // The precision of arithmetic operations
int MP_H; // An integer such that 2^MP_H <= SQRT(M) / M < 2^{MP_H +1}. Used in rounding
int MP_J; // An integer such that 2^MP_J <= (M - SQRT(M)) / M < 2^{MP_J +1}. Used in exponent alignment
mp_float_t MP_ZERO; //Zero in the used multiple-precision representation
mp_float_t MP_MIN; //The smallest number in MP format (largest negative value)

//Constants for GPU
namespace cuda {
    __device__ __constant__ int MP_H;
    __device__ __constant__ int MP_J;
    __device__ __constant__ mp_float_t MP_ZERO;
    __device__ __constant__ mp_float_t MP_MIN;
}

void mp_const_init() {
    //Computating MP_PRECISION
    mpfr_t mpfr_tmp;
    mpfr_init2(mpfr_tmp, 20000);
    mpfr_sqrt(mpfr_tmp, RNS_MODULI_PRODUCT_MPFR, MPFR_RNDD);
    mpfr_log2(mpfr_tmp, mpfr_tmp, MPFR_RNDD);
    MP_PRECISION = floor(mpfr_get_ui(mpfr_tmp, MPFR_RNDD)) - 1;
    // Сomputing MP_H
    mpfr_sqrt(mpfr_tmp, RNS_MODULI_PRODUCT_MPFR, MPFR_RNDD);
    mpfr_div(mpfr_tmp, mpfr_tmp, RNS_MODULI_PRODUCT_MPFR, MPFR_RNDD);
    mpfr_log2(mpfr_tmp, mpfr_tmp, MPFR_RNDD);
    MP_H = mpfr_get_si(mpfr_tmp, MPFR_RNDD);
    //Сomputing MP_J
    mpfr_sqrt(mpfr_tmp, RNS_MODULI_PRODUCT_MPFR, MPFR_RNDD);
    mpfr_sub(mpfr_tmp, RNS_MODULI_PRODUCT_MPFR, mpfr_tmp, MPFR_RNDD);
    mpfr_div(mpfr_tmp, mpfr_tmp, RNS_MODULI_PRODUCT_MPFR, MPFR_RNDD);
    mpfr_log2(mpfr_tmp, mpfr_tmp, MPFR_RNDD);
    MP_J = mpfr_get_si(mpfr_tmp, MPFR_RNDD);
    mpfr_clear(mpfr_tmp);
    //Setting MP_ZERO
    MP_ZERO.sign = 0;
    MP_ZERO.exp = 0;
    MP_ZERO.eval[0].exp = 0;
    MP_ZERO.eval[1].exp = 0;
    MP_ZERO.eval[0].frac = 0.0;
    MP_ZERO.eval[1].frac = 0.0;
    for (int i = 0; i < RNS_MODULI_SIZE; ++i) {
        MP_ZERO.digits[i] = 0;
    }
    //Setting MP_MIN
    MP_MIN.sign = 1;
    MP_MIN.exp = MP_EXP_MAX;
    for (int i = 0; i < RNS_MODULI_SIZE; ++i) {
        MP_MIN.digits[i] = RNS_MODULI[i] - 1;
    }
    rns_eval_compute(&MP_MIN.eval[0], &MP_MIN.eval[1], MP_MIN.digits);
    //Copying constants to the GPU memory
    cudaMemcpyToSymbol(cuda::MP_H, &MP_H, sizeof(int));
    cudaMemcpyToSymbol(cuda::MP_J, &MP_J, sizeof(int));
    cudaMemcpyToSymbol(cuda::MP_ZERO, &MP_ZERO, sizeof(mp_float_t));
    cudaMemcpyToSymbol(cuda::MP_MIN, &MP_MIN, sizeof(mp_float_t));
}

/*
 * Print main constants of the MP arithmetic
 */
void mp_const_print(){
    std::cout << "Constants of the RNS-based floating-point arithmetic:" << std::endl;
    printf("- MP_PRECISION: %i\n", MP_PRECISION);
    printf("- MP_H: %i\n", MP_H);
    printf("- MP_J: %i\n", MP_J);
}


/********************* Assignment, conversion and formatted output functions *********************/

/*!
 * Assign the value of x to result
 */
GCC_FORCEINLINE void mp_set(mp_float_ptr result, mp_float_ptr x) {
    rns_set(result->digits, x->digits);
    result->sign = x->sign;
    result->exp = x->exp;
    result->eval[0] = x->eval[0];
    result->eval[1] = x->eval[1];
}

/*!
 * Forming a multiple-precision number from the significand, exponent and sign
 */
GCC_FORCEINLINE void mp_set(mp_float_ptr result, long significand, int exp, int sign) {
    result->sign = sign;
    result->exp = exp;
    for (int i = 0; i < RNS_MODULI_SIZE; ++i) {
        long residue = significand % (long)RNS_MODULI[i];
        result->digits[i] = (int) residue;
    }
    rns_eval_compute(&result->eval[0], &result->eval[1], result->digits);
}

/*!
 * Assign the value of a double precision variable x to result
 */
GCC_FORCEINLINE void mp_set_d(mp_float_ptr result, const double x) {
    int sign;
    int exp;
    long significand;
    RealIntUnion u = {x};
    //Extraction of the sign, exponent and significand from the union
    sign = (int) (u.ivalue >> DBL_SIGN_OFFSET);
    exp = (u.ivalue >> DBL_EXP_OFFSET) & (DBL_EXP_BIAS * 2 + 1);
    if (exp == 0) {
        significand = (u.ivalue & ((long) 1 << DBL_EXP_OFFSET) - 1);
    } else {
        significand = (u.ivalue & ((long) 1 << DBL_EXP_OFFSET) - 1 | (long) 1 << DBL_EXP_OFFSET);
    }
    exp = exp - DBL_EXP_BIAS - DBL_EXP_OFFSET;
    //Trimming least significant zeros
    if (significand != 0) {
        while ((significand & 1) == 0) {
            significand = significand >> 1;
            exp++;
        }
    } else {
        exp = 0;
    }
    mp_set(result, significand, exp, sign);
}

/*!
 * Assign the value of a multiple-precision (mpfr) variable x to result
 */
GCC_FORCEINLINE void mp_set_mpfr(mp_float_ptr result, mpfr_srcptr x) {
    mpz_t mpz_mant;
    mpz_t rem;
    mpz_init(mpz_mant);
    mpz_init(rem);
    mp_exp_t exp;
    std::string mantissa = mpfr_get_str(NULL, &exp, 2, 0, x, MPFR_RNDN);
    unsigned int num = 0;
    for (int i = (int) (mantissa.length() - 1); i > -1; i--) {
        if (mantissa[i] != '0') {
            num = i;
            break;
        }
    }
    if (num != mantissa.length() - 1) {
        std::string subbuff = "";
        subbuff = mantissa.substr(0, (unsigned long) (num + 1));
        mpz_set_str(mpz_mant, subbuff.c_str(), 2);
        result->exp = exp - subbuff.length();
        std::string().swap(subbuff);
    } else {
        mpz_set_str(mpz_mant, mantissa.c_str(), 2);
        result->exp = exp - mantissa.length();
    }
    if (mantissa[0] == '-') {
        result->sign = 1;
        result->exp++;
        mpz_mul_si(mpz_mant, mpz_mant, -1);
    } else {
        result->sign = 0;
    }
    for (int i = 0; i < RNS_MODULI_SIZE; i++) {
        mpz_mod_ui(rem, mpz_mant, RNS_MODULI[i]);
        result->digits[i] = mpz_get_ui(rem);
    }
    rns_eval_compute(&result->eval[0], &result->eval[1], result->digits);
    mpz_clear(mpz_mant);
    mpz_clear(rem);
    std::string().swap(mantissa);
}

/*!
 * Convert x to a double
 */
GCC_FORCEINLINE double mp_get_d(mp_float_ptr x) {
    mpfr_t disp_mpfr;
    mpfr_init(disp_mpfr);
    mpz_t temp_mpz, full_mpz;
    mpz_init(temp_mpz);
    mpz_init(full_mpz);
    mpz_set_si(full_mpz, 0);
    for (int i = 0; i < RNS_MODULI_SIZE; i++) {
        mpz_mul_si(temp_mpz, RNS_ORTHOGONAL_BASE[i], x->digits[i]);
        mpz_add(full_mpz, full_mpz, temp_mpz);
    }
    mpz_mod(full_mpz, full_mpz, RNS_MODULI_PRODUCT);
    char *str = mpz_get_str(NULL, 2, full_mpz);
    std::string mystring = "";
    if (x->sign == 1) {
        mystring += "-";
    }
    mystring += str;
    mystring += "e";
    mystring += toString(x->exp);
    mpfr_set_str(disp_mpfr, mystring.c_str(), 2, MPFR_RNDN);
    double a = mpfr_get_d(disp_mpfr, MPFR_RNDN);
    mpz_clear(full_mpz);
    mpz_clear(temp_mpz);
    mpfr_clear(disp_mpfr);
    return a;
}

/*!
 * Convert x to the mpfr_t number result
 */
GCC_FORCEINLINE void mp_get_mpfr(mpfr_t result, mp_float_ptr x) {
    mpz_t temp_mpz, full_mpz;
    mpz_init(temp_mpz);
    mpz_init(full_mpz);
    mpz_set_si(full_mpz, 0);
    for (int i = 0; i < RNS_MODULI_SIZE; i++) {
        mpz_mul_si(temp_mpz, RNS_ORTHOGONAL_BASE[i], x->digits[i]);
        mpz_add(full_mpz, full_mpz, temp_mpz);
    }
    mpz_mod(full_mpz, full_mpz, RNS_MODULI_PRODUCT);
    char *str = mpz_get_str(NULL, 2, full_mpz);
    std::string mystring = "";
    if (x->sign == 1)
        mystring += "-";
    mystring += str;
    mystring += "e";
    mystring += toString(x->exp);
    mpfr_set_str(result, mystring.c_str(), 2, MPFR_RNDN);
    mpz_clear(full_mpz);
    mpz_clear(temp_mpz);
    // In order to free the memory we need to get the right free function:
    void (*freefunc)(void *, size_t);
    mp_get_memory_functions(NULL, NULL, &freefunc);
    // In order to use free one needs to give both the pointer and the block
    // size. For tmp this is strlen(tmp) + 1, see [1].
    freefunc(str, strlen(str) + 1);
}

/*!
 * Print the parts (fields) of a multiple-precision number
 */
void mp_print(mp_float_ptr x) {
    printf("\nMultiple-precision value:\n");
    printf("- Sign: %d\n", x->sign);
    printf("- Significand: <");
    for (int i = 0; i < RNS_MODULI_SIZE; i++)
        printf("%d ", x->digits[i]);
    printf("\b>\n");
    printf("- Exp: %i\n", x->exp);
    printf("- Eval.low: ");
    er_print(&x->eval[0]);
    printf("\n- Eval.exact: ");
    er_float_t er_temp;
    rns_fractional(&er_temp, x->digits);
    er_print(&er_temp);
    printf("\n- Eval.upp: ");
    er_print(&x->eval[1]);
    printf("\n\n");
}


/********************* Basic multiple-precision arithmetic operations *********************/

/*
 * Returns the number of rounding bits for x
 */
GCC_FORCEINLINE int mp_get_rnd_bits(mp_float_ptr x) {
    #if defined(DEBUG) || defined(_DEBUG)
    if(x->eval[1].exp != x->eval[0].exp){
        printf("\n [CPU WARNING] Possible loss of accuracy");
    }
    #endif
    return x->eval[1].exp - MP_H + 1;
}

/*
 * Rounding x by n bits
 */
GCC_FORCEINLINE void mp_round(mp_float_ptr x, int n) {
    if (n > 0) {
        x->exp = x->exp + n;
        rns_scale2pow(x->digits, x->digits, (unsigned) n);
        //After rounding, the significand will be small enough,
        //so the interval evaluation can be computed faster.
        rns_eval_compute_fast(&x->eval[0], &x->eval[1], x->digits);
    }
}

/*
 * Sign identification in addition and subtraction operations when an ambiguous case occurs.
 * To compare the aligned significands, the mixed-radix conversion is used.
 * */
GCC_FORCEINLINE int sign_estimate(const int * digx, const int * digy, const int sx, const int sy,
                                  const int gamma, const int theta,  const unsigned char nzx, const unsigned char nzy){
    int lx[RNS_MODULI_SIZE];
    int ly[RNS_MODULI_SIZE];
    for(int i = 0; i < RNS_MODULI_SIZE; i++){
        lx[i] = mod_mul(digx[i], RNS_POW2[gamma][i] * nzx, RNS_MODULI[i]);
        ly[i] = mod_mul(digy[i], RNS_POW2[theta][i] * nzy, RNS_MODULI[i]);
    }
    int cmp = mrc_compare_rns(lx, ly);
    return (cmp < 0 ? sy : sx) * (cmp != 0);
}

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

/*!
 * Multiplication of two multiple-precision numbers
 * result = x * y
 */
GCC_FORCEINLINE void mp_mul(mp_float_ptr result, mp_float_ptr x, mp_float_ptr y) {
    result->exp = x->exp + y->exp;
    result->sign = x->sign ^ y->sign;
    er_md_rd(&result->eval[0], &x->eval[0], &y->eval[0], &RNS_EVAL_UNIT.upp);
    er_md_ru(&result->eval[1], &x->eval[1], &y->eval[1], &RNS_EVAL_UNIT.low);
    for(int i = 0; i < RNS_MODULI_SIZE; i ++){
        result->digits[i] = mod_mul(x->digits[i], y->digits[i], RNS_MODULI[i]);
    }
    if (result->eval[1].frac != 0 && result->eval[1].exp >= MP_H) {
        mp_round(result, mp_get_rnd_bits(result));
    }
}

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

/*!
 * Comparison of the absolute values of x and y
 * Returns 1 if |x| > |y|, -1 if |x| < |y|, and 0 otherwise
 */
GCC_FORCEINLINE int mp_cmp_abs(mp_float_ptr x, mp_float_ptr y) {
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
    return rns_cmp(digits_x, &eval_x[0], &eval_x[1], digits_y, &eval_y[0], & eval_y[1]);
}


/*
 * GPU functions
 */
namespace cuda {

    /*!
     * Assign the value of x to result
     */
    DEVICE_CUDA_FORCEINLINE void mp_set(mp_float_ptr result, mp_float_ptr x) {
        for(int i = 0; i < RNS_MODULI_SIZE; i++){
            result->digits[i] = x->digits[i];
        }
        result->sign = x->sign;
        result->exp = x->exp;
        result->eval[0] = x->eval[0];
        result->eval[1] = x->eval[1];
    }

    /*!
     * Assign the value of x to result[idr]
     * @param idr - index in the result vector to write the value of x
     * @param result - pointer to the mp_array_t vector with result[idr] = x
     */
    DEVICE_CUDA_FORCEINLINE void mp_set(mp_array_t result, int idr, mp_float_ptr x) {
        for(int i = 0; i < RNS_MODULI_SIZE; i++){
            result.digits[RNS_MODULI_SIZE * idr + i] = x->digits[i];
        }
        result.sign[idr] = x->sign;
        result.exp[idr] = x->exp;
        result.eval[idr] = x->eval[0];
        result.eval[idr + result.len[0]] = x->eval[1];
    }

    /*
     * Returns the number of rounding bits for x
     */
    DEVICE_CUDA_FORCEINLINE int mp_get_rnd_bits(mp_float_ptr x) {
        #if defined(DEBUG) || defined(_DEBUG)
        if(x->eval[1].exp != x->eval[0].exp){
            printf("\n [CUDA WARNING] Possible loss of accuracy");
         }
        #endif
        return x->eval[1].exp - cuda::MP_H + 1;
    }

    /*
     * Rounding x by n bits
     */
    DEVICE_CUDA_FORCEINLINE void mp_round(mp_float_ptr x, int n) {
        while (n > 0) {
            x->exp += n;
            cuda::rns_scale2pow(x->digits, x->digits, (unsigned int) n);
            //After rounding, the significand will be small enough,
            //so the interval evaluation can be computed faster.
            cuda::rns_eval_compute_fast(&x->eval[0], &x->eval[1], x->digits);
            n = -1;
        }
    }

    /*
     * Parallel (n threads) rounding x by n bits
     * This function must be performed by n threads simultaneously within a single thread block
     */
    DEVICE_CUDA_FORCEINLINE void mp_round_thread(mp_float_ptr x, int n) {
        while (n > 0) {
            cuda::rns_scale2pow_thread(x->digits, x->digits, (unsigned int) n);
            if (threadIdx.x == 0) {
                x->exp += n;
                cuda::rns_eval_compute_fast(&x->eval[0], &x->eval[1], x->digits);
            }
            n = -1;
        }
    }

    ////////////////////////////////////////////////////////////////
    // Addition routines
    ////////////////////////////////////////////////////////////////

    /*
     * Sign identification in addition and subtraction operations when an ambiguous case occurs.
     * To compare the aligned significands, the mixed-radix conversion is used.
     */
    DEVICE_CUDA_FORCEINLINE int sign_estimate(const int * digx, const int * digy, const int sx, const int sy,
                                              const int gamma, const int theta,  const unsigned char nzx, const unsigned char nzy){
        int lx[RNS_MODULI_SIZE];
        int ly[RNS_MODULI_SIZE];
        for(int i = 0; i < RNS_MODULI_SIZE; i++){
            lx[i] = cuda::mod_mul(digx[i], cuda::RNS_POW2[gamma][i] * nzx, cuda::RNS_MODULI[i]);
            ly[i] = cuda::mod_mul(digy[i], cuda::RNS_POW2[theta][i] * nzy, cuda::RNS_MODULI[i]);
        }
        int cmp = cuda::mrc_compare_rns(lx, ly);
        return (cmp < 0 ? sy : sx) * (cmp != 0);
    }

    /*!
     * General routine for adding multiple-precision numbers (result = x + y)
     * The routines below call this routine
     */
    DEVICE_CUDA_FORCEINLINE void mp_add_common(int * sr, int * er, er_float_ptr * evalr, int * digr,
                                               int sx, int ex, er_float_ptr * evlx, const int * digx,
                                               int sy, int ey, er_float_ptr * evly, const int * digy)
    {
        er_float_t evalx[2];
        er_float_t evaly[2];
        evalx[0] = *evlx[0];
        evalx[1] = *evlx[1];
        evaly[0] = *evly[0];
        evaly[1] = *evly[1];

        int dexp = ex - ey;
        int gamma =  dexp * (dexp > 0);
        int theta = -dexp * (dexp < 0);

        unsigned char  nzx = ((evaly[1].frac == 0) || (theta + evaly[1].exp) < cuda::MP_J);
        unsigned char  nzy = ((evalx[1].frac == 0) || (gamma + evalx[1].exp) < cuda::MP_J);

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

        cuda::er_add_rd(evalr[0], &evalx[sx], &evaly[sy]);
        cuda::er_add_ru(evalr[1], &evalx[1 - sx], &evaly[1 - sy]);

        //Sign identification
        unsigned char sign;
        if(evalr[0]->frac * evalr[1]->frac >= 0){
            sign = (evalr[0]->frac < 0);
        } else{
            sign = sign_estimate(digx, digy, sx, sy, gamma, theta, nzx, nzy);
            evalr[sign]->frac = cuda::RNS_EVAL_UNIT.low.frac * (1 - 2 * sign);
            evalr[sign]->exp =  cuda::RNS_EVAL_UNIT.low.exp;
        }
        *sr = sign;
        *er = (ex == 0) ? ey : ex;

        for (int i = 0; i < RNS_MODULI_SIZE; i++) {
            int residue = cuda::mod_axby(
                    digx[i], cuda::RNS_POW2[gamma][i] * factor_x,
                    digy[i], cuda::RNS_POW2[theta][i] * factor_y,
                    cuda::RNS_MODULI[i],
                    cuda::RNS_MODULI_RECIPROCAL[i]);
            digr[i] = residue < 0 ? residue + cuda::RNS_MODULI[i] : residue;
        }

        if(sign == 1){
            for (int i = 0; i < RNS_MODULI_SIZE; i++) {
                digr[i] = (digr[i] != 0) * (cuda::RNS_MODULI[i] - digr[i]);
            }
            er_float_t tmp = *evalr[0];
            evalr[0]->frac = -evalr[1]->frac;
            evalr[0]->exp  = evalr[1]->exp;
            evalr[1]->frac = -1 * tmp.frac;
            evalr[1]->exp  = tmp.exp;
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
       /* evalx[0] = &x->eval[0];
        evalx[1] = &x->eval[1];
        evaly[0] = &y.eval[idy];
        evaly[1] = &y.eval[idy + y.len[0]];
        evalr[0] = &result->eval[0];
        evalr[1] = &result->eval[1];*/

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

    ////////////////////////////////////////////////////////////////
    // Addition of absolute values routines
    ////////////////////////////////////////////////////////////////

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

    ////////////////////////////////////////////////////////////////
    // Multiplication routines
    ////////////////////////////////////////////////////////////////

    /*!
     * Multiplication of two multiple-precision numbers
     * result = x * y
     */
    DEVICE_CUDA_FORCEINLINE void mp_mul(mp_float_ptr result, mp_float_ptr x, mp_float_ptr y) {
        result->exp = x->exp + y->exp;
        result->sign = x->sign ^ y->sign;
        cuda::er_md_rd(&result->eval[0], &x->eval[0], &y->eval[0], &cuda::RNS_EVAL_UNIT.upp);
        cuda::er_md_ru(&result->eval[1], &x->eval[1], &y->eval[1], &cuda::RNS_EVAL_UNIT.low);
        for(int i = 0; i < RNS_MODULI_SIZE; i ++){
            result->digits[i] = cuda::mod_mul(x->digits[i], y->digits[i], cuda::RNS_MODULI[i]);
        }
        if (result->eval[1].frac != 0 && result->eval[1].exp >= cuda::MP_H) {
            cuda::mp_round(result, cuda::mp_get_rnd_bits(result));
        }
    }

    /*!
     * Multiplication of two multiple-precision numbers using the mp_array_t type for the first argument
     * result = x[idx] * y
     */
    DEVICE_CUDA_FORCEINLINE void mp_mul(mp_float_ptr result, mp_array_t x, int idx, mp_float_ptr y) {
        int digx[RNS_MODULI_SIZE]; //digits of x
        for(int i = 0; i < RNS_MODULI_SIZE; i++){
            digx[i] = x.digits[RNS_MODULI_SIZE * idx + i];
        }
        result->exp = x.exp[idx] + y->exp;
        result->sign = x.sign[idx] ^ y->sign;
        cuda::er_md_rd(&result->eval[0], &x.eval[idx], &y->eval[0], &cuda::RNS_EVAL_UNIT.upp);
        cuda::er_md_ru(&result->eval[1], &x.eval[idx + x.len[0]], &y->eval[1], &cuda::RNS_EVAL_UNIT.low);
        for(int i = 0; i < RNS_MODULI_SIZE; i ++){
            result->digits[i] = cuda::mod_mul(digx[i], y->digits[i], cuda::RNS_MODULI[i]);
        }
        if (result->eval[1].frac != 0 && result->eval[1].exp >= cuda::MP_H) {
            cuda::mp_round(result, cuda::mp_get_rnd_bits(result));
        }
    }

    /*!
     * Multiplication of two multiple-precision numbers using the mp_array_t type for the arguments
     * result = x[idx] * y[idy]
     */
    DEVICE_CUDA_FORCEINLINE void mp_mul(mp_float_ptr result, mp_array_t x, int idx, mp_array_t y, int idy) {
        int digx[RNS_MODULI_SIZE]; //digits of x
        int digy[RNS_MODULI_SIZE]; //digits of y
        for(int i = 0; i < RNS_MODULI_SIZE; i++){
            digx[i] = x.digits[RNS_MODULI_SIZE * idx + i];
            digy[i] = y.digits[RNS_MODULI_SIZE * idy + i];
        }
        result->exp = x.exp[idx] + y.exp[idy];
        result->sign = x.sign[idx] ^ y.sign[idy];
        cuda::er_md_rd(&result->eval[0], &x.eval[idx], &y.eval[idy], &cuda::RNS_EVAL_UNIT.upp);
        cuda::er_md_ru(&result->eval[1], &x.eval[idx + x.len[0]], &y.eval[idy + y.len[0]], &cuda::RNS_EVAL_UNIT.low);
        for(int i = 0; i < RNS_MODULI_SIZE; i ++){
            result->digits[i] = cuda::mod_mul(digx[i], digy[i], cuda::RNS_MODULI[i]);
        }
        if (result->eval[1].frac != 0 && result->eval[1].exp >= cuda::MP_H) {
            cuda::mp_round(result, cuda::mp_get_rnd_bits(result));
        }
    }

    /*!
     * Parallel (n threads) multiplication of two multiple-precision numbers
     * This function must be performed by n threads simultaneously within a single thread block
     * result = x * y
     */
    /*DEVICE_CUDA_FORCEINLINE void mp_mul_thread(mp_float_ptr result, mp_float_ptr x, mp_float_ptr y) {
        result->digits[threadIdx.x] = cuda::mod_mul(x->digits[threadIdx.x], y->digits[threadIdx.x], cuda::RNS_MODULI[threadIdx.x]);
        if(threadIdx.x == 0) {
            result->exp = x->exp + y->exp;
            result->sign = x->sign ^ y->sign;
            cuda::er_md_rd(&result->eval[0], &x->eval[0], &y->eval[0], &cuda::RNS_EVAL_UNIT.upp);
            cuda::er_md_ru(&result->eval[1], &x->eval[1], &y->eval[1], &cuda::RNS_EVAL_UNIT.low);
        }
        __syncthreads();
        int bits = (result->eval[1].exp - cuda::MP_H + 1)*(result->eval[1].frac != 0);
        while(bits > 0){
            cuda::mp_round_thread(result, bits);
            bits = -1;
        }
    }*/

    ////////////////////////////////////////////////////////////////
    // Comparison routines
    ////////////////////////////////////////////////////////////////

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

    /*!
     * Comparison of the absolute values of x and y
     * Returns 1 if |x| > |y|, -1 if |x| < |y|, and 0 otherwise
     */
    DEVICE_CUDA_FORCEINLINE int mp_cmp_abs(mp_float_ptr x, mp_float_ptr y) {
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
        return cuda::rns_cmp(digits_x, &eval_x[0], &eval_x[1], digits_y, &eval_y[0], & eval_y[1]);
    }

    /*!
     * Comparison of the absolute values of x and y using the mp_array_t type for the first argument
     * Returns 1 if |x[idx]| > |y|, -1 if |x[idx]| < |y|, and 0 otherwise
     */
    DEVICE_CUDA_FORCEINLINE int mp_cmp_abs(mp_array_t x, int idx, mp_float_ptr y) {
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
        return cuda::rns_cmp(digits_x, &eval_x[0], &eval_x[1], digits_y, &eval_y[0], & eval_y[1]);
    }


} //end of namespace

#endif //MPRES_MPFLOAT_CUH
