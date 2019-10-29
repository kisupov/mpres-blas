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
//TODO: device constant??
//Constants for GPU
namespace cuda {
    __device__ __constant__ int MP_H;
    __device__ __constant__ int MP_J;
    __device__ __constant__ mp_float_t MP_ZERO;
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
    //Copying constants to the GPU memory
    cudaMemcpyToSymbol(cuda::MP_H, &MP_H, sizeof(int));
    cudaMemcpyToSymbol(cuda::MP_J, &MP_J, sizeof(int));
    cudaMemcpyToSymbol(cuda::MP_ZERO, &MP_ZERO, sizeof(mp_float_t));
}

/*
 * Print main constants of the MP arithmetic
 */
void mp_const_print(){
    std::cout << "Constants of the RNS-based floating-point arithmetic:" << std::endl;
    printf("- MP_PRECISION %i\n", MP_PRECISION);
    printf("- MP_H %i\n", MP_H);
    printf("- MP_J %i\n", MP_J);
}


/********************* Assignment, conversion and formatted output functions *********************/

/*!
 * Set the value of result from x
 */
GCC_FORCEINLINE void mp_set(mp_float_ptr result, mp_float_ptr x) {
    rns_set(result->digits, x->digits);
    result->sign = x->sign;
    result->exp = x->exp;
    result->eval[0] = x->eval[1];
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
    interval_t eval;
    rns_eval_compute(&eval, result->digits);
    result->eval[0] = eval.low;
    result->eval[1] = eval.upp;
}

/*!
 * Set the value of result from double-precision x
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
 * Set the value of result from the mpfr number x
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
    interval_t eval;
    rns_eval_compute(&eval, result->digits);
    result->eval[0] = eval.low;
    result->eval[1] = eval.upp;
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
        interval_t eval;
        rns_eval_compute_fast(&eval, x->digits);
        x->eval[0] = eval.low;
        x->eval[1] = eval.upp;
    }
}

/*!
 * Addition of two multiple-precision numbers
 * result = x + y
 */
GCC_FORCEINLINE void mp_add(mp_float_ptr result, mp_float_ptr x, mp_float_ptr y) {

    //The local copies of the terms are required since the exponent alignment can be performed
    er_float_t eval_x[2];
    er_float_t eval_y[2];
    eval_x[0] = x->eval[0];
    eval_x[1] = x->eval[1];
    eval_y[0] = y->eval[0];
    eval_y[1] = y->eval[1];

    int exp_x = x->exp;
    int exp_y = y->exp;
    int sign_x = x->sign;
    int sign_y = y->sign;

    //Exponent alignment
    int dexp = exp_x - exp_y;
    int gamma =  dexp * (dexp > 0); //if dexp > 0, then gamma =  dexp; otherwise gamma = 0
    int theta = -dexp * (dexp < 0); //if dexp < 0, then theta = -dexp; otherwise theta = 0

    int nzx = ((eval_y[1].frac == 0) || (theta + eval_y[1].exp) < MP_J); //nzx (u) = 1 if x not need be zeroed; otherwise nzx = 0
    int nzy = ((eval_x[1].frac == 0) || (gamma + eval_x[1].exp) < MP_J); //nzy (v) = 1 if y not need be zeroed; otherwise nzy = 0

    gamma = gamma * nzy; //if nzy = 0 (y needs to be zeroed), then gamma = 0, i.e. we will multiply x by 2^0 without actually changing the value of x
    theta = theta * nzx; //if nzx = 0 (x needs to be zeroed), then theta = 0, i.e. we will multiply y by 2^0 without actually changing the value of y

    //Correction of the exponents
    exp_x = (exp_x - gamma) * nzx; //if x needs to be zeroed, exp_x will be equal to 0
    exp_y = (exp_y - theta) * nzy; //if y needs to be zeroed, exp_y will be equal to 0

    //Correction of the signs
    sign_x *= nzx;
    sign_y *= nzy;

    int factor_x = (1 - 2 * sign_x) * nzx; //-1 if  x is negative, 1 if x is positive, 0 if x needs to be zeroed (the exponent of x is too small)
    int factor_y = (1 - 2 * sign_y) * nzy; //-1 if  y is negative, 1 if y is positive, 0 if y needs to be zeroed (the exponent of y is too small)

    //Correction of the interval evaluations (multiplication by 2^gamma or 2^theta)
    eval_x[0].exp += gamma;
    eval_x[1].exp += gamma;
    eval_y[0].exp += theta;
    eval_y[1].exp += theta;

    //Change the signs of the interval evaluation bounds when the number is negative
    //The signs will not change when the number is positive
    //If the number needs to be reset, then the bounds will also be reset
    eval_x[0].frac *=  factor_x;
    eval_x[1].frac *=  factor_x;
    eval_y[0].frac *=  factor_y;
    eval_y[1].frac *=  factor_y;

    //Interval addition
    round_down_mode();
    er_add(&result->eval[0], &eval_x[sign_x], &eval_y[sign_y]);
    round_up_mode();
    er_add(&result->eval[1], &eval_x[1 - sign_x], &eval_y[1 - sign_y]);
    round_nearest_mode();

    //Calculation of the exponent and preliminary calculation of the sign (the sign will be changed if restoring is required)
    result->sign = 0;
    result->exp = (exp_x == 0) ? exp_y : exp_x;

    //Addition of the RNS significands with multiplication by a power of two
    for (int i = 0; i < RNS_MODULI_SIZE; i++) {
        int residue = mod_axby(x->digits[i], RNS_POW2[gamma][i] * factor_x, y->digits[i], RNS_POW2[theta][i] * factor_y, RNS_MODULI[i]);
        result->digits[i] = residue < 0 ? residue + RNS_MODULI[i] : residue;
    }

    //Restoring the negative result
    //int plus  = result->eval[0].frac >= 0 && result->eval[1].frac >= 0;
    int minus = result->eval[0].frac < 0 && result->eval[1].frac < 0;
    //One observation (should be proven in the future):
    //when both plus and minus are equal to zero, the actual result is always non-negative.
    if(minus){
        result->sign = 1;
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

    round_down_mode();
    er_add(&result->eval[0], &eval_x[0], &eval_y[0]);
    round_up_mode();
    er_add(&result->eval[1], &eval_x[1], &eval_y[1]);
    round_nearest_mode();

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

/*
 * GPU functions
 */
namespace cuda {

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
            interval_t eval;
            cuda::rns_eval_compute_fast(&eval, x->digits);
            x->eval[0] = eval.low;
            x->eval[1] = eval.upp;
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
                interval_t eval;
                cuda::rns_eval_compute_fast(&eval, x->digits);
                x->eval[0] = eval.low;
                x->eval[1] = eval.upp;
            }
            n = -1;
        }
    }

    /*!
     * Addition of two multiple-precision numbers
     * result = x + y
     */
    DEVICE_CUDA_FORCEINLINE void mp_add(mp_float_ptr result, mp_float_ptr x, mp_float_ptr y) {
        er_float_t eval_x[2];
        er_float_t eval_y[2];
        eval_x[0] = x->eval[0];
        eval_x[1] = x->eval[1];
        eval_y[0] = y->eval[0];
        eval_y[1] = y->eval[1];

        int exp_x = x->exp;
        int exp_y = y->exp;
        int sign_x = x->sign;
        int sign_y = y->sign;

        int dexp = exp_x - exp_y;
        int gamma =  dexp * (dexp > 0);
        int theta = -dexp * (dexp < 0);

        int nzx = ((eval_y[1].frac == 0) || (theta + eval_y[1].exp) < cuda::MP_J);
        int nzy = ((eval_x[1].frac == 0) || (gamma + eval_x[1].exp) < cuda::MP_J);

        gamma = gamma * nzy;
        theta = theta * nzx;

        exp_x = (exp_x - gamma) * nzx;
        exp_y = (exp_y - theta) * nzy;

        sign_x *= nzx;
        sign_y *= nzy;

        int factor_x = (1 - 2 * sign_x) * nzx;
        int factor_y = (1 - 2 * sign_y) * nzy;

        eval_x[0].exp += gamma;
        eval_x[1].exp += gamma;
        eval_y[0].exp += theta;
        eval_y[1].exp += theta;

        eval_x[0].frac *=  factor_x;
        eval_x[1].frac *=  factor_x;
        eval_y[0].frac *=  factor_y;
        eval_y[1].frac *=  factor_y;

        cuda::er_add_rd(&result->eval[0], &eval_x[sign_x], &eval_y[sign_y]);
        cuda::er_add_ru(&result->eval[1], &eval_x[1 - sign_x], &eval_y[1 - sign_y]);

        result->sign = 0;
        result->exp = (exp_x == 0) ? exp_y : exp_x;

        for (int i = 0; i < RNS_MODULI_SIZE; i++) {
            int residue = cuda::mod_axby(
                    x->digits[i], cuda::RNS_POW2[gamma][i] * factor_x,
                    y->digits[i], cuda::RNS_POW2[theta][i] * factor_y,
                    cuda::RNS_MODULI[i],
                    cuda::RNS_MODULI_RECIPROCAL[i]);
            result->digits[i] = residue < 0 ? residue + cuda::RNS_MODULI[i] : residue;
        }
        //int plus  = result->eval[0].frac >= 0 && result->eval[1].frac >= 0; // see mp_add for CPU
        int minus = result->eval[0].frac < 0 && result->eval[1].frac < 0;
        if(minus){
            result->sign = 1;
            for (int i = 0; i < RNS_MODULI_SIZE; i++) {
                result->digits[i] = (cuda::RNS_MODULI[i] - result->digits[i]) % cuda::RNS_MODULI[i];
            }
            er_float_t tmp = result->eval[0];
            result->eval[0].frac = -1 * result->eval[1].frac;
            result->eval[0].exp  = result->eval[1].exp;
            result->eval[1].frac = -1 * tmp.frac;
            result->eval[1].exp  = tmp.exp;
        }
        if (result->eval[1].frac != 0 && result->eval[1].exp >= cuda::MP_H) {
            cuda::mp_round(result, cuda::mp_get_rnd_bits(result));
        }
    }

    /*!
     * Addition of two multiple-precision numbers using mp_array_t as the second argument
     * @param index - index of the desired element in the vector y
     * @param n - length of the vector y
     * @param result - pointer to the computed sum, result = x + y[index]
     */
    DEVICE_CUDA_FORCEINLINE void mp_add(mp_float_ptr result, mp_float_ptr x, mp_array_t y, int index, int n) {
        er_float_t eval_x[2];
        er_float_t eval_y[2];
        eval_x[0] = x->eval[0];
        eval_x[1] = x->eval[1];
        eval_y[0] = y.eval[index];
        eval_y[1] = y.eval[index + n];

        int exp_x = x->exp;
        int exp_y = y.exp[index];
        int sign_x = x->sign;
        int sign_y = y.sign[index];

        int dexp = exp_x - exp_y;
        int gamma =  dexp * (dexp > 0);
        int theta = -dexp * (dexp < 0);

        int nzx = ((eval_y[1].frac == 0) || (theta + eval_y[1].exp) < cuda::MP_J);
        int nzy = ((eval_x[1].frac == 0) || (gamma + eval_x[1].exp) < cuda::MP_J);

        gamma = gamma * nzy;
        theta = theta * nzx;

        exp_x = (exp_x - gamma) * nzx;
        exp_y = (exp_y - theta) * nzy;

        sign_x *= nzx;
        sign_y *= nzy;

        int factor_x = (1 - 2 * sign_x) * nzx;
        int factor_y = (1 - 2 * sign_y) * nzy;

        eval_x[0].exp += gamma;
        eval_x[1].exp += gamma;
        eval_y[0].exp += theta;
        eval_y[1].exp += theta;

        eval_x[0].frac *=  factor_x;
        eval_x[1].frac *=  factor_x;
        eval_y[0].frac *=  factor_y;
        eval_y[1].frac *=  factor_y;

        cuda::er_add_rd(&result->eval[0], &eval_x[sign_x], &eval_y[sign_y]);
        cuda::er_add_ru(&result->eval[1], &eval_x[1 - sign_x], &eval_y[1 - sign_y]);

        result->sign = 0;
        result->exp = (exp_x == 0) ? exp_y : exp_x;

        for (int i = 0; i < RNS_MODULI_SIZE; i++) {
            int residue = cuda::mod_axby(
                    x->digits[i],
                    cuda::RNS_POW2[gamma][i] * factor_x,
                    y.digits[RNS_MODULI_SIZE * index + i],
                    cuda::RNS_POW2[theta][i] * factor_y,
                    cuda::RNS_MODULI[i],
                    cuda::RNS_MODULI_RECIPROCAL[i]);
            result->digits[i] = residue < 0 ? residue + cuda::RNS_MODULI[i] : residue;
        }
        //int plus  = result->eval[0].frac >= 0 && result->eval[1].frac >= 0; // see mp_add for CPU
        int minus = result->eval[0].frac < 0 && result->eval[1].frac < 0;
        if(minus){
            result->sign = 1;
            for (int i = 0; i < RNS_MODULI_SIZE; i++) {
                result->digits[i] = (cuda::RNS_MODULI[i] - result->digits[i]) % cuda::RNS_MODULI[i];
            }
            er_float_t tmp = result->eval[0];
            result->eval[0].frac = -1 * result->eval[1].frac;
            result->eval[0].exp  = result->eval[1].exp;
            result->eval[1].frac = -1 * tmp.frac;
            result->eval[1].exp  = tmp.exp;
        }
        if (result->eval[1].frac != 0 && result->eval[1].exp >= cuda::MP_H) {
            cuda::mp_round(result, cuda::mp_get_rnd_bits(result));
        }
    }

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
     * Addition of the absolute values of two multiple-precision numbers using mp_array_t as the second argument
     * @param index - index of the desired element in the vector y
     * @param n - length of the vector y
     * @param result - pointer to the computed sum, result = x + y[index]
     */
    DEVICE_CUDA_FORCEINLINE void mp_add_abs(mp_float_ptr result, mp_float_ptr x, mp_array_t &y, int index, int n){
        er_float_t eval_x[2];
        er_float_t eval_y[2];
        eval_x[0] = x->eval[0];
        eval_x[1] = x->eval[1];
        eval_y[0] = y.eval[index];
        eval_y[1] = y.eval[index+n];

        int exp_x = x->exp;
        int exp_y = y.exp[index];

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
                    y.digits[RNS_MODULI_SIZE * index + i] * nzy,
                    cuda::RNS_POW2[theta][i],
                    cuda::RNS_MODULI[i],
                    cuda::RNS_MODULI_RECIPROCAL[i]);
        }
        if (result->eval[1].frac != 0 && result->eval[1].exp >= cuda::MP_H) {
            cuda::mp_round(result, cuda::mp_get_rnd_bits(result));
        }
    }

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
     * Parallel (n threads) multiplication of two multiple-precision numbers
     * This function must be performed by n threads simultaneously within a single thread block
     * result = x * y
     */
    DEVICE_CUDA_FORCEINLINE void mp_mul_thread(mp_float_ptr result, mp_float_ptr x, mp_float_ptr y) {
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
    }

} //end of namespace

#endif //MPRES_MPFLOAT_CUH
