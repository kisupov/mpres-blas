/*
 *  Multiple-precision floating-point assignment and conversion routines
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

#ifndef MPRES_ASSIGN_CUH
#define MPRES_ASSIGN_CUH

#include "arith_utils.cuh"

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
    char* cstr;
    cstr = mpfr_get_str(NULL, &exp, 2, 0, x, MPFR_RNDN);
    std::string mantissa(cstr);
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
    mpfr_free_str(cstr);
}

/*GCC_FORCEINLINE void mp_set_mpfr2(mp_float_ptr result, mpfr_srcptr x) {
    mpfr_t rounded;
    mpz_t mantissa, remainder;
    mpfr_init2(rounded, MP_PRECISION);
    mpz_init(mantissa);
    mpz_init(remainder);

    mpfr_set(rounded, x, MPFR_RNDN);
    result->exp = (int) mpfr_get_z_exp(mantissa, rounded);
   // gmp_printf ("result: %.70Zd\n", mantissa);

    result->sign = (mpz_cmp_ui(mantissa, 0) == -1);
    for (int i = 0; i < RNS_MODULI_SIZE; i++) {
        mpz_mod_ui(remainder, mantissa, RNS_MODULI[i]);
        result->digits[i] = mpz_get_ui(remainder);
    }
    rns_eval_compute(&result->eval[0], &result->eval[1], result->digits);
    mpfr_clear(rounded);
    mpz_clear(mantissa);
    mpz_clear(remainder);
}*/

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

} //namespace cuda

#endif //MPRES_ASSIGN_CUH
