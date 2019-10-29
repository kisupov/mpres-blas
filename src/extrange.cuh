/*
 *  Extended-range floating-point arithmetic for both the CPU and GPU platforms.
 *  2 is used as the exponent base, i.e., an extended-range number is represented
 *  as f * 2^i, where f is the double precision float, and i is the integer.
 *  In many cases, this simplifies and speeds up arithmetic operations.
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


#ifndef MPRES_EXTRANGE_CUH
#define MPRES_EXTRANGE_CUH

#include "types.cuh"
#include "common.cuh"
#include "bitwise.cuh"
#include "roundings.cuh"


/********************* Assignment and formatted output functions *********************/


/*!
 * Set the value of result from x
 */
GCC_FORCEINLINE void er_set(er_float_ptr result, er_float_ptr x) {
    result->frac = x->frac;
    result->exp = x->exp;
}

/*!
 * Set the value of result from double-precision x
 */
GCC_FORCEINLINE void er_set_d(er_float_ptr result, const double x) {
    RealIntUnion u;
    u.dvalue = x;
    if (x != 0) {
        //Zeroing the sign and displacing the significand by the exponent value
        result->exp = ((u.ivalue & ~((uint64_t) 1 << DBL_SIGN_OFFSET)) >> DBL_EXP_OFFSET) - DBL_EXP_BIAS;
        //Zeroing the exponent without changing the sign and significand
        u.ivalue = u.ivalue & DBL_ZERO_EXP_MASK | DBL_BIAS_EXP_MASK;
        result->frac = u.dvalue;
    } else {
        result->exp = 0;
        result->frac = 0;
    }
}

/*!
 * Convert x to a double. The procedure may yield an overflow or an underflow
 * if x is not representable in the exponent range of the double precision format
 */
GCC_FORCEINLINE void er_get_d(double *result, er_float_ptr x) {
    *result = fast_scalbn(x->frac, x->exp);
}

/*!
 * Inline print of x
 */
GCC_FORCEINLINE void er_print(er_float_ptr x) {
    printf("number->frac %.18e | number->exp %li", x->frac, x->exp);
}

/*
 * GPU functions
 */
namespace cuda {

    /*!
     * Set the value of result from x
     */
    DEVICE_CUDA_FORCEINLINE void er_set(er_float_ptr result, er_float_ptr x) {
        result->frac = x->frac;
        result->exp = x->exp;
    }

    /*!
     * Set the value of result from double-precision x
     */
    DEVICE_CUDA_FORCEINLINE void er_set_d(er_float_ptr result, const double x) {
        RealIntUnion u;
        u.dvalue = x;
        if (x != 0) {
            //Zeroing the sign and displacing the significand by the exponent value
            result->exp = ((u.ivalue & ~((uint64_t) 1 << DBL_SIGN_OFFSET)) >> DBL_EXP_OFFSET) - DBL_EXP_BIAS;
            //Zeroing the exponent without changing the sign and significand
            u.ivalue = u.ivalue & DBL_ZERO_EXP_MASK | DBL_BIAS_EXP_MASK;
            result->frac = u.dvalue;
        } else {
            result->exp = 0;
            result->frac = 0;
        }
    }

    /*!
     * Inline print of x
     */
    DEVICE_CUDA_FORCEINLINE void er_print(er_float_ptr x) {
        printf("number->frac %.18e | number->exp %i\n", x->frac, x->exp);
    }

} //end of namespace


/********************* Basic arithmetic and comparison functions *********************/


/*!
 * Adjustment of x. Places the significand, x.frac,
 * in the range of [1, 2). This routine exploits the bitwise
 * operators and is only correct for an extended-range base = 2
 */
GCC_FORCEINLINE void er_adjust(er_float_ptr x){
    RealIntUnion u;
    u.dvalue = x->frac;
    //Zeroing the sign and displacing the significand by the exponent value. If the significand is zero, then the exponent is also zeroed.
    x->exp = (x->exp + ((u.ivalue & ~((uint64_t) 1 << DBL_SIGN_OFFSET)) >> DBL_EXP_OFFSET) - DBL_EXP_BIAS) * (x->frac != 0);
    //Zeroing the exponent without changing the sign and significand
    u.ivalue = u.ivalue & DBL_ZERO_EXP_MASK | DBL_BIAS_EXP_MASK;
    //Setting the significand if it is not zero
    x->frac = u.dvalue * (x->frac != 0);
}

/*!
 * Addition of two extended-range numbers
 */
GCC_FORCEINLINE void er_add(er_float_ptr result, er_float_ptr x, er_float_ptr y) {
    x->exp *= (x->frac != 0); //Reset the exponent of x when the significand is zero
    y->exp *= (y->frac != 0); //Reset the exponent of y when the significand is zero
    // If either x or y is zero, then dexp will also be zero
    // If x->exp > y->exp, then dexp > 0
    // If x->exp < y->exp, then dexp < 0
    int dexp = (x->exp - y->exp) * (x->frac != 0) * (y->frac != 0);
    if (dexp > 0) {
        result->exp = x->exp;
        result->frac = x->frac + fast_scalbn(y->frac, -dexp);
    } else if (dexp < 0) {
        result->exp = y->exp;
        result->frac = y->frac + fast_scalbn(x->frac, dexp);
    } else {
        result->exp = (x->exp == 0) ? y->exp : x->exp;
        result->frac = x->frac + y->frac;
    }
    er_adjust(result);
}

/*!
 * Subtraction of two extended-range numbers
 */
GCC_FORCEINLINE void er_sub(er_float_ptr result, er_float_ptr x, er_float_ptr y) {
    x->exp *= (x->frac != 0);
    y->exp *= (y->frac != 0);
    int dexp = (x->exp - y->exp) * (x->frac != 0) * (y->frac != 0);
    if (dexp > 0) {
        result->exp = x->exp;
        result->frac = x->frac - fast_scalbn(y->frac, -dexp);
    } else if (dexp < 0) {
        result->exp = y->exp;
        result->frac = fast_scalbn(x->frac, dexp) - y->frac;
    } else {
        result->exp = (x->exp == 0) ? y->exp : x->exp;
        result->frac = x->frac - y->frac;
    }
    er_adjust(result);
}

/*!
 * Multiplication of two extended-range numbers
 */
GCC_FORCEINLINE void er_mul(er_float_ptr result, er_float_ptr x, er_float_ptr y) {
    result->exp = x->exp + y->exp;
    result->frac = x->frac * y->frac;
    er_adjust(result);
}

/*!
 * Division of two extended-range numbers
 */
GCC_FORCEINLINE void er_div(er_float_ptr result, er_float_ptr x, er_float_ptr y) {
    result->exp = x->exp - y->exp;
    result->frac = x->frac / y->frac;
    er_adjust(result);
}

/*!
 * Compute x * y / z with a single adjust call in rounding-up mode
 */
GCC_FORCEINLINE void er_md_ru(er_float_ptr result, er_float_ptr x, er_float_ptr y, er_float_ptr z) {
    result->exp = x->exp + y->exp - z->exp;
    round_up_mode();
    result->frac = (x->frac * y->frac) / z->frac;
    round_nearest_mode();
    er_adjust(result);
}

/*!
 * Compute x * y / z with a single adjust call in rounding-down mode
 */
GCC_FORCEINLINE void er_md_rd(er_float_ptr result, er_float_ptr x, er_float_ptr y, er_float_ptr z) {
    result->exp = x->exp + y->exp - z->exp;
    round_down_mode();
    result->frac = (x->frac * y->frac) / z->frac;
    er_adjust(result);
}

/*!
 * Comparison of two extended-range numbers
 * Returns 1, if x > y; -1, if x < y; 0, if x = y
 * The function is correct only when 1 <= |x.frac|, |y.frac| < 2
 */
GCC_FORCEINLINE int er_cmp(er_float_ptr x, er_float_ptr y) {
    int diff = x->exp - y->exp;
    if (diff == 0) {
        return (x->frac == y->frac ? 0 : x->frac > y->frac ? 1 : -1);
    }
    if (x->frac == 0) {
        return (y->frac > 0 ? -1 : 1); // x < y when y is positive
    }
    if (y->frac == 0) {
        return (x->frac > 0 ? 1 : -1); // x > y when x is positive
    }
    return (diff > 0 && x->frac > 0) || (diff < 0 && y->frac < 0) ? 1 : -1;
}

/*!
 * Comparison of two non-negative extended-range numbers
 * Performs faster than er_cmp
 * Returns 1, if x > y; -1, if x < y; 0, if x = y
 * The function is correct only when 1 <= x.frac, y.frac < 2
 */
GCC_FORCEINLINE int er_ucmp(er_float_ptr x, er_float_ptr y) {
    int diff = x->exp - y->exp;
    if (diff == 0) {
        return (x->frac == y->frac ? 0 : x->frac > y->frac ? 1 : -1);
    }
    return x->frac != 0 && (diff > 0 || (diff < 0 && y->frac == 0)) ? 1 : -1;
}


/*
 * GPU functions
 */
namespace cuda {

    /*!
     * Adjustment of x. Places the significand, x.frac,
     * in the range of [1, 2). This routine exploits the bitwise
     * operators and is only correct for an extended-range base = 2
     */
    DEVICE_CUDA_FORCEINLINE void er_adjust(er_float_ptr x) {
        RealIntUnion u;
        u.dvalue = x->frac;
        //Zeroing the sign and displacing the significand by the exponent value. If the significand is zero, then the exponent is also zeroed.
        x->exp = (x->exp + ((u.ivalue & ~((uint64_t) 1 << DBL_SIGN_OFFSET)) >> DBL_EXP_OFFSET) - DBL_EXP_BIAS) * (x->frac != 0);
        //Zeroing the exponent without changing the sign and significand
        u.ivalue = u.ivalue & DBL_ZERO_EXP_MASK | DBL_BIAS_EXP_MASK;
        //Setting the significand if it is not zero
        x->frac = u.dvalue * (x->frac != 0);
    }

    /*!
     * Addition of two extended-range numbers in rounding-up mode
     */
     DEVICE_CUDA_FORCEINLINE void er_add_ru(er_float_ptr result, er_float_ptr x, er_float_ptr y){
        x->exp *= (x->frac != 0); //Reset the exponent of x when the significand is zero
        y->exp *= (y->frac != 0); //Reset the exponent of y when the significand is zero
        // If either x or y is zero, then dexp will also be zero
        // If x->exp > y->exp, then dexp > 0
        // If x->exp < y->exp, then dexp < 0
        int dexp = (x->exp - y->exp) * (x->frac != 0) * (y->frac != 0);
        if (dexp > 0) {
            result->exp = x->exp;
            result->frac = __dadd_ru(x->frac, scalbn(y->frac, -dexp)); // may be cuda::fast_scalbn ?
        } else if (dexp < 0) {
            result->exp = y->exp;
            result->frac = __dadd_ru(y->frac, scalbn(x->frac, dexp));
        } else {
            result->exp = (x->exp == 0) ? y->exp : x->exp;
            result->frac = __dadd_ru(x->frac, y->frac);
        }
        cuda::er_adjust(result);
    }

    /*!
     * Addition of two extended-range numbers in rounding-down mode
     */
     DEVICE_CUDA_FORCEINLINE void er_add_rd(er_float_ptr result, er_float_ptr x, er_float_ptr y) {
        x->exp *= (x->frac != 0);
        y->exp *= (y->frac != 0);
        int dexp = (x->exp - y->exp) * (x->frac != 0) * (y->frac != 0);
        if (dexp > 0) {
            result->exp = x->exp;
            result->frac = __dadd_rd(x->frac, scalbn(y->frac, -dexp));
        } else if (dexp < 0) {
            result->exp = y->exp;
            result->frac = __dadd_rd(y->frac, scalbn(x->frac, dexp));
        } else {
            result->exp = (x->exp == 0) ? y->exp : x->exp;
            result->frac = __dadd_rd(x->frac, y->frac);
        }
        cuda::er_adjust(result);
    }

    /*!
     * Subtraction of two extended-range numbers in rounding-up mode
     */
    DEVICE_CUDA_FORCEINLINE void er_sub_ru(er_float_ptr result, er_float_ptr x, er_float_ptr y) {
        x->exp *= (x->frac != 0);
        y->exp *= (y->frac != 0);
        int dexp = (x->exp - y->exp) * (x->frac != 0) * (y->frac != 0);
        if (dexp > 0) {
            result->exp = x->exp;
            result->frac = __dsub_ru(x->frac, scalbn(y->frac, -dexp));
        } else if (dexp < 0) {
            result->exp = y->exp;
            result->frac = __dsub_ru(scalbn(x->frac, dexp), y->frac);
        } else {
            result->exp = (x->exp == 0) ? y->exp : x->exp;
            result->frac = __dsub_ru(x->frac, y->frac);
        }
        cuda::er_adjust(result);
    }

    /*!
     * Subtraction of two extended-range numbers in rounding-down mode
     */
    DEVICE_CUDA_FORCEINLINE void er_sub_rd(er_float_ptr result, er_float_ptr x, er_float_ptr y) {
        x->exp *= (x->frac != 0);
        y->exp *= (y->frac != 0);
        int dexp = (x->exp - y->exp) * (x->frac != 0) * (y->frac != 0);
        if (dexp > 0) {
            result->exp = x->exp;
            result->frac = __dsub_rd(x->frac, scalbn(y->frac, -dexp));
        } else if (dexp < 0) {
            result->exp = y->exp;
            result->frac = __dsub_rd(scalbn(x->frac, dexp), y->frac);
        } else {
            result->exp = (x->exp == 0) ? y->exp : x->exp;
            result->frac = __dsub_rd(x->frac, y->frac);
        }
        cuda::er_adjust(result);
    }

    /*!
     * Multiplication of two extended-range numbers
     */
    DEVICE_CUDA_FORCEINLINE void er_mul(er_float_ptr result, er_float_ptr x, er_float_ptr y) {
        result->exp = x->exp + y->exp;
        result->frac = x->frac * y->frac;
        cuda::er_adjust(result);
    }

    /*!
     * Division of two extended-range numbers
     */
    DEVICE_CUDA_FORCEINLINE void er_div(er_float_ptr result, er_float_ptr x, er_float_ptr y) {
        result->exp = x->exp - y->exp;
        result->frac = x->frac / y->frac;
        cuda::er_adjust(result);
    }

    /*!
     * Compute x * y / z with a single adjust call in rounding-up mode
     */
    DEVICE_CUDA_FORCEINLINE void er_md_ru(er_float_ptr result, er_float_ptr x, er_float_ptr y, er_float_ptr z) {
        result->exp = x->exp + y->exp - z->exp;
        result->frac = __ddiv_ru(__dmul_ru(x->frac, y->frac), z->frac);
        cuda::er_adjust(result);
    }

    /*!
     * Compute x * y / z with a single adjust call in rounding-down mode
     */
    DEVICE_CUDA_FORCEINLINE void er_md_rd(er_float_ptr result, er_float_ptr x, er_float_ptr y, er_float_ptr z) {
        result->exp = x->exp + y->exp - z->exp;
        result->frac = __ddiv_rd(__dmul_rd(x->frac, y->frac), z->frac);
        cuda::er_adjust(result);
    }

    /*!
     * Comparison of two extended-range numbers
     * Returns 1, if x > y; -1, if x < y; 0, if x = y
     * The function is correct only when 1 <= |x.frac|, |y.frac| < 2
     */
    DEVICE_CUDA_FORCEINLINE int er_cmp(er_float_ptr x, er_float_ptr y) {
        int diff = x->exp - y->exp;
        if (diff == 0) {
            return (x->frac == y->frac ? 0 : x->frac > y->frac ? 1 : -1);
        }
        if (x->frac == 0) {
            return (y->frac > 0 ? -1 : 1); // x < y when y is positive
        }
        if (y->frac == 0) {
            return (x->frac > 0 ? 1 : -1); // x > y when x is positive
        }
        return (diff > 0 && x->frac > 0) || (diff < 0 && y->frac < 0) ? 1 : -1;
    }

    /*!
   * Comparison of two non-negative extended-range numbers
   * Performs faster than er_cmp
   * Returns 1, if x > y; -1, if x < y; 0, if x = y
   * The function is correct only when 1 <= x.frac, y.frac < 2
   */
    DEVICE_CUDA_FORCEINLINE int er_ucmp(er_float_ptr x, er_float_ptr y) {
        int diff = x->exp - y->exp;
        if (diff == 0) {
            return x->frac == y->frac ? 0 : x->frac > y->frac ? 1 : -1;
        }
        return x->frac != 0 && (diff > 0 || (diff < 0 && y->frac == 0)) ? 1 : -1;
    }

} //end of namespace


#endif //MPRES_EXTRANGE_CUH
