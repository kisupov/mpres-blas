/*
*  Defines and functions for efficient handling double precision
*  floating-point numbers by manipulations of the bits.
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

#ifndef MPRES_BITWISE_CUH
#define MPRES_BITWISE_CUH

#include "common.cuh"

/*
 * Precision of IEEE 754 double precision arithmetic (binary64)
 */
#define DBL_PRECISION (53)

/*
 * Sign offset in IEEE 754 double precision (binary64)
 */
#define DBL_SIGN_OFFSET (63)

/*
 * Exponent bits offset in IEEE 754 double precision (binary64)
 */
#define DBL_EXP_OFFSET (52)

/*
 * Exponent bias in IEEE 754 double precision (binary64)
 */
#define DBL_EXP_BIAS (1023)

/*
 * The distance from 1 to the next smaller floating-point number, 2^{-53}
 */
#define DBL_EPS scalbn(1, -53)

/*
 * The smallest positive subnormal floating-point number, 1 * 2^{-1074}
 */
#define DBL_ETA scalbn(1, -1074)

/*
 * Auxiliary numeric constant for double-precision interval arithmetic
 */
#define DBL_PHI1 DBL_EPS*(1+2*DBL_EPS)

/*
 * When the logical AND operator (&) is applied to the double x with this mask (i.e. x & DBL_ZERO_EXP_MASK),
 * all exponent bits of x are set to zero, while the sign bit and the significand bits of x do not change.
 */
#define DBL_ZERO_EXP_MASK   ((((uint64_t) 1 << DBL_EXP_OFFSET) - 1) | ((uint64_t) 1 << DBL_SIGN_OFFSET))

/*
 * When the logical OR operator (|) is applied to the double x with this mask (i.e. x | DBL_BIAS_EXP_MASK),
 * DBL_EXP_BIAS is added to the exponent bits of x, while the sign bit and the significand bits of x do not change.
 */
#define DBL_BIAS_EXP_MASK   ((uint64_t) DBL_EXP_BIAS << DBL_EXP_OFFSET)

/*
 * Union for bitwise operations on IEEE 754 double precision floats
 */
union RealIntUnion {
    double dvalue;
    uint64_t ivalue;
};

/*!
 * Multiplies a floating point value x by 2^n
 * This procedure has proven faster than scalbn from <cmath>
 * This is a bitwise operation that can produce an incorrect result when n is extremely large (n < -1022 or n > 1023).
 * The result is guaranteed to be correct only if -1023 < n < 1024 AND 1 <= |x| < 2
 * @param x - extended-range float whose significand, x.frac, needs to be scaled
 * @param n - exponent of the integer power of two in the range (-1023, 1024)
 * @return - if no errors (overflow / underflow) occur, x multiplied by 2 to the power of n (x * 2^n) is returned.
 */
GCC_FORCEINLINE double fast_scalbn(const double x, const long n) {
    RealIntUnion diu;
    diu.dvalue = x;
    diu.ivalue += (uint64_t) n << DBL_EXP_OFFSET; //Add pow to the exponent
    return diu.dvalue * (x != 0 && n >= -1023);
}

/**
 * Returns the sign of a floating-point variable x
 * NaNs, infinities, and denormals unhandled
 * @param x - input value
 * @return - sign of x
 */
GCC_FORCEINLINE int dbl_get_sign(const double x) {
    RealIntUnion u = {x};
    return (int) (u.ivalue >> DBL_SIGN_OFFSET);
}

/**
 * Returns the unbiased exponent of a floating-point variable x corresponding to the integral significand
 * NaNs, infinities, and denormals unhandled
 * @param x - input value
 * @return - exponent of x
 */
GCC_FORCEINLINE int dbl_get_exp(const double x) {
    RealIntUnion u = {x};
    return ( (u.ivalue >> DBL_EXP_OFFSET) & (DBL_EXP_BIAS * 2 + 1) ) - (DBL_EXP_BIAS + DBL_EXP_OFFSET) * (x != 0);
}

/**
 * Returns the integral significand of a floating-point variable x
 * NaNs, infinities, and denormals unhandled
 * @param x - input value
 * @return - integral significand of x
 */
GCC_FORCEINLINE long dbl_get_significand(const double x) {
    RealIntUnion u = {x};
    return (u.ivalue & ((long) 1 << DBL_EXP_OFFSET) - 1 | (long) 1 << DBL_EXP_OFFSET) * (x != 0);
    //TODO: Проверить, может быть на CUDA быстрее будет так как ниже
    /*
        long significand;
        int exp = (u.ivalue >> DBL_EXP_OFFSET) & (DBL_EXP_BIAS * 2 + 1);
          if (exp == 0) {
            significand = (u.ivalue & ((long) 1 << DBL_EXP_OFFSET) - 1);
        } else {
            significand = (u.ivalue & ((long) 1 << DBL_EXP_OFFSET) - 1 | (long) 1 << DBL_EXP_OFFSET);
        }
        return significand;
     */
}

/**
 * Returns the sign, exponent, and integral significand (mantissa) of x
 * such that x = -1^s * significand * 2^exp, where significand is an integer
 * NaNs, infinities, and denormals unhandled
 */
GCC_FORCEINLINE void dbl_decompose(const double x, int& sign, int& exp, long& significand) {
    RealIntUnion u = {x};
    //Extraction of the sign, exponent and significand from the union
    sign = (int) (u.ivalue >> DBL_SIGN_OFFSET);
    exp  = ( (u.ivalue >> DBL_EXP_OFFSET) & (DBL_EXP_BIAS * 2 + 1) ) - (DBL_EXP_BIAS + DBL_EXP_OFFSET) * (x != 0);
    significand = (u.ivalue & ((long) 1 << DBL_EXP_OFFSET) - 1 | (long) 1 << DBL_EXP_OFFSET) * (x != 0);
}

namespace cuda {

    /*!
     * Multiplies a floating point value x by 2^n
     * Perhaps this procedure is faster than scalbn from the CUDA Math API
     * This is a bitwise operation that can produce an incorrect result when n is extremely large (n < -1022 or n > 1023).
     * The result is guaranteed to be correct only if -1023 < n < 1024 AND 1 <= |x| < 2
     * @param x - extended-range float whose significand, x.frac, needs to be scaled
     * @param n - exponent of the integer power of two in the range (-1023, 1024)
     * @return - if no errors (overflow / underflow) occur, x multiplied by 2 to the power of n (x * 2^n) is returned.
     */
     DEVICE_CUDA_FORCEINLINE double fast_scalbn(const double x, const long n) {
        RealIntUnion diu;
        diu.dvalue = x;
        diu.ivalue += (uint64_t) n << DBL_EXP_OFFSET;        // Add pow to exponent
        return diu.dvalue * (x != 0 && n >= -1023);
    }

    /**
     * Returns the sign of a floating-point variable x
     * NaNs, infinities, and denormals unhandled
     * @param x - input value
     * @return - sign of x
     */
    DEVICE_CUDA_FORCEINLINE int dbl_get_sign(const double x) {
        RealIntUnion u = {x};
        return (int) (u.ivalue >> DBL_SIGN_OFFSET);
    }

    /**
     * Returns the unbiased exponent of a floating-point variable x corresponding to the integral significand
     * NaNs, infinities, and denormals unhandled
     * @param x - input value
     * @return - exponent of x
     */
    DEVICE_CUDA_FORCEINLINE int dbl_get_exp(const double x) {
        RealIntUnion u = {x};
        return ( (u.ivalue >> DBL_EXP_OFFSET) & (DBL_EXP_BIAS * 2 + 1) ) - (DBL_EXP_BIAS + DBL_EXP_OFFSET) * (x != 0);
    }

    /**
     * Returns the integral significand of a floating-point variable x
     * NaNs, infinities, and denormals unhandled
     * @param x - input value
     * @return - integral significand of x
     */
    DEVICE_CUDA_FORCEINLINE long dbl_get_significand(const double x) {
        RealIntUnion u = {x};
        return (u.ivalue & ((long) 1 << DBL_EXP_OFFSET) - 1 | (long) 1 << DBL_EXP_OFFSET) * (x != 0);
        /*
            long significand;
            int exp = (u.ivalue >> DBL_EXP_OFFSET) & (DBL_EXP_BIAS * 2 + 1);
              if (exp == 0) {
                significand = (u.ivalue & ((long) 1 << DBL_EXP_OFFSET) - 1);
            } else {
                significand = (u.ivalue & ((long) 1 << DBL_EXP_OFFSET) - 1 | (long) 1 << DBL_EXP_OFFSET);
            }
            return significand;
         */
    }

    /**
     * Returns the sign, exponent, and integral significand (mantissa) of x
     * such that x = -1^s * significand * 2^exp, where significand is an integer
     * NaNs, infinities, and denormals unhandled
     */
    DEVICE_CUDA_FORCEINLINE void dbl_decompose(const double x, int& sign, int& exp, long& significand) {
        RealIntUnion u = {x};
        //Extraction of the sign, exponent and significand from the union
        sign = (int) (u.ivalue >> DBL_SIGN_OFFSET);
        exp  = ( (u.ivalue >> DBL_EXP_OFFSET) & (DBL_EXP_BIAS * 2 + 1) ) - (DBL_EXP_BIAS + DBL_EXP_OFFSET) * (x != 0);
        significand = (u.ivalue & ((long) 1 << DBL_EXP_OFFSET) - 1 | (long) 1 << DBL_EXP_OFFSET) * (x != 0);
    }

} //end of namespace


#endif //MPRES_BITWISE_CUH
