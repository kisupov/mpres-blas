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
extern "C"
GCC_FORCEINLINE double fast_scalbn(const double x, const int n) {
    RealIntUnion diu;
    diu.dvalue = x;
    diu.ivalue += (uint64_t) n << DBL_EXP_OFFSET; //Add pow to the exponent
    return diu.dvalue * (x != 0 && n >= -1023);
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
     DEVICE_CUDA_FORCEINLINE double fast_scalbn(const double x, const int n) {
        RealIntUnion diu;
        diu.dvalue = x;
        diu.ivalue += (uint64_t) n << DBL_EXP_OFFSET;        // Add pow to exponent
        return diu.dvalue * (x != 0 && n >= -1023);
    }

} //end of namespace


#endif //MPRES_BITWISE_CUH
