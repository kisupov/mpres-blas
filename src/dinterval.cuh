/*
 *  Simulation of double precision interval arithmetic operations
 *  with fixed rounding mode (rounding to nearest) according to
 *  Algorithm 4 (BoundNear1) proposed in:
 *  Siegfried M. Rump, Takeshi Ogita, Yusuke Morikura, Shin'ichi Oishi,
 *  Interval arithmetic with fixed rounding mode //
 *  Nonlinear Theory and Its Applications, IEICE, 2016, Volume 7, Issue 3, Pages 362-373,
 *  https://doi.org/10.1587/nolta.7.362
 *
 *  Copyright 2020 by Konstantin Isupov.
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


#ifndef MPRES_DINTERVAL_CUH
#define MPRES_DINTERVAL_CUH

#include "bitwise.cuh"
#include <cmath>

/*!
 *	Computing the predecessor of a + b (addition with rounding downwards)
 *	Returns pred(a + b) that less than or equal to (a + b)
 */
GCC_FORCEINLINE double dadd_rd(double a, double b){
    double c = a + b;
    return c - std::fma(DBL_PHI1, std::abs(c), DBL_ETA);
}

/*!
 *	Computing the successor of a + b (addition with rounding upwards).
 *	Returns succ(a + b) greater than or equal to (a + b)
 */
GCC_FORCEINLINE double dadd_ru(double a, double b){
    double c = a + b;
    return c + std::fma(DBL_PHI1, std::abs(c), DBL_ETA);
}

/*!
 *	Enclosing interval of a + b. Returns the interval [low, upp] that includes (a + b),
 *	where low and upp are the pointers to the result lower and upper bounds, respectively.
 */
GCC_FORCEINLINE void dadd_rdu(double * low, double * upp, double a, double b){
    double c = a + b;
    double e = std::fma(DBL_PHI1, std::abs(c), DBL_ETA);
    *low = c - e;
    *upp = c + e;
}

/*!
 *	Computing the predecessor of a + b (subtraction with rounding downwards)
 *	Returns pred(a - b) that less than or equal to (a - b)
 */
GCC_FORCEINLINE double dsub_rd(double a, double b){
    double c = a - b;
    return c - std::fma(DBL_PHI1, std::abs(c), DBL_ETA);
}

/*!
 *	Computing the successor of a - b (subtraction with rounding upwards)
 *	Returns succ(a - b) greater than or equal to (a - b)
 */
GCC_FORCEINLINE double dsub_ru(double a, double b){
    double c = a - b;
    return c + std::fma(DBL_PHI1, std::abs(c), DBL_ETA);
}

/*!
 *	Enclosing interval of a - b. Returns the interval [low, upp] that includes (a - b),
 *	where low and upp are the pointers to the result lower and upper bounds, respectively.
 */
GCC_FORCEINLINE void dsub_rdu(double * low, double * upp, double a, double b){
    double c = a - b;
    double e = std::fma(DBL_PHI1, std::abs(c), DBL_ETA);
    *low = c - e;
    *upp = c + e;
}

/*!
 *	Computing the predecessor of a * b (multiplication with rounding downwards)
 *	Returns pred(a * b) that less than or equal to (a * b)
 */
GCC_FORCEINLINE double dmul_rd(double a, double b){
    double c = a * b;
    return c - std::fma(DBL_PHI1, std::abs(c), DBL_ETA);
}

/*!
 *	Computing the successor of a * b (multiplication with rounding upwards)
 *	Returns succ(a * b) greater than or equal to (a * b)
 */
GCC_FORCEINLINE double dmul_ru(double a, double b){
    double c = a * b;
    return c + std::fma(DBL_PHI1, std::abs(c), DBL_ETA);
}

/*!
 *	Enclosing interval of a * b. Returns the interval [low, upp] that includes (a * b),
 *	where low and upp are the pointers to the result lower and upper bounds, respectively.
 */
GCC_FORCEINLINE void dmul_rdu(double * low, double * upp, double a, double b){
    double c = a * b;
    double e = std::fma(DBL_PHI1, std::abs(c), DBL_ETA);
    *low = c - e;
    *upp = c + e;
}

/*!
 *	Computing the predecessor of a / b (division with rounding downwards)
 *	Returns pred(a / b) that less than or equal to (a / b)
 */
GCC_FORCEINLINE double ddiv_rd(double a, double b){
    double c = a / b;
    return c - std::fma(DBL_PHI1, std::abs(c), DBL_ETA);
}

/*!
 *	Computing the successor of a / b (division with rounding upwards)
 *	Returns succ(a / b) greater than or equal to (a / b)
 */
GCC_FORCEINLINE double ddiv_ru(double a, double b){
    double c = a / b;
    return c + std::fma(DBL_PHI1, std::abs(c), DBL_ETA);
}

/*!
 *	Enclosing interval of a / b. Returns the interval [low, upp] that includes (a/b),
 *	where low and upp are the pointers to the result lower and upper bounds, respectively.
 */
GCC_FORCEINLINE void ddiv_rdu(double * low, double * upp, double a, double b){
    double c = a / b;
    double e = std::fma(DBL_PHI1, std::abs(c), DBL_ETA);
    *low = c - e;
    *upp = c + e;
}

#endif //MPRES_DINTERVAL_CUH
