/*
 *  Multiple-precision floating-point division.
 *  In the current version, we rely on MPFR to perform the division.
 *  In the future, we will try to implement internal division algorithms using Newton's iteration or Goldschmidt's division.
 *
 *  Copyright 2021 by Konstantin Isupov.
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

#ifndef MPRES_DIV_CUH
#define MPRES_DIV_CUH

#include "assign.cuh"

/*!
 * Division of two multiple-precision numbers
 * result = x / y
 */
GCC_FORCEINLINE void mp_div(mp_float_ptr result, mp_float_ptr x, mp_float_ptr y) {
    //первый вариант - использовать rns_div
    //второй вариант - newton-rapson
    mpfr_t dividend, divisor;
    mpfr_init2(dividend, MP_PRECISION);
    mpfr_init2(divisor, MP_PRECISION);
    mp_get_mpfr(dividend, x);
    mp_get_mpfr(divisor, y);
    mpfr_div(dividend, dividend, divisor, MPFR_RNDN);
    mp_set_mpfr(result, dividend);
    mpfr_clear(dividend);
    mpfr_clear(divisor);
}

/*
* GPU functions
*/
namespace cuda {
    /*!
     * Division of two multiple-precision numbers
     * result = x / y
     * Note that this is a HOST procedure
     */
    GCC_FORCEINLINE void mp_div(mp_float_ptr result, mp_float_ptr x, mp_float_ptr y) {
        auto dividend = new mp_float_t[1];
        auto divisor = new mp_float_t[1];
        cudaMemcpy(dividend, x, sizeof(mp_float_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(divisor, y, sizeof(mp_float_t), cudaMemcpyDeviceToHost);
        ::mp_div(dividend, dividend, divisor);
        cudaMemcpy(result, dividend, sizeof(mp_float_t), cudaMemcpyHostToDevice);
        delete[] dividend;
        delete[] divisor;
    }
}

#endif //MPRES_DIV_CUH
