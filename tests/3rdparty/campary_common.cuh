/*
 *  CAMPARY configuration and common routines for benchmarks.
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

#ifndef MPRES_TEST_CAMPARY_COMMON_CUH
#define MPRES_TEST_CAMPARY_COMMON_CUH

#include "mpfr.h"
#include "campary/Doubles/src_gpu/multi_prec.h"
#include "../../src/params.h"

/*
 * Precision of CAMPARY in n-double
 * For predefined RNS moduli sets from the src/32-bit-n-double-moduli/ directory:
 * 8 moduli give 2-double, 16 moduli give 4-double, 24 moduli give 6-double, etc.
 */
#define CAMPARY_PRECISION (RNS_MODULI_SIZE / 4)

//Execution configuration
#define CAMPARY_REDUCTION_BLOCKS 1024
#define CAMPARY_REDUCTION_THREADS 32
#define CAMPARY_VECTOR_MULTIPLY_THREADS 32
#define CAMPARY_MATRIX_THREADS_X 32
#define CAMPARY_MATRIX_THREADS_Y 8

// Printing the result, which is a CAMPARY's floating-point expansion (ONE multiple precision number)
// prec specifies the number of terms (precision), i.e. the size of the floating point expansion
template<int nterms>
static void printResult(multi_prec<nterms> result){
    int p = 8192;
    mpfr_t x;
    mpfr_t r;
    mpfr_init2(x, p);
    mpfr_init2(r, p);
    mpfr_set_d(r, 0.0, MPFR_RNDN);
    for(int i = nterms - 1; i >= 0; i--){
        mpfr_set_d(x, result.getData()[i], MPFR_RNDN);
        mpfr_add(r, r, x, MPFR_RNDN);
    }
    mpfr_printf("result: %.70Rf \n", r);
    /* printf("RAW Data:\n");
    result.prettyPrint(); */
    mpfr_clear(x);
    mpfr_clear(r);
}

#endif //MPRES_TEST_CAMPARY_COMMON_CUH