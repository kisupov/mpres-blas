/*
 *  Data structures for representing multiple-precision floating-point numbers
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

#ifndef MPRES_TYPES_CUH
#define MPRES_TYPES_CUH

#include "params.h"
#include <limits.h>

/*!
 * Predefined constants
 */

/*
* The largest exponent value in the mp_float_t type
*/
#define MP_EXP_MAX (INT_MAX/2)

/*
* The smallest exponent value in the mp_float_t type
*/
#define MP_EXP_MIN (INT_MIN/2)


/*!
 * Extended-range floating-point representation
 */
typedef struct {
    double frac; // Significand
    long exp;    // Exponent
} er_float_t;

typedef er_float_t * er_float_ptr;


/*!
 * Interval evaluation for the fractional representation of a number represented in the Residue Number System (RNS).
 * We called this 'RNS interval evaluation'
 */
typedef struct {
    er_float_t low; // Lower bound
    er_float_t upp; // Upper bound
} interval_t;

typedef interval_t * interval_ptr;


/*!
 * Multiple-precision floating-point representation
 */
typedef struct {
    int digits[RNS_MODULI_SIZE];  // Significand part of the number in RNS (residues)
    int sign;                     // Sign
    int exp;                      // Exponent
    er_float_t eval[2];           // Interval evaluation of the significand: eval[0] - lower bound, eval[1] - upper bound
} mp_float_t;

typedef mp_float_t * mp_float_ptr;


/*!
 * Data type for representing a general multiple-precision array in the structure-of-arrays layout.
 * This data type contains a 'buf' field for transfer auxiliary variables between computational kernels
 * in addition operations. This data type is intended for use primarily in BLAS kernels (for storing dense matrices and vectors)
 * where addition operations may be required.
 */
typedef struct {
    int * digits;        // Significand parts of the numbers in RNS (residues): [all residues of x1]...[all residues of x_N]
    int * sign;          // Signs: [sign of x1][sign of x2]...[sign of x_N]
    int * exp;           // Exponents: [exp of x1][exp of x2]...[exp of x_N]
    er_float_t * eval;   // Interval evaluations: [low bound of x1]...[low bound of x_N] [upp bound of x1]...[upp bound of x_N]
    int4 * buf;          // Temporary buffer: buf[idx].x = gamma, buf[idx].y = theta, buf[idx].z = factor_x, buf[idx].w = factor_y
    int * len;           // Length of the vector
} mp_array_t;

/*!
 * Data type for representing a multiple-precision array in the structure-of-arrays layout without 'buf' and 'len' fields.
 * This type is intended for use primarily in sparse linear algebra kernels (for storing a sparse matrix)
 * where addition operations are usually not required.
 */
typedef struct {
    int * digits;        // Significand parts of the numbers in RNS (residues): [all residues of x1]...[all residues of x_N]
    int * sign;          // Signs: [sign of x1][sign of x2]...[sign of x_N]
    int * exp;           // Exponents: [exp of x1][exp of x2]...[exp of x_N]
    er_float_t * eval;   // Interval evaluations: [low bound of x1]...[low bound of x_N] [upp bound of x1]...[upp bound of x_N]
} mp_collection_t;


/*!
 * Jagged Diagonals (JAD) structure for representing a sparse matrix
 */
typedef struct {
    double * as;  // Nonzero matrix entries
    int * ja;     // Column indices
    int * jcp;    // Column start pointers
    int * perm;   // Row permutation
} jad_t;

#endif //MPRES_TYPES_CUH