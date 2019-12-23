/*
 *  Global parameters of MPRES-BLAS.
 *  48 moduli of the RNS system, each of 16 bits
 *  722-bit dynamic range of the RNS
 *  360 bits of precision
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

#ifndef MPRES_PARAMS_H
#define MPRES_PARAMS_H

#define RNS_MODULI_SIZE (48)

#define RNS_PARALLEL_REDUCTION_IDX (32)

#define RNS_MODULI_PRODUCT_LOG2 (722)

#define RNS_P2_SCALING_THRESHOLD (30)

#define RNS_EVAL_ACCURACY (0.0000001)

#define RNS_EVAL_MIN_LIMIT (5.1159076974727213e-06)

#define RNS_EVAL_OFFSET_VEC_SIZE (46)

const int RNS_MODULI[] = {
        33721,33723,33725,33727,33731,33733,33737,33739,33743,33749,33751,33757,33767,33769,33773,33779,
        33787,33791,33793,33797,33799,33809,33811,33821,33823,33827,33829,33841,33851,33853,33857,33863,
        33871,33889,33893,33899,33911,33919,33923,33931,33937,33941,33947,33961,33967,33973,33997,34003
};

#endif  //MPRES_PARAMS_H
