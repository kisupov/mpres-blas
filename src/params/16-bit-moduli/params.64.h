/*
 *  Global parameters of MPRES-BLAS.
 *  64 moduli of the RNS system, each of 16 bits
 *  962-bit dynamic range of the RNS
 *  480 bits of precision
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

#define RNS_MODULI_SIZE (64)

#define RNS_PARALLEL_REDUCTION_IDX (32)

#define RNS_MODULI_PRODUCT_LOG2 (962)

#define RNS_P2_SCALING_THRESHOLD (30)

#define RNS_EVAL_RELATIVE_ERROR (0.0000001)

#define RNS_EVAL_MIN_LIMIT (9.0949470177292824e-06)

#define RNS_EVAL_OFFSET_VEC_SIZE (65)

const int RNS_MODULI[] = {
        33357,33359,33361,33365,33367,33371,33373,33377,33379,33383,33389,33391,33401,33403,33407,33409,
        33413,33427,33431,33433,33443,33457,33461,33463,33469,33479,33487,33491,33493,33499,33503,33521,
        33529,33533,33547,33559,33563,33569,33571,33577,33581,33587,33589,33599,33601,33613,33617,33619,
        33623,33629,33637,33641,33647,33659,33661,33667,33679,33697,33703,33713,33721,33731,33739,33749
};

#endif  //MPRES_PARAMS_H
