/*
 *  Global parameters of MPRES-BLAS.
 *  40 moduli of the RNS system, each of 32 bits
 *  1062-bit dynamic range of the RNS
 *  530 bits of precision (10-double)
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

#define RNS_MODULI_SIZE (40)

#define RNS_PARALLEL_REDUCTION_IDX (32)

#define RNS_MODULI_PRODUCT_LOG2 (1062)

#define RNS_P2_SCALING_THRESHOLD (30)

#define RNS_EVAL_RELATIVE_ERROR (0.0000001)

#define RNS_EVAL_MIN_LIMIT (3.5527136788005009e-06)

#define RNS_EVAL_OFFSET_VEC_SIZE (65)

const int RNS_MODULI[] = {
        98412003,98412005,98412007,98412011,98412013,98412017,98412019,98412023,
        98412029,98412031,98412037,98412043,98412047,98412049,98412053,98412059,
        98412067,98412071,98412077,98412079,98412089,98412091,98412101,98412103,
        98412107,98412109,98412113,98412121,98412131,98412137,98412143,98412151,
        98412157,98412161,98412163,98412169,98412173,98412179,98412187,98412199
};

#endif  //MPRES_PARAMS_H
