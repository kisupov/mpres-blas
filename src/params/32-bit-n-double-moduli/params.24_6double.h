/*
 *  Global parameters of MPRES-BLAS.
 *  24 moduli of the RNS system, each of 32 bits
 *  638-bit dynamic range of the RNS
 *  318 bits of precision (6-double)
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

#define RNS_MODULI_SIZE (24)

#define RNS_PARALLEL_REDUCTION_IDX (16)

#define RNS_MODULI_PRODUCT_LOG2 (638)

#define RNS_P2_SCALING_THRESHOLD (30)

#define RNS_EVAL_RELATIVE_ERROR (0.0000001)

#define RNS_EVAL_MIN_LIMIT (1.2789769243681803e-06)

#define RNS_EVAL_OFFSET_VEC_SIZE (36)

const int RNS_MODULI[] = {
        100772103,100772105,100772107,100772111,100772113,100772117,100772123,100772129,
        100772131,100772137,100772149,100772153,100772159,100772167,100772171,100772173,
        100772179,100772183,100772197,100772201,100772207,100772213,100772233,100772237
};

#endif  //MPRES_PARAMS_H
