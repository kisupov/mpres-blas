/*
 *  Global parameters of MPRES-BLAS.
 *  64 moduli of the RNS system, each of 32 bits
 *  1922-bit dynamic range of the RNS
 *  960 bits of precision
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

#define RNS_MODULI_PRODUCT_LOG2 (1922)

#define RNS_P2_SCALING_THRESHOLD (30)

#define RNS_EVAL_ACCURACY (0.0000001)

#define RNS_EVAL_MIN_LIMIT (9.0949470177292824e-06)

#define RNS_EVAL_OFFSET_VEC_SIZE (130)

const int RNS_MODULI[] = {
        1097551117,1097551119,1097551121,1097551123,1097551127,1097551129,1097551135,1097551139,
        1097551141,1097551151,1097551153,1097551157,1097551159,1097551163,1097551171,1097551177,
        1097551181,1097551183,1097551187,1097551193,1097551199,1097551201,1097551207,1097551211,
        1097551213,1097551219,1097551223,1097551237,1097551241,1097551243,1097551253,1097551261,
        1097551267,1097551271,1097551277,1097551289,1097551291,1097551303,1097551309,1097551319,
        1097551327,1097551333,1097551337,1097551339,1097551349,1097551361,1097551363,1097551369,
        1097551379,1097551381,1097551387,1097551391,1097551409,1097551439,1097551447,1097551451,
        1097551453,1097551457,1097551459,1097551463,1097551471,1097551489,1097551493,1097551517
};

#endif  //MPRES_PARAMS_H
