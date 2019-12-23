/*
 *  Global parameters of MPRES-BLAS.
 *  16 moduli of the RNS system, each of 32 bits
 *  426-bit dynamic range of the RNS
 *  212 bits of precision (4-double)
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

#define RNS_MODULI_SIZE (16)

#define RNS_PARALLEL_REDUCTION_IDX (8)

#define RNS_MODULI_PRODUCT_LOG2 (426)

#define RNS_P2_SCALING_THRESHOLD (30)

#define RNS_EVAL_ACCURACY (0.0000001)

#define RNS_EVAL_MIN_LIMIT (5.6843418860808015e-07)

#define RNS_EVAL_OFFSET_VEC_SIZE (22)

const int RNS_MODULI[] = {103812103,103812105,103812107,103812109,103812113,103812119,103812127,103812131,
                          103812133,103812137,103812139,103812151,103812157,103812161,103812167,103812169
};

#endif  //MPRES_PARAMS_H
