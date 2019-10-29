/*
 *  Global parameters of MPRES-BLAS.
 *  80 moduli of the RNS system, each of 16 bits
 *  1202-bit dynamic range of the RNS
 *  600 bits of precision
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

#define RNS_MODULI_SIZE 80

#define RNS_PARALLEL_REDUCTION_IDX 64

#define RNS_MODULI_PRODUCT_LOG2 1202

#define RNS_P2_SCALING_THRESHOLD 30

#define RNS_EVAL_ACCURACY 0.0000001

#define RNS_EVAL_MIN_LIMIT 1.4210854715202004e-05

#define RNS_EVAL_OFFSET_VEC_SIZE 85

const int RNS_MODULI[] = {
        33117,33119,33121,33125,33127,33133,33137,33139,33149,33151,33157,33161,33169,33179,33181,33191,
        33197,33199,33203,33211,33217,33221,33223,33227,33233,33239,33247,33259,33277,33281,33287,33289,
        33301,33311,33317,33323,33329,33331,33343,33347,33349,33353,33359,33361,33367,33377,33379,33389,
        33391,33401,33403,33409,33413,33427,33431,33433,33457,33461,33463,33469,33479,33487,33493,33503,
        33521,33529,33533,33547,33559,33563,33569,33577,33581,33587,33589,33599,33601,33613,33617,33619
};

#endif  //MPRES_PARAMS_H
