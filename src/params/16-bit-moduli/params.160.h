/*
 *  Global parameters of MPRES-BLAS.
 *  160 moduli of the RNS system, each of 16 bits
 *  2402-bit dynamic range of the RNS
 *  1200 bits of precision
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

#define RNS_MODULI_SIZE (160)

#define RNS_PARALLEL_REDUCTION_IDX (128)

#define RNS_MODULI_PRODUCT_LOG2 (2402)

#define RNS_P2_SCALING_THRESHOLD (30)

#define RNS_EVAL_ACCURACY (0.0000001)

#define RNS_EVAL_MIN_LIMIT (5.6843418860808015e-05)

#define RNS_EVAL_OFFSET_VEC_SIZE (198)

const int RNS_MODULI[] = {
        32533,32535,32537,32539,32543,32549,32551,32561,32563,32567,32569,32573,32579,32581,32587,32597,
        32603,32609,32611,32621,32623,32629,32633,32639,32647,32651,32653,32663,32677,32687,32689,32693,
        32701,32707,32713,32717,32719,32729,32731,32743,32749,32761,32771,32777,32779,32783,32789,32797,
        32801,32803,32831,32833,32839,32843,32849,32863,32869,32881,32887,32899,32909,32911,32917,32933,
        32939,32941,32947,32951,32957,32969,32971,32983,32987,32993,32999,33001,33013,33023,33029,33037,
        33043,33049,33053,33071,33073,33083,33091,33101,33107,33109,33113,33119,33127,33149,33151,33161,
        33179,33181,33191,33199,33203,33211,33223,33227,33247,33287,33289,33301,33311,33317,33329,33331,
        33343,33347,33349,33353,33359,33377,33391,33403,33409,33413,33427,33457,33461,33469,33479,33487,
        33493,33503,33521,33529,33533,33547,33563,33569,33577,33581,33587,33589,33599,33601,33613,33617,
        33619,33623,33629,33637,33641,33647,33679,33703,33713,33721,33739,33749,33751,33757,33767,33769
};

#endif  //MPRES_PARAMS_H
