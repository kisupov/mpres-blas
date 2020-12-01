/*
 *  Global parameters of MPRES-BLAS.
 *  112 moduli of the RNS system, each of 16 bits
 *  1682-bit dynamic range of the RNS
 *  840 bits of precision
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

#define RNS_MODULI_SIZE (112)

#define RNS_MODULI_PRODUCT_LOG2 (1682)

#define RNS_P2_SCALING_THRESHOLD (30)

#define RNS_EVAL_RELATIVE_ERROR (0.0000001)

const int RNS_MODULI[] = {
        32869,32871,32873,32875,32879,32881,32887,32891,32897,32899,32903,32909,32911,32917,32927,32933,
        32939,32941,32947,32951,32957,32959,32969,32971,32983,32987,32993,32999,33013,33017,33023,33029,
        33037,33043,33049,33053,33067,33071,33073,33083,33091,33101,33107,33109,33113,33119,33127,33133,
        33149,33151,33161,33169,33179,33181,33191,33197,33199,33203,33211,33217,33221,33223,33227,33247,
        33277,33287,33289,33301,33311,33317,33329,33331,33343,33347,33349,33353,33359,33361,33377,33379,
        33391,33403,33409,33413,33427,33431,33443,33457,33461,33463,33469,33479,33487,33493,33503,33521,
        33529,33533,33547,33559,33563,33569,33577,33581,33587,33589,33599,33601,33613,33617,33619,33623
};

#define EMPLOY_STD_FMA false

#endif  //MPRES_PARAMS_H
