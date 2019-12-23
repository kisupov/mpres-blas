/*
 *  Global parameters of MPRES-BLAS.
 *  48 moduli of the RNS system, each of 32 bits
 *  1442-bit dynamic range of the RNS
 *  720 bits of precision
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

#define RNS_MODULI_PRODUCT_LOG2 (1442)

#define RNS_P2_SCALING_THRESHOLD (30)

#define RNS_EVAL_ACCURACY (0.0000001)

#define RNS_EVAL_MIN_LIMIT (5.1159076974727213e-06)

#define RNS_EVAL_OFFSET_VEC_SIZE (92)

const int RNS_MODULI[] = {
        1105651103,1105651105,1105651107,1105651109,1105651111,1105651117,1105651121,1105651123,
        1105651127,1105651133,1105651139,1105651147,1105651153,1105651159,1105651163,1105651171,
        1105651177,1105651181,1105651187,1105651189,1105651199,1105651201,1105651207,1105651213,
        1105651223,1105651229,1105651231,1105651237,1105651241,1105651243,1105651247,1105651249,
        1105651259,1105651271,1105651273,1105651277,1105651279,1105651289,1105651291,1105651297,
        1105651307,1105651319,1105651333,1105651343,1105651357,1105651367,1105651369,1105651373
};

#endif  //MPRES_PARAMS_H
