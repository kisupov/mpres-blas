/*
 *  Global parameters of MPRES-BLAS.
 *  48 moduli of the RNS system, each of 32 bits
 *  1274-bit dynamic range of the RNS
 *  636 bits of precision (12-double)
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

#define RNS_MODULI_PRODUCT_LOG2 (1274)

#define RNS_P2_SCALING_THRESHOLD (30)

#define RNS_EVAL_ACCURACY (0.0000001)

#define RNS_EVAL_MIN_LIMIT (5.1159076974727213e-06)

#define RNS_EVAL_OFFSET_VEC_SIZE (81)

const int RNS_MODULI[] = {
        97712999,97713001,97713003,97713005,97713007,97713011,97713013,97713017,
        97713019,97713023,97713029,97713037,97713041,97713043,97713047,97713053,
        97713059,97713061,97713071,97713073,97713079,97713089,97713097,97713101,
        97713103,97713107,97713113,97713131,97713137,97713139,97713149,97713151,
        97713163,97713167,97713169,97713173,97713179,97713191,97713193,97713197,
        97713199,97713211,97713221,97713227,97713233,97713247,97713251,97713277
};

#endif  //MPRES_PARAMS_H
