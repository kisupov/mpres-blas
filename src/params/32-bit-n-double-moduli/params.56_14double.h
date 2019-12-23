/*
 *  Global parameters of MPRES-BLAS.
 *  56 moduli of the RNS system, each of 32 bits
 *  1486-bit dynamic range of the RNS
 *  742 bits of precision (14-double)
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

#define RNS_MODULI_SIZE (56)

#define RNS_PARALLEL_REDUCTION_IDX (32)

#define RNS_MODULI_PRODUCT_LOG2 (1486)

#define RNS_P2_SCALING_THRESHOLD (30)

#define RNS_EVAL_ACCURACY (0.0000001)

#define RNS_EVAL_MIN_LIMIT (6.9633188104489818e-06)

#define RNS_EVAL_OFFSET_VEC_SIZE (98)

const int RNS_MODULI[] = {
        97291787,97291789,97291791,97291793,97291795,97291799,97291801,97291807,
        97291811,97291813,97291819,97291823,97291829,97291841,97291847,97291849,
        97291853,97291861,97291871,97291877,97291879,97291883,97291889,97291891,
        97291897,97291903,97291907,97291921,97291927,97291933,97291937,97291939,
        97291949,97291963,97291967,97291973,97291979,97291991,97291993,97292009,
        97292017,97292023,97292029,97292033,97292047,97292051,97292053,97292059,
        97292087,97292089,97292099,97292101,97292113,97292119,97292123,97292131
};

#endif  //MPRES_PARAMS_H
