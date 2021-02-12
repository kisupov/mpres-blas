/*
 *  Global parameters of MPRES-BLAS.
 *  128 moduli of the RNS system, each of 32 bits
 *  3842-bit dynamic range of the RNS
 *  1920 bits of precision
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

#define RNS_MODULI_SIZE (128)

#define RNS_MODULI_PRODUCT_LOG2 (3842)

#define RNS_P2_SCALING_THRESHOLD (30)

#define RNS_EVAL_RELATIVE_ERROR (0.0000001)

#define RNS_MODULI_VALUES {1085471863,1085471865,1085471867,1085471869,1085471873,1085471879,1085471881,1085471887, \
                           1085471893,1085471897,1085471899,1085471903,1085471909,1085471911,1085471917,1085471921, \
                           1085471923,1085471927,1085471929,1085471939,1085471941,1085471951,1085471953,1085471963, \
                           1085471969,1085471971,1085471977,1085471983,1085471987,1085471993,1085471999,1085472001, \
                           1085472007,1085472013,1085472019,1085472023,1085472041,1085472043,1085472053,1085472061, \
                           1085472067,1085472077,1085472079,1085472103,1085472107,1085472119,1085472121,1085472127, \
                           1085472131,1085472133,1085472137,1085472139,1085472149,1085472161,1085472163,1085472169, \
                           1085472173,1085472181,1085472187,1085472197,1085472203,1085472209,1085472217,1085472221, \
                           1085472229,1085472239,1085472251,1085472257,1085472259,1085472263,1085472277,1085472301, \
                           1085472313,1085472317,1085472319,1085472329,1085472331,1085472341,1085472343,1085472359, \
                           1085472361,1085472371,1085472391,1085472397,1085472431,1085472433,1085472439,1085472449, \
                           1085472461,1085472463,1085472467,1085472469,1085472481,1085472497,1085472503,1085472517, \
                           1085472529,1085472539,1085472541,1085472547,1085472571,1085472587,1085472601,1085472611, \
                           1085472617,1085472623,1085472629,1085472631,1085472643,1085472659,1085472667,1085472671, \
                           1085472677,1085472691,1085472709,1085472721,1085472727,1085472733,1085472743,1085472749, \
                           1085472757,1085472761,1085472767,1085472769,1085472799,1085472803,1085472809,1085472821}

constexpr int RNS_MODULI[RNS_MODULI_SIZE] = RNS_MODULI_VALUES;

#define EMPLOY_STD_FMA false

#endif  //MPRES_PARAMS_H
