/*
 *  Global parameters of MPRES-BLAS.
 *  96 moduli of the RNS system, each of 32 bits
 *  2882-bit dynamic range of the RNS
 *  1440 bits of precision
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

#define RNS_MODULI_SIZE (96)

#define RNS_MODULI_PRODUCT_LOG2 (2882)

#define RNS_P2_SCALING_THRESHOLD (30)

#define RNS_EVAL_RELATIVE_ERROR (0.0000001)

#define RNS_MODULI_VALUES {1089541111,1089541113,1089541115,1089541117,1089541121,1089541123,1089541127,1089541129, \
                           1089541133,1089541139,1089541147,1089541151,1089541153,1089541157,1089541163,1089541169, \
                           1089541171,1089541177,1089541181,1089541183,1089541199,1089541207,1089541213,1089541217, \
                           1089541223,1089541231,1089541241,1089541247,1089541249,1089541253,1089541259,1089541261, \
                           1089541273,1089541283,1089541289,1089541291,1089541303,1089541307,1089541309,1089541319, \
                           1089541333,1089541337,1089541351,1089541361,1089541363,1089541367,1089541379,1089541381, \
                           1089541391,1089541393,1089541399,1089541403,1089541417,1089541421,1089541423,1089541429, \
                           1089541433,1089541447,1089541457,1089541459,1089541463,1089541477,1089541483,1089541489, \
                           1089541493,1089541499,1089541501,1089541507,1089541511,1089541547,1089541559,1089541561, \
                           1089541573,1089541589,1089541591,1089541597,1089541601,1089541603,1089541619,1089541631, \
                           1089541633,1089541637,1089541657,1089541669,1089541693,1089541699,1089541703,1089541709, \
                           1089541711,1089541721,1089541727,1089541729,1089541741,1089541751,1089541753,1089541759}

constexpr int RNS_MODULI[RNS_MODULI_SIZE] = RNS_MODULI_VALUES;

#define EMPLOY_STD_FMA false

#endif  //MPRES_PARAMS_H
