/*
 *  Global parameters of MPRES-BLAS.
 *  80 moduli of the RNS system, each of 32 bits
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

#define RNS_MODULI_SIZE (80)

#define RNS_MODULI_PRODUCT_LOG2 (2402)

#define RNS_P2_SCALING_THRESHOLD (30)

#define RNS_EVAL_RELATIVE_ERROR (0.0000001)

#define RNS_MODULI_VALUES {1092715871,1092715873,1092715875,1092715877,1092715879,1092715883,1092715889,1092715891, \
                           1092715901,1092715903,1092715907,1092715909,1092715913,1092715919,1092715927,1092715931, \
                           1092715933,1092715937,1092715951,1092715961,1092715963,1092715969,1092715973,1092715979, \
                           1092715991,1092715999,1092716003,1092716017,1092716029,1092716033,1092716041,1092716047, \
                           1092716057,1092716059,1092716063,1092716069,1092716077,1092716081,1092716083,1092716099, \
                           1092716101,1092716111,1092716113,1092716137,1092716147,1092716159,1092716161,1092716171, \
                           1092716173,1092716189,1092716197,1092716203,1092716213,1092716221,1092716231,1092716243, \
                           1092716249,1092716257,1092716263,1092716267,1092716281,1092716291,1092716299,1092716309, \
                           1092716311,1092716321,1092716323,1092716329,1092716347,1092716351,1092716357,1092716369, \
                           1092716371,1092716377,1092716381,1092716393,1092716399,1092716411,1092716419,1092716431}

constexpr int RNS_MODULI[RNS_MODULI_SIZE] = RNS_MODULI_VALUES;

#define EMPLOY_STD_FMA false

#endif  //MPRES_PARAMS_H
