/*
 *  Global parameters of MPRES-BLAS.
 *  144 moduli of the RNS system, each of 32 bits
 *  4322-bit dynamic range of the RNS
 *  2160 bits of precision
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

#define RNS_MODULI_SIZE (144)

#define RNS_MODULI_PRODUCT_LOG2 (4322)

#define RNS_P2_SCALING_THRESHOLD (30)

#define RNS_EVAL_RELATIVE_ERROR (0.0000001)

#define RNS_MODULI_VALUES {1084171475,1084171477,1084171479,1084171481,1084171483,1084171487,1084171489,1084171493, \
                           1084171499,1084171507,1084171511,1084171513,1084171517,1084171519,1084171523,1084171531, \
                           1084171537,1084171541,1084171547,1084171553,1084171567,1084171573,1084171577,1084171579, \
                           1084171589,1084171591,1084171601,1084171603,1084171609,1084171619,1084171633,1084171637, \
                           1084171639,1084171643,1084171657,1084171661,1084171663,1084171667,1084171679,1084171681, \
                           1084171687,1084171691,1084171703,1084171709,1084171721,1084171729,1084171741,1084171757, \
                           1084171783,1084171789,1084171793,1084171799,1084171801,1084171807,1084171811,1084171817, \
                           1084171819,1084171831,1084171841,1084171853,1084171859,1084171867,1084171871,1084171877, \
                           1084171883,1084171909,1084171919,1084171927,1084171937,1084171951,1084171961,1084171969, \
                           1084171973,1084171981,1084171987,1084171993,1084171997,1084172003,1084172009,1084172017, \
                           1084172021,1084172027,1084172041,1084172051,1084172057,1084172069,1084172071,1084172087, \
                           1084172099,1084172107,1084172119,1084172123,1084172147,1084172149,1084172153,1084172179, \
                           1084172189,1084172191,1084172197,1084172203,1084172209,1084172213,1084172227,1084172231, \
                           1084172233,1084172237,1084172239,1084172249,1084172251,1084172267,1084172273,1084172291, \
                           1084172303,1084172317,1084172329,1084172333,1084172347,1084172351,1084172363,1084172371, \
                           1084172377,1084172381,1084172387,1084172407,1084172417,1084172423,1084172429,1084172431, \
                           1084172437,1084172443,1084172447,1084172491,1084172503,1084172513,1084172519,1084172521, \
                           1084172539,1084172543,1084172549,1084172561,1084172563,1084172587,1084172591,1084172599}

constexpr int RNS_MODULI[RNS_MODULI_SIZE] = RNS_MODULI_VALUES;

#define EMPLOY_STD_FMA false

#endif  //MPRES_PARAMS_H
