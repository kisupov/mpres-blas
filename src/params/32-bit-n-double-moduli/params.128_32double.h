/*
 *  Global parameters of MPRES-BLAS.
 *  128 moduli of the RNS system, each of 32 bits
 *  3394-bit dynamic range of the RNS
 *  1696 bits of precision (32-double)
 *
 *  Copyright 2020 by Konstantin Isupov.
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

#define RNS_MODULI_PRODUCT_LOG2 (3394)

#define RNS_P2_SCALING_THRESHOLD (30)

#define RNS_EVAL_RELATIVE_ERROR (0.0000001)

#define RNS_MODULI_VALUES {95955537,95955539,95955541,95955545,95955547,95955551,95955553,95955557, \
                           95955569,95955571,95955577,95955581,95955583,95955589,95955593,95955599, \
                           95955611,95955613,95955617,95955619,95955623,95955631,95955637,95955641, \
                           95955647,95955649,95955653,95955659,95955661,95955677,95955679,95955683, \
                           95955697,95955707,95955709,95955721,95955731,95955733,95955737,95955749, \
                           95955767,95955773,95955779,95955781,95955787,95955791,95955793,95955817, \
                           95955823,95955829,95955841,95955857,95955859,95955863,95955869,95955877, \
                           95955883,95955887,95955901,95955907,95955913,95955929,95955941,95955947, \
                           95955953,95955961,95955967,95955971,95955991,95955997,95956001,95956013, \
                           95956019,95956027,95956033,95956037,95956039,95956043,95956057,95956073, \
                           95956079,95956087,95956097,95956103,95956117,95956121,95956123,95956139, \
                           95956141,95956153,95956163,95956169,95956171,95956181,95956183,95956193, \
                           95956199,95956219,95956229,95956241,95956247,95956249,95956271,95956283, \
                           95956291,95956303,95956309,95956321,95956337,95956339,95956351,95956363, \
                           95956369,95956373,95956379,95956387,95956397,95956409,95956411,95956417, \
                           95956433,95956451,95956453,95956457,95956459,95956463,95956481,95956489}

constexpr int RNS_MODULI[RNS_MODULI_SIZE] = RNS_MODULI_VALUES;

#define EMPLOY_STD_FMA false

#endif  //MPRES_PARAMS_H
