/*
 *  Global parameters of MPRES-BLAS.
 *  24 moduli of the RNS system, each of 32 bits
 *  722-bit dynamic range of the RNS
 *  360 bits of precision
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

#define RNS_MODULI_SIZE (24)

#define RNS_MODULI_PRODUCT_LOG2 (722)

#define RNS_P2_SCALING_THRESHOLD (30)

#define RNS_EVAL_RELATIVE_ERROR (0.0000001)

const int RNS_MODULI[] = {
        1140141131,1140141133,1140141135,1140141137,1140141139,1140141143,1140141151,1140141157,
        1140141161,1140141167,1140141169,1140141173,1140141179,1140141181,1140141187,1140141193,
        1140141197,1140141199,1140141203,1140141209,1140141217,1140141221,1140141229,1140141239
};

#define EMPLOY_STD_FMA false

#endif  //MPRES_PARAMS_H
