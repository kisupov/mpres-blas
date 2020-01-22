/*
 *  Global parameters of MPRES-BLAS.
 *  32 moduli of the RNS system, each of 32 bits
 *  850-bit dynamic range of the RNS
 *  424 bits of precision (8-double)
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

#define RNS_MODULI_SIZE (32)

#define RNS_PARALLEL_REDUCTION_IDX (16)

#define RNS_MODULI_PRODUCT_LOG2 (850)

#define RNS_P2_SCALING_THRESHOLD (30)

#define RNS_EVAL_RELATIVE_ERROR (0.0000001)

const int RNS_MODULI[] = {
        99312103,99312105,99312107,99312109,99312113,99312119,99312121,99312127,
        99312131,99312133,99312137,99312139,99312143,99312149,99312151,99312163,
        99312167,99312173,99312179,99312193,99312197,99312203,99312211,99312217,
        99312223,99312233,99312247,99312253,99312263,99312271,99312277,99312287
};

#endif  //MPRES_PARAMS_H
