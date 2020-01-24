/*
 *  Global parameters of MPRES-BLAS.
 *  32 moduli of the RNS system, each of 32 bits
 *  962-bit dynamic range of the RNS
 *  480 bits of precision
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

#define RNS_MODULI_PRODUCT_LOG2 (962)

#define RNS_P2_SCALING_THRESHOLD (30)

#define RNS_EVAL_RELATIVE_ERROR (0.0000001)

const int RNS_MODULI[] = {
        1122512117,1122512119,1122512121,1122512123,1122512125,1122512129,1122512131,1122512137,
        1122512141,1122512143,1122512147,1122512149,1122512161,1122512173,1122512177,1122512179,
        1122512189,1122512191,1122512197,1122512201,1122512203,1122512207,1122512213,1122512219,
        1122512221,1122512231,1122512233,1122512239,1122512243,1122512249,1122512257,1122512269
};

#define EMPLOY_STD_FMA false

#endif  //MPRES_PARAMS_H
