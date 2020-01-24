/*
 *  Global parameters of MPRES-BLAS.
 *  32 moduli of the RNS system, each of 16 bits
 *  482-bit dynamic range of the RNS
 *  240 bits of precision
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

#define RNS_MODULI_PRODUCT_LOG2 (482)

#define RNS_P2_SCALING_THRESHOLD (30)

#define RNS_EVAL_RELATIVE_ERROR (0.0000001)

const int RNS_MODULI[] = {
        34403,34405,34407,34409,34411,34417,34421,34423,34427,34429,34439,34441,34451,34453,34457,34459,
        34469,34471,34477,34483,34487,34499,34501,34511,34513,34519,34537,34543,34547,34549,34553,34571
};

#define EMPLOY_STD_FMA false

#endif  //MPRES_PARAMS_H
