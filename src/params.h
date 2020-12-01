/*
 *  Global parameters of MPRES-BLAS.
 *  8 moduli of the RNS system, each of 32 bits
 *  242-bit dynamic range of the RNS
 *  120 bits of precision
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

/*
 * Size of the RNS moduli set
 */
#define RNS_MODULI_SIZE (8)

/*
 * Binary logarithm of the full RNS moduli product
 */
#define RNS_MODULI_PRODUCT_LOG2 (242)

/*
 * Maximal power-of-two for one scaling step in the RNS system.
 * It should be such that operations modulo 2^RNS_P2_SCALING_THRESHOLD are performed efficiently.
 */
#define RNS_P2_SCALING_THRESHOLD (30)

/*
 * Upper bound for the relative forward error of an RNS interval evaluation
 */
#define RNS_EVAL_RELATIVE_ERROR (0.0000001)

/*
 * The set of RNS moduli
 */
const int RNS_MODULI[] = {1283742825,1283742827,1283742829,1283742833,1283742839,1283742841,1283742847,1283742851};

/*
 * Specifies whether to use std::fma to compute (x * y) + z
 */
#define EMPLOY_STD_FMA false

#endif  //MPRES_PARAMS_H
