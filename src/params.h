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
#define RNS_MODULI_SIZE 8

/*
 * Initial index for parallel reduction in loops over the RNS moduli.
 * The largest power of two which strictly less than RNS_MODULI_SIZE
 */
#define RNS_PARALLEL_REDUCTION_IDX 4

/*
 * Binary logarithm of the full RNS moduli product
 */
#define RNS_MODULI_PRODUCT_LOG2 242

/*
 * Maximal power-of-two for one scaling step in the RNS system.
 * It should be such that operations modulo 2^RNS_P2_SCALING_THRESHOLD are performed efficiently.
 */
#define RNS_P2_SCALING_THRESHOLD 30

/*
 * Upper bound for the relative forward error of an RNS interval evaluation
 */
#define RNS_EVAL_ACCURACY 0.0000001

/*
 * The minimum value for the upper bound of an RNS interval evaluation at which a refinement loop is not required.
 * RNS_EVAL_MIN_LIMIT = RNS_MODULI_SIZE * RNS_MODULI_SIZE * pow(2.0, 1 - 53) / RNS_EVAL_ACCURACY
 */
#define RNS_EVAL_MIN_LIMIT 1.4210854715202004e-07

/*
 * The size of the offset vector for an RNS interval evaluation refinement loop.
 * RNS_EVAL_OFFSET_VEC_SIZE = ceil(-(LOG2(M) + log2_psi) / (2 + log2_psi))
 */
#define RNS_EVAL_OFFSET_VEC_SIZE 11

/*
 * The set of RNS moduli
 */
const int RNS_MODULI[] = {1283742825,1283742827,1283742829,1283742833,1283742839,1283742841,1283742847,1283742851};


#endif  //MPRES_PARAMS_H
