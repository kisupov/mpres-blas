/*
 *  Global parameters of MPRES-BLAS.
 *  64 moduli of the RNS system, each of 32 bits
 *  1698-bit dynamic range of the RNS
 *  848 bits of precision (16-double)
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

#define RNS_MODULI_SIZE (64)

#define RNS_PARALLEL_REDUCTION_IDX (32)

#define RNS_MODULI_PRODUCT_LOG2 (1698)

#define RNS_P2_SCALING_THRESHOLD (30)

#define RNS_EVAL_RELATIVE_ERROR (0.0000001)

const int RNS_MODULI[] = {
        96997537,96997539,96997541,96997543,96997547,96997549,96997553,96997555,
        96997559,96997561,96997567,96997573,96997577,96997583,96997591,96997597,
        96997603,96997609,96997613,96997619,96997627,96997633,96997639,96997643,
        96997651,96997657,96997661,96997669,96997673,96997687,96997697,96997709,
        96997711,96997723,96997727,96997741,96997751,96997753,96997759,96997763,
        96997777,96997781,96997783,96997787,96997799,96997807,96997811,96997819,
        96997847,96997861,96997867,96997871,96997883,96997889,96997891,96997909,
        96997913,96997919,96997921,96997931,96997933,96997939,96997949,96997951
};

#define EMPLOY_STD_FMA false

#endif  //MPRES_PARAMS_H
