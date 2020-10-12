/*
 *  CUMP configuration and common routines for benchmarks.
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

#ifndef MPRES_TEST_CUMP_COMMON_CUH
#define MPRES_TEST_CUMP_COMMON_CUH

#include <stdio.h>
#include "mpfr.h"
#include "cump/cump.cuh"

#define CUMP_MAX_THREADS_PER_BLOCK 1024
using cump::mpf_array_t;

/*
 * Set the elements of an array to zero
 */
__global__ void cump_reset_array(int n, mpf_array_t temp) {
    using namespace cump;
    unsigned int numberIdx =  blockDim.x * blockIdx.x + threadIdx.x;
    while (numberIdx < n) {
        mpf_sub(temp[numberIdx], temp[numberIdx], temp[numberIdx]); // set to zero
        numberIdx +=  gridDim.x * blockDim.x;
    }
}

#endif //MPRES_TEST_CUMP_COMMON_CUH