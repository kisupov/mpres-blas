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

/*
 * Return the rough size  in bytes of the CUMP n-element nbit-precision array (cumpf_array_t)
 */
size_t cump_get_array_size_in_bytes(size_t n, int nbit){
    return n * sizeof (cump_limb_t) * __CUMPF_ARRAY_ELEMSIZE (__CUMPF_BITS_TO_PREC(nbit));
}

/*
 * Return the rough size  in MB of the CUMP n-element nbit-precision array (cumpf_array_t)
 */
double cump_get_array_size_in_mb(size_t n, int nbit){
    return (double)(n * sizeof (cump_limb_t)) * (double)(__CUMPF_ARRAY_ELEMSIZE (__CUMPF_BITS_TO_PREC(nbit))) / double(1024 * 1024);
}

#endif //MPRES_TEST_CUMP_COMMON_CUH