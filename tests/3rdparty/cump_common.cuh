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
size_t get_cump_array_size_in_bytes(size_t n, int nbit){
    return n * sizeof (cump_limb_t) * __CUMPF_ARRAY_ELEMSIZE (__CUMPF_BITS_TO_PREC(nbit));
}

/*
 * Return the rough size  in MB of the CUMP n-element nbit-precision array (cumpf_array_t)
 */
double get_cump_array_size_in_mb(size_t n, int nbit){
    return (double)(n * sizeof (cump_limb_t)) * (double)(__CUMPF_ARRAY_ELEMSIZE (__CUMPF_BITS_TO_PREC(nbit))) / double(1024 * 1024);
}

/*
 * Returns the memory consumption of a double precision CSR structure and returns the total size of the structure in MB
 */
double get_cump_csr_memory_consumption(const int m, const int nnz, const int prec){
    double sizeOfAs = get_cump_array_size_in_mb(nnz, prec);
    double sizeOfCsr = sizeOfAs + get_int_array_size_in_mb(nnz) + get_int_array_size_in_mb(m + 1);
    return sizeOfCsr;
}

/*
 * Returns the memory consumption of a double precision ELLPACK structure and returns the total size of the structure in MB
 */
double get_cump_ell_memory_consumption(const int m, const int maxnzr, const int prec){
    double sizeOfAs = get_cump_array_size_in_mb(m * maxnzr, prec);
    double sizeOfEll = sizeOfAs + get_int_array_size_in_mb(m * maxnzr);
    return sizeOfEll;
}

/*
 * Returns the memory consumption of a double precision JAD structure and returns the total size of the structure in MB
 */
double get_cump_jad_memory_consumption(const int m, const int n, const int nnz, const int maxnzr, const int prec){
    double sizeOfAs = get_cump_array_size_in_mb(nnz, prec);
    double sizeOfJad = sizeOfAs + get_int_array_size_in_mb(nnz) + get_int_array_size_in_mb(maxnzr + 1)+ get_int_array_size_in_mb(m);
    return sizeOfJad;
}

/*
 * Returns the memory consumption of a double precision DIA structure and returns the total size of the structure in MB
 */
double get_cump_dia_memory_consumption(const int m, const int ndiag, const int prec){
    double sizeOfAs = get_cump_array_size_in_mb(m * ndiag, prec);
    double sizeOfDia = sizeOfAs + get_int_array_size_in_mb(ndiag);
    return sizeOfDia;
}


#endif //MPRES_TEST_CUMP_COMMON_CUH