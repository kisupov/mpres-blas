/*
 *  Test for validating the mp_cmp routines
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


#include "../../src/arith/cmp.cuh"
#include "../../src/arith/assign.cuh"

static __global__ void performCudaCmp(mp_float_ptr dx, mp_float_ptr dy){
    printf("\n[CUDA] Result = %i", cuda::mp_cmp(dx[0], dy[0]));
}

static void testCuda(mp_float_ptr x, mp_float_ptr y){
    mp_float_ptr dx;
    mp_float_ptr dy;
    cudaMalloc(&dx, sizeof(mp_float_t));
    cudaMalloc(&dy, sizeof(mp_float_t));
    cudaMemcpy(dx, x, sizeof(mp_float_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dy, y, sizeof(mp_float_t), cudaMemcpyHostToDevice);
    performCudaCmp<<<1,1>>>(dx, dy);
    cudaFree(dx);
    cudaFree(dy);
}

int main() {
    rns_const_init();
    mp_const_init();

    mp_float_t x, y;
    mp_set_d(&x, 0.550000001);
    mp_set_d(&y, 0.550000000);
    printf("\n x = %.16f", mp_get_d(x));
    printf("\n y = %.16f", mp_get_d(y));
    printf("\n[CPU ] Result = %i", mp_cmp(x, y));
    testCuda(&x, &y);
    return 0;
}