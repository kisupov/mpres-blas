/*
 *  Test for validating the mp_sub routines
 *
 *  Copyright 2021 by Konstantin Isupov.
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


#include "../../src/arith/sub.cuh"
#include "../../src/arith/assign.cuh"

static __global__ void testCudaSub(mp_float_ptr dz, mp_float_ptr dx, mp_float_ptr dy){
    cuda::mp_sub(dz, dx[0], dy[0]);
}

int main() {
    rns_const_init();
    mp_const_init();
    rns_const_print(true);
    rns_eval_const_print();

    mp_float_t x, y, z;
    mp_set_d(&x, 1.1e-10);
    mp_set_d(&y, 1.1e10);

    printf("\nARG X = %.16f", mp_get_d(x));
    //mp_print(&x);
    //printf("\n");

    printf("\nARG Y = %.16f", mp_get_d(y));
    //mp_print(&y);
    //printf("\n");

    //CPU SUB
    mp_sub(&z, x, y);

    //mp_print(&z);
    //printf("\n");

    mp_float_ptr dx;
    mp_float_ptr dy;
    mp_float_ptr dz;
    cudaMalloc(&dx, sizeof(mp_float_t));
    cudaMalloc(&dy, sizeof(mp_float_t));
    cudaMalloc(&dz, sizeof(mp_float_t));
    cudaMemcpy(dx, &x, sizeof(mp_float_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dy, &y, sizeof(mp_float_t), cudaMemcpyHostToDevice);

    //CUDA SUB
    testCudaSub<<<1,1>>>(dz, dx, dy);
    mp_set_d(&z, 0.0);
    cudaMemcpy(&z, dz, sizeof(mp_float_t), cudaMemcpyDeviceToHost);
    printf("\nCUDA RESULT = %.16f", mp_get_d(z));
    //mp_print(&z);
    //printf("\n");

    return 0;
}