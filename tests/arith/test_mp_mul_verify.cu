/*
 *  Test for validating the mp_mul routines
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


#include "../../src/arith/mpmul.cuh"
#include "../../src/arith/mpassign.cuh"
#include "../../src/mparray.cuh"

static __global__ void testCudaMul(mp_float_ptr dz, mp_float_ptr dx, mp_float_ptr dy){
    cuda::mp_mul(dz, dx, dy);
}

static __global__ void testCudaMuld(mp_float_ptr dz, mp_float_ptr dx, const double dy){
    cuda::mp_mul_d(dz, dx, dy);
}

int main() {
   rns_const_init();
    mp_const_init();
    rns_const_print(true);
    rns_eval_const_print();

    double dblx = 1;
    double dbly = -1;

    mp_float_t x, y, z;
    mp_set_d(&x, dblx);
    mp_set_d(&y, dbly);

    printf("\nARG X = %.16f", mp_get_d(&x));
    mp_print(&x);
    printf("\n");

    printf("\nARG Y = %.16f", mp_get_d(&y));
    mp_print(&y);
    printf("\n");

    mp_set_d(&z, 0);
    mp_mul(&z, &x, &y);
    printf("\nCPU mp_mul = %.16f", mp_get_d(&z));
    mp_print(&z);
    printf("\n");

    mp_set_d(&z, 0);
    mp_mul_d(&z, &x, dbly);
    printf("\nCPU mp_mul_d = %.16f", mp_get_d(&z));
    mp_print(&z);
    printf("\n");


    mp_float_ptr dx;
    mp_float_ptr dy;
    mp_float_ptr dz;
    cudaMalloc(&dx, sizeof(mp_float_t));
    cudaMalloc(&dy, sizeof(mp_float_t));
    cudaMalloc(&dz, sizeof(mp_float_t));
    cudaMemcpy(dx, &x, sizeof(mp_float_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dy, &y, sizeof(mp_float_t), cudaMemcpyHostToDevice);

    //CUDA ADD 1
    testCudaMul<<<1,1>>>(dz, dx, dy);
    mp_set_d(&z, 0.0);
    cudaMemcpy(&z, dz, sizeof(mp_float_t), cudaMemcpyDeviceToHost);
    printf("\nCUDA mp_mul = %.16f", mp_get_d(&z));
    mp_print(&z);
    printf("\n");

    //CUDA ADD 2
    testCudaMuld<<<1,1>>>(dz, dx, dbly);
    mp_set_d(&z, 0.0);
    cudaMemcpy(&z, dz, sizeof(mp_float_t), cudaMemcpyDeviceToHost);
    printf("\nCUDA mp_mul_d = %.16f", mp_get_d(&z));
    mp_print(&z);
    printf("\n");
}