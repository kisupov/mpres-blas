/*
 *  Test for validating the mp_add routine
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


#include "../../src/arith/mpadd.cuh"
#include "../../src/arith/mpassign.cuh"
#include "../../src/mparray.cuh"

static __global__ void testCudaAdd(mp_float_ptr dz, mp_float_ptr dx, mp_float_ptr dy){
    cuda::mp_add(dz, dx, dy);
}

static __global__ void testCudaAdd2(mp_float_ptr dz, mp_float_ptr dx, mp_array_t dy){
    cuda::mp_add(dz, dx, dy, 0);
}

static __global__ void testCudaAdd3(mp_array_t dz, mp_array_t dx, mp_float_ptr dy){
    cuda::mp_add(dz, 0, dx, 0, dy);
}


int main() {
    rns_const_init();
    mp_const_init();
    rns_const_print(true);
    rns_eval_const_print();

    mp_float_t x, y, z;
    mp_set_d(&x, -1000.01);
    mp_set_d(&y, 10000.000000002);

    printf("\nARG X = %lf", mp_get_d(&x));
    mp_print(&x);
    printf("\n");

    printf("\nARG Y = %lf", mp_get_d(&y));
    mp_print(&y);
    printf("\n");

    mp_add(&z, &x, &y);
    printf("\nCPU RESULT = %lf", mp_get_d(&z));
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

    mp_array_t arrx;
    mp_array_t arry;
    mp_array_t arrz;
    cuda::mp_array_init(arrx, 1);
    cuda::mp_array_init(arry, 1);
    cuda::mp_array_init(arrz, 1);
    cuda::mp_array_host2device(arrx, &x, 1);
    cuda::mp_array_host2device(arry, &y, 1);

    //CUDA ADD 1
    testCudaAdd<<<1,1>>>(dz, dx, dy);
    mp_set_d(&z, 0.0);
    cudaMemcpy(&z, dz, sizeof(mp_float_t), cudaMemcpyDeviceToHost);
    printf("\nCUDA RESULT 1 = %lf", mp_get_d(&z));
    mp_print(&z);
    printf("\n");

    //CUDA ADD 2
    testCudaAdd2<<<1,1>>>(dz, dx, arry);
    mp_set_d(&z, 0.0);
    cudaMemcpy(&z, dz, sizeof(mp_float_t), cudaMemcpyDeviceToHost);
    printf("\nCUDA RESULT 2 = %lf", mp_get_d(&z));
    mp_print(&z);
    printf("\n");

    testCudaAdd3<<<1,1>>>(arrz, arrx, dy);
    mp_set_d(&z, 0.0);
    cuda::mp_array_device2host(&z,arrz,1);
    printf("\nCUDA RESULT 3 = %lf", mp_get_d(&z));
    mp_print(&z);
    printf("\n");

    return 0;
}