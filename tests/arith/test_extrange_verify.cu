/*
 *  Test for checking the correctness of the extended-range floating-point routines
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


#include "../../src/extrange.cuh"
#include "../logger.cuh"
#include <iostream>

enum er_test_type {
    add_test,
    sub_test,
    mul_test,
    div_test,
    mul_div_test
};

static void printResult(const  char * name, er_float_t result){
    double d = er_get_d(result);
    printf("%s = %.8f \t\t", name, d);
    er_print(result);
    printf("\n");
}

namespace cuda{
    __device__ static void printResult(const  char * name, er_float_t result){
        double d = cuda::er_get_d(result);
        printf("%s = %.8f \t\t", name, d);
        cuda::er_print(result);
        printf("\n");
    }
}


/*
 * GPU tests
 */

static __global__ void testCudaAdd(er_float_ptr dr, er_float_ptr dx, er_float_ptr dy){
    *dr = cuda::er_add_rd(dx[0], dy[0]);
    cuda::printResult("[CUDA-rd] x + y", dr[0]);
    *dr = cuda::er_add(dx[0], dy[0]);
    cuda::printResult("[CUDA]    x + y", dr[0]);
    *dr = cuda::er_add_ru(dx[0], dy[0]);
    cuda::printResult("[CUDA-ru] x + y", dr[0]);
}

static __global__ void testCudaSub(er_float_ptr dr, er_float_ptr dx, er_float_ptr dy){
    *dr = cuda::er_sub_rd(dx[0], dy[0]);
    cuda::printResult("[CUDA-rd] x - y", dr[0]);
    *dr = cuda::er_sub(dx[0], dy[0]);
    cuda::printResult("[CUDA]    x - y", dr[0]);
    *dr = cuda::er_sub_ru(dx[0], dy[0]);
    cuda::printResult("[CUDA-ru] x - y", dr[0]);
}

static __global__ void testCudaMul(er_float_ptr dr, er_float_ptr dx, er_float_ptr dy){
    *dr = cuda::er_mul_rd(dx[0], dy[0]);
    cuda::printResult("[CUDA-rd] x * y", dr[0]);
    *dr = cuda::er_mul(dx[0], dy[0]);
    cuda::printResult("[CUDA]    x * y", dr[0]);
    *dr = cuda::er_mul_ru(dx[0], dy[0]);
    cuda::printResult("[CUDA-ru] x * y", dr[0]);
}

static __global__ void testCudaDiv(er_float_ptr dr, er_float_ptr dx, er_float_ptr dy){
    *dr = cuda::er_div_rd(dx[0], dy[0]);
    cuda::printResult("[CUDA-rd] x / y", dr[0]);
    *dr = cuda::er_div(dx[0], dy[0]);
    cuda::printResult("[CUDA]    x / y", dr[0]);
    *dr = cuda::er_div_ru(dx[0], dy[0]);
    cuda::printResult("[CUDA-ru] x / y", dr[0]);
}

static __global__ void testCudaMulDiv(er_float_ptr dr, er_float_ptr dx, er_float_ptr dy, er_float_ptr dz){
    *dr = cuda::er_md_rd(dx[0], dy[0], dz[0]);
    cuda::printResult("[CUDA-rd] x * y / z", dr[0]);
    *dr = cuda::er_md_ru(dx[0], dy[0], dz[0]);
    cuda::printResult("[CUDA-ru] x * y / z", dr[0]);
}

void testCuda(er_float_ptr x, er_float_ptr y, er_float_ptr z, er_test_type type){
    er_float_ptr dx;
    er_float_ptr dy;
    er_float_ptr dz;
    er_float_ptr dr;
    cudaMalloc(&dx, sizeof(er_float_t));
    cudaMalloc(&dy, sizeof(er_float_t));
    cudaMalloc(&dz, sizeof(er_float_t));
    cudaMalloc(&dr, sizeof(er_float_t));
    cudaMemcpy(dx, x, sizeof(er_float_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dy, y, sizeof(er_float_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dz, z, sizeof(er_float_t), cudaMemcpyHostToDevice);

    switch(type){
        case add_test:
            testCudaAdd<<<1,1>>>(dr, dx, dy);
            break;
        case sub_test:
            testCudaSub<<<1,1>>>(dr, dx, dy);
            break;
        case mul_test:
            testCudaMul<<<1,1>>>(dr, dx, dy);
            break;
        case div_test:
            testCudaDiv<<<1,1>>>(dr, dx, dy);
            break;
        case mul_div_test:
            testCudaMulDiv<<<1,1>>>(dr, dx, dy, dz);
            break;
        default:
            break;
    }
    cudaFree(dx);
    cudaFree(dy);
    cudaFree(dr);
}


void base_test() {

    double arg_x = 0.00003687544343;
    double arg_y = 0.41352;
    double arg_z = 12345.1443575;

    er_float_t x;
    er_float_t y;
    er_float_t z;
    er_float_t r;

    er_set_d(&x, arg_x);
    er_set_d(&y, arg_y);
    er_set_d(&z, arg_z);

    Logger::printDash();
    printf("x = %.8f\n", arg_x);
    printf("y = %.8f\n", arg_y);
    printf("z = %.8f\n", arg_z);

    Logger::printDash();
    printf("\nEXACT           = %.8f \t\t", arg_x + arg_y);
    er_set_d(&r, arg_x + arg_y);
    er_print(r);
    printf("\n");
    r = er_add_rd(x, y);
    printResult("[CPU-rd]  x + y", r);
    r = er_add(x, y);
    printResult("[CPU]     x + y", r);
    r = er_add_ru(x, y);
    printResult("[CPU-ru]  x + y", r);
    Logger::printSpace();
    testCuda(&x, &y, &z, add_test);
    //--------------------------------------------------------------
    Logger::printSpace();
    Logger::printDash();
    printf("\nEXACT           = %.8f \t\t", arg_x - arg_y);
    er_set_d(&r, arg_x - arg_y);
    er_print(r);
    printf("\n");
    r= er_sub_rd(x, y);
    printResult("[CPU-rd]  x - y", r);
    r= er_sub(x, y);
    printResult("[CPU]     x - y", r);
    r= er_sub_ru(x, y);
    printResult("[CPU-ru]  x - y", r);
    Logger::printSpace();
    testCuda(&x, &y,  &z, sub_test);
    //--------------------------------------------------------------
    Logger::printSpace();
    Logger::printDash();
    printf("\nEXACT           = %.8f \t\t", arg_x * arg_y);
    er_set_d(&r, arg_x * arg_y);
    er_print(r);
    printf("\n");
    r= er_mul_rd(x, y);
    printResult("[CPU-rd]  x * y", r);
    r= er_mul(x, y);
    printResult("[CPU]     x * y", r);
    r= er_mul_ru(x, y);
    printResult("[CPU-ru]  x * y", r);
    Logger::printSpace();
    testCuda(&x, &y, &z, mul_test);
    //--------------------------------------------------------------
    Logger::printSpace();
    Logger::printDash();
    printf("\nEXACT           = %.8f \t\t", arg_x / arg_y);
    er_set_d(&r, arg_x / arg_y);
    er_print(r);
    printf("\n");
    r= er_div_rd(x, y);
    printResult("[CPU-rd]  x / y", r);
    r= er_div(x, y);
    printResult("[CPU]     x / y", r);
    r= er_div_ru(x, y);
    printResult("[CPU-ru]  x / y", r);
    Logger::printSpace();
    testCuda(&x, &y, &z, div_test);
    //--------------------------------------------------------------
    Logger::printSpace();
    Logger::printDash();
    printf("\nEXACT               = %.8f \t\t", arg_x * arg_y / arg_z);
    er_set_d(&r, arg_x * arg_y / arg_z);
    er_print(r);
    printf("\n");
    r= er_md_rd(x, y, z);
    printResult("[CPU-rd]  x * y / z", r);
    r= er_md_ru(x, y, z);
    printResult("[CPU-ru]  x * y / z", r);
    Logger::printSpace();
    testCuda(&x, &y,  &z, mul_div_test);
}

int main() {
    Logger::beginTestDescription(Logger::RNS_EVAL_ACCURACY_TEST);
    Logger::printSpace();
    base_test();
    Logger::endTestDescription();
    return 0;
}