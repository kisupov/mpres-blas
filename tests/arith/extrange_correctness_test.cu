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
    mul_div_test,
    cmp_test,
    ucmp_test,
    scalbn_test
};

__host__ __device__
static void printResult(const  char * name, er_float_ptr result){
    double d;
    er_get_d(&d, result);
    printf("%s = %.8f \t\t", name, d);
    er_print(result);
    printf("\n");

}

/*
 * GPU tests
 */

static __global__ void testCudaAdd(er_float_ptr dr, er_float_ptr dx, er_float_ptr dy){
    cuda::er_add_rd(dr, dx, dy);
    printResult("[CUDA-rd] x + y", dr);
    cuda::er_add(dr, dx, dy);
    printResult("[CUDA]    x + y", dr);
    cuda::er_add_ru(dr, dx, dy);
    printResult("[CUDA-ru] x + y", dr);
}

static __global__ void testCudaSub(er_float_ptr dr, er_float_ptr dx, er_float_ptr dy){
    cuda::er_sub_rd(dr, dx, dy);
    printResult("[CUDA-rd] x - y", dr);
    cuda::er_sub(dr, dx, dy);
    printResult("[CUDA]    x - y", dr);
    cuda::er_sub_ru(dr, dx, dy);
    printResult("[CUDA-ru] x - y", dr);
}

static __global__ void testCudaMul(er_float_ptr dr, er_float_ptr dx, er_float_ptr dy){
    cuda::er_mul_rd(dr, dx, dy);
    printResult("[CUDA-rd] x * y", dr);
    cuda::er_mul(dr, dx, dy);
    printResult("[CUDA]    x * y", dr);
    cuda::er_mul_ru(dr, dx, dy);
    printResult("[CUDA-ru] x * y", dr);
}

static __global__ void testCudaDiv(er_float_ptr dr, er_float_ptr dx, er_float_ptr dy){
    cuda::er_div_rd(dr, dx, dy);
    printResult("[CUDA-rd] x / y", dr);
    cuda::er_div(dr, dx, dy);
    printResult("[CUDA]    x / y", dr);
    cuda::er_div_ru(dr, dx, dy);
    printResult("[CUDA-ru] x / y", dr);
}

static __global__ void testCudaMulDiv(er_float_ptr dr, er_float_ptr dx, er_float_ptr dy, er_float_ptr dz){
    cuda::er_md_rd(dr, dx, dy, dz);
    printResult("[CUDA-rd] x * y / z", dr);
    cuda::er_md_ru(dr, dx, dy, dz);
    printResult("[CUDA-ru] x * y / z", dr);
}


/*
static __global__ void testCudaCmp(er_ptr d_x, er_ptr d_y){
    int i = cuda::er_cmp(d_x,d_y);
    printf("[CUDA] cmp:   %s\n", (i == -1 ? "x < y" : i == 0 ? "x = y" : i == 1 ? "x > y" : "incorrect cmp"));
}

static __global__ void testCudaUcmp(er_ptr d_x, er_ptr d_y){
    int i = cuda::er_ucmp(d_x,d_y);
    printf("[CUDA] ucmp:  %s\n", (i == -1 ? "x < y" : i == 0 ? "x = y" : i == 1 ? "x > y" : "incorrect cmp"));
}
*/


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

    er_float_ptr x = new er_float_t;
    er_float_ptr y = new er_float_t;
    er_float_ptr z = new er_float_t;
    er_float_ptr r = new er_float_t;

    er_set_d(x, arg_x);
    er_set_d(y, arg_y);
    er_set_d(z, arg_z);

    Logger::printDash();
    printf("x = %.8f\n", arg_x);
    printf("y = %.8f\n", arg_y);
    printf("z = %.8f\n", arg_z);

    Logger::printDash();
    printf("\nEXACT           = %.8f \t\t", arg_x + arg_y);
    er_set_d(r, arg_x + arg_y);
    er_print(r);
    printf("\n");
    er_add_rd(r, x, y);
    printResult("[CPU-rd]  x + y", r);
    er_add(r, x, y);
    printResult("[CPU]     x + y", r);
    er_add_ru(r, x, y);
    printResult("[CPU-ru]  x + y", r);
    Logger::printSpace();
    testCuda(x, y, z, add_test);
    //--------------------------------------------------------------
    Logger::printSpace();
    Logger::printDash();
    printf("\nEXACT           = %.8f \t\t", arg_x - arg_y);
    er_set_d(r, arg_x - arg_y);
    er_print(r);
    printf("\n");
    er_sub_rd(r, x, y);
    printResult("[CPU-rd]  x - y", r);
    er_sub(r, x, y);
    printResult("[CPU]     x - y", r);
    er_sub_ru(r, x, y);
    printResult("[CPU-ru]  x - y", r);
    Logger::printSpace();
    testCuda(x, y,  z, sub_test);
    //--------------------------------------------------------------
    Logger::printSpace();
    Logger::printDash();
    printf("\nEXACT           = %.8f \t\t", arg_x * arg_y);
    er_set_d(r, arg_x * arg_y);
    er_print(r);
    printf("\n");
    er_mul_rd(r, x, y);
    printResult("[CPU-rd]  x * y", r);
    er_mul(r, x, y);
    printResult("[CPU]     x * y", r);
    er_mul_ru(r, x, y);
    printResult("[CPU-ru]  x * y", r);
    Logger::printSpace();
    testCuda(x, y, z, mul_test);
    //--------------------------------------------------------------
    Logger::printSpace();
    Logger::printDash();
    printf("\nEXACT           = %.8f \t\t", arg_x / arg_y);
    er_set_d(r, arg_x / arg_y);
    er_print(r);
    printf("\n");
    er_div_rd(r, x, y);
    printResult("[CPU-rd]  x / y", r);
    er_div(r, x, y);
    printResult("[CPU]     x / y", r);
    er_div_ru(r, x, y);
    printResult("[CPU-ru]  x / y", r);
    Logger::printSpace();
    testCuda(x, y,  z, div_test);
    //--------------------------------------------------------------
    Logger::printSpace();
    Logger::printDash();
    printf("\nEXACT               = %.8f \t\t", arg_x * arg_y / arg_z);
    er_set_d(r, arg_x * arg_y / arg_z);
    er_print(r);
    printf("\n");
    er_md_rd(r, x, y, z);
    printResult("[CPU-rd]  x * y / z", r);
    er_md_ru(r, x, y, z);
    printResult("[CPU-ru]  x * y / z", r);
    Logger::printSpace();
    testCuda(x, y,  z, mul_div_test);


    /*PRINT_RULE;
    er_sub(z, x, y, MPRES_RNDF);
    er_get_d(&t, z);
    std::cout << "[CPU]  x - y = " << t << std::tab << std::tab;
    er_print(z);
    testCuda(x, y, sub_test);
    std::cout << "exact =        " << arg_x - arg_y << std::endl;

    PRINT_RULE;
    er_mul(z, x, y, MPRES_RNDF);
    er_get_d(&t, z);
    std::cout << "[CPU]  x * y = " << t << std::tab << std::tab;
    er_print(z);
    testCuda(x, y, mul_test);
    std::cout << "exact =        " << arg_x * arg_y << std::endl;

    PRINT_RULE;
    er_div(z, x, y, MPRES_RNDF);
    er_get_d(&t, z);
    std::cout << "[CPU]  x / y = " << t << std::tab << std::tab;
    er_print(z);
    testCuda(x, y, div_test);
    std::cout << "exact =        " << arg_x / arg_y << std::endl;

    PRINT_RULE;
    int i = er_cmp(x,y);
    std::cout << "[CPU]  cmp:   " << (i == -1 ? "x < y" : i == 0 ? "x = y" : i == 1 ? "x > y" : "incorrect cmp") << std::endl;
    testCuda(x, y, cmp_test);
    std::cout << "exact =       " << (arg_x < arg_y ? "x < y" : arg_x == arg_y ? "x = y" : "x > y") << std::endl;

    PRINT_RULE;
    i = er_ucmp(x,y);
    std::cout << "[CPU]  ucmp:  " << (i == -1 ? "x < y" : i == 0 ? "x = y" : i == 1 ? "x > y" : "incorrect cmp") << std::endl;
    testCuda(x, y, ucmp_test);
    std::cout << "exact =       " << (arg_x < arg_y ? "x < y" : arg_x == arg_y ? "x = y" : "x > y") << std::endl;

    PRINT_RULE;
    int n = 1023;
    x->frac = -1;
    std::cout << "[CPU] fast_scalbn("<< x->frac << ", " << n << ") = " << fast_scalbn(x->frac, n)  << std::endl;
    std::cout << "exact = " << x->frac * pow(2, n) << std::endl;
    PRINT_RULE;*/
}

int main() {
    Logger::beginTestDescription(Logger::RNS_EVAL_ACCURACY_TEST);
    Logger::printSpace();
    base_test();

    //End logging
    Logger::endTestDescription();
    return 0;
}