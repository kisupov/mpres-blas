/*
 *  Test for checking the algorithms that calculate the interval evaluation of an RNS number
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

#include <stdio.h>
#include <iostream>
#include "../../src/rns.cuh"
#include "../tsthelper.cuh"
#include "../logger.cuh"

/*
 *  Printing the error of the computed interval evaluation with respect
 *  to the exact relative value of an RNS number
 */
void printError(interval_ptr eval, er_float_ptr exact) {
    std::cout << "\n\t eval_low  = ";
    er_print(&eval->low);
    std::cout << "\t eval_upp  = ";
    er_print(&eval->upp);

    er_adjust(exact);
    if((er_cmp(&eval->low, exact) == 1) || (er_cmp(exact, &eval->upp) == 1)){
        std::cout << "\t error    = 100%. The RNS Interval Evaluation is wrong!";
    }
    else{
        er_float_ptr error = new er_float_t[1];
        er_sub(error, &eval->upp, &eval->low);
        er_div(error, error, exact);
        double derror;
        er_get_d(&derror, error);
        std::cout << "\t rel.error    = " << (derror);
        delete error;
    }
}

void resetResult(interval_ptr eval){
    er_set_d(&eval->low, 0.0);
    er_set_d(&eval->upp, 0.0);
}

__global__ void resetResultCuda(interval_ptr eval) {
    cuda::er_set_d(&eval->low, 0.0);
    cuda::er_set_d(&eval->upp, 0.0);
}

/*
 * CUDA tests
 */
__global__ void testCudaEvalCompute(interval_ptr d_eval, int * d_number) {
    cuda::rns_eval_compute(&d_eval->low, &d_eval->upp, d_number);
}

__global__ void testCudaEvalFastCompute(interval_ptr d_eval, int * d_number) {
    cuda::rns_eval_compute_fast(&d_eval->low, &d_eval->upp, d_number);
}

int main() {

    rns_const_init();
    Logger::beginTestDescription(Logger::RNS_EVAL_ACCURACY_TEST);
    rns_const_print(true);
    Logger::printDash();
    rns_eval_const_print();
    Logger::endSection(true);
    Logger::printSpace();

    bool asc = true; //start with x = 0
    char c;

    int * number = new int[RNS_MODULI_SIZE];;
    int * d_number;
    interval_ptr eval = new interval_t; // host result
    interval_ptr d_eval; // device result
    er_float_ptr exact = new er_float_t[1];
    mpz_t binary;

    cudaMalloc(&d_number, RNS_MODULI_SIZE * sizeof(int));
    cudaMalloc(&d_eval, sizeof(interval_t));
    mpz_init(binary);

    std::cout << "Enter 'y' for start (or any key for exit): ";
    std::cin >> c;
    if (c == 'y') {
        for (int j = 0; j < RNS_MODULI_SIZE; j++) {
            if (asc)
                number[j] = 0;
            else
                number[j] = RNS_MODULI[j] - 1;
        }
        cudaMemcpy(d_number, number, RNS_MODULI_SIZE * sizeof(int), cudaMemcpyHostToDevice);

        while (c == 'y') {
            for (int j = 0; j < 10; j++) {
                printf("\n");
                Logger::printDash();

                rns_to_binary(binary, number);
                printf("\t number = %s\n", mpz_get_str(NULL, 10, binary));
                rns_fractional(exact, number);
                std::cout << "\t relative = ";
                er_print(exact);
                //-------------------------------------------
                printf("\n[CPU] rns_eval_compute: ");
                resetResult(eval);
                rns_eval_compute(&eval->low, &eval->upp, number);
                printError(eval, exact);
                //-------------------------------------------
                printf("\n[CPU] rns_eval_compute_fast: ");
                resetResult(eval);
                rns_eval_compute_fast(&eval->low, &eval->upp, number);
                printError(eval, exact);
                //-------------------------------------------
                printf("\n[CUDA] rns_eval_compute: ");
                resetResult(eval);
                resetResultCuda<<< 1, 1 >>>(d_eval);
                testCudaEvalCompute<<< 1, 1 >>>(d_eval, d_number);
                cudaMemcpy(eval, d_eval, sizeof(interval_t), cudaMemcpyDeviceToHost);
                printError(eval, exact);
                //-------------------------------------------
                printf("\n[CUDA] rns_eval_compute_fast: ");
                resetResult(eval);
                resetResultCuda<<< 1, 1 >>>(d_eval);
                testCudaEvalFastCompute<<< 1, 1 >>>(d_eval, d_number);
                cudaMemcpy(eval, d_eval, sizeof(interval_t), cudaMemcpyDeviceToHost);
                printError(eval, exact);
                //-------------------------------------------

                for (int i = 0; i < RNS_MODULI_SIZE; i++) {
                    if (asc)
                        number[i] = (number[i] + 1) % RNS_MODULI[i];
                    else {
                        number[i] -= 1;
                        if (number[i] < 0)
                            number[i] += RNS_MODULI[i];
                    }
                }
                cudaMemcpy(d_number, number, RNS_MODULI_SIZE * sizeof(int), cudaMemcpyHostToDevice);
                printf("\n");
            }
            Logger::printDash();
            std::cout << "\n\nEnter 'y' for continue (or any key for exit): ";
            std::cin >> c;
        }
    }

    delete [] number;
    delete eval;
    cudaFree(d_number);
    cudaFree(d_eval);
    delete exact;
    mpz_clear(binary);

    //End logging
    Logger::endTestDescription();
    return 1;
}