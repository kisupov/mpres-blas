/*
 *  Test for measure the performance of the RNS magnitude comparison algorithms
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

#include <iostream>
#include "gmp.h"
#include "../../src/rns.cuh"
#include "../logger.cuh"
#include "../timers.cuh"


#define ITERATIONS 1000


/*
 * CUDA tests
 */

__global__ static void testCudaRnsCmp(int * dr, int * dx, int * dy, int vectorSize) {
    for(int i = 0; i < vectorSize; i++){
        dr[i] = cuda::rns_cmp(&dx[i * RNS_MODULI_SIZE], &dy[i * RNS_MODULI_SIZE]);
    }
}

__global__ static void testCudaMrcCmp(int * dr, int * dx, int * dy, int vectorSize) {
    for(int i = 0; i < vectorSize; i++){
        dr[i] = cuda::mrc_compare_rns(&dx[i * RNS_MODULI_SIZE], &dy[i * RNS_MODULI_SIZE]);
    }
}


/*
 * Common methods
 */

static void resetResult(int * r, int vectorSize){
    memset(r, 0, vectorSize * sizeof(int));
}

__global__ static void resetResultCuda(int * r, int vectorSize) {
    for(int i = 0; i < vectorSize; i++){
        r[i] = 0;
    }
}

static void checkResult(int * ref, int * res, int vectorSize){
    int errors = 0;
    for(int i = 0; i < vectorSize; i++){
        if(ref[i] != res[i]){
            errors++;
        }
    }
    if(errors == 0){
        printf("All results match\n");
    }else{
        printf("Count of errors: %i\n", errors);
    }
}

static void fill_random_array(mpz_t *array, int n) {
    mpz_t rndBnd;
    gmp_randstate_t state;
    gmp_randinit_default(state);
    gmp_randseed_ui(state, (unsigned) time(NULL));
    mpz_init(rndBnd);
    mpz_sub_ui(rndBnd, RNS_MODULI_PRODUCT, 1);
    for (int i = 0; i < n; i++) {
        mpz_urandomm(array[i], state, rndBnd);
    }
    gmp_randclear(state);
    mpz_clear(rndBnd);
}


/*
 * Main test
 */

static void run_test(int iterations) {
    InitCpuTimer();
    InitCudaTimer();

    // Multiple-precision host data
    mpz_t * hx  = new mpz_t[iterations];
    mpz_t * hy  = new mpz_t[iterations];
    int   * ref = new int[iterations]; //reference result

    // RNS host data
    int * hrx = new int[iterations * RNS_MODULI_SIZE];
    int * hry = new int[iterations * RNS_MODULI_SIZE];
    int * hres = new int[iterations];

    //GPU data
    int * drx;
    int * dry;
    int * dres;

    //Memory allocation
    for(int i = 0; i < iterations; i++){
        mpz_init(hx[i]);
        mpz_init(hy[i]);
    }

    cudaMalloc(&drx, sizeof(int) * RNS_MODULI_SIZE * iterations);
    cudaMalloc(&dry, sizeof(int) * RNS_MODULI_SIZE * iterations);
    cudaMalloc(&dres, sizeof(int) * iterations);
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    //Generate inputs
    fill_random_array(hx, iterations);
    waitFor(5);
    fill_random_array(hy, iterations);

    //Convert to the RNS
    for(int i = 0; i < iterations; i++){
        rns_from_binary(&hrx[i * RNS_MODULI_SIZE], hx[i]);
        rns_from_binary(&hry[i * RNS_MODULI_SIZE], hy[i]);
    }

    // Copying to the GPU
    cudaMemcpy(drx, hrx, sizeof(int) * RNS_MODULI_SIZE * iterations, cudaMemcpyHostToDevice);
    cudaMemcpy(dry, hry, sizeof(int) * RNS_MODULI_SIZE * iterations, cudaMemcpyHostToDevice);
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    // Computing exact results
    //---------------------------------------------------------
    for(int i = 0; i < iterations; i++){
        ref[i] = mpz_cmp(hx[i], hy[i]);
    }
    //---------------------------------------------------------
    Logger::printDash();
    PrintTimerName("[CPU] rns_cmp");
    resetResult(hres, iterations);
    //Launch
    StartCpuTimer();
    for(int i = 0; i < iterations; i++){
        hres[i] = rns_cmp(&hrx[i * RNS_MODULI_SIZE], &hry[i * RNS_MODULI_SIZE]);
    }
    EndCpuTimer();
    PrintCpuTimer("took");
    checkResult(ref, hres, iterations);
    //---------------------------------------------------------
    Logger::printDash();
    PrintTimerName("[CUDA] rns_cmp");
    resetResult(hres, iterations);
    resetResultCuda<<<1,1>>>(dres, iterations);
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();
    //Launch
    StartCudaTimer();
    testCudaRnsCmp<<<1,1>>>(dres, drx, dry, iterations);
    EndCudaTimer();
    PrintCudaTimer("took");
    //Copying to the host
    cudaMemcpy(hres, dres, sizeof(int) * iterations , cudaMemcpyDeviceToHost);
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();
    checkResult(ref, hres, iterations);
    //---------------------------------------------------------
    Logger::printDash();
    PrintTimerName("[CUDA] mrc_compare_rns");
    resetResult(hres, iterations);
    resetResultCuda<<<1,1>>>(dres, iterations);
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();
    //Launch
    StartCudaTimer();
    testCudaMrcCmp<<<1,1>>>(dres, drx, dry, iterations);
    EndCudaTimer();
    PrintCudaTimer("took");
    //Copying to the host
    cudaMemcpy(hres, dres, sizeof(int) * iterations , cudaMemcpyDeviceToHost);
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();
    checkResult(ref, hres, iterations);
    //---------------------------------------------------------
    
    // Cleanup
    for(int i = 0; i < iterations; i++){
        mpz_clear(hx[i]);
        mpz_clear(hy[i]);
    }
    delete [] hx;
    delete [] hy;
    delete [] ref;
    delete [] hrx;
    delete [] hry;
    delete [] hres;
    cudaFree(drx);
    cudaFree(dry);
    cudaFree(dres);
}

int main() {
    cudaDeviceReset();
    rns_const_init();
    Logger::beginTestDescription(Logger::RNS_CMP_PERFORMANCE_TEST);
    Logger::printParam("ITERATIONS", ITERATIONS);
    Logger::endSection(true);
    Logger::printSpace();
    run_test(ITERATIONS);
    Logger::endTestDescription();
    return 0;
}