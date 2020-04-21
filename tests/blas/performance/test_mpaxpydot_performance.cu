/*
 *  Performance test for BLAS AXPY_DOT routines
 *
 *  Copyright 2019 by Konstantin Isupov and Alexander Kuvaev.
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

#include "omp.h"
#include "../../logger.cuh"
#include "../../timers.cuh"
#include "../../tsthelper.cuh"
#include "../../../src/blas/mpaxpydot.cuh"
#include "3rdparty.cuh"


#define N 1000000 //Operation size
#define REPEAT_TEST 10 //Number of repeats

//Execution configuration for mpaxpydot
#define MPRES_CUDA_BLOCKS_FIELDS_ROUND   512
#define MPRES_CUDA_THREADS_FIELDS_ROUND  128
#define MPRES_CUDA_BLOCKS_RESIDUES       8192
#define MPRES_CUDA_BLOCKS_REDUCE         256
#define MPRES_CUDA_THREADS_REDUCE        64

int MP_PRECISION_DEC; //in decimal digits
int INP_BITS; //in bits
int INP_DIGITS; //in decimal digits

void setPrecisions(){
    MP_PRECISION_DEC = (int)(MP_PRECISION / 3.32 + 1);
    INP_BITS = (int)(MP_PRECISION / 4);
    INP_DIGITS = (int)(INP_BITS / 3.32 + 1);
}

void initialize(){
    cudaDeviceReset();
    rns_const_init();
    mp_const_init();
    setPrecisions();
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();
}

void finalize(){
    cudaDeviceReset();
}

static void print_mp_sum(mp_float_ptr result, int v_length, const char *name) {
    mp_float_t print_result;
    print_result = MP_ZERO;

    mpfr_t mpfr_result;
    mpfr_init2(mpfr_result, 100000);
    mpfr_set_d(mpfr_result, 0, MPFR_RNDN);

    for (int i = 0; i < v_length; i++) {
        mp_add(&print_result, &print_result, &result[i]);
    }

    mp_get_mpfr(mpfr_result, &print_result);
    mpfr_printf("result %s: %.70Rf \n", name, mpfr_result);
    mpfr_clear(mpfr_result);
}


/********************* Benchmarks *********************/


/////////
// MPRES-BLAS
/////////
void mpres_test(int n, mpfr_t alpha, mpfr_t * w, mpfr_t * v, mpfr_t * u) {
    Logger::printDash();
    InitCudaTimer();
    PrintTimerName("[GPU] MPRES-BLAS axpy_dot");

    //Host data
    mp_float_ptr hw = new mp_float_t[n];
    mp_float_ptr hv = new mp_float_t[n];
    mp_float_ptr hu = new mp_float_t[n];
    mp_float_ptr halpha = new mp_float_t[1];
    mp_float_ptr hr = new mp_float_t[1];

    //GPU data
    mp_array_t dw;
    mp_array_t dv;
    mp_array_t du;
    mp_array_t dalpha;
    mp_array_t dr;
    mp_array_t dbuffer;

    cuda::mp_array_init(dw, n);
    cuda::mp_array_init(dv, n);
    cuda::mp_array_init(du, n);
    cuda::mp_array_init(dalpha, 1);
    cuda::mp_array_init(dbuffer, n);
    cuda::mp_array_init(dr, 1);

    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    //Convert from MPFR
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        mp_set_mpfr(&hw[i], w[i]);
        mp_set_mpfr(&hv[i], v[i]);
        mp_set_mpfr(&hu[i], u[i]);
    }
    mp_set_mpfr(halpha, alpha);

    //Copying to the GPU
    cuda::mp_array_host2device(dv, hv, n);
    cuda::mp_array_host2device(du, hu, n);
    cuda::mp_array_host2device(dalpha, halpha, 1);

    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    //Launch
    for (int i = 0; i < REPEAT_TEST; i++) {
        cuda::mp_array_host2device(dw, hw, n);
        StartCudaTimer();
        cuda::mpaxpydot<
                MPRES_CUDA_BLOCKS_FIELDS_ROUND,
                MPRES_CUDA_THREADS_FIELDS_ROUND,
                MPRES_CUDA_BLOCKS_RESIDUES,
                MPRES_CUDA_BLOCKS_REDUCE,
                MPRES_CUDA_THREADS_REDUCE>
                (n, dalpha, dw, 1, dv, 1, du, 1, dr, dbuffer);
        EndCudaTimer();
    }
    PrintCudaTimer("took");

    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    mpfr_t mpfr_result;
    mpfr_init2(mpfr_result, MP_PRECISION);
    mpfr_set_d(mpfr_result, 0, MPFR_RNDN);

    //Copying to the host
    cuda::mp_array_device2host(hr, dr, 1);
    cuda::mp_array_device2host(hw, dw, n);

    print_mp_sum(hw, n, "w");
    mp_get_mpfr(mpfr_result, hr);
    mpfr_printf("result r: %.70Rf \n", mpfr_result);
    
    //Cleanup
    delete [] hv;
    delete [] hu;
    delete [] hw;
    delete [] hr;
    delete [] halpha;
    cuda::mp_array_clear(dv);
    cuda::mp_array_clear(du);
    cuda::mp_array_clear(dw);
    cuda::mp_array_clear(dalpha);
    cuda::mp_array_clear(dr);
    cuda::mp_array_clear(dbuffer);
    mpfr_clear(mpfr_result);
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();
}


/********************* Main test *********************/


int main() {

    initialize();

    //Start logging
    Logger::beginTestDescription(Logger::BLAS_AXPY_DOT_PERFORMANCE_TEST);
    Logger::printTestParameters(N, REPEAT_TEST, MP_PRECISION, MP_PRECISION_DEC);
    Logger::beginSection("Additional info:");
    Logger::printParam("RNS_MODULI_SIZE", RNS_MODULI_SIZE);
    Logger::printParam("MPRES_CUDA_BLOCKS_FIELDS_ROUND", MPRES_CUDA_BLOCKS_FIELDS_ROUND);
    Logger::printParam("MPRES_CUDA_THREADS_FIELDS_ROUND", MPRES_CUDA_THREADS_FIELDS_ROUND);
    Logger::printParam("MPRES_CUDA_BLOCKS_RESIDUES", MPRES_CUDA_BLOCKS_RESIDUES);
    Logger::printParam("MPRES_CUDA_BLOCKS_REDUCE", MPRES_CUDA_BLOCKS_REDUCE);
    Logger::printParam("MPRES_CUDA_THREADS_REDUCE", MPRES_CUDA_THREADS_REDUCE);
    Logger::printParam("CAMPARY_PRECISION (n-double)", CAMPARY_PRECISION);
    Logger::endSection(true);

    //Inputs
    mpfr_t * vectorV;
    mpfr_t * vectorU;
    mpfr_t * vectorW;
    mpfr_t * alpha;
    vectorU = create_random_array(N, INP_BITS);
    vectorV = create_random_array(N, INP_BITS);
    vectorW = create_random_array(N, INP_BITS);
    alpha = create_random_array(1, INP_BITS);

    // Multiple-precision tests
    mpres_test(N, alpha[0], vectorW, vectorV, vectorU);
    cudaDeviceReset();
    garprec_axpy_dot_test(N, alpha[0], vectorW, vectorV, vectorU, MP_PRECISION_DEC, INP_DIGITS, REPEAT_TEST);
    campary_axpy_dot_test<CAMPARY_PRECISION>(N, alpha[0], vectorW, vectorV, vectorU, INP_DIGITS, REPEAT_TEST);
    cump_axpy_dot_test(N, alpha[0], vectorW, vectorV, vectorU, MP_PRECISION, INP_DIGITS, REPEAT_TEST);
    checkDeviceHasErrors(cudaDeviceSynchronize());
    //cudaCheckErrors(); //CUMP gives failure

    //Cleanup
    mpfr_clear(alpha[0]);
    for(int i = 0; i < N; i++){
        mpfr_clear(vectorW[i]);
        mpfr_clear(vectorV[i]);
        mpfr_clear(vectorU[i]);
    }
    delete [] alpha;
    delete [] vectorV;
    delete [] vectorU;
    delete [] vectorW;

    //Finalize
    finalize();

    //End logging
    Logger::endTestDescription();

    return 0;
}