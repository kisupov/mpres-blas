/*
 *  Performance test for BLAS NORM routines
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

/*
 * Exclude some benchmarks
 */
#define EXCLUDE_OPENBLAS
#define EXCLUDE_XBLAS
#define EXCLUDE_ARPREC
#define EXCLUDE_MPACK
#define EXCLUDE_MPDECIMAL
#define EXCLUDE_CUBLAS
#define EXCLUDE_CUMP
#define EXCLUDE_GARPREC
#define EXCLUDE_CAMPARY

#include "../../logger.cuh"
#include "../../timers.cuh"
#include "../../tsthelper.cuh"
#include "../../../src/mparray.cuh"
#include "../../../src/blas/norm.cuh"
#include "3rdparty.cuh"


#define N 1000000 //Operation size
#define INCX (1) // Specifies the increment for the elements of x
#define REPEAT_TEST 10 //Number of repeats
#define NORM_TYPE "ONE" //ONE = one norm, INF = infinity-norm

//Execution configuration for mpres
#define MPRES_CUDA_BLOCKS_REDUCE   128
#define MPRES_CUDA_THREADS_REDUCE  32

int MP_PRECISION_DEC; //in decimal digits
int INP_BITS; //in bits
int INP_DIGITS; //in decimal digits

static void setPrecisions(){
    MP_PRECISION_DEC = (int)(MP_PRECISION / 3.32 + 1);
    INP_BITS = (int)(MP_PRECISION / 4);
    INP_DIGITS = (int)(INP_BITS / 3.32 + 1);
}

static void initialize(){
    cudaDeviceReset();
    rns_const_init();
    mp_const_init();
    setPrecisions();
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();
}

static void finalize(){
    cudaDeviceReset();
}

/********************* NORM implementations and benchmarks *********************/

/////////
// MPFR
/////////

//one-norm
static void mpfr_norm_one(mpfr_t result, mpfr_t *x, int incx, int n, int precision) {
    #pragma omp parallel shared(result)
    {
        mpfr_t sum;
        mpfr_t absval;
        mpfr_init2(sum, precision);
        mpfr_init2(absval, precision);
        mpfr_set_d(sum, 0, MPFR_RNDN);
        mpfr_set_d(absval, 0, MPFR_RNDN);
        #pragma omp for
        for (int i = 0; i < n; i++) {
            mpfr_abs(absval, x[i * incx], MPFR_RNDN);
            mpfr_add(sum, sum, absval, MPFR_RNDN);
        }
        #pragma omp critical
        {
            mpfr_add(result, result, sum, MPFR_RNDN);
        }
        mpfr_clear(absval);
        mpfr_clear(sum);
    }
}

//infinity-norm
static void mpfr_norm_inf(mpfr_t result, mpfr_t *x, int incx, int n, int precision) {
        mpfr_t absval;
        mpfr_init2(absval, precision);
        mpfr_set_d(absval, 0, MPFR_RNDN);
        mpfr_set_d(result, 0, MPFR_RNDN);
        for (int i = 0; i < n; i++) {
            mpfr_abs(absval, x[i*incx], MPFR_RNDN);
            if(mpfr_cmp(absval, result) > 0){
                mpfr_set(result, absval, MPFR_RNDN);
            }
        }
        mpfr_clear(absval);
}

static void mpfr_test(int n, mpfr_t *x, int incx){
    InitCpuTimer();
    Logger::printDash();
    PrintTimerName("[CPU] MPFR norm");

    //Init
    mpfr_t result;
    mpfr_init2(result, MP_PRECISION);

    //Launch
   if(NORM_TYPE == "ONE"){
       for (int i = 0; i < REPEAT_TEST; i++) {
           mpfr_set_d(result, 0, MPFR_RNDN);
           StartCpuTimer();
           mpfr_norm_one(result, x, incx, n, MP_PRECISION);
           EndCpuTimer();
       }
    } else{
        for (int i = 0; i < REPEAT_TEST; i++) {
            StartCpuTimer();
            mpfr_norm_inf(result, x, incx, n, MP_PRECISION);
            EndCpuTimer();
        }
    }
    PrintCpuTimer("took");
    mpfr_printf("result: %.70Rf \n", result);

    //Clear
    mpfr_clear(result);
}



/////////
// MPRES-BLAS
/////////
static void mpres_test(int n, mpfr_t *x, int incx){
    InitCudaTimer();
    Logger::printDash();
    PrintTimerName("[GPU] MPRES-BLAS norm");

    int lenx = (1 + (n - 1) * abs(incx));

    //Host data
    mp_float_ptr hx = new mp_float_t[lenx];
    mp_float_ptr hresult = new mp_float_t[1];

    //GPU data
    mp_array_t dx;
    mp_array_t dresult;

    cuda::mp_array_init(dx, lenx);
    cuda::mp_array_init(dresult, 1);

    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    //Convert from MPFR
    #pragma omp parallel for
    for(int i =0; i < lenx; i ++){
        mp_set_mpfr(&hx[i], x[i]);
    }
    *hresult = MP_ZERO;

    //Copying to the GPU
    cuda::mp_array_host2device(dx, hx, lenx);

    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    //Launch
    StartCudaTimer();
    for (int i = 0; i < REPEAT_TEST; i++) {
        cuda::mp_norm<MPRES_CUDA_BLOCKS_REDUCE, MPRES_CUDA_THREADS_REDUCE>
                ((NORM_TYPE == "ONE" ? mblas_one_norm : mblas_inf_norm), n, dx, incx, dresult);
    }
    EndCudaTimer();
    PrintCudaTimer("took");

    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    mpfr_t mpfr_result;
    mpfr_init2(mpfr_result, MP_PRECISION);
    mpfr_set_d(mpfr_result, 0, MPFR_RNDN);

    //Copying to the host
    cuda::mp_array_device2host(hresult, dresult, 1);
    mp_get_mpfr(mpfr_result, hresult);
    mpfr_printf("result: %.70Rf \n", mpfr_result);

    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    //Cleanup
    delete [] hx;
    delete [] hresult;
    mpfr_clear(mpfr_result);
    cuda::mp_array_clear(dx);
    cuda::mp_array_clear(dresult);
}


/********************* Main test *********************/


int main() {

    initialize();

    //Start logging
    Logger::beginTestDescription(Logger::BLAS_NORM_PERFORMANCE_TEST);
    Logger::printTestParameters(N, REPEAT_TEST, MP_PRECISION, MP_PRECISION_DEC);
    Logger::beginSection("Operation info:");
    Logger::printParam("INCX", INCX);
    Logger::printParam("NORM_TYPE", NORM_TYPE);
    Logger::beginSection("Additional info:");
    Logger::printParam("RNS_MODULI_SIZE", RNS_MODULI_SIZE);
    Logger::printParam("MPRES_CUDA_BLOCKS_REDUCE", MPRES_CUDA_BLOCKS_REDUCE);
    Logger::printParam("MPRES_CUDA_THREADS_REDUCE", MPRES_CUDA_THREADS_REDUCE);
    Logger::endSection(true);

    int LENX = (1 + (N - 1) * abs(INCX));
    mpfr_t * vectorX = create_random_array(LENX, INP_BITS);

    mpfr_test(N, vectorX, INCX);
    mpres_test(N, vectorX, INCX);

    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    // Cleanup
    for(int i = 0; i < LENX; i++){
        mpfr_clear(vectorX[i]);
    }
    delete[] vectorX;

    //Finalize
    finalize();
    Logger::endTestDescription();

    return 0;
}