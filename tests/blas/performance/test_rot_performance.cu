/*
 *  Performance test for BLAS ROT routines
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
#include "../../../src/blas/rot.cuh"
#include "blas/external/3rdparty.cuh"

#define N 1000000 //Operation size
#define REPEAT_TEST 10 //Number of repeats

//Execution configuration for mp_rot
#define MPRES_CUDA_BLOCKS_FIELDS_ROUND   512
#define MPRES_CUDA_THREADS_FIELDS_ROUND  128
#define MPRES_CUDA_BLOCKS_RESIDUES       8192

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
    mp_float_t print_result = MP_ZERO;

    mpfr_t mpfr_result;
    mpfr_init2(mpfr_result, 100000);
    mpfr_set_d(mpfr_result, 0, MPFR_RNDN);

    for (int i = 0; i < v_length; i++) {
        mp_add(&print_result, print_result, result[i]);
    }

    mp_get_mpfr(mpfr_result, print_result);
    mpfr_printf("result %s: %.70Rf \n", name, mpfr_result);
    mpfr_clear(mpfr_result);
}

/********************* Benchmarks *********************/

/////////
// MPRES-BLAS
/////////
void mpres_test(int n, mpfr_t * x, mpfr_t * y, mpfr_t c, mpfr_t s) {
    Logger::printDash();
    InitCudaTimer();
    PrintTimerName("[GPU] MPRES-BLAS rot");

    //Host data
    mp_float_ptr hx = new mp_float_t[n];
    mp_float_ptr hy = new mp_float_t[n];
    mp_float_t hc;
    mp_float_t hs;

    //GPU data
    mp_array_t dx;
    mp_array_t dy;
    mp_array_t dc;
    mp_array_t ds;
    mp_array_t dbuffer1;
    mp_array_t dbuffer2;

    cuda::mp_array_init(dx, n);
    cuda::mp_array_init(dy, n);
    cuda::mp_array_init(dc, 1);
    cuda::mp_array_init(ds, 1);
    cuda::mp_array_init(dbuffer1, n);
    cuda::mp_array_init(dbuffer2, n);

    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    //Convert from MPFR
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        mp_set_mpfr(&hx[i], x[i]);
        mp_set_mpfr(&hy[i], y[i]);
    }
    mp_set_mpfr(&hc, c);
    mp_set_mpfr(&hs, s);

    //Copying to the GPU
    cuda::mp_array_host2device(dc, &hc, 1);
    cuda::mp_array_host2device(ds, &hs, 1);

    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    //Launch
    for (int i = 0; i < REPEAT_TEST; i++) {
        cuda::mp_array_host2device(dx, hx, n);
        cuda::mp_array_host2device(dy, hy, n);
        StartCudaTimer();
        cuda::mp_rot<
                MPRES_CUDA_BLOCKS_FIELDS_ROUND,
                MPRES_CUDA_THREADS_FIELDS_ROUND,
                MPRES_CUDA_BLOCKS_RESIDUES>
                (n, dx, 1, dy, 1, dc, ds, dbuffer1, dbuffer2);
        EndCudaTimer();
    }
    PrintCudaTimer("took");

    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    //Copying to the host
    cuda::mp_array_device2host(hx, dx, n);
    cuda::mp_array_device2host(hy, dy, n);
    print_mp_sum(hx, N, "x");
    print_mp_sum(hy, N, "y");

    //Cleanup
    delete [] hx;
    delete [] hy;
    cuda::mp_array_clear(dx);
    cuda::mp_array_clear(dy);
    cuda::mp_array_clear(dc);
    cuda::mp_array_clear(ds);
    cuda::mp_array_clear(dbuffer1);
    cuda::mp_array_clear(dbuffer2);
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();
}


/********************* Main test *********************/


int main() {

    initialize();

    //Start logging
    Logger::beginTestDescription(Logger::BLAS_ROT_PERFORMANCE_TEST);
    Logger::printTestParameters(N, REPEAT_TEST, MP_PRECISION, MP_PRECISION_DEC);
    Logger::beginSection("Additional info:");
    Logger::printParam("RNS_MODULI_SIZE", RNS_MODULI_SIZE);
    Logger::printParam("MPRES_CUDA_BLOCKS_FIELDS_ROUND", MPRES_CUDA_BLOCKS_FIELDS_ROUND);
    Logger::printParam("MPRES_CUDA_THREADS_FIELDS_ROUND", MPRES_CUDA_THREADS_FIELDS_ROUND);
    Logger::printParam("MPRES_CUDA_BLOCKS_RESIDUES", MPRES_CUDA_BLOCKS_RESIDUES);
    #ifndef EXCLUDE_CAMPARY
    Logger::printParam("CAMPARY_PRECISION (n-double)", CAMPARY_PRECISION);
    #endif
    Logger::endSection(true);

    //Inputs
    mpfr_t * vectorX;
    mpfr_t * vectorY;
    mpfr_t * c;
    mpfr_t * s;
    vectorX = create_random_array(N, INP_BITS);
    vectorY = create_random_array(N, INP_BITS);
    c = create_random_array(1, INP_BITS);
    s = create_random_array(1, INP_BITS);

    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    // Multiple-precision tests
    #ifndef EXCLUDE_MPACK
    mpack_rot_test(N, vectorX, vectorY, c[0], s[0], MP_PRECISION, REPEAT_TEST);
    #endif
    mpres_test(N, vectorX, vectorY, c[0], s[0]);
    #ifndef EXCLUDE_GARPREC
    garprec_rot_test(N, vectorX, vectorY, c[0], s[0], MP_PRECISION_DEC, INP_DIGITS, REPEAT_TEST);
    #endif
    #ifndef EXCLUDE_CAMPARY
    campary_rot_test<CAMPARY_PRECISION>(N, vectorX, vectorY, c[0], s[0], INP_DIGITS, REPEAT_TEST);
    #endif
    #ifndef EXCLUDE_CUMP
    cump_rot_test(N, vectorX, vectorY, c[0], s[0], MP_PRECISION, INP_DIGITS, REPEAT_TEST);
    #endif

    checkDeviceHasErrors(cudaDeviceSynchronize());
    //cudaCheckErrors(); //CUMP gives failure

    //Cleanup
    mpfr_clear(c[0]);
    mpfr_clear(s[0]);
    for(int i = 0; i < N; i++){
        mpfr_clear(vectorX[i]);
        mpfr_clear(vectorY[i]);
    }
    delete [] vectorX;
    delete [] vectorY;
    delete [] c;
    delete [] s;

    //Finalize
    finalize();

    //End logging
    Logger::endTestDescription();

    return 0;
}