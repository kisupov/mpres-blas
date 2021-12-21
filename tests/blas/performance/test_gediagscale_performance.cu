/*
 *  Performance test for BLAS GE_DIAG_SCALE routines
 *
 *  Copyright 2020 by Konstantin Isupov
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
#define EXCLUDE_MPACK
#define EXCLUDE_ARPREC
#define EXCLUDE_XBLAS
#define EXCLUDE_MPDECIMAL
#define EXCLUDE_GARPREC
#define EXCLUDE_CUBLAS
#define EXCLUDE_OPENBLAS

#include "../../logger.cuh"
#include "../../timers.cuh"
#include "../../tsthelper.cuh"
#include "../../../src/mparray.cuh"
#include "../../../src/blas/gediagscale.cuh"
#include "blas/external/3rdparty.cuh"

#define M 1000  // Number of matrix rows and the vector X dimension
#define N 1000  // Number of matrix columns and the vector Y dimension
#define LDA (M) // Specifies the leading dimension of A as declared in the calling (sub)program.
#define INCD (1) // Specifies the increment for the elements of D.
#define REPEAT_TEST 10 //Number of repeats
#define SIDE "R" // specifies the type of operation to be performed


//Execution configuration for mpres
#define MPRES_GRID_SIZE_X_ESI 128
#define MPRES_BLOCK_SIZE_ESI 128
#define MPRES_GRID_SIZE_DIGITS 128

int MP_PRECISION_DEC; //in decimal digits
int INP_BITS; //in bits
int INP_DIGITS; //in decimal digits

void setPrecisions(){
    MP_PRECISION_DEC = (int)(MP_PRECISION / 3.32 + 1);
    INP_BITS = (int)(MP_PRECISION / 2);
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
}

/********************* GE_DIAG_SCALE implementations and benchmarks *********************/

/////////
// MPFR
/////////
void mpfr_ge_diag_scale_r(int m, int n, mpfr_t *D, int incd, mpfr_t *A, int lda){
        #pragma omp parallel for shared(m, n, A, D)
        for (int j = 0; j < n; j++) {
            int id = incd > 0 ? j * incd : (-n + 1 + j)*incd;
            for (int i = 0; i < m; i++) {
                mpfr_mul(A[i + j * lda], A[i + j * lda], D[id], MPFR_RNDN);
            }
        }
}


void mpfr_ge_diag_scale_l(int m, int n, mpfr_t *D, int incd, mpfr_t *A, int lda){
#pragma omp parallel for shared(m, n, A, D)
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
            int id = incd > 0 ? i * incd : (-m + 1 + i)*incd;
            mpfr_mul(A[i + j * lda], A[i + j * lda], D[id], MPFR_RNDN);
        }
    }
}

void mpfr_test(int m, int n, mpfr_t *D, int incd, mpfr_t *A, int lda){
    InitCpuTimer();
    Logger::printDash();
    PrintTimerName("[CPU] MPFR ge_diag_scale");

    // Init
    mpfr_t *mA = new mpfr_t[lda * n];
    #pragma omp parallel for
    for(int j = 0; j < lda * n; j++){
        mpfr_init2(mA[j], MP_PRECISION);
    }

    // Launch
    if(SIDE == "R"){
        for(int i = 0; i < REPEAT_TEST; i++){
            #pragma omp parallel for
            for(int j = 0; j < lda * n; j++){
                mpfr_set(mA[j], A[j], MPFR_RNDN);
            }
            StartCpuTimer();
            mpfr_ge_diag_scale_r(m, n, D, incd, mA, lda);
            EndCpuTimer();
        }
    }
    else{
        for(int i = 0; i < REPEAT_TEST; i++){
            #pragma omp parallel for
            for(int j = 0; j < lda * n; j++){
                mpfr_set(mA[j], A[j], MPFR_RNDN);
            }
            StartCpuTimer();
            mpfr_ge_diag_scale_l(m, n, D, incd, mA, lda);
            EndCpuTimer();
        }
    }
    PrintAndResetCpuTimer("took");
    print_mpfr_sum(mA, lda * n);

    //Cleanup
    for(int i = 0; i < lda * n; i++){
        mpfr_clear(mA[i]);
    }
    delete [] mA;
}

/////////
// MPRES-BLAS
/////////
void mpres_test(int m, int n, mpfr_t *D, int incd, mpfr_t *A, int lda, int lend){
    InitCudaTimer();
    Logger::printDash();
    PrintTimerName("[GPU] MPRES-BLAS ge_diag_scale");


    // Host data
    mp_float_ptr hA = new mp_float_t[lda * n];
    mp_float_ptr hD = new mp_float_t[lend];

    //GPU data
    mp_array_t dA;
    mp_array_t dD;

    //Init data
    cuda::mp_array_init(dA, lda * n);
    cuda::mp_array_init(dD, lend);

    // Convert from MPFR
    convert_matrix(hA, A, lda, n);
    convert_matrix(hD, D, lend, 1);

    //Copying the diagonal matrix to the GPU
    cuda::mp_array_host2device(dD, hD, lend);

    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    //Launch
    for (int i = 0; i < REPEAT_TEST; i++) {
        cuda::mp_array_host2device(dA, hA, lda * n);
        StartCudaTimer();
        cuda::mp_ge_diag_scale<
                        MPRES_GRID_SIZE_X_ESI,
                        MPRES_BLOCK_SIZE_ESI,
                        MPRES_GRID_SIZE_DIGITS>
                ( ((SIDE == "R") ? mblas_right_side : mblas_left_side), m, n, dD, incd, dA, lda);
        EndCudaTimer();
    }
    PrintAndResetCudaTimer("took");
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    //Copying to the host
    cuda::mp_array_device2host(hA, dA, lda * n);
    print_mp_sum(hA, lda * n);

    //Cleanup
    delete [] hA;
    delete [] hD;
    cuda::mp_array_clear(dA);
    cuda::mp_array_clear(dD);
}


/********************* Main test *********************/

void runTest(){

    int LEND = (SIDE == "R") ? (1 + (N - 1) * abs(INCD)) : (1 + (M - 1) * abs(INCD));

    //Inputs
    mpfr_t *matrixA = create_random_array(LDA * N, INP_BITS);
    mpfr_t *matrixD = create_random_array(LEND, INP_BITS);
    //Multiple-precision tests
    mpfr_test(M, N, matrixD, INCD, matrixA, LDA);
    mpres_test(M, N, matrixD, INCD, matrixA, LDA, LEND);
    #ifndef EXCLUDE_CAMPARY
    campary_ge_diag_scale_test<CAMPARY_PRECISION>(((SIDE == "R") ? mblas_right_side : mblas_left_side), M, N, LEND, matrixD, INCD, matrixA, LDA, INP_DIGITS, REPEAT_TEST);
    #endif
    #ifndef EXCLUDE_CUMP
    cump_ge_diag_scale_test(((SIDE == "R") ? mblas_right_side : mblas_left_side), M, N, LEND, matrixD, INCD, matrixA, LDA, MP_PRECISION, INP_DIGITS, REPEAT_TEST);
    #endif

    checkDeviceHasErrors(cudaDeviceSynchronize());

    //Cleanup
    for(int i = 0; i < LDA * N; i++){
        mpfr_clear(matrixA[i]);
    }
    for(int i = 0; i < LEND; i++){
        mpfr_clear(matrixD[i]);
    }

    delete [] matrixA;
    delete [] matrixD;
    cudaDeviceReset();
}


int main(){
    initialize();

    //Start logging
    Logger::beginTestDescription(Logger::BLAS_GE_DIAG_SCALE_PERFORMANCE_TEST);
    Logger::printTestParameters(M * N, REPEAT_TEST, MP_PRECISION, MP_PRECISION_DEC);
    Logger::beginSection("Operation info:");
    Logger::printParam("Matrix rows, M", M);
    Logger::printParam("Matrix columns, N", N);
    Logger::printParam("LDA", LDA);
    Logger::printParam("INCD", INCD);
    Logger::printParam("SIDE", SIDE);
    Logger::printDash();
    Logger::beginSection("Additional info:");
    Logger::printParam("RNS_MODULI_SIZE", RNS_MODULI_SIZE);
    Logger::printParam("MPRES_GRID_SIZE_X_ESI", MPRES_GRID_SIZE_X_ESI);
    Logger::printParam("MPRES_BLOCK_SIZE_ESI", MPRES_BLOCK_SIZE_ESI);
    Logger::printParam("MPRES_GRID_SIZE_DIGITS", MPRES_GRID_SIZE_DIGITS);
    #ifndef EXCLUDE_CAMPARY
    Logger::printParam("CAMPARY_PRECISION (n-double)", CAMPARY_PRECISION);
    #endif
    Logger::endSection(true);

    //Run the test
    runTest();

    //Finalize
    finalize();

    //End logging
    Logger::endTestDescription();
    return 0;
}