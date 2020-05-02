/*
 *  Performance test for BLAS GE_LRSCALE routines
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

#define EXCLUDE_ARPREC 1
#define EXCLUDE_XBLAS 1
#define EXCLUDE_MPDECIMAL 1
#define EXCLUDE_GARPREC 1
#define EXCLUDE_CUBLAS 1
#define EXCLUDE_OPENBLAS 1

#include "omp.h"
#include "../../logger.cuh"
#include "../../timers.cuh"
#include "../../tsthelper.cuh"
#include "../../../src/blas/mpgelrscale.cuh"
#include "3rdparty.cuh"

#define M 300  // Number of matrix rows and the vector X dimension
#define N 300  // Number of matrix columns and the vector Y dimension
#define LDA (M+1) // Specifies the leading dimension of A as declared in the calling (sub)program.
#define INCDL (-1) // Specifies the increment for the elements of DL.
#define INCDR (1) // Specifies the increment for the elements of DR.
#define REPEAT_TEST 10 //Number of repeats


//Execution configuration for mpres
#define MPRES_GRID_SIZE_X_ESI 128
#define MPRES_BLOCK_SIZE_ESI 128
#define MPRES_GRID_SIZE_DIGITS 128

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
}

static void convert_matrix(mp_float_ptr dest, mpfr_t *source, int rows, int cols){
    int width = rows * cols;
    for( int i = 0; i < width; i++ ){
        mp_set_mpfr(&dest[i], source[i]);
    }
}


/********************* GE_ADD implementations and benchmarks *********************/

/////////
// MPFR
/////////
void mpfr_ge_lrscale(int m, int n, mpfr_t *DL, int incdl, mpfr_t *DR, int incdr, mpfr_t *A, int lda){
    #pragma omp parallel for shared(m, n, A, DL, DR)
    for (int j = 0; j < n; j++) {
        int idr = incdr > 0 ? j * incdr : (-n + 1 + j)*incdr;
        for (int i = 0; i < m; i++) {
            int idl = incdl > 0 ? i * incdl : (-m + 1 + i)*incdl;
            mpfr_mul(A[i + j * lda], A[i + j * lda], DL[idl], MPFR_RNDN);
            mpfr_mul(A[i + j * lda], A[i + j * lda], DR[idr], MPFR_RNDN);
        }
    }
}

void mpfr_test(int m, int n, mpfr_t *DL, int incdl, mpfr_t *DR, int incdr, mpfr_t *A, int lda){
    InitCpuTimer();
    Logger::printDash();
    PrintTimerName("[CPU] MPFR ge_lrscale");

    // Init
    mpfr_t *mA = new mpfr_t[lda * n];
    #pragma omp parallel for
    for(int j = 0; j < lda * n; j++){
        mpfr_init2(mA[j], MP_PRECISION);
    }

    // Launch
    for(int i = 0; i < REPEAT_TEST; i++){
        #pragma omp parallel for
        for(int j = 0; j < lda * n; j++){
            mpfr_set(mA[j], A[j], MPFR_RNDN);
        }
        StartCpuTimer();
        mpfr_ge_lrscale(m, n, DL, incdl, DR, incdr, mA, lda);
        EndCpuTimer();
    }

    PrintCpuTimer("took");
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
void mpres_test(int m, int n, mpfr_t *DL, int incdl, mpfr_t *DR, int incdr, mpfr_t *A, int lda){
    InitCudaTimer();
    Logger::printDash();
    PrintTimerName("[GPU] MPRES-BLAS ge_lrscale");

    int lendl = (1 + (m - 1) * abs(incdl));
    int lendr = (1 + (n - 1) * abs(incdr));

    // Host data
    mp_float_ptr hA = new mp_float_t[lda * n];
    mp_float_ptr hDL = new mp_float_t[lendl];
    mp_float_ptr hDR = new mp_float_t[lendr];

    //GPU data
    mp_array_t dA;
    mp_array_t dDL;
    mp_array_t dDR;

    //Init data
    cuda::mp_array_init(dA, lda * n);
    cuda::mp_array_init(dDL, lendl);
    cuda::mp_array_init(dDR, lendr);

    // Convert from MPFR
    convert_matrix(hA, A, lda, n);
    convert_matrix(hDL, DL, lendl, 1);
    convert_matrix(hDR, DR, lendr, 1);

    //Copying the diagonal matrix to the GPU
    cuda::mp_array_host2device(dDL, hDL, lendl);
    cuda::mp_array_host2device(dDR, hDR, lendr);

    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();
    //Launch
    for (int i = 0; i < REPEAT_TEST; i++) {
        cuda::mp_array_host2device(dA, hA, lda * n);
        StartCudaTimer();
                cuda::mpgelrscale<
                        MPRES_GRID_SIZE_X_ESI,
                        MPRES_BLOCK_SIZE_ESI,
                        MPRES_GRID_SIZE_DIGITS>
                        (m, n, dDL, incdl, dDR, incdr, dA, lda);
        EndCudaTimer();
    }
    PrintCudaTimer("took");
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    //Copying to the host
    cuda::mp_array_device2host(hA, dA, lda * n);
    print_mp_sum(hA, lda * n);

    //Cleanup
    delete [] hA;
    delete [] hDL;
    delete [] hDR;
    cuda::mp_array_clear(dA);
    cuda::mp_array_clear(dDL);
    cuda::mp_array_clear(dDR);
}


/********************* Main test *********************/

void runTest(){

    int LENDL = (1 + (M - 1) * abs(INCDL));
    int LENDR = (1 + (N - 1) * abs(INCDR));

    //Inputs
    mpfr_t *matrixA = create_random_array(LDA * N, INP_BITS);
    mpfr_t *matrixDL = create_random_array(LENDL, INP_BITS);
    mpfr_t *matrixDR = create_random_array(LENDR, INP_BITS);
    //Multiple-precision tests
    mpfr_test(M, N, matrixDL, INCDL, matrixDR, INCDR, matrixA, LDA);
    mpres_test(M, N, matrixDL, INCDL, matrixDR, INCDR, matrixA, LDA);
    campary_ge_lrscale_test<CAMPARY_PRECISION>(M, N, matrixDL, INCDL, matrixDR, INCDR, matrixA, LDA, INP_DIGITS, REPEAT_TEST);
    cump_ge_lrscale_test(M, N, matrixDL, INCDL, matrixDR, INCDR, matrixA, LDA, MP_PRECISION, INP_DIGITS, REPEAT_TEST);

    checkDeviceHasErrors(cudaDeviceSynchronize());

    //Cleanup
    for(int i = 0; i < LDA * N; i++){
        mpfr_clear(matrixA[i]);
    }
    for(int i = 0; i < LENDL; i++){
        mpfr_clear(matrixDL[i]);
    }
    for(int i = 0; i < LENDR; i++){
        mpfr_clear(matrixDR[i]);
    }
    delete [] matrixA;
    delete [] matrixDL;
    delete [] matrixDR;
    cudaDeviceReset();
}


int main(){
    initialize();

    //Start logging
    Logger::beginTestDescription(Logger::BLAS_GE_LRSCALE_PERFORMANCE_TEST);
    Logger::printTestParameters(M * N, REPEAT_TEST, MP_PRECISION, MP_PRECISION_DEC);
    Logger::beginSection("Operation info:");
    Logger::printParam("Matrix rows, M", M);
    Logger::printParam("Matrix columns, N", N);
    Logger::printParam("LDA", LDA);
    Logger::printParam("INCDL", INCDL);
    Logger::printParam("INCDR", INCDR);
    Logger::printDash();
    Logger::beginSection("Additional info:");
    Logger::printParam("RNS_MODULI_SIZE", RNS_MODULI_SIZE);
    Logger::printParam("MPRES_GRID_SIZE_X_ESI", MPRES_GRID_SIZE_X_ESI);
    Logger::printParam("MPRES_BLOCK_SIZE_ESI", MPRES_BLOCK_SIZE_ESI);
    Logger::printParam("MPRES_GRID_SIZE_DIGITS", MPRES_GRID_SIZE_DIGITS);
    Logger::printParam("CAMPARY_PRECISION (n-double)", CAMPARY_PRECISION);
    Logger::endSection(true);

    //Run the test
    runTest();

    //Finalize
    finalize();

    //End logging
    Logger::endTestDescription();
    return 0;
}