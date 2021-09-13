/*
 *  Performance test for BLAS GE_ACC routines
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
#include "../../../src/blas/geacc.cuh"
#include "blas/external/3rdparty.cuh"

#define M 500  // Number of matrix rows and the vector X dimension
#define N 500  // Number of matrix columns and the vector Y dimension
#define LDA (M) // Specifies the leading dimension of A as declared in the calling (sub)program.
#define LDB (M) // Specifies the leading dimension of B as declared in the calling (sub)program.
#define REPEAT_TEST 10 //Number of repeats

//Execution configuration for mp_ger
#define MPRES_BLOCK_SIZE_X_ESI 32
#define MPRES_BLOCK_SIZE_Y_ESI 1
#define MPRES_GRID_SIZE_X_DIGITS 128
#define MPRES_GRID_SIZE_Y_DIGITS 64

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
}

/********************* GE_ADD implementations and benchmarks *********************/

/////////
// MPFR
/////////
void mpfr_ge_acc(int m, int n, mpfr_t alpha, mpfr_t *A, int lda, mpfr_t beta, mpfr_t *B, int ldb){
    #pragma omp parallel shared(m, n, A, B)
    {
        mpfr_t temp1;
        mpfr_init2(temp1, MP_PRECISION);
        #pragma omp for
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < m; i++) {
                mpfr_mul(temp1, alpha, A[i + j * lda], MPFR_RNDN);
                mpfr_mul(B[i + j * ldb], beta, B[i + j * ldb], MPFR_RNDN);
                mpfr_add(B[i + j * ldb], temp1, B[i + j * ldb], MPFR_RNDN);
            }
        }
        mpfr_clear(temp1);
    }
}

void mpfr_test(int m, int n, mpfr_t alpha, mpfr_t *A, int lda, mpfr_t beta, mpfr_t *B, int ldb){
    InitCpuTimer();
    Logger::printDash();
    PrintTimerName("[CPU] MPFR ge_acc");

    // Init
    mpfr_t *mB = new mpfr_t[ldb * n];
    #pragma omp parallel for
    for(int i = 0; i < ldb * n; i++){
        mpfr_init2(mB[i], MP_PRECISION);
    }

    // Launch
    for(int i = 0; i < REPEAT_TEST; i++){
        #pragma omp parallel for
        for(int j = 0; j < ldb * n; j++){
            mpfr_set(mB[j], B[j], MPFR_RNDN);
        }
        StartCpuTimer();
        mpfr_ge_acc(m, n, alpha, A, lda, beta, B, ldb);
        EndCpuTimer();
    }
    PrintCpuTimer("took");
    print_mpfr_sum(mB, ldb * n);

    //Cleanup
    for(int i = 0; i < ldb * n; i++){
        mpfr_clear(mB[i]);
    }
    delete [] mB;
}

/////////
// MPRES-BLAS
/////////
void mpres_test(int m, int n, mpfr_t alpha, mpfr_t *A, int lda, mpfr_t beta, mpfr_t *B, int ldb){
    InitCudaTimer();
    Logger::printDash();
    PrintTimerName("[GPU] MPRES-BLAS ge_acc");

    // Host data
    mp_float_t halpha;
    mp_float_t hbeta;
    mp_float_ptr hA = new mp_float_t[lda * n];
    mp_float_ptr hB = new mp_float_t[ldb * n];

    //GPU data
    mp_array_t dalpha;
    mp_array_t dbeta;
    mp_array_t dA;
    mp_array_t dB;
    mp_array_t dbuf;

    //Init data
    cuda::mp_array_init(dalpha, 1);
    cuda::mp_array_init(dbeta, 1);
    cuda::mp_array_init(dA, lda * n);
    cuda::mp_array_init(dB, ldb * n);
    cuda::mp_array_init(dbuf, m * n);

    // Convert from MPFR
    mp_set_mpfr(&halpha, alpha);
    mp_set_mpfr(&hbeta, beta);
    convert_matrix(hA, A, lda, n);
    convert_matrix(hB, B, ldb, n);

    //Copying to the GPU
    cuda::mp_array_host2device(dalpha, &halpha, 1);
    cuda::mp_array_host2device(dbeta, &hbeta, 1);
    cuda::mp_array_host2device(dA, hA, lda * n);

    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    //Cleanup
    delete [] hA;

    //Launch
    for (int i = 0; i < REPEAT_TEST; i++) {
        cuda::mp_array_host2device(dB, hB, ldb * n);
        StartCudaTimer();
        cuda::mp_ge_acc<
                MPRES_BLOCK_SIZE_X_ESI,
                MPRES_BLOCK_SIZE_Y_ESI,
                MPRES_GRID_SIZE_X_DIGITS,
                MPRES_GRID_SIZE_Y_DIGITS>
                (m, n, dalpha, dA, lda, dbeta, dB, ldb, dbuf);
        EndCudaTimer();
    }
    PrintCudaTimer("took");
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    //Copying to the host
    cuda::mp_array_device2host(hB, dB, ldb * n);
    print_mp_sum(hB, ldb * n);

    //Cleanup
    delete [] hB;
    cuda::mp_array_clear(dalpha);
    cuda::mp_array_clear(dbeta);
    cuda::mp_array_clear(dA);
    cuda::mp_array_clear(dB);
    cuda::mp_array_clear(dbuf);
}


/********************* Main test *********************/

void runTest(){

    //Inputs
    mpfr_t *alpha = create_random_array(1, INP_BITS);
    mpfr_t *beta = create_random_array(1, INP_BITS);
    mpfr_t *matrixA = create_random_array(LDA * N, INP_BITS);
    mpfr_t *matrixB = create_random_array(LDB * N, INP_BITS);

    //Multiple-precision tests
    mpfr_test(M, N, alpha[0], matrixA, LDA, beta[0], matrixB, LDB);
    mpres_test(M, N, alpha[0], matrixA, LDA, beta[0], matrixB, LDB);
    #ifndef EXCLUDE_CAMPARY
    campary_ge_acc_test<CAMPARY_PRECISION>(M, N, alpha[0], matrixA, LDA, beta[0], matrixB, LDB, INP_DIGITS, REPEAT_TEST);
    #endif
    #ifndef EXCLUDE_CUMP
    cump_ge_acc_test(M, N, alpha[0], matrixA, LDA, beta[0], matrixB, LDB, MP_PRECISION, INP_DIGITS, REPEAT_TEST);
    #endif

    checkDeviceHasErrors(cudaDeviceSynchronize());

    //Cleanup
    for(int i = 0; i < LDA * N; i++){
        mpfr_clear(matrixA[i]);
    }
    for(int i = 0; i < LDB * N; i++){
        mpfr_clear(matrixB[i]);
    }
    mpfr_clear(alpha[0]);
    mpfr_clear(beta[0]);
    delete [] matrixA;
    delete [] matrixB;
    delete [] alpha;
    delete [] beta;
    cudaDeviceReset();
}


int main(){
    initialize();

    //Start logging
    Logger::beginTestDescription(Logger::BLAS_GE_ACC_PERFORMANCE_TEST);
    Logger::printTestParameters(M * N, REPEAT_TEST, MP_PRECISION, MP_PRECISION_DEC);
    Logger::beginSection("Operation info:");
    Logger::printParam("Matrix rows, M", M);
    Logger::printParam("Matrix columns, N", N);
    Logger::printParam("LDA", LDA);
    Logger::printParam("LDB", LDB);
    Logger::printDash();
    Logger::beginSection("Additional info:");
    Logger::printParam("RNS_MODULI_SIZE", RNS_MODULI_SIZE);
    Logger::printParam("MPRES_BLOCK_SIZE_X_ESI", MPRES_BLOCK_SIZE_X_ESI);
    Logger::printParam("MPRES_BLOCK_SIZE_Y_ESI", MPRES_BLOCK_SIZE_Y_ESI);
    Logger::printParam("MPRES_GRID_SIZE_X_DIGITS", MPRES_GRID_SIZE_X_DIGITS);
    Logger::printParam("MPRES_GRID_SIZE_Y_DIGITS", MPRES_GRID_SIZE_Y_DIGITS);
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