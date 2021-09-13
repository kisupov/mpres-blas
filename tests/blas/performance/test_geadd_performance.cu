/*
 *  Performance test for BLAS GE_ADD routines
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
#include "../../../src/blas/geadd.cuh"
#include "blas/external/3rdparty.cuh"

#define M 500  // Number of matrix rows and the vector X dimension
#define N 500  // Number of matrix columns and the vector Y dimension
#define LDA (M) // Specifies the leading dimension of A as declared in the calling (sub)program.
#define LDB (M) // Specifies the leading dimension of B as declared in the calling (sub)program.
#define LDC (M) // Specifies the leading dimension of C as declared in the calling (sub)program.
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
void mpfr_ge_add(int m, int n, mpfr_t alpha, mpfr_t *A, int lda, mpfr_t beta, mpfr_t *B, int ldb, mpfr_t *C, int ldc){
    #pragma omp parallel shared(m, n, A, B, C)
    {
        mpfr_t temp1;
        mpfr_t temp2;
        mpfr_init2(temp1, MP_PRECISION);
        mpfr_init2(temp2, MP_PRECISION);
        #pragma omp for
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < m; i++) {
                mpfr_mul(temp1, alpha, A[i + j * lda], MPFR_RNDN);
                mpfr_mul(temp2, beta, B[i + j * ldb], MPFR_RNDN);
                mpfr_add(C[i + j * ldc], temp1, temp2, MPFR_RNDN);
            }
        }
        mpfr_clear(temp1);
        mpfr_clear(temp2);
    }
}

void mpfr_test(int m, int n, mpfr_t alpha, mpfr_t *A, int lda, mpfr_t beta, mpfr_t *B, int ldb, mpfr_t *C, int ldc){
    InitCpuTimer();
    Logger::printDash();
    PrintTimerName("[CPU] MPFR ge_add");

    // Init
    mpfr_t *mC = new mpfr_t[ldc * n];
    #pragma omp parallel for
    for(int i = 0; i < ldc * n; i++){
        mpfr_init2(mC[i], MP_PRECISION);
        mpfr_set(mC[i], C[i], MPFR_RNDN);
    }

    // Launch
    for(int i = 0; i < REPEAT_TEST; i++){
        StartCpuTimer();
        mpfr_ge_add(m, n, alpha, A, lda, beta, B, ldb, mC, ldc);
        EndCpuTimer();
    }
    PrintCpuTimer("took");
    print_mpfr_sum(mC, ldc * n);

    //Cleanup
    for(int i = 0; i < ldc * n; i++){
        mpfr_clear(mC[i]);
    }
    delete [] mC;
}

/////////
// MPRES-BLAS
/////////
void mpres_test(int m, int n, mpfr_t alpha, mpfr_t *A, int lda, mpfr_t beta, mpfr_t *B, int ldb, mpfr_t *C, int ldc){
    InitCudaTimer();
    Logger::printDash();
    PrintTimerName("[GPU] MPRES-BLAS ge_add");

    // Host data
    mp_float_t halpha;
    mp_float_t hbeta;
    mp_float_ptr hA = new mp_float_t[lda * n];
    mp_float_ptr hB = new mp_float_t[ldb * n];
    mp_float_ptr hC = new mp_float_t[ldc * n];

    //GPU data
    mp_array_t dalpha;
    mp_array_t dbeta;
    mp_array_t dA;
    mp_array_t dB;
    mp_array_t dC;
    mp_array_t dbuf;

    //Init data
    cuda::mp_array_init(dalpha, 1);
    cuda::mp_array_init(dbeta, 1);
    cuda::mp_array_init(dA, lda * n);
    cuda::mp_array_init(dB, ldb * n);
    cuda::mp_array_init(dC, ldc * n);
    cuda::mp_array_init(dbuf, m * n);

    // Convert from MPFR
    mp_set_mpfr(&halpha, alpha);
    mp_set_mpfr(&hbeta, beta);
    convert_matrix(hA, A, lda, n);
    convert_matrix(hB, B, ldb, n);
    convert_matrix(hC, C, ldc, n);

    //Copying to the GPU
    cuda::mp_array_host2device(dalpha, &halpha, 1);
    cuda::mp_array_host2device(dbeta, &hbeta, 1);
    cuda::mp_array_host2device(dA, hA, lda * n);
    cuda::mp_array_host2device(dB, hB, ldb * n);
    cuda::mp_array_host2device(dC, hC, ldc * n);

    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    //Cleanup
    delete [] hA;
    delete [] hB;

    //Launch
    for (int i = 0; i < REPEAT_TEST; i++) {
        StartCudaTimer();
        cuda::mp_ge_add<
                MPRES_BLOCK_SIZE_X_ESI,
                MPRES_BLOCK_SIZE_Y_ESI,
                MPRES_GRID_SIZE_X_DIGITS,
                MPRES_GRID_SIZE_Y_DIGITS>
                (m, n, dalpha, dA, lda, dbeta, dB, ldb, dC, ldc, dbuf);
        EndCudaTimer();
    }
    PrintCudaTimer("took");
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    //Copying to the host
    cuda::mp_array_device2host(hC, dC, ldc * n);
    print_mp_sum(hC, ldc * n);

    //Cleanup
    delete [] hC;
    cuda::mp_array_clear(dalpha);
    cuda::mp_array_clear(dbeta);
    cuda::mp_array_clear(dA);
    cuda::mp_array_clear(dB);
    cuda::mp_array_clear(dC);
    cuda::mp_array_clear(dbuf);
}


/********************* Main test *********************/

void runTest(){

    //Inputs
    mpfr_t *alpha = create_random_array(1, INP_BITS);
    mpfr_t *beta = create_random_array(1, INP_BITS);
    mpfr_t *matrixA = create_random_array(LDA * N, INP_BITS);
    mpfr_t *matrixB = create_random_array(LDB * N, INP_BITS);
    mpfr_t *matrixC = create_random_array(LDC * N, INP_BITS);

    //Multiple-precision tests
    mpfr_test(M, N, alpha[0], matrixA, LDA, beta[0], matrixB, LDB, matrixC, LDC);
    mpres_test(M, N, alpha[0], matrixA, LDA, beta[0], matrixB, LDB, matrixC, LDC);
    #ifndef EXCLUDE_CAMPARY
    campary_ge_add_test<CAMPARY_PRECISION>(M, N, alpha[0], matrixA, LDA, beta[0], matrixB, LDB, matrixC, LDC, INP_DIGITS, REPEAT_TEST);
    #endif
    #ifndef EXCLUDE_CUMP
    cump_ge_add_test(M, N, alpha[0], matrixA, LDA, beta[0], matrixB, LDB, matrixC, LDC, MP_PRECISION, INP_DIGITS, REPEAT_TEST);
    #endif

    checkDeviceHasErrors(cudaDeviceSynchronize());

    //Cleanup
    for(int i = 0; i < LDA * N; i++){
        mpfr_clear(matrixA[i]);
    }
    for(int i = 0; i < LDB * N; i++){
        mpfr_clear(matrixB[i]);
    }
    for(int i = 0; i < LDC * N; i++){
        mpfr_clear(matrixC[i]);
    }
    mpfr_clear(alpha[0]);
    mpfr_clear(beta[0]);
    delete [] matrixA;
    delete [] matrixB;
    delete [] matrixC;
    delete [] alpha;
    delete [] beta;
    cudaDeviceReset();
}


int main(){
    initialize();

    //Start logging
    Logger::beginTestDescription(Logger::BLAS_GE_ADD_PERFORMANCE_TEST);
    Logger::printTestParameters(M * N, REPEAT_TEST, MP_PRECISION, MP_PRECISION_DEC);
    Logger::beginSection("Operation info:");
    Logger::printParam("Matrix rows, M", M);
    Logger::printParam("Matrix columns, N", N);
    Logger::printParam("LDA", LDA);
    Logger::printParam("LDB", LDB);
    Logger::printParam("LDC", LDC);
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