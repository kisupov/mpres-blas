/*
 *  Performance test for BLAS GE_NORM routines
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
#define EXCLUDE_OPENBLAS
#define EXCLUDE_XBLAS
#define EXCLUDE_ARPREC
#define EXCLUDE_MPACK
#define EXCLUDE_MPDECIMAL
#define EXCLUDE_CUMP
#define EXCLUDE_GARPREC
#define EXCLUDE_CAMPARY
#define EXCLUDE_CUBLAS

#include "../../logger.cuh"
#include "../../timers.cuh"
#include "../../tsthelper.cuh"
#include "../../../src/mparray.cuh"
#include "../../../src/blas/genorm.cuh"
#include "blas/external/3rdparty.cuh"

#define M 500  // Number of matrix rows and the vector X dimension
#define N 500  // Number of matrix columns and the vector Y dimension
#define LDA (M) // Specifies the leading dimension of A as declared in the calling (sub)program.
#define REPEAT_TEST 10 //Number of repeats
#define NORM_TYPE "INF" //ONE = one norm, INF = infinity-norm

//Execution configuration for mpres
#define MPRES_CUDA_BLOCKS_REDUCE   128
#define MPRES_CUDA_THREADS_REDUCE  32

int MP_PRECISION_DEC; //in decimal digits
int INP_BITS; //in bits
int INP_DIGITS; //in decimal digits

static void setPrecisions(){
    MP_PRECISION_DEC = (int)(MP_PRECISION / 3.32 + 1);
    INP_BITS = (int)(MP_PRECISION / 2);
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

/********************* GE_NORM implementations and benchmarks *********************/

/////////
// MPFR
/////////
static void mpfr_max_val(int n, mpfr_t *x, mpfr_t result) {
    mpfr_set_d(result, 0, MPFR_RNDN);
    for (int i = 0; i < n; i++) {
        if(mpfr_cmp(x[i], result) > 0){
            mpfr_set(result, x[i], MPFR_RNDN);
        }
    }
}

//Sum of elements in each column
static void mpfr_matrix_col_sum(int m, int n, mpfr_t *A, int lda, mpfr_t *result, int precision){
    #pragma omp parallel shared(m, n, A, result)
    {
        mpfr_t absval;
        mpfr_init2(absval, precision);
        #pragma omp for
        for (int j = 0; j < n; j++) {
            mpfr_set_d(result[j], 0, MPFR_RNDN);
            for (int i = 0; i < m; i++) {
                mpfr_abs(absval, A[i + j * lda], MPFR_RNDN);
                mpfr_add(result[j], result[j], absval, MPFR_RNDN);
            }
        }
        mpfr_clear(absval);
    }
}

//Sum of elements in each row
static void mpfr_matrix_row_sum(int m, int n, mpfr_t *A, int lda, mpfr_t *result, int precision){
    #pragma omp parallel shared(m, n, A, result)
    {
        mpfr_t absval;
        mpfr_init2(absval, precision);
        #pragma omp for
        for (int i = 0; i < m; i++) {
            mpfr_set_d(result[i], 0, MPFR_RNDN);
            for (int j = 0; j < n; j++) {
                mpfr_abs(absval, A[i + j * lda], MPFR_RNDN);
                mpfr_add(result[i], result[i], absval, MPFR_RNDN);
            }
        }
        mpfr_clear(absval);
    }
}

void mpfr_test(int m, int n, mpfr_t *A, int lda){
    InitCpuTimer();
    Logger::printDash();
    PrintTimerName("[CPU] MPFR ge_norm");

    int lenb = (NORM_TYPE == "ONE") ? n : m;

    // Init
    mpfr_t *mbuf = new mpfr_t[lenb];
    mpfr_t result;

    #pragma omp parallel for
    for(int j = 0; j < lenb; j++){
        mpfr_init2(mbuf[j], MP_PRECISION);
    }
    mpfr_init2(result, MP_PRECISION);

    // Launch
    if(NORM_TYPE == "ONE"){
        for(int i = 0; i < REPEAT_TEST; i++){
            StartCpuTimer();
            mpfr_matrix_col_sum(m, n, A, lda, mbuf, MP_PRECISION);
            mpfr_max_val(n, mbuf, result);
            EndCpuTimer();
        }
    }
    else{
        for(int i = 0; i < REPEAT_TEST; i++){
            StartCpuTimer();
            mpfr_matrix_row_sum(m, n, A, lda, mbuf, MP_PRECISION);
            mpfr_max_val(m, mbuf, result);
            EndCpuTimer();
        }
    }
    PrintAndResetCpuTimer("took");
    mpfr_printf("result: %.70Rf \n", result);

    //Cleanup
    for(int i = 0; i < lenb; i++){
        mpfr_clear(mbuf[i]);
    }
    mpfr_clear(result);
    delete [] mbuf;
}

/////////
// MPRES-BLAS
/////////
void mpres_test(int m, int n, mpfr_t *A, int lda){
    InitCudaTimer();
    Logger::printDash();
    PrintTimerName("[GPU] MPRES-BLAS ge_norm");

    int lenb = (NORM_TYPE == "ONE") ? n : m;

    // Host data
    mp_float_ptr hA = new mp_float_t[lda * n];
    mp_float_t hresult = MP_ZERO;

    //GPU data
    mp_array_t dA;
    mp_array_t dbuf;
    mp_array_t dresult;

    //Init data
    cuda::mp_array_init(dA, lda * n);
    cuda::mp_array_init(dbuf, lenb);
    cuda::mp_array_init(dresult, 1);

    // Convert from MPFR
    convert_matrix(hA, A, lda, n);

    //Copying the matrix to the GPU
    cuda::mp_array_host2device(dA, hA, lda * n);

    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    //Launch
    for (int i = 0; i < REPEAT_TEST; i++) {
        StartCudaTimer();
        cuda::mp_ge_norm<
                        MPRES_CUDA_BLOCKS_REDUCE,
                        MPRES_CUDA_THREADS_REDUCE>
                ( ((NORM_TYPE == "ONE") ? mblas_one_norm : mblas_inf_norm), m, n, dA, lda, dresult, dbuf);
        EndCudaTimer();
    }
    PrintAndResetCudaTimer("took");
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    mpfr_t mpfr_result;
    mpfr_init2(mpfr_result, MP_PRECISION);
    mpfr_set_d(mpfr_result, 0, MPFR_RNDN);

    //Copying to the host
    cuda::mp_array_device2host(&hresult, dresult, 1);
    mp_get_mpfr(mpfr_result, hresult);
    mpfr_printf("result: %.70Rf \n", mpfr_result);

    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    //Cleanup
    delete [] hA;
    mpfr_clear(mpfr_result);
    cuda::mp_array_clear(dA);
    cuda::mp_array_clear(dbuf);
    cuda::mp_array_clear(dresult);

}


/********************* Main test *********************/

void runTest(){

    //Inputs
    mpfr_t *matrixA = create_random_array(LDA * N, INP_BITS);
    mpfr_test(M, N, matrixA, LDA);
    mpres_test(M, N, matrixA, LDA);
    checkDeviceHasErrors(cudaDeviceSynchronize());

    //Cleanup
    for(int i = 0; i < LDA * N; i++){
        mpfr_clear(matrixA[i]);
    }
    delete [] matrixA;
    cudaDeviceReset();
}


int main(){
    initialize();

    //Start logging
    Logger::beginTestDescription(Logger::BLAS_GE_NORM_PERFORMANCE_TEST);
    Logger::printTestParameters(M * N, REPEAT_TEST, MP_PRECISION, MP_PRECISION_DEC);
    Logger::beginSection("Operation info:");
    Logger::printParam("Matrix rows, M", M);
    Logger::printParam("Matrix columns, N", N);
    Logger::printParam("LDA", LDA);
    Logger::printParam("NORM_TYPE", NORM_TYPE);
    Logger::printDash();
    Logger::beginSection("Additional info:");
    Logger::printParam("RNS_MODULI_SIZE", RNS_MODULI_SIZE);
    Logger::printParam("MPRES_CUDA_BLOCKS_REDUCE", MPRES_CUDA_BLOCKS_REDUCE);
    Logger::printParam("MPRES_CUDA_THREADS_REDUCE", MPRES_CUDA_THREADS_REDUCE);
    Logger::endSection(true);

    //Run the test
    runTest();

    //Finalize
    finalize();

    //End logging
    Logger::endTestDescription();
    return 0;
}