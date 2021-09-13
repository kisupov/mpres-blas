/*
 *  Performance test for BLAS GER routines
 *
 *  Copyright 2020 by Konstantin Isupov and Ivan Babeshko
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
#define EXCLUDE_XBLAS
#define EXCLUDE_ARPREC
#define EXCLUDE_MPDECIMAL
#define EXCLUDE_CUBLAS
#define EXCLUDE_GARPREC
#define EXCLUDE_CAMPARY
#define EXCLUDE_CUMP

#include "../../logger.cuh"
#include "../../timers.cuh"
#include "../../tsthelper.cuh"
#include "../../../src/mparray.cuh"
#include "../../../src/blas/ger.cuh"
#include "blas/external/3rdparty.cuh"

#define M 500  // Number of matrix rows and the vector X dimension
#define N 500  // Number of matrix columns and the vector Y dimension
#define LDA (M) // Specifies the leading dimension of A as declared in the calling (sub)program.
#define INCX 1 // Specifies the increment for the elements of x.
#define INCY 1 // Specifies the increment for the elements of y.
#define REPEAT_TEST 10 //Number of repeats

//Execution configuration for mp_ger
#define MPRES_BLOCK_SIZE_X_ESI 32
#define MPRES_BLOCK_SIZE_Y_ESI 1
#define MPRES_GRID_SIZE_X_DIGITS 128
#define MPRES_GRID_SIZE_Y_DIGITS 64

#define OPENBLAS_THREADS 4

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

/********************* GER implementations and benchmarks *********************/

/////////
// OpenBLAS
/////////
extern "C" void openblas_set_num_threads(int num_threads);

void openblas_test(int m, int n, int lenx, int leny, mpfr_t alpha, mpfr_t *A, int lda, mpfr_t *x, int incx, mpfr_t *y, int incy){
    InitCpuTimer();
    Logger::printDash();
    PrintTimerName("[CPU] OpenBLAS ger");

    openblas_set_num_threads(OPENBLAS_THREADS);

    //CPU data
    double *dx = new double[lenx];
    double *dy = new double[leny];
    double *dr = new double[lda * n];
    double *dA = new double[lda * n];
    double dalpha = mpfr_get_d(alpha, MPFR_RNDN);

    for (int i = 0; i < lenx; ++i) {
        dx[i] = mpfr_get_d(x[i], MPFR_RNDN);
    }

    for (int i = 0; i < leny; ++i) {
        dy[i] = mpfr_get_d(y[i], MPFR_RNDN);
    }

    for (int i = 0; i < lda * n; ++i) {
        dA[i] = mpfr_get_d(A[i], MPFR_RNDN);
    }

    //Launch
    for(int i = 0; i < REPEAT_TEST; i++){
        for (int j = 0; j < lda * n; j++) {
            dr[j] = dA[j];
        }
        StartCpuTimer();
        cblas_dger(CblasColMajor, m, n, dalpha, dx, incx, dy, incy, dr, lda);
        EndCpuTimer();
    }
    PrintCpuTimer("took");
    print_double_sum(dr, lda * n);
    delete [] dx;
    delete [] dy;
    delete [] dr;
    delete [] dA;
}

/////////
// MPRES-BLAS (structure of arrays)
/////////
void mpres_test(int m, int n, int lenx, int leny, mpfr_t alpha, mpfr_t *A, int lda, mpfr_t *x, int incx, mpfr_t *y, int incy){
    InitCudaTimer();
    Logger::printDash();
    PrintTimerName("[GPU] MPRES-BLAS ger");

    // Host data
    mp_float_ptr hx = new mp_float_t[lenx];
    mp_float_ptr hy = new mp_float_t[leny];
    mp_float_ptr hA = new mp_float_t[lda * n];
    mp_float_t halpha;

    //GPU data
    mp_array_t dx;
    mp_array_t dy;
    mp_array_t dA;
    mp_array_t dalpha;
    mp_array_t dbuf1;
    mp_array_t dbuf2;

    //Init data
    cuda::mp_array_init(dx, lenx);
    cuda::mp_array_init(dy, leny);
    cuda::mp_array_init(dA, lda * n);
    cuda::mp_array_init(dalpha, 1);
    cuda::mp_array_init(dbuf1, n);
    cuda::mp_array_init(dbuf2, m * n);

    // Convert from MPFR
    convert_vector(hx, x, lenx);
    convert_vector(hy, y, leny);
    convert_matrix(hA, A, lda, n);
    mp_set_mpfr(&halpha, alpha);

    //Copying to the GPU
    cuda::mp_array_host2device(dx, hx, lenx);
    cuda::mp_array_host2device(dy, hy, leny);
    cuda::mp_array_host2device(dalpha, &halpha, 1);


    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    //Launch
    for (int i = 0; i < REPEAT_TEST; i++) {
        cuda::mp_array_host2device(dA, hA, lda * n);
        StartCudaTimer();
        cuda::mp_ger<
                MPRES_BLOCK_SIZE_X_ESI,
                MPRES_BLOCK_SIZE_Y_ESI,
                MPRES_GRID_SIZE_X_DIGITS,
                MPRES_GRID_SIZE_Y_DIGITS>
                (m, n, dalpha, dx, incx,dy, incy, dA, lda, dbuf1, dbuf2);
        EndCudaTimer();
    }
    PrintCudaTimer("took");
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    //Copying to the host
    cuda::mp_array_device2host(hA, dA, lda * n);
    print_mp_sum(hA, lda * n);

    //Cleanup
    delete [] hx;
    delete [] hy;
    delete [] hA;
    cuda::mp_array_clear(dx);
    cuda::mp_array_clear(dy);
    cuda::mp_array_clear(dA);
    cuda::mp_array_clear(dalpha);
    cuda::mp_array_clear(dbuf1);
    cuda::mp_array_clear(dbuf2);
}


/********************* Main test *********************/

/*
 * x is of size m
 * y is of size n
 * a is of size lda * n, where the value of lda must be at least max(1, m).
 */
void test(){
    //Actual length of the vectors
    int lenx = (1 + (M - 1) * abs(INCX));
    int leny = (1 + (N - 1) * abs(INCY));

    //Inputs
    mpfr_t *vectorX = create_random_array(lenx, INP_BITS);
    mpfr_t *vectorY = create_random_array(leny, INP_BITS);
    mpfr_t *matrixA = create_random_array(LDA * N, INP_BITS);
    mpfr_t *alpha = create_random_array(1, INP_BITS);

    //Tests
    openblas_test(M, N, lenx, leny, alpha[0], matrixA, LDA, vectorX, INCX, vectorY, INCY);
    #ifndef EXCLUDE_MPACK
    mpack_ger_test(M, N, lenx, leny, alpha[0], matrixA, LDA, vectorX, INCX, vectorY, INCY, MP_PRECISION, REPEAT_TEST);
    #endif
    mpres_test(M, N, lenx, leny, alpha[0], matrixA, LDA, vectorX, INCX, vectorY, INCY);
    #ifndef EXCLUDE_CUMP
    cump_ger_test(M, N, alpha[0], matrixA, LDA, vectorX, vectorY, MP_PRECISION, INP_DIGITS, REPEAT_TEST);
    #endif


    checkDeviceHasErrors(cudaDeviceSynchronize());
    // cudaCheckErrors(); //CUMP gives failure

    //Cleanup
    for(int i = 0; i < LDA * N; i++){
        mpfr_clear(matrixA[i]);
    }
    for(int i = 0; i < lenx; i++){
        mpfr_clear(vectorX[i]);
    }
    for(int i = 0; i < leny; i++){
        mpfr_clear(vectorY[i]);
    }

    mpfr_clear(alpha[0]);
    delete [] matrixA;
    delete [] vectorX;
    delete [] vectorY;
    delete [] alpha;
    cudaDeviceReset();
}


int main(){

    initialize();

    //Start logging
    Logger::beginTestDescription(Logger::BLAS_GER_PERFORMANCE_TEST);
    Logger::printTestParameters(N * M, REPEAT_TEST, MP_PRECISION, MP_PRECISION_DEC);
    Logger::beginSection("Operation info:");
    Logger::printParam("Matrix rows, M", M);
    Logger::printParam("Matrix columns, N", N);
    Logger::printParam("LDA", LDA);
    Logger::printDash();
    Logger::beginSection("Additional info:");
    Logger::printParam("RNS_MODULI_SIZE", RNS_MODULI_SIZE);
    Logger::printParam("MPRES_BLOCK_SIZE_X_ESI", MPRES_BLOCK_SIZE_X_ESI);
    Logger::printParam("MPRES_BLOCK_SIZE_Y_ESI", MPRES_BLOCK_SIZE_Y_ESI);
    Logger::printParam("MPRES_GRID_SIZE_X_DIGITS", MPRES_GRID_SIZE_X_DIGITS);
    Logger::printParam("MPRES_GRID_SIZE_Y_DIGITS", MPRES_GRID_SIZE_Y_DIGITS);
    Logger::endSection(true);

    //Run the test
    test();

    //Finalize
    finalize();

    //End logging
    Logger::endTestDescription();

    return 0;
}