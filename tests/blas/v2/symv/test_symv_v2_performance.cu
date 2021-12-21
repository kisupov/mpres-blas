/*
 *  Performance test for BLAS SYMV routines
 *
 *  Copyright 2021 by Konstantin Isupov.
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

#include "logger.cuh"
#include "tsthelper.cuh"
#include "blas/mblas_enum.cuh"
#include "test_double_symv.cuh"
#include "test_mpfr_symv.cuh"
#include "test_openblas_symv.cuh"
#include "test_cublas_symv.cuh"
#include "test_mpres_symv.cuh"
#include "test_campary_symv.cuh"

#define N 5000  // Number of matrix rows / column and the vectors dimension
#define LDA (N) // Specifies the leading dimension of A as declared in the calling (sub)program.
#define UPLO mblas_upper // Specifies whether the upper or lower triangular part of the array a is used.
#define INCX 1 // Specifies the increment for the elements of x.
#define INCY 1 // Specifies the increment for the elements of y.
#define REPEAT_TEST 10 //Number of repeats

int MP_PRECISION_DEC;
int INP_BITS;
int INP_DIGITS;

void setPrecisions() {
    MP_PRECISION_DEC = (int) (MP_PRECISION / 3.32 + 1);
    INP_BITS = (int) (MP_PRECISION / 4);
    INP_DIGITS = (int) (INP_BITS / 3.32 + 1);
}

void initialize() {
    cudaDeviceReset();
    rns_const_init();
    mp_const_init();
    setPrecisions();
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();
}

void finalize() {
    cudaDeviceReset();
}

void test() {
    //Actual length of the vectors
    int lenx = (1 + (N - 1) * abs(INCX));
    int leny = (1 + (N - 1) * abs(INCY));
    //Inputs
    mpfr_t *vectorX = create_random_array(lenx, INP_BITS);
    mpfr_t *vectorY = create_random_array(leny, INP_BITS);
    mpfr_t *matrixA = create_random_array(LDA * N, INP_BITS);
    mpfr_t *alpha = create_random_array(1, INP_BITS);
    mpfr_t *beta = create_random_array(1, INP_BITS);
    //Launch tests
    test_openblas(UPLO, N, lenx, leny, alpha[0], matrixA, LDA, vectorX, INCX, beta[0], vectorY, INCY, REPEAT_TEST);
    test_double(UPLO, N, lenx, leny, alpha[0], matrixA, LDA, vectorX, beta[0], vectorY, REPEAT_TEST);
    test_mpfr(UPLO, N, alpha[0], matrixA, LDA, vectorX, beta[0], vectorY, REPEAT_TEST);
    test_cublas(UPLO, N, lenx, leny, alpha[0], matrixA, LDA, vectorX, INCX, beta[0], vectorY, INCY, REPEAT_TEST);
    test_double_symv_cuda(UPLO, N, lenx, leny, alpha[0], matrixA, LDA, vectorX, INCX, beta[0], vectorY, INCY, REPEAT_TEST);
    test_mpres_symv(UPLO, N, lenx, leny, alpha[0], matrixA, LDA, vectorX, INCX, beta[0], vectorY, INCY, REPEAT_TEST);
    test_campary_symv<CAMPARY_PRECISION>(UPLO, N, lenx, leny, alpha[0], matrixA, LDA, vectorX, INCX, beta[0], vectorY, INCY, INP_DIGITS, REPEAT_TEST);
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();
    //Cleanup
    for (int i = 0; i < LDA * N; i++) {
        mpfr_clear(matrixA[i]);
    }
    for (int i = 0; i < lenx; i++) {
        mpfr_clear(vectorX[i]);
    }
    for (int i = 0; i < leny; i++) {
        mpfr_clear(vectorY[i]);
    }
    mpfr_clear(alpha[0]);
    mpfr_clear(beta[0]);
    delete[] matrixA;
    delete[] vectorX;
    delete[] vectorY;
    delete[] alpha;
    delete[] beta;
    cudaDeviceReset();
}
int main() {
    initialize();
    Logger::beginTestDescription(Logger::BLAS_SYMV_PERFORMANCE_TEST);
    Logger::printTestParameters(N * N, REPEAT_TEST, MP_PRECISION, MP_PRECISION_DEC);
    Logger::beginSection("Operation info:");
    Logger::printParam("Matrix rows and columns, N", N);
    Logger::printParam("LDA", LDA);
    Logger::printParam("INCX", INCX);
    Logger::printParam("INCY", INCY);
    Logger::printParam("UPLO", UPLO);
    Logger::printDash();
    Logger::beginSection("Additional info:");
    Logger::printParam("RNS_MODULI_SIZE", RNS_MODULI_SIZE);
    Logger::printParam("CAMPARY_PRECISION (n-double)", CAMPARY_PRECISION);
    Logger::endSection(true);
    test();
    finalize();
    Logger::endTestDescription();
    return 0;
}