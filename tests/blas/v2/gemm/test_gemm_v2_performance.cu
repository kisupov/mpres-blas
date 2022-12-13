/*
 *  Performance test for BLAS GEMM routines (matrix-matrix product with general matrices)
 *
 *  Copyright 2022 by Konstantin Isupov.
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
#include "test_mpres_gemm.cuh"
#include "test_openblas_gemm.cuh"

/*
#include "test_double_ger.cuh"
#include "test_mpfr_ger.cuh"
#include "test_cublas_ger.cuh"
#include "test_campary_ger.cuh"
*/

#define M 500  // Specifies the number of rows of the matrix A and of the matrix C.
#define N 500   // Specifies the number of columns of the matrix B and the number of columns of the matrix C.
#define K 500  // Specifies the number of columns of the matrix A and the number of rows of the matrix B.
#define LDA (M) // Specifies the leading dimension of A.
#define LDB (K) // Specifies the leading dimension of B.
#define LDC (M) // Specifies the leading dimension of C.
#define TRANSA mblas_no_trans // Specifies the form of op(A) used in the matrix multiplication
#define TRANSB mblas_no_trans // Specifies the form of op(B) used in the matrix multiplication
#define REPEAT_TEST 1 //Number of repeats

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


    //Inputs
    auto SIZE_A = LDA * K;
    auto SIZE_B = LDB * N;
    if(TRANSA == mblas_trans){
        SIZE_A = LDA * M;
    }
    if(TRANSB == mblas_trans){
        SIZE_B = LDB * K;
    }
    mpfr_t *matrixA = create_random_array(SIZE_A, INP_BITS);
    mpfr_t *matrixB = create_random_array(SIZE_B, INP_BITS);
    mpfr_t *matrixC = create_random_array(LDC * N, INP_BITS);
    mpfr_t *alpha = create_random_array(1,  INP_BITS);
    mpfr_t *beta = create_random_array(1, INP_BITS);
    //Launch tests
    test_openblas(TRANSA, TRANSB, M, N, K, alpha[0], matrixA, LDA, matrixB, LDB, beta[0], matrixC, LDC, REPEAT_TEST);
   // test_double(M, N, alpha[0], vectorX, INCX, vectorY, INCY, matrixA, LDA, REPEAT_TEST);
   // test_mpfr(M, N, alpha[0], vectorX, INCX, vectorY, INCY, matrixA, LDA, REPEAT_TEST);
   // test_cublas(M, N, alpha[0], vectorX, INCX, vectorY, INCY, matrixA, LDA, REPEAT_TEST);
   // test_double_cuda(M, N, alpha[0], vectorX, INCX, vectorY, INCY, matrixA, LDA, REPEAT_TEST);
    test_mpres_gemm(TRANSA, TRANSB, M, N, K, alpha[0], matrixA, LDA, matrixB, LDB, beta[0], matrixC, LDC, REPEAT_TEST);
   // test_campary_ger<CAMPARY_PRECISION>(M, N, alpha[0], vectorX, INCX, vectorY, INCY, matrixA, LDA, INP_DIGITS, REPEAT_TEST);
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();
    //Cleanup
    for(int i = 0; i < SIZE_A; i++){
        mpfr_clear(matrixA[i]);
    }
    for(int i = 0; i < SIZE_B; i++){
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
int main() {
    initialize();
    Logger::beginTestDescription(Logger::BLAS_GEMM_PERFORMANCE_TEST);
    Logger::printTestParameters(0, REPEAT_TEST, MP_PRECISION, MP_PRECISION_DEC);
    Logger::beginSection("Operation info:");
    Logger::printParam("M", M);
    Logger::printParam("N", N);
    Logger::printParam("K", K);
    Logger::printParam("LDA", LDA);
    Logger::printParam("LDB", LDB);
    Logger::printParam("LDC", LDC);
    Logger::printParam("TRANSA", TRANSA);
    Logger::printParam("TRANSB", TRANSB);
    Logger::printDash();
    Logger::beginSection("Additional info:");
    Logger::printParam("RNS_MODULI_SIZE", RNS_MODULI_SIZE);
   // Logger::printParam("CAMPARY_PRECISION (n-double)", CAMPARY_PRECISION);
    Logger::endSection(true);
    test();
    finalize();
    Logger::endTestDescription();
    return 0;
}