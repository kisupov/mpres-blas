/*
 *  Performance test for BLAS SYRK routines (symmetric rank-k update)
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
#include "test_mpres_syrk.cuh"
//#include "test_openblas_gemm.cuh"
//#include "test_double_gemm.cuh"
//#include "test_mpfr_gemm.cuh"
//#include "test_campary_gemm.cuh"
//#include "test_cublas_gemm.cuh"

#define N 500 // Specifies the number of rows and columns in matrix C.
#define K 500  // Specifies the number of columns of matrix A.
#define LDA (N) // Specifies the leading dimension of A.
#define LDC (N) // Specifies the leading dimension of C.
#define UPLO mblas_upper // Specifies whether the upper or lower triangular part of the array C is used.
#define TRANS mblas_no_trans // Specifies the form of op(A) used in the matrix multiplication
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
    if(TRANSA == mblas_trans){
        SIZE_A = LDA * N;
    }
    mpfr_t *matrixA = create_random_array(SIZE_A, INP_BITS);
    mpfr_t *matrixC = create_random_array(LDC * N, INP_BITS);
    mpfr_t *alpha = create_random_array(1,  INP_BITS);
    mpfr_t *beta = create_random_array(1, INP_BITS);
    //Launch tests
//    test_openblas(TRANSA, TRANSB, M, N, K, alpha[0], matrixA, LDA, matrixB, LDB, beta[0], matrixC, LDC, REPEAT_TEST);
//    test_double(TRANSA, TRANSB, M, N, K, alpha[0], matrixA, LDA, matrixB, LDB, beta[0], matrixC, LDC, REPEAT_TEST);
//    test_mpfr(TRANSA, TRANSB, M, N, K, alpha[0], matrixA, LDA, matrixB, LDB, beta[0], matrixC, LDC, REPEAT_TEST);
//    test_cublas(TRANSA, TRANSB, M, N, K, alpha[0], matrixA, LDA, matrixB, LDB, beta[0], matrixC, LDC, REPEAT_TEST);
//    test_double_cuda(TRANSA, TRANSB, M, N, K, alpha[0], matrixA, LDA, matrixB, LDB, beta[0], matrixC, LDC, REPEAT_TEST);
    test_mpres_gemm(UPLO, TRANS, N, K, alpha[0], matrixA, LDA, beta[0], matrixC, LDC, REPEAT_TEST);
//    test_campary_gemm<CAMPARY_PRECISION>(TRANSA, TRANSB, M, N, K, alpha[0], matrixA, LDA, matrixB, LDB, beta[0], matrixC, LDC, INP_DIGITS, REPEAT_TEST);
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();
    //Cleanup
    for(int i = 0; i < SIZE_A; i++){
        mpfr_clear(matrixA[i]);
    }
    for(int i = 0; i < LDC * N; i++){
        mpfr_clear(matrixC[i]);
    }
    mpfr_clear(alpha[0]);
    mpfr_clear(beta[0]);
    delete [] matrixA;
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
    Logger::printParam("N", N);
    Logger::printParam("K", K);
    Logger::printParam("LDA", LDA);
    Logger::printParam("LDC", LDC);
    Logger::printParam("TRANS", TRANS);
    Logger::printParam("UPLO", UPLO);
    Logger::printDash();
    Logger::beginSection("Additional info:");
    Logger::printParam("RNS_MODULI_SIZE", RNS_MODULI_SIZE);
    //Logger::printParam("CAMPARY_PRECISION (n-double)", CAMPARY_PRECISION);
    Logger::endSection(true);
    test();
    finalize();
    Logger::endTestDescription();
    return 0;
}