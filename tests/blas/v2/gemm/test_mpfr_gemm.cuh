/*
 *  Performance test for MPFR matrix-matrix product with general matrices (GEMM)
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
#ifndef TEST_MPFR_GEMM_CUH
#define TEST_MPFR_GEMM_CUH

#include "logger.cuh"
#include "timers.cuh"
#include "tsthelper.cuh"
#include "mpfr.h"

void mpfr_gemm(enum mblas_trans_type transa, enum mblas_trans_type transb, const int m, const int n, const int k, mpfr_t alpha, mpfr_t *A, const int lda, mpfr_t *B, const int ldb, mpfr_t beta, mpfr_t *C, const int ldc) {
    #pragma omp parallel shared(m, n, k, lda, ldb, ldc, A, B, C)
    {
        mpfr_t dot;
        mpfr_t prod;
        mpfr_init2(dot, MP_PRECISION);
        mpfr_init2(prod, MP_PRECISION);
        #pragma omp for
        for (int row = 0; row < m; row++) {
            for (int col = 0; col < n; col++) {
                mpfr_set_d(dot, 0.0, MPFR_RNDN);
                for (int i = 0; i < k; i++) {
                    unsigned int indexA = row + lda * i;
                    unsigned int indexB = i + ldb * col;
                    if (transa == mblas_trans) {
                        indexA = i + lda * row;
                    }
                    if (transb == mblas_trans) {
                        indexB = col + ldb * i;
                    }
                    mpfr_mul(prod, A[indexA], B[indexB], MPFR_RNDN);
                    mpfr_add(dot, dot, prod, MPFR_RNDN);
                }
                mpfr_mul(dot, dot, alpha, MPFR_RNDN);
                mpfr_mul(prod, beta, C[row + col * ldc], MPFR_RNDN);
                mpfr_add(C[row + col * ldc], dot, prod, MPFR_RNDN);
            }
        }
        mpfr_clear(dot);
        mpfr_clear(prod);
    }
}

void test_mpfr(enum mblas_trans_type transa, enum mblas_trans_type transb, const int m, const int n, const int k, mpfr_t alpha, mpfr_t *A, const int lda, mpfr_t *B, const int ldb, mpfr_t beta, mpfr_t *C, const int ldc, const int repeats) {
    InitCpuTimer();
    Logger::printDash();
    PrintTimerName("[CPU] MPFR gemm");

    // Init
    mpfr_t *result = new mpfr_t[ldc * n];
    #pragma omp parallel for
    for (int i = 0; i < ldc * n; i++) {
        mpfr_init2(result[i], MP_PRECISION);
    }

    // Launch
    for (int i = 0; i < repeats; i++) {
        #pragma omp parallel for
        for (int j = 0; j < ldc * n; j++) {
            mpfr_set(result[j], C[j], MPFR_RNDN);
        }
        StartCpuTimer();
        mpfr_gemm(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, result, ldc);
        EndCpuTimer();
    }
    PrintAndResetCpuTimer("took");
    print_mpfr_sum(result, ldc * n);

    //Cleanup
    for (int i = 0; i < ldc * n; i++) {
        mpfr_clear(result[i]);
    }
    delete[] result;
}

#endif //TEST_MPFR_GEMM_CUH
