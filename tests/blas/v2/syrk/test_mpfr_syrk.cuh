/*
 *  Performance test for MPFR symmetric rank-k update (SYRK)
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
#ifndef TEST_MPFR_SYRK_CUH
#define TEST_MPFR_SYRK_CUH

#include "logger.cuh"
#include "timers.cuh"
#include "tsthelper.cuh"
#include "mpfr.h"

void mpfr_syrk(enum mblas_uplo_type uplo, enum mblas_trans_type trans, const int n, const int k, mpfr_t alpha, mpfr_t *A, const int lda, mpfr_t beta, mpfr_t *C, const int ldc) {
        #pragma omp parallel shared(n, k, lda, ldc, A, C)
        {
            mpfr_t dot;
            mpfr_t prod;
            mpfr_init2(dot, MP_PRECISION);
            mpfr_init2(prod, MP_PRECISION);
            #pragma omp for
            for (int col = 0; col < n; col++) {
                for (int row = ((uplo == mblas_upper) ? 0 : col); row <= ((uplo == mblas_upper) ? col : n - 1); row++) {
                    mpfr_set_d(dot, 0.0, MPFR_RNDN);
                    for (int i = 0; i < k; i++) {
                        auto indexA = row + lda * i;
                        auto indexAT = col + lda * i;
                        if (trans == mblas_trans) {
                            indexA = i + lda * row;
                            indexAT = i + lda * col;
                        }
                        mpfr_mul(prod, A[indexA], A[indexAT], MPFR_RNDN);
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

void test_mpfr(enum mblas_uplo_type uplo, enum mblas_trans_type trans, const int n, const int k, mpfr_t alpha, mpfr_t *A, const int lda, mpfr_t beta, mpfr_t *C, const int ldc, const int repeats) {
    InitCpuTimer();
    Logger::printDash();
    PrintTimerName("[CPU] MPFR syrk");

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
        mpfr_syrk(uplo, trans, n, k, alpha, A, lda, beta, result, ldc);
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

#endif //TEST_MPFR_SYRK_CUH
