/*
 *  Performance test for MPFR symmetric rank-1 update (SYR)
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
#ifndef TEST_MPFR_SYR_CUH
#define TEST_MPFR_SYR_CUH

#include "logger.cuh"
#include "timers.cuh"
#include "tsthelper.cuh"
#include "mpfr.h"

void mpfr_syr(enum mblas_uplo_type uplo, int n, mpfr_t alpha, mpfr_t *x, int incx, mpfr_t *A, int lda) {
    if (uplo == mblas_upper) { //Access the upper part of the matrix
        #pragma omp parallel shared(n, A, x)
        {
            mpfr_t axx;
            mpfr_init2(axx, MP_PRECISION);
            #pragma omp for
            for (int col = 0; col < n; col++) {
                for (int row = 0; row <= col; row++) {
                    auto ir = incx > 0 ? row * incx : (-n + row + 1) * incx;
                    auto ic = incx > 0 ? col * incx : (-n + col + 1) * incx;
                    mpfr_mul(axx, x[ic], x[ir], MPFR_RNDN);
                    mpfr_mul(axx, axx, alpha, MPFR_RNDN);
                    mpfr_add(A[row + col * lda], A[row + col * lda], axx, MPFR_RNDN);
                }
            }
        }
    } else { //Access the upper part of the matrix
        #pragma omp parallel shared(n, A, x)
        {
            mpfr_t axx;
            mpfr_init2(axx, MP_PRECISION);
            #pragma omp for
            for (int col = 0; col < n; col++) {
                for (int row = col; row < n; row++) {
                    auto ir = incx > 0 ? row * incx : (-n + row + 1) * incx;
                    auto ic = incx > 0 ? col * incx : (-n + col + 1) * incx;
                    mpfr_mul(axx, x[ic], x[ir], MPFR_RNDN);
                    mpfr_mul(axx, axx, alpha, MPFR_RNDN);
                    mpfr_add(A[row + col * lda], A[row + col * lda], axx, MPFR_RNDN);
                }
            }
        }
    }
}

void test_mpfr(enum mblas_uplo_type uplo, const int n, mpfr_t alpha, mpfr_t *x, const int incx, mpfr_t *A, const int lda, const int repeats) {
    InitCpuTimer();
    Logger::printDash();
    PrintTimerName("[CPU] MPFR syr");

    // Init
    mpfr_t *result = new mpfr_t[lda * n];
    #pragma omp parallel for
    for (int i = 0; i < lda * n; i++) {
        mpfr_init2(result[i], MP_PRECISION);
    }

    // Launch
    for (int i = 0; i < repeats; i++) {
        #pragma omp parallel for
        for (int j = 0; j < lda * n; j++) {
            mpfr_set(result[j], A[j], MPFR_RNDN);
        }
        StartCpuTimer();
        mpfr_syr(uplo, n, alpha, x, incx, result, lda);
        EndCpuTimer();
    }
    PrintAndResetCpuTimer("took");
    print_mpfr_sum(result, lda * n);

    //Cleanup
    for (int i = 0; i < lda * n; i++) {
        mpfr_clear(result[i]);
    }
    delete[] result;
}

#endif //TEST_MPFR_SYR_CUH
