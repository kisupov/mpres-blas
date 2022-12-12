/*
 *  Performance test for MPFR general rank-1 update (GER)
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
#ifndef TEST_MPFR_GER_CUH
#define TEST_MPFR_GER_CUH

#include "logger.cuh"
#include "timers.cuh"
#include "tsthelper.cuh"
#include "mpfr.h"

void mpfr_ger(int m, int n, mpfr_t alpha, mpfr_t *x, int incx, mpfr_t *y, int incy, mpfr_t *A, int lda) {
    #pragma omp parallel shared(m, n, A, x, y)
    {
        mpfr_t axy;
        mpfr_init2(axy, MP_PRECISION);
        #pragma omp for
        for (int col = 0; col < n; col++) {
            for (int row = 0; row < m; row++) {
                auto ix = incx > 0 ? row * incx : (-m + row + 1) * incx;
                auto iy = incy > 0 ? col * incy : (-n + col + 1) * incy;
                mpfr_mul(axy, alpha, x[ix], MPFR_RNDN);
                mpfr_mul(axy, axy, y[iy], MPFR_RNDN);
                mpfr_add(A[row + col * lda], A[row + col * lda], axy, MPFR_RNDN);
            }
        }
        mpfr_clear(axy);
    }
}

void test_mpfr(const int m, const int n, mpfr_t alpha, mpfr_t *x, const int incx, mpfr_t *y, const int incy, mpfr_t *A, const int lda, const int repeats) {
    InitCpuTimer();
    Logger::printDash();
    PrintTimerName("[CPU] MPFR ger");

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
        mpfr_ger(m, n, alpha, x, incx, y, incy, result, lda);
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

#endif //TEST_MPFR_GER_CUH
