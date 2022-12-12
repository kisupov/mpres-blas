/*
 *  Performance test for MPFR symmetric rank-2 update (SYR2)
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
#ifndef TEST_MPFR_SYR2_CUH
#define TEST_MPFR_SYR2_CUH

#include "logger.cuh"
#include "timers.cuh"
#include "tsthelper.cuh"
#include "mpfr.h"

void mpfr_syr2(enum mblas_uplo_type uplo, int n, mpfr_t alpha, mpfr_t *x, int incx, mpfr_t *y, int incy, mpfr_t *A, int lda) {
    if (uplo == mblas_upper) { //Access the upper part of the matrix
        #pragma omp parallel shared(n, A, x)
        {
            mpfr_t axy;
            mpfr_t ayx;
            mpfr_init2(axy, MP_PRECISION);
            mpfr_init2(ayx, MP_PRECISION);
            #pragma omp for
            for (int col = 0; col < n; col++) {
                for (int row = 0; row <= col; row++) {
                    auto irx = incx > 0 ? row * incx : (-n + row + 1) * incx;
                    auto icx = incx > 0 ? col * incx : (-n + col + 1) * incx;
                    auto iry = incy > 0 ? row * incy : (-n + row + 1) * incy;
                    auto icy = incy > 0 ? col * incy : (-n + col + 1) * incy;
                    mpfr_mul(axy, x[irx], y[icy], MPFR_RNDN);
                    mpfr_mul(axy, axy, alpha, MPFR_RNDN);
                    mpfr_mul(ayx, y[iry], x[icx], MPFR_RNDN);
                    mpfr_mul(ayx, ayx, alpha, MPFR_RNDN);
                    mpfr_add(axy, axy, ayx, MPFR_RNDN);
                    mpfr_add(A[row + col * lda], A[row + col * lda], axy, MPFR_RNDN);
                }
            }
            mpfr_clear(axy);
            mpfr_clear(ayx);
        }
    } else { //Access the upper part of the matrix
        #pragma omp parallel shared(n, A, x)
        {
            mpfr_t axy;
            mpfr_t ayx;
            mpfr_init2(axy, MP_PRECISION);
            mpfr_init2(ayx, MP_PRECISION);
            #pragma omp for
            for (int col = 0; col < n; col++) {
                for (int row = col; row < n; row++) {
                    auto irx = incx > 0 ? row * incx : (-n + row + 1) * incx;
                    auto icx = incx > 0 ? col * incx : (-n + col + 1) * incx;
                    auto iry = incy > 0 ? row * incy : (-n + row + 1) * incy;
                    auto icy = incy > 0 ? col * incy : (-n + col + 1) * incy;
                    mpfr_mul(axy, x[irx], y[icy], MPFR_RNDN);
                    mpfr_mul(axy, axy, alpha, MPFR_RNDN);
                    mpfr_mul(ayx, y[iry], x[icx], MPFR_RNDN);
                    mpfr_mul(ayx, ayx, alpha, MPFR_RNDN);
                    mpfr_add(axy, axy, ayx, MPFR_RNDN);
                    mpfr_add(A[row + col * lda], A[row + col * lda], axy, MPFR_RNDN);
                }
            }
            mpfr_clear(axy);
            mpfr_clear(ayx);
        }
    }
}

void test_mpfr(enum mblas_uplo_type uplo, const int n, mpfr_t alpha, mpfr_t *x, const int incx, mpfr_t *y, const int incy, mpfr_t *A, const int lda, const int repeats) {
    InitCpuTimer();
    Logger::printDash();
    PrintTimerName("[CPU] MPFR syr2");

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
        mpfr_syr2(uplo, n, alpha, x, incx, y, incy, result, lda);
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

#endif //TEST_MPFR_SYR2_CUH
