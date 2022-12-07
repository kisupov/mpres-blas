/*
 *  Performance test for OpenBLAS symmetric rank-1 update (SYR)
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
#ifndef TEST_OPENBLAS_SYR_CUH
#define TEST_OPENBLAS_SYR_CUH

#include "logger.cuh"
#include "timers.cuh"
#include "tsthelper.cuh"
#include "cblas.h"

#define OPENBLAS_THREADS 4

extern "C" void openblas_set_num_threads(int num_threads);

void test_openblas(enum mblas_uplo_type uplo, const int n, mpfr_t alpha, mpfr_t *x, const int incx, mpfr_t *A, const int lda, const int repeats) {
    InitCpuTimer();
    Logger::printDash();
    PrintTimerName("[CPU] OpenBLAS syr");

    openblas_set_num_threads(OPENBLAS_THREADS);

    //Actual length of the vector
    int lenx = (1 + (n - 1) * abs(incx));
    //CPU data
    double *dx = new double[lenx];
    double *dr = new double[lda * n];
    double *dA = new double[lda * n];
    double dalpha = mpfr_get_d(alpha, MPFR_RNDN);
    for (int i = 0; i < lenx; ++i) {
        dx[i] = mpfr_get_d(x[i], MPFR_RNDN);
    }
    for (int i = 0; i < lda * n; ++i) {
        dA[i] = mpfr_get_d(A[i], MPFR_RNDN);
    }
    //Launch
    for (int i = 0; i < repeats; i++) {
        for (int j = 0; j < lda * n; j++) {
            dr[j] = dA[j];
        }
        StartCpuTimer();
        if (uplo == mblas_upper) {
            cblas_dsyr(CblasColMajor, CblasUpper, n, dalpha, dx, incx, dr, lda);
        } else {
            cblas_dsyr(CblasColMajor, CblasLower, n, dalpha, dx, incx, dr, lda);
        }
        EndCpuTimer();
    }
    PrintAndResetCpuTimer("took");
    print_double_sum(dr, lda * n);
    delete[] dx;
    delete[] dr;
    delete[] dA;
}

#endif //TEST_OPENBLAS_SYR_CUH
