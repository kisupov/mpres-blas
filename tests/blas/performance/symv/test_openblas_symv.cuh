/*
 *  Performance test for OpenBLAS SYMV
 *  http://homepages.laas.fr/mmjoldes/campary/
 *
 *  Copyright 2020 by Konstantin Isupov.
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
#ifndef TEST_OPENBLAS_SYMV_CUH
#define TEST_OPENBLAS_SYMV_CUH

#include "logger.cuh"
#include "timers.cuh"
#include "tsthelper.cuh"
#include "cblas.h"

#define OPENBLAS_THREADS 4

extern "C" void openblas_set_num_threads(int num_threads);

void test_openblas(enum mblas_uplo_type uplo, const int n, int lenx, int leny, mpfr_t alpha, mpfr_t *A, int lda, mpfr_t *x,
        int incx, mpfr_t beta, mpfr_t *y, int incy, const int repeats) {
    InitCpuTimer();
    Logger::printDash();
    PrintTimerName("[CPU] OpenBLAS symv");

    openblas_set_num_threads(OPENBLAS_THREADS);

    //CPU data
    double *dx = new double[lenx];
    double *dy = new double[leny];
    double *dr = new double[leny];
    double *dA = new double[lda * n];
    double dalpha = mpfr_get_d(alpha, MPFR_RNDN);
    double dbeta = mpfr_get_d(beta, MPFR_RNDN);

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
    for (int i = 0; i < repeats; i++) {
        for (int j = 0; j < leny; j++) {
            dr[j] = dy[j];
        }
        StartCpuTimer();
        if (uplo == mblas_upper) {
            cblas_dsymv(CblasColMajor, CblasUpper, n, dalpha, dA, lda, dx, incx, dbeta, dr, incy);
        } else {
            cblas_dsymv(CblasColMajor, CblasLower, n, dalpha, dA, lda, dx, incx, dbeta, dr, incy);
        }
        EndCpuTimer();
    }
    PrintCpuTimer("took");
    print_double_sum(dr, leny);
    delete[] dx;
    delete[] dy;
    delete[] dr;
    delete[] dA;
}

#endif //TEST_OPENBLAS_SYMV_CUH
