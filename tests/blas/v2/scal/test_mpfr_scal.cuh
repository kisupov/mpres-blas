/*
 *  Performance test for MPFR SCAL
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
#ifndef TEST_MPFR_SCAL_CUH
#define TEST_MPFR_SCAL_CUH

#include "logger.cuh"
#include "timers.cuh"
#include "tsthelper.cuh"
#include "mpfr.h"

void mpfr_scal(int n, mpfr_t alpha, mpfr_t *x, mpfr_t *r) {
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        mpfr_mul(r[i], x[i], alpha, MPFR_RNDN);
    }
}

void test_mpfr(const int n, mpfr_t alpha, mpfr_t *x, const int repeats){
    InitCpuTimer();
    Logger::printDash();
    PrintTimerName("[CPU] MPFR scal");

    // Init
    mpfr_t *hr = new mpfr_t[n];
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        mpfr_init2(hr[i], MP_PRECISION);
    }
    for(int i = 0; i < repeats; i ++) {
        #pragma omp parallel for
        for (int j = 0; j < n; j++) {
            mpfr_set(hr[j], x[j], MPFR_RNDN);
        }
        StartCpuTimer();
        mpfr_scal(n, alpha, hr, hr);
        EndCpuTimer();
    }
    PrintCpuTimer("took");
    print_mpfr_sum(hr, n);

    //Cleanup
    for (int i = 0; i < n; i++) {
        mpfr_clear(hr[i]);
    }
    delete[] hr;
}

#endif //TEST_MPFR_SCAL_CUH
