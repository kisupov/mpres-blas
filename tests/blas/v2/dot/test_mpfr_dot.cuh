/*
 *  Performance test for MPFR DOT
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
#ifndef TEST_MPFR_DOT_CUH
#define TEST_MPFR_DOT_CUH

#include "logger.cuh"
#include "timers.cuh"
#include "tsthelper.cuh"
#include "mpfr.h"

void mpfr_dot(const int n, mpfr_t *x, mpfr_t *y, mpfr_t r){
    #pragma omp parallel shared(r)
    {
        mpfr_t tmp_sum;
        mpfr_t tmp;
        mpfr_init2(tmp_sum, MP_PRECISION);
        mpfr_init2(tmp, MP_PRECISION);
        mpfr_set_d(tmp_sum, 0, MPFR_RNDN);
        mpfr_set_d(tmp, 0, MPFR_RNDN);

        #pragma omp for
        for(int i =0; i < n; i ++){
            mpfr_mul(tmp, x[i], y[i], MPFR_RNDN);
            mpfr_add(tmp_sum, tmp_sum, tmp, MPFR_RNDN);
        }
        #pragma omp critical
        {
            mpfr_add(r, r, tmp_sum, MPFR_RNDN);
        }
        mpfr_clear(tmp);
        mpfr_clear(tmp_sum);
    }
}

void test_mpfr(const int n, mpfr_t *x, mpfr_t *y, const int repeats){
    InitCpuTimer();
    Logger::printDash();
    PrintTimerName("[CPU] MPFR dot");
    mpfr_t dot;
    mpfr_init2(dot, MP_PRECISION);
    for(int i = 0; i < repeats; i ++) {
        mpfr_set_d(dot, 0, MPFR_RNDN);
        StartCpuTimer();
        mpfr_dot(n, x, y, dot);
        EndCpuTimer();
    }
    PrintAndResetCpuTimer("took");
    mpfr_printf("result: %.70Rf \n", dot);
    mpfr_clear(dot);
}

#endif //TEST_MPFR_DOT_CUH
