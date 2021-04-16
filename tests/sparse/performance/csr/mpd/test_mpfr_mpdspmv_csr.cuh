/*
 *  Performance test for SpMV CSR routine using MPFR
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

#ifndef TEST_MPFR_MPDSPMV_CSR_CUH
#define TEST_MPFR_MPDSPMV_CSR_CUH

#include "mpfr.h"
#include "../../../../tsthelper.cuh"
#include "../../../../logger.cuh"
#include "../../../../timers.cuh"

/////////
// MPFR SpMV routine using double precision matrix
/////////
void mpfr_mpdspmv_csr(const int m, const int *irp, const int *ja, const double *as, const mpfr_t *x, mpfr_t *y, const int prec) {
    mpfr_t prod;
    mpfr_init2(prod, prec);
    for(int row = 0; row < m; row++){
        mpfr_set_d(y[row], 0.0, MPFR_RNDN);
        int row_start = irp[row];
        int row_end = irp[row+1];
        for (int i = row_start; i < row_end; i++) {
            mpfr_mul_d(prod, x[ja[i]], as[i], MPFR_RNDN);
            mpfr_add(y[row],y[row],prod, MPFR_RNDN);
        }
    }
}


void test_mpfr_mpdspmv_csr(const int m, const int n, const int nnz, const int *irp, const int *ja, const double *as, const mpfr_t *x, const int prec) {
    InitCpuTimer();
    Logger::printDash();
    PrintTimerName("[CPU] MPFR SpMV CSR");

    // Init
    auto *y = new mpfr_t[m];
    #pragma omp parallel for
    for(int i = 0; i < m; i++){
        mpfr_init2(y[i], prec);
    }

    //Launch
    StartCpuTimer();
    mpfr_mpdspmv_csr(m, irp, ja, as, x, y, prec);
    EndCpuTimer();
    PrintCpuTimer("took");
    print_mpfr_sum(y, m);

    //Cleanup
    for(int i = 0; i < m; i++){
        mpfr_clear(y[i]);
    }
    delete [] y;
}

#endif //TEST_MPFR_MPDSPMV_CSR_CUH
