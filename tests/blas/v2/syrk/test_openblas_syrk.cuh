/*
 *  Performance test for OpenBLAS symmetric rank-k update (SYRK)
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
#ifndef TEST_OPENBLAS_SYRK_CUH
#define TEST_OPENBLAS_SYRK_CUH

#include "logger.cuh"
#include "timers.cuh"
#include "tsthelper.cuh"
#include "cblas.h"

#define OPENBLAS_THREADS 4

extern "C" void openblas_set_num_threads(int num_threads);

void test_openblas(enum mblas_uplo_type uplo, enum mblas_trans_type trans, const int n, const int k, mpfr_t alpha, mpfr_t *A, const int lda, mpfr_t beta, mpfr_t *C, const int ldc, const int repeats) {
    InitCpuTimer();
    Logger::printDash();
    PrintTimerName("[CPU] OpenBLAS syrk");

    openblas_set_num_threads(OPENBLAS_THREADS);

    //Size of an array storing matrix A
    auto sizeA = (trans == mblas_trans) ? lda * n : lda * k;

    //CPU data
    double *dA = new double[sizeA];
    double *dC = new double[ldc * n];
    double *result = new double[ldc * n];
    double dalpha = mpfr_get_d(alpha, MPFR_RNDN);
    double dbeta = mpfr_get_d(beta, MPFR_RNDN);

    for (int i = 0; i < sizeA; i++) {
        dA[i] = mpfr_get_d(A[i], MPFR_RNDN);
    }
    for (int i = 0; i < ldc * n; i++) {
        dC[i] = mpfr_get_d(C[i], MPFR_RNDN);
    }

    //Transpose and uplo
    CBLAS_TRANSPOSE cblasTranspose = (trans == mblas_trans) ? CblasTrans : CblasNoTrans;
    CBLAS_UPLO cblasUplo = (uplo == mblas_upper) ? CblasUpper : CblasLower;

    //Launch
    for(int i = 0; i < repeats; i ++){
        for (int j = 0; j < ldc * n; j++) {
            result[j] = dC[j];
        }
        StartCpuTimer();
        cblas_dsyrk(CblasColMajor, cblasUplo, cblasTranspose, n, k, dalpha, dA, lda, dbeta, result, ldc);
        EndCpuTimer();
    }
    PrintAndResetCpuTimer("took");
    print_double_sum(result, ldc * n);
    delete [] dA;
    delete [] dC;
    delete [] result;
}

#endif //TEST_OPENBLAS_SYRK_CUH
