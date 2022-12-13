/*
 *  Performance test for OpenBLAS matrix-matrix product with general matrices (GEMM)
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
#ifndef TEST_OPENBLAS_GEMM_CUH
#define TEST_OPENBLAS_GEMM_CUH

#include "logger.cuh"
#include "timers.cuh"
#include "tsthelper.cuh"
#include "cblas.h"

#define OPENBLAS_THREADS 4

extern "C" void openblas_set_num_threads(int num_threads);

void test_openblas(enum mblas_trans_type transa, enum mblas_trans_type transb, const int m, const int n, const int k, mpfr_t alpha, mpfr_t *A, const int lda, mpfr_t *B, const int ldb, mpfr_t beta, mpfr_t *C, const int ldc, const int repeats) {
    InitCpuTimer();
    Logger::printDash();
    PrintTimerName("[CPU] OpenBLAS gemm");

    openblas_set_num_threads(OPENBLAS_THREADS);

    //Size of arrays storing matrices
    auto sizeA = lda * k;
    auto sizeB = ldb * n;
    if(transa == mblas_trans){
        sizeA = lda * m;
    }
    if(transb == mblas_trans){
        sizeB = ldb * k;
    }

    //CPU data
    double *dA = new double[sizeA];
    double *dB = new double[sizeB];
    double *dC = new double[ldc * n];
    double dalpha = mpfr_get_d(alpha, MPFR_RNDN);
    double dbeta = mpfr_get_d(beta, MPFR_RNDN);

    for (int i = 0; i < sizeA; i++) {
        dA[i] = mpfr_get_d(A[i], MPFR_RNDN);
    }

    for (int i = 0; i < sizeB; i++) {
        dB[i] = mpfr_get_d(B[i], MPFR_RNDN);
    }

    //Transpose
    CBLAS_TRANSPOSE transposeA = CblasNoTrans;
    CBLAS_TRANSPOSE transposeB = CblasNoTrans;
    if(transa == mblas_trans){
        transposeA = CblasTrans;
    }
    if(transb == mblas_trans){
        transposeB = CblasTrans;
    }

    //Launch
    for(int i = 0; i < repeats; i ++){
        for (int j = 0; j < ldc * n; j++) {
            dC[j] = mpfr_get_d(C[j], MPFR_RNDN);
        }
        StartCpuTimer();
        cblas_dgemm(CblasColMajor, transposeA, transposeB, m, n, k, dalpha, dA, lda, dB, ldb, dbeta, dC, ldc);
        EndCpuTimer();
    }
    PrintAndResetCpuTimer("took");
    print_double_sum(dC, ldc * n);
    delete [] dA;
    delete [] dB;
    delete [] dC;
}

#endif //TEST_OPENBLAS_GEMM_CUH
