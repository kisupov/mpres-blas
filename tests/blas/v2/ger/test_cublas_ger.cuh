/*
 *  Performance test for cuBLAS general rank-1 update (GER)
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
#ifndef TEST_CUBLAS_GER_CUH
#define TEST_CUBLAS_GER_CUH

#include "logger.cuh"
#include "timers.cuh"
#include "tsthelper.cuh"
#include "cublas_v2.h"

void test_cublas(const int m, const int n, mpfr_t alpha, mpfr_t *x, const int incx, mpfr_t *y, const int  incy, mpfr_t *A, const int lda, const int repeats){
    InitCudaTimer();
    Logger::printDash();
    PrintTimerName("[GPU] cuBLAS ger");

    cublasStatus_t stat;
    cublasHandle_t handle;
    stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("cuBLAS initialization failed\n");
        return;
    }

    //Actual length of the vectors
    int lenx = (1 + (m - 1) * abs(incx));
    int leny = (1 + (n - 1) * abs(incy));

    //Host data
    double *hx = new double[lenx];
    double *hy = new double[leny];
    double *hA = new double[lda * n];

    for (int i = 0; i < lenx; i++) {
        hx[i] = mpfr_get_d(x[i], MPFR_RNDN);
    }

    for (int i = 0; i < leny; i++) {
        hy[i] = mpfr_get_d(y[i], MPFR_RNDN);
    }

    for (int i = 0; i < lda * n; i++) {
        hA[i] = mpfr_get_d(A[i], MPFR_RNDN);
    }

    //GPU data
    double *dx;
    double *dy;
    double *dA;
    double dalpha = mpfr_get_d(alpha, MPFR_RNDN);

    cudaMalloc(&dx, sizeof(double) * lenx);
    cudaMalloc(&dy, sizeof(double) * leny);
    cudaMalloc(&dA, sizeof(double) * lda * n);
    cublasSetVector(n, sizeof(double), hx, incx, dx, incx);
    cublasSetVector(n, sizeof(double), hy, incy, dy, incy);

    for(int i = 0; i < repeats; i ++) {
        cublasSetVector(lda * n, sizeof(double), hA, 1, dA, 1);
        StartCudaTimer();
        cublasDger(handle, m, n, &dalpha, dx, incx, dy, incy, dA, lda);
        EndCudaTimer();

    }
    PrintAndResetCudaTimer("took");
    cublasGetVector(lda * n, sizeof(double), dA, 1, hA, 1);
    print_double_sum(hA, lda * n);
    cublasDestroy ( handle );
    delete[] hx;
    delete[] hy;
    delete[] hA;
    cudaFree(dx);
    cudaFree(dy);
    cudaFree(dA);
}

#endif //TEST_CUBLAS_GER_CUH
