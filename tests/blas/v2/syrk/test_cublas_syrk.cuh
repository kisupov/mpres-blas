/*
 *  Performance test for cuBLAS symmetric rank-k update (SYRK)
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
#ifndef TEST_CUBLAS_SYRK_CUH
#define TEST_CUBLAS_SYRK_CUH

#include "logger.cuh"
#include "timers.cuh"
#include "tsthelper.cuh"
#include "cublas_v2.h"

void test_cublas(enum mblas_uplo_type uplo, enum mblas_trans_type trans, const int n, const int k, mpfr_t alpha, mpfr_t *A, const int lda, mpfr_t beta, mpfr_t *C, const int ldc, const int repeats){
    InitCudaTimer();
    Logger::printDash();
    PrintTimerName("[GPU] cuBLAS syrk");

    cublasStatus_t stat;
    cublasHandle_t handle;
    stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("cuBLAS initialization failed\n");
        return;
    }

    //Size of an array storing matrix A
    auto sizeA = (trans == mblas_trans) ? lda * n : lda * k;

    //Host data
    double *hA = new double[sizeA];
    double *hC = new double[ldc * n];
    convert_vector(hA, A, sizeA);
    convert_vector(hC, C, ldc * n);

    //GPU data
    double *dA;
    double *dC;
    double dalpha = mpfr_get_d(alpha, MPFR_RNDN);
    double dbeta = mpfr_get_d(beta, MPFR_RNDN);

    cudaMalloc(&dA, sizeof(double) * sizeA);
    cudaMalloc(&dC, sizeof(double) * ldc * n);
    cublasSetVector(sizeA, sizeof(double), hA, 1, dA, 1);

    //Transpose and uplo
    cublasOperation_t cublasTranspose = (trans == mblas_trans) ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasFillMode_t cublasUplo = (uplo == mblas_upper) ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;

    for(int i = 0; i < repeats; i ++) {
        cublasSetVector(ldc * n, sizeof(double), hC, 1, dC, 1);
        StartCudaTimer();
        cublasDsyrk_v2(handle, cublasUplo, cublasTranspose, n, k, &dalpha, dA, lda, &dbeta, dC, ldc);
        EndCudaTimer();
    }
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();
    PrintAndResetCudaTimer("took");
    cublasGetVector(ldc * n, sizeof(double), dC, 1, hC, 1);
    print_double_sum(hC, ldc * n);
    cublasDestroy ( handle );
    delete[] hA;
    delete[] hC;
    cudaFree(dA);
    cudaFree(dC);
}

#endif //TEST_CUBLAS_SYRK_CUH
