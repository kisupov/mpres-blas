/*
 *  Performance test for cuBLAS matrix-matrix product with general matrices (GEMM)
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
#ifndef TEST_CUBLAS_GEMM_CUH
#define TEST_CUBLAS_GEMM_CUH

#include "logger.cuh"
#include "timers.cuh"
#include "tsthelper.cuh"
#include "cublas_v2.h"

void test_cublas(enum mblas_trans_type transa, enum mblas_trans_type transb, const int m, const int n, const int k, mpfr_t alpha, mpfr_t *A, const int lda,
        mpfr_t *B, const int ldb, mpfr_t beta, mpfr_t *C, const int ldc, const int repeats){
    InitCudaTimer();
    Logger::printDash();
    PrintTimerName("[GPU] cuBLAS gemm");

    cublasStatus_t stat;
    cublasHandle_t handle;
    stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("cuBLAS initialization failed\n");
        return;
    }

    //Size of arrays storing matrices
    auto sizeA = lda * k;
    auto sizeB = ldb * n;
    if(transa == mblas_trans){
        sizeA = lda * m;
    }
    if(transb == mblas_trans){
        sizeB = ldb * k;
    }

    //Host data
    double *hA = new double[sizeA];
    double *hB = new double[sizeB];
    double *hC = new double[ldc * n];
    convert_vector(hA, A, sizeA);
    convert_vector(hB, B, sizeB);
    convert_vector(hC, C, ldc * n);

    //GPU data
    double *dA;
    double *dB;
    double *dC;
    double dalpha = mpfr_get_d(alpha, MPFR_RNDN);
    double dbeta = mpfr_get_d(beta, MPFR_RNDN);

    cudaMalloc(&dA, sizeof(double) * sizeA);
    cudaMalloc(&dB, sizeof(double) * sizeB);
    cudaMalloc(&dC, sizeof(double) * ldc * n);
    cublasSetVector(sizeA, sizeof(double), hA, 1, dA, 1);
    cublasSetVector(sizeB, sizeof(double), hB, 1, dB, 1);

    //Transpose
    cublasOperation_t transposeA = CUBLAS_OP_N;
    cublasOperation_t transposeB = CUBLAS_OP_N;
    if(transa == mblas_trans){
        transposeA = CUBLAS_OP_T;
    }
    if(transb == mblas_trans){
        transposeB = CUBLAS_OP_T;
    }

    for(int i = 0; i < repeats; i ++) {
        cublasSetVector(ldc * n, sizeof(double), hC, 1, dC, 1);
        StartCudaTimer();
        cublasDgemm(handle, transposeA, transposeB, m, n, k, &dalpha, dA, lda, dB, ldb, &dbeta, dC, ldc);
        EndCudaTimer();
    }
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();
    PrintAndResetCudaTimer("took");
    cublasGetVector(ldc * n, sizeof(double), dC, 1, hC, 1);
    print_double_sum(hC, ldc * n);
    cublasDestroy ( handle );
    delete[] hA;
    delete[] hB;
    delete[] hC;
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
}

#endif //TEST_CUBLAS_GEMM_CUH
