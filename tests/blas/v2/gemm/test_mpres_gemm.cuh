/*
 *  Performance test for MPRES matrix-matrix product with general matrices (GEMM)
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
#ifndef TEST_MPRES_GEMM_CUH
#define TEST_MPRES_GEMM_CUH

#include "logger.cuh"
#include "timers.cuh"
#include "tsthelper.cuh"
#include "blas/v2/gemm_v2.cuh"

void test_mpres_gemm(enum mblas_trans_type transa, enum mblas_trans_type transb, const int m, const int n, const int k, mpfr_t alpha, mpfr_t *A, const int lda, mpfr_t *B, const int ldb, mpfr_t beta, mpfr_t *C, const int ldc, const int repeats) {
    InitCudaTimer();
    Logger::printDash();
    PrintTimerName("[GPU] MPRES-BLAS gemm");

    //Execution configuration
    auto threadsX = 16;
    auto threadsY = 16;
    dim3 dimBlock(threadsX, threadsY);
    auto blocksX = (m + dimBlock.x - 1) / dimBlock.x;
    auto blocksY = (n + dimBlock.y - 1) / dimBlock.y;
    dim3 dimGrid(blocksX, blocksY);
    Logger::printKernelExecutionConfig2D(threadsX, threadsY, blocksX, blocksY);

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
    mp_float_ptr hA = new mp_float_t[sizeA];
    mp_float_ptr hB = new mp_float_t[sizeB];
    mp_float_ptr hC = new mp_float_t[ldc * n];
    mp_float_t halpha;
    mp_float_t hbeta;

    // GPU data
    mp_float_ptr dA;
    mp_float_ptr dB;
    mp_float_ptr dC;
    mp_float_ptr dalpha;
    mp_float_ptr dbeta;

    cudaMalloc(&dA, sizeof(mp_float_t) * sizeA);
    cudaMalloc(&dB, sizeof(mp_float_t) * sizeB);
    cudaMalloc(&dC, sizeof(mp_float_t) * ldc * n);
    cudaMalloc(&dalpha, sizeof(mp_float_t));
    cudaMalloc(&dbeta, sizeof(mp_float_t));

    // Convert from MPFR
    convert_matrix(hA, A, lda, (transa == mblas_trans ? m : k));
    convert_matrix(hB, B, ldb, (transb == mblas_trans ? k : n));
    convert_matrix(hC, C, ldc, n);
    mp_set_mpfr(&halpha, alpha);
    mp_set_mpfr(&hbeta, beta);

    //Copying to the GPU
    cudaMemcpy(dA, hA, sizeA * sizeof(mp_float_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, sizeB * sizeof(mp_float_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dalpha, &halpha, sizeof(mp_float_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dbeta, &hbeta, sizeof(mp_float_t), cudaMemcpyHostToDevice);
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    for (int i = 0; i < repeats; i++) {
        cudaMemcpy(dC, hC, ldc * n * sizeof(mp_float_t), cudaMemcpyHostToDevice);
        StartCudaTimer();
        cuda::mp_gemm<<<dimGrid, dimBlock>>>(transa, transb, m, n, k, dalpha, dA, lda, dB, ldb, dbeta, dC, ldc);
        EndCudaTimer();
    }
    PrintAndResetCudaTimer("took");
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    //Copying to the host
    cudaMemcpy(hC, dC, ldc * n * sizeof(mp_float_t), cudaMemcpyDeviceToHost);
    print_mp_sum(hC, ldc * n);

    //Cleanup
    delete[] hA;
    delete[] hB;
    delete[] hC;
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    cudaFree(dalpha);
    cudaFree(dbeta);
}

#endif //TEST_MPRES_GEMM_CUH
