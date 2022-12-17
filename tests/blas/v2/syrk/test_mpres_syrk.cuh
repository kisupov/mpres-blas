/*
 *  Performance test for MPRES symmetric rank-k update (SYRK)
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
#ifndef TEST_MPRES_SYRK_CUH
#define TEST_MPRES_SYRK_CUH

#include "logger.cuh"
#include "timers.cuh"
#include "tsthelper.cuh"
#include "blas/v2/syrk_v2.cuh"

void test_mpres_gemm(enum mblas_uplo_type uplo, enum mblas_trans_type trans, const int n, const int k, mpfr_t alpha, mpfr_t *A, const int lda, mpfr_t beta, mpfr_t *C, const int ldc, const int repeats) {
    InitCudaTimer();
    Logger::printDash();
    PrintTimerName("[GPU] MPRES-BLAS syrk");

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
    if(trans == mblas_trans){
        sizeA = lda * n;
    }

    //Host data
    mp_float_ptr hA = new mp_float_t[sizeA];
    mp_float_ptr hC = new mp_float_t[ldc * n];
    mp_float_t halpha;
    mp_float_t hbeta;

    // GPU data
    mp_float_ptr dA;
    mp_float_ptr dC;
    mp_float_ptr dalpha;
    mp_float_ptr dbeta;

    cudaMalloc(&dA, sizeof(mp_float_t) * sizeA);
    cudaMalloc(&dC, sizeof(mp_float_t) * ldc * n);
    cudaMalloc(&dalpha, sizeof(mp_float_t));
    cudaMalloc(&dbeta, sizeof(mp_float_t));

    // Convert from MPFR
    convert_matrix(hA, A, lda, (trans == mblas_trans ? n : k));
    convert_matrix(hC, C, ldc, n);
    mp_set_mpfr(&halpha, alpha);
    mp_set_mpfr(&hbeta, beta);

    //Copying to the GPU
    cudaMemcpy(dA, hA, sizeA * sizeof(mp_float_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dalpha, &halpha, sizeof(mp_float_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dbeta, &hbeta, sizeof(mp_float_t), cudaMemcpyHostToDevice);
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    for (int i = 0; i < repeats; i++) {
        cudaMemcpy(dC, hC, ldc * n * sizeof(mp_float_t), cudaMemcpyHostToDevice);
        StartCudaTimer();
        cuda::mp_syrk<<<dimGrid, dimBlock>>>(uplo, trans, n, k, dalpha, dA, lda, dbeta, dC, ldc);
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
    delete[] hC;
    cudaFree(dA);
    cudaFree(dC);
    cudaFree(dalpha);
    cudaFree(dbeta);
}

#endif //TEST_MPRES_SYRK_CUH
