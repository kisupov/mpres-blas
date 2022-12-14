/*
 *   Performance test for CAMPARY matrix-matrix product with general matrices (GEMM)
 *  http://homepages.laas.fr/mmjoldes/campary/
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
#ifndef TEST_CAMPARY_GEMM_CUH
#define TEST_CAMPARY_GEMM_CUH

#include "logger.cuh"
#include "timers.cuh"
#include "tsthelper.cuh"
#include "lib/campary_common.cuh"
#include "blas/mblas_enum.cuh"

template<int prec>
__global__ static void campary_gemm_kernel(enum mblas_trans_type transa, enum mblas_trans_type transb, const int m, const int n, const int k,
        multi_prec<prec> *alpha, multi_prec<prec> *A, const int lda, multi_prec<prec> *B, const int ldb, multi_prec<prec> *beta, multi_prec<prec> *C, const int ldc) {
    auto row = blockIdx.x * blockDim.x + threadIdx.x;
    auto col = blockIdx.y * blockDim.y + threadIdx.y;
    while (col < n && row < m) {
        multi_prec<prec> dot;
        dot = 0.0;
        for (int i = 0; i < k; i++) {
            unsigned int indexA = row + lda * i;
            unsigned int indexB = i + ldb * col;
            if (transa == mblas_trans) {
                indexA = i + lda * row;
            }
            if (transb == mblas_trans) {
                indexB = col + ldb * i;
            }
            dot = dot + A[indexA] * B[indexB];
        }
        dot = dot * alpha[0];
        C[row + col * ldc] = beta[0] * C[row + col * ldc] + dot;
        row += gridDim.x * blockDim.x;
        col += gridDim.y * blockDim.y;
    }
}

template<int prec>
void test_campary_gemm(enum mblas_trans_type transa, enum mblas_trans_type transb, const int m, const int n, const int k, mpfr_t alpha, mpfr_t *A, const int lda,
        mpfr_t *B, const int ldb, mpfr_t beta, mpfr_t *C, const int ldc, const int convert_prec, const int repeats) {
    InitCudaTimer();
    Logger::printDash();
    PrintTimerName("[GPU] CAMPARY gemm");

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
    multi_prec<prec> halpha;
    multi_prec<prec> hbeta;
    auto *hA = new multi_prec<prec>[sizeA];
    auto *hB = new multi_prec<prec>[sizeB];
    auto *hC = new multi_prec<prec>[ldc * n];

    // GPU data
    multi_prec<prec> *dalpha;
    multi_prec<prec> *dbeta;
    multi_prec<prec> *dA;
    multi_prec<prec> *dB;
    multi_prec<prec> *dC;
    cudaMalloc(&dA, sizeof(multi_prec<prec>) * sizeA);
    cudaMalloc(&dB, sizeof(multi_prec<prec>) * sizeB);
    cudaMalloc(&dC, sizeof(multi_prec<prec>) * ldc * n);
    cudaMalloc(&dalpha, sizeof(multi_prec<prec>));
    cudaMalloc(&dbeta, sizeof(multi_prec<prec>));

    //Convert from MPFR
    #pragma omp parallel for
    for (int i = 0; i < sizeA; i++) {
        hA[i] = convert_to_string_sci(A[i], convert_prec).c_str();
    }
    #pragma omp parallel for
    for (int i = 0; i < sizeB; i++) {
        hB[i] = convert_to_string_sci(B[i], convert_prec).c_str();
    }
    #pragma omp parallel for
    for (int i = 0; i < ldc * n; i++) {
        hC[i] = convert_to_string_sci(C[i], convert_prec).c_str();
    }
    halpha = convert_to_string_sci(alpha, convert_prec).c_str();
    hbeta = convert_to_string_sci(beta, convert_prec).c_str();

    //Copying to the GPU
    cudaMemcpy(dA, hA, sizeA * sizeof(multi_prec<prec>), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, sizeB * sizeof(multi_prec<prec>), cudaMemcpyHostToDevice);
    cudaMemcpy(dalpha, &halpha, sizeof(multi_prec<prec>), cudaMemcpyHostToDevice);
    cudaMemcpy(dbeta, &hbeta, sizeof(multi_prec<prec>), cudaMemcpyHostToDevice);
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    for (int i = 0; i < repeats; i++) {
        cudaMemcpy(dC, hC, ldc * n * sizeof(multi_prec<prec>), cudaMemcpyHostToDevice);
        StartCudaTimer();
        campary_gemm_kernel<<<dimGrid, dimBlock>>>(transa, transb, m, n, k, dalpha, dA, lda, dB, ldb, dbeta, dC, ldc);
        EndCudaTimer();
    }PrintAndResetCudaTimer("took");
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    //Copying to the host
    cudaMemcpy(hC, dC, sizeof(multi_prec<prec>) * ldc * n, cudaMemcpyDeviceToHost);
    for (int i = 1; i < ldc * n; i++) {
        hC[0] += hC[i];
    }
    printResult<prec>(hC[0]);

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

#endif //TEST_CAMPARY_GEMM_CUH
