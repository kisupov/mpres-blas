/*
 *  Performance test for double precision matrix-matrix product with general matrices (GEMM)
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
#ifndef TEST_DOUBLE_GEMM_CUH
#define TEST_DOUBLE_GEMM_CUH

#include "logger.cuh"
#include "timers.cuh"
#include "tsthelper.cuh"

/////////
// CPU
/////////
void double_gemm(enum mblas_trans_type transa, enum mblas_trans_type transb, const int m, const int n, const int k, double alpha, double *A, const int lda, double *B, const int ldb, double beta, double *C, const int ldc) {
    #pragma omp parallel shared(m, n, k, lda, ldb, ldc, A, B, C)
    {
        #pragma omp for
        for (int row = 0; row < m; row++) {
            for (int col = 0; col < n; col++) {
                double dot = 0;
                for (int i = 0; i < k; i++) {
                    unsigned int indexA = row + lda * i;
                    unsigned int indexB = i + ldb * col;
                    if (transa == mblas_trans) {
                        indexA = i + lda * row;
                    }
                    if (transb == mblas_trans) {
                        indexB = col + ldb * i;
                    }
                    dot += A[indexA] * B[indexB];
                }
                dot *= alpha;
                C[row + col * ldc] = beta * C[row + col * ldc] + dot;
            }
        }
    }
}

void test_double(enum mblas_trans_type transa, enum mblas_trans_type transb, const int m, const int n, const int k, mpfr_t alpha, mpfr_t *A, const int lda, mpfr_t *B, const int ldb, mpfr_t beta, mpfr_t *C, const int ldc, const int repeats) {
    InitCpuTimer();
    Logger::printDash();
    PrintTimerName("[CPU] double gemm");

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

    //Launch
    for(int i = 0; i < repeats; i ++){
        for (int j = 0; j < ldc * n; j++) {
            dC[j] = mpfr_get_d(C[j], MPFR_RNDN);
        }
        StartCpuTimer();
        double_gemm(transa, transb, m, n, k, dalpha, dA, lda, dB, ldb, dbeta, dC, ldc);
        EndCpuTimer();
    }
    PrintAndResetCpuTimer("took");
    print_double_sum(dC, ldc * n);
    delete [] dA;
    delete [] dB;
    delete [] dC;
}


/////////
// CUDA
/////////
__global__ static void double_gemm_kernel(enum mblas_trans_type transa, enum mblas_trans_type transb, const int m, const int n, const int k, double alpha, double *A, const int lda, double *B, const int ldb, double beta, double *C, const int ldc) {
    auto row = blockIdx.x * blockDim.x + threadIdx.x;
    auto col = blockIdx.y * blockDim.y + threadIdx.y;
    while (col < n && row < m) {
        double dot = 0;
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
        dot *= alpha;
        C[row + col * ldc] = beta * C[row + col * ldc] + dot;
        row += gridDim.x * blockDim.x;
        col += gridDim.y * blockDim.y;
    }
}

void test_double_cuda(enum mblas_trans_type transa, enum mblas_trans_type transb, const int m, const int n, const int k, mpfr_t alpha, mpfr_t *A, const int lda, mpfr_t *B, const int ldb, mpfr_t beta, mpfr_t *C, const int ldc, const int repeats) {
    InitCudaTimer();
    Logger::printDash();
    PrintTimerName("[GPU] double gemm");

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
    double *hA = new double[sizeA];
    double *hB = new double[sizeB];
    double *hC = new double[ldc * n];

    //GPU data
    double *dA;
    double *dB;
    double *dC;
    double dalpha = mpfr_get_d(alpha, MPFR_RNDN);
    double dbeta = mpfr_get_d(beta, MPFR_RNDN);

    cudaMalloc(&dA, sizeof(double) * sizeA);
    cudaMalloc(&dB, sizeof(double) * sizeB);
    cudaMalloc(&dC, sizeof(double) * ldc * n);
    convert_vector(hA, A, sizeA);
    convert_vector(hB, B, sizeB);
    convert_vector(hC, C, ldc * n);

    cudaMemcpy(dA, hA, sizeA * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, sizeB * sizeof(double), cudaMemcpyHostToDevice);
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    for (int i = 0; i < repeats; i++) {
        cudaMemcpy(dC, hC, ldc * n * sizeof(double), cudaMemcpyHostToDevice);
        StartCudaTimer();
        double_gemm_kernel<<<dimGrid, dimBlock>>>(transa, transb, m, n, k, dalpha, dA, lda, dB, ldb, dbeta, dC, ldc);
        EndCudaTimer();
    }
    PrintAndResetCudaTimer("took");
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();
    //Copying to the host
    cudaMemcpy(hC, dC, ldc * n * sizeof(double), cudaMemcpyDeviceToHost);
    print_double_sum(hC, ldc * n);
    delete[] hA;
    delete[] hB;
    delete[] hC;
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
}

#endif //TEST_DOUBLE_GEMM_CUH
