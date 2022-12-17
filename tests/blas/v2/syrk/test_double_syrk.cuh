/*
 *  Performance test for double precision symmetric rank-k update (SYRK)
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
#ifndef TEST_DOUBLE_SYRK_CUH
#define TEST_DOUBLE_SYRK_CUH

#include "logger.cuh"
#include "timers.cuh"
#include "tsthelper.cuh"

/////////
// CPU
/////////

void double_syrk(enum mblas_uplo_type uplo, enum mblas_trans_type trans, const int n, const int k, double alpha, double *A, const int lda, double beta, double *C, const int ldc) {
    #pragma omp parallel shared(n, k, lda, ldc, A, C)
    {
        #pragma omp for
        for (int col = 0; col < n; col++) {
            for (int row = ((uplo == mblas_upper) ? 0 : col); row <= ((uplo == mblas_upper) ? col : n - 1); row++) {
                double dot = 0;
                for (int i = 0; i < k; i++) {
                    auto indexA = row + lda * i;
                    auto indexAT = col + lda * i;
                    if (trans == mblas_trans) {
                        indexA = i + lda * row;
                        indexAT = i + lda * col;
                    }
                    dot += A[indexA] * A[indexAT];
                }
                dot *= alpha;
                C[row + col * ldc] = beta * C[row + col * ldc] + dot;
            }
        }
    }
}

void test_double(enum mblas_uplo_type uplo, enum mblas_trans_type trans, const int n, const int k, mpfr_t alpha, mpfr_t *A, const int lda, mpfr_t beta, mpfr_t *C, const int ldc, const int repeats) {
    InitCpuTimer();
    Logger::printDash();
    PrintTimerName("[CPU] double syrk");

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
    for (int j = 0; j < ldc * n; j++) {
        dC[j] = mpfr_get_d(C[j], MPFR_RNDN);
    }

    //Launch
    for(int i = 0; i < repeats; i ++){
        for (int j = 0; j < ldc * n; j++) {
            result[j] = dC[j];
        }
        StartCpuTimer();
        double_syrk(uplo, trans, n, k, dalpha, dA, lda, dbeta, result, ldc);
        EndCpuTimer();
    }
    PrintAndResetCpuTimer("took");
    print_double_sum(result, ldc * n);
    delete [] dA;
    delete [] dC;
    delete [] result;
}


/////////
// CUDA
/////////
__global__ static void double_syrk_kernel(enum mblas_uplo_type uplo, enum mblas_trans_type trans, const int n, const int k, double alpha, double *A, const int lda, double beta, double *C, const int ldc) {
    auto row = blockIdx.x * blockDim.x + threadIdx.x;
    auto col = blockIdx.y * blockDim.y + threadIdx.y;
    while ((uplo == mblas_upper && col < n && row <= col) || (uplo == mblas_lower && row < n && col <= row)) {
        double dot = 0.0;
        for (int i = 0; i < k; i++) {
            auto indexA = row + lda * i;
            auto indexAT = col + lda * i;
            if (trans == mblas_trans) {
                indexA = i + lda * row;
                indexAT = i + lda * col;
            }
            dot = dot + A[indexA] * A[indexAT];
        }
        dot *= alpha;
        C[row + col * ldc] = beta * C[row + col * ldc] + dot;
        row += gridDim.x * blockDim.x;
        col += gridDim.y * blockDim.y;
    }
}

void test_double_cuda(enum mblas_uplo_type uplo, enum mblas_trans_type trans, const int n, const int k, mpfr_t alpha, mpfr_t *A, const int lda, mpfr_t beta, mpfr_t *C, const int ldc, const int repeats) {
    InitCudaTimer();
    Logger::printDash();
    PrintTimerName("[GPU] double syrk");

    //Execution configuration
    auto threadsX = 16;
    auto threadsY = 16;
    dim3 dimBlock(threadsX, threadsY);
    auto blocksX = (n + dimBlock.x - 1) / dimBlock.x;
    auto blocksY = (n + dimBlock.y - 1) / dimBlock.y;
    dim3 dimGrid(blocksX, blocksY);
    Logger::printKernelExecutionConfig2D(threadsX, threadsY, blocksX, blocksY);

    //Size of an array storing matrix A
    auto sizeA = (trans == mblas_trans) ? lda * n : lda * k;

    //Host data
    double *hA = new double[sizeA];
    double *hC = new double[ldc * n];

    //GPU data
    double *dA;
    double *dC;
    double dalpha = mpfr_get_d(alpha, MPFR_RNDN);
    double dbeta = mpfr_get_d(beta, MPFR_RNDN);

    cudaMalloc(&dA, sizeof(double) * sizeA);
    cudaMalloc(&dC, sizeof(double) * ldc * n);
    convert_vector(hA, A, sizeA);
    convert_vector(hC, C, ldc * n);

    cudaMemcpy(dA, hA, sizeA * sizeof(double), cudaMemcpyHostToDevice);
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    for (int i = 0; i < repeats; i++) {
        cudaMemcpy(dC, hC, ldc * n * sizeof(double), cudaMemcpyHostToDevice);
        StartCudaTimer();
        double_syrk_kernel<<<dimGrid, dimBlock>>>(uplo, trans, n, k, dalpha, dA, lda, dbeta, dC, ldc);
        EndCudaTimer();
    }
    PrintAndResetCudaTimer("took");
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();
    //Copying to the host
    cudaMemcpy(hC, dC, ldc * n * sizeof(double), cudaMemcpyDeviceToHost);
    print_double_sum(hC, ldc * n);
    delete[] hA;
    delete[] hC;
    cudaFree(dA);
    cudaFree(dC);
}

#endif //TEST_DOUBLE_SYRK_CUH
