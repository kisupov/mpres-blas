/*
 *  Performance test for double precision symmetric rank-1 update (SYR)
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
#ifndef TEST_DOUBLE_SYR_CUH
#define TEST_DOUBLE_SYR_CUH

#include "logger.cuh"
#include "timers.cuh"
#include "tsthelper.cuh"

/////////
// CPU
/////////
void double_syr(enum mblas_uplo_type uplo, const int n, double alpha, double *x, const int incx, double *A, const int lda) {
    if (uplo == mblas_upper) { //Access the upper part of the matrix
        #pragma omp parallel shared(n, A, x)
        {
            #pragma omp for
            for (int col = 0; col < n; col++) {
                for (int row = 0; row <= col; row++) {
                    auto ir = incx > 0 ? row * incx : (-n + row + 1) * incx;
                    auto ic = incx > 0 ? col * incx : (-n + col + 1) * incx;
                    A[row + col * lda] += alpha * x[ir] * x[ic];
                }
            }
        }
    } else { //Access the upper part of the matrix
        #pragma omp parallel shared(n, A, x)
        {
            #pragma omp for
            for (int col = 0; col < n; col++) {
                for (int row = col; row < n; row++) {
                    auto ir = incx > 0 ? row * incx : (-n + row + 1) * incx;
                    auto ic = incx > 0 ? col * incx : (-n + col + 1) * incx;
                    A[row + col * lda] += alpha * x[ir] * x[ic];
                }
            }
        }
    }
}

void test_double(enum mblas_uplo_type uplo, const int n, mpfr_t alpha, mpfr_t *x, const int incx, mpfr_t *A, const int lda, const int repeats) {
    InitCpuTimer();
    Logger::printDash();
    PrintTimerName("[CPU] double syr");
    //Actual length of the vector
    int lenx = (1 + (n- 1) * abs(incx));
    //CPU data
    double *dx = new double[lenx];
    double *dr = new double[lda * n];
    double *dA = new double[lda * n];
    double dalpha = mpfr_get_d(alpha, MPFR_RNDN);
    for (int i = 0; i < lenx; ++i) {
        dx[i] = mpfr_get_d(x[i], MPFR_RNDN);
    }
    for (int i = 0; i < lda * n; ++i) {
        dA[i] = mpfr_get_d(A[i], MPFR_RNDN);
    }
    //Launch
    for (int i = 0; i < repeats; i++) {
        for (int j = 0; j < lda * n; j++) {
            dr[j] = dA[j];
        }
        StartCpuTimer();
        double_syr(uplo, n, dalpha, dx, incx, dr, lda);
        EndCpuTimer();
    }
    PrintAndResetCpuTimer("took");
    print_double_sum(dr, lda * n);
    delete[] dx;
    delete[] dr;
    delete[] dA;
}

/////////
// CUDA
/////////
__global__ static void double_syr_kernel(enum mblas_uplo_type uplo, const int n, double alpha, double *x, const int incx, double *A, const int lda) {
    auto row = blockIdx.x * blockDim.x + threadIdx.x;
    auto col = blockIdx.y * blockDim.y + threadIdx.y;
    if (uplo == mblas_upper) { //Access the upper part of the matrix
        while (col < n && row <= col) {
            auto ir = incx > 0 ? row * incx : (-n + row + 1) * incx;
            auto ic = incx > 0 ? col * incx : (-n + col + 1) * incx;
            A[row + col * lda] += alpha * x[ir] * x[ic];
            row += gridDim.x * blockDim.x;
            col += gridDim.y * blockDim.y;
        }
    } else { //Access the lower part of the matrix
        while (row < n && col <= row) {
            auto ir = incx > 0 ? row * incx : (-n + row + 1) * incx;
            auto ic = incx > 0 ? col * incx : (-n + col + 1) * incx;
            A[row + col * lda] += alpha * x[ir] * x[ic];
            row += gridDim.x * blockDim.x;
            col += gridDim.y * blockDim.y;
        }
    }
}

void test_double_cuda(enum mblas_uplo_type uplo, const int n, mpfr_t alpha, mpfr_t *x, const int incx, mpfr_t *A, const int lda, const int repeats) {
    InitCudaTimer();
    Logger::printDash();
    PrintTimerName("[GPU] double syr");
    //Actual length of the vector
    int lenx = (1 + (n - 1) * abs(incx));
    //Execution configuration
    int threadsX = 16;
    int threadsY = 16;
    dim3 dimBlock(threadsX, threadsY);
    int blocksX = (n + dimBlock.x - 1) / dimBlock.x;
    int blocksY = (n + dimBlock.y - 1) / dimBlock.y;
    dim3 dimGrid(blocksX, blocksY);
    printf("\tExec. config: threads.x = %i, threads.y = %i, blocks.x = %i, blocks.y = %i\n", threadsX, threadsY, blocksX, blocksY);
    //Host data
    double *hx = new double[lenx];
    double *hA = new double[lda * n];
    //GPU data
    double *dx;
    double *dA;
    double dalpha = mpfr_get_d(alpha, MPFR_RNDN);
    cudaMalloc(&dx, sizeof(double) * lenx);
    cudaMalloc(&dA, sizeof(double) * lda * n);
    convert_vector(hx, x, lenx);
    convert_vector(hA, A, lda * n);
    cudaMemcpy(dx, hx, sizeof(double) * lenx, cudaMemcpyHostToDevice);
    for (int i = 0; i < repeats; i++) {
        cudaMemcpy(dA, hA, sizeof(double) * lda * n, cudaMemcpyHostToDevice);
        StartCudaTimer();
        double_syr_kernel<<<dimGrid, dimBlock>>>(uplo, n, dalpha, dx, incx, dA, lda);
        EndCudaTimer();
    }
    PrintAndResetCudaTimer("took");
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();
    //Copying to the host
    cudaMemcpy(hA, dA, lda * n * sizeof(double), cudaMemcpyDeviceToHost);
    print_double_sum(hA, lda * n);
    /*for (int i = 0; i < n; i++) {
        printf("\n");
        for (int j = 0; j < n; j++) {
            printf("%.5f\t", hA[i + j * lda]);
        }
    }
    printf("\n");*/
    delete[] hx;
    delete[] hA;
    cudaFree(dx);
    cudaFree(dA);
}

#endif //TEST_DOUBLE_SYR_CUH
