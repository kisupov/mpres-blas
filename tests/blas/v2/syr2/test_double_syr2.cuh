/*
 *  Performance test for double precision symmetric rank-2 update (SYR2)
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
#ifndef TEST_DOUBLE_SYR2_CUH
#define TEST_DOUBLE_SYR2_CUH

#include "logger.cuh"
#include "timers.cuh"
#include "tsthelper.cuh"

/////////
// CPU
/////////
void double_syr2(enum mblas_uplo_type uplo, const int n, double alpha, double *x, const int incx, double *y, const int incy, double *A, const int lda) {
    if (uplo == mblas_upper) { //Access the upper part of the matrix
        #pragma omp parallel shared(n, A, x, y)
        {
            #pragma omp for
            for (int col = 0; col < n; col++) {
                for (int row = 0; row <= col; row++) {
                    auto irx = incx > 0 ? row * incx : (-n + row + 1) * incx;
                    auto icx = incx > 0 ? col * incx : (-n + col + 1) * incx;
                    auto iry = incy > 0 ? row * incy : (-n + row + 1) * incy;
                    auto icy = incy > 0 ? col * incy : (-n + col + 1) * incy;
                    A[row + col * lda] = A[row + col * lda] + alpha * x[irx] * y[icy] + alpha * y[iry] * x[icx];
                }
            }
        }
    } else { //Access the upper part of the matrix
        #pragma omp parallel shared(n, A, x, y)
        {
            #pragma omp for
            for (int col = 0; col < n; col++) {
                for (int row = col; row < n; row++) {
                    auto irx = incx > 0 ? row * incx : (-n + row + 1) * incx;
                    auto icx = incx > 0 ? col * incx : (-n + col + 1) * incx;
                    auto iry = incy > 0 ? row * incy : (-n + row + 1) * incy;
                    auto icy = incy > 0 ? col * incy : (-n + col + 1) * incy;
                    A[row + col * lda] = A[row + col * lda] + alpha * x[irx] * y[icy] + alpha * y[iry] * x[icx];
                }
            }
        }
    }
}

void test_double(enum mblas_uplo_type uplo, const int n, mpfr_t alpha, mpfr_t *x, const int incx, mpfr_t *y, const int incy, mpfr_t *A, const int lda, const int repeats) {
    InitCpuTimer();
    Logger::printDash();
    PrintTimerName("[CPU] double syr2");
    //Actual length of the vectors
    int lenx = (1 + (n- 1) * abs(incx));
    int leny = (1 + (n- 1) * abs(incy));
    //CPU data
    double *dx = new double[lenx];
    double *dy = new double[leny];
    double *dr = new double[lda * n];
    double *dA = new double[lda * n];
    double dalpha = mpfr_get_d(alpha, MPFR_RNDN);
    for (int i = 0; i < lenx; ++i) {
        dx[i] = mpfr_get_d(x[i], MPFR_RNDN);
    }
    for (int i = 0; i < leny; ++i) {
        dy[i] = mpfr_get_d(y[i], MPFR_RNDN);
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
        double_syr2(uplo, n, dalpha, dx, incx, dy, incy, dr, lda);
        EndCpuTimer();
    }
    PrintAndResetCpuTimer("took");
    print_double_sum(dr, lda * n);
    delete[] dx;
    delete[] dy;
    delete[] dr;
    delete[] dA;
}

/////////
// CUDA
/////////
__global__ static void double_syr2_kernel(enum mblas_uplo_type uplo, const int n, double alpha, double *x, const int incx, double *y, const int incy, double *A, const int lda) {
    auto row = blockIdx.x * blockDim.x + threadIdx.x;
    auto col = blockIdx.y * blockDim.y + threadIdx.y;
    if (uplo == mblas_upper) { //Access the upper part of the matrix
        while (col < n && row <= col) {
            auto irx = incx > 0 ? row * incx : (-n + row + 1) * incx;
            auto icx = incx > 0 ? col * incx : (-n + col + 1) * incx;
            auto iry = incy > 0 ? row * incy : (-n + row + 1) * incy;
            auto icy = incy > 0 ? col * incy : (-n + col + 1) * incy;
            A[row + col * lda] = A[row + col * lda] + alpha * x[irx] * y[icy] + alpha * y[iry] * x[icx];
            row += gridDim.x * blockDim.x;
            col += gridDim.y * blockDim.y;
        }
    } else { //Access the lower part of the matrix
        while (row < n && col <= row) {
            auto irx = incx > 0 ? row * incx : (-n + row + 1) * incx;
            auto icx = incx > 0 ? col * incx : (-n + col + 1) * incx;
            auto iry = incy > 0 ? row * incy : (-n + row + 1) * incy;
            auto icy = incy > 0 ? col * incy : (-n + col + 1) * incy;
            A[row + col * lda] = A[row + col * lda] + alpha * x[irx] * y[icy] + alpha * y[iry] * x[icx];
            row += gridDim.x * blockDim.x;
            col += gridDim.y * blockDim.y;
        }
    }
}

void test_double_cuda(enum mblas_uplo_type uplo, const int n, mpfr_t alpha, mpfr_t *x, const int incx, mpfr_t *y, const int incy, mpfr_t *A, const int lda, const int repeats) {
    InitCudaTimer();
    Logger::printDash();
    PrintTimerName("[GPU] double syr2");
    //Actual length of the vectors
    int lenx = (1 + (n - 1) * abs(incx));
    int leny = (1 + (n - 1) * abs(incy));
    //Execution configuration
    auto threadsX = 16;
    auto threadsY = 16;
    dim3 dimBlock(threadsX, threadsY);
    auto blocksX = (n + dimBlock.x - 1) / dimBlock.x;
    auto blocksY = (n + dimBlock.y - 1) / dimBlock.y;
    dim3 dimGrid(blocksX, blocksY);
    Logger::printKernelExecutionConfig2D(threadsX, threadsY, blocksX, blocksY);
    //Host data
    double *hx = new double[lenx];
    double *hy = new double[leny];
    double *hA = new double[lda * n];
    //GPU data
    double *dx;
    double *dy;
    double *dA;
    double dalpha = mpfr_get_d(alpha, MPFR_RNDN);
    cudaMalloc(&dx, sizeof(double) * lenx);
    cudaMalloc(&dy, sizeof(double) * leny);
    cudaMalloc(&dA, sizeof(double) * lda * n);
    convert_vector(hx, x, lenx);
    convert_vector(hy, y, leny);
    convert_vector(hA, A, lda * n);
    cudaMemcpy(dx, hx, sizeof(double) * lenx, cudaMemcpyHostToDevice);
    cudaMemcpy(dy, hy, sizeof(double) * leny, cudaMemcpyHostToDevice);
    for (int i = 0; i < repeats; i++) {
        cudaMemcpy(dA, hA, sizeof(double) * lda * n, cudaMemcpyHostToDevice);
        StartCudaTimer();
        double_syr2_kernel<<<dimGrid, dimBlock>>>(uplo, n, dalpha, dx, incx, dy, incy, dA, lda);
        EndCudaTimer();
    }
    PrintAndResetCudaTimer("took");
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();
    //Copying to the host
    cudaMemcpy(hA, dA, lda * n * sizeof(double), cudaMemcpyDeviceToHost);
    print_double_sum(hA, lda * n);
    delete[] hx;
    delete[] hy;
    delete[] hA;
    cudaFree(dx);
    cudaFree(dy);
    cudaFree(dA);
}

#endif //TEST_DOUBLE_SYR2_CUH
