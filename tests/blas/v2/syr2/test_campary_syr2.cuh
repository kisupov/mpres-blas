/*
 *  Performance test for CAMPARY symmetric rank-2 update (SYR2)
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
#ifndef TEST_CAMPARY_SYR2_CUH
#define TEST_CAMPARY_SYR2_CUH

#include "logger.cuh"
#include "timers.cuh"
#include "tsthelper.cuh"
#include "lib/campary_common.cuh"
#include "blas/mblas_enum.cuh"

template<int prec>
__global__ static void campary_syr2_kernel(enum mblas_uplo_type uplo, int n, multi_prec<prec> *alpha, multi_prec<prec> *x, const int incx, multi_prec<prec> *y, const int incy, multi_prec<prec> *A, int lda) {
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

template<int prec>
void test_campary_syr2(enum mblas_uplo_type uplo, const int n, mpfr_t alpha, mpfr_t *x, const int incx, mpfr_t *y, const int incy, mpfr_t *A, const int lda, const int convert_prec, const int repeats) {
    InitCudaTimer();
    Logger::printDash();
    PrintTimerName("[GPU] CAMPARY syr2");
    //Actual length of the vectors
    int lenx = (1 + (n - 1) * abs(incx));
    int leny = (1 + (n - 1) * abs(incy));

    //Execution configuration
    auto threadsX = 32;
    auto threadsY = 1;
    dim3 dimBlock(threadsX, threadsY);
    auto blocksX = (n + dimBlock.x - 1) / dimBlock.x;
    auto blocksY = (n + dimBlock.y - 1) / dimBlock.y;
    dim3 dimGrid(blocksX, blocksY);
    Logger::printKernelExecutionConfig2D(threadsX, threadsY, blocksX, blocksY);

    //Host data
    multi_prec<prec> halpha;
    auto *hx = new multi_prec<prec>[lenx];
    auto *hy = new multi_prec<prec>[leny];
    auto *hA = new multi_prec<prec>[lda * n];

    // GPU data
    multi_prec<prec> *dalpha;
    multi_prec<prec> *dx;
    multi_prec<prec> *dy;
    multi_prec<prec> *dA;

    cudaMalloc(&dx, sizeof(multi_prec<prec>) * lenx);
    cudaMalloc(&dy, sizeof(multi_prec<prec>) * leny);
    cudaMalloc(&dA, sizeof(multi_prec<prec>) * lda * n);
    cudaMalloc(&dalpha, sizeof(multi_prec<prec>));

    //Convert from MPFR
    #pragma omp parallel for
    for (int i = 0; i < lenx; i++) {
        hx[i] = convert_to_string_sci(x[i], convert_prec).c_str();
    }
    #pragma omp parallel for
    for (int i = 0; i < leny; i++) {
        hy[i] = convert_to_string_sci(y[i], convert_prec).c_str();
    }
    #pragma omp parallel for
    for (int i = 0; i < lda * n; i++) {
        hA[i] = convert_to_string_sci(A[i], convert_prec).c_str();
    }
    halpha = convert_to_string_sci(alpha, convert_prec).c_str();

    //Copying to the GPU
    cudaMemcpy(dx, hx, sizeof(multi_prec<prec>) * lenx, cudaMemcpyHostToDevice);
    cudaMemcpy(dy, hy, sizeof(multi_prec<prec>) * leny, cudaMemcpyHostToDevice);
    cudaMemcpy(dalpha, &halpha, sizeof(multi_prec<prec>), cudaMemcpyHostToDevice);
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    for (int i = 0; i < repeats; i++) {
        cudaMemcpy(dA, hA, lda * n * sizeof(multi_prec<prec>), cudaMemcpyHostToDevice);
        StartCudaTimer();
        campary_syr2_kernel<<<dimGrid, dimBlock>>>(uplo, n, dalpha, dx, incx, dy, incy, dA, lda);
        EndCudaTimer();
    }
    PrintAndResetCudaTimer("took");
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    //Copying to the host
    cudaMemcpy(hA, dA, sizeof(multi_prec<prec>) * lda * n, cudaMemcpyDeviceToHost);
    for (int i = 1; i < lda * n; i++) {
        hA[0] += hA[i];
    }
    printResult<prec>(hA[0]);

    //Cleanup
    delete[] hx;
    delete[] hy;
    delete[] hA;
    cudaFree(dx);
    cudaFree(dy);
    cudaFree(dA);
    cudaFree(dalpha);
}

#endif //TEST_CAMPARY_SYR2_CUH
