/*
 *  Performance test for MPRES symmetric rank-1 update (SYR)
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
#ifndef TEST_MPRES_SYR_CUH
#define TEST_MPRES_SYR_CUH

#include "logger.cuh"
#include "timers.cuh"
#include "tsthelper.cuh"
#include "blas/v2/syr_v2.cuh"

void test_mpres_syr(enum mblas_uplo_type uplo, const int n, mpfr_t alpha, mpfr_t *x, const int incx, mpfr_t *A, const int lda, const int repeats) {
    InitCudaTimer();
    Logger::printDash();
    PrintTimerName("[GPU] MPRES-BLAS syr");

    //Actual length of the vector x
    int lenx = (1 + (n - 1) * abs(incx));

    //Execution configuration
    auto threadsX = 32;
    auto threadsY = 1;
    dim3 dimBlock(threadsX, threadsY);
    auto blocksX = (n + dimBlock.x - 1) / dimBlock.x;
    auto blocksY = (n + dimBlock.y - 1) / dimBlock.y;
    dim3 dimGrid(blocksX, blocksY);
    Logger::printKernelExecutionConfig2D(threadsX, threadsY, blocksX, blocksY);

    //Host data
    mp_float_ptr hx = new mp_float_t[lenx];
    mp_float_ptr hA = new mp_float_t[lda * n];
    mp_float_t halpha;

    // GPU data
    mp_float_ptr dx;
    mp_float_ptr dA;
    mp_float_ptr dalpha;

    cudaMalloc(&dx, sizeof(mp_float_t) * lenx);
    cudaMalloc(&dA, sizeof(mp_float_t) * lda * n);
    cudaMalloc(&dalpha, sizeof(mp_float_t));

    // Convert from MPFR
    convert_vector(hx, x, lenx);
    convert_matrix(hA, A, lda, n);
    mp_set_mpfr(&halpha, alpha);

    //Copying to the GPU
    cudaMemcpy(dx, hx, sizeof(mp_float_t) * lenx, cudaMemcpyHostToDevice);
    cudaMemcpy(dalpha, &halpha, sizeof(mp_float_t), cudaMemcpyHostToDevice);
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    for(int i = 0; i < repeats; i ++) {
        cudaMemcpy(dA, hA, lda * n * sizeof(mp_float_t), cudaMemcpyHostToDevice);
        StartCudaTimer();
        cuda::mp_syr<<<dimGrid, dimBlock>>>(uplo, n, dalpha, dx, incx, dA, lda);
        EndCudaTimer();
    }
    PrintAndResetCudaTimer("took");
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    //Copying to the host
    cudaMemcpy(hA, dA, lda * n * sizeof(mp_float_t), cudaMemcpyDeviceToHost);
    print_mp_sum(hA, lda * n);

    //Cleanup
    delete [] hx;
    delete [] hA;
    cudaFree(dx);
    cudaFree(dA);
    cudaFree(dalpha);
}

#endif //TEST_MPRES_SYR_CUH
