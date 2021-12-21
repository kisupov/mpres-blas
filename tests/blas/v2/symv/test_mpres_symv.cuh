/*
 *  Performance test for MPRES SYMV
 *
 *  Copyright 2021 by Konstantin Isupov.
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
#ifndef TEST_MPRES_SYMV_CUH
#define TEST_MPRES_SYMV_CUH

#include "logger.cuh"
#include "timers.cuh"
#include "tsthelper.cuh"
#include "blas/v2/symv_v2.cuh"

void test_mpres_symv(enum mblas_uplo_type uplo, const int n, int lenx, int leny, mpfr_t alpha, mpfr_t *A, int lda, mpfr_t *x, int incx, mpfr_t beta, mpfr_t *y, int incy, const int repeats) {
    InitCudaTimer();
    Logger::printDash();
    PrintTimerName("[GPU] MPRES-BLAS symv");

    //Execution configuration
    int threads = 32;
    int blocks = n / threads + 1;
    printf("\tExec. config: blocks = %i, threads = %i\n", blocks, threads);

    //Host data
    mp_float_ptr hx = new mp_float_t[lenx];
    mp_float_ptr hy = new mp_float_t[leny];
    mp_float_ptr hA = new mp_float_t[lda * n];
    mp_float_t halpha;
    mp_float_t hbeta;

    // GPU data
    mp_float_ptr dA;
    mp_float_ptr dx;
    mp_float_ptr dy;
    mp_float_ptr dalpha;
    mp_float_ptr dbeta;

    cudaMalloc(&dx, sizeof(mp_float_t) * lenx);
    cudaMalloc(&dy, sizeof(mp_float_t) * leny);
    cudaMalloc(&dA, sizeof(mp_float_t) * lda * n);
    cudaMalloc(&dalpha, sizeof(mp_float_t));
    cudaMalloc(&dbeta, sizeof(mp_float_t));

    // Convert from MPFR
    convert_vector(hx, x, lenx);
    convert_vector(hy, y, leny);
    convert_matrix(hA, A, lda, n);
    mp_set_mpfr(&halpha, alpha);
    mp_set_mpfr(&hbeta, beta);

    //Copying to the GPU
    cudaMemcpy(dx, hx, sizeof(mp_float_t) * lenx, cudaMemcpyHostToDevice);
    cudaMemcpy(dA, hA, lda * n * sizeof(mp_float_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dalpha, &halpha, sizeof(mp_float_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dbeta, &hbeta, sizeof(mp_float_t), cudaMemcpyHostToDevice);

    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    for(int i = 0; i < repeats; i ++) {
        cudaMemcpy(dy, hy, sizeof(mp_float_t) * leny, cudaMemcpyHostToDevice);
        StartCudaTimer();
        cuda::mp_symv<32><<<blocks, threads>>>(uplo, n, dalpha, dA, lda, dx, incx, dbeta, dy, incy);
        EndCudaTimer();
    }
    PrintAndResetCudaTimer("took");
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    //Copying to the host
    cudaMemcpy(hy, dy, leny * sizeof(mp_float_t), cudaMemcpyDeviceToHost);
    print_mp_sum(hy, leny);

    //Cleanup
    delete [] hx;
    delete [] hy;
    delete [] hA;
    cudaFree(dx);
    cudaFree(dy);
    cudaFree(dA);
    cudaFree(dalpha);
    cudaFree(dbeta);
}

#endif //TEST_MPRES_SYMV_CUH
