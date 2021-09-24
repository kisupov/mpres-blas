/*
 *  Performance test for MPRES AXPY
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
#ifndef TEST_MPRES_AXPY_CUH
#define TEST_MPRES_AXPY_CUH

#include "logger.cuh"
#include "timers.cuh"
#include "tsthelper.cuh"
#include "blas/v2/axpy_v2.cuh"

void test_mpres(const int n, mpfr_t alpha, mpfr_t *x, mpfr_t *y, const int repeats) {
    InitCudaTimer();
    Logger::printDash();
    PrintTimerName("[GPU] MPRES-BLAS axpy");
    //Execution configuration
    int threads = 32;
    int blocks = n / threads + 1;
    printf("\tExec. config: blocks = %i, threads = %i\n", blocks, threads);

    //Host data
    mp_float_ptr hx = new mp_float_t[n];
    mp_float_ptr hy = new mp_float_t[n];
    mp_float_t halpha;

    // GPU data
    mp_float_ptr dx;
    mp_float_ptr dy;
    mp_float_ptr dalpha;

    cudaMalloc(&dx, sizeof(mp_float_t) * n);
    cudaMalloc(&dy, sizeof(mp_float_t) * n);
    cudaMalloc(&dalpha, sizeof(mp_float_t));

    // Convert from MPFR
    convert_vector(hx, x, n);
    convert_vector(hy, y, n);
    mp_set_mpfr(&halpha, alpha);

    //Copying to the GPU
    cudaMemcpy(dx, hx, sizeof(mp_float_t) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(dalpha, &halpha, sizeof(mp_float_t), cudaMemcpyHostToDevice);

    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    for(int i = 0; i < repeats; i ++) {
        cudaMemcpy(dy, hy, sizeof(mp_float_t) * n, cudaMemcpyHostToDevice);
        StartCudaTimer();
        cuda::mp_axpy<<<blocks, threads>>>(n, dalpha, dx, dy, dy);
        EndCudaTimer();
    }
    PrintCudaTimer("took");
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    //Copying to the host
    cudaMemcpy(hy, dy, n * sizeof(mp_float_t), cudaMemcpyDeviceToHost);
    print_mp_sum(hy, n);

    //Cleanup
    delete [] hx;
    delete [] hy;
    cudaFree(dx);
    cudaFree(dy);
    cudaFree(dalpha);
}

#endif //TEST_MPRES_AXPY_CUH
