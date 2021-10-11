/*
 *  Performance test for mp_add and mp_mul_d
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
#ifndef TEST_MP_PEAK_PERFORMANCE_CUH
#define TEST_MP_PEAK_PERFORMANCE_CUH

#include "logger.cuh"
#include "timers.cuh"
#include "tsthelper.cuh"
#include "arith/muld.cuh"

static __global__ void arith_kernel(const int n, mp_float_ptr x, mp_float_ptr y, mp_float_ptr z, mp_float_ptr w) {
    auto i = threadIdx.x + blockIdx.x * blockDim.x;
    while (i < n) {
        cuda::mp_add(&z[i], x[i], x[i]);
        cuda::mp_mul_d(&w[i], y[i], 3.14159268);
        i += gridDim.x * blockDim.x;
    }
}

static __global__ void copy_kernel(const int n, mp_float_ptr x, mp_float_ptr y, mp_float_ptr z, mp_float_ptr w) {
    auto i = threadIdx.x + blockIdx.x * blockDim.x;
    while (i < n) {
        cuda::mp_set(&z[i], x[i]);
        cuda::mp_set(&w[i], y[i]);
        i += gridDim.x * blockDim.x;
    }
}

static double launchArithKernel(const int blocks, const int threads, const int n, mp_float_ptr dx, mp_float_ptr dy, mp_float_ptr dz, mp_float_ptr dw){
    InitCudaTimer();
    StartCudaTimer();
    arith_kernel<<<blocks, threads>>>(n, dx, dy, dz, dw);
    EndCudaTimer();
    double milliseconds = _cuda_time;
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();
    return milliseconds;
}

static double launchCopyKernel(const int blocks, const int threads, const int n, mp_float_ptr dx, mp_float_ptr dy, mp_float_ptr dz, mp_float_ptr dw){
    InitCudaTimer();
    StartCudaTimer();
    copy_kernel<<<blocks, threads>>>(n, dx, dy, dz, dw);
    EndCudaTimer();
    double milliseconds = _cuda_time;
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();
    return milliseconds;
}

static void runKernels(const int blocks, const int threads, const int n, mp_float_ptr dx, mp_float_ptr dy, mp_float_ptr dz, mp_float_ptr dw){
    printf("\nExec. config: blocks = %i, threads = %i\n", blocks, threads);
    double copyTime = launchCopyKernel(blocks, threads, n, dx, dy, dz, dw);
    double arithTime = launchArithKernel(blocks, threads, n, dx, dy, dz, dw);
    std::cout << "- took copy: " << copyTime << std::endl;
    std::cout << "- took arith: " << arithTime << std::endl;
    std::cout << "- bandwidth copy (GB/s): " << sizeof(mp_float_t) * n * 4 / copyTime / 1e6 << std::endl;
    double performance = (double)n * 2 / (arithTime - copyTime) / 1e6;
    if(performance > 0){
        std::cout<< "- peak perf. (mp-flop/s x 10^9): " << performance << std::endl;
    }
}

void test_mp_peak_performance(const int n, const int prec) {
    Logger::printDash();
    PrintTimerName("[GPU] MPRES-BLAS addition and by-double-multiplication peak performance");
    Logger::printDash();

    //Inputs
    mpfr_t *vectorX = create_random_array(n, prec, -10000, 10000);
    //Host data
    auto hx = new mp_float_t[n];
    // GPU data
    mp_float_ptr dx;
    mp_float_ptr dy;
    mp_float_ptr dz;
    mp_float_ptr dw;
    cudaMalloc(&dx, sizeof(mp_float_t) * n);
    cudaMalloc(&dy, sizeof(mp_float_t) * n);
    cudaMalloc(&dz, sizeof(mp_float_t) * n);
    cudaMalloc(&dw, sizeof(mp_float_t) * n);
    // Convert from MPFR
    convert_vector(hx, vectorX, n);
    //Copying to the GPU
    cudaMemcpy(dx, hx, sizeof(mp_float_t) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(dy, hx, sizeof(mp_float_t) * n, cudaMemcpyHostToDevice);
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();
    //Launch
    int blocks = 128;
    for(int k = 0; k < 5; k++){
        int threads = 32;
        for(int j = 0; j < 4; j++){
            runKernels(blocks, threads, n, dx, dy, dz, dw);
            threads *= 2;
        }
        blocks *= 2;
    }

    int threads = 32;
    blocks = n / threads;
    runKernels(blocks, threads, n, dx, dy, dz, dw);

    threads = 64;
    blocks = n / threads;
    runKernels(blocks, threads, n, dx, dy, dz, dw);

    threads = 128;
    blocks = n / threads;
    runKernels(blocks, threads, n, dx, dy, dz, dw);

    threads = 256;
    blocks = n / threads;
    runKernels(blocks, threads, n, dx, dy, dz, dw);

    //Cleanup
    for (int i = 0; i < n; i++) {
        mpfr_clear(vectorX[i]);
    }
    delete[] vectorX;
    delete [] hx;
    cudaFree(dx);
    cudaFree(dy);
    cudaFree(dz);
    cudaFree(dw);
}


#endif //TEST_MP_PEAK_PERFORMANCE_CUH
