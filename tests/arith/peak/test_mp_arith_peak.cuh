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

#define N 20000000
#define REPEAT 20

static __global__ void arith_kernel(mp_float_ptr x, mp_float_ptr y, mp_float_ptr z, mp_float_ptr w) {
    __shared__ mp_float_t operand;
    if(threadIdx.x == 0){
        operand = x[0];
    }
    __syncthreads();
    for(int j = 0; j < REPEAT; j++){
        auto i = threadIdx.x + blockIdx.x * blockDim.x;
        while (i < N) {
            cuda::mp_mul_d(&z[i], x[i], 3.14159268); //1 global memory read + 1 global memory write
            cuda::mp_add(&w[i], y[i], operand); //1 global memory read + 1 global memory write
            i += gridDim.x * blockDim.x;
        }
    }
}

static __global__ void copy_kernel(mp_float_ptr x, mp_float_ptr y, mp_float_ptr z, mp_float_ptr w) {
    for(int j = 0; j < REPEAT; j++) {
        auto i = threadIdx.x + blockIdx.x * blockDim.x;
        while (i < N) {
            cuda::mp_set(&z[i], x[i]); //1 global memory read + 1 global memory write
            cuda::mp_set(&w[i], y[i]); //1 global memory read + 1 global memory write
            i += gridDim.x * blockDim.x;
        }
    }
}

static double launchArithKernel(const int blocks, const int threads, mp_float_ptr dx, mp_float_ptr dy, mp_float_ptr dz, mp_float_ptr dw){
    InitCudaTimer();
    StartCudaTimer();
    arith_kernel<<<blocks, threads>>>(dx, dy, dz, dw);
    EndCudaTimer();
    double milliseconds = _cuda_time;
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();
    return milliseconds;
}

static double launchCopyKernel(const int blocks, const int threads, mp_float_ptr dx, mp_float_ptr dy, mp_float_ptr dz, mp_float_ptr dw){
    InitCudaTimer();
    StartCudaTimer();
    copy_kernel<<<blocks, threads>>>(dx, dy, dz, dw);
    EndCudaTimer();
    double milliseconds = _cuda_time;
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();
    return milliseconds;
}

static void runKernels(const int blocks, const int threads, mp_float_ptr dx, mp_float_ptr dy, mp_float_ptr dz, mp_float_ptr dw){
    double copyTime = launchCopyKernel(blocks, threads, dx, dy, dz, dw);
    double arithTime = launchArithKernel(blocks, threads, dx, dy, dz, dw);
    double avgCopyTime = copyTime / (double) REPEAT;
    double avgArithTime = arithTime / (double) REPEAT;
    double bandwidth = sizeof(mp_float_t) * N * 4 / avgCopyTime / 1e6;
    double performance = N * 2 / (avgArithTime - avgCopyTime) / 1e6;
    if(performance > 0){
        std::cout << std::endl;
        std::cout << "Exec. config: blocks = " << blocks << " threads = " << threads << std::endl;
        std::cout << "- took copy: " << copyTime << std::endl;
        std::cout << "- took arith: " << arithTime << std::endl;
        std::cout << "- bandwidth copy (GB/s): " << bandwidth << std::endl;
        std::cout << "- peak perf. (mp-flop/s x 10^9): " << performance << std::endl;
    }
}

void test_mp_peak_performance(const int prec) {
    Logger::printDash();
    PrintTimerName("[GPU] MPRES-BLAS addition and by-double-multiplication peak performance");
    Logger::printDash();

    //Inputs
    mpfr_t *vectorX = create_random_array(N, prec, 0, 1);
    //Host data
    auto hx = new mp_float_t[N];
    // GPU data
    mp_float_ptr dx;
    mp_float_ptr dy;
    mp_float_ptr dz;
    mp_float_ptr dw;
    cudaMalloc(&dx, sizeof(mp_float_t) * N);
    cudaMalloc(&dy, sizeof(mp_float_t) * N);
    cudaMalloc(&dz, sizeof(mp_float_t) * N);
    cudaMalloc(&dw, sizeof(mp_float_t) * N);
    // Convert from MPFR
    convert_vector(hx, vectorX, N);
    //Copying to the GPU
    cudaMemcpy(dx, hx, sizeof(mp_float_t) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(dy, hx, sizeof(mp_float_t) * N, cudaMemcpyHostToDevice);
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();
    //Launch
    int blocks = 128;
    for(int k = 0; k < 5; k++){
        int threads = 32;
        for(int j = 0; j < 4; j++){
            runKernels(blocks, threads, dx, dy, dz, dw);
            threads *= 2;
        }
        blocks *= 2;
    }

    int threads = 32;
    blocks = N / threads;
    runKernels(blocks, threads, dx, dy, dz, dw);

    threads = 64;
    blocks = N / threads;
    runKernels(blocks, threads, dx, dy, dz, dw);

    threads = 128;
    blocks = N / threads;
    runKernels(blocks, threads, dx, dy, dz, dw);

    threads = 256;
    blocks = N / threads;
    runKernels(blocks, threads, dx, dy, dz, dw);

    //Cleanup
    for (int i = 0; i < N; i++) {
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
