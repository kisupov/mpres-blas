/*
 *  Run time measurement macros for CPU (with OpenMP support) and CUDA
 *
 *  Copyright 2018, 2019 by Konstantin Isupov and Alexander Kuvaev.
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

#ifndef MPRES_TEST_TIMERS_CUH
#define MPRES_TEST_TIMERS_CUH

#include <iostream>
#include <chrono>

/*
 * Return CPU time in milliseconds between start and end
 */
inline double calcTimeCPU(struct timespec start, struct timespec end){
    long long start_nanos = start.tv_sec * 1000000000LL + start.tv_nsec;
    long long end_nanos = end.tv_sec * 1000000000LL + end.tv_nsec;
    return (end_nanos - start_nanos) * 1e-6f;
}

/*
 * Return GPU time in milliseconds between start and stop
 */
float calcTimeCUDA(cudaEvent_t start, cudaEvent_t stop) {
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    return milliseconds;
};

#define PrintTimerName(msg) std::cout << msg << " testing..." << std::endl;

#define InitCpuTimer()       struct timespec start, end; double _cpu_time = 0;

#define StartCpuTimer();      clock_gettime(CLOCK_MONOTONIC, &start);

#define EndCpuTimer();        clock_gettime(CLOCK_MONOTONIC, &end);_cpu_time += calcTimeCPU(start, end);

#define PrintCpuTimer(msg) std::cout << msg << "(ms): " << _cpu_time << std::endl;_cpu_time=0;

#define InitCudaTimer();                         \
      float _cuda_time = 0;                         \
      cudaEvent_t cuda_timer_start, cuda_timer_end; \
      cudaEventCreate(&cuda_timer_start);           \
      cudaEventCreate(&cuda_timer_end);

#define StartCudaTimer(); cudaEventRecord(cuda_timer_start, 0);

#define EndCudaTimer();                                           \
      checkDeviceHasErrors(cudaEventRecord(cuda_timer_end, 0));      \
      checkDeviceHasErrors(cudaEventSynchronize(cuda_timer_end));  \
      _cuda_time += calcTimeCUDA(cuda_timer_start, cuda_timer_end);

#define PrintCudaTimer(msg);   \
    std::cout << msg << "(ms): " << _cuda_time << std::endl; \
    _cuda_time = 0;


#endif //MPRES_TEST_TIMERS_CUH
