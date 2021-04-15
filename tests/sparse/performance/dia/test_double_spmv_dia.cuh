/*
 *  Performance test for the regular double precision SpMV DIA routine (scalar kernel)
 *
 *  Copyright 2020 by Konstantin Isupov.
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

#ifndef TEST_DOUBLE_SPMV_DIA_CUH
#define TEST_DOUBLE_SPMV_DIA_CUH

#include "tsthelper.cuh"
#include "logger.cuh"
#include "timers.cuh"

/////////
// Double precision
/////////
__global__ static void double_spmv_dia_kernel(const int m, const int n, const int ndiag, const int *offset, const double *as, const double *x, double *y) {
    unsigned int row = threadIdx.x + blockIdx.x * blockDim.x;
    if(row < m) {
        double dot = 0;
        for (int i = 0; i < ndiag; i++) {
            int col = row + offset[i];
            double val = as[m * i + row];
            if(col  >= 0 && col < n)
                dot += val * x[col];
        }
        y[row] = dot;
    }
}

void test_double_spmv_dia(const int m, const int n, const int ndiag, const int *offset, const double *as, const mpfr_t *x) {
    InitCudaTimer();
    Logger::printDash();
    PrintTimerName("[GPU] double SpMV DIA");

    //Execution configuration
    int threads = 32;
    int blocks = m / threads + 1;
    printf("Exec. config: blocks = %i, threads = %i\n", blocks, threads);
    printf("Matrix (AS array) size (MB): %lf\n", get_double_array_size_in_mb(m * ndiag));

    //host data
    auto *hx = new double[n];
    auto *hy = new double[m];

    //GPU data
    auto *das = new double[m * ndiag];
    auto *doffset = new int[ndiag];
    auto *dx = new double[n];
    auto *dy = new double[m];

    cudaMalloc(&das, sizeof(double) * m * ndiag);
    cudaMalloc(&doffset, sizeof(int) * ndiag);
    cudaMalloc(&dx, sizeof(double) * n);
    cudaMalloc(&dy, sizeof(double) * m);

    // Convert from MPFR
    convert_vector(hx, x, n);

    //Copying data to the GPU
    cudaMemcpy(das, as, sizeof(double) * m * ndiag, cudaMemcpyHostToDevice);
    cudaMemcpy(doffset, offset, sizeof(int) * ndiag, cudaMemcpyHostToDevice);
    cudaMemcpy(dx, hx, sizeof(double) * n, cudaMemcpyHostToDevice);

    //Launch
    StartCudaTimer();
    double_spmv_dia_kernel<<<blocks, threads>>>(m, n, ndiag, doffset, das, dx, dy);
    EndCudaTimer();
    PrintCudaTimer("took");
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    //Copying to the host
    cudaMemcpy(hy, dy, sizeof(double) * m , cudaMemcpyDeviceToHost);
    print_double_sum(hy, m);

    delete [] hx;
    delete [] hy;
    cudaFree(das);
    cudaFree(doffset);
    cudaFree(dx);
    cudaFree(dy);
}

#endif //TEST_DOUBLE_SPMV_DIA_CUH
