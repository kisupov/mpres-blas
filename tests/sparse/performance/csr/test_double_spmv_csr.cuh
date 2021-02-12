/*
 *  Performance test for the regular double precision SpMV CSR routine (scalar kernel)
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

#ifndef TEST_DOUBLE_SPMV_CSR_CUH
#define TEST_DOUBLE_SPMV_CSR_CUH

#include "tsthelper.cuh"
#include "logger.cuh"
#include "timers.cuh"

/////////
// Double precision
/////////
__global__ static void double_spmv_csr_kernel(const int m, const int *irp, const int *ja, const double *as, const double *x, double *y) {
    unsigned int row = threadIdx.x + blockIdx.x * blockDim.x;
    if(row < m){
        double dot = 0;
        int row_start = irp[row];
        int row_end = irp[row+1];
        for (int i = row_start; i < row_end; i++) {
            dot += as[i] * x[ja[i]];
        }
        y[row] = dot;
    }
}


void test_double_spmv_csr(const int m, const int n, const int nnz, const int *irp, const int *ja, const double *as, const mpfr_t *x) {
    InitCudaTimer();
    Logger::printDash();
    PrintTimerName("[GPU] double SpMV CSR");

    //Execution configuration
    int threads = 32;
    int blocks = m / threads + 1;
    printf("\tExec. config: blocks = %i, threads = %i\n", blocks, threads);

    //host data
    auto *hx = new double[n];
    auto *hy = new double[m];

    //GPU data
    double *das;
    int *dirp;
    int *dja;
    double *dx;
    double *dy;

    cudaMalloc(&das, sizeof(double) * nnz);
    cudaMalloc(&dirp, sizeof(int) * (m +1));
    cudaMalloc(&dja, sizeof(int) * nnz);
    cudaMalloc(&dx, sizeof(double) * n);
    cudaMalloc(&dy, sizeof(double) * m);

    // Convert from MPFR
    convert_vector(hx, x, n);

    //Copying data to the GPU
    cudaMemcpy(das, as, sizeof(double) * nnz, cudaMemcpyHostToDevice);
    cudaMemcpy(dja, ja, sizeof(int) * nnz, cudaMemcpyHostToDevice);
    cudaMemcpy(dirp, irp, sizeof(int) * (m +1), cudaMemcpyHostToDevice);
    cudaMemcpy(dx, hx, sizeof(double) * n, cudaMemcpyHostToDevice);

    //Launch
    StartCudaTimer();
    double_spmv_csr_kernel<<<blocks, threads>>>(m, dirp, dja, das, dx, dy);
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
    cudaFree(dja);
    cudaFree(dirp);
    cudaFree(dx);
    cudaFree(dy);
}

#endif //TEST_DOUBLE_SPMV_CSR_CUH
