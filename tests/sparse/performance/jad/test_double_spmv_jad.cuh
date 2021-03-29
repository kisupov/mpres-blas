/*
 *  Performance test for the regular double precision SpMV JAD (JDS) routine (scalar kernel)
 *
 *  Copyright 2020 by Konstantin Isupov and Ivan Babeshko.
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

#ifndef TEST_DOUBLE_SPMV_JAD_CUH
#define TEST_DOUBLE_SPMV_JAD_CUH

#include "tsthelper.cuh"
#include "logger.cuh"
#include "timers.cuh"

/////////
// Double precision
/////////
__global__ static void double_spmv_jad_kernel(const int m, const int nzr, const int *ja, const double *as, const int *jcp, const int *perm_rows, const double *x, double *y) {
    unsigned int row = threadIdx.x + blockIdx.x * blockDim.x;
    if (row < m) {
        double dot = 0;
        int j = 0;
        int index = row;

        while (j < nzr && index < jcp[j + 1]) {
            dot += as[index] * x[ja[index]];
            index += jcp[j+1] - jcp[j];
            j++;
        }
        y[perm_rows[row]] = dot;
    }
}

void test_double_spmv_jad(const int m, const int n, const int nzr, const int nnz, const int *ja, const int *jcp, const double *as, const int *perm_rows, const mpfr_t *x) {
    InitCudaTimer();
    Logger::printDash();
    PrintTimerName("[GPU] double SpMV JAD (JDS)");

    //Execution configuration
    int threads = 32;
    int blocks = m / threads + 1;
    printf("\tExec. config: blocks = %i, threads = %i\n", blocks, threads);

    //host data
    auto *hx = new double[n];
    auto *hy = new double[m];

    //GPU data
    double *das;
    int *dja;
    int *djcp;
    int *dperm_rows;
    double *dx;
    double *dy;

    cudaMalloc(&das, sizeof(double) * nnz);
    cudaMalloc(&dja, sizeof(int) * nnz);
    cudaMalloc(&djcp, sizeof(int) * (nzr + 1));
    cudaMalloc(&dperm_rows, sizeof(int) * m);
    cudaMalloc(&dx, sizeof(double) * n);
    cudaMalloc(&dy, sizeof(double) * m);

    // Convert from MPFR
    convert_vector(hx, x, n);

    //Copying data to the GPU
    cudaMemcpy(das, as, sizeof(double) * nnz, cudaMemcpyHostToDevice);
    cudaMemcpy(dja, ja, sizeof(int) * nnz, cudaMemcpyHostToDevice);
    cudaMemcpy(djcp, jcp, sizeof(int) * (nzr + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(dperm_rows, perm_rows, sizeof(int) * m, cudaMemcpyHostToDevice);
    cudaMemcpy(dx, hx, sizeof(double) * n, cudaMemcpyHostToDevice);

    //Launch
    StartCudaTimer();
    double_spmv_jad_kernel<<<blocks, threads>>>(m, nzr, dja, das, djcp, dperm_rows, dx, dy);
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
    cudaFree(djcp);
    cudaFree(dperm_rows);
    cudaFree(dx);
    cudaFree(dy);
}

#endif //TEST_DOUBLE_SPMV_JAD_CUH
