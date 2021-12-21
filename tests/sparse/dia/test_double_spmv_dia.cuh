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

#include "../../tsthelper.cuh"
#include "../../logger.cuh"
#include "../../timers.cuh"

/////////
// Double precision
/////////
__global__ static void double_spmv_dia_kernel(const int m, const int n, const int ndiag, const dia_t dia, const double *x, double *y) {
    unsigned int row = threadIdx.x + blockIdx.x * blockDim.x;
    if(row < m) {
        double dot = 0;
        for (int i = 0; i < ndiag; i++) {
            int col = row + dia.offset[i];
            double val = dia.as[m * i + row];
            if(col  >= 0 && col < n && val != 0)
                dot += val * x[col];
        }
        y[row] = dot;
    }
}

void test_double_spmv_dia(const int m, const int n, const int ndiag, const dia_t &dia, const mpfr_t *x) {
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

    //GPU vectors
    double *dx;
    double *dy;
    cudaMalloc(&dx, sizeof(double) * n);
    cudaMalloc(&dy, sizeof(double) * m);
    convert_vector(hx, x, n);
    cudaMemcpy(dx, hx, sizeof(double) * n, cudaMemcpyHostToDevice);

    //GPU matrix
    dia_t ddia;
    cuda::dia_init(ddia, m, ndiag);
    cuda::dia_host2device(ddia, dia, m, ndiag);

    //Launch
    StartCudaTimer();
    double_spmv_dia_kernel<<<blocks, threads>>>(m, n, ndiag, ddia, dx, dy);
    EndCudaTimer();
    PrintAndResetCudaTimer("took");
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    //Copying to the host
    cudaMemcpy(hy, dy, sizeof(double) * m , cudaMemcpyDeviceToHost);
    print_double_sum(hy, m);

    delete [] hx;
    delete [] hy;
    cudaFree(dx);
    cudaFree(dy);
    cuda::dia_clear(ddia);
}

#endif //TEST_DOUBLE_SPMV_DIA_CUH
