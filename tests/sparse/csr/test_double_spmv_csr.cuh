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

#include "../../tsthelper.cuh"
#include "../../logger.cuh"
#include "../../timers.cuh"

/////////
// Double precision
/////////
__global__ static void double_spmv_csr_kernel(const int m, const csr_t csr, const double *x, double *y) {
    unsigned int row = threadIdx.x + blockIdx.x * blockDim.x;
    if(row < m){
        double dot = 0;
        int row_start = csr.irp[row];
        int row_end = csr.irp[row+1];
        for (int i = row_start; i < row_end; i++) {
            dot += csr.as[i] * x[csr.ja[i]];
        }
        y[row] = dot;
    }
}


void test_double_spmv_csr(const int m, const int n, const int nnz, const csr_t &csr, const mpfr_t *x) {
    InitCudaTimer();
    Logger::printDash();
    PrintTimerName("[GPU] double SpMV CSR");

    //Execution configuration
    int threads = 32;
    int blocks = m / threads + 1;
    printf("\tExec. config: blocks = %i, threads = %i\n", blocks, threads);

    //Host data
    auto *hx = new double[n];
    auto *hy = new double[m];

    // GPU vectors
    double *dx;
    double *dy;
    cudaMalloc(&dx, sizeof(double) * n);
    cudaMalloc(&dy, sizeof(double) * m);
    convert_vector(hx, x, n);
    cudaMemcpy(dx, hx, sizeof(double) * n, cudaMemcpyHostToDevice);

    //GPU matrix
    csr_t dcsr;
    cuda::csr_init(dcsr, m, nnz);
    cuda::csr_host2device(dcsr, csr, m, nnz);

    //Launch
    StartCudaTimer();
    double_spmv_csr_kernel<<<blocks, threads>>>(m, dcsr, dx, dy);
    EndCudaTimer();
    PrintCudaTimer("took");
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    //Copying to the host
    cudaMemcpy(hy, dy, sizeof(double) * m , cudaMemcpyDeviceToHost);
    print_double_sum(hy, m);

    delete [] hx;
    delete [] hy;
    cudaFree(dx);
    cudaFree(dy);
    cuda::csr_clear(dcsr);
}

#endif //TEST_DOUBLE_SPMV_CSR_CUH
