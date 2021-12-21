/*
 *  Performance test for the regular double precision SpMV ELLPACK routine (scalar kernel)
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

#ifndef TEST_DOUBLE_SPMV_ELL_CUH
#define TEST_DOUBLE_SPMV_ELL_CUH

#include "../../tsthelper.cuh"
#include "../../logger.cuh"
#include "../../timers.cuh"

/////////
// Double precision
/////////
__global__ static void double_spmv_ell_kernel(const int m, const int maxnzr, const ell_t ell, const double *x, double *y) {
    unsigned int row = threadIdx.x + blockIdx.x * blockDim.x;
    if(row < m){
        double dot = 0;
        for (int col = 0; col < maxnzr; col++) {
            int j = ell.ja[col * m + row];
            double val = ell.as[col * m + row];
            if(val != 0){
                dot += val * x[j];
            }
        }
        y[row] = dot;
    }
}

void test_double_spmv_ell(const int m, const int n, const int maxnzr, const ell_t &ell, const mpfr_t *x) {
    InitCudaTimer();
    Logger::printDash();
    PrintTimerName("[GPU] double SpMV ELLPACK");

    //Execution configuration
    int threads = 32;
    int blocks = m / threads + 1;
    printf("\tExec. config: blocks = %i, threads = %i\n", blocks, threads);

    //host data
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
    ell_t dell;
    cuda::ell_init(dell, m, maxnzr);
    cuda::ell_host2device(dell, ell, m, maxnzr);

    //Launch
    StartCudaTimer();
    double_spmv_ell_kernel<<<blocks, threads>>>(m, maxnzr, dell, dx, dy);
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
    cuda::ell_clear(dell);
}

#endif //TEST_DOUBLE_SPMV_ELL_CUH
