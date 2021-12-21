/*
 *  Performance test for the regular double precision SpMV JAD (JDS) routine
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

#include "../../tsthelper.cuh"
#include "../../logger.cuh"
#include "../../timers.cuh"
#include "sparse/utils/jad_utils.cuh"

/////////
// Double precision
/////////
__global__ static void double_spmv_jad_kernel(const int m, const int maxnzr, const jad_t jad, const double *x, double *y) {
    auto row = threadIdx.x + blockIdx.x * blockDim.x;
    while (row < m) {
        double dot = 0;
        auto j = 0;
        auto index = row;
        while (j < maxnzr && index < jad.jcp[j + 1]) {
            dot += jad.as[index] * x[jad.ja[index]];
            index = row + jad.jcp[++j];
        }
        y[jad.perm[row]] = dot;
        row +=  gridDim.x * blockDim.x;
    }
}

void test_double_spmv_jad(const int m, const int n, const int maxnzr, const int nnz, const jad_t &jad, const mpfr_t *x) {
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

    //GPU vectors
    double *dx;
    double *dy;
    cudaMalloc(&dx, sizeof(double) * n);
    cudaMalloc(&dy, sizeof(double) * m);
    convert_vector(hx, x, n);
    cudaMemcpy(dx, hx, sizeof(double) * n, cudaMemcpyHostToDevice);

    //GPU matrix
    jad_t djad;
    cuda::jad_init(djad, m, maxnzr, nnz);
    cuda::jad_host2device(djad, jad, m, maxnzr, nnz);


    //Launch
    StartCudaTimer();
    double_spmv_jad_kernel<<<blocks, threads>>>(m, maxnzr, djad, dx, dy);
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
    cuda::jad_clear(djad);
}


#endif //TEST_DOUBLE_SPMV_JAD_CUH
