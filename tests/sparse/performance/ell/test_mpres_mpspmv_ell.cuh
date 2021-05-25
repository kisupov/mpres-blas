/*
 *  Performance test for the MPRES-BLAS library SpMV routine mpspmv_ell (double precision matrix)
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

#ifndef TEST_MPRES_MPSPMV_ELL_CUH
#define TEST_MPRES_MPSPMV_ELL_CUH

#include "../../../tsthelper.cuh"
#include "../../../logger.cuh"
#include "../../../timers.cuh"
#include "../../../../src/sparse/mpspmv_ell.cuh"

/////////
//  SpMV ELLPACK scalar kernel
/////////
void test_mpres_mpspmv_ell(const int m, const int n, const int maxnzr, const int *ja, const double *as, const mpfr_t *x){
    InitCudaTimer();
    Logger::printDash();
    PrintTimerName("[GPU] MPRES-BLAS ELLPACK (mpspmv_ell)");

    //Execution configuration
    int threads = 32;
    int blocks = m / threads + 1;
    printf("\tExec. config: blocks = %i, threads = %i\n", blocks, threads);

    // Host data
    auto hx = new mp_float_t[n];
    auto hy = new mp_float_t[m];

    // GPU data
    mp_float_ptr dx;
    mp_float_ptr dy;
    double *das;
    int *dja;

    //Init data
    cudaMalloc(&dx, sizeof(mp_float_t) * n);
    cudaMalloc(&dy, sizeof(mp_float_t) * m);
    cudaMalloc(&das, sizeof(double) * m * maxnzr);
    cudaMalloc(&dja, sizeof(int) * m * maxnzr);

    // Convert from MPFR
    convert_vector(hx, x, n);

    //Copying to the GPU
    cudaMemcpy(dx, hx, n * sizeof(mp_float_t), cudaMemcpyHostToDevice);
    cudaMemcpy(das, as, m * maxnzr * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dja, ja, m * maxnzr * sizeof(int), cudaMemcpyHostToDevice);

    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    //Launch
    StartCudaTimer();
    cuda::mpspmv_ell<32><<<blocks, threads>>>(m, maxnzr, dja, das, dx, dy);
    EndCudaTimer();
    PrintCudaTimer("took");
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    //Copying to the host
    cudaMemcpy(hy, dy, m * sizeof(mp_float_t), cudaMemcpyDeviceToHost);
    print_mp_sum(hy, m);

    //Cleanup
    delete [] hx;
    delete [] hy;
    cudaFree(dx);
    cudaFree(dy);
    cudaFree(das);
    cudaFree(dja);
}

#endif //TEST_MPRES_MPSPMV_ELL_CUH