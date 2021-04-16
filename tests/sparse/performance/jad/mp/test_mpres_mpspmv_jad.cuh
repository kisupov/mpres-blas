/*
 *  Performance test for the MPRES-BLAS library SpMV routine mpspmv_jad (multiple precision matrix)
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

#ifndef TEST_MPRES_MPSPMV_JAD_CUH
#define TEST_MPRES_MPSPMV_JAD_CUH

#include "../../../../tsthelper.cuh"
#include "../../../../logger.cuh"
#include "../../../../timers.cuh"
#include "../../../../../src/sparse/jad/mpspmv_jad.cuh"

/////////
//  SpMV jad scalar kernel
/////////
void test_mpres_mpspmv_jad(const int m, const int n, const int maxnz, const int nnz, const int *ja, const int *jcp,
                      const double *as, const int *perm_rows, const mpfr_t *x) {
    InitCudaTimer();
    Logger::printDash();
    PrintTimerName("[GPU] MPRES-BLAS mpspmv_jad");

    //Execution configuration
    int threads = 32;
    int blocks = m / threads + 1;
    printf("\tExec. config: blocks = %i, threads = %i\n", blocks, threads);
    printf("\tMatrix (AS array) size (MB): %lf\n", get_mp_float_array_size_in_mb(nnz));

    // Host data
    auto hx = new mp_float_t[n];
    auto hy = new mp_float_t[m];
    auto has = new mp_float_t[nnz];

    // GPU data
    mp_float_ptr dx;
    mp_float_ptr dy;
    mp_float_ptr das;
    int *dja;
    int *djcp;
    int *dperm_rows;

    //Init data
    cudaMalloc(&dx, sizeof(mp_float_t) * n);
    cudaMalloc(&dy, sizeof(mp_float_t) * m);
    cudaMalloc(&das, sizeof(mp_float_t) * nnz);
    cudaMalloc(&dja, sizeof(int) * nnz);
    cudaMalloc(&djcp, sizeof(int) * (maxnz + 1));
    cudaMalloc(&dperm_rows, sizeof(int) * m);

    // Convert from MPFR
    convert_vector(hx, x, n);
    convert_vector(has, as, nnz);

    //Copying to the GPU
    cudaMemcpy(dx, hx, n * sizeof(mp_float_t), cudaMemcpyHostToDevice);
    cudaMemcpy(das, has, nnz * sizeof(mp_float_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dja, ja, nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(djcp, jcp, (maxnz + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dperm_rows, perm_rows, m * sizeof(int), cudaMemcpyHostToDevice);

    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    //Launch
    StartCudaTimer();
    cuda::mpspmv_jad<<<blocks, threads>>>(m, maxnz, das, dja, djcp, dperm_rows, dx, dy);
    EndCudaTimer();
    PrintCudaTimer("took");
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    //Copying to the host
    cudaMemcpy(hy, dy, m * sizeof(mp_float_t), cudaMemcpyDeviceToHost);
    print_mp_sum(hy, m);

    //Cleanup
    delete[] hx;
    delete[] hy;
    delete[] has;
    cudaFree(dx);
    cudaFree(dy);
    cudaFree(das);
    cudaFree(dja);
    cudaFree(djcp);
    cudaFree(dperm_rows);
}

#endif //TEST_MPRES_MPSPMV_JAD_CUH