/*
 *  Performance test for the MPRES-BLAS library SpMV routine mpspmv_jad (double precision matrix)
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

#ifndef TEST_MPRES_MPDSPMV_JAD_CUH
#define TEST_MPRES_MPDSPMV_JAD_CUH

#include "tsthelper.cuh"
#include "logger.cuh"
#include "timers.cuh"
#include "sparse/jad/mpdspmv_jad.cuh"

/////////
//  SpMV jad kernel
/////////
void test_mpres_mpdspmv_jad(const int m, const int n, const int nzr, const int nnz, const int *ja, const int *jcp,
                       const double *as, const int *perm_rows, const mpfr_t *x) {
    InitCudaTimer();
    Logger::printDash();
    PrintTimerName("[GPU] MPRES-BLAS mpdspmv_jad");

    //Execution configuration
    int threads = 32;
    int blocks = m / threads + 1;
    printf("\tExec. config: blocks = %i, threads = %i\n", blocks, threads);

    //Memory requirements
    double sizeOfAs = get_double_array_size_in_mb(nnz);
    double sizeOfJad = sizeOfAs + get_int_array_size_in_mb(nnz) + get_int_array_size_in_mb(nzr + 1)+ get_int_array_size_in_mb(m);
    double sizeOfVectors = get_mp_float_array_size_in_mb(m + n);
    printf("\tMatrix (AS array) size (MB): %lf\n", sizeOfAs);
    printf("\tJAD structure size (MB): %lf\n", sizeOfJad);
    printf("\tVectors x and y size (MB): %lf\n", sizeOfVectors);
    printf("\tTOTAL Memory Consumption (MB): %lf\n", sizeOfJad + sizeOfVectors);

    // Host data
    auto hx = new mp_float_t[n];
    auto hy = new mp_float_t[m];

    // GPU data
    mp_float_ptr dx;
    mp_float_ptr dy;
    double *das;
    int *dja;
    int *djcp;
    int *dperm_rows;

    //Init data
    cudaMalloc(&dx, sizeof(mp_float_t) * n);
    cudaMalloc(&dy, sizeof(mp_float_t) * m);
    cudaMalloc(&das, sizeof(double) * nnz);
    cudaMalloc(&dja, sizeof(int) * nnz);
    cudaMalloc(&djcp, sizeof(int) * (nzr + 1));
    cudaMalloc(&dperm_rows, sizeof(int) * m);

    // Convert from MPFR
    convert_vector(hx, x, n);

    //Copying to the GPU
    cudaMemcpy(dx, hx, n * sizeof(mp_float_t), cudaMemcpyHostToDevice);
    cudaMemcpy(das, as, nnz * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dja, ja, nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(djcp, jcp, (nzr + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dperm_rows, perm_rows, m * sizeof(int), cudaMemcpyHostToDevice);

    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    //Launch
    StartCudaTimer();
    cuda::mpdspmv_jad<<<blocks, threads>>>(m, nzr, das, dja, djcp, dperm_rows, dx, dy);
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
    cudaFree(dx);
    cudaFree(dy);
    cudaFree(das);
    cudaFree(dja);
    cudaFree(djcp);
    cudaFree(dperm_rows);
}

#endif //TEST_MPRES_MPDSPMV_JAD_CUH