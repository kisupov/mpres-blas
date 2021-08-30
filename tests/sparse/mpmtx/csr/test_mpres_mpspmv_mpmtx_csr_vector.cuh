/*
 *  Performance test for the MPRES-BLAS library SpMV routine mpspmv_mpmtx_csr_vector (multiple precision matrix)
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

#ifndef TEST_MPRES_MPSPMV_MPMTX_CSR_VECTOR_CUH
#define TEST_MPRES_MPSPMV_MPMTX_CSR_VECTOR_CUH

#include "../../../tsthelper.cuh"
#include "../../../logger.cuh"
#include "../../../timers.cuh"
#include "../../../../src/sparse/mpmtx/spmv_mpmtx_csrv.cuh"

/////////
//  SpMV CSR vector kernel with multiple threads per matrix row
/////////
template<int threadsPerRow>
void test_mpres_mpspmv_mpmtx_csr_vector(const int m, const int n, const int nnz, const csr_t &csr, const mpfr_t * x){
    InitCudaTimer();
    Logger::printDash();
    PrintTimerName("[GPU] MPRES-BLAS mpspmv_mpmtx_csr_vector");

    //Execution configuration
    const int threads = 256;
    const int blocks = m / (threads/threadsPerRow) + 1;
    //int blocks = 32;
    printf("\tThreads per row = %i\n", threadsPerRow);
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
    int *dirp;
    int *dja;

    //Init data
    cudaMalloc(&dx, sizeof(mp_float_t) * n);
    cudaMalloc(&dy, sizeof(mp_float_t) * m);
    cudaMalloc(&das, sizeof(mp_float_t) * nnz);
    cudaMalloc(&dirp, sizeof(int) * (m + 1));
    cudaMalloc(&dja, sizeof(int) * nnz);

    // Convert from MPFR
    convert_vector(hx, x, n);
    convert_vector(has, csr.as, nnz);

    //Copying to the GPU
    cudaMemcpy(dx, hx, n * sizeof(mp_float_t), cudaMemcpyHostToDevice);
    cudaMemcpy(das, has, nnz * sizeof(mp_float_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dirp, csr.irp, (m + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dja, csr.ja, nnz * sizeof(int), cudaMemcpyHostToDevice);

    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    //Launch
    StartCudaTimer();
    cuda::mpspmv_mpmtx_csr_vector<threadsPerRow><<<blocks, threads, sizeof(mp_float_t) * threads>>>(m, dirp, dja, das, dx, dy);
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
    delete [] has;
    cudaFree(dx);
    cudaFree(dy);
    cudaFree(das);
    cudaFree(dirp);
    cudaFree(dja);
}

#endif //TEST_MPRES_MPSPMV_MPMTX_CSR_VECTOR_CUH