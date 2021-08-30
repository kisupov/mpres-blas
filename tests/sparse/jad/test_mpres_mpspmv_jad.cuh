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

#ifndef TEST_MPRES_MPSPMV_JAD_CUH
#define TEST_MPRES_MPSPMV_JAD_CUH

#include "../../tsthelper.cuh"
#include "../../logger.cuh"
#include "../../timers.cuh"
#include "sparse/spmv_jad.cuh"
#include "sparse/utils/jad_utils.cuh"

/////////
//  SpMV jad kernel test
/////////
void test_mpres_mpspmv_jad(const int m, const int n, const int maxnzr, const int nnz, const jad_t &jad, const mpfr_t *x) {
    InitCudaTimer();
    Logger::printDash();
    PrintTimerName("[GPU] MPRES-BLAS JAD (mpspmv_jad)");

    //Execution configuration
    int threads = 32;
    int blocks = m / threads + 1;
    printf("\tExec. config: blocks = %i, threads = %i\n", blocks, threads);

    // Host data
    auto hx = new mp_float_t[n];
    auto hy = new mp_float_t[m];

    // GPU vectors
    mp_float_ptr dx;
    mp_float_ptr dy;
    cudaMalloc(&dx, sizeof(mp_float_t) * n);
    cudaMalloc(&dy, sizeof(mp_float_t) * m);
    convert_vector(hx, x, n);
    cudaMemcpy(dx, hx, n * sizeof(mp_float_t), cudaMemcpyHostToDevice);

    //GPU matrix
    jad_t djad;
    cuda::jad_init(djad, m, maxnzr, nnz);
    cuda::jad_host2device(djad, jad, m, maxnzr, nnz);

    //Launch
    StartCudaTimer();
    cuda::mpspmv_jad<32><<<blocks, threads>>>(m, maxnzr, djad, dx, dy);
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
    cuda::jad_clear(djad);

}

#endif //TEST_MPRES_MPSPMV_JAD_CUH