/*
 *  Test for multiple precision unpreconditioned CG linear solver using CSR
 *
 *  Copyright 2021 by Konstantin Isupov.
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

#ifndef TEST_MPRES_CGCSRLS_CUH
#define TEST_MPRES_CGCSRLS_CUH

#include "../../tsthelper.cuh"
#include "../../logger.cuh"
#include "../../timers.cuh"
#include "sparse/cgcsrls/cgcsrls.cuh"


void test_mpres_cgcsrls(const int n, const int nnz, const csr_t &matrix, const double * rhs, const double tol, const double maxit) {
    InitCudaTimer();
    Logger::printDash();
    PrintTimerName("[GPU] MPRES-BLAS CG solver (mp_cgcsrls)");
    //Execution configuration
    printf("\tExec. config: blocks = %i, threads = %i, blocks for dot product and norm2 = %i\n", (n / THREADS + 1), THREADS, BLOCKS_SCALAR);
    //Host data
    auto hx = new mp_float_t[n];
    // GPU vectors
    mp_float_ptr drhs;
    mp_float_ptr dx;
    mp_float_ptr dr;
    mp_float_ptr dp;
    mp_float_ptr dq;
    cudaMalloc(&drhs, sizeof(mp_float_t) * n);
    cudaMalloc(&dx, sizeof(mp_float_t) * n);
    cudaMalloc(&dr, sizeof(mp_float_t) * n);
    cudaMalloc(&dp, sizeof(mp_float_t) * n);
    cudaMalloc(&dq, sizeof(mp_float_t) * n);
    //Initial guess
    #pragma omp parallel for
    for(int i = 0; i < n; i++){
        mp_set(&hx[i], MP_ZERO);
    }
    cudaMemcpy(dx, hx, sizeof(mp_float_t) * n, cudaMemcpyHostToDevice);
    //Right-hand-side vector
    #pragma omp parallel for
    for(int i = 0; i < n; i++){
        mp_set_d(&hx[i], rhs[i]);
    }
    cudaMemcpy(drhs, hx, sizeof(mp_float_t) * n, cudaMemcpyHostToDevice);
    //GPU matrix
    csr_t dmatrix;
    cuda::csr_init(dmatrix, n, nnz);
    cuda::csr_host2device(dmatrix, matrix, n, nnz);
    //Launch
    StartCudaTimer();
    int iters = mp_cgcsrls(n, dmatrix, drhs, dx, dr, dp, dq, tol, maxit)
    EndCudaTimer();
    PrintCudaTimer("took");
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();
    //Copying to the host
    cudaMemcpy(hx, dx, sizeof(mp_float_t) * n, cudaMemcpyDeviceToHost);
    std::cout << "iterations: " << iters << std::endl;
    print_residual(n, matrix, hx, rhs);
    //Cleanup
    delete [] hx;
    cudaFree(drhs);
    cudaFree(dx);
    cudaFree(dr);
    cudaFree(dp);
    cudaFree(dq);
    cuda::csr_clear(dmatrix);
}

#endif //TEST_MPRES_CGCSRLS_CUH
