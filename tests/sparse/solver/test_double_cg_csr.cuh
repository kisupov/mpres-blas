/*
 *  Test for double precision unpreconditioned CG linear solver using CSR
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

#ifndef TEST_DOUBLE_CG_CSR_CUH
#define TEST_DOUBLE_CG_CSR_CUH

#include "../../tsthelper.cuh"
#include "../../logger.cuh"
#include "../../timers.cuh"
#include "sparse/solver/double/double_cg_csr_gpu.cuh"

void test_double_cg_csr(const char * RESIDUAL_PATH, const int n, const int nnz, const csr_t &A, const double tol, const int maxit) {
    int threads = 256;
    int blocks = n / threads + 1;
    InitCudaTimer();
    Logger::printDash();
    PrintTimerName("[GPU] double CG CSR solver");
    auto *hx = new double[n];
    auto *hb = new double[n];
    std::vector<double> resvec;
    #pragma omp parallel for
    for(int i = 0; i < n; i++){
        hx[i] = 0;
        hb[i] = 1;
    }
    // GPU vectors
    double *db;
    double *dx;
    cudaMalloc(&db, sizeof(double) * n);
    cudaMalloc(&dx, sizeof(double) * n);
    cudaMemcpy(dx, hx, sizeof(double) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(db, hb, sizeof(double) * n, cudaMemcpyHostToDevice);
    //GPU matrix
    csr_t dA;
    cuda::csr_init(dA, n, nnz);
    cuda::csr_host2device(dA, A, n, nnz);
    StartCudaTimer();
    int iters = double_cg_csr(n, dA, db, tol, maxit, dx, resvec, blocks, threads)
    EndCudaTimer();
    PrintAndResetCudaTimer("took");
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();
    cudaMemcpy(hx, dx, sizeof(double) * n, cudaMemcpyDeviceToHost);
    std::cout << "iterations: " << iters << std::endl;
    print_residual(n, A, hx, hb);
    //Write residual history
    store_residual_history(std::string (RESIDUAL_PATH) + "_cg_res_double.txt", resvec);
    //Cleanup
    delete [] hx;
    delete [] hb;
    cudaFree(db);
    cudaFree(dx);
    cuda::csr_clear(dA);
    resvec.clear();
    resvec.shrink_to_fit();
}

#endif //TEST_DOUBLE_CG_CSR_CUH
