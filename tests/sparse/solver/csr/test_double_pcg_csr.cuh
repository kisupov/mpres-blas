/*
 *  Test for double precision diagonal (Jacobi) preconditioned CG linear solver using CSR
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

#ifndef TEST_DOUBLE_PCG_CSR_CUH
#define TEST_DOUBLE_PCG_CSR_CUH

#include "sparse/msparse_enum.cuh"
#include "../../../tsthelper.cuh"
#include "../../../logger.cuh"
#include "../../../timers.cuh"
#include "../double/double_pcg_csr_gpu.cuh"

void test_double_pcg_csr(const char * RESIDUAL_PATH, const int n, const int nnz, const csr_t &A, const double tol, const int maxit) {
    int threads = 256;
    int blocks = n / threads + 1;
    InitCudaTimer();
    Logger::printDash();
    PrintTimerName("[GPU] double PCG CSR solver with diagonal preconditioner");
    //Host data
    auto *hx = new double[n];
    auto *hb = new double[n];
    auto *hM = new double [n];
    std::vector<double> resvec;
    //Construct Jacobi preconditioner M and compute M^{-1}
    csr_mdiag(A, n, hM); //M = main diagonal of A
    #pragma omp parallel for
    for(int i = 0; i < n; i++){
        hx[i] = 0;
        hb[i] = 1;
        assert(hM[i] != 0.0);
        hM[i] = 1.0 / hM[i]; //vector reciprocal
    }
    // GPU vectors
    double *db;
    double *dx;
    double *dM;
    cudaMalloc(&db, sizeof(double) * n);
    cudaMalloc(&dx, sizeof(double) * n);
    cudaMalloc(&dM, sizeof(double) * n);
    cudaMemcpy(dx, hx, sizeof(double) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(db, hb, sizeof(double) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(dM, hM, sizeof(double) * n, cudaMemcpyHostToDevice);
    //GPU matrix
    csr_t dA;
    cuda::csr_init(dA, n, nnz);
    cuda::csr_host2device(dA, A, n, nnz);
    //Launch
    StartCudaTimer();
    int iters = double_pcg_csr(n, dA, db, tol, maxit, diag, dM, dx, resvec, blocks, threads)
    EndCudaTimer();
    PrintAndResetCudaTimer("took");
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();
    cudaMemcpy(hx, dx, sizeof(double) * n, cudaMemcpyDeviceToHost);
    std::cout << "iterations: " << iters << std::endl;
    print_residual(n, A, hx, hb);
    //Write residual history
    store_residual_history(std::string (RESIDUAL_PATH) + "_pcg_res_double.txt", resvec);
    //Cleanup
    delete [] hx;
    delete [] hb;
    delete [] hM;
    cudaFree(dx);
    cudaFree(db);
    cudaFree(dM);
    cuda::csr_clear(dA);
    resvec.clear();
    resvec.shrink_to_fit();
}

#endif //TEST_DOUBLE_PCG_CSR_CUH
