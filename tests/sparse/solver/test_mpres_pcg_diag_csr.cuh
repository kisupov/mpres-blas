/*
 *  Test for multiple precision PCG linear solver using CSR and diagonal preconditioner
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

#ifndef TEST_MPRES_PCG_DIAG_CSR_CUH
#define TEST_MPRES_PCG_DIAG_CSR_CUH

#include "../../tsthelper.cuh"
#include "../../logger.cuh"
#include "../../timers.cuh"
#include "sparse/solver/pcg_csr.cuh"
#include "sparse/msparse_enum.cuh"


void test_mpres_pcg_diag_csr(const int n, const int nnz, const csr_t &A, const double tol, const int maxit) {
    InitCudaTimer();
    Logger::printDash();
    PrintTimerName("[GPU] MPRES-BLAS PCG CSR solver (mp_pcg_csr) with diagonal preconditioner");
    //Host data
    auto hx = new mp_float_t[n];
    auto hb = new mp_float_t[n];
    auto *hM = new double [n];
    std::vector<double> resvec;
    //Construct Jacobi preconditioner M and compute M^{-1}, set solution to zero and right-hand-side vector to 1
    mdiag(A, n, hM); //hM = main diagonal of A
    #pragma omp parallel for
    for(int i = 0; i < n; i++){
        mp_set(&hx[i], MP_ZERO);
        mp_set(&hb[i], 1, 0, 0);
        assert(hM[i] != 0.0);
        hM[i] = 1.0 / hM[i]; //reciprocals
    }
    //GPU vectors
    mp_float_ptr db;
    mp_float_ptr dx;
    double *dM;
    cudaMalloc(&db, sizeof(mp_float_t) * n);
    cudaMalloc(&dx, sizeof(mp_float_t) * n);
    cudaMalloc(&dM, sizeof(double) * n);
    cudaMemcpy(dx, hx, sizeof(mp_float_t) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(db, hb, sizeof(mp_float_t) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(dM, hM, sizeof(double) * n, cudaMemcpyHostToDevice);
    //GPU matrix
    csr_t dA;
    cuda::csr_init(dA, n, nnz);
    cuda::csr_host2device(dA, A, n, nnz);
    //Launch
    StartCudaTimer();
    int iters = mp_pcg_csr(n, dA, db, tol, maxit, diag, dM, dx, resvec);
    EndCudaTimer();
    PrintCudaTimer("took");
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();
    //Copying to the host
    cudaMemcpy(hx, dx, sizeof(mp_float_t) * n, cudaMemcpyDeviceToHost);
    std::cout << "iterations: " << iters << std::endl;
    print_residual(n, A, hx, hb);
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

#endif //TEST_MPRES_PCG_DIAG_CSR_CUH
