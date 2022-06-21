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

#ifndef TEST_MPRES_CG_CSR_CUH
#define TEST_MPRES_CG_CSR_CUH

#include "sparse/solver/cg_csr.cuh"
#include "../../../tsthelper.cuh"
#include "../../../logger.cuh"
#include "../../../timers.cuh"


void test_mpres_cg_csr(const char * RESIDUAL_PATH, const int n, const int nnz, const csr_t &A, const double tol, const int maxit) {
    InitCudaTimer();
    Logger::printDash();
    PrintTimerName("[GPU] MPRES-BLAS CG CSR solver (mp_cg_csr)");
    //Host data
    auto hx = new mp_float_t[n];
    auto hb = new mp_float_t[n];
    std::vector<double> resvec;
    //Set solution to zero and right-hand-side vector to 1
    #pragma omp parallel for
    for(int i = 0; i < n; i++){
        mp_set(&hx[i], MP_ZERO); //initial residual
        mp_set(&hb[i], 1, 0, 0);
    }
    // GPU vectors
    mp_float_ptr dx;
    mp_float_ptr db;
    cudaMalloc(&dx, sizeof(mp_float_t) * n);
    cudaMalloc(&db, sizeof(mp_float_t) * n);
    cudaMemcpy(dx, hx, sizeof(mp_float_t) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(db, hb, sizeof(mp_float_t) * n, cudaMemcpyHostToDevice);
    //GPU matrix
    csr_t dA;
    cuda::csr_init(dA, n, nnz);
    cuda::csr_host2device(dA, A, n, nnz);
    //Launch
    StartCudaTimer();
    int iters = mp_cg_csr(n, dA, db, tol, maxit, dx, resvec);
    EndCudaTimer();
    PrintAndResetCudaTimer("took");
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();
    //Copying to the host
    cudaMemcpy(hx, dx, sizeof(mp_float_t) * n, cudaMemcpyDeviceToHost);
    std::cout << "iterations: " << iters << std::endl;
    print_residual(n, A, hx, hb);
    //Write residual history
    string postfix = "_cg_res_" + std::to_string(RNS_MODULI_SIZE) +"-moduli.txt";
    store_residual_history(std::string (RESIDUAL_PATH) + postfix, resvec);
    //Cleanup
    delete [] hx;
    delete [] hb;
    cudaFree(dx);
    cudaFree(db);
    cuda::csr_clear(dA);
    resvec.clear();
    resvec.shrink_to_fit();
}

#endif //TEST_MPRES_CG_CSR_CUH
