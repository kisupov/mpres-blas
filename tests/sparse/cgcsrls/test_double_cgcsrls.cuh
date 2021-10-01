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

#ifndef TEST_DOUBLE_CGCSRLS_CUH
#define TEST_DOUBLE_CGCSRLS_CUH

#include "../../tsthelper.cuh"
#include "../../logger.cuh"
#include "../../timers.cuh"
#include "double_ls_kernels.cuh"

/////////
// Unpreconditioned double precision CG iterative method
/////////
/**
 * Unpreconditioned double precision conjugate gradient linear solver, Ax = b
 * ASSUMPTIONS: The appropriate memory has been allocated and set to zero.
 * The relative residual or maximum number of iterations are used as a stopping criteria: ||r_k|| <= ||r_0|| * tol or k > maxiter
 *
 * @param n - operation size
 * @param matrix - matrix A in gpu memory
 * @param rhs - right-hand side vector b in gpu memory
 * @param x - initial guess, overwritten by the solution; before entry, x should be set to some value, typical x = 0; in gpu memory
 * @param r - residual vector in gpu memory
 * @param p - search direction vector
 * @param q - matrix-vector product in gpu memory
 * @param tol - tolerance
 * @param maxit - maximum number of iterations
 */
int double_cgcsrls(const int n, const csr_t &matrix, const double *rhs, double *x, double *r, double *p, double *q, const double tol, const double maxit, int blocks, int threads){
    //variables
    double epsilon = 0;     //stopping critera
    double norm0 = 0;       //norm 2 of the initial residual, r0
    double normk = 0;       //norm 2 of the kth residual, rk
    double rho = 0;         //rho_{k-1}
    double rhop = 0;        //previous rho, rho_{k-2}
    double alpha = 0;
    double beta = 0;
    double pqDot = 0;
    int k = 0;
    //Initialize cuBLAS
    cublasStatus_t stat;
    cublasHandle_t handle;
    stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("cuBLAS initialization failed\n");
        return 0;
    }
    //1: compute initial residual r = b − Ax (using initial guess in x)
    double_spmv_csr_kernel<<<blocks, threads>>>(n, matrix, x, r);
    double_diff_kernel<<<blocks, threads>>>(n, rhs, r, r);
    //compute  and update epsilon
    cublasDnrm2(handle, n, r, 1, &norm0); // initial residual norm, norm0 = ||r||
    epsilon = norm0 * tol; //stopping criteria (based on relative residual)
    normk = norm0; //to exit if r = 0
    //2: repeat until convergence (based on max. it. and relative residual)
    while(normk > epsilon && k < maxit){
        rhop = rho;
        cublasDdot(handle, n, r, 1, r, 1, &rho); //rho = r^{T} r
        if(k == 0){
            double_copy_kernel<<<blocks, threads>>>(n, r, p); //p = r
        } else{
            beta = rho / rhop; //beta = rho_{k-1} / rho_{k-2}
            double_axpy_kernel<<<blocks, threads>>>(n, beta, p, r, p); // pk = r_{k−1} + β_{k−1} * p_{k−1}
        }
        double_spmv_csr_kernel<<<blocks, threads>>>(n, matrix, p, q);
        cublasDdot(handle, n, p, 1, q, 1, &pqDot); //pqDot = p^{T} q
        alpha = rho / pqDot;
        double_axpy_kernel<<<blocks, threads>>>(n, alpha, p, x, x); // xk = x_{k−1} + α_k * p_k
        double_axpy_kernel<<<blocks, threads>>>(n, -alpha, q, r, r); // rk = r_{k−1} − α_k * q_k
        cublasDnrm2(handle, n, r, 1, &normk); // current residual norm,
        k++;
    }
    //3: cleanup
    cublasDestroy (handle);
    return k;
}

void test_double_cgcsrls(const int n, const int nnz, const csr_t &matrix, const double * rhs, const double tol, const double maxit) {
    InitCudaTimer();
    Logger::printDash();
    PrintTimerName("[GPU] double CG solver (CSR)");

    //Execution configuration
    int threads = 32;
    int blocks = n / threads + 1;
    printf("\tExec. config: blocks = %i, threads = %i\n", blocks, threads);

    //Host data
    auto *hx = new double[n]; //Initial guess
    for(int i = 0; i < n; i++){
        hx[i] = 0;
    }

    // GPU vectors
    double *drhs;
    double *dx;
    double *dr;
    double *dp;
    double *dq;
    cudaMalloc(&drhs, sizeof(double) * n);
    cudaMalloc(&dx, sizeof(double) * n);
    cudaMalloc(&dr, sizeof(double) * n);
    cudaMalloc(&dp, sizeof(double) * n);
    cudaMalloc(&dq, sizeof(double) * n);
    cudaMemcpy(dx, hx, sizeof(double) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(drhs, rhs, sizeof(double) * n, cudaMemcpyHostToDevice);

    //GPU matrix
    csr_t dmatrix;
    cuda::csr_init(dmatrix, n, nnz);
    cuda::csr_host2device(dmatrix, matrix, n, nnz);

    //Launch
    StartCudaTimer();
    int iters = double_cgcsrls(n, dmatrix, drhs, dx, dr, dp, dq, tol, maxit, blocks, threads)
    EndCudaTimer();
    PrintCudaTimer("took");
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    //Copying to the host
    cudaMemcpy(hx, dx, sizeof(double) * n, cudaMemcpyDeviceToHost);
    std::cout << "iterations: " << iters << std::endl;
    print_residual(n, matrix, hx, rhs);

    delete [] hx;
    cudaFree(drhs);
    cudaFree(dx);
    cudaFree(dr);
    cudaFree(dp);
    cudaFree(dq);
    cuda::csr_clear(dmatrix);
}

#endif //TEST_DOUBLE_CGCSRLS_CUH
