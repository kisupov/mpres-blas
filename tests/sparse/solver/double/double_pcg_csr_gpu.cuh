/*
 *  Preconditioned double precision CG iterative method for solving linear systems using the CSR matrix storage format
 *
 *  Copyright 2021 by Konstantin Isupov
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


#ifndef DOUBLE_PCG_CSR_GPU_CUH
#define DOUBLE_PCG_CSR_GPU_CUH

#include "double_ls_kernels_gpu.cuh"
#include "sparse/msparse_enum.cuh"

/////////
// Preconditioned double precision CG iterative method using diagonal (Jacobi) preconditioner
/////////
/**
 * Preconditioned double precision conjugate gradient linear solver (Ax = b) using CSR for storing the matrix
 * ASSUMPTIONS:
 * - The appropriate GPU memory for x has been allocated and set to zero.
 * - The relative residual or maximum number of iterations are used as a stopping criteria: ||r_k|| <= ||r_0|| * tol or k > maxiter
 * - When the diagonal (Jacobi) preconditioner is used (ptype = diag), M should contain the INVERTED main diagonal of A
 * @tparam threads - number of threads fo cuda kernels
 * @tparam blocks_reduce - number of blocks for norm2 and dot product (reduction operations)
 * @param n - operation size
 * @param A - sparse double-precision matrix in the CSR storage format
 * @param b - right-hand side vector in gpu memory
 * @param tol - tolerance
 * @param maxit - maximum number of iterations
 * @param ptype - type of preconditioner
 * @param M - preconditioner matrix
 * @param x - initial residual and linear system solution
 * @param resvec - residual error, returned as vector (residual history)
 * @return number of iterations
 */
int double_pcg_csr(const int n, const csr_t &A, double * b, const double tol, const int maxit, enum preconditioner_type ptype, double *M, double * x, vector<double> &resvec, int blocks, int threads){
    //variables
    double *r;
    double *p;
    double *q;
    double *z;
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
    //Allocate memory
    cudaMalloc(&r, sizeof(double) * n);
    cudaMalloc(&p, sizeof(double) * n);
    cudaMalloc(&q, sizeof(double) * n);
    cudaMalloc(&z, sizeof(double) * n);

    //1: compute initial residual r = b − Ax (using initial guess in x)
    double_spmv_csr_kernel<<<blocks, threads>>>(n, A, x, r);
    double_diff_kernel<<<blocks, threads>>>(n, b, r, r);
    //compute  and update epsilon
    cublasDnrm2(handle, n, r, 1, &norm0); // initial residual norm, norm0 = ||r||
    epsilon = norm0 * tol; //stopping criteria (based on relative residual)
    normk = norm0; //to exit if r = 0
    resvec.push_back(normk/norm0);
    //2: repeat until convergence (based on max. it. and relative residual)
    while(normk > epsilon && k < maxit){
        //3: preconditioning
        if(ptype == diag){
            //Solve Mz = r -> z = M^{-1} r and this is simple component-wise vector product.
            //In this case, it is assumed that M already contains the reciprocals of the main diagonal of A.
            double_prod_kernel<<<blocks, threads>>>(n, r, M, z);
        }
        rhop = rho;
        cublasDdot(handle, n, r, 1, z, 1, &rho); //rho = r^{T} z
        if(k == 0){
            double_copy_kernel<<<blocks, threads>>>(n, z, p); //p = z
        } else{
            beta = rho / rhop; //beta = rho_{k-1} / rho_{k-2}
            double_axpy_kernel<<<blocks, threads>>>(n, beta, p, z, p); // pk = z_{k−1} + β_{k−1} * p_{k−1}
        }
        double_spmv_csr_kernel<<<blocks, threads>>>(n, A, p, q);
        cublasDdot(handle, n, p, 1, q, 1, &pqDot); //pqDot = p^{T} q
        alpha = rho / pqDot;
        double_axpy_kernel<<<blocks, threads>>>(n, alpha, p, x, x); // xk = x_{k−1} + α_k * p_k
        double_axpy_kernel<<<blocks, threads>>>(n, -alpha, q, r, r); // rk = r_{k−1} − α_k * q_k
        cublasDnrm2(handle, n, r, 1, &normk); // current residual norm
        resvec.push_back(normk/norm0);
        k++;
    }
    //3: cleanup
    cublasDestroy (handle);
    cudaFree(r);
    cudaFree(p);
    cudaFree(q);
    cudaFree(z);
    return k;
}

#endif //DOUBLE_PCG_CSR_GPU_CUH
