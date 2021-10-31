/*
 *  Unpreconditioned double precision CG iterative method for solving linear systems using the CSR matrix storage format on CPU
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


#ifndef DOUBLE_CG_CSR_CPU_CUH
#define DOUBLE_CG_CSR_CPU_CUH

#include "double_ls_kernels_cpu.cuh"

/**
 * Unpreconditioned double precision conjugate gradient linear solver, Ax = b
 * We use the relative residual as a stopping criteria: ||r_k|| <= ||r_0|| * tol or k > maxiter
 * @param n - operation size
 * @param matrix - matrix A
 * @param rhs - right-hand side vector b
 * @param x - initial guess, overwritten by the solution; before entry, x should be set to some value, typical x = 0
 */
int double_cg_csr_cpu(const int n, const csr_t &matrix, double *rhs, double *x){
    //variables
    double tol = 1e-12;     //tolerance
    double epsilon = 0;     //stopping critera
    double maxit = 40000;   //maximum number of iterations
    double norm0 = 0;       //norm 2 of the initial residual, r0
    double normk = 0;       //norm 2 of the kth residual, rk
    double rho = 0;         //rho_{k-1}
    double rhop = 0;        //previous rho, rho_{k-2}
    double alpha = 0;
    double beta = 0;
    double pqDot = 0;
    int k = 0;
    double *r = new double[n]; //residual vector
    double *p = new double[n]; //search direction vector
    double *q = new double[n]; //matrix-vector product

    //1: compute initial residual r = b − Ax (using initial guess in x)
    double_spmv_csr(n, matrix, x, r); //r = Ax
    double_diff(n, rhs, r, r); //r = b - r
    //compute  and update epsilon
    norm0 = double_norm2(n, r); // initial residual norm, ||r||
    epsilon = norm0 * tol; //stopping criteria (based on relative residual)
    normk = norm0; //to exit if r = 0

    //2: repeat until convergence (based on max. it. and relative residual)
    while(normk > epsilon && k < maxit){
        rhop = rho;
        rho = double_dot(n, r, r); //rho = r^{T} r
        if(k == 0){
            double_copy(n, r, p); //p = r
        } else{
            beta = rho / rhop; //beta = rho_{k-1} / rho_{k-2}
            double_axpy(n, beta, p, r, p); // pk = r_{k−1} + β_{k−1} * p_{k−1}
        }
        double_spmv_csr(n,matrix,p,q);
        pqDot = double_dot(n, p, q);
        alpha = rho / pqDot;
        double_axpy(n, alpha, p, x, x); // xk = x_{k−1} + α_k * p_k
        double_axpy(n, -alpha, q, r, r); // rk = r_{k−1} − α_k * q_k
        normk = double_norm2(n, r);
        k++;
    }

    //3: cleanup
    delete [] r;
    delete [] p;
    delete [] q;
    return k;
}


#endif //DOUBLE_CG_CSR_CPU_CUH
