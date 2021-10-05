/*
 *  Unpreconditioned multiple precision CG iterative method for solving linear systems using the CSR matrix storage format
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


#ifndef MPRES_CGCSRLS_H
#define MPRES_CGCSRLS_H

#include "arith/assign.cuh"
#include "arith/div.cuh"
#include "blas//v2/diff_v2.cuh"
#include "blas/v2/norm2_v2.cuh"
#include "blas/v2/dot_v2.cuh"
#include "blas/v2/axpy_v2.cuh"
#include "blas/v2/maxpy_v2.cuh"
#include "sparse/spmv/spmv_csr.cuh"

//Execution configuration
#define THREADS 64
#define BLOCKS_SCALAR 256 //for norm2 and dot product

/**
 * TODO: Current timings is suboptimal, for convergence studies only
 * Unpreconditioned multiple precision conjugate gradient linear solver (Ax = b) using CSR for storing the matrix
 *
 * @note ASSUMPTIONS: The appropriate GPU memory for `matrix`, `rhs`, `x`, `r`, `p`, and `q` has been allocated and set to zero.
 * @note The relative residual or maximum number of iterations are used as a stopping criteria: ||r_k|| <= ||r_0|| * tol or k > maxiter
 * @param n - operation size
 * @param matrix - sparse double-precision matrix in the CSR storage format
 * @param rhs - right-hand side vector b in gpu memory
 * @param x - initial guess, overwritten by the solution; before entry, x should be set to some value, typical x = 0; in gpu memory
 * @param r - residual vector in gpu memory
 * @param p - search direction vector
 * @param q - matrix-vector product in gpu memory
 * @param tol - tolerance
 * @param maxit - maximum number of iterations
 */
int mp_cgcsrls(const int n, const csr_t &matrix, mp_float_ptr rhs, mp_float_ptr x, mp_float_ptr r, mp_float_ptr p, mp_float_ptr q, const double tol, const double maxit){
    const int BLOCKS = n / THREADS + 1;
    //Variables
    double epsilon = 0;     //stopping criteria based on relative residual
    double norm0 = 0;       //norm 2 of the initial residual, r0
    double normk = 0;       //norm 2 of the kth residual, rk
    mp_float_ptr rho;       //rho_{k-1}
    mp_float_ptr rhop;      //previous rho, rho_{k-2}
    mp_float_ptr alpha;
    mp_float_ptr beta;
    mp_float_ptr pqDot;
    int k = 0;
    //Init host memory
    cudaMalloc(&rho, sizeof(mp_float_t));
    cudaMalloc(&rhop, sizeof(mp_float_t));
    cudaMalloc(&alpha, sizeof(mp_float_t));
    cudaMalloc(&beta, sizeof(mp_float_t));
    cudaMalloc(&pqDot, sizeof(mp_float_t));

    //1: Compute initial residual r = b − Ax (using initial guess in x)
    cuda::mp_spmv_csr<THREADS><<<BLOCKS, THREADS>>>(n, matrix, x, r);
    cuda::mp_diff<<<BLOCKS, THREADS>>>(n, rhs, r, r);
    //2: Compute initial residual norm ||r|| and update epsilon (stopping criteria )
    norm0 = cuda::mp_norm2<BLOCKS_SCALAR, THREADS>(n, r);
    epsilon = norm0 * tol;
    normk = norm0; //in order to terminate when r = 0
    //3: Repeat until convergence (based on max. it. and relative residual)
    while(normk > epsilon && k < maxit){
        checkDeviceHasErrors(cudaMemcpy(rhop, rho, sizeof(mp_float_t), cudaMemcpyDeviceToDevice)); //rhop = rho
        cuda::mp_dot<BLOCKS_SCALAR,THREADS>(n, r, r, rho); //rho = r^{T} r
        if(k == 0){
            checkDeviceHasErrors(cudaMemcpy(p, r, sizeof(mp_float_t) * n, cudaMemcpyDeviceToDevice)); //p = r
        } else{
            cuda::mp_div(beta, rho, rhop); //beta = rho_{k-1} / rho_{k-2}
            cuda::mp_axpy<<<BLOCKS, THREADS>>>(n, beta, p, r, p); // pk = r_{k−1} + β_{k−1} * p_{k−1}
        }
        cuda::mp_spmv_csr<THREADS><<<BLOCKS, THREADS>>>(n, matrix, p, q);
        cuda::mp_dot<BLOCKS_SCALAR,THREADS>(n, p, q, pqDot); //pqDot = p^{T} q
        cuda::mp_div(alpha, rho, pqDot); //alpha = rho / pqDot
        cuda::mp_axpy<<<BLOCKS, THREADS>>>(n, alpha, p, x, x); // xk = x_{k−1} + α_k * p_k
        cuda::mp_maxpy<<<BLOCKS, THREADS>>>(n, alpha, q, r, r); // rk = r_{k−1} − α_k * q_k
        normk = cuda::mp_norm2<BLOCKS_SCALAR, THREADS>(n, r); // current residual norm
        k++;
    }
    //4: Cleanup
    cudaFree(rho);
    cudaFree(rhop);
    cudaFree(alpha);
    cudaFree(beta);
    cudaFree(pqDot);
    return k;
}

#endif //MPRES_CGCSRLS_H
