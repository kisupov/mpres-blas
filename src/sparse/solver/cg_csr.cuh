/*
 *  Multiple precision CG iterative method for solving linear systems using the CSR matrix storage format
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


#ifndef MPRES_CG_CSR_CUH
#define MPRES_CG_CSR_CUH

#include "arith/assign.cuh"
#include "arith/div.cuh"
#include "blas//v2/diff_v2.cuh"
#include "blas/v2/norm2_v2.cuh"
#include "blas/v2/dot_v2.cuh"
#include "blas/v2/axpy_v2.cuh"
#include "blas/v2/maxpy_v2.cuh"
#include "sparse/spmv/spmv_csr.cuh"

/**
 * Multiple precision conjugate gradient linear solver (Ax = b) using CSR for storing the matrix
 * TODO: Current timings is suboptimal, for convergence and numerical robustness studies only
 * ASSUMPTIONS:
 * - The appropriate GPU memory for x has been allocated and set to zero.
 * - The relative residual or maximum number of iterations are used as a stopping criteria: ||r_k|| <= ||r_0|| * tol or k > maxiter
 * @tparam threads - number of threads fo cuda kernels
 * @tparam blocks_reduce - number of blocks for norm2 and dot product (reduction operations)
 * @param n - operation size
 * @param A - sparse double-precision matrix in the CSR storage format
 * @param b - right-hand side vector in gpu memory
 * @param tol - tolerance
 * @param maxit - maximum number of iterations
 * @param x - initial residual and linear system solution
 * @param resvec - residual error returned as vector (residual history)
 * @return number of iterations
 */
template <int threads = 64, int blocks_reduce = 256>
int mp_cg_csr(const int n, const csr_t &A, mp_float_ptr b, const double tol, const int maxit, mp_float_ptr x, std::vector<double> resvec){
    const int blocks = n / threads + 1;
    //Variables
    double epsilon = 0; //stopping criteria based on relative residual
    double norm0 = 0;   //norm 2 of the initial residual, r0
    double normk = 0;   //norm 2 of the kth residual, rk
    mp_float_ptr r;     //residual vector
    mp_float_ptr p;     //search direction vector
    mp_float_ptr q;     //matrix-vector product
    mp_float_ptr rho;   //rho_{k-1}
    mp_float_ptr rhop;  //previous rho, rho_{k-2}
    mp_float_ptr alpha; //scalar
    mp_float_ptr beta;  //scalar
    mp_float_ptr pqDot; //dot product
    int k = 0;          //iters
    //Allocate memory
    cudaMalloc(&r, sizeof(mp_float_t) * n);
    cudaMalloc(&p, sizeof(mp_float_t) * n);
    cudaMalloc(&q, sizeof(mp_float_t) * n);
    cudaMalloc(&rho, sizeof(mp_float_t));
    cudaMalloc(&rhop, sizeof(mp_float_t));
    cudaMalloc(&alpha, sizeof(mp_float_t));
    cudaMalloc(&beta, sizeof(mp_float_t));
    cudaMalloc(&pqDot, sizeof(mp_float_t));
    //Initial residual r = b − Ax (using initial guess in x):
    cuda::mp_spmv_csr<threads><<<blocks, threads>>>(n, A, x, r);
    cuda::mp_diff<<<blocks, threads>>>(n, b, r, r);
    //Initial residual norm ||r||
    norm0 = cuda::mp_norm2<blocks_reduce, threads>(n, r);
    //Calc stopping criteria
    epsilon = norm0 * tol;
    //In order to terminate when r = 0:
    normk = norm0;
    resvec.push_back(normk/norm0);
    //Repeat until convergence based on relative residual or maxit reached
    while(normk > epsilon && k < maxit){
        checkDeviceHasErrors(cudaMemcpy(rhop, rho, sizeof(mp_float_t), cudaMemcpyDeviceToDevice)); //rhop = rho
        cuda::mp_dot<blocks_reduce,threads>(n, r, r, rho); //rho = r^{T} r
        if(k == 0){
            checkDeviceHasErrors(cudaMemcpy(p, r, sizeof(mp_float_t) * n, cudaMemcpyDeviceToDevice)); //p = r
        } else{
            cuda::mp_div(beta, rho, rhop); //beta = rho_{k-1} / rho_{k-2}
            cuda::mp_axpy<<<blocks, threads>>>(n, beta, p, r, p); // pk = r_{k−1} + β_{k−1} * p_{k−1}
        }
        cuda::mp_spmv_csr<threads><<<blocks, threads>>>(n, A, p, q);
        cuda::mp_dot<blocks_reduce,threads>(n, p, q, pqDot); //pqDot = p^{T} q
        cuda::mp_div(alpha, rho, pqDot); //alpha = rho / pqDot
        cuda::mp_axpy<<<blocks, threads>>>(n, alpha, p, x, x); // xk = x_{k−1} + α_k * p_k
        cuda::mp_maxpy<<<blocks_reduce, threads>>>(n, alpha, q, r, r); // rk = r_{k−1} − α_k * q_k
        normk = cuda::mp_norm2<blocks_reduce, threads>(n, r); // current (implicit) residual norm
        resvec.push_back(normk/norm0); //update residual history
        k++;
    }
    cudaFree(r);
    cudaFree(p);
    cudaFree(q);
    cudaFree(rho);
    cudaFree(rhop);
    cudaFree(alpha);
    cudaFree(beta);
    cudaFree(pqDot);
    return k;
}

#endif //MPRES_CG_CSR_CUH
