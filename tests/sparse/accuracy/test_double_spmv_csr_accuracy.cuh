/*
 *  Accuracy test for the MPRES-BLAS library SpMV routine mp_spmv_csr (double precision matrix)
 *
 *  Copyright 2020 by Konstantin Isupov.
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

#ifndef TEST_DOUBLE_SPMV_CSR_ACCURACY_CUH
#define TEST_DOUBLE_SPMV_CSR_ACCURACY_CUH

#include "../../tsthelper.cuh"
#include "sparse/spmv/spmv_csr.cuh"

__global__ static void double_spmv_csr_kernel(const int m, const csr_t csr, const double *x, double *y) {
    unsigned int row = threadIdx.x + blockIdx.x * blockDim.x;
    if(row < m){
        double dot = 0;
        int row_start = csr.irp[row];
        int row_end = csr.irp[row+1];
        for (int i = row_start; i < row_end; i++) {
            dot += csr.as[i] * x[csr.ja[i]];
        }
        y[row] = dot;
    }
}

//returns result in y
void test_double_spmv_csr_accuracy(const int m, const int n, const int nnz, const int maxnzr, const csr_t &csr, const double * x, mpfr_t *y){

    //Execution configuration
    int threads = 32;
    int blocks = m / threads + 1;

    // Host data
    auto *hy = new double[m];

    // GPU vectors
    double *dx;
    double *dy;
    cudaMalloc(&dx, sizeof(double) * n);
    cudaMalloc(&dy, sizeof(double) * m);
    cudaMemcpy(dx, x, sizeof(double) * n, cudaMemcpyHostToDevice);

    //GPU matrix
    csr_t dcsr;
    cuda::csr_init(dcsr, m, nnz);
    cuda::csr_host2device(dcsr, csr, m, nnz);

    //Launch
    double_spmv_csr_kernel<<<blocks, threads>>>(m, dcsr, dx, dy);
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    //Set output vector
    cudaMemcpy(hy, dy, sizeof(double) * m , cudaMemcpyDeviceToHost);
    for(int i = 0; i < m; i++){
        mpfr_set_d(y[i], hy[i], MPFR_RNDN);
    }
    //Cleanup
    delete [] hy;
    cudaFree(dx);
    cudaFree(dy);
    cuda::csr_clear(dcsr);
}

#endif //TEST_DOUBLE_SPMV_CSR_ACCURACY_CUH