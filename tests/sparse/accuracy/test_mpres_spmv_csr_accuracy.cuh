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

#ifndef TEST_MPRES_SPMV_CSR_ACCURACY_CUH
#define TEST_MPRES_SPMV_CSR_ACCURACY_CUH

#include "../../tsthelper.cuh"
#include "sparse/spmv/spmv_csr.cuh"


//returns result in y
void test_mpres_spmv_csr_accuracy(const int m, const int n, const int nnz, const int maxnzr, const csr_t &csr, const double * x, mpfr_t *y){

    //Execution configuration
    int threads = 32;
    int blocks = m / threads + 1;

    // Host data
    auto hx = new mp_float_t[n];
    auto hy = new mp_float_t[m];

    // GPU vectors
    mp_float_ptr dx;
    mp_float_ptr dy;
    cudaMalloc(&dx, sizeof(mp_float_t) * n);
    cudaMalloc(&dy, sizeof(mp_float_t) * m);
    convert_vector(hx, x, n);
    cudaMemcpy(dx, hx, n * sizeof(mp_float_t), cudaMemcpyHostToDevice);

    //GPU matrix
    csr_t dcsr;
    cuda::csr_init(dcsr, m, nnz);
    cuda::csr_host2device(dcsr, csr, m, nnz);

    //Launch
    cuda::mp_spmv_csr<32><<<blocks, threads>>>(m, dcsr, dx, dy);
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    //Set output vector
    cudaMemcpy(hy, dy, m * sizeof(mp_float_t), cudaMemcpyDeviceToHost);
    for(int i = 0; i < m; i++){
        mp_get_mpfr(y[i], hy[i]); //actual result
    }
    //Cleanup
    delete [] hx;
    delete [] hy;
    cudaFree(dx);
    cudaFree(dy);
    cuda::csr_clear(dcsr);
}

#endif //TEST_MPRES_SPMV_CSR_ACCURACY_CUH