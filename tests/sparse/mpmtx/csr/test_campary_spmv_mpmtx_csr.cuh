/*
 *  Performance test for the CAMPARY library SpMV routine campary_spmv_mpmtx_csr_kernel (multiple precision matrix)
 *  http://homepages.laas.fr/mmjoldes/campary/
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

#ifndef TEST_CAMPARY_SPMV_MPMTX_CSR_CUH
#define TEST_CAMPARY_SPMV_MPMTX_CSR_CUH

#include "../../../tsthelper.cuh"
#include "../../../logger.cuh"
#include "../../../timers.cuh"
#include "../../../lib/campary_common.cuh"

/*
 * Performs the matrix-vector operation y = A * x where x and y are dense vectors and A is a sparse matrix.
 * The matrix is in multiple precision
 * Scalar kernel - one thread is assigned to each row of the matrix, i.e. one element of the vector y.
 * The matrix should be stored in the CSR format: entries are stored in a dense array in column major order and explicit zeros are stored if necessary (zero padding)
 */
template<int prec>
__global__ void campary_spmv_mpmtx_csr_kernel(const int m, const int *irp, const int *ja, const multi_prec<prec> *as, const multi_prec<prec> *x, multi_prec<prec> *y) {
    unsigned int row = threadIdx.x + blockIdx.x * blockDim.x;
    if (row < m) {
        multi_prec<prec> dot = 0.0;
        int row_start = irp[row];
        int row_end = irp[row + 1];
        for (int i = row_start; i < row_end; i++) {
            dot += as[i] * x[ja[i]];
        }
        y[row] = dot;
    }
}

template<int prec>
void test_campary_spmv_mpmtx_csr(const int m, const int n, const int nnz, const csr_t &csr, mpfr_t *x, const int convert_prec) {
    Logger::printDash();
    InitCudaTimer();
    PrintTimerName("[GPU] CAMPARY SpMV CSR scalar (multiple precision matrix)");

    //Execution configuration
    int threads = 32;
    int blocks = m / threads + 1;
    printf("\tExec. config: blocks = %i, threads = %i\n", blocks, threads);
    printf("\tMatrix (AS array) size (MB): %lf\n", get_campary_array_size_in_mb<prec>(nnz));

    //Host data
    multi_prec<prec> *hx = new multi_prec<prec>[n];
    multi_prec<prec> *hy = new multi_prec<prec>[m];
    multi_prec<prec> *has = new multi_prec<prec>[nnz];

    //GPU data
    multi_prec<prec> *dx;
    multi_prec<prec> *dy;
    multi_prec<prec> *das;
    int *dirp;
    int *dja;

    cudaMalloc(&dx, sizeof(multi_prec<prec>) * n);
    cudaMalloc(&dy, sizeof(multi_prec<prec>) * m);
    cudaMalloc(&das, sizeof(multi_prec<prec>) * nnz);
    cudaMalloc(&dirp, sizeof(int) * (m + 1));
    cudaMalloc(&dja, sizeof(int) * nnz);

    //Convert from MPFR
    #pragma omp parallel for
    for(int i = 0; i < n; i++){
        hx[i] = convert_to_string_sci(x[i], convert_prec).c_str();
    }
    //Convert from double
    #pragma omp parallel for
    for(int i = 0; i < nnz; i++){
        has[i] = csr.as[i];
    }

    //Copying to the GPU
    cudaMemcpy(dx, hx, sizeof(multi_prec<prec>) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(das, has, sizeof(multi_prec<prec>) * nnz, cudaMemcpyHostToDevice);
    cudaMemcpy(dirp, csr.irp, sizeof(int) * (m + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(dja, csr.ja, sizeof(int) * nnz, cudaMemcpyHostToDevice);
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    //Launch
    StartCudaTimer();
    campary_spmv_mpmtx_csr_kernel<prec><<<blocks, threads>>>(m, dirp, dja, das, dx, dy);
    EndCudaTimer();
    PrintAndResetCudaTimer("took");
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    //Copying to the host
    cudaMemcpy(hy, dy, sizeof(multi_prec<prec>) * m, cudaMemcpyDeviceToHost);
    for(int i = 1; i < m; i++){
        hy[0] += hy[i];
    }
    printResult<prec>(hy[0]);

    //Cleanup
    delete [] hx;
    delete [] hy;
    delete [] has;
    cudaFree(dx);
    cudaFree(dy);
    cudaFree(dirp);
    cudaFree(dja);
    cudaFree(das);
}


#endif //TEST_CAMPARY_SPMV_MPMTX_CSR_CUH
