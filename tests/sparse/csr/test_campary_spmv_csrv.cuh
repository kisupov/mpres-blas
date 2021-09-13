/*
 *  Performance test for the CAMPARY library SpMV routine campary_spmv_csr_kernel (double precision matrix)
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

#ifndef TEST_CAMPARY_SPMV_CSRV_CUH
#define TEST_CAMPARY_SPMV_CSRV_CUH

#include "../../tsthelper.cuh"
#include "../../logger.cuh"
#include "../../timers.cuh"
#include "../../lib/campary_common.cuh"

/*
 * Performs the matrix-vector operation y = A * x where x and y are dense vectors and A is a sparse matrix.
 * Vector kernel - a group of threads (up to 32 threads) are assigned to each row of the matrix, i.e. one element of the vector y.
 * The matrix is in double precision
 * The matrix should be stored in the CSR format: entries are stored in a dense array in column major order and explicit zeros are stored if necessary (zero padding)
 */
template<int threads, int threadsPerRow, int prec>
__global__ void campary_spmv_csrv_kernel(const int m, const csr_t csr, const multi_prec<prec> *x, multi_prec<prec> *y) {
    __shared__ multi_prec<prec> sums[threads];
    __shared__ multi_prec<prec> prods[threads];
    auto threadId = threadIdx.x + blockIdx.x * blockDim.x; // global thread index
    auto groupId = threadId / threadsPerRow; // global thread group index
    auto lane = threadId & (threadsPerRow - 1); // thread index within the group
    auto row = groupId; // one group per row
    while (row < m) {
        int row_start = csr.irp[row];
        int row_end = csr.irp[row + 1];
        // compute running sum per thread
        sums[threadIdx.x] = 0.0;
        for (auto i = row_start + lane; i < row_end; i += threadsPerRow) {
            prods[threadIdx.x] = x[csr.ja[i]] * csr.as[i];
            sums[threadIdx.x] += prods[threadIdx.x];
        }
        // parallel reduction in shared memory
        if (threadsPerRow >= 32 && lane < 16) {
            sums[threadIdx.x] = sums[threadIdx.x] + sums[threadIdx.x + 16];
        }
        if (threadsPerRow >= 16 && lane < 8) {
            sums[threadIdx.x] = sums[threadIdx.x] + sums[threadIdx.x + 8];
        }
        if (threadsPerRow >= 8 && lane < 4) {
            sums[threadIdx.x] = sums[threadIdx.x] + sums[threadIdx.x + 4];
        }
        if (threadsPerRow >= 4 && lane < 2) {
            sums[threadIdx.x] = sums[threadIdx.x] + sums[threadIdx.x + 2];
        }
        if (threadsPerRow >= 2 && lane < 1) {
            sums[threadIdx.x] = sums[threadIdx.x] + sums[threadIdx.x + 1];
        }
        // first thread writes the result
        if (lane == 0) {
            y[row] = sums[threadIdx.x];
        }
        row +=  gridDim.x * blockDim.x / threadsPerRow;
    }
}

template<int prec, int threadsPerRow>
void test_campary_spmv_csrv(const int m, const int n, const int nnz, const csr_t &csr, mpfr_t *x, const int convert_prec) {
    Logger::printDash();
    InitCudaTimer();
    PrintTimerName("[GPU] CAMPARY SpMV CSR Vector (double precision matrix)");

    //Execution configuration
    const int threads = 32;
    const int blocks = m / (threads/threadsPerRow) + 1;
    printf("\tThreads per row = %i\n", threadsPerRow);

    //Host data
    auto *hx = new multi_prec<prec>[n];
    auto *hy = new multi_prec<prec>[m];

    //GPU vectors
    multi_prec<prec> *dx;
    multi_prec<prec> *dy;
    cudaMalloc(&dx, sizeof(multi_prec<prec>) * n);
    cudaMalloc(&dy, sizeof(multi_prec<prec>) * m);
    #pragma omp parallel for
    for(int i = 0; i < n; i++){
        hx[i] = convert_to_string_sci(x[i], convert_prec).c_str();
    }
    cudaMemcpy(dx, hx, sizeof(multi_prec<prec>) * n, cudaMemcpyHostToDevice);

    //GPU matrix
    csr_t dcsr;
    cuda::csr_init(dcsr, m, nnz);
    cuda::csr_host2device(dcsr, csr, m, nnz);

    //Launch
    StartCudaTimer();
    campary_spmv_csrv_kernel<32, threadsPerRow, prec><<<blocks, threads>>>(m, dcsr, dx, dy);
    EndCudaTimer();
    PrintCudaTimer("took");
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
    cudaFree(dx);
    cudaFree(dy);
    cuda::csr_clear(dcsr);
}

#endif //TEST_CAMPARY_SPMV_CSRV_CUH
