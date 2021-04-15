/*
 *  Performance test for the CAMPARY library SpMV routine campary_mpspmv_csr_vector_kernel (multiple precision matrix)
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

#ifndef TEST_CAMPARY_MPSPMV_CSR_VECTOR_CUH
#define TEST_CAMPARY_MPSPMV_CSR_VECTOR_CUH

#include "tsthelper.cuh"
#include "logger.cuh"
#include "timers.cuh"
#include "3rdparty/campary_common.cuh"

/*
 * Performs the matrix-vector operation y = A * x where x and y are dense vectors and A is a sparse matrix.
 * Vector kernel - a group of threads (up to 32 threads) are assigned to each row of the matrix, i.e. one element of the vector y.
 * The matrix should be stored in the CSR format: entries are stored in a dense array in column major order and explicit zeros are stored if necessary (zero padding)
 */
template<int prec, int threadsPerRow>
__global__ void campary_mpspmv_csr_vector_kernel(const int m, const int *irp, const int *ja, const multi_prec<prec> *as, const multi_prec<prec> *x, multi_prec<prec> *y) {
    extern __shared__ multi_prec<prec> vals[];

    auto threadId = threadIdx.x + blockIdx.x * blockDim.x; // global thread index
    auto groupId = threadId / threadsPerRow; // global thread group index
    auto lane = threadId & (threadsPerRow - 1); // thread index within the group
    auto row = groupId; // one group per row
    while (row < m) {
        multi_prec<prec> prod;
        int row_start = irp[row];
        int row_end = irp[row + 1];
        // compute running sum per thread
        vals[threadIdx.x] = 0.0;
        for (auto i = row_start + lane; i < row_end; i += threadsPerRow) {
            prod = x[ja[i]] * as[i];
            vals[threadIdx.x] += prod;
        }
        // parallel reduction in shared memory
        if (threadsPerRow >= 32 && lane < 16) {
            vals[threadIdx.x] += vals[threadIdx.x + 16];
        }
        if (threadsPerRow >= 16 && lane < 8) {
            vals[threadIdx.x] += vals[threadIdx.x + 8];
        }
        if (threadsPerRow >= 8 && lane < 4) {
            vals[threadIdx.x] += vals[threadIdx.x + 4];
        }
        if (threadsPerRow >= 4 && lane < 2) {
            vals[threadIdx.x] += vals[threadIdx.x + 2];
        }
        if (threadsPerRow >= 2 && lane < 1) {
            vals[threadIdx.x] += vals[threadIdx.x + 1];
        }
        // first thread writes the result
        if (lane == 0) {
            y[row] = vals[threadIdx.x];
        }
        row +=  gridDim.x * blockDim.x / threadsPerRow;
    }
}


template<int prec, int threadsPerRow>
void test_campary_mpspmv_csr_vector(const int m, const int n, const int nnz, const int *irp, const int *ja, const double *as,  mpfr_t *x, const int convert_prec) {
    Logger::printDash();
    InitCudaTimer();
    PrintTimerName("[GPU] CAMPARY SpMV CSR vector (multiple precision matrix)");

    //Execution configuration
    const int threads = 256;
    const int blocks = m / (threads/threadsPerRow) + 1;
    //int blocks = 32;
    printf("\tThreads per row = %i\n", threadsPerRow);
    printf("\tExec. config: blocks = %i, threads = %i\n", blocks, threads);
    printf("\tMatrix (AS array) size (MB): %lf\n", get_campary_array_size_in_mb<prec>(nnz));

    //Host data
    auto *hx = new multi_prec<prec>[n];
    auto *hy = new multi_prec<prec>[m];
    auto *has = new multi_prec<prec>[nnz];

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
        has[i] = as[i];
    }

    //Copying to the GPU
    cudaMemcpy(dx, hx, sizeof(multi_prec<prec>) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(das, has, sizeof(multi_prec<prec>) * nnz, cudaMemcpyHostToDevice);
    cudaMemcpy(dirp, irp, sizeof(int) * (m + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(dja, ja, sizeof(int) * nnz, cudaMemcpyHostToDevice);
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    //Launch
    StartCudaTimer();
    campary_mpspmv_csr_vector_kernel<prec, threadsPerRow><<<blocks, threads, sizeof(multi_prec<prec>) * threads>>>(m, dirp, dja, das, dx, dy);
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
    delete [] has;
    cudaFree(dx);
    cudaFree(dy);
    cudaFree(dirp);
    cudaFree(dja);
    cudaFree(das);
}

#endif //TEST_CAMPARY_MPSPMV_CSR_VECTOR_CUH
