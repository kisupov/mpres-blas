/*
 *  Multiple-precision sparse linear algebra kernels using CAMPARY as well as corresponding performance benchmarks.
 *
 *  Copyright 2020 by Konstantin Isupov and Ivan Babeshko.
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

#ifndef MPRES_TEST_CAMPARY_SPARSE_CUH
#define MPRES_TEST_CAMPARY_SPARSE_CUH

#include "tsthelper.cuh"
#include "logger.cuh"
#include "timers.cuh"
#include "3rdparty/campary_common.cuh"

/********************* Computational kernels *********************/

/*
 * Performs the matrix-vector operation y = A * x
 * where x and y are dense vectors and A is a sparse matrix.
 * The matrix should be stored in the ELLPACK format: entries are stored in a dense array in column major order and explicit zeros are stored if necessary (zero padding)
 */
template<int prec>
__global__ void campary_spmv_ell_kernel(const int m, const int nzr, const int *ja, const multi_prec<prec> *as, const multi_prec<prec> *x, multi_prec<prec> *y) {
    unsigned int row = threadIdx.x + blockIdx.x * blockDim.x;
    if (row < m) {
        multi_prec<prec> dot = 0.0;
        for (int col = 0; col < nzr; col++) {
            int index = ja[col * m + row];
            if(index >= 0){
                dot += as[col * m + row] * x[index];
            }
        }
        y[row] = dot;
    }
}

/*
 * Performs the matrix-vector operation y = A * x
 * where x and y are dense vectors and A is a sparse matrix.
 * The matrix should be stored in the CSR format: entries are stored in a dense array in column major order and explicit zeros are stored if necessary (zero padding)
 */
template<int prec>
__global__ void campary_spmv_csr_kernel(const int m, const int *irp, const int *ja, const multi_prec<prec> *as, const multi_prec<prec> *x, multi_prec<prec> *y) {
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


/********************* Benchmarks *********************/

/*
 * SpMV ELLPACK test
 */
template<int prec>
void campary_spmv_ell_test(const int m, const int n, const int nzr, const int *ja, const double *as,  mpfr_t *x, const int convert_prec) {
    Logger::printDash();
    InitCudaTimer();
    PrintTimerName("[GPU] CAMPARY SpMV ELLPACK");

    //Execution configuration
    int threads = 32;
    int blocks = m / threads + 1;
    printf("(exec. config: blocks = %i, threads = %i)\n", blocks, threads);

    //Host data
    multi_prec<prec> *hx = new multi_prec<prec>[n];
    multi_prec<prec> *hy = new multi_prec<prec>[m];
    multi_prec<prec> *has = new multi_prec<prec>[m * nzr];

    //GPU data
    multi_prec<prec> *dx;
    multi_prec<prec> *dy;
    multi_prec<prec> *das;
    int *dja = new int[m * nzr];

    cudaMalloc(&dx, sizeof(multi_prec<prec>) * n);
    cudaMalloc(&dy, sizeof(multi_prec<prec>) * m);
    cudaMalloc(&das, sizeof(multi_prec<prec>) * m * nzr);
    cudaMalloc(&dja, sizeof(int) * m * nzr);

    //Convert from MPFR
    #pragma omp parallel for
    for(int i = 0; i < n; i++){
        hx[i] = convert_to_string_sci(x[i], convert_prec).c_str();
    }
    //Convert from double
    #pragma omp parallel for
    for(int i = 0; i < m * nzr; i++){
        has[i] = as[i];
    }

    //Copying to the GPU
    cudaMemcpy(dx, hx, sizeof(multi_prec<prec>) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(das, has, sizeof(multi_prec<prec>) * m * nzr, cudaMemcpyHostToDevice);
    cudaMemcpy(dja, ja, sizeof(int) * m * nzr, cudaMemcpyHostToDevice);
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    //Launch
    StartCudaTimer();
    campary_spmv_ell_kernel<prec><<<blocks, threads>>>(m, nzr, dja, das, dx, dy);
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
    cudaFree(das);
    cudaFree(dja);
}

/*
 * SpMV CSR test
 */
template<int prec>
void campary_spmv_csr_test(const int m, const int n, const int nnz, const int *irp, const int *ja, const double *as,  mpfr_t *x, const int convert_prec) {
    Logger::printDash();
    InitCudaTimer();
    PrintTimerName("[GPU] CAMPARY SpMV CSR");

    //Execution configuration
    int threads = 32;
    int blocks = m / threads + 1;
    printf("(exec. config: blocks = %i, threads = %i)\n", blocks, threads);

    //Host data
    multi_prec<prec> *hx = new multi_prec<prec>[n];
    multi_prec<prec> *hy = new multi_prec<prec>[m];
    multi_prec<prec> *has = new multi_prec<prec>[nnz];

    //GPU data
    multi_prec<prec> *dx;
    multi_prec<prec> *dy;
    multi_prec<prec> *das;
    int *dirp = new int[m + 1];
    int *dja = new int[nnz];

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
    campary_spmv_csr_kernel<prec><<<blocks, threads>>>(m, dirp, dja, das, dx, dy);
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

#endif //MPRES_TEST_CAMPARY_SPARSE_CUH
