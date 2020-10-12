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

#ifndef MPRES_TEST_CAMPARY_BLAS_CUH
#define MPRES_TEST_CAMPARY_BLAS_CUH

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
__global__ void campary_spmv_ell_kernel(int num_rows, int num_cols_per_row, int *indices, multi_prec<prec> *data, multi_prec<prec> *x, multi_prec<prec> *y) {
    unsigned int threadId = threadIdx.x + blockIdx.x * blockDim.x;
    if (threadId < num_rows) {
        multi_prec<prec> dot = 0.0;
        for (int colId = 0; colId < num_cols_per_row; colId++) {
            int index = indices[colId * num_rows + threadId];
            if(index >= 0){
                dot += data[colId * num_rows + threadId] * x[index];
            }
        }
        y[threadId] = dot;
    }
}


/********************* Benchmarks *********************/

/*
 * SpMV ELLPACK test
 */
template<int prec>
void campary_spmv_ell_test(const int num_rows, const int num_cols, const int num_cols_per_row,  double const *data, int const *indices, mpfr_t *x, int convert_prec) {
    Logger::printDash();
    InitCudaTimer();
    PrintTimerName("[GPU] CAMPARY SpMV ELLPACK");

    //Execution configuration
    int BLOCKS = num_rows / CAMPARY_VECTOR_MULTIPLY_THREADS + 1;

    //Host data
    multi_prec<prec> *hx = new multi_prec<prec>[num_cols];
    multi_prec<prec> *hy = new multi_prec<prec>[num_rows];
    multi_prec<prec> *hdata = new multi_prec<prec>[num_rows * num_cols_per_row];

    //GPU data
    multi_prec<prec> *dx;
    multi_prec<prec> *dy;
    multi_prec<prec> *ddata;
    int *dindices = new int[num_rows * num_cols_per_row];

    cudaMalloc(&dx, sizeof(multi_prec<prec>) * num_cols);
    cudaMalloc(&dy, sizeof(multi_prec<prec>) * num_rows);
    cudaMalloc(&ddata, sizeof(multi_prec<prec>) * num_rows * num_cols_per_row);
    cudaMalloc(&dindices, sizeof(int) * num_rows * num_cols_per_row);

    //Convert from MPFR
    #pragma omp parallel for
    for(int i = 0; i < num_cols; i++){
        hx[i] = convert_to_string_sci(x[i], convert_prec).c_str();
    }

    //Convert from double
    #pragma omp parallel for
    for(int i = 0; i < num_rows * num_cols_per_row; i++){
        hdata[i] = data[i];
    }

    //Copying to the GPU
    cudaMemcpy(dx, hx, sizeof(multi_prec<prec>) * num_cols, cudaMemcpyHostToDevice);
    cudaMemcpy(ddata, hdata, sizeof(multi_prec<prec>) * num_rows * num_cols_per_row, cudaMemcpyHostToDevice);
    cudaMemcpy(dindices, indices, sizeof(int) * num_rows * num_cols_per_row, cudaMemcpyHostToDevice);
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    //Launch
    StartCudaTimer();
    campary_spmv_ell_kernel<prec><<<BLOCKS, CAMPARY_VECTOR_MULTIPLY_THREADS>>>(num_rows, num_cols_per_row, dindices, ddata, dx, dy);
    EndCudaTimer();
    PrintCudaTimer("took");
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    //Copying to the host
    cudaMemcpy(hy, dy, sizeof(multi_prec<prec>) * num_rows, cudaMemcpyDeviceToHost);
    for(int i = 1; i < num_rows; i++){
        hy[0] += hy[i];
    }
    printResult<prec>(hy[0]);

    //Cleanup
    delete [] hx;
    delete [] hy;
    delete [] hdata;
    cudaFree(dx);
    cudaFree(dy);
    cudaFree(ddata);
    cudaFree(dindices);
}

#endif //MPRES_TEST_CAMPARY_BLAS_CUH