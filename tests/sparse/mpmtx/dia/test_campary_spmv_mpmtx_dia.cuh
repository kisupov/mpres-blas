/*
 *  Performance test for the CAMPARY library SpMV routine campary_spmv_mpmtx_dia_kernel (multiple precision matrix)
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

#ifndef TEST_CAMPARY_SPMV_MPMTX_DIA_CUH
#define TEST_CAMPARY_SPMV_MPMTX_DIA_CUH

#include "../../../tsthelper.cuh"
#include "../../../logger.cuh"
#include "../../../timers.cuh"
#include "../../../lib/campary_common.cuh"

/*
 * Performs the matrix-vector operation y = A * x
 * where x and y are dense vectors and A is a sparse matrix.
 * The matrix should be stored in the DIA format: entries are stored in a dense array in column major order and explicit zeros are stored if necessary (zero padding)
 */
template<int prec>
__global__ void campary_spmv_mpmtx_dia_kernel(const int m, const int n, const int ndiag, const int *offset, const multi_prec<prec> *as, const multi_prec<prec> *x, multi_prec<prec> *y) {
    unsigned int row = threadIdx.x + blockIdx.x * blockDim.x;
    if (row < m) {
        multi_prec<prec> dot = 0.0;
        for (int i = 0; i < ndiag; i++) {
            int col = row + offset[i];
            multi_prec<prec> val = as[m * i + row];
            if(col  >= 0 && col < n)
                dot += val * x[col];
        }
        y[row] = dot;
    }
}

template<int prec>
void test_campary_spmv_mpmtx_dia(const int m, const int n, const int ndiag, const int *offset, const double *as, mpfr_t *x, const int convert_prec) {
    Logger::printDash();
    InitCudaTimer();
    PrintTimerName("[GPU] CAMPARY SpMV DIA");

    //Execution configuration
    int threads = 32;
    int blocks = m / threads + 1;
    printf("(exec. config: blocks = %i, threads = %i)\n", blocks, threads);
    printf("\tMatrix (AS array) size (MB): %lf\n", get_campary_array_size_in_mb<prec>(m * ndiag));

    //Host data
    multi_prec<prec> *hx = new multi_prec<prec>[n];
    multi_prec<prec> *hy = new multi_prec<prec>[m];
    multi_prec<prec> *has = new multi_prec<prec>[m * ndiag];

    //GPU data
    multi_prec<prec> *dx;
    multi_prec<prec> *dy;
    multi_prec<prec> *das;
    int *doffset = new int[ndiag];

    cudaMalloc(&dx, sizeof(multi_prec<prec>) * n);
    cudaMalloc(&dy, sizeof(multi_prec<prec>) * m);
    cudaMalloc(&das, sizeof(multi_prec<prec>) * m * ndiag);
    cudaMalloc(&doffset, sizeof(int) * ndiag);

    //Convert from MPFR
    #pragma omp parallel for
    for(int i = 0; i < n; i++){
        hx[i] = convert_to_string_sci(x[i], convert_prec).c_str();
    }
    //Convert from double
    #pragma omp parallel for
    for(int i = 0; i < m * ndiag; i++){
        has[i] = as[i];
    }

    //Copying to the GPU
    cudaMemcpy(dx, hx, sizeof(multi_prec<prec>) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(das, has, sizeof(multi_prec<prec>) * m * ndiag, cudaMemcpyHostToDevice);
    cudaMemcpy(doffset, offset, sizeof(int) * ndiag, cudaMemcpyHostToDevice);
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    //Launch
    StartCudaTimer();
    campary_spmv_mpmtx_dia_kernel<prec><<<blocks, threads>>>(m, n, ndiag, doffset, das, dx, dy);
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
    cudaFree(doffset);
}

#endif //TEST_CAMPARY_SPMV_MPMTX_DIA_CUH
