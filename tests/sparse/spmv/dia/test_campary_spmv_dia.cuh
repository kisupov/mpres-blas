/*
 *  Performance test for the CAMPARY library SpMV routine campary_spmv_dia_kernel (double precision matrix)
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

#ifndef TEST_CAMPARY_SPMV_DIA_CUH
#define TEST_CAMPARY_SPMV_DIA_CUH

#include "tsthelper.cuh"
#include "logger.cuh"
#include "timers.cuh"
#include "lib/campary_common.cuh"

/*
 * Performs the matrix-vector operation y = A * x
 * where x and y are dense vectors and A is a sparse matrix.
 * The matrix should be stored in the DIA format: entries are stored in a dense array in column major order and explicit zeros are stored if necessary (zero padding)
 */
template<int threads, int prec>
__global__ void campary_spmv_dia_kernel(const int m, const int n, const int ndiag, const dia_t dia, const multi_prec<prec> *x, multi_prec<prec> *y) {
    int row = threadIdx.x + blockIdx.x * blockDim.x;
    __shared__ multi_prec<prec> sums[threads];
    __shared__ multi_prec<prec> prods[threads];
    if (row < m) {
        sums[threadIdx.x] = 0.0;
        for (int i = 0; i < ndiag; i++) {
            int j = row + dia.offset[i];
            double val = dia.as[m * i + row];
            if(j  >= 0 && j < n && val != 0)
                prods[threadIdx.x] = val * x[j];
                sums[threadIdx.x] += prods[threadIdx.x];
        }
        y[row] = sums[threadIdx.x];
    }
}

template<int prec>
void test_campary_spmv_dia(const int m, const int n, const int ndiag, const dia_t &dia, mpfr_t *x, const int convert_prec) {
    Logger::printDash();
    InitCudaTimer();
    PrintTimerName("[GPU] CAMPARY SpMV DIA (double precision matrix)");

    //Execution configuration
    int threads = 32;
    int blocks = m / threads + 1;
    printf("Exec. config: blocks = %i, threads = %i\n", blocks, threads);

    //Host data
    multi_prec<prec> *hx = new multi_prec<prec>[n];
    multi_prec<prec> *hy = new multi_prec<prec>[m];

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
    dia_t ddia;
    cuda::dia_init(ddia, m, ndiag);
    cuda::dia_host2device(ddia, dia, m, ndiag);

    //Launch
    StartCudaTimer();
    campary_spmv_dia_kernel<32, prec><<<blocks, threads>>>(m, n, ndiag, ddia, dx, dy);
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
    cudaFree(dx);
    cudaFree(dy);
    cuda::dia_clear(ddia);
}

#endif //TEST_CAMPARY_SPMV_DIA_CUH
