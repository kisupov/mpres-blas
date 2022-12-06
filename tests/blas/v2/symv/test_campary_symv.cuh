/*
 *  Performance test for CAMPARY SYMV
 *  http://homepages.laas.fr/mmjoldes/campary/
 *
 *  Copyright 2021 by Konstantin Isupov.
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
#ifndef TEST_CAMPARY_SYMV_CUH
#define TEST_CAMPARY_SYMV_CUH

#include "logger.cuh"
#include "timers.cuh"
#include "tsthelper.cuh"
#include "lib/campary_common.cuh"
#include "blas/mblas_enum.cuh"

template<int prec>
__global__ static void campary_symv_kernel(enum mblas_uplo_type uplo, int n, multi_prec<prec> *alpha, multi_prec<prec> *A, int lda, multi_prec<prec> *x, const int incx, multi_prec<prec> *beta, multi_prec<prec> *y, const int incy) {
    auto row = threadIdx.x + blockIdx.x * blockDim.x;
    auto iy = incy > 0 ? row * incy : (-n + row + 1)*incy;
    //Each thread works with its own row, i.e. goes through the columns
    while (row < n) {
        multi_prec<prec> dot;
        dot = 0.0;
        if (uplo == mblas_upper) { //Use the upper part of the matrix
            for (int colId = 0; colId < n; colId++) {
                auto ix = incx > 0 ? colId * incx : (-n + colId + 1) * incx;
                multi_prec<prec> dx = x[ix];
                if(row <= colId){
                    dot += dx * A[row + colId * lda];
                } else{
                    dot += dx * A[colId + row * lda];
                }
            }
        } else{ //Use the lower part of the matrix
            for (int colId = 0; colId < n; colId++) {
                auto ix = incx > 0 ? colId * incx : (-n + colId + 1) * incx;
                multi_prec<prec> dx = x[ix];
                if(row <= colId){
                    dot += dx * A[colId + row * lda];
                } else{
                    dot += dx * A[row + colId * lda];
                }
            }
        }
        y[iy] = beta * y[iy] + alpha[0] * dot;
        row +=  gridDim.x * blockDim.x;
        iy += gridDim.x * blockDim.x * incy;
    }
}

template<int prec>
void test_campary_symv(enum mblas_uplo_type uplo, const int n, int lenx, int leny, mpfr_t alpha, mpfr_t *A, int lda, mpfr_t *x, int incx, mpfr_t beta, mpfr_t *y, int incy, const int convert_prec, const int repeats) {
    InitCudaTimer();
    Logger::printDash();
    PrintTimerName("[GPU] CAMPARY symv");

    //Execution configuration
    int threads = 32;
    int blocks = n / threads + 1;
    printf("\tExec. config: blocks = %i, threads = %i\n", blocks, threads);

    //Host data
    multi_prec<prec> halpha;
    multi_prec<prec> hbeta;
    multi_prec<prec> *hx = new multi_prec<prec>[lenx];
    multi_prec<prec> *hy = new multi_prec<prec>[leny];
    multi_prec<prec> *hA = new multi_prec<prec>[lda * n];

    // GPU data
    multi_prec<prec> *dalpha;
    multi_prec<prec> *dbeta;
    multi_prec<prec> *dx;
    multi_prec<prec> *dy;
    multi_prec<prec> *dA;

    cudaMalloc(&dx, sizeof(multi_prec<prec>) * lenx);
    cudaMalloc(&dy, sizeof(multi_prec<prec>) * leny);
    cudaMalloc(&dA, sizeof(multi_prec<prec>) * lda * n);
    cudaMalloc(&dalpha, sizeof(multi_prec<prec>));
    cudaMalloc(&dbeta, sizeof(multi_prec<prec>));

    //Convert from MPFR
    #pragma omp parallel for
    for(int i = 0; i < lenx; i ++){
        hx[i] = convert_to_string_sci(x[i], convert_prec).c_str();
    }
    #pragma omp parallel for
    for(int i = 0; i < leny; i ++){
        hy[i] = convert_to_string_sci(y[i], convert_prec).c_str();
    }
    #pragma omp parallel for
    for(int i = 0; i < lda * n; i ++){
        hA[i] = convert_to_string_sci(A[i], convert_prec).c_str();
    }
    halpha = convert_to_string_sci(alpha, convert_prec).c_str();
    hbeta = convert_to_string_sci(beta, convert_prec).c_str();

    //Copying to the GPU
    cudaMemcpy(dx, hx, sizeof(multi_prec<prec>) * lenx, cudaMemcpyHostToDevice);
    cudaMemcpy(dA, hA, lda * n * sizeof(multi_prec<prec>), cudaMemcpyHostToDevice);
    cudaMemcpy(dalpha, &halpha, sizeof(multi_prec<prec>), cudaMemcpyHostToDevice);
    cudaMemcpy(dbeta, &hbeta, sizeof(multi_prec<prec>), cudaMemcpyHostToDevice);

    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    for(int i = 0; i < repeats; i ++) {
        cudaMemcpy(dy, hy, sizeof(multi_prec<prec>) * leny, cudaMemcpyHostToDevice);
        StartCudaTimer();
        campary_symv_kernel<<<blocks, threads>>>(uplo, n, dalpha, dA, lda, dx, incx, dbeta, dy, incy);
        EndCudaTimer();
    }
    PrintAndResetCudaTimer("took");
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    //Copying to the host
    cudaMemcpy(hy, dy, sizeof(multi_prec<prec>) * leny, cudaMemcpyDeviceToHost);
    for(int i = 1; i < leny; i ++){
        hy[0] += hy[i];
    }
    printResult<prec>(hy[0]);

    //Cleanup
    delete [] hx;
    delete [] hy;
    delete [] hA;
    cudaFree(dx);
    cudaFree(dy);
    cudaFree(dA);
    cudaFree(dalpha);
    cudaFree(dbeta);
}
#endif //TEST_CAMPARY_SYMV_CUH
