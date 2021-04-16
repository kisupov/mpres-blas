/*
 *  Performance test for the CAMPARY library SpMV routine campary_mpspmv_ellpack_kernel (double precision matrix)
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

#ifndef TEST_CAMPARY_MPDSPMV_JAD_CUH
#define TEST_CAMPARY_MPDSPMV_JAD_CUH

#include "../../../../tsthelper.cuh"
#include "../../../../logger.cuh"
#include "../../../../timers.cuh"
#include "../../../../3rdparty/campary_common.cuh"

/*
 * Performs the matrix-vector operation y = A * x where x and y are dense vectors and A is a sparse matrix.
 * The matrix is in multiple precision
 * The matrix should be stored in the ELLPACK format: entries are stored in a dense array in column major order and explicit zeros are stored if necessary (zero padding)
 */
template<int prec>
__global__ void campary_mpdspmv_jad_kernel(const int m, const int maxnz, const int *ja, const double *as, const int *jcp, const int *perm_rows, const multi_prec<prec> *x, multi_prec<prec> *y) {
    auto row = threadIdx.x + blockIdx.x * blockDim.x;
    while (row < m) {
        multi_prec<prec> dot = 0.0;
        auto j = 0;
        auto index = row;
        while (j < maxnz && index < jcp[j + 1]) {
            dot += as[index] * x[ja[index]];
            index = row + jcp[++j];
        }
        y[perm_rows[row]] = dot;
        row +=  gridDim.x * blockDim.x;
    }
}

template<int prec>
void test_campary_mpdspmv_jad(const int m, const int n, const int maxnz, const int nnz, const int *ja, const int *jcp, const double *as, const int *perm_rows, mpfr_t *x, const int convert_prec) {
    Logger::printDash();
    InitCudaTimer();
    PrintTimerName("[GPU] CAMPARY SpMV JAD (JDS) (double precision matrix)");

    //Execution configuration
    int threads = 32;
    int blocks = m / threads + 1;
    printf("\tExec. config: blocks = %i, threads = %i\n", blocks, threads);

    //Memory requirements
    double sizeOfAs = get_double_array_size_in_mb(nnz);
    double sizeOfJad = sizeOfAs + get_int_array_size_in_mb(nnz) + get_int_array_size_in_mb(maxnz + 1)+ get_int_array_size_in_mb(m);
    double sizeOfVectors = get_campary_array_size_in_mb<prec>(m + n);
    printf("\tMatrix (AS array) size (MB): %lf\n", sizeOfAs);
    printf("\tJAD structure size (MB): %lf\n", sizeOfJad);
    printf("\tVectors x and y size (MB): %lf\n", sizeOfVectors);
    printf("\tTOTAL Memory Consumption (MB): %lf\n", sizeOfJad + sizeOfVectors);

    //Host data
    multi_prec<prec> *hx = new multi_prec<prec>[n];
    multi_prec<prec> *hy = new multi_prec<prec>[m];

    //GPU data
    multi_prec<prec> *dx;
    multi_prec<prec> *dy;
    double *das;
    int *dja;
    int *djcp;
    int *dperm_rows;

    cudaMalloc(&dx, sizeof(multi_prec<prec>) * n);
    cudaMalloc(&dy, sizeof(multi_prec<prec>) * m);
    cudaMalloc(&das, sizeof(double) * nnz);
    cudaMalloc(&dja, sizeof(int) * nnz);
    cudaMalloc(&djcp, sizeof(int) * (maxnz + 1));
    cudaMalloc(&dperm_rows, sizeof(int) * m);

    //Convert from MPFR
    #pragma omp parallel for
    for(int i = 0; i < n; i++){
        hx[i] = convert_to_string_sci(x[i], convert_prec).c_str();
    }

    //Copying to the GPU
    cudaMemcpy(dx, hx, sizeof(multi_prec<prec>) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(das, as, sizeof(double) * nnz, cudaMemcpyHostToDevice);
    cudaMemcpy(dja, ja, sizeof(int) * nnz, cudaMemcpyHostToDevice);
    cudaMemcpy(djcp, jcp, sizeof(int) * (maxnz + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(dperm_rows, perm_rows, sizeof(int) * m, cudaMemcpyHostToDevice);
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    //Launch
    StartCudaTimer();
    campary_mpdspmv_jad_kernel<prec><<<blocks, threads>>>(m, maxnz, dja, das, djcp, dperm_rows, dx, dy);
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
    cudaFree(das);
    cudaFree(dja);
    cudaFree(djcp);
    cudaFree(dperm_rows);
}

#endif //TEST_CAMPARY_MPDSPMV_JAD_CUH
