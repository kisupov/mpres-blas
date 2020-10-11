/*
 *  Multiple-precision sparse linear algebra kernels using CUMP as well as corresponding performance benchmarks.
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

#ifndef MPRES_TEST_CUMP_SPARSE_CUH
#define MPRES_TEST_CUMP_SPARSE_CUH

#include <stdio.h>
#include "mpfr.h"
#include "../../../src/mblas_enum.cuh"
#include "../../tsthelper.cuh"
#include "../../logger.cuh"
#include "../../timers.cuh"
#include "cump/cump.cuh"

#define CUMP_MAX_THREADS_PER_BLOCK 1024
using cump::mpf_array_t;

/********************* Computational kernels *********************/

/*
 * Performs the matrix-vector operation y = A * x
 * where x and y are dense vectors and A is a sparse matrix.
 * The matrix should be stored in the ELLPACK format: entries are stored in a dense array in column major order and explicit zeros are stored if necessary (zero padding)
 */
__global__ void cump_spmv_ell_kernel(int num_rows, int num_cols_per_row, int *indices, mpf_array_t data, mpf_array_t x, mpf_array_t y, mpf_array_t tmp) {
    using namespace cump;
    unsigned int threadId = threadIdx.x + blockIdx.x * blockDim.x;
    if( threadId < num_rows ) {
        for (int colId = 0; colId < num_cols_per_row; colId++) {
            int index = indices[colId * num_rows + threadId];
            if(index >= 0){
                mpf_mul(tmp[threadId], x[index], data[colId * num_rows + threadId]);
                mpf_add(y[threadId], y[threadId], tmp[threadId]);
            }
        }
    }
}

/********************* Benchmarks *********************/

/*
 * SpMV ELLPACK test
 */
void cump_spmv_ell_test(const int num_rows, const int num_cols, const int num_cols_per_row,  double const  * data, int const *indices,  mpfr_t *x, int prec, int convert_digits){
    Logger::printDash();
    InitCudaTimer();
    PrintTimerName("[GPU] CUMP SpMV ELLPACK");

    //Set precision
    mpf_set_default_prec(prec);
    cumpf_set_default_prec(prec);

    //Execution configuration
    int threads = 64;
    int blocks = num_rows / (threads) + (num_rows % (threads) ? 1 : 0);

    //Host data
    mpf_t *hx = new mpf_t[num_cols];
    mpf_t *hy = new mpf_t[num_rows];
    mpf_t *hdata = new mpf_t[num_rows * num_cols_per_row];

    //GPU data
    cumpf_array_t dx;
    cumpf_array_t dy;
    cumpf_array_t ddata;
    cumpf_array_t dtemp;
    int *dindices = new int[num_rows * num_cols_per_row];

    cumpf_array_init2(dx, num_cols, prec);
    cumpf_array_init2(dy, num_rows, prec);
    cumpf_array_init2(ddata, num_rows * num_cols_per_row, prec);
    cumpf_array_init2(dtemp, num_rows, prec);
    cudaMalloc(&dindices, sizeof(int) * num_rows * num_cols_per_row);

    //Convert from MPFR
    for(int i = 0; i < num_cols; i++){
        mpf_init2(hx[i], prec);
        mpf_set_str(hx[i], convert_to_string_sci(x[i], convert_digits).c_str(), 10);
    }
    //Init output vector
    for(int i = 0; i < num_rows; i++){
        mpf_init2(hy[i], prec);
        mpf_set_d(hy[i], 0.0);
    }
    //Convert from double
    for(int i = 0; i < num_rows * num_cols_per_row; i++){
        mpf_init2(hdata[i], prec);
        mpf_set_d(hdata[i], data[i]);
    }

    //Copying to the GPU
    cumpf_array_set_mpf(dx, hx, num_cols);
    cumpf_array_set_mpf(dy, hy, num_rows);
    cumpf_array_set_mpf(ddata, hdata, num_rows * num_cols_per_row);
    cudaMemcpy(dindices, indices, sizeof(int) * num_rows * num_cols_per_row, cudaMemcpyHostToDevice);

    //Launch
    StartCudaTimer();
    cump_spmv_ell_kernel<<<blocks, threads>>>(num_rows, num_cols_per_row, dindices, ddata, dx, dy, dtemp);
    EndCudaTimer();
    PrintCudaTimer("took");

    //Copying to the host
    mpf_array_set_cumpf(hy, dy, num_rows);
    for(int i = 1; i < num_rows; i++){
        mpf_add(hy[0], hy[i], hy[0]);
    }
    gmp_printf ("result: %.70Ff \n", hy[0]);

    //Cleanup
    for(int i = 0; i < num_cols; i++){
        mpf_clear(hx[i]);
    }
    for(int i = 0; i < num_rows; i++){
        mpf_clear(hy[i]);
    }
    for(int i = 0; i < num_rows * num_cols_per_row; i++){
        mpf_clear(hdata[i]);
    }
    delete [] hx;
    delete [] hy;
    delete [] hdata;
    cumpf_array_clear(dx);
    cumpf_array_clear(dy);
    cumpf_array_clear(ddata);
    cumpf_array_clear(dtemp);
    cudaFree(dindices);
}

#endif //MPRES_TEST_CUMP_SPARSE_CUH