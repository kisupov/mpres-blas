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
__global__ void cump_spmv_ellpack_kernel(int num_rows, int num_cols_per_row, mpf_array_t data, int *indices, mpf_array_t x, mpf_array_t y, mpf_array_t tmp1) {
    using namespace cump;
    int indice = 0;
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    for (int j = 0; j < num_cols_per_row; j++) {
        if( i < num_rows ){
            indice = indices[j * num_rows + i];
            mpf_mul(tmp1[i], x[indice], data[j * num_rows + i]);
            mpf_add(y[i], y[i], tmp1[i]);
        }
    }
}

/********************* Benchmarks *********************/

/*
 * SpMV ELLPACK test
 */
void cump_spmv_ellpack_test(int num_rows, int num_cols, int num_cols_per_row, mp_float_ptr data, int *indices, mp_float_ptr x, mp_float_ptr y, int prec, int convert_digits, int repeats){
    Logger::printDash();
    InitCudaTimer();
    PrintTimerName("[GPU] CUMP SpMV ELLPACK");

    //Set precision
    mpf_set_default_prec(prec);
    cumpf_set_default_prec(prec);

    //Execution configuration
    int threads = 64;
    //todo тут точно так?
    int blocks_spmv = num_rows / (threads) + (num_rows % (threads) ? 1 : 0);

    //Host data
    mpf_t *hdata = new mpf_t[num_rows * num_cols_per_row];
    mpf_t *hx = new mpf_t[num_cols];
    mpf_t *hy = new mpf_t[num_rows];

    //mpfr_data for conversion mp_float -> mpfr -> string_sci
    mpfr_t *mpfrX = new mpfr_t[num_cols];
    mpfr_t *mpfrY = new mpfr_t[num_rows];
    mpfr_t *mpfrData = new mpfr_t[num_rows * num_cols_per_row];

    //GPU data
    cumpf_array_t ddata;
    cumpf_array_t dx;
    cumpf_array_t dy;
    cumpf_array_t dtemp;
    int *dindices = new int[num_rows * num_cols_per_row];


    cumpf_array_init2(ddata, num_rows * num_cols_per_row, prec);
    cumpf_array_init2(dx, num_cols, prec);
    cumpf_array_init2(dy, num_rows, prec);
    cumpf_array_init2(dtemp, num_rows, prec);
    cudaMalloc(&dindices, sizeof(int) * num_rows * num_cols_per_row);

    //Convert from MPFR
    for(int i = 0; i < num_rows * num_cols_per_row; i++){
        mpfr_init2(mpfrData[i], prec);
        mp_get_mpfr(mpfrData[i], &data[i]);
        mpf_init2(hdata[i], prec);
        mpf_set_str(hdata[i], convert_to_string_sci(mpfrData[i], convert_digits).c_str(), 10);
    }
    for(int i = 0; i < num_cols; i++){
        mpfr_init2(mpfrX[i], prec);
        mp_get_mpfr(mpfrX[i], &x[i]);
        mpf_init2(hx[i], prec);
        mpf_set_str(hx[i], convert_to_string_sci(mpfrX[i], convert_digits).c_str(), 10);
    }
    for(int i = 0; i < num_rows; i++){
        mpfr_init2(mpfrY[i], prec);
        mp_get_mpfr(mpfrY[i], &y[i]);
        mpf_init2(hy[i], prec);
        mpf_set_str(hy[i], convert_to_string_sci(mpfrY[i], convert_digits).c_str(), 10);
    }
    //Copying to the GPU

    cumpf_array_set_mpf(dx, hx, num_cols);
    cumpf_array_set_mpf(ddata, hdata, num_rows * num_cols_per_row);
    cudaMemcpy(dindices, indices, sizeof(int) * num_rows * num_cols_per_row, cudaMemcpyHostToDevice);

    //Launch
    for(int i = 0; i < repeats; i++){
        cumpf_array_set_mpf(dy, hy, num_rows);
        cudaDeviceSynchronize();
        StartCudaTimer();
        cump_spmv_ellpack_kernel<<<blocks_spmv, threads>>>(num_rows, num_cols_per_row, ddata, dindices, dx, dy, dtemp);
        EndCudaTimer();
    }
    PrintCudaTimer("took");

    //Copying to the host
    mpf_array_set_cumpf(hy, dy, num_rows);
    for(int i = 1; i < num_rows; i++){
        mpf_add(hy[0], hy[i], hy[0]);
    }
    gmp_printf ("result: %.70Ff \n", hy[0]);

    //Cleanup
    for(int i = 0; i < num_rows * num_cols_per_row; i++){
        mpf_clear(hdata[i]);
        mpfr_clear(mpfrData[i]);
    }
    for(int i = 0; i < num_cols; i++){
        mpf_clear(hx[i]);
        mpfr_clear(mpfrX[i]);
    }
    for(int i = 0; i < num_rows; i++){
        mpf_clear(hy[i]);
        mpfr_clear(mpfrY[i]);
    }
    delete [] hdata;
    delete [] hx;
    delete [] hy;
    delete [] mpfrData;
    delete [] mpfrX;
    delete [] mpfrY;
    cumpf_array_clear(ddata);
    cumpf_array_clear(dx);
    cumpf_array_clear(dy);
    cumpf_array_clear(dtemp);
    cudaFree(dindices);
}

#endif //MPRES_TEST_CUMP_SPARSE_CUH