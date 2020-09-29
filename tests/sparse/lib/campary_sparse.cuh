/*
 *  Multiple-precision BLAS routines using CAMPARY as well as corresponding performance benchmarks.
 *
 *  Copyright 2018, 2019 by Konstantin Isupov and Alexander Kuvaev.
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

#include <stdio.h>
#include "mpfr.h"
#include "../../../src/params.h"
#include "../../../src/mblas_enum.cuh"
#include "../../tsthelper.cuh"
#include "../../logger.cuh"
#include "../../timers.cuh"
#include "../../3rdparty/campary/Doubles/src_gpu/multi_prec.h"

/*
 * Precision of CAMPARY in n-double
 * For predefined RNS moduli sets from the src/32-bit-n-double-moduli/ directory:
 * 8 moduli give 2-double, 16 moduli give 4-double, 24 moduli give 6-double, etc.
 */
#define CAMPARY_PRECISION (RNS_MODULI_SIZE / 4)

//Execution configuration
#define CAMPARY_REDUCTION_BLOCKS 1024
#define CAMPARY_REDUCTION_THREADS 32
#define CAMPARY_VECTOR_MULTIPLY_THREADS 32
#define CAMPARY_MATRIX_THREADS_X 32
#define CAMPARY_MATRIX_THREADS_Y 8



/********************* Computational kernels *********************/

/*
 * Performs the matrix-vector operation  y := A*x + beta*y,
 * where beta is a scalar, x and y are vectors and A is an m by n matrix
 */
template<int prec>
__global__ void campary_spmv_ellpack_kernel(int num_rows, int num_cols_per_row, multi_prec<prec> *data, int *indices, multi_prec<prec> *x, multi_prec<prec> *y) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int indice = 0;
    for (int j = 0; j < num_cols_per_row; j++) {
        if( i < num_rows ){
            indice = indices[j * num_rows + i];
            //TODO опять же суммирование (нехорошо)
            y[i] = y[i] + data[j * num_rows + i] * x[indice];
        }
        __syncthreads();
    }
    __syncthreads();
}


/********************* Benchmarks *********************/

// Printing the result, which is a CAMPARY's floating-point expansion (ONE multiple precision number)
// prec specifies the number of terms (precision), i.e. the size of the floating point expansion
template<int nterms>
static void printResult(multi_prec<nterms> result){
    int p = 8192;
    mpfr_t x;
    mpfr_t r;
    mpfr_init2(x, p);
    mpfr_init2(r, p);
    mpfr_set_d(r, 0.0, MPFR_RNDN);
    for(int i = nterms - 1; i >= 0; i--){
        mpfr_set_d(x, result.getData()[i], MPFR_RNDN);
        mpfr_add(r, r, x, MPFR_RNDN);
    }
    mpfr_printf("result: %.70Rf \n", r);
    /* printf("RAW Data:\n");
    result.prettyPrint(); */
    mpfr_clear(x);
    mpfr_clear(r);
}

/*
 * SpMV ELLPACK test
 */
template<int prec>
void campary_spmv_ellpack_test(int num_rows, int num_cols, int num_cols_per_row, mp_float_ptr data, int *indices,
                               mp_float_ptr x, mp_float_ptr y, int convert_prec, int repeats) {
    Logger::printDash();
    InitCudaTimer();
    PrintTimerName("[GPU] CAMPARY SpMV ELLPACK");

    //Execution configuration
    int BLOCKS = num_rows / CAMPARY_VECTOR_MULTIPLY_THREADS + 1;

    //Host data

    multi_prec<prec> *hx = new multi_prec<prec>[num_cols];
    multi_prec<prec> *hy = new multi_prec<prec>[num_rows];
    multi_prec<prec> *hdata = new multi_prec<prec>[num_rows * num_cols_per_row];

    //mpfr_data for conversion mp_float -> mpfr -> string_sci
    mpfr_t *mpfrX = new mpfr_t[num_cols];
    mpfr_t *mpfrY = new mpfr_t[num_rows];
    mpfr_t *mpfrData = new mpfr_t[num_rows * num_cols_per_row];

    //GPU data
    multi_prec<prec> *dx;
    multi_prec<prec> *dy;
    multi_prec<prec> *ddata;
    int *dindices = new int[num_rows * num_cols_per_row];

    cudaMalloc(&dx, sizeof(multi_prec<prec>) * num_cols);
    cudaMalloc(&dy, sizeof(multi_prec<prec>) * num_rows);
    cudaMalloc(&ddata, sizeof(multi_prec<prec>) * num_rows * num_cols_per_row);
    cudaMalloc(&dindices, sizeof(int) * num_rows * num_cols_per_row);

    //convert to string
    #pragma omp parallel for
    for(int i = 0; i < num_cols; i ++){
        mpfr_init2(mpfrX[i], prec);
        mp_get_mpfr(mpfrX[i], &x[i]);
        hx[i] = convert_to_string_sci(mpfrX[i], convert_prec).c_str();
    }
    #pragma omp parallel for
    for(int i = 0; i < num_rows; i ++){
        mpfr_init2(mpfrY[i], prec);
        mp_get_mpfr(mpfrY[i], &y[i]);
        hy[i] = convert_to_string_sci(mpfrY[i], convert_prec).c_str();
    }
    #pragma omp parallel for
    for(int i = 0; i < num_rows * num_cols_per_row; i ++){
        mpfr_init2(mpfrData[i], prec);
        mp_get_mpfr(mpfrData[i], &data[i]);
        hdata[i] = convert_to_string_sci(mpfrData[i], convert_prec).c_str();
    }


    //Copying to the GPU
    cudaMemcpy(dx, hx, sizeof(multi_prec<prec>) * num_cols, cudaMemcpyHostToDevice);
    cudaMemcpy(ddata, hdata, sizeof(multi_prec<prec>) * num_rows * num_cols_per_row, cudaMemcpyHostToDevice);
    cudaMemcpy(dindices, indices, sizeof(int) * num_rows * num_cols_per_row, cudaMemcpyHostToDevice);
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    //Launch
    for(int i = 0; i < repeats; i ++){
        //TODO нужно ли перезаписывать?
        cudaMemcpy(dy, hy, sizeof(multi_prec<prec>) * num_rows, cudaMemcpyHostToDevice);
        StartCudaTimer();
        campary_spmv_ellpack_kernel<prec><<<BLOCKS, CAMPARY_VECTOR_MULTIPLY_THREADS>>>(num_rows, num_cols, ddata, dindices, dx, dy);
        EndCudaTimer();
    }
    PrintCudaTimer("took");
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    //Copying to the host
    cudaMemcpy(hy, dy, sizeof(multi_prec<prec>) * num_rows, cudaMemcpyDeviceToHost);
    for(int i = 1; i < num_rows; i ++){
        hy[0] += hy[i];
    }
    printResult<prec>(hy[0]);

    //Cleanup
    for(int i = 0; i < num_rows * num_cols_per_row; i++){
        mpfr_clear(mpfrData[i]);
    }
    for(int i = 0; i < num_cols; i++){
        mpfr_clear(mpfrX[i]);
    }
    for(int i = 0; i < num_rows; i++){
        mpfr_clear(mpfrY[i]);
    }
    delete [] mpfrData;
    delete [] mpfrX;
    delete [] mpfrY;
    delete [] hdata;
    delete [] hx;
    delete [] hy;
    cudaFree(ddata);
    cudaFree(dindices);
    cudaFree(dx);
    cudaFree(dy);
}



#endif //MPRES_TEST_CAMPARY_BLAS_CUH