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
 * Performs the matrix-vector operation y = A * x
 * where x and y are dense vectors and A is a sparse matrix.
 * The matrix should be stored in the ELLPACK format: entries are stored in a dense array in column major order and explicit zeros are stored if necessary (zero padding)
 */
template<int prec>
__global__ void campary_spmv_ellpack_kernel(int num_rows, int num_cols_per_row, int *indices, multi_prec<prec> *data, multi_prec<prec> *x, multi_prec<prec> *y) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    multi_prec<prec> dot = 0.0;
    for (int colId = 0; colId < num_cols_per_row; colId++) {
        if (i < num_rows) {
            int index = indices[colId * num_rows + i];
            dot += data[colId * num_rows + i] * x[index];
        }
    }
    if (i < num_rows) {
        y[i] = dot;
    }
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
    mpfr_clear(x);
    mpfr_clear(r);
}

/*
 * SpMV ELLPACK test
 */
template<int prec>
void campary_spmv_ellpack_test(const int num_rows, const int num_cols, const int num_cols_per_row,  double const *data, int *indices, mpfr_t *x, int convert_prec, int repeats) {
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
    for(int i = 0; i < repeats; i ++){
        StartCudaTimer();
        campary_spmv_ellpack_kernel<prec><<<BLOCKS, CAMPARY_VECTOR_MULTIPLY_THREADS>>>(num_rows, num_cols_per_row, dindices, ddata, dx, dy);
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
    delete [] hx;
    delete [] hy;
    delete [] hdata;
    cudaFree(dx);
    cudaFree(dy);
    cudaFree(ddata);
    cudaFree(dindices);
}



#endif //MPRES_TEST_CAMPARY_BLAS_CUH