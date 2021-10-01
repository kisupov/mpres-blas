/*
 *  Common routines for double precision iterative methods
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

#ifndef DOUBLE_LS_KERNELS_CUH
#define DOUBLE_LS_KERNELS_CUH

#include "cublas_v2.h"
#include <cassert>

//Sparse matrix-vector product, y = Ax
__global__ static void double_spmv_csr_kernel(const int m, const csr_t csr, const double *x, double *y) {
    auto row = threadIdx.x + blockIdx.x * blockDim.x;
    if(row < m){
        double dot = 0;
        int row_start = csr.irp[row];
        int row_end = csr.irp[row+1];
        for (int i = row_start; i < row_end; i++) {
            dot += csr.as[i] * x[csr.ja[i]];
        }
        y[row] = dot;
    }
}

//vector difference, z = x - y
__global__ void double_diff_kernel(const int n, const double *x, const double *y, double *z) {
    auto i = threadIdx.x + blockIdx.x * blockDim.x;
    while(i < n){
        z[i] = x[i] - y[i];
        i += gridDim.x * blockDim.x;
    }
}

//component-wise vector product, z = x * y
__global__ void double_prod_kernel(const int n, const double *x, const double *y, double *z){
    auto i = threadIdx.x + blockIdx.x * blockDim.x;
    while(i < n){
        z[i] = x[i] * y[i];
        i += gridDim.x * blockDim.x;
    }
}

//z = alpha * x + y
__global__ void double_axpy_kernel(const int n, double alpha, const double *x, const double *y, double *z){
    auto i = threadIdx.x + blockIdx.x * blockDim.x;
    while(i < n){
        z[i] = alpha*x[i] + y[i];
        i += gridDim.x * blockDim.x;
    }
}

//vector copy, y = x;
__global__ void double_copy_kernel(const int n, const double *x, double *y){
    auto i = threadIdx.x + blockIdx.x * blockDim.x;
    while(i < n){
        y[i] = x[i];
        i += gridDim.x * blockDim.x;
    }
}

//vector reciprocal, y = 1 / x;
__global__ void double_invert_kernel(const int n, const double *x, double *y){
    auto i = threadIdx.x + blockIdx.x * blockDim.x;
    while(i < n){
        assert(x[i] != 0.0);
        y[i] = 1.0 / x[i];
        i += gridDim.x * blockDim.x;
    }
}


#endif //DOUBLE_LS_KERNELS_CUH
