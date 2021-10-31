/*
 *  Common routines for double precision iterative methods on CPU
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

#ifndef DOUBLE_LS_KERNELS_CPU_CUH
#define DOUBLE_LS_KERNELS_CPU_CUH

#include "cassert"

//Sparse matrix-vector product, y = Ax
void double_spmv_csr(const int m, const csr_t matrix, const double *x, double *y) {
    for(int row = 0; row < m; row++){
        y[row] = 0;
        int row_start = matrix.irp[row];
        int row_end = matrix.irp[row+1];
        for (int i = row_start; i < row_end; i++) {
            y[row] = y[row] + x[matrix.ja[i]] * matrix.as[i];
        }
    }
}

//vector difference, z = x - y
void double_diff(const int n, const double *x, const double *y, double *z){
    for(int i = 0; i < n; i++){
        z[i] = x[i] - y[i];
    }
}

//component-wise vector product, z = x * y
void double_prod(const int n, const double *x, const double *y, double *z){
    for(int i = 0; i < n; i++){
        z[i] = x[i] * y[i];
    }
}

//z = alpha * x + y
void double_axpy(const int n, double alpha, const double *x, const double *y, double *z){
    for(int i = 0; i < n; i++){
        z[i] = alpha*x[i] + y[i];
    }
}

//vector copy, y = x;
void double_copy(const int n, const double *x, double *y){
    for(int i = 0; i < n; i++){
        y[i] = x[i];
    }
}

//vector reciprocal, y = 1 / x;
void double_invert(const int n, const double *x, double *y){
    for(int i = 0; i < n; i++){
        assert(x[i] != 0.0);
        y[i] = 1.0 / x[i];
    }
}

// Euclidean norm
double double_norm2(const int n, const double *x){
    double norm = 0;
    for(int i = 0; i < n; i++){
        norm += x[i]*x[i];
    }
    return sqrt(norm);
}

//inner product
double double_dot(const int n, const double *x, const double *y){
    double dot = 0;
    for(int i = 0; i < n; i++){
        dot += x[i] * y[i];
    }
    return dot;
}

#endif //DOUBLE_LS_KERNELS_CPU_CUH
