/*
 *  Common utils for SpMV accuracy evaluation
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

#ifndef SPMV_CSR_ACCURACY_UTILS_CUH
#define SPMV_CSR_ACCURACY_UTILS_CUH

#include "mpfr.h"
#include "rns.cuh"

#define REFERENCE_PRECISION 10000 //Precision for reference results

//machine epsilon
void unit_roundoff_mpres(mpfr_t u){
    mpfr_sqrt(u, RNS_MODULI_PRODUCT_MPFR, MPFR_RNDN);
    mpfr_ui_div(u, 4, u, MPFR_RNDN);
}

void unit_roundoff_double(mpfr_t u){
    mpfr_set_d(u, 2, MPFR_RNDN);
    mpfr_pow_si(u, u, -53,MPFR_RNDN);
}

// u * n / (1 - u * n)
void gamma(mpfr_t gamma, mpfr_t u, int n){
    mpfr_t denominator;
    mpfr_init2(denominator, REFERENCE_PRECISION);
    mpfr_mul_si(gamma, u, n, MPFR_RNDN);
    mpfr_si_sub(denominator, 1, gamma, MPFR_RNDN);
    mpfr_div(gamma, gamma, denominator, MPFR_RNDN);
    mpfr_clear(denominator);
}

// exact matrix-vector product
void exact_spmv(const int m, const int *irp, const int *ja, const double *as, const double *x, mpfr_t *y) {
    #pragma omp parallel shared(m, irp, ja, as, x, y)
    {
        mpfr_t prod;
        mpfr_init2(prod, REFERENCE_PRECISION);
        #pragma omp for
        for(int row = 0; row < m; row++){
            mpfr_set_d(y[row], 0.0, MPFR_RNDN);
            int row_start = irp[row];
            int row_end = irp[row+1];
            for (int i = row_start; i < row_end; i++) {
                mpfr_set_d(prod, x[ja[i]], MPFR_RNDN);
                mpfr_mul_d(prod, prod, as[i], MPFR_RNDN);
                mpfr_add(y[row],y[row],prod, MPFR_RNDN);
            }
        }
        mpfr_clear(prod);
    }
}

// exact matrix-vector product when the matrix and the vector are in their absolute values
void exact_spmv_abs(const int m, const int *irp, const int *ja, const double *as, const double *x, mpfr_t *y) {
    #pragma omp parallel shared(m, irp, ja, as, x, y)
    {
        mpfr_t prod;
        mpfr_init2(prod, REFERENCE_PRECISION);
        #pragma omp for
        for(int row = 0; row < m; row++){
            mpfr_set_d(y[row], 0.0, MPFR_RNDN);
            int row_start = irp[row];
            int row_end = irp[row+1];
            for (int i = row_start; i < row_end; i++) {
                mpfr_set_d(prod, x[ja[i]], MPFR_RNDN);
                mpfr_mul_d(prod, prod, as[i], MPFR_RNDN);
                mpfr_abs(prod, prod, MPFR_RNDN);
                mpfr_add(y[row],y[row],prod, MPFR_RNDN);
            }
        }
        mpfr_clear(prod);
    }
}

// Euclidean norm of a vector
void norm2(mpfr_t norm, mpfr_t * vec, const int n) {
    mpfr_t sqr;
    mpfr_init2(sqr, REFERENCE_PRECISION);
    mpfr_set_d(norm, 0.0, MPFR_RNDN);
    for(int i = 0; i < n; i++){
        mpfr_sqr(sqr, vec[i], MPFR_RNDN);
        mpfr_add(norm, norm, sqr, MPFR_RNDN);
    }
    mpfr_sqrt(norm, norm, MPFR_RNDN);
    mpfr_clear(sqr);
}

// calculates condition number of the SpMV: || exact_spmv_abs || / || exact_norm ||
void spmv_condition_number(mpfr_t cond, const int m, const int *irp, const int *ja, const double *as, const double *x, mpfr_t exact_norm){
    mpfr_t * y = new mpfr_t[m];
    mpfr_t norm;
    for(int i = 0; i < m; i++){
        mpfr_init2(y[i], REFERENCE_PRECISION);
    }
    mpfr_init2(norm, REFERENCE_PRECISION);
    exact_spmv_abs(m, irp, ja, as, x, y);
    norm2(norm, y, m); // norm of the SpMV in its absolute values
    mpfr_div(cond, norm, exact_norm, MPFR_RNDN);
    mpfr_clear(norm);
    for(int i = 0; i < m; i++){
        mpfr_clear(y[i]);
    }
    delete[] y;
}

// relative residual of the spmv: || exact - y || / || exact ||, where exact and y are the exact and computed matrix-vector products, respectively
void spmv_relative_residual(mpfr_t residual, const int m, mpfr_t * y, mpfr_t * exact, mpfr_t exact_norm){
    mpfr_t norm;
    mpfr_init2(norm, REFERENCE_PRECISION);
    norm2(norm, exact, m); // norm of the difference result
    #pragma omp parallel for
    for(int i = 0; i < m; i++){
        mpfr_sub(y[i], exact[i], y[i], MPFR_RNDN); //difference
    }
    norm2(norm, y, m); // norm of the vector-difference
    mpfr_div(residual, norm, exact_norm, MPFR_RNDN);
    mpfr_clear(norm);
}

#endif //SPMV_CSR_ACCURACY_UTILS_CUH