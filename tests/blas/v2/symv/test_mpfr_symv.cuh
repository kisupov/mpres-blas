/*
 *  Performance test for MPFR SYMV
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
#ifndef TEST_MPFR_SYMV_CUH
#define TEST_MPFR_SYMV_CUH

#include "logger.cuh"
#include "timers.cuh"
#include "tsthelper.cuh"
#include "mpfr.h"

void mpfr_symv_up_reference(int n, mpfr_t alpha, mpfr_t *A, int lda, mpfr_t *x, mpfr_t beta, mpfr_t *y) {
    mpfr_t temp1;
    mpfr_t temp2;
    mpfr_t acc1;
    mpfr_t acc2;
    mpfr_init2(temp1, MP_PRECISION);
    mpfr_init2(temp2, MP_PRECISION);
    mpfr_init2(acc1, MP_PRECISION);
    mpfr_init2(acc2, MP_PRECISION);
    int i = 0;
    int j = 0;
    //#pragma omp parallel for
    for (i = 0; i < n; i++) {
        mpfr_mul(y[i], beta, y[i], MPFR_RNDN);
    }
    for (j = 0; j < n; j++) {
        mpfr_mul(temp1, alpha, x[j], MPFR_RNDN);
        mpfr_set_d(temp2, 0, MPFR_RNDN);
        //     #pragma omp for
        for (i = 0; i < j; i++) {
            mpfr_mul(acc1, temp1, A[i + j * lda], MPFR_RNDN); //temp1 * a(i,j)
            mpfr_add(y[i], y[i], acc1, MPFR_RNDN); // y(i) = y(i) + temp1*a(i,j)
            mpfr_mul(acc2, x[i], A[i + j * lda], MPFR_RNDN); //a(i,j)*x(i)
            mpfr_add(temp2, temp2, acc2, MPFR_RNDN); // temp2 = temp2 + a(i,j)*x(i)
        }
        mpfr_mul(temp1, temp1, A[j + j * lda], MPFR_RNDN);
        mpfr_mul(temp2, alpha, temp2, MPFR_RNDN);
        mpfr_add(temp1, temp1, temp2, MPFR_RNDN); // y(j) = y(j) + temp1*a(j,j) + alpha*temp2
        mpfr_add(y[j], y[j], temp1, MPFR_RNDN); // y(j) = y(j) + temp1*a(j,j) + alpha*temp2
    }
    mpfr_clear(temp1);
    mpfr_clear(temp2);
    mpfr_clear(acc1);
    mpfr_clear(acc2);
}

void mpfr_symv(enum mblas_uplo_type uplo, int n, mpfr_t alpha, mpfr_t *A, int lda, mpfr_t *x, mpfr_t beta, mpfr_t *y) {
    if (uplo == mblas_upper) { //Use the upper part of the matrix
        #pragma omp parallel shared(n, A, x, y)
        {
            mpfr_t mul_acc;
            mpfr_t ax;
            mpfr_init2(ax, MP_PRECISION);
            mpfr_init2(mul_acc, MP_PRECISION);
            int i = 0;
            int j = 0;
            #pragma omp for
            for (i = 0; i < n; i++) {
                mpfr_mul(y[i], beta, y[i], MPFR_RNDN);
            }

            for (j = 0; j < n; j++) {
                mpfr_mul(ax, alpha, x[j], MPFR_RNDN);
                #pragma omp for
                for (i = 0; i <= j; i++) {
                    mpfr_mul(mul_acc, ax, A[i + j * lda], MPFR_RNDN);
                    mpfr_add(y[i], y[i], mul_acc, MPFR_RNDN);
                }
                #pragma omp for
                for (i = j + 1; i < n; i++) {
                    mpfr_mul(mul_acc, ax, A[j + i * lda], MPFR_RNDN);
                    mpfr_add(y[i], y[i], mul_acc, MPFR_RNDN);
                }
            }
            mpfr_clear(mul_acc);
            mpfr_clear(ax);
        }
    } else{ //Use the lower part of the matrix
        #pragma omp parallel shared(n, A, x, y)
        {
            mpfr_t mul_acc;
            mpfr_t ax;
            mpfr_init2(ax, MP_PRECISION);
            mpfr_init2(mul_acc, MP_PRECISION);
            int i = 0;
            int j = 0;
            #pragma omp for
            for (i = 0; i < n; i++) {
                mpfr_mul(y[i], beta, y[i], MPFR_RNDN);
            }
            for (j = 0; j < n; j++) {
                mpfr_mul(ax, alpha, x[j], MPFR_RNDN);
                #pragma omp for
                for (i = 0; i <= j; i++) {
                    mpfr_mul(mul_acc, ax, A[j + i * lda], MPFR_RNDN);
                    mpfr_add(y[i], y[i], mul_acc, MPFR_RNDN);
                }
                #pragma omp for
                for (i = j + 1; i < n; i++) {
                    mpfr_mul(mul_acc, ax, A[i + j * lda], MPFR_RNDN);
                    mpfr_add(y[i], y[i], mul_acc, MPFR_RNDN);
                }
            }
            mpfr_clear(mul_acc);
            mpfr_clear(ax);
        }
    }
}

void test_mpfr(enum mblas_uplo_type uplo, int n, mpfr_t alpha, mpfr_t *A, int lda, mpfr_t *x, mpfr_t beta, mpfr_t *y, const int repeats) {
    InitCpuTimer();
    Logger::printDash();
    PrintTimerName("[CPU] MPFR symv");

    // Init
    mpfr_t *result = new mpfr_t[n];
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        mpfr_init2(result[i], MP_PRECISION);
    }

    // Launch
    for (int i = 0; i < repeats; i++) {
        #pragma omp parallel for
        for (int j = 0; j < n; j++) {
            mpfr_set(result[j], y[j], MPFR_RNDN);
        }
        StartCpuTimer();
        mpfr_symv(uplo, n, alpha, A, lda, x, beta, result);
        EndCpuTimer();
    }
    PrintCpuTimer("took");
    print_mpfr_sum(result, n);

    //Cleanup
    for (int i = 0; i < n; i++) {
        mpfr_clear(result[i]);
    }
    delete[] result;
}

#endif //TEST_MPFR_SYMV_CUH
