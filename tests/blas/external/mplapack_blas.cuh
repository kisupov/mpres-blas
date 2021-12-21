/*
 *  Performance benchmarks for the MPLAPACK (formerly MPACK) routines.
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

#ifndef EXCLUDE_MPACK
#ifndef MPRES_TEST_MPLAPACK_BLAS_CUH
#define MPRES_TEST_MPLAPACK_BLAS_CUH

#include "mpack/mpreal.h"
#include "mpack/mblas_mpfr.h"
#include "timers.cuh"
#include "omp.h"

/*
 * ASUM test
 */
void mpack_asum_test(mpfr_t *x, int n, int precision, int repeats) {
    InitCpuTimer();
    Logger::printDash();
    PrintTimerName("[CPU] MPACK asum");

    //Set precision
    mpfr::mpreal::set_default_prec( precision );

    //Init
    mpreal *mpreal_x = new mpreal[n];
    mpreal mpreal_result;
    mpreal_result = 0;
    for (int i = 0; i < n; i++) {
        mpreal_x[i] = x[i];
    }

    //Launch
    StartCpuTimer();
    for (int i = 0; i < repeats; i++)
        mpreal_result = Rasum(n, mpreal_x, 1); // Call MPACK
    EndCpuTimer();
    PrintAndResetCpuTimer("took");
    mpfr_printf("result: %.70Rf\n", &mpreal_result);

    //Clear
    delete[] mpreal_x;
}

/*
 * DOT test
 */
void mpack_dot_test(mpfr_t *x, mpfr_t *y, int n, int precision, int repeats) {
    InitCpuTimer();
    Logger::printDash();
    PrintTimerName("[CPU] MPACK dot");

    //Set precision
    mpfr::mpreal::set_default_prec ( precision );

    //Init
    mpreal *mpreal_x = new mpreal[n];
    mpreal *mpreal_y = new mpreal[n];
    mpreal mpreal_result = 0;
    for (int i = 0; i < n; i++) {
        mpreal_x[i] = x[i];
        mpreal_y[i] = y[i];
    }

    //Launch
    StartCpuTimer();
    for(int i = 0; i < repeats; i ++){
        mpreal_result = Rdot(n, mpreal_x, 1, mpreal_y, 1);
    }
    EndCpuTimer();
    PrintAndResetCpuTimer("took");
    mpfr_printf("result: %.70Rf\n", &mpreal_result);

    //Cleanup
    delete [] mpreal_x;
    delete [] mpreal_y;
}

/*
 * SCAL test
 */
void mpack_scal_test(mpfr_t *x, mpfr_t alpha, int n, int precision, int repeats) {
    Logger::printDash();
    InitCpuTimer();
    PrintTimerName("[CPU] MPACK scal");

    //Set precision
    mpfr::mpreal::set_default_prec ( precision );

    //Init
    mpreal *lx = new mpreal[n];
    mpreal lalpha = alpha;

    //Launch
    for(int i = 0; i < repeats; i ++) {
        for (int j = 0; j < n; j++) {
            lx[j] = x[j];
        }
        StartCpuTimer();
        Rscal(n, lalpha, lx, 1);
        EndCpuTimer();
    }
    PrintAndResetCpuTimer("took");

    //Print
    mpreal result = 0.0;
    for (int i = 0; i < n; i++) {
        result += lx[i];
    }
    mpfr_printf("result: %.70Rf\n", &result);

    //Cleanup
    delete [] lx;
}

/*
 * AXPY test
 */
void mpack_axpy_test(mpfr_t * x, mpfr_t *y, mpfr_t alpha, int n, int precision, int repeats) {
    Logger::printDash();
    InitCpuTimer();
    PrintTimerName("[CPU] MPACK axpy");

    //Set precision
    mpfr::mpreal::set_default_prec ( precision );

    //Init
    mpreal *mpreal_x = new mpreal[n];
    mpreal *mpreal_y = new mpreal[n];
    mpreal mpreal_alpha = alpha;
    for (int j = 0; j < n; j++) {
        mpreal_x[j] = x[j];
    }

    //Launch
    for(int i = 0; i < repeats; i ++) {
        for (int j = 0; j < n; j++) {
            mpreal_y[j] = y[j];
        }
        StartCpuTimer();
        Raxpy(n, mpreal_alpha, mpreal_x, 1, mpreal_y, 1);
        EndCpuTimer();
    }
    PrintAndResetCpuTimer("took");

    //Print
    mpreal mpreal_result = 0.0;
    for (int i = 0; i < n; i++) {
        mpreal_result += mpreal_y[i];
    }
    mpfr_printf("result: %.70Rf\n", &mpreal_result);

    //Cleanup
    delete [] mpreal_x;
    delete [] mpreal_y;
}

/*
 * ROT test
 */
void mpack_rot_test( int n, mpfr_t * x, mpfr_t *y, mpfr_t c, mpfr_t s, int precision, int repeats) {
    Logger::printDash();
    InitCpuTimer();
    PrintTimerName("[CPU] MPACK rot");

    //Set precision
    mpfr::mpreal::set_default_prec ( precision );

    //Init
    mpreal *mpreal_x = new mpreal[n];
    mpreal *mpreal_y = new mpreal[n];
    mpreal mpreal_c = c;
    mpreal mpreal_s = s;
    for (int j = 0; j < n; j++) {
        mpreal_x[j] = x[j];
    }

    //Launch
    for(int i = 0; i < repeats; i++) {
        #pragma omp parallel for
        for (int j = 0; j < n; j++) {
            mpreal_x[j] = x[j];
            mpreal_y[j] = y[j];
        }
        StartCpuTimer();
        Rrot(n, mpreal_x, 1, mpreal_y, 1, mpreal_c, mpreal_s);
        EndCpuTimer();
    }
    PrintAndResetCpuTimer("took");

    //Print
    for (int i = 1; i < n; i++) {
        mpreal_x[0] += mpreal_x[i];
        mpreal_y[0] += mpreal_y[i];
    }
    mpfr_printf("result x: %.70Rf\n", &mpreal_x[0]);
    mpfr_printf("result y: %.70Rf\n", &mpreal_y[0]);

    //Cleanup
    delete [] mpreal_x;
    delete [] mpreal_y;
}

/*
 * GEMV test
 */
void mpack_gemv_test(const char * trans, int m, int n, int lenx, int leny, mpfr_t alpha, mpfr_t *A, int lda, mpfr_t *x, int incx, mpfr_t beta, mpfr_t *y, int incy, int precision, int repeats){
    Logger::printDash();
    InitCpuTimer();
    PrintTimerName("[CPU] MPACK gemv");

    //Set precision
    mpfr::mpreal::set_default_prec ( precision );

    //Init
    mpreal *lA = new mpreal[lda * n];
    mpreal *ly = new mpreal[leny];
    mpreal *lx = new mpreal[lenx];
    mpreal lalpha = alpha;
    mpreal lbeta = beta;

    #pragma omp parallel for
    for(int i = 0; i < lda * n; i++){
        lA[i] = A[i];
    }
    #pragma omp parallel for
    for(int i = 0; i < lenx; i++){
        lx[i] = x[i];
    }

    //Launch
    for(int j = 0; j < repeats; j ++){
        #pragma omp parallel for
        for(int i = 0; i < leny; i++){
            ly[i] = y[i];
        }
        StartCpuTimer();
        Rgemv(trans, m, n, lalpha, lA, lda, lx, incx, lbeta, ly, incy);
        EndCpuTimer();
    }
    PrintAndResetCpuTimer("took");

    //Print
    for (int i = 1; i < leny; i+= 1) {
        ly[0] += ly[i];
    }
    mpfr_printf("result: %.70Rf\n", &ly[0]);

    //Cleanup
    delete [] lA;
    delete [] lx;
    delete [] ly;
}

/*
 * GEMM test
 */
void mpack_gemm_test(const char *transA, const char *transB, const int m, const int n, const int k, mpfr_t alpha, mpfr_t *A, const int lda, mpfr_t *B, const int ldb, mpfr_t beta, mpfr_t *C, const int ldc, int precision, int repeats){
    Logger::printDash();
    InitCpuTimer();
    PrintTimerName("[CPU] MPACK gemm");

    //Set precision
    mpfr::mpreal::set_default_prec ( precision );

    //Init
    mpreal *lA = new mpreal[lda * k];
    mpreal *lB = new mpreal[ldb * n];
    mpreal *lC = new mpreal[ldc * n];
    mpreal lalpha = alpha;
    mpreal lbeta = beta;

    #pragma omp parallel for
    for (int i = 0; i < lda * k; i++) {
        lA[i] = A[i];
    }

    #pragma omp parallel for
    for (int i = 0; i < ldb * n; i++) {
        lB[i] = B[i];
    }

    //Launch
    for(int j = 0; j < repeats; j ++){
        #pragma omp parallel for
        for(int i = 0; i < ldc * n; i++){
            lC[i] = C[i];
        }
        StartCpuTimer();
        Rgemm(transA, transB, m, n, k, lalpha, lA, lda, lB, ldb, lbeta, lC, ldc);
        EndCpuTimer();
    }
    PrintAndResetCpuTimer("took");

    //Print
    for (int i = 1; i < ldc * n; i+= 1) {
        lC[0] += lC[i];
    }
    mpfr_printf("result: %.70Rf\n", &lC[0]);

    //Cleanup
    delete [] lA;
    delete [] lB;
    delete [] lC;
}

/*
 * GER test
 */
void mpack_ger_test(int m, int n, int lenx, int leny, mpfr_t alpha, mpfr_t *A, int lda, mpfr_t *x, int incx, mpfr_t *y, int incy, int precision, int repeats){
    Logger::printDash();
    InitCpuTimer();
    PrintTimerName("[CPU] MPACK ger");

    //Set precision
    mpfr::mpreal::set_default_prec ( precision );

    //Init
    mpreal *lA = new mpreal[lda * n];
    mpreal *ly = new mpreal[leny];
    mpreal *lx = new mpreal[lenx];
    mpreal lalpha = alpha;

    #pragma omp parallel for
    for(int i = 0; i < lenx; i++){
        lx[i] = x[i];
    }
    #pragma omp parallel for
    for(int i = 0; i < leny; i++){
        ly[i] = y[i];
    }

    //Launch
    for(int j = 0; j < repeats; j ++){
        #pragma omp parallel for
        for(int i = 0; i < lda * n; i++){
            lA[i] = A[i];
        }
        StartCpuTimer();
        Rger(m, n, lalpha, lx, incx, ly, incy, lA, lda);
        EndCpuTimer();
    }
    PrintAndResetCpuTimer("took");

    //Print
    for (int i = 1; i < lda * n; i+= 1) {
        lA[0] += lA[i];
    }
    mpfr_printf("result: %.70Rf\n", &lA[0]);

    //Cleanup
    delete [] lA;
    delete [] lx;
    delete [] ly;
}

#endif //MPRES_TEST_MPLAPACK_BLAS_CUH
#endif //EXCLUDE_MPACK