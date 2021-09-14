/*
 *  Performance test for double precision SYMV
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
#ifndef TEST_DOUBLE_SYMV_CUH
#define TEST_DOUBLE_SYMV_CUH

#include "logger.cuh"
#include "timers.cuh"
#include "tsthelper.cuh"

void double_symv_up_reference(int n, double alpha, const double *A, int lda, const double *x, double beta, double *y) {
    double temp1;
    double temp2;
    int i = 0;
    int j = 0;
    //   #pragma omp for
    for (i = 0; i < n; i++) {
        y[i] = beta * y[i];
    }
    for (j = 0; j < n; j++) {
        temp1 = alpha * x[j];
        temp2 = 0;
        //     #pragma omp for
        for (i = 0; i < j; i++) {
            y[i] = y[i] + temp1 * A[i + j * lda];
            temp2 = temp2 + A[i + j * lda] * x[i];
        }
        y[j] = y[j] + temp1 * A[i + j * lda] + alpha * temp2;
    }
}

/////////
// CPU
/////////
void double_symv(enum mblas_uplo_type uplo, int n, double alpha, const double *A, int lda, const double *x, double beta, double *y) {
    if (uplo == mblas_upper) { //Use the upper part of the matrix
        #pragma omp parallel shared(n, A, x, y)
        {
            double mul_acc;
            double ax;
            int i = 0;
            int j = 0;
            #pragma omp for
            for (i = 0; i < n; i++) {
                y[i] = beta * y[i];
            }

            for (j = 0; j < n; j++) {
                ax = alpha * x[j];
                #pragma omp for
                for (i = 0; i <= j; i++) {
                    mul_acc = ax * A[i + j * lda];
                    y[i] = y[i] + mul_acc;
                }
                #pragma omp for
                for (i = j + 1; i < n; i++) {
                    mul_acc = ax * A[j + i * lda];
                    y[i] = y[i] + mul_acc;
                }
            }
        }
    } else{ //Use the lower part of the matrix
        #pragma omp parallel shared(n, A, x, y)
        {
            double mul_acc;
            double ax;
            int i = 0;
            int j = 0;
            #pragma omp for
            for (i = 0; i < n; i++) {
                y[i] = beta * y[i];
            }

            for (j = 0; j < n; j++) {
                ax = alpha * x[j];
                #pragma omp for
                for (i = 0; i <= j; i++) {
                    mul_acc = ax * A[j + i * lda];
                    y[i] = y[i] + mul_acc;
                }
                #pragma omp for
                for (i = j + 1; i < n; i++) {
                    mul_acc = ax * A[i + j * lda];
                    y[i] = y[i] + mul_acc;
                }
            }
        }
    }
}

void test_double(enum mblas_uplo_type uplo, const int n, int lenx, int leny, mpfr_t alpha, mpfr_t *A, int lda, mpfr_t *x, mpfr_t beta, mpfr_t *y, const int repeats) {
    InitCpuTimer();
    Logger::printDash();
    PrintTimerName("[CPU] double symv");

    //CPU data
    double *dx = new double[lenx];
    double *dy = new double[leny];
    double *dr = new double[leny];
    double *dA = new double[lda * n];
    double dalpha = mpfr_get_d(alpha, MPFR_RNDN);
    double dbeta = mpfr_get_d(beta, MPFR_RNDN);

    for (int i = 0; i < lenx; ++i) {
        dx[i] = mpfr_get_d(x[i], MPFR_RNDN);
    }

    for (int i = 0; i < leny; ++i) {
        dy[i] = mpfr_get_d(y[i], MPFR_RNDN);
    }

    for (int i = 0; i < lda * n; ++i) {
        dA[i] = mpfr_get_d(A[i], MPFR_RNDN);
    }

    //Launch
    for (int i = 0; i < repeats; i++) {
        for (int j = 0; j < leny; j++) {
            dr[j] = dy[j];
        }
        StartCpuTimer();
        double_symv(uplo, n, dalpha, dA, lda, dx, dbeta, dr);
        EndCpuTimer();
    }
    PrintCpuTimer("took");
    print_double_sum(dr, leny);
    delete[] dx;
    delete[] dy;
    delete[] dr;
    delete[] dA;
}


/////////
// CUDA
/////////
__global__ static void double_symv_cuda(enum mblas_uplo_type uplo, int n, double alpha, const double *A, int lda, const double *x, double beta, double *y) {
    unsigned int threadId = threadIdx.x + blockIdx.x * blockDim.x;
    //Each thread works with its own row, i.e. goes through the columns
    if (threadId < n) {
        double dot = 0;
        if (uplo == mblas_upper) { //Use the upper part of the matrix
            for (int colId = 0; colId < n; colId++) {
                double dx = x[colId];
                if(threadId <= colId){
                    dot += dx * A[threadId + colId * lda];
                } else{
                    dot += dx * A[colId + threadId * lda];
                }
            }
        } else{ //Use the lower part of the matrix
            for (int colId = 0; colId < n; colId++) {
                double dx = x[colId];
                if(threadId <= colId){
                    dot += dx * A[colId + threadId * lda];
                } else{
                    dot += dx * A[threadId + colId * lda];
                }
            }
        }
        y[threadId] = beta * y[threadId] + alpha * dot;
    }
}

void test_double_symv_cuda(enum mblas_uplo_type uplo, const int n, int lenx, int leny, mpfr_t alpha, mpfr_t *A, int lda, mpfr_t *x, int incx, mpfr_t beta, mpfr_t *y, int incy, int repeats) {
    InitCudaTimer();
    Logger::printDash();
    PrintTimerName("[GPU] double symv");

    //Execution configuration
    int threads = 32;
    int blocks = n / threads + 1;
    printf("\tExec. config: blocks = %i, threads = %i\n", blocks, threads);

    //Host data
    double *hx = new double[lenx];
    double *hy = new double[leny];
    double *hA = new double[lda * n];

    //GPU data
    double *dx;
    double *dy;
    double *dA;

    double dalpha = mpfr_get_d(alpha, MPFR_RNDN);
    double dbeta = mpfr_get_d(beta, MPFR_RNDN);

    cudaMalloc(&dx, sizeof(double) * lenx);
    cudaMalloc(&dy, sizeof(double) * leny);
    cudaMalloc(&dA, sizeof(double) * lda * n);

    convert_vector(hx, x, lenx);
    convert_vector(hy, y, leny);
    convert_vector(hA, A, lda * n);

    cudaMemcpy(dx, hx, sizeof(double) * lenx, cudaMemcpyHostToDevice);
    cudaMemcpy(dA, hA, sizeof(double) * lda * n, cudaMemcpyHostToDevice);

    for(int i = 0; i < repeats; i ++) {
        cudaMemcpy(dy, hy, sizeof(double) * leny, cudaMemcpyHostToDevice);
        StartCudaTimer();
        double_symv_cuda<<<blocks, threads>>>(uplo, n, dalpha, dA, lda, dx, dbeta, dy);
        EndCudaTimer();
    }
    PrintCudaTimer("took");
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    //Copying to the host
    cudaMemcpy(hy, dy, sizeof(double) * leny , cudaMemcpyDeviceToHost);
    print_double_sum(hy, leny);

    delete[] hx;
    delete[] hy;
    delete[] hA;
    cudaFree(dx);
    cudaFree(dy);
    cudaFree(dA);
}


#endif //TEST_DOUBLE_SYMV_CUH
