/*
 *  Performance test for BLAS SYMV routines
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

/*
 * Exclude some benchmarks
 */
#define EXCLUDE_MPACK
#define EXCLUDE_XBLAS
#define EXCLUDE_MPDECIMAL
#define EXCLUDE_GARPREC

#include "../../logger.cuh"
#include "../../timers.cuh"
#include "../../tsthelper.cuh"
#include "../../../src/blas/symv.cuh"
#include "blas/external/3rdparty.cuh"

#define N 5000  // Number of matrix rows / column and the vectors dimension
#define LDA (N) // Specifies the leading dimension of A as declared in the calling (sub)program.
#define UPLO mblas_upper // Specifies whether the upper or lower triangular part of the array a is used.
#define INCX 1 // Specifies the increment for the elements of x.
#define INCY 1 // Specifies the increment for the elements of y.
#define REPEAT_TEST 1 //Number of repeats

#define OPENBLAS_THREADS 4

int MP_PRECISION_DEC; //in decimal digits
int INP_BITS; //in bits
int INP_DIGITS; //in decimal digits

void setPrecisions() {
    MP_PRECISION_DEC = (int) (MP_PRECISION / 3.32 + 1);
    INP_BITS = (int) (MP_PRECISION / 4);
    INP_DIGITS = (int) (INP_BITS / 3.32 + 1);
}

void initialize() {
    cudaDeviceReset();
    rns_const_init();
    mp_const_init();
    setPrecisions();
    //mp_real::mp_init(MP_PRECISION_DEC);
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();
}

void finalize() {
    //mp_real::mp_finalize();
}

/********************* GEMV implementations and benchmarks *********************/

/////////
// Reference implementations
/////////
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

/////////
// OpenBLAS
/////////
extern "C" void openblas_set_num_threads(int num_threads);

void test_openblas(enum mblas_uplo_type uplo, const int n, int lenx, int leny, mpfr_t alpha, mpfr_t *A, int lda, mpfr_t *x,
              int incx, mpfr_t beta, mpfr_t *y, int incy) {
    InitCpuTimer();
    Logger::printDash();
    PrintTimerName("[CPU] OpenBLAS symv");

    openblas_set_num_threads(OPENBLAS_THREADS);

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
    for (int i = 0; i < REPEAT_TEST; i++) {
        for (int j = 0; j < leny; j++) {
            dr[j] = dy[j];
        }
        StartCpuTimer();
        if (uplo == mblas_upper) {
            cblas_dsymv(CblasColMajor, CblasUpper, n, dalpha, dA, lda, dx, incx, dbeta, dr, incy);
        } else {
            cblas_dsymv(CblasColMajor, CblasLower, n, dalpha, dA, lda, dx, incx, dbeta, dr, incy);
        }
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
// double upper
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

void test_double(enum mblas_uplo_type uplo, const int n, int lenx, int leny, mpfr_t alpha, mpfr_t *A, int lda, mpfr_t *x, mpfr_t beta, mpfr_t *y) {
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
    for (int i = 0; i < REPEAT_TEST; i++) {
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
// MPFR upper
/////////
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

void test_mpfr(enum mblas_uplo_type uplo, int n, mpfr_t alpha, mpfr_t *A, int lda, mpfr_t *x, mpfr_t beta, mpfr_t *y) {
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
    for (int i = 0; i < REPEAT_TEST; i++) {
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

/////////
// cuBLAS
/////////
void test_cublas(enum mblas_uplo_type uplo, const int n, int lenx, int leny, mpfr_t alpha, mpfr_t *A, int lda, mpfr_t *x,
        int incx, mpfr_t beta, mpfr_t *y, int incy){
    InitCudaTimer();
    Logger::printDash();
    PrintTimerName("[GPU] cuBLAS symv");

    cublasStatus_t stat;
    cublasHandle_t handle;
    stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("cuBLAS initialization failed\n");
        return;
    }

    //Host data
    double *hx = new double[lenx];
    double *hy = new double[leny];
    double *hA = new double[lda * n];

    for (int i = 0; i < lenx; i++) {
        hx[i] = mpfr_get_d(x[i], MPFR_RNDN);
    }

    for (int i = 0; i < leny; i++) {
        hy[i] = mpfr_get_d(y[i], MPFR_RNDN);
    }

    for (int i = 0; i < lda * n; i++) {
        hA[i] = mpfr_get_d(A[i], MPFR_RNDN);
    }

    //GPU data
    double *dx;
    double *dy;
    double *dA;
    double dalpha = mpfr_get_d(alpha, MPFR_RNDN);
    double dbeta = mpfr_get_d(beta, MPFR_RNDN);

    cudaMalloc(&dx, sizeof(double) * lenx);
    cudaMalloc(&dy, sizeof(double) * leny);
    cudaMalloc(&dA, sizeof(double) * lda * n);
    cublasSetVector(n, sizeof(double), hx, incx, dx, incx);
    cublasSetVector(lda * n, sizeof(double), hA, 1, dA, 1);

    for(int i = 0; i < REPEAT_TEST; i ++) {
        cublasSetVector(n, sizeof(double), hy, incy, dy, incy);
        StartCudaTimer();
        if (uplo == mblas_upper) {
            cublasDsymv(handle, CUBLAS_FILL_MODE_UPPER, n, &dalpha, dA, lda, dx, incx, &dbeta, dy, incy);
        } else{
            cublasDsymv(handle, CUBLAS_FILL_MODE_LOWER, n, &dalpha, dA, lda, dx, incx, &dbeta, dy, incy);
        }
        EndCudaTimer();

    }
    PrintCudaTimer("took");
    cublasGetVector(n, sizeof(double), dy, incy, hy, incy);
    print_double_sum(hy, leny);
    cublasDestroy ( handle );
    delete[] hx;
    delete[] hy;
    delete[] hA;
    cudaFree(dx);
    cudaFree(dy);
    cudaFree(dA);
}


/////////
// double CUDA
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

void test_double_symv_cuda(enum mblas_uplo_type uplo, const int n, int lenx, int leny, mpfr_t alpha, mpfr_t *A, int lda, mpfr_t *x, int incx, mpfr_t beta, mpfr_t *y, int incy) {
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

    for(int i = 0; i < REPEAT_TEST; i ++) {
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

/////////
// MPRES-BLAS
/////////
void test_mpres_symv(enum mblas_uplo_type uplo, const int n, int lenx, int leny, mpfr_t alpha, mpfr_t *A, int lda, mpfr_t *x, int incx, mpfr_t beta, mpfr_t *y, int incy) {
    InitCudaTimer();
    Logger::printDash();
    PrintTimerName("[GPU] MPRES-BLAS symv");

    //Execution configuration
    int threads = 32;
    int blocks = n / threads + 1;
    printf("\tExec. config: blocks = %i, threads = %i\n", blocks, threads);

    //Host data
    mp_float_ptr hx = new mp_float_t[lenx];
    mp_float_ptr hy = new mp_float_t[leny];
    mp_float_ptr hA = new mp_float_t[lda * n];
    mp_float_t halpha;
    mp_float_t hbeta;

    // GPU data
    mp_float_ptr dA;
    mp_float_ptr dx;
    mp_float_ptr dy;
    mp_float_ptr dalpha;
    mp_float_ptr dbeta;

    cudaMalloc(&dx, sizeof(mp_float_t) * lenx);
    cudaMalloc(&dy, sizeof(mp_float_t) * leny);
    cudaMalloc(&dA, sizeof(mp_float_t) * lda * n);
    cudaMalloc(&dalpha, sizeof(mp_float_t));
    cudaMalloc(&dbeta, sizeof(mp_float_t));

    // Convert from MPFR
    convert_vector(hx, x, lenx);
    convert_vector(hy, y, leny);
    convert_matrix(hA, A, lda, n);
    mp_set_mpfr(&halpha, alpha);
    mp_set_mpfr(&hbeta, beta);

    //Copying to the GPU
    cudaMemcpy(dx, hx, sizeof(mp_float_t) * lenx, cudaMemcpyHostToDevice);
    cudaMemcpy(dA, hA, lda * n * sizeof(mp_float_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dalpha, &halpha, sizeof(mp_float_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dbeta, &hbeta, sizeof(mp_float_t), cudaMemcpyHostToDevice);

    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    for(int i = 0; i < REPEAT_TEST; i ++) {
        cudaMemcpy(dy, hy, sizeof(mp_float_t) * leny, cudaMemcpyHostToDevice);
        StartCudaTimer();
        cuda::mp_symv<32><<<blocks, threads>>>(uplo, n, dalpha, dA, lda, dx, incx, dbeta, dy, incy);
        EndCudaTimer();
    }
    PrintCudaTimer("took");
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    //Copying to the host
    cudaMemcpy(hy, dy, leny * sizeof(mp_float_t), cudaMemcpyDeviceToHost);
    print_mp_sum(hy, leny);

    //Cleanup
    delete [] hx;
    delete [] hy;
    delete [] hA;
    cudaFree(dx);
    cudaFree(dy);
    cudaFree(dA);
    cudaFree(dalpha);
    cudaFree(dbeta);
}

/********************* Main test *********************/

/*
 * Test for non-transposed matrix
 * x is of size n
 * y is of size m
 * a is of size lda * n, where the value of lda must be at least max(1, m).
 */
void test() {
    //Actual length of the vectors
    int lenx = (1 + (N - 1) * abs(INCX));
    int leny = (1 + (N - 1) * abs(INCY));

    //Inputs
    mpfr_t *vectorX = create_random_array(lenx, INP_BITS);
    mpfr_t *vectorY = create_random_array(leny, INP_BITS);
    mpfr_t *matrixA = create_random_array(LDA * N, INP_BITS);
    mpfr_t *alpha = create_random_array(1, INP_BITS);
    mpfr_t *beta = create_random_array(1, INP_BITS);
    //Launch tests
    test_openblas(UPLO, N, lenx, leny, alpha[0], matrixA, LDA, vectorX, INCX, beta[0], vectorY, INCY);
    test_double(UPLO, N, lenx, leny, alpha[0], matrixA, LDA, vectorX, beta[0], vectorY);
    test_mpfr(UPLO, N, alpha[0], matrixA, LDA, vectorX, beta[0], vectorY);
    test_cublas(UPLO, N, lenx, leny, alpha[0], matrixA, LDA, vectorX, INCX, beta[0], vectorY, INCY);
    test_double_symv_cuda(UPLO, N, lenx, leny, alpha[0], matrixA, LDA, vectorX, INCX, beta[0], vectorY, INCY);
    test_mpres_symv(UPLO, N, lenx, leny, alpha[0], matrixA, LDA, vectorX, INCX, beta[0], vectorY, INCY);

    checkDeviceHasErrors(cudaDeviceSynchronize());
    // cudaCheckErrors(); //CUMP gives failure

    //Cleanup
    for (int i = 0; i < LDA * N; i++) {
        mpfr_clear(matrixA[i]);
    }
    for (int i = 0; i < lenx; i++) {
        mpfr_clear(vectorX[i]);
    }
    for (int i = 0; i < leny; i++) {
        mpfr_clear(vectorY[i]);
    }
    mpfr_clear(alpha[0]);
    mpfr_clear(beta[0]);
    delete[] matrixA;
    delete[] vectorX;
    delete[] vectorY;
    delete[] alpha;
    delete[] beta;
    cudaDeviceReset();
}
int main() {

    initialize();

    //Start logging
    Logger::beginTestDescription(Logger::BLAS_SYMV_PERFORMANCE_TEST);
    Logger::printTestParameters(N * N, REPEAT_TEST, MP_PRECISION, MP_PRECISION_DEC);
    Logger::beginSection("Operation info:");
    Logger::printParam("Matrix rows and columns, N", N);
    Logger::printParam("LDA", LDA);
    Logger::printParam("INCX", INCX);
    Logger::printParam("INCY", INCY);
    Logger::printParam("UPLO", UPLO);
    Logger::printDash();
    Logger::beginSection("Additional info:");
    Logger::printParam("RNS_MODULI_SIZE", RNS_MODULI_SIZE);
#ifndef EXCLUDE_CAMPARY
    //Logger::printParam("CAMPARY_PRECISION (n-double)", CAMPARY_PRECISION);
#endif
    Logger::endSection(true);
    //Run the test
    test();
    //Finalize
    finalize();
    //End logging
    Logger::endTestDescription();

    return 0;
}