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
#define EXCLUDE_CUBLAS
#define EXCLUDE_GARPREC
#define EXCLUDE_CAMPARY
#define EXCLUDE_CUMP

#include "../../logger.cuh"
#include "../../timers.cuh"
#include "../../tsthelper.cuh"
#include "../../../src/mparray.cuh"
#include "../../../src/arith/mul.cuh"
#include "../../../src/blas/gemv.cuh"
#include "3rdparty.cuh"


#define N 1000  // Number of matrix rows / column and the vectors dimension
#define LDA (N) // Specifies the leading dimension of A as declared in the calling (sub)program.
#define UPLO mblas_upper // Specifies whether the upper or lower triangular part of the array a is used.
#define INCX 1 // Specifies the increment for the elements of x.
#define INCY 1 // Specifies the increment for the elements of y.
#define REPEAT_TEST 1 //Number of repeats

//Execution configuration for mp_gemv
#define MPRES_CUDA_BLOCKS_FIELDS_ROUND 256
#define MPRES_CUDA_THREADS_FIELDS_ROUND 128
#define MPRES_CUDA_BLOCKS_RESIDUES 256
#define MPRES_CUDA_THREADS_REDUCE 32

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

static int get_mtx_index_up(const int i, const int j, const int lda){
    return (i <= j) ? i + j * lda : j + i * lda;
}

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

void mpfr_symv_up(int n, mpfr_t alpha, mpfr_t *A, int lda, mpfr_t *x, mpfr_t beta, mpfr_t *y) {
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

void
test_openblas(enum mblas_uplo_type uplo, const int n, int lenx, int leny, mpfr_t alpha, mpfr_t *A, int lda, mpfr_t *x,
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
        }StartCpuTimer();
        if (uplo == mblas_upper) {
            cblas_dsymv(CblasColMajor, CblasUpper, n, dalpha, dA, lda, dx, incx, dbeta, dr, incy);
        } else {
            cblas_dsymv(CblasColMajor, CblasLower, n, dalpha, dA, lda, dx, incx, dbeta, dr, incy);
        }EndCpuTimer();
    }
    PrintCpuTimer("took");
    print_double_sum(dr, leny);
    delete[] dx;
    delete[] dy;
    delete[] dr;
    delete[] dA;
}

/////////
// double
/////////
void double_symv_up(int n, double alpha, const double *A, int lda, const double *x, double beta, double *y) {
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
}

void test_double(enum mblas_uplo_type uplo, const int n, int lenx, int leny, mpfr_t alpha, mpfr_t *A, int lda, mpfr_t *x, int incx, mpfr_t beta, mpfr_t *y, int incy) {
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
        if (uplo == mblas_upper) {
            double_symv_up(n, dalpha, dA, lda, dx, dbeta, dr);
        } else {
            //TODO:
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
// double CUDA
/////////
/*__global__ static void double_symv_up_cuda(int n, double alpha, const double *A, int lda, const double *x, double beta, double *y) {
    unsigned int threadId = threadIdx.x + blockIdx.x * blockDim.x;
    //Работаем по строкам, генерируя один элемент вектра y.
    if (threadId < n) {
        double dot = 0;
        for (int colId = 0; colId < n; colId++) {
            double da = alpha * x[colId];
            if(threadId <= colId){
                dot += da * A[threadId + colId * lda];
            } else{
                dot += da * A[colId + threadId * lda];
            }
        }
        y[threadId] *= beta;
        y[threadId] += dot;
    }
}

void test_double_symv_cuda(enum mblas_uplo_type uplo, const int n, int lenx, int leny, mpfr_t alpha, mpfr_t *A, int lda, mpfr_t *x, int incx, mpfr_t beta, mpfr_t *y, int incy) {
    InitCudaTimer();
    Logger::printDash();
    PrintTimerName("[GPU] double SYMV");

    //Execution configuration
    int threads = 32;
    int blocks = n / threads + 1;
    printf("\tExec. config: blocks = %i, threads = %i\n", blocks, threads);

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


}*/


/////////
// cuBLAS
/////////
/*
void cublas_test(double *x, double *y, int n){
    InitCudaTimer();
    Logger::printDash();
    PrintTimerName("[GPU] cuBLAS dot");

    cublasStatus_t stat;
    cublasHandle_t handle;

    stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("cuBLAS initialization failed\n");
        return;
    }

    double *dev_x, *dev_y;
    double *res = new double[n];
    cudaMalloc(&dev_x, sizeof(double) * n);
    cudaMalloc(&dev_y, sizeof(double) * n);
    cublasSetVector(n, sizeof(double), x, 1, dev_x, 1);
    cublasSetVector(n, sizeof(double), y, 1, dev_y, 1);

    StartCudaTimer();
    for(int i = 0; i < REPEAT_TEST; i ++) {
        cublasDdot(handle, n, dev_x, 1, dev_y, 1, res);
    }
    EndCudaTimer();
    PrintCudaTimer("took");
    printf("result: %lf\n", *res);

    cublasDestroy ( handle );
    cudaFree(dev_x);
    cudaFree(dev_y);
    delete [] res;
}
*/





/////////
// MPFR upper serial
/////////


void mpfr_test(enum mblas_uplo_type uplo, int n, mpfr_t alpha, mpfr_t *A, int lda, mpfr_t *x, mpfr_t beta, mpfr_t *y) {
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
        if (uplo == mblas_upper) {
            mpfr_symv_up(n, alpha, A, lda, x, beta, result);
        }
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
// MPFR upper openmp
/////////
void mpfr_symv_up_omp(int n, mpfr_t alpha, mpfr_t *A, int lda, mpfr_t *x, mpfr_t beta, mpfr_t *y) {
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
}

void mpfr_test_omp(enum mblas_uplo_type uplo, int n, mpfr_t alpha, mpfr_t *A, int lda, mpfr_t *x, mpfr_t beta, mpfr_t *y) {
    InitCpuTimer();
    Logger::printDash();
    PrintTimerName("[CPU] MPFR symv omp");

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
        }StartCpuTimer();
        if (uplo == mblas_upper) {
            mpfr_symv_up_omp(n, alpha, A, lda, x, beta, result);
        }EndCpuTimer();
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
// MPRES-BLAS (structure of arrays)
/////////
/*void mpres_test(enum mblas_trans_type trans, int m, int n, int lenx, int leny, mpfr_t alpha, mpfr_t *A, int lda, mpfr_t *x, int incx, mpfr_t beta, mpfr_t *y, int incy){
    InitCudaTimer();
    Logger::printDash();
    PrintTimerName("[GPU] MPRES-BLAS gemv");

    // Host data
    mp_float_ptr hx = new mp_float_t[lenx];
    mp_float_ptr hy = new mp_float_t[leny];
    mp_float_ptr hA = new mp_float_t[lda * n];
    mp_float_t halpha;
    mp_float_t hbeta;

    //GPU data
    mp_array_t dx;
    mp_array_t dy;
    mp_array_t dA;
    mp_array_t dalpha;
    mp_array_t dbeta;
    mp_array_t dbuf1;
    mp_array_t dbuf2;

    //Init data
    cuda::mp_array_init(dx, lenx);
    cuda::mp_array_init(dy, leny);
    cuda::mp_array_init(dA, lda * n);
    cuda::mp_array_init(dalpha, 1);
    cuda::mp_array_init(dbeta, 1);
    cuda::mp_array_init(dbuf1, (trans == mblas_no_trans) ? n : m);
    cuda::mp_array_init(dbuf2, m * n);

    // Convert from MPFR
    convert_vector(hx, x, lenx);
    convert_vector(hy, y, leny);
    convert_matrix(hA, A, lda, n);
    mp_set_mpfr(&halpha, alpha);
    mp_set_mpfr(&hbeta, beta);

    //Copying to the GPU
    cuda::mp_array_host2device(dx, hx, lenx);
    cuda::mp_array_host2device(dA, hA, lda * n);
    cuda::mp_array_host2device(dalpha, &halpha, 1);
    cuda::mp_array_host2device(dbeta, &hbeta, 1);

    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();
    //Launch
    for (int i = 0; i < REPEAT_TEST; i++) {
        cuda::mp_array_host2device(dy, hy, leny);
        StartCudaTimer();

        cuda::mp_gemv<
                MPRES_CUDA_BLOCKS_FIELDS_ROUND,
                MPRES_CUDA_THREADS_FIELDS_ROUND,
                MPRES_CUDA_BLOCKS_RESIDUES,
                MPRES_CUDA_THREADS_REDUCE>
                (trans, m, n, dalpha, dA, lda, dx, incx, dbeta, dy, incy, dbuf1, dbuf2);
        EndCudaTimer();
    }
    PrintCudaTimer("took");
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    //Copying to the host
    cuda::mp_array_device2host(hy, dy, leny);
    print_mp_sum(hy, leny);

    //Cleanup
    delete [] hx;
    delete [] hy;
    delete [] hA;
    cuda::mp_array_clear(dx);
    cuda::mp_array_clear(dy);
    cuda::mp_array_clear(dA);
    cuda::mp_array_clear(dalpha);
    cuda::mp_array_clear(dbeta);
    cuda::mp_array_clear(dbuf1);
    cuda::mp_array_clear(dbuf2);
}*/

/////////
// MPRES-BLAS straightforward (array of structures)
// Each multiple-precision operation is performed by a single thread
/////////
/*
__global__ static void mp_scal_straightforward(int n, mp_float_ptr alpha, mp_float_ptr x){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i < n){
        cuda::mp_mul(&x[i], &alpha[0], &x[i]);
    }
}

__global__ static void mp_gemv_straightforward(int m, int n, mp_float_ptr A, int lda, mp_float_ptr x, mp_float_ptr beta, mp_float_ptr y) {
    unsigned int threadId = threadIdx.x + blockIdx.x * blockDim.x;
    if (threadId < m) {
        mp_float_t prod;
        mp_float_t dot = cuda::MP_ZERO;
        for (int colId = 0; colId < n; colId++) {
            cuda::mp_mul(&prod, &x[colId], &A[colId * lda + threadId]);
            cuda::mp_add(&dot, &dot, &prod);
        }
        cuda::mp_mul(&y[threadId], &beta[0], &y[threadId]);
        cuda::mp_add(&y[threadId], &y[threadId], &dot);
    }
}

void mpres_test_straightforward(int m, int n, mpfr_t alpha, mpfr_t *A, int lda, mpfr_t *x, mpfr_t beta, mpfr_t *y){
    InitCudaTimer();
    Logger::printDash();
    PrintTimerName("[GPU] MPRES-BLAS gemv (straightforward)");

    //Execution configuration
    int threads = 32;
    int blocks_gemv = m / (threads) + (m % (threads) ? 1 : 0);
    int blocks_scal = n / (threads) + (n % (threads) ? 1 : 0);

    // Host data
    mp_float_t halpha;
    mp_float_t hbeta;
    mp_float_ptr hx = new mp_float_t[n];
    mp_float_ptr hy = new mp_float_t[m];
    mp_float_ptr hA = new mp_float_t[lda * n];

    // GPU data
    mp_float_ptr dA;
    mp_float_ptr dx;
    mp_float_ptr dy;
    mp_float_ptr dalpha;
    mp_float_ptr dbeta;

    cudaMalloc(&dA, sizeof(mp_float_t) * lda * n);
    cudaMalloc(&dx, sizeof(mp_float_t) * n);
    cudaMalloc(&dy, sizeof(mp_float_t) * m);
    cudaMalloc(&dalpha, sizeof(mp_float_t));
    cudaMalloc(&dbeta, sizeof(mp_float_t));

    // Convert from MPFR
    mp_set_mpfr(&halpha, alpha);
    mp_set_mpfr(&hbeta, beta);
    convert_vector(hx, x, n);
    convert_vector(hy, y, m);
    convert_matrix(hA, A, lda, n);

    //Copying to the GPU
    cudaMemcpy(dA, hA, lda * n * sizeof(mp_float_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dalpha, &halpha, sizeof(mp_float_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dbeta, &hbeta, sizeof(mp_float_t), cudaMemcpyHostToDevice);

    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    //Launch
    for (int i = 0; i < REPEAT_TEST; i++) {
        cudaMemcpy(dx, hx, n * sizeof(mp_float_t), cudaMemcpyHostToDevice);
        cudaMemcpy(dy, hy, m * sizeof(mp_float_t), cudaMemcpyHostToDevice);
        StartCudaTimer();
        mp_scal_straightforward<<<blocks_scal, threads>>>(n, dalpha, dx);
        mp_gemv_straightforward<<<blocks_gemv, threads>>>(m, n, dA, lda, dx, dbeta, dy);
        EndCudaTimer();
    }
    PrintCudaTimer("took");
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    //Copying to the host
    cudaMemcpy(hy, dy, m * sizeof(mp_float_t), cudaMemcpyDeviceToHost);
    print_mp_sum(hy, m);

    //Cleanup
    delete [] hA;
    delete [] hx;
    delete [] hy;
    cudaFree(dA);
    cudaFree(dx);
    cudaFree(dy);
    cudaFree(dalpha);
    cudaFree(dbeta);
}
*/

/********************* Main test *********************/

/*
 * Test for non-transposed matrix
 * x is of size n
 * y is of size m
 * a is of size lda * n, where the value of lda must be at least max(1, m).
 */
void testNoTrans() {
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
    test_double(UPLO, N, lenx, leny, alpha[0], matrixA, LDA, vectorX, INCX, beta[0], vectorY, INCY);
    mpfr_test(UPLO, N, alpha[0], matrixA, LDA, vectorX, beta[0], vectorY);
    mpfr_test_omp(UPLO, N, alpha[0], matrixA, LDA, vectorX, beta[0], vectorY);
    //arprec_test(M, N, alpha[0], matrixA, LDA, vectorX, beta[0], vectorY);
#ifndef EXCLUDE_MPACK
    //mpack_gemv_test(TRANS, M, N, lenx, leny, alpha[0], matrixA, LDA, vectorX, INCX, beta[0], vectorY, INCY, MP_PRECISION, REPEAT_TEST);
#endif
    //mpres_test(mblas_no_trans, M, N, lenx, leny, alpha[0], matrixA, LDA, vectorX, INCX, beta[0], vectorY, INCY);
    //mpres_test_straightforward(M, N, alpha[0], matrixA, LDA, vectorX, beta[0], vectorY);
#ifndef EXCLUDE_GARPREC
    //garprec_gemv_test(M, N, alpha[0], matrixA, LDA, vectorX, beta[0], vectorY, MP_PRECISION_DEC, INP_DIGITS, REPEAT_TEST);
#endif
#ifndef EXCLUDE_CAMPARY
    //campary_gemv_test<CAMPARY_PRECISION>(M, N, alpha[0], matrixA, LDA, vectorX, beta[0], vectorY, INP_DIGITS, REPEAT_TEST);
#endif
#ifndef EXCLUDE_CUMP
    //cump_gemv_test(M, N, alpha[0], matrixA, LDA, vectorX, beta[0], vectorY, MP_PRECISION, INP_DIGITS, REPEAT_TEST);
#endif

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

/*
 * Test for transposed matrix
 * x is of size m
 * y is of size n
 * a is of size lda * n, where the value of lda must be at least max(1, m).
 */
/*void testTrans(){

    //Actual length of the vectors
    int lenx = (1 + (M - 1) * abs(INCX));
    int leny = (1 + (N - 1) * abs(INCY));

    //Inputs
    mpfr_t *vectorX = create_random_array(lenx, INP_BITS);
    mpfr_t *vectorY = create_random_array(leny, INP_BITS);
    mpfr_t *matrixA = create_random_array(LDA * N, INP_BITS);
    mpfr_t *alpha = create_random_array(1, INP_BITS);
    mpfr_t *beta = create_random_array(1, INP_BITS);

    //Launch tests
    openblas_test(TRANS, M, N, lenx, leny, alpha[0], matrixA, LDA, vectorX, INCX, beta[0], vectorY, INCY);
    #ifndef EXCLUDE_MPACK
        mpack_gemv_test(TRANS, M, N, lenx, leny, alpha[0], matrixA, LDA, vectorX, INCX, beta[0], vectorY, INCY, MP_PRECISION, REPEAT_TEST);
    #endif
    mpres_test(mblas_trans, M, N, lenx, leny, alpha[0], matrixA, LDA, vectorX, INCX, beta[0], vectorY, INCY);

    checkDeviceHasErrors(cudaDeviceSynchronize());
    // cudaCheckErrors(); //CUMP gives failure

    //Cleanup
    for(int i = 0; i < LDA * N; i++){
        mpfr_clear(matrixA[i]);
    }
    for(int i = 0; i < lenx; i++){
        mpfr_clear(vectorX[i]);
    }
    for(int i = 0; i < leny; i++){
        mpfr_clear(vectorY[i]);
    }

    mpfr_clear(alpha[0]);
    mpfr_clear(beta[0]);
    delete [] matrixA;
    delete [] vectorX;
    delete [] vectorY;
    delete [] alpha;
    delete [] beta;
    cudaDeviceReset();
}*/


int main() {

    initialize();

    //Start logging
    Logger::beginTestDescription(Logger::BLAS_SYMV_PERFORMANCE_TEST);
    Logger::printTestParameters(N * N, REPEAT_TEST, MP_PRECISION, MP_PRECISION_DEC);
    Logger::beginSection("Operation info:");
    Logger::printParam("Matrix rows and columns, N", N);
    Logger::printParam("LDA", LDA);
    Logger::printParam("UPLO", UPLO);
    Logger::printDash();
    Logger::beginSection("Additional info:");
    Logger::printParam("RNS_MODULI_SIZE", RNS_MODULI_SIZE);
    Logger::printParam("MPRES_CUDA_BLOCKS_FIELDS_ROUND", MPRES_CUDA_BLOCKS_FIELDS_ROUND);
    Logger::printParam("MPRES_CUDA_THREADS_FIELDS_ROUND", MPRES_CUDA_THREADS_FIELDS_ROUND);
    Logger::printParam("MPRES_CUDA_BLOCKS_RESIDUES", MPRES_CUDA_BLOCKS_RESIDUES);
    Logger::printParam("MPRES_CUDA_THREADS_REDUCE", MPRES_CUDA_THREADS_REDUCE);
#ifndef EXCLUDE_CAMPARY
    //Logger::printParam("CAMPARY_PRECISION (n-double)", CAMPARY_PRECISION);
#endif
    Logger::endSection(true);

    //Run the test
    testNoTrans();

    //Finalize
    finalize();

    //End logging
    Logger::endTestDescription();

    return 0;
}