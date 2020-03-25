/*
 *  Performance test for BLAS GEMV routines
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

#include "omp.h"
#include "../../logger.cuh"
#include "../../timers.cuh"
#include "../../tsthelper.cuh"
#include "../../../src/blas/mpgemv.cuh"
#include "3rdparty.cuh"


#define M 100  // Number of matrix rows and the vector Y dimension
#define N 100  // Number of matrix columns and the vector X dimension
#define LDA (M) // Specifies the leading dimension of A as declared in the calling (sub)program.
#define TRANS "N" // Specifies the operation: if trans = 'N' or 'n', then y := alpha*A*x + beta*y; if trans = 'T' or 't' or 'C' or 'c' then y = alpha*A**T*x + beta*y (transposed matrix).
#define INCX 1 // Specifies the increment for the elements of x.
#define INCY 1 // Specifies the increment for the elements of y.
#define REPEAT_TEST 10 //Number of repeats

//Execution configuration for mpgemv
#define MPRES_CUDA_BLOCKS_FIELDS_ROUND   512
#define MPRES_CUDA_THREADS_FIELDS_ROUND  128
#define MPRES_CUDA_BLOCKS_RESIDUES       8192
#define MPRES_CUDA_THREADS_REDUCE        32

#define OPENBLAS_THREADS 4

int MP_PRECISION_DEC; //in decimal digits
int INP_BITS; //in bits
int INP_DIGITS; //in decimal digits

void setPrecisions(){
    MP_PRECISION_DEC = (int)(MP_PRECISION / 3.32 + 1);
    INP_BITS = (int)(MP_PRECISION / 4);
    INP_DIGITS = (int)(INP_BITS / 3.32 + 1);
}

void initialize(){
    cudaDeviceReset();
    rns_const_init();
    mp_const_init();
    setPrecisions();
    mp_real::mp_init(MP_PRECISION_DEC);
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();
}

void finalize(){
    mp_real::mp_finalize();
}

void print_double_sum(double *result, int v_length) {
    double print_result = 0;
    for (int i = 0; i < v_length; i++) {
        print_result += result[i];
    }
    printf("result: %.70f\n", print_result);
}

void print_mp_sum(mp_float_ptr result, int v_length) {
    mp_float_t print_result;
    print_result = MP_ZERO;

    mpfr_t mpfr_result;
    mpfr_init2(mpfr_result, MP_PRECISION * 10);
    mpfr_set_d(mpfr_result, 0.0, MPFR_RNDN);

    for (int i = 0; i < v_length; i+= 1) {
        mp_add(&print_result, &print_result, &result[i]);
    }

    mp_get_mpfr(mpfr_result, &print_result);
    mpfr_printf("result: %.70Rf \n", mpfr_result);
    mpfr_clear(mpfr_result);
}

void print_mpfr_sum(mpfr_t *result, int v_length) {
    mpfr_t tmp_sum;
    mpfr_init2(tmp_sum, MP_PRECISION * 10);
    mpfr_set_d(tmp_sum, 0.0, MPFR_RNDN);

    for (int i = 0; i < v_length; i++) {
        mpfr_add(tmp_sum, tmp_sum, result[i], MPFR_RNDN);
    }
    mpfr_printf("result: %.70Rf\n", tmp_sum);
    mpfr_clear(tmp_sum);
}

void convert_vector(mp_float_ptr dest, mpfr_t *source, int width){
    for( int i = 0; i < width; i++ ){
        mp_set_mpfr(&dest[i], source[i]);
    }
}

void convert_matrix(mp_float_ptr dest, mpfr_t *source, int rows, int cols){
    int width = rows * cols;
    for( int i = 0; i < width; i++ ){
        mp_set_mpfr(&dest[i], source[i]);
    }
}



/********************* GEMV implementations and benchmarks *********************/

/////////
// OpenBLAS
/////////
extern "C" void openblas_set_num_threads(int num_threads);

void openblas_test(const char *trans, const int m, const int n, int lenx, int leny, mpfr_t alpha, mpfr_t *A, int lda, mpfr_t *x, int incx, mpfr_t beta, mpfr_t *y, int incy){
    InitCpuTimer();
    Logger::printDash();
    PrintTimerName("[CPU] OpenBLAS gemv");

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
    for(int i = 0; i < REPEAT_TEST; i ++){
        for (int i = 0; i < leny; ++i) {
            dr[i] = dy[i];
        }
        StartCpuTimer();
        if(trans == "N" || trans == "n") {
            cblas_dgemv(CblasColMajor, CblasNoTrans, m, n, dalpha, dA, lda, dx, incx, dbeta, dr, incy);
        } else {
            cblas_dgemv(CblasColMajor, CblasTrans, m, n, dalpha, dA, lda, dx, incx, dbeta, dr, incy);
        }
        EndCpuTimer();
    }
    PrintCpuTimer("took");
    print_double_sum(dr, leny);
    delete [] dx;
    delete [] dy;
    delete [] dr;
    delete [] dA;
}


/////////
// MPFR (only for non-transposed matrix and unit strides)
/////////
void mpfr_gemv(int m, int n, mpfr_t alpha, mpfr_t *A, int lda, mpfr_t *x, mpfr_t beta, mpfr_t *y){
    #pragma omp parallel shared(m, n, A, x, y)
    {
        mpfr_t mul_acc, ax;
        mpfr_init2(ax, MP_PRECISION);
        mpfr_init2(mul_acc, MP_PRECISION);
        int i = 0;
        int j = 0;
        #pragma omp for
        for(i = 0; i < m; i++){
            mpfr_mul(y[i], beta, y[i], MPFR_RNDN);
        }

        for (j = 0; j < n; j++) {
            mpfr_mul(ax, alpha, x[j], MPFR_RNDN);
            #pragma omp for
            for (i = 0; i < m; i++) {
                mpfr_mul(mul_acc, ax, A[i + j * lda], MPFR_RNDN);
                mpfr_add(y[i], y[i], mul_acc, MPFR_RNDN);
            }
        }
        mpfr_clear(mul_acc);
        mpfr_clear(ax);
    }
}

void mpfr_test(int m, int n, mpfr_t alpha, mpfr_t *A, int lda, mpfr_t *x, mpfr_t beta, mpfr_t *y){
    InitCpuTimer();
    Logger::printDash();
    PrintTimerName("[CPU] MPFR gemv");

    // Init
    mpfr_t *result = new mpfr_t[m];
    #pragma omp parallel for
    for(int i = 0; i < m; i++){
        mpfr_init2(result[i], MP_PRECISION);
    }

    // Launch
    for(int i = 0; i < REPEAT_TEST; i++){
        #pragma omp parallel for
        for(int j = 0; j < m; j ++){
            mpfr_set(result[j], y[j], MPFR_RNDN);
        }
        StartCpuTimer();
        mpfr_gemv(m, n, alpha, A, lda, x, beta, result);
        EndCpuTimer();
    }
    PrintCpuTimer("took");
    print_mpfr_sum(result, m);

    //Cleanup
    for(int i = 0; i < m; i++){
        mpfr_clear(result[i]);
    }
    delete [] result;
}

/////////
// ARPREC (only for non-transposed matrix and unit strides)
/////////
void arprec_gemv(int m, int n, mp_real alpha, mp_real *A, int lda, mp_real *x, mp_real beta, mp_real *y){
    mp_real temp;
    int i = 0;
    #pragma omp parallel for
    for (i = 0; i < m; i++) {
        y[i] = beta * y[i];
    }

    for (int j = 0; j < n; j++) {
        temp = alpha * x[j];
        #pragma omp parallel for
        for (i = 0; i < m; i++) {
            y[i] = y[i] + temp * A[i + j * lda];
        }
    }
}

void arprec_test(int m, int n, mpfr_t alpha, mpfr_t *A, int lda, mpfr_t *x, mpfr_t beta, mpfr_t *y){
    InitCpuTimer();
    Logger::printDash();
    PrintTimerName("[CPU] ARPREC gemv");

    //Init
    mp_real *lx = new mp_real[n];
    mp_real *ly = new mp_real[m];
    mp_real *lA = new mp_real[lda * n];
    mp_real mp_alpha;
    mp_real mp_beta;

    //Convert from MPFR
    #pragma omp parallel for
    for(int i = 0; i < lda * n; i++){
        lA[i].read(convert_to_string_sci(A[i], INP_DIGITS));
    }
    #pragma omp parallel for
    for(int i = 0; i < n; i++){
        lx[i].read(convert_to_string_sci(x[i], INP_DIGITS));
    }
    mp_alpha.read(convert_to_string_sci(alpha, INP_DIGITS));
    mp_beta.read(convert_to_string_sci(beta, INP_DIGITS));

    //Launch
    for(int j = 0; j < REPEAT_TEST; j ++){
        for(int i = 0; i < m; i++){
            ly[i].read(convert_to_string_sci(y[i], INP_DIGITS));
        }
        StartCpuTimer();
        arprec_gemv(m, n, mp_alpha, lA, lda, lx, mp_beta, ly);
        EndCpuTimer();
    }
    PrintCpuTimer("took");
    for (int i = 1; i < m; i++) {
        ly[0] += ly[i];
    }
    printf("result %s \n", ly[0].to_string().c_str());

    //Cleanup
    delete [] lA;
    delete [] lx;
    delete [] ly;
}

/////////
// MPACK
/////////
void mpack_test(const char * trans, int m, int n, int lenx, int leny, mpfr_t alpha, mpfr_t *A, int lda, mpfr_t *x, int incx, mpfr_t beta, mpfr_t *y, int incy){
    Logger::printDash();
    InitCpuTimer();
    PrintTimerName("[CPU] MPACK gemv");

    //Set precision
    mpfr::mpreal::set_default_prec ( MP_PRECISION );

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
    for(int j = 0; j < REPEAT_TEST; j ++){
        #pragma omp parallel for
        for(int i = 0; i < leny; i++){
            ly[i] = y[i];
        }
        StartCpuTimer();
        Rgemv(trans, m, n, lalpha, lA, lda, lx, incx, lbeta, ly, incy);
        EndCpuTimer();
    }
    PrintCpuTimer("took");

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

/////////
// MPRES-BLAS (structure of arrays)
/////////
void mpres_test(const char *trans, int m, int n, int lenx, int leny, mpfr_t alpha, mpfr_t *A, int lda, mpfr_t *x, int incx, mpfr_t beta, mpfr_t *y, int incy){
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
    bool noTrans = (TRANS == "N" || TRANS == "n");
    cuda::mp_array_init(dbuf1, noTrans ? n : m);
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
        cuda::mpgemv<
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
}

/////////
// MPRES-BLAS straightforward (array of structures)
// Each multiple-precision operation is performed by a single thread
/////////
__global__ static void mp_scal_straightforward(int n, mp_float_ptr alpha, mp_float_ptr x){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i < n){
        cuda::mp_mul(&x[i], &alpha[0], &x[i]);
    }
}

__global__ static void mp_gemv_straightforward(int m, int n, mp_float_ptr alpha, mp_float_ptr A, int lda, mp_float_ptr x, mp_float_ptr beta, mp_float_ptr y){
    mp_float_t sum;
    int i = (threadIdx.x + blockIdx.x * blockDim.x);
    if(i < m){
        cuda::mp_mul(&y[i], &beta[0], &y[i]);
    }
    __syncthreads();
    for (int j = 0; j < n; j++) {
        if( i < m ){
            cuda::mp_mul(&sum, &x[j], &A[i + j * lda]);
            cuda::mp_add(&y[i], &y[i], &sum);
        }
        __syncthreads();
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
        mp_gemv_straightforward<<<blocks_gemv, threads>>>(m, n, dalpha, dA, lda, dx, dbeta, dy);
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

/********************* Main test *********************/

/*
 * Test for non-transposed matrix
 * x is of size n
 * y is of size m
 * a is of size lda * n, where the value of lda must be at least max(1, m).
 */
void testNoTrans(){
    //Actual length of the vectors
    int lenx = (1 + (N - 1) * abs(INCX));
    int leny = (1 + (M - 1) * abs(INCY));

    //Inputs
    mpfr_t *vectorX = create_random_array(lenx, INP_BITS);
    mpfr_t *vectorY = create_random_array(leny, INP_BITS);
    mpfr_t *matrixA = create_random_array(LDA * N, INP_BITS);
    mpfr_t *alpha = create_random_array(1, INP_BITS);
    mpfr_t *beta = create_random_array(1, INP_BITS);

    //Multiple-precision tests
    openblas_test(TRANS, M, N, lenx, leny, alpha[0], matrixA, LDA, vectorX, INCX, beta[0], vectorY, INCY);
    mpfr_test(M, N, alpha[0], matrixA, LDA, vectorX, beta[0], vectorY);
    arprec_test(M, N, alpha[0], matrixA, LDA, vectorX, beta[0], vectorY);
    mpack_test(TRANS, M, N, lenx, leny, alpha[0], matrixA, LDA, vectorX, INCX, beta[0], vectorY, INCY);
    mpres_test(TRANS, M, N, lenx, leny, alpha[0], matrixA, LDA, vectorX, INCX, beta[0], vectorY, INCY);
    mpres_test_straightforward(M, N, alpha[0], matrixA, LDA, vectorX, beta[0], vectorY);
    garprec_gemv_test(M, N, alpha[0], matrixA, LDA, vectorX, beta[0], vectorY, MP_PRECISION_DEC, INP_DIGITS, REPEAT_TEST);
    campary_gemv_test<CAMPARY_PRECISION>(M, N, alpha[0], matrixA, LDA, vectorX, beta[0], vectorY, INP_DIGITS, REPEAT_TEST);
    cump_gemv_test(M, N, alpha[0], matrixA, LDA, vectorX, beta[0], vectorY, MP_PRECISION, INP_DIGITS, REPEAT_TEST);

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
}

/*
 * Test for transposed matrix
 * x is of size m
 * y is of size n
 * a is of size lda * n, where the value of lda must be at least max(1, m).
 */
void testTrans(){

    //Actual length of the vectors
    int lenx = (1 + (M - 1) * abs(INCX));
    int leny = (1 + (N - 1) * abs(INCY));

    //Inputs
    mpfr_t *vectorX = create_random_array(lenx, INP_BITS);
    mpfr_t *vectorY = create_random_array(leny, INP_BITS);
    mpfr_t *matrixA = create_random_array(LDA * N, INP_BITS);
    mpfr_t *alpha = create_random_array(1, INP_BITS);
    mpfr_t *beta = create_random_array(1, INP_BITS);

    //Multiple-precision tests
    openblas_test(TRANS, M, N, lenx, leny, alpha[0], matrixA, LDA, vectorX, INCX, beta[0], vectorY, INCY);
    mpack_test(TRANS, M, N, lenx, leny, alpha[0], matrixA, LDA, vectorX, INCX, beta[0], vectorY, INCY);
    mpres_test(TRANS, M, N, lenx, leny, alpha[0], matrixA, LDA, vectorX, INCX, beta[0], vectorY, INCY);

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
}


int main(){

    initialize();

    //Start logging
    Logger::beginTestDescription(Logger::BLAS_GEMV_PERFORMANCE_TEST);
    Logger::printTestParameters(N * M, REPEAT_TEST, MP_PRECISION, MP_PRECISION_DEC);
    Logger::beginSection("Operation info:");
    Logger::printParam("Matrix rows, M", M);
    Logger::printParam("Matrix columns, N", N);
    Logger::printParam("LDA", LDA);
    Logger::printParam("TRANS", TRANS);
    Logger::printDash();
    Logger::beginSection("Additional info:");
    Logger::printParam("RNS_MODULI_SIZE", RNS_MODULI_SIZE);
    Logger::printParam("MPRES_CUDA_BLOCKS_FIELDS_ROUND", MPRES_CUDA_BLOCKS_FIELDS_ROUND);
    Logger::printParam("MPRES_CUDA_THREADS_FIELDS_ROUND", MPRES_CUDA_THREADS_FIELDS_ROUND);
    Logger::printParam("MPRES_CUDA_BLOCKS_RESIDUES", MPRES_CUDA_BLOCKS_RESIDUES);
    Logger::printParam("MPRES_CUDA_THREADS_REDUCE", MPRES_CUDA_THREADS_REDUCE);
    Logger::printParam("CAMPARY_PRECISION (n-double)", CAMPARY_PRECISION);
    Logger::endSection(true);

    //Run the test
    if(TRANS == "N" || TRANS == "n") {
        testNoTrans();
    } else{
        testTrans();
    }

    //Finalize
    finalize();

    //End logging
    Logger::endTestDescription();

    return 0;
}