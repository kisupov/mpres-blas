/*
 *  Performance test for BLAS GEMM routines
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

/*
 * Exclude some benchmarks
 */
#define EXCLUDE_MPACK
#define EXCLUDE_XBLAS
#define EXCLUDE_ARPREC
#define EXCLUDE_MPDECIMAL
#define EXCLUDE_CUBLAS
#define EXCLUDE_GARPREC
#define EXCLUDE_CAMPARY
#define EXCLUDE_CUMP

#include "../../logger.cuh"
#include "../../timers.cuh"
#include "../../tsthelper.cuh"
#include "../../../src/mparray.cuh"
#include "../../../src/blas/gemm.cuh"
#include "blas/external/3rdparty.cuh"


#define M 300  // Specifies the number of rows of the matrix op(A) and of the matrix C.
#define N 300  // Specifies the number of columns of the matrix op(B) and the number of columns of the matrix C.
#define K 300  // Specifies the number of columns of the matrix op(A) and the number of rows of the matrix op(B).
#define LDA (M) // Specifies the leading dimension of A as declared in the calling (sub)program.
#define LDB (K) // Specifies the leading dimension of B as declared in the calling (sub)program.
#define LDC (M) // Specifies the leading dimension of C as declared in the calling (sub)program.
#define TRANSA "N" // Specifies the form of op(A) used in the matrix multiplication
#define TRANSB "N" // Specifies the form of op(B) used in the matrix multiplication
#define REPEAT_TEST 1 //Number of repeats

//Execution configuration for mp_gemm
#define MPRES_BLOCK_SIZE_X_ESI 32
#define MPRES_BLOCK_SIZE_Y_ESI 1
#define MPRES_GRID_SIZE_X_DIGITS 128
#define MPRES_GRID_SIZE_Y_DIGITS 64
#define MPRES_BLOCK_SIZE_MATRIX_MULT 16

#define OPENBLAS_THREADS 4

int MP_PRECISION_DEC; //in decimal digits
int INP_BITS; //in bits
int INP_DIGITS; //in decimal digits

static void setPrecisions(){
    MP_PRECISION_DEC = (int)(MP_PRECISION / 3.32 + 1);
    INP_BITS = (int)(MP_PRECISION / 4);
    INP_DIGITS = (int)(INP_BITS / 3.32 + 1);
}

static void initialize(){
    cudaDeviceReset();
    rns_const_init();
    mp_const_init();
    setPrecisions();
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();
}

static void finalize(){
}


/********************* GEMM implementations and benchmarks *********************/

/////////
// OpenBLAS
/////////
extern "C" void openblas_set_num_threads(int num_threads);

void openblas_test(CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB, const int m, const int n, const int k, mpfr_t alpha, mpfr_t *A, const int lda, mpfr_t *B, const int ldb, mpfr_t beta, mpfr_t *C, const int ldc){
    InitCpuTimer();
    Logger::printDash();
    PrintTimerName("[CPU] OpenBLAS gemm");

    openblas_set_num_threads(OPENBLAS_THREADS);

    //CPU data
    double *dA = new double[lda * k];
    double *dB = new double[ldb * n];
    double *dC = new double[ldc * n];
    double dalpha = mpfr_get_d(alpha, MPFR_RNDN);
    double dbeta = mpfr_get_d(beta, MPFR_RNDN);

    for (int i = 0; i < lda * k; i++) {
        dA[i] = mpfr_get_d(A[i], MPFR_RNDN);
    }

    for (int i = 0; i < ldb * n; i++) {
        dB[i] = mpfr_get_d(B[i], MPFR_RNDN);
    }

    //Launch
    for(int i = 0; i < REPEAT_TEST; i ++){
        for (int j = 0; j < ldc * n; j++) {
            dC[j] = mpfr_get_d(C[j], MPFR_RNDN);
        }
        StartCpuTimer();
        cblas_dgemm(CblasColMajor, transA, transB, m, n, k, dalpha, dA, lda, dB, ldb, dbeta, dC, ldc);
        EndCpuTimer();
    }
    PrintCpuTimer("took");
    print_double_sum(dC, ldc * n);
    delete [] dA;
    delete [] dB;
    delete [] dC;
}

/////////
// MPRES-BLAS
/////////
void mpres_test_notrans(const int m, const int n, const int k, mpfr_t alpha, mpfr_t *A, const int lda, mpfr_t *B, const int ldb, mpfr_t beta, mpfr_t *C, const int ldc){
    InitCudaTimer();
    Logger::printDash();
    PrintTimerName("[GPU] MPRES-BLAS gemm");

    // Host data
    mp_float_ptr hA = new mp_float_t[lda * k];
    mp_float_ptr hB = new mp_float_t[ldb * n];
    mp_float_ptr hC = new mp_float_t[ldc * n];
    mp_float_t halpha;
    mp_float_t hbeta;

    //GPU data
    mp_array_t dA;
    mp_array_t dB;
    mp_array_t dC;
    mp_array_t dalpha;
    mp_array_t dbeta;
    mp_array_t dbuffer;

    //Init data
    cuda::mp_array_init(dA, lda * k);
    cuda::mp_array_init(dB, ldb * n);
    cuda::mp_array_init(dC, ldc * n);
    cuda::mp_array_init(dalpha, 1);
    cuda::mp_array_init(dbeta, 1);
    cuda::mp_array_init(dbuffer, m * n);

    // Convert from MPFR
    convert_matrix(hA, A, lda, k);
    convert_matrix(hB, B, ldb, n);
    convert_matrix(hC, C, ldc, n);
    mp_set_mpfr(&halpha, alpha);
    mp_set_mpfr(&hbeta, beta);

    //Copying to the GPU
    cuda::mp_array_host2device(dA, hA, lda * k);
    cuda::mp_array_host2device(dB, hB, ldb * n);
    cuda::mp_array_host2device(dalpha, &halpha, 1);
    cuda::mp_array_host2device(dbeta, &hbeta, 1);

    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();
    //Launch
    for (int i = 0; i < REPEAT_TEST; i++) {
        cuda::mp_array_host2device(dC, hC, ldc * n);
        StartCudaTimer();
        cuda::mp_gemm<
                MPRES_BLOCK_SIZE_X_ESI,
                MPRES_BLOCK_SIZE_Y_ESI,
                MPRES_GRID_SIZE_X_DIGITS,
                MPRES_GRID_SIZE_Y_DIGITS,
                MPRES_BLOCK_SIZE_MATRIX_MULT>
                (mblas_no_trans, mblas_no_trans, m, n, k, dalpha, dA, lda, dB, ldb, dbeta, dC, ldc, dbuffer);
        EndCudaTimer();
    }
    PrintCudaTimer("took");
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    //Copying to the host
    cuda::mp_array_device2host(hC, dC, ldc * n);
    print_mp_sum(hC, ldc * n);

    //Cleanup
    delete [] hA;
    delete [] hB;
    delete [] hC;
    cuda::mp_array_clear(dA);
    cuda::mp_array_clear(dB);
    cuda::mp_array_clear(dC);
    cuda::mp_array_clear(dalpha);
    cuda::mp_array_clear(dbeta);
    cuda::mp_array_clear(dbuffer);
}


/////////
// MPRES-BLAS (straightforward implementation)
// Only for non-transposed matrices
/////////
__global__ void mpgemm_straightforward_kernel(const unsigned int m, const unsigned int n, const unsigned int k, mp_array_t alpha, mp_array_t A, const int lda, mp_array_t B, const int ldb, mp_array_t C, const int ldc) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int indexC = row + col * ldc;
    if(col < n && row < m){
        mp_float_t mul = cuda::MP_ZERO;
        mp_float_t sum = cuda::MP_ZERO;
        for(int i = 0; i < k; i++){
            // if(col == 0 && row == 0)
            // cuda::mp_mul(&mul, B, (col * ldb + i), B, (col * ldb + i));
            cuda::mp_mul(&mul, A, (lda * i + row), B, (col * ldb + i) );
            cuda::mp_mul(&mul, mul, alpha, 0);
            cuda::mp_add(&sum, sum, mul);
        }
        cuda::mp_add(C, indexC, C, indexC, sum);
    }
}

template<int blockDim1x, int blockDim1y, int gridDim2x, int gridDim2y, int blockDim3>
void mpgemm_straightforward(const int m, const int n, const int k, mp_array_t &alpha, mp_array_t &A, const int lda, mp_array_t &B, const int ldb, mp_array_t &beta, mp_array_t &C, const int ldc){

    //Quick return if possible
    if( (m <= 0) || (n <= 0) || (k <= 0) ){
        return;
    }

    //Test the input parameters
    if( lda < MAX(1, m) || (ldb < MAX(1, k)) || (ldc < MAX(1, m)) ){
        return;
    }

    //Execution configuration for C = beta * C
    dim3 block1(blockDim1x, blockDim1y);
    dim3 grid1((m + block1.x - 1) / block1.x, (n + block1.y - 1) / block1.y);
    //  To compute the digits in RNS
    dim3 grid2(gridDim2x, gridDim2y);
    //  To rounding the result (we do not currently parameterize rounding)
    dim3 block3(16, 16);
    dim3 grid3((m + block3.x - 1) / block3.x, (n + block3.y - 1) / block3.y);

    //Multiplication C = beta * C - Computing the signs, exponents, and interval evaluations
    cuda::mp_mat2scal_mul_esi_kernel<<< grid1, block1 >>> (C, ldc, C, ldc, beta, m, n);

    //Multiplication C = beta * C - Multiplying the digits in the RNS
    cuda::mp_mat2scal_mul_digits_kernel<<< grid2, BLOCK_SIZE_FOR_RESIDUES >>> (C, ldc, C, ldc, beta, m, n);

    //Multiplication C = beta * C  - Rounding the result
    cuda::mp_matrix_round_kernel<<< grid3, block3 >>> (C, ldc, m, n);

    //Execution configuration for C = alpha * A * B + C
    dim3 block(blockDim3, blockDim3);

    //C = alpha * A * B + C
    dim3 grid((n + blockDim3 - 1) / blockDim3, (m + blockDim3 - 1) / blockDim3);
    mpgemm_straightforward_kernel<<< grid, block >>> (m, n, k, alpha, A, lda, B, ldb, C, ldc);
}

void mpres_test_straightforward(const int m, const int n, const int k, mpfr_t alpha, mpfr_t *A, const int lda, mpfr_t *B, const int ldb, mpfr_t beta, mpfr_t *C, const int ldc){
    InitCudaTimer();
    Logger::printDash();
    PrintTimerName("[GPU] MPRES-BLAS gemm straightforward");

    // Host data
    mp_float_ptr hA = new mp_float_t[lda * k];
    mp_float_ptr hB = new mp_float_t[ldb * n];
    mp_float_ptr hC = new mp_float_t[ldc * n];
    mp_float_t halpha;
    mp_float_t hbeta;

    //GPU data
    mp_array_t dA;
    mp_array_t dB;
    mp_array_t dC;
    mp_array_t dalpha;
    mp_array_t dbeta;

    //Init data
    cuda::mp_array_init(dA, lda * k);
    cuda::mp_array_init(dB, ldb * n);
    cuda::mp_array_init(dC, ldc * n);
    cuda::mp_array_init(dalpha, 1);
    cuda::mp_array_init(dbeta, 1);

    // Convert from MPFR
    convert_matrix(hA, A, lda, k);
    convert_matrix(hB, B, ldb, n);
    convert_matrix(hC, C, ldc, n);
    mp_set_mpfr(&halpha, alpha);
    mp_set_mpfr(&hbeta, beta);

    //Copying to the GPU
    cuda::mp_array_host2device(dA, hA, lda * k);
    cuda::mp_array_host2device(dB, hB, ldb * n);
    cuda::mp_array_host2device(dalpha, &halpha, 1);
    cuda::mp_array_host2device(dbeta, &hbeta, 1);

    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();
    //Launch
    for (int i = 0; i < REPEAT_TEST; i++) {
        cuda::mp_array_host2device(dC, hC, ldc * n);
        StartCudaTimer();
        mpgemm_straightforward<
                MPRES_BLOCK_SIZE_X_ESI,
                MPRES_BLOCK_SIZE_Y_ESI,
                MPRES_GRID_SIZE_X_DIGITS,
                MPRES_GRID_SIZE_Y_DIGITS,
                MPRES_BLOCK_SIZE_MATRIX_MULT>
                (m, n, k, dalpha, dA, lda, dB, ldb, dbeta, dC, ldc);
        EndCudaTimer();
    }
    PrintCudaTimer("took");
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    //Copying to the host
    cuda::mp_array_device2host(hC, dC, ldc * n);
    print_mp_sum(hC, ldc * n);

    //Cleanup
    delete [] hA;
    delete [] hB;
    delete [] hC;
    cuda::mp_array_clear(dA);
    cuda::mp_array_clear(dB);
    cuda::mp_array_clear(dC);
    cuda::mp_array_clear(dalpha);
    cuda::mp_array_clear(dbeta);
}



/********************* Main test *********************/

/*
 * Test for non-transposed matrices
 * A is of size lda * k, where the value of lda must be at least max(1, m).
 * B is of size ldb * n, where the value of ldb must be at least max(1, k).
 * C is of size ldc * n, where the value of lda must be at least max(1, m).
 */
void testNoTrans(){

    //Inputs
    mpfr_t *matrixA = create_random_array(LDA * K, INP_BITS);
    mpfr_t *matrixB = create_random_array(LDB * N, INP_BITS);
    mpfr_t *matrixC = create_random_array(LDC * N, INP_BITS);
    mpfr_t *alpha = create_random_array(1,  INP_BITS);
    mpfr_t *beta = create_random_array(1, INP_BITS);

    openblas_test(CblasNoTrans, CblasNoTrans, M, N, K, alpha[0], matrixA, LDA, matrixB, LDB, beta[0], matrixC, LDC);
    #ifndef EXCLUDE_MPACK
    mpack_gemm_test(TRANSA, TRANSB, M, N, K, alpha[0], matrixA, LDA, matrixB, LDB, beta[0], matrixC, LDC, MP_PRECISION, REPEAT_TEST);
    #endif
    mpres_test_notrans(M, N, K, alpha[0], matrixA, LDA, matrixB, LDB, beta[0], matrixC, LDC);
    mpres_test_straightforward(M, N, K, alpha[0], matrixA, LDA, matrixB, LDB, beta[0], matrixC, LDC);
    #ifndef EXCLUDE_CAMPARY
    campary_gemm_test<CAMPARY_PRECISION>(M, N, K, alpha[0], matrixA, LDA,  matrixB, LDB, beta[0], matrixC, LDC, INP_DIGITS, REPEAT_TEST);
    #endif
    #ifndef EXCLUDE_CUMP
    cump_gemm_test(M, N, K, alpha[0], matrixA, LDA,  matrixB, LDB, beta[0], matrixC, LDC, MP_PRECISION, INP_DIGITS, REPEAT_TEST);
    #endif

    checkDeviceHasErrors(cudaDeviceSynchronize());
    // cudaCheckErrors(); //CUMP gives failure

    //Cleanup
    for(int i = 0; i < LDA * K; i++){
        mpfr_clear(matrixA[i]);
    }
    for(int i = 0; i < LDB * N; i++){
        mpfr_clear(matrixB[i]);
    }
    for(int i = 0; i < LDC * N; i++){
        mpfr_clear(matrixC[i]);
    }
    mpfr_clear(alpha[0]);
    mpfr_clear(beta[0]);
    delete [] matrixA;
    delete [] matrixB;
    delete [] matrixC;
    delete [] alpha;
    delete [] beta;
    cudaDeviceReset();
}

int main(){

    initialize();

    //Start logging
    Logger::beginTestDescription(Logger::BLAS_GEMM_PERFORMANCE_TEST);
    Logger::printTestParameters(0, REPEAT_TEST, MP_PRECISION, MP_PRECISION_DEC);
    Logger::beginSection("Operation info:");
    Logger::printParam("M", M);
    Logger::printParam("N", N);
    Logger::printParam("K", K);
    Logger::printParam("LDA", LDA);
    Logger::printParam("LDB", LDB);
    Logger::printParam("LDC", LDC);
    Logger::printParam("TRANSA", TRANSA);
    Logger::printParam("TRANSB", TRANSB);
    Logger::printDash();
    Logger::beginSection("Additional info:");
    Logger::printParam("RNS_MODULI_SIZE", RNS_MODULI_SIZE);
    Logger::printParam("MPRES_BLOCK_SIZE_X_ESI", MPRES_BLOCK_SIZE_X_ESI);
    Logger::printParam("MPRES_BLOCK_SIZE_Y_ESI", MPRES_BLOCK_SIZE_Y_ESI);
    Logger::printParam("MPRES_GRID_SIZE_X_DIGITS", MPRES_GRID_SIZE_X_DIGITS);
    Logger::printParam("MPRES_GRID_SIZE_X_DIGITS", MPRES_GRID_SIZE_X_DIGITS);
    Logger::printParam("MPRES_GRID_SIZE_Y_DIGITS", MPRES_GRID_SIZE_Y_DIGITS);
    Logger::printParam("MPRES_BLOCK_SIZE_MATRIX_MULT", MPRES_BLOCK_SIZE_MATRIX_MULT);
    /* Logger::printParam("CAMPARY_PRECISION (n-double)", CAMPARY_PRECISION); */
    Logger::endSection(true);

    //Run the test
    if(TRANSA == "N" && TRANSB == "N") {
        testNoTrans();
    } else{
        //do something
    }

    //Finalize
    finalize();

    //End logging
    Logger::endTestDescription();

    return 0;
}