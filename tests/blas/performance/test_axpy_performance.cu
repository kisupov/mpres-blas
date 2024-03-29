/*
 *  Performance test for BLAS AXPY routines
 *
 *  Copyright 2018, 2019 by Konstantin Isupov and Alexander Kuvaev.
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
#define EXCLUDE_GARPREC
#define EXCLUDE_CAMPARY
#define EXCLUDE_CUMP

#include "omp.h"
#include "../../logger.cuh"
#include "../../timers.cuh"
#include "../../tsthelper.cuh"
#include "../../../src/mparray.cuh"
#include "../../../src/blas/axpy.cuh"
#include "blas/external/3rdparty.cuh"

#define N 1000000 //Operation size
#define REPEAT_TEST 10 //Number of repeats

//Execution configuration for mp_axpy
#define MPRES_CUDA_BLOCKS_FIELDS_ROUND   512
#define MPRES_CUDA_THREADS_FIELDS_ROUND  128
#define MPRES_CUDA_BLOCKS_RESIDUES       8192

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
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();
}

void finalize(){
    cudaDeviceReset();
}

/********************* Multiple-precision axpy implementations and benchmarks *********************/


/////////
// cuBLAS
/////////
void cublas_test(double *x, double *y, double alpha, int n){
    InitCudaTimer();
    Logger::printDash();
    PrintTimerName("[GPU] cuBLAS daxpy");

    cublasStatus_t stat;
    cublasHandle_t handle;

    stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("CUBLAS initialization failed\n");
        return;
    }

    double *dev_x, *dev_y;
    double *res_y = new double[n];

    cudaMalloc(&dev_x, sizeof(double) * n);
    cudaMalloc(&dev_y, sizeof(double) * n);

    cublasSetVector(n, sizeof(double), x, 1, dev_x, 1);
    cublasSetVector(n, sizeof(double), y, 1, dev_y, 1);

    for(int i = 0; i < REPEAT_TEST; i ++) {
        cublasSetVector(n, sizeof(double), y, 1, dev_y, 1);
        StartCudaTimer();
        cublasDaxpy(handle, n, &alpha, dev_x, 1, dev_y, 1);
        EndCudaTimer();
    }
    PrintAndResetCudaTimer("took");
    cublasGetVector(n, sizeof(double), dev_y, 1, res_y, 1);
    print_double_sum(res_y, n);

    cublasDestroy ( handle );
    cudaFree(dev_x);
    cudaFree(dev_y);
    delete [] res_y;
}

/////////
// OpenBLAS
/////////
extern "C" void openblas_set_num_threads(int num_threads); //https://github.com/xianyi/OpenBLAS/issues/131

void openblas_test(double *x, double *y, double alpha, int n){
    InitCpuTimer();
    Logger::printDash();
    PrintTimerName("[CPU] OpenBLAS axpy");

    openblas_set_num_threads(OPENBLAS_THREADS);

    // y changed in axpy, result used instead
    double *result = new double[n];

    for(int i = 0; i < REPEAT_TEST; i ++){
        for(int j = 0; j < n; j ++){
            result[j] = y[j];
        }
        StartCpuTimer();
        cblas_daxpy(n, alpha, x, 1, result, 1);
        EndCpuTimer();
    }
    PrintAndResetCpuTimer("took");
    print_double_sum(result, n);
    delete [] result;
}

/////////
// XBLAS
/////////
void xblas_test(double *x, double *y, double alpha, int n) {
    Logger::printDash();
    InitCpuTimer();
    PrintTimerName("[CPU] XBLAS axpy");

    double *result = new double[n];

    for(int j = 0; j < REPEAT_TEST; j ++) {
        for(int i = 0; i < n; i ++){
            result[i] = y[i];
        }
        StartCpuTimer();
        //XBLAS have no axpy so axpby used instead
        BLAS_dwaxpby_x(n, alpha, x, 1, 1, result, 1, result, 1, blas_prec_extra);
        EndCpuTimer();
    }
    PrintAndResetCpuTimer("took");
    print_double_sum(result, n);
    delete [] result;
}

/////////
// MPFR
/////////
void mpfr_axpy(int n, mpfr_t *x, mpfr_t *y, mpfr_t alpha, int precision) {
    #pragma omp parallel 
    {
        mpfr_t temp;
        mpfr_init2(temp, precision);
        #pragma omp for
            for (int i = 0; i < n; i++) {     
                mpfr_mul(temp, x[i], alpha, MPFR_RNDN);
                mpfr_add(y[i], y[i], temp, MPFR_RNDN);
            }
        mpfr_clear(temp);           
    }
}

void mpfr_test(mpfr_t *x, mpfr_t *y, mpfr_t alpha, int n) {
    Logger::printDash();
    InitCpuTimer();
    PrintTimerName("[CPU] MPFR axpy");

    // Init
    mpfr_t  *mpfr_y = new mpfr_t[n];
    for (int i = 0; i < n; i++) {
        mpfr_init2(mpfr_y[i], MP_PRECISION);
        mpfr_set(mpfr_y[i], y[i], MPFR_RNDN);
    }

    // Launch
    for(int j = 0; j < REPEAT_TEST; j ++) {
        #pragma omp parallel for
        for (int i = 0; i < n; i++) {
            mpfr_set(mpfr_y[i], y[i], MPFR_RNDN);
        }
        StartCpuTimer();
        mpfr_axpy(n, x, mpfr_y, alpha, MP_PRECISION);
        EndCpuTimer();
    }
    PrintAndResetCpuTimer("took");
    print_mpfr_sum(mpfr_y, n);

    //Cleanup
    for(int i = 0; i < n; i++){
        mpfr_clear(mpfr_y[i]);
    }
    delete [] mpfr_y;
}

/////////
// ARPREC
/////////
void arprec_axpy(int n, mp_real *x, mp_real *y, mp_real alpha) {
    #pragma omp parallel
    {
        mp_real temp_x = 0.0;
        #pragma omp for
        for(int i = 0; i < n; i++){
            temp_x = alpha * x[i];
            y[i] = y[i] + temp_x;
        }
    }
}

void arprec_test(mpfr_t *x, mpfr_t *y, mpfr_t alpha, int n) {
    Logger::printDash();
    InitCpuTimer();
    PrintTimerName("[CPU] ARPREC axpy");

    //Init
    mp_real::mp_init(MP_PRECISION_DEC);
    mp_real *mp_real_x = new mp_real[n];
    mp_real *mp_real_y = new mp_real[n];
    mp_real mp_real_alpha;

    mp_real_alpha.read(convert_to_string_sci(alpha, INP_DIGITS));
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        mp_real_x[i].read(convert_to_string_sci(x[i], INP_DIGITS));
    }
    //Launch
    for(int i = 0; i < REPEAT_TEST; i ++) {
        #pragma omp parallel for
        for (int j = 0; j < n; j++) {
            mp_real_y[j] = 0;
            mp_real_y[j].read(convert_to_string_sci(y[j], INP_DIGITS));
        }
        StartCpuTimer();
        arprec_axpy(n, mp_real_x, mp_real_y, mp_real_alpha);
        EndCpuTimer();
    }
    PrintAndResetCpuTimer("took");
    for (int i = 1; i < n; i++) {
        mp_real::mpadd(mp_real_y[i], mp_real_y[0], mp_real_y[0], MP_PRECISION_DEC);
    }
    printf("result: %.83s \n", mp_real_y[0].to_string().c_str());

    //Cleanup
    delete [] mp_real_x;
    delete [] mp_real_y;
    mp_real::mp_finalize();
}

//////////
// MPDECIMAL
//////////
void mpdecimal_axpy(int n, mpd_t **x, mpd_t *alpha, mpd_t **y, mpd_context_t *ctx){
    #pragma omp parallel
    {
        mpd_t *tmp;
        tmp = mpd_new(ctx);

        #pragma omp for
        for(int i = 0; i < n; i ++){
            mpd_mul(tmp, alpha, x[i], ctx);
            mpd_add(y[i], tmp, y[i], ctx);
        }
        mpd_del(tmp);
    }
}

void mpdecimal_test(mpfr_t *x, mpfr_t *y, mpfr_t alpha, int n){
    InitCpuTimer();
    Logger::printDash();
    PrintTimerName("[CPU] MPDECIMAL axpy");

    // Loading the library context with needed precision
    mpd_context_t ctx;
    mpd_init(&ctx, MP_PRECISION_DEC);

    //Init
    mpd_t *malpha = mpd_new(&ctx);
    mpd_t **mx = new mpd_t*[n];
    mpd_t **my = new mpd_t*[n];

    mpd_set_string(malpha, convert_to_string_fix(alpha, INP_DIGITS).c_str(), &ctx);
    for(int i = 0; i < n; i ++){
        mx[i] = mpd_new(&ctx);
        my[i] = mpd_new(&ctx);
        mpd_set_string(mx[i], convert_to_string_fix(x[i], INP_DIGITS).c_str(), &ctx);
    }

    //Launch
    for(int j = 0; j < REPEAT_TEST; j ++){
        #pragma omp parallel for
        for(int i = 0; i < n; i ++){
            mpd_set_string(my[i], convert_to_string_fix(y[i], INP_DIGITS).c_str(), &ctx);
        }
        StartCpuTimer();
        mpdecimal_axpy(n, mx, malpha, my, &ctx);
        EndCpuTimer();
    }
    PrintAndResetCpuTimer("took");
    for(int i = 1; i < n; i ++){
        mpd_add(my[0], my[i], my[0], &ctx);
    }
    printf("result: %.83s \n", mpd_to_sci(my[0], 1));

    //Cleanup
    mpd_del(malpha);
    for(int i = 0; i < n; i ++){
        mpd_del(mx[i]);
        mpd_del(my[i]);
    }
    delete [] mx;
    delete [] my;
}

/////////
// MPRES-BLAS
/////////
void mpres_test(mpfr_t * x, mpfr_t * y, mpfr_t alpha, int n) {
    Logger::printDash();
    InitCudaTimer();
    PrintTimerName("[GPU] MPRES-BLAS axpy");

    //Host data
    mp_float_ptr hx = new mp_float_t[n];
    mp_float_ptr hy = new mp_float_t[n];
    mp_float_t halpha;

    //GPU data
    mp_array_t dx;
    mp_array_t dy;
    mp_array_t dalpha;
    mp_array_t dbuffer;

    cuda::mp_array_init(dx, n);
    cuda::mp_array_init(dy, n);
    cuda::mp_array_init(dalpha, 1);
    cuda::mp_array_init(dbuffer, n);

    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    //Convert from MPFR
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        mp_set_mpfr(&hx[i], x[i]);
        mp_set_mpfr(&hy[i], y[i]);
    }
    mp_set_mpfr(&halpha, alpha);

    //Copying to the GPU
    cuda::mp_array_host2device(dx, hx, n);
    cuda::mp_array_host2device(dy, hy, n);
    cuda::mp_array_host2device(dalpha, &halpha, 1);

    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    //Launch
    for (int i = 0; i < REPEAT_TEST; i++) {
        cuda::mp_array_host2device(dy, hy, n);
        StartCudaTimer();
        cuda::mp_axpy<
                MPRES_CUDA_BLOCKS_FIELDS_ROUND,
                MPRES_CUDA_THREADS_FIELDS_ROUND,
                MPRES_CUDA_BLOCKS_RESIDUES>
                (n, dalpha, dx, 1, dy, 1, dbuffer);
        EndCudaTimer();
    }
    PrintAndResetCudaTimer("took");

    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    //Copying to the host
    cuda::mp_array_device2host(hy, dy, n);
    print_mp_sum(hy, N);

    //Cleanup
    delete [] hx;
    delete [] hy;
    cuda::mp_array_clear(dx);
    cuda::mp_array_clear(dy);
    cuda::mp_array_clear(dalpha);
    cuda::mp_array_clear(dbuffer);
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();
}


/********************* Main test *********************/


int main() {

    initialize();

    //Start logging
    Logger::beginTestDescription(Logger::BLAS_AXPY_PERFORMANCE_TEST);
    Logger::printTestParameters(N, REPEAT_TEST, MP_PRECISION, MP_PRECISION_DEC);
    Logger::beginSection("Additional info:");
    Logger::printParam("RNS_MODULI_SIZE", RNS_MODULI_SIZE);
    Logger::printParam("MPRES_CUDA_BLOCKS_FIELDS_ROUND", MPRES_CUDA_BLOCKS_FIELDS_ROUND);
    Logger::printParam("MPRES_CUDA_THREADS_FIELDS_ROUND", MPRES_CUDA_THREADS_FIELDS_ROUND);
    Logger::printParam("MPRES_CUDA_BLOCKS_RESIDUES", MPRES_CUDA_BLOCKS_RESIDUES);
    #ifndef EXCLUDE_CAMPARY
    Logger::printParam("CAMPARY_PRECISION (n-double)", CAMPARY_PRECISION);
    #endif
    Logger::printParam("OPENBLAS_THREADS", OPENBLAS_THREADS);
    Logger::endSection(true);

    //Inputs
    mpfr_t * vectorX;
    mpfr_t * vectorY;
    mpfr_t * alpha;
    vectorX = create_random_array(N, INP_BITS);
    vectorY = create_random_array(N, INP_BITS);
    alpha = create_random_array(1, INP_BITS);

    //Double and double-double tests
    double  *dx = new double[N];
    double  *dy = new double[N];
    double dalpha;
    dalpha = mpfr_get_d(alpha[0], MPFR_RNDN);
    for(int i = 0; i < N; i ++){
        dx[i] = mpfr_get_d(vectorX[i], MPFR_RNDN);
        dy[i] = mpfr_get_d(vectorY[i], MPFR_RNDN);
    }
    //xblas_test(dx, dy, dalpha, N);
    openblas_test(dx, dy, dalpha, N);
    cublas_test(dx, dy, dalpha, N);
    delete [] dx;
    delete [] dy;

    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    // Multiple-precision tests
    mpfr_test(vectorX, vectorY, alpha[0], N);
    arprec_test(vectorX, vectorY, alpha[0], N);
    #ifndef EXCLUDE_MPACK
    mpack_axpy_test(vectorX, vectorY, alpha[0], N, MP_PRECISION, REPEAT_TEST);
    #endif
    mpdecimal_test(vectorX, vectorY, alpha[0], N);
    mpres_test(vectorX, vectorY, alpha[0], N);
    #ifndef EXCLUDE_GARPREC
    garprec_axpy_test(N, alpha[0], vectorX, vectorY, MP_PRECISION_DEC, INP_DIGITS, REPEAT_TEST);
    #endif
    #ifndef EXCLUDE_CAMPARY
    campary_axpy_test<CAMPARY_PRECISION>(N, alpha[0], vectorX, vectorY, INP_DIGITS, REPEAT_TEST);
    #endif
    #ifndef EXCLUDE_CUMP
    cump_axpy_test(N, alpha[0], vectorX, vectorY, MP_PRECISION, INP_DIGITS, REPEAT_TEST);
    #endif
    checkDeviceHasErrors(cudaDeviceSynchronize());
    // cudaCheckErrors(); //CUMP gives failure

    //Cleanup
    mpfr_clear(alpha[0]);
    for(int i = 0; i < N; i++){
        mpfr_clear(vectorX[i]);
        mpfr_clear(vectorY[i]);
    }
    delete [] vectorX;
    delete [] vectorY;
    delete [] alpha;

    //Finalize
    finalize();

    //End logging
    Logger::endTestDescription();

    return 0;
}