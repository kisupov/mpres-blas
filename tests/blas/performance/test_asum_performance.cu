/*
 *  Performance test for BLAS ASUM routines
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

#include "../../logger.cuh"
#include "../../timers.cuh"
#include "../../tsthelper.cuh"
#include "../../../src/mparray.cuh"
#include "../../../src/blas/asum.cuh"
#include "blas/external/3rdparty.cuh"


#define N 1000000 //Operation size
#define REPEAT_TEST 10 //Number of repeats

//Execution configuration for mp_asum
#define MPRES_CUDA_BLOCKS_REDUCE   256
#define MPRES_CUDA_THREADS_REDUCE  64

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


/********************* Multiple-precision asum implementations and benchmarks *********************/


/////////
// cuBLAS
/////////
void cublas_test(double *x, int n){
    InitCudaTimer();
    Logger::printDash();
    PrintTimerName("[GPU] cuBLAS asum");

    cublasStatus_t stat;
    cublasHandle_t handle;

    stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("CUBLAS initialization failed\n");
        return;
    }
    double *dev_x;
    double *res = new double[1];
    cudaMalloc(&dev_x, sizeof(double) * n);
    cublasSetVector(n, sizeof(double), x, 1, dev_x, 1);

    StartCudaTimer();
    for(int i = 0; i < REPEAT_TEST; i ++) {
        cublasDasum(handle, n, dev_x, 1, res); // Call cuBLAS
    }
    EndCudaTimer();
    PrintAndResetCudaTimer("took");
    printf("result: %.70f\n", *res);

    cublasDestroy ( handle );
    cudaFree(dev_x);
    delete[] res;
}

/////////
// OpenBLAS
/////////
void openblas_test(double *x, int n){
    InitCpuTimer();
    Logger::printDash();
    PrintTimerName("[CPU] OpenBLAS asum");

    double result;
    StartCpuTimer();
    for (int i = 0; i < REPEAT_TEST; i++) {
        result = 0;
        result = cblas_dasum(n, x, 1); // Call OpenBLAS
    }
    EndCpuTimer();
    PrintAndResetCpuTimer("took");
    printf("result: %.70f \n", result);
}

/////////
// XBLAS
/////////
void xblas_test(double *x, int n){
    InitCpuTimer();
    Logger::printDash();
    PrintTimerName("[CPU] XBLAS asum");

    double result;
    StartCpuTimer();
    for (int i = 0; i < REPEAT_TEST; i++) {
        result = 0;
        BLAS_dsum_x(N, (double const *) x, 1, (double *) &result, blas_prec_extra); // Call XBLAS
    }
    EndCpuTimer();
    PrintAndResetCpuTimer("took");
    printf("result: %.70f \n", result);
}

/////////
// MPFR
/////////
void mpfr_asum(mpfr_t result, mpfr_t *x, int n, int precision) {
    #pragma omp parallel shared(result)
    {
        mpfr_t sum;
        mpfr_t absval;
        mpfr_init2(sum, precision);
        mpfr_init2(absval, precision);
        mpfr_set_d(sum, 0, MPFR_RNDN);
        mpfr_set_d(absval, 0, MPFR_RNDN);

        #pragma omp for
        for (int i = 0; i < n; ++i) {
            mpfr_abs(absval, x[i], MPFR_RNDN);
            mpfr_add(sum, sum, absval, MPFR_RNDN);
        }
        #pragma omp critical
        {
            mpfr_add(result, result, sum, MPFR_RNDN);
        }
        mpfr_clear(absval);
        mpfr_clear(sum);
    }
}

void mpfr_test(mpfr_t *x, int n){
    InitCpuTimer();
    Logger::printDash();
    PrintTimerName("[CPU] MPFR asum");

    //Init
    mpfr_t result;
    mpfr_init2(result, MP_PRECISION);

    //Launch
    for (int i = 0; i < REPEAT_TEST; i++) {
        mpfr_set_d(result, 0, MPFR_RNDN);
        StartCpuTimer();
        mpfr_asum(result, x, n, MP_PRECISION);
        EndCpuTimer();
    }
    PrintAndResetCpuTimer("took");
    mpfr_printf("result: %.70Rf \n", result);

    //Clear
    mpfr_clear(result);
}

/////////
// ARPREC
/////////
void arprec_asum(int n, mp_real *x, mp_real *result) {
    #pragma omp parallel
    {
        mp_real sum = 0.0;
        mp_real temp = 0.0;
        #pragma omp for
        for (int i = 0; i < n; ++i) {
            temp = abs(x[i]);
            sum = sum + temp;
        }
        #pragma omp critical
        {
            result[0] = result[0] + sum;
        }
    }
}

void arprec_test(mpfr_t *x, int n){
    InitCpuTimer();
    Logger::printDash();
    PrintTimerName("[CPU] ARPREC asum");

    //Init
    mp_real::mp_init(MP_PRECISION_DEC);
    mp_real mp_real_result;
    mp_real *mp_real_x = new mp_real[n];
    mp_real_result = 0.0;
    for (int i = 0; i < n; i++) {
        mp_real_x[i].read(convert_to_string_sci(x[i], INP_DIGITS));
    }

    //Launch
    for (int i = 0; i < REPEAT_TEST; i++) {
        mp_real_result = 0;
        StartCpuTimer();
        arprec_asum(n, mp_real_x, &mp_real_result);
        EndCpuTimer();
    }
    PrintAndResetCpuTimer("took");
    printf("result: %.83s \n", mp_real_result.to_string().c_str());

    //Clear
    delete [] mp_real_x;
    mp_real::mp_finalize();
}

//////////
// MPDECIMAL
//////////
void mpdecimal_asum(int n, mpd_t **x, mpd_t *result, mpd_context_t *ctx){
     #pragma omp parallel
    {
        mpd_t *tmp_sum;
        mpd_t *tmp;
        tmp = mpd_new(ctx);
        tmp_sum = mpd_new(ctx);
        mpd_set_string(tmp, "0", ctx);
        mpd_set_string(tmp_sum, "0", ctx);
        #pragma omp for
        for(int i = 0; i < n; i ++){
            mpd_abs(tmp, x[i], ctx);
            mpd_add(tmp_sum, tmp_sum, tmp, ctx);
        }
        #pragma omp critical
        {
            mpd_add(result, result, tmp_sum, ctx);
        }
        mpd_del(tmp_sum);
        mpd_del(tmp);
    }
}

void mpdecimal_test(mpfr_t *x, int n){
    InitCpuTimer();
    Logger::printDash();
    PrintTimerName("[CPU] MPDECIMAL asum");

    //Set precision
    mpd_context_t ctx;
    mpd_init(&ctx, MP_PRECISION_DEC);

    //Init
	mpd_t *sum = mpd_new(&ctx);
    mpd_t **mx = new mpd_t*[n];
    #pragma omp parallel for
    for(int i = 0; i < n; i ++){
        mx[i] = mpd_new(&ctx);
        mpd_set_string(mx[i], convert_to_string_fix(x[i], INP_DIGITS).c_str(), &ctx);
    }

    //Lunch
    for(int j = 0; j < REPEAT_TEST; j ++){
        mpd_set_string(sum, "0", &ctx);
        StartCpuTimer();
        mpdecimal_asum(n, mx, sum, &ctx);
        EndCpuTimer();
    }
    PrintAndResetCpuTimer("took");
    printf("result: %.83s \n", mpd_to_sci(sum, 1));

    //Cleanup
    mpd_del(sum);
    for(int i = 0; i < n; i ++){
        mpd_del(mx[i]);
    }
    delete [] mx;
}

/////////
// MPRES-BLAS
/////////
void mpres_test(mpfr_t *x, int n){
    InitCudaTimer();
    Logger::printDash();
    PrintTimerName("[GPU] MPRES-BLAS asum");

    //Host data
    mp_float_ptr hx = new mp_float_t[n];
    mp_float_t hresult = MP_ZERO;

    //GPU data
    mp_array_t dx;
    mp_array_t dresult;

    cuda::mp_array_init(dx, n);
    cuda::mp_array_init(dresult, 1);

    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    //Convert from MPFR
    #pragma omp parallel for
    for(int i =0; i < n; i ++){
        mp_set_mpfr(&hx[i], x[i]);
    }

    //Copying to the GPU
    cuda::mp_array_host2device(dx, hx, n);

    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    //Launch
    StartCudaTimer();
    for (int i = 0; i < REPEAT_TEST; i++) {
        cuda::mp_asum<MPRES_CUDA_BLOCKS_REDUCE, MPRES_CUDA_THREADS_REDUCE>(n, dx, 1, dresult);
    }
    EndCudaTimer();
    PrintAndResetCudaTimer("took");

    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    mpfr_t mpfr_result;
    mpfr_init2(mpfr_result, MP_PRECISION);
    mpfr_set_d(mpfr_result, 0, MPFR_RNDN);

    //Copying to the host
    cuda::mp_array_device2host(&hresult, dresult, 1);
    mp_get_mpfr(mpfr_result, hresult);
    mpfr_printf("result: %.70Rf \n", mpfr_result);

    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    //Cleanup
    delete [] hx;
    mpfr_clear(mpfr_result);
    cuda::mp_array_clear(dx);
    cuda::mp_array_clear(dresult);
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();
}


/********************* Main test *********************/


int main() {

    initialize();

    //Start logging
    Logger::beginTestDescription(Logger::BLAS_ASUM_PERFORMANCE_TEST);
    Logger::printTestParameters(N, REPEAT_TEST, MP_PRECISION, MP_PRECISION_DEC);
    Logger::beginSection("Additional info:");
    Logger::printParam("RNS_MODULI_SIZE", RNS_MODULI_SIZE);
    Logger::printParam("MPRES_CUDA_BLOCKS_REDUCE", MPRES_CUDA_BLOCKS_REDUCE);
    Logger::printParam("MPRES_CUDA_THREADS_REDUCE", MPRES_CUDA_THREADS_REDUCE);
    #ifndef EXCLUDE_CAMPARY
    Logger::printParam("CAMPARY_PRECISION (n-double)", CAMPARY_PRECISION);
    #endif
    Logger::endSection(true);

    //Inputs
    mpfr_t * vectorX;
    vectorX = create_random_array(N, INP_BITS);

   //Double and double-double tests
    double *dx = new double[N];
    for(int i = 0; i < N; i ++){
        dx[i] = mpfr_get_d(vectorX[i], MPFR_RNDN);
    }
    //xblas_test(dx, N);
    openblas_test(dx, N);
    cublas_test(dx, N);
    delete [] dx;

    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    //Multiple-precision tests
    mpfr_test(vectorX, N);
    arprec_test(vectorX, N);
    #ifndef EXCLUDE_MPACK
    mpack_asum_test(vectorX, N, MP_PRECISION, REPEAT_TEST);
    #endif
    mpdecimal_test(vectorX, N);
    mpres_test(vectorX, N);
    #ifndef EXCLUDE_GARPREC
    garprec_sum_test(N, vectorX, MP_PRECISION_DEC, INP_DIGITS, REPEAT_TEST);
    #endif
    #ifndef EXCLUDE_CAMPARY
    campary_asum_test<CAMPARY_PRECISION>(N, vectorX, INP_DIGITS, REPEAT_TEST);
    #endif
    #ifndef EXCLUDE_CUMP
    cump_sum_test(N, vectorX, MP_PRECISION, INP_DIGITS, REPEAT_TEST);
    #endif
    checkDeviceHasErrors(cudaDeviceSynchronize());
    //cudaCheckErrors(); //CUMP gives failure

    // Cleanup
    for(int i = 0; i < N; i++){
        mpfr_clear(vectorX[i]);
    }
    delete[] vectorX;

    //Finalize
    finalize();

    //End logging
    Logger::endTestDescription();

    return 0;
}