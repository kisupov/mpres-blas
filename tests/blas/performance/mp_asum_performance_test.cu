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

#include "omp.h"
#include "../../logger.cuh"
#include "../../timers.cuh"
#include "../../tsthelper.cuh"
#include "../../../src/mparray.cuh"
#include "../../../src/blas/mpasum.cuh"
#include "3rdparty.cuh"


#define N 1000000 //Operation size
#define REPEAT_TEST 10 //Number of repeats

//Execution configuration for mp_array_asum
#define MPRES_CUDA_BLOCKS_REDUCE   256
#define MPRES_CUDA_THREADS_REDUCE  128

#define CAMPARY_PRECISION 8 //in n-double (e.g., 2-double, 3-double, 4-double, 8-double, etc.)

int INPUT_PRECISION; //in bits
int INPUT_PRECISION_DEC; //in decimal digits
int MP_PRECISION_DEC; //in decimal digits

void setPrecisions(){
    INPUT_PRECISION = (int)(MP_PRECISION / 4);
    INPUT_PRECISION_DEC = (int)(INPUT_PRECISION / 3.32 + 1);
    MP_PRECISION_DEC = (int)(MP_PRECISION / 3.32 + 1);
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
    PrintCudaTimer("took");
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
    PrintCpuTimer("took");
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
    PrintCpuTimer("took");
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
    PrintCpuTimer("took");
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
    mp_real mp_real_result;
    mp_real *mp_real_x = new mp_real[n];
    mp_real_result = 0.0;
    for (int i = 0; i < n; i++) {
        mp_real_x[i].read(convert_to_string_sci(x[i], INPUT_PRECISION_DEC));
    }

    //Launch
    for (int i = 0; i < REPEAT_TEST; i++) {
        mp_real_result = 0;
        StartCpuTimer();
        arprec_asum(n, mp_real_x, &mp_real_result);
        EndCpuTimer();
    }
    PrintCpuTimer("took");
    printf("result: %.83s \n", mp_real_result.to_string().c_str());

    //Clear
    delete [] mp_real_x;
}

/////////
// MPACK
/////////
void mpack_test(mpfr_t *x, int n) {
    InitCpuTimer();
    Logger::printDash();
    PrintTimerName("[CPU] MPACK asum");

    //Set precision
    mpfr::mpreal::set_default_prec(MP_PRECISION);

    //Init
    mpreal *mpreal_x = new mpreal[n];
    mpreal mpreal_result;
    mpreal_result = 0;
    for (int i = 0; i < n; i++) {
        mpreal_x[i] = x[i];
    }

    //Launch
    StartCpuTimer();
    for (int i = 0; i < REPEAT_TEST; i++)
        mpreal_result = Rasum(n, mpreal_x, 1); // Call MPACK
    EndCpuTimer();
    PrintCpuTimer("took");
    mpfr_printf("result: %.70Rf\n", &mpreal_result);

    //Clear
    delete[] mpreal_x;
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
        mpd_set_string(mx[i], convert_to_string_fix(x[i], INPUT_PRECISION_DEC).c_str(), &ctx);
    }

    //Lunch
    for(int j = 0; j < REPEAT_TEST; j ++){
        mpd_set_string(sum, "0", &ctx);
        StartCpuTimer();
        mpdecimal_asum(n, mx, sum, &ctx);
        EndCpuTimer();
    }
    PrintCpuTimer("took");
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
    mp_float_ptr hresult = new mp_float_t[1];

    //GPU data
    mp_array_t dx;
    mp_float_ptr dresult;

    cuda::mp_array_init(dx, n);
    cudaMalloc((void **) &dresult, sizeof(mp_float_t));

    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    //Convert from MPFR
    #pragma omp parallel for
    for(int i =0; i < n; i ++){
        mp_set_mpfr(&hx[i], x[i]);
    }
    *hresult = MP_ZERO;

    //Copying to the GPU
    cuda::mp_array_host2device(dx, hx, n);

    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    //Launch
    StartCudaTimer();
    for (int i = 0; i < REPEAT_TEST; i++) {
        cuda::mp_array_asum<MPRES_CUDA_BLOCKS_REDUCE, MPRES_CUDA_THREADS_REDUCE>(n, dx, 1, dresult);
    }
    EndCudaTimer();
    PrintCudaTimer("took");

    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    mpfr_t mpfr_result;
    mpfr_init2(mpfr_result, MP_PRECISION);
    mpfr_set_d(mpfr_result, 0, MPFR_RNDN);

    //Copying to the host
    cudaMemcpy(hresult, dresult, sizeof(mp_float_t), cudaMemcpyDeviceToHost);
    mp_get_mpfr(mpfr_result, hresult);
    mpfr_printf("result: %.70Rf \n", mpfr_result);

    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    //Cleanup
    delete [] hx;
    delete [] hresult;
    mpfr_clear(mpfr_result);
    cudaFree(dresult);
    cuda::mp_array_clear(dx);
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();
}


//////
// CUMP (Note that the sum is calculated instead of the sum of absolute values)
//////
using cump::mpf_array_t;

//Reset array
__global__ void cump_reset(int n, mpf_array_t temp) {
    using namespace cump;
    int numberIdx =  blockDim.x * blockIdx.x + threadIdx.x;
    while (numberIdx < n) {
        mpf_sub(temp[numberIdx], temp[numberIdx], temp[numberIdx]); // set to zero
        numberIdx +=  gridDim.x * blockDim.x;
    }
}

//First summation kernel
__global__ void cump_reduce_kernel1(int n, mpf_array_t result, mpf_array_t x, mpf_array_t temp){
    using namespace cump;
    // parameters
    unsigned int tid = threadIdx.x;
    unsigned int bid = blockIdx.x;
    unsigned int bsize = blockDim.x;
    unsigned int globalIdx = bid * bsize + tid;
    unsigned int i = bid * bsize * 2 + tid;
    unsigned int k = 2 * gridDim.x * bsize;

    while (i < n) {
        mpf_add(temp[globalIdx], temp[globalIdx], x[i]);
        if (i + bsize < n){
            mpf_add(temp[globalIdx], temp[globalIdx], x[i + bsize]);
        }
        i += k;
    }
    __syncthreads();
    i = bsize;
    while(i >= 2){
        unsigned int half = i >> 1;
        if ((bsize >= i) && (tid < half) && (globalIdx + half < n)) {
            mpf_add(temp[globalIdx], temp[globalIdx], temp[globalIdx + half]);
        }
        i = i >> 1;
        __syncthreads();
    }
    // write result for this block to global mem
    if (tid == 0) {
        mpf_set(result[bid], temp[globalIdx]);
    };
    __syncthreads();
}

//Second summation kernel (optimized)
__global__ void cump_reduce_kernel2(mpf_array_t x, mpf_array_t result){
    using namespace cump;
    unsigned int tid = threadIdx.x;
    for(unsigned int s = blockDim.x / 2; s > 0; s >>= 1){
        if(tid < s){
            mpf_add(x[tid], x[tid], x[tid + s]);
        }
        __syncthreads();
    }
    // write result for this block to global mem
    if (tid == 0) {
        mpf_set(result[0], x[tid]);
    }
}

void cump_test(mpfr_t *x, int n){
    InitCudaTimer();
    Logger::printDash();
    PrintTimerName("[GPU] CUMP sum");

    //Set precision
    mpf_set_default_prec(MP_PRECISION);
    cumpf_set_default_prec(MP_PRECISION);

    //Execution configuration
    int threads = 64;
    int blocks = 1024;

    //Host data
    mpf_t *hx = new mpf_t[n];
    mpf_t hresult;

    //GPU data
    cumpf_array_t dx;
    cumpf_array_t dresult;
    cumpf_array_t dtemp;
    cumpf_array_t dblock_result;

    cumpf_array_init2(dx, n, MP_PRECISION);
    cumpf_array_init2(dresult, 1, MP_PRECISION);
    cumpf_array_init2(dtemp, n, MP_PRECISION);
    cumpf_array_init2(dblock_result, blocks, MP_PRECISION);

    //Convert from MPFR
    for(int i = 0; i < n; i ++){
        mpf_init2(hx[i], MP_PRECISION);
        mpf_set_str(hx[i], convert_to_string_sci(x[i], INPUT_PRECISION_DEC).c_str(), 10);
    }
    mpf_init2(hresult, MP_PRECISION);
    mpf_set_d(hresult, 0);

    //Copying to the GPU
    cumpf_array_set_mpf(dx, hx, n);

    //Launch
    for(int i = 0; i < REPEAT_TEST; i ++){
        cump_reset<<<blocks, threads>>>(n, dtemp);
        StartCudaTimer();
        cump_reduce_kernel1<<<blocks, threads>>>(n, dblock_result, dx, dtemp);
        cump_reduce_kernel2<<<1, blocks>>>(dblock_result, dresult);
        EndCudaTimer();
    }
    PrintCudaTimer("took");

    //Copying to the host
    mpf_array_set_cumpf(&hresult, dresult, 1);
    gmp_printf ("result: %.70Ff \n", hresult);

    //Cleanup
    mpf_clear(hresult);
    for(int i = 0; i < n; i ++){
        mpf_clear(hx[i]);
    }
    delete[] hx;
    cumpf_array_clear(dx);
    cumpf_array_clear(dresult);
    cumpf_array_clear(dblock_result);
    cumpf_array_clear(dtemp);
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
    Logger::endSection(true);

    //Inputs
    mpfr_t * vectorX;
    vectorX = create_random_array(N, INPUT_PRECISION);

   //Double and double-double tests
    double *dx = new double[N];
    for(int i = 0; i < N; i ++){
        dx[i] = mpfr_get_d(vectorX[i], MPFR_RNDN);
    }
    xblas_test(dx, N);
    openblas_test(dx, N);
    cublas_test(dx, N);
    delete [] dx;

    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    //Multiple-precision tests
    mpfr_test(vectorX, N);
    arprec_test(vectorX, N);
    mpack_test(vectorX, N);
    mpdecimal_test(vectorX, N);
    mpres_test(vectorX, N);
    garprec_sum_test(N, vectorX, MP_PRECISION_DEC, INPUT_PRECISION_DEC, REPEAT_TEST);
    //campary_asum_test<CAMPARY_PRECISION>(N, vectorX, INPUT_PRECISION_DEC, REPEAT_TEST);
    cump_test(vectorX, N);

    checkDeviceHasErrors(cudaDeviceSynchronize());
    //cudaCheckErrors(); //CUMP gives failure

    // Cleanup
    for(int i = 0; i < N; i++){
        mpfr_clear(vectorX[i]);
    }
    delete[] vectorX;
    cudaDeviceReset();

    //Finalize
    finalize();

    //End logging
    Logger::endTestDescription();

    return 0;
}