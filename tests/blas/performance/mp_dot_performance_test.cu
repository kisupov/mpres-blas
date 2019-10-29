/*
 *  Performance test for BLAS DOT routines
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
#include "../../../src/blas/mpdot.cuh"
#include "3rdparty.cuh"


#define N 1000000 //Operation size
#define REPEAT_TEST 10 //Number of repeats

//Execution configuration for mp_array_dot
#define MPRES_CUDA_BLOCKS_FIELDS_ROUND   512
#define MPRES_CUDA_THREADS_FIELDS_ROUND  128
#define MPRES_CUDA_BLOCKS_RESIDUES       8192
#define MPRES_CUDA_BLOCKS_REDUCE         256
#define MPRES_CUDA_THREADS_REDUCE        128

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


/********************* Multiple-precision dot implementations and benchmarks *********************/


/////////
// cuBLAS
/////////
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

/////////
// OpenBLAS
/////////
void openblas_test(double *dbl_x, double *dbl_y, int n){
    InitCpuTimer();
    Logger::printDash();
    PrintTimerName("[CPU] OpenBLAS dot");

    double dbl_product = 0;
    StartCpuTimer();
    for(int i = 0; i < REPEAT_TEST; i ++) {
        dbl_product = cblas_ddot((const int) n, (const double *) dbl_x, (const int) 1, (const double *) dbl_y, (const int) 1);
    }
    EndCpuTimer();
    PrintCpuTimer("took");
    printf("result: %.70f \n", dbl_product);
}

/////////
// XBLAS
/////////
void xblas_test(double *dbl_x, double *dbl_y, int n){
    InitCpuTimer();
    Logger::printDash();
    PrintTimerName("[CPU] XBLAS dot")

    double dbl_product = 0;
    StartCpuTimer();
    for(int i = 0; i < REPEAT_TEST; i ++) {
        dbl_product = 0;
        BLAS_ddot_x(blas_conj, n, 1, dbl_x, 1, 1, dbl_y, 1, &dbl_product, blas_prec_extra);
    }
    EndCpuTimer();
    PrintCpuTimer("took");
    printf("result: %.70f \n", dbl_product);
}

/////////
// MPFR
/////////
void mpfr_dot(mpfr_t result, mpfr_t * x, mpfr_t * y, int n, int precision){
    #pragma omp parallel shared(result)
    {
        mpfr_t tmp_sum;
        mpfr_t tmp;
        mpfr_init2(tmp_sum, precision);
        mpfr_init2(tmp, precision);
        mpfr_set_d(tmp_sum, 0, MPFR_RNDN);
        mpfr_set_d(tmp, 0, MPFR_RNDN);

        #pragma omp for
        for(int i =0; i < n; i ++){
            mpfr_mul(tmp, x[i], y[i], MPFR_RNDN);
            mpfr_add(tmp_sum, tmp_sum, tmp, MPFR_RNDN);
         }
        #pragma omp critical
        {
            mpfr_add(result, result, tmp_sum, MPFR_RNDN);
        }
        mpfr_clear(tmp);
        mpfr_clear(tmp_sum);
    }
}

void mpfr_test(mpfr_t *x, mpfr_t *y, int n){
    InitCpuTimer();
    Logger::printDash();
    PrintTimerName("[CPU] MPFR dot");

    //Init
    mpfr_t mpfr_product;
    mpfr_init2(mpfr_product, MP_PRECISION);

    //Launch
    for(int i = 0; i < REPEAT_TEST; i ++) {
        mpfr_set_d(mpfr_product, 0, MPFR_RNDN);
        StartCpuTimer();
        mpfr_dot(mpfr_product, x, y, n, MP_PRECISION);
        EndCpuTimer();
    }
    PrintCpuTimer("took");
    mpfr_printf("result: %.70Rf \n", mpfr_product);

    //Clear
    mpfr_clear(mpfr_product);
}

/////////
// ARPREC
/////////
void arprec_dot(int n, mp_real *x, mp_real *y, mp_real *result){
    #pragma omp parallel
    {
        mp_real temp = 0.0;
        mp_real dot = 0.0;
        #pragma omp for
        for(int i = 0; i < n; i++){
            temp = x[i] * y[i];
            dot = dot + temp;
        }
        #pragma omp critical
        {
            result[0] = result[0] + dot;
        }
    }
}

void arprec_test(mpfr_t *x, mpfr_t *y, int n){
    InitCpuTimer();
    Logger::printDash();
    PrintTimerName("[CPU] ARPREC dot");

    //Init
    mp_real *mp_real_x = new mp_real[n]; // Arprec vector
    mp_real *mp_real_y = new mp_real[n]; // Arprec vector
    mp_real mp_real_product = 0.0;
    for (int i = 0; i < n; i++) {
        mp_real_x[i].read(convert_to_string_sci(x[i], INPUT_PRECISION_DEC));
        mp_real_y[i].read(convert_to_string_sci(y[i], INPUT_PRECISION_DEC));
    }

    //Launch
    for(int i = 0; i < REPEAT_TEST; i ++) {
        mp_real_product = 0.0;
        StartCpuTimer();
        arprec_dot(n, mp_real_x, mp_real_y, &mp_real_product);
        EndCpuTimer();
    }
    PrintCpuTimer("took");
    printf("result: %.83s \n", mp_real_product.to_string().c_str());

    //Clear
    delete [] mp_real_x;
    delete [] mp_real_y;
}

/////////
// MPACK
/////////
void mpack_test(mpfr_t *x, mpfr_t *y, int n) {
    InitCpuTimer();
    Logger::printDash();
    PrintTimerName("[CPU] MPACK dot");

    //Set precision
    mpfr::mpreal::set_default_prec ( MP_PRECISION );

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
    for(int i = 0; i < REPEAT_TEST; i ++){
        mpreal_result = Rdot(n, mpreal_x, 1, mpreal_y, 1);
    }
    EndCpuTimer();
    PrintCpuTimer("took");
    mpfr_printf("result: %.70Rf\n", &mpreal_result);

    //Cleanup
    delete [] mpreal_x;
    delete [] mpreal_y;
}

//////////
// MPDECIMAL
//////////
void mpdecimal_dot(int n, mpd_t **x, mpd_t **y, mpd_t *result, mpd_context_t *ctx){
     #pragma omp parallel
    {
        mpd_t *mul_result = mpd_new(ctx);
        mpd_set_string(mul_result, "0", ctx);
        mpd_t *partial_sum = mpd_new(ctx);
        mpd_set_string(partial_sum, "0", ctx);

        #pragma omp for
        for(int i =0; i < n; i ++){
            mpd_mul(mul_result, x[i], y[i], ctx);
            mpd_add(partial_sum, partial_sum, mul_result, ctx);
        }
        #pragma omp critical
        {
            mpd_add(result, result, partial_sum, ctx);
        }
        mpd_del(mul_result);
        mpd_del(partial_sum);
    }
}

void mpdecimal_test(mpfr_t *x, mpfr_t *y, int n){
    InitCpuTimer();
    Logger::printDash();
    PrintTimerName("[CPU] MPDECIMAL dot");

    // Loading the library context with needed precision
    mpd_context_t ctx;
    mpd_init(&ctx, MP_PRECISION_DEC);

    // Init
	mpd_t *product = mpd_new(&ctx);
    mpd_t **mx = new mpd_t*[n];
    mpd_t **my = new mpd_t*[n];
    #pragma omp parallel for
    for(int i = 0; i < n; i ++){
        mx[i] = mpd_new(&ctx);
        my[i] = mpd_new(&ctx);
        mpd_set_string(mx[i], convert_to_string_fix(x[i], INPUT_PRECISION_DEC).c_str(), &ctx);
        mpd_set_string(my[i], convert_to_string_fix(y[i], INPUT_PRECISION_DEC).c_str(), &ctx);
    }

    //Launch
    for(int j = 0; j < REPEAT_TEST; j ++){
        // reset result
        mpd_set_string(product, "0", &ctx);
        StartCpuTimer();
        mpdecimal_dot(n, mx, my, product, &ctx);
        EndCpuTimer();
    }
    PrintCpuTimer("took");
    printf("result: %.83s \n", mpd_to_sci(product, 1));

    //Cleanup
    mpd_del(product);
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
void mpres_test(mpfr_t *x, mpfr_t *y, int n){
    InitCudaTimer();
    Logger::printDash();
    PrintTimerName("[GPU] MPRES-BLAS dot");

    //Host data
    mp_float_ptr hx = new mp_float_t[n];
    mp_float_ptr hy = new mp_float_t[n];
    mp_float_ptr hresult = new mp_float_t[1];

    //GPU data
    mp_array_t dx;
    mp_array_t dy;
    mp_array_t dbuffer;
    mp_float_ptr dresult;

    cuda::mp_array_init(dx, n);
    cuda::mp_array_init(dy, n);
    cuda::mp_array_init(dbuffer, n);
    cudaMalloc((void **) &dresult, sizeof(mp_float_t));

    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    //Convert from MPFR
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        mp_set_mpfr(&hx[i], x[i]);
        mp_set_mpfr(&hy[i], y[i]);
    }
    *hresult = MP_ZERO;

    //Copying to the GPU
    cuda::mp_array_host2device(dx, hx, n);
    cuda::mp_array_host2device(dy, hy, n);

    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    //Launch
    StartCudaTimer();
    for (int i = 0; i < REPEAT_TEST; i++) {
        cuda::mp_array_dot<
                MPRES_CUDA_BLOCKS_FIELDS_ROUND,
                MPRES_CUDA_THREADS_FIELDS_ROUND,
                MPRES_CUDA_BLOCKS_RESIDUES,
                MPRES_CUDA_BLOCKS_REDUCE,
                MPRES_CUDA_THREADS_REDUCE>
                (n, dx, 1, dy, 1, dresult, dbuffer);
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
    delete [] hy;
    delete [] hresult;
    mpfr_clear(mpfr_result);
    cudaFree(dresult);
    cuda::mp_array_clear(dx);
    cuda::mp_array_clear(dy);
    cuda::mp_array_clear(dbuffer);
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();
}

//////
// CUMP
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

//Componentwise vector-vector multiplication
__global__ void cump_vector_mul(int n, mpf_array_t result, mpf_array_t x, mpf_array_t y) {
    using namespace cump;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    while (idx < n) {
        mpf_mul(result[idx], y[idx], x[idx]);
        idx += gridDim.x * blockDim.x;
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

void cump_test(mpfr_t *x, mpfr_t *y, int n){
    InitCudaTimer();
    Logger::printDash();
    PrintTimerName("[GPU] CUMP dot");

    //Set precision
    mpf_set_default_prec(MP_PRECISION);
    cumpf_set_default_prec(MP_PRECISION);

    //Execution configuration
    int threads = 64;
    int blocks_mul = n / threads + (n % threads ? 1 : 0);
    int blocks_red = 1024;

    //Host data
    mpf_t *hx = new mpf_t[n];
    mpf_t *hy = new mpf_t[n];
    mpf_t hresult;

    //GPU data
    cumpf_array_t dx;
    cumpf_array_t dy;
    cumpf_array_t dresult;
    cumpf_array_t dvecprod;
    cumpf_array_t dtemp;
    cumpf_array_t dblock_result;

    cumpf_array_init2(dx, n, MP_PRECISION);
    cumpf_array_init2(dy, n, MP_PRECISION);
    cumpf_array_init2(dresult, 1, MP_PRECISION);
    cumpf_array_init2(dvecprod, n, MP_PRECISION);
    cumpf_array_init2(dtemp, n, MP_PRECISION);
    cumpf_array_init2(dblock_result, blocks_red, MP_PRECISION);

    //Convert from MPFR
    for(int i = 0; i < n; i ++){
        mpf_init2(hx[i], MP_PRECISION);
        mpf_init2(hy[i], MP_PRECISION);
        mpf_set_str(hx[i], convert_to_string_sci(x[i], INPUT_PRECISION_DEC).c_str(), 10);
        mpf_set_str(hy[i], convert_to_string_sci(y[i], INPUT_PRECISION_DEC).c_str(), 10);
    }
    mpf_init2(hresult, MP_PRECISION);
    mpf_set_d(hresult, 0);

    //Copying to the GPU
    cumpf_array_set_mpf(dx, hx, n);
    cumpf_array_set_mpf(dy, hy, n);

    //Launch
    for(int i = 0; i < REPEAT_TEST; i ++){
        cump_reset<<<blocks_mul, threads>>>(n, dtemp);
        StartCudaTimer();
        cump_vector_mul<<<blocks_mul, threads>>>(n, dvecprod, dx, dy);
        cump_reduce_kernel1<<<blocks_red, threads>>>(n, dblock_result, dvecprod, dtemp);
        cump_reduce_kernel2<<<1, blocks_red>>>(dblock_result, dresult);
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
        mpf_clear(hy[i]);
    }
    delete [] hx;
    delete [] hy;
    cumpf_array_clear(dx);
    cumpf_array_clear(dy);
    cumpf_array_clear(dresult);
    cumpf_array_clear(dvecprod);
    cumpf_array_clear(dblock_result);
    cumpf_array_clear(dtemp);
}


/********************* Main test *********************/


int main() {

    initialize();

    //Start logging
    Logger::beginTestDescription(Logger::BLAS_DOT_PERFORMANCE_TEST);
    Logger::printTestParameters(N, REPEAT_TEST, MP_PRECISION, MP_PRECISION_DEC);
    Logger::beginSection("Additional info:");
    Logger::printParam("RNS_MODULI_SIZE", RNS_MODULI_SIZE);
    Logger::printParam("MPRES_CUDA_BLOCKS_FIELDS_ROUND", MPRES_CUDA_BLOCKS_FIELDS_ROUND);
    Logger::printParam("MPRES_CUDA_THREADS_FIELDS_ROUND", MPRES_CUDA_THREADS_FIELDS_ROUND);
    Logger::printParam("MPRES_CUDA_BLOCKS_RESIDUES", MPRES_CUDA_BLOCKS_RESIDUES);
    Logger::printParam("MPRES_CUDA_BLOCKS_REDUCE", MPRES_CUDA_BLOCKS_REDUCE);
    Logger::printParam("MPRES_CUDA_THREADS_REDUCE", MPRES_CUDA_THREADS_REDUCE);
    Logger::endSection(true);

    //Inputs
    mpfr_t * vectorX;
    mpfr_t * vectorY;
    vectorX = create_random_array(N, INPUT_PRECISION);
    vectorY = create_random_array(N, INPUT_PRECISION);

    //Double and double-double tests
    double  *dx;
    double  *dy;
    dx = new double[N];
    dy = new double[N];
    for(int i = 0; i < N; i ++){
        dx[i] = mpfr_get_d(vectorX[i], MPFR_RNDN);
        dy[i] = mpfr_get_d(vectorY[i], MPFR_RNDN);
    }
    xblas_test(dx, dy, N);
    openblas_test(dx, dy, N);
    cublas_test(dx, dy, N);
    delete [] dx;
    delete [] dy;

    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    // Multiple precision tests
    mpfr_test(vectorX, vectorY, N);
    arprec_test(vectorX, vectorY, N);
    mpack_test(vectorX, vectorY, N);
    mpdecimal_test(vectorX, vectorY, N);
    mpres_test(vectorX, vectorY, N);
    garprec_dot_test(N, vectorX, vectorY, MP_PRECISION_DEC, INPUT_PRECISION_DEC, REPEAT_TEST);
    //campary_dot_test<CAMPARY_PRECISION>(N, vectorX, vectorY, INPUT_PRECISION_DEC, REPEAT_TEST);
    cump_test(vectorX, vectorY, N);

    checkDeviceHasErrors(cudaDeviceSynchronize());
    // cudaCheckErrors(); //CUMP gives failure

    //Cleanup
    for(int i = 0; i < N; i ++){
        mpfr_clear(vectorX[i]);
        mpfr_clear(vectorY[i]);
    }
    delete [] vectorX;
    delete [] vectorY;
    cudaDeviceReset();

    //Finalize
    finalize();

    //End logging
    Logger::endTestDescription();

    return 0;
}
