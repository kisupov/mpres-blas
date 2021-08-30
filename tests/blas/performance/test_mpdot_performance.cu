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
#include "../../../src/blas/dot.cuh"
#include "3rdparty.cuh"

#define N 1000000 //Operation size
#define REPEAT_TEST 10 //Number of repeats

//Execution configuration for mpdot
#define MPRES_CUDA_BLOCKS_FIELDS_ROUND   512
#define MPRES_CUDA_THREADS_FIELDS_ROUND  128
#define MPRES_CUDA_BLOCKS_RESIDUES       8192
#define MPRES_CUDA_BLOCKS_REDUCE         256
#define MPRES_CUDA_THREADS_REDUCE        64

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
    mp_real::mp_init(MP_PRECISION_DEC);
    mp_real *mp_real_x = new mp_real[n]; // Arprec vector
    mp_real *mp_real_y = new mp_real[n]; // Arprec vector
    mp_real mp_real_product = 0.0;
    for (int i = 0; i < n; i++) {
        mp_real_x[i].read(convert_to_string_sci(x[i], INP_DIGITS));
        mp_real_y[i].read(convert_to_string_sci(y[i], INP_DIGITS));
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
    mp_real::mp_finalize();
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
        mpd_set_string(mx[i], convert_to_string_fix(x[i], INP_DIGITS).c_str(), &ctx);
        mpd_set_string(my[i], convert_to_string_fix(y[i], INP_DIGITS).c_str(), &ctx);
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
    mp_array_t dresult;

    cuda::mp_array_init(dx, n);
    cuda::mp_array_init(dy, n);
    cuda::mp_array_init(dbuffer, n);
    cuda::mp_array_init(dresult, 1);

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
        cuda::mpdot<
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
    cuda::mp_array_device2host(hresult, dresult, 1);
    mp_get_mpfr(mpfr_result, hresult);
    mpfr_printf("result: %.70Rf \n", mpfr_result);

    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    //Cleanup
    delete [] hx;
    delete [] hy;
    delete [] hresult;
    mpfr_clear(mpfr_result);
    cuda::mp_array_clear(dx);
    cuda::mp_array_clear(dy);
    cuda::mp_array_clear(dbuffer);
    cuda::mp_array_clear(dresult);
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();
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
    #ifndef EXCLUDE_CAMPARY
    Logger::printParam("CAMPARY_PRECISION (n-double)", CAMPARY_PRECISION);
    #endif
    Logger::endSection(true);

    //Inputs
    mpfr_t * vectorX;
    mpfr_t * vectorY;
    vectorX = create_random_array(N, INP_BITS);
    vectorY = create_random_array(N, INP_BITS);

    //Double and double-double tests
    double  *dx;
    double  *dy;
    dx = new double[N];
    dy = new double[N];
    for(int i = 0; i < N; i ++){
        dx[i] = mpfr_get_d(vectorX[i], MPFR_RNDN);
        dy[i] = mpfr_get_d(vectorY[i], MPFR_RNDN);
    }
    //xblas_test(dx, dy, N);
    openblas_test(dx, dy, N);
    cublas_test(dx, dy, N);
    delete [] dx;
    delete [] dy;

    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    // Multiple precision tests
    mpfr_test(vectorX, vectorY, N);
    arprec_test(vectorX, vectorY, N);
    #ifndef EXCLUDE_MPACK
    mpack_dot_test(vectorX, vectorY, N, MP_PRECISION, REPEAT_TEST);
    #endif
    mpdecimal_test(vectorX, vectorY, N);
    mpres_test(vectorX, vectorY, N);
    #ifndef EXCLUDE_GARPREC
    garprec_dot_test(N, vectorX, vectorY, MP_PRECISION_DEC, INP_DIGITS, REPEAT_TEST);
    #endif
    #ifndef EXCLUDE_CAMPARY
    campary_dot_test<CAMPARY_PRECISION>(N, vectorX, vectorY, INP_DIGITS, REPEAT_TEST);
    #endif
    #ifndef EXCLUDE_CUMP
    cump_dot_test(N, vectorX, vectorY, MP_PRECISION, INP_DIGITS, REPEAT_TEST);
    #endif

    checkDeviceHasErrors(cudaDeviceSynchronize());
    // cudaCheckErrors(); //CUMP gives failure

    //Cleanup
    for(int i = 0; i < N; i ++){
        mpfr_clear(vectorX[i]);
        mpfr_clear(vectorY[i]);
    }
    delete [] vectorX;
    delete [] vectorY;

    //Finalize
    finalize();

    //End logging
    Logger::endTestDescription();

    return 0;
}
