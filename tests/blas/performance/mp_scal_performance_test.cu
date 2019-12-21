/*
 *  Performance test for BLAS SCAL routines
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
#include "../../../src/blas/mpscal.cuh"
#include "3rdparty.cuh"


#define N 1000000 //Operation size
#define REPEAT_TEST 10 //Number of repeats

//Execution configuration for mp_array_scal
#define MPRES_CUDA_BLOCKS_FIELDS_ROUND   512
#define MPRES_CUDA_THREADS_FIELDS_ROUND  128
#define MPRES_CUDA_BLOCKS_RESIDUES       8192

#define CAMPARY_PRECISION 8 //in n-double (e.g., 2-double, 3-double, 4-double, 8-double, etc.)

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
    mpfr_set_d(mpfr_result, 0, MPFR_RNDN);

    for (int i = 0; i < v_length; i++) {
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


/********************* Multiple-precision scal implementations and benchmarks *********************/


/////////
// cuBLAS
/////////
void cublas_test(double *x, double alpha, int n){
    InitCudaTimer();
    Logger::printDash();
    PrintTimerName("[GPU] cuBLAS dscal");

    cublasStatus_t stat;
    cublasHandle_t handle;

    stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("CUBLAS initialization failed\n");
        return;
    }

    double *dev_x;
    double *res_x = new double[n];

    cudaMalloc(&dev_x, sizeof(double) * n);
    cublasSetVector(n, sizeof(double), x, 1, dev_x, 1);

    for(int i = 0; i < REPEAT_TEST; i ++) {
        cublasSetVector(n, sizeof(double), x, 1, dev_x, 1);
        StartCudaTimer();
        cublasDscal(handle, n, &alpha, dev_x, 1);
        EndCudaTimer();
    }
    PrintCudaTimer("took");
    cublasGetVector(n, sizeof(double), dev_x, 1, res_x, 1);
    print_double_sum(res_x, n);

    cublasDestroy ( handle );
    cudaFree(dev_x);
    delete [] res_x;
}

/////////
// OpenBLAS
/////////
extern "C" void openblas_set_num_threads(int num_threads); //https://github.com/xianyi/OpenBLAS/issues/131

void openblas_test(double *x, double alpha, int n){
    InitCpuTimer();
    Logger::printDash();
    PrintTimerName("[CPU] OpenBLAS scal");

    openblas_set_num_threads(OPENBLAS_THREADS);
    double *result = new double[n];

    for(int i = 0; i < REPEAT_TEST; i ++){
        for(int j = 0; j < n; j ++){
            result[j] = x[j];
        }
        StartCpuTimer();
        cblas_dscal(n, alpha, result, 1); // Call OpenBLAS
        EndCpuTimer();
    }
    PrintCpuTimer("took");
    print_double_sum(result, n);
    delete [] result;
}

/////////
// MPFR
/////////
void mpfr_scal(int n, mpfr_t *x, mpfr_t alpha) {
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        mpfr_mul(x[i], x[i], alpha, MPFR_RNDN);
    }
}

void mpfr_test(mpfr_t *x, mpfr_t alpha, int n) {
    InitCpuTimer();
    Logger::printDash();
    PrintTimerName("[CPU] MPFR scal");

    // Init
    mpfr_t  lalpha;
    mpfr_t *lx = new mpfr_t[n];
    mpfr_init2(lalpha, MP_PRECISION);
    mpfr_set(lalpha, alpha, MPFR_RNDN);
    for (int i = 0; i < n; i++) {
        mpfr_init2(lx[i], MP_PRECISION);
        mpfr_set(lx[i], x[i], MPFR_RNDN);
    }

    // Launch
    for(int i = 0; i < REPEAT_TEST; i ++) {
        #pragma omp parallel for
        for (int j = 0; j < n; j++) {
            mpfr_set(lx[j], x[j], MPFR_RNDN);
        }
        StartCpuTimer();
        mpfr_scal(n, lx, lalpha);
        EndCpuTimer();
    }
    PrintCpuTimer("took");
    print_mpfr_sum(lx, n);

    //Cleanup
    mpfr_clear(lalpha);
    for(int i = 0; i < n; i++){
        mpfr_clear(lx[i]);
    }
    delete [] lx;
}

/////////
// ARPREC
/////////
void arprec_scal(int n, mp_real *x, mp_real alpha) {
    #pragma omp parallel for
    for(int i = 0; i < n; i++){
        x[i] = alpha * x[i];
    }
}

void arprec_test(mpfr_t *x, mpfr_t alpha, int n) {
    Logger::printDash();
    InitCpuTimer();
    PrintTimerName("[CPU] ARPREC scal");

    //Init
    mp_real *mp_real_x = new mp_real[n];
    mp_real mp_real_alpha;
    mp_real_alpha.read(convert_to_string_sci(alpha, INP_DIGITS));

    //Launch
    for(int i = 0; i < REPEAT_TEST; i ++){
        #pragma omp parallel for
        for (int j = 0; j < n; j++) {
            mp_real_x[j] = 0;
            mp_real_x[j].read(convert_to_string_sci(x[j], INP_DIGITS));
        }
        StartCpuTimer();
        arprec_scal(n, mp_real_x, mp_real_alpha);
        EndCpuTimer();
    }
    PrintCpuTimer("took");
    for (int i = 1; i < n; i++) {
        mp_real_x[0] += mp_real_x[i];
    }
    printf("result: %.83s \n", mp_real_x[0].to_string().c_str());

    //Cleanup
    delete [] mp_real_x;
}


/////////
// MPACK
/////////
void mpack_test(mpfr_t *x, mpfr_t alpha, int n) {
    Logger::printDash();
    InitCpuTimer();
    PrintTimerName("[CPU] MPACK scal");

    //Set precision
    mpfr::mpreal::set_default_prec ( MP_PRECISION );

    //Init
    mpreal *lx = new mpreal[n];
    mpreal lalpha = alpha;

    //Launch
    for(int i = 0; i < REPEAT_TEST; i ++) {
        for (int j = 0; j < n; j++) {
            lx[j] = x[j];
        }
        StartCpuTimer();
        Rscal(n, lalpha, lx, 1);
        EndCpuTimer();
    }
    PrintCpuTimer("took");

    //Print
    mpreal result = 0.0;
    for (int i = 0; i < n; i++) {
        result += lx[i];
    }
    mpfr_printf("result: %.70Rf\n", &result);

    //Cleanup
    delete [] lx;
}

/////////
// arbitraire, https://github.com/hlibc/arbitraire
/////////
/*
void arbitraire_scal(int n, fxdpnt **x, fxdpnt *alpha){
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        x[i] = arb_mul(x[i], alpha, x[i], 10, 1);
    }
}

void arbitraire_test(int n, mpfr_t *x, mpfr_t alpha, int precision, int repeat){
    fxdpnt *falpha, **fx;
    InitCpuTimer();

    fx = new fxdpnt*[n];
    Logger::printDash();
    falpha = arb_str2fxdpnt(convert_to_string_fix(alpha, precision).c_str());

    PrintTimerName("[CPU] arbitraire test");

    for (int j = 0; j < repeat; j += 1){
        for(int i = 0; i < n; i++){
            fx[i] = arb_str2fxdpnt(convert_to_string_fix(x[i], precision).c_str());
        }
        StartCpuTimer();
        arbitraire_scal(n, fx, falpha);
        EndCpuTimer();
    }
    PrintCpuTimer("took");

    for(int i = 1; i < n; i ++){
        fx[0] = arb_add(fx[0], fx[i], fx[0], 10);
    }
    printf("result: ");
    arb_print(fx[0]);

    for(int i = 0; i < n; i ++){
        arb_free(fx[i]);
    }
    arb_free(falpha);
}
*/

//////////
// MPDECIMAL
//////////
void mpdecimal_scal(int n, mpd_t **x, mpd_t *alpha, mpd_context_t *ctx){
    #pragma omp parallel for
    for(int i = 0; i < n; i ++){
        mpd_mul(x[i], alpha, x[i], ctx);
    }
}

void mpdecimal_test(mpfr_t *x, mpfr_t alpha, int n){
    InitCpuTimer();
    Logger::printDash();
    PrintTimerName("[CPU] MPDECIMAL scal");

    // Loading the library context with needed precision
    mpd_context_t ctx;
    mpd_init(&ctx, MP_PRECISION_DEC);

    //Init
    mpd_t *malpha  = mpd_new(&ctx);
    mpd_t **mx = new mpd_t*[n];
    mpd_set_string(malpha, convert_to_string_fix(alpha, INP_DIGITS).c_str(), &ctx);
    for(int i = 0; i < n; i ++){
        mx[i] = mpd_new(&ctx);
    }

    //Launch
    for(int j = 0; j < REPEAT_TEST; j ++){
        #pragma omp parallel for
        for(int i = 0; i < n; i ++){
            mpd_set_string(mx[i], convert_to_string_fix(x[i], INP_DIGITS).c_str(), &ctx);
        }
        StartCpuTimer();
        mpdecimal_scal(n, mx, malpha, &ctx);
        EndCpuTimer();
    }
    PrintCpuTimer("took");

    for(int i = 1; i < n; i ++){
        mpd_add(mx[0], mx[i], mx[0], &ctx);
    }
    printf("result: %.83s \n", mpd_to_sci(mx[0], 1));

    //Cleanup
    mpd_del(malpha);
    for(int i = 0; i < n; i ++){
        mpd_del(mx[i]);
    }
    delete [] mx;
}

/////////
// MPRES
/////////
void mpres_test(mpfr_t *x, mpfr_t alpha, int n) {
    Logger::printDash();
    InitCudaTimer();
    PrintTimerName("[GPU] MPRES-BLAS scal");

    //Host data
    mp_float_ptr hx = new mp_float_t[n];
    mp_float_t halpha;

    //GPU data
    mp_array_t dx;
    mp_array_t dalpha;

    cuda::mp_array_init(dx, n);
    cuda::mp_array_init(dalpha, 1);

    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    //Convert from MPFR
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        mp_set_mpfr(&hx[i], x[i]);
    }
    mp_set_mpfr(&halpha, alpha);

    //Copying to the GPU
    cuda::mp_array_host2device(dx, hx, n);
    cuda::mp_array_host2device(dalpha, &halpha, 1);

    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    //Launch
    for (int i = 0; i < REPEAT_TEST; i++) {
        cuda::mp_array_host2device(dx, hx, n);
        StartCudaTimer();
        cuda::mp_array_scal<
                MPRES_CUDA_BLOCKS_FIELDS_ROUND,
                MPRES_CUDA_THREADS_FIELDS_ROUND,
                MPRES_CUDA_BLOCKS_RESIDUES>
                (n, dalpha, dx, 1);
        EndCudaTimer();
    }
    PrintCudaTimer("took");

    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    //Copying to the host
    cuda::mp_array_device2host(hx, dx, n);
    print_mp_sum(hx, n);

    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    //Cleanup
    delete [] hx;
    cuda::mp_array_clear(dx);
    cuda::mp_array_clear(dalpha);
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();
}


/********************* Main test *********************/


int main() {

    initialize();

    //Start logging
    Logger::beginTestDescription(Logger::BLAS_SCAL_PERFORMANCE_TEST);
    Logger::printTestParameters(N, REPEAT_TEST, MP_PRECISION, MP_PRECISION_DEC);
    Logger::beginSection("Additional info:");
    Logger::printParam("RNS_MODULI_SIZE", RNS_MODULI_SIZE);
    Logger::printParam("MPRES_CUDA_BLOCKS_FIELDS_ROUND", MPRES_CUDA_BLOCKS_FIELDS_ROUND);
    Logger::printParam("MPRES_CUDA_THREADS_FIELDS_ROUND", MPRES_CUDA_THREADS_FIELDS_ROUND);
    Logger::printParam("MPRES_CUDA_BLOCKS_RESIDUES", MPRES_CUDA_BLOCKS_RESIDUES);
    Logger::printParam("CAMPARY_PRECISION (n-double)", CAMPARY_PRECISION);
    Logger::printParam("OPENBLAS_THREADS", OPENBLAS_THREADS);
    Logger::endSection(true);

    //Inputs
    mpfr_t * vectorX;
    mpfr_t * alpha;
    vectorX = create_random_array(N, INP_BITS);
    alpha = create_random_array(1, INP_BITS);

    //Double tests
    double dalpha;
    double *dx = new double[N];
    dalpha = mpfr_get_d(alpha[0], MPFR_RNDN);
    for(int i = 0; i < N; i ++){
        dx[i] = mpfr_get_d(vectorX[i], MPFR_RNDN);
    }
    openblas_test(dx, dalpha, N);
    cublas_test(dx, dalpha, N);
    delete [] dx;

    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    // Multiple-precision tests
    mpfr_test(vectorX, alpha[0], N);
    arprec_test(vectorX, alpha[0], N);
    mpack_test(vectorX, alpha[0], N);
    mpdecimal_test(vectorX, alpha[0], N);
    //arbitraire_test(N, vectorX, alpha[0], INP_BITS, REPEAT_TEST); // performs too long
    mpres_test(vectorX, alpha[0], N);
    garprec_scal_test(N, alpha[0], vectorX, MP_PRECISION_DEC, INP_DIGITS, REPEAT_TEST);
    //campary_scal_test<CAMPARY_PRECISION>(N, alpha[0], vectorX, INP_DIGITS, REPEAT_TEST);
    cump_scal_test(N, alpha[0], vectorX, MP_PRECISION, INP_DIGITS, REPEAT_TEST);
    
    checkDeviceHasErrors(cudaDeviceSynchronize());
    // cudaCheckErrors(); //CUMP gives failure

    // Cleanup
    for(int i = 0; i < N; i++){
        mpfr_clear(vectorX[i]);
    }
    mpfr_clear(alpha[0]);
    delete []alpha;
    delete []vectorX;
    cudaDeviceReset();

    //Finalize
    finalize();

    //End logging
    Logger::endTestDescription();

    return 0;
}
