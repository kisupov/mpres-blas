#  Copyright 2018, 2019 by Konstantin Isupov and Alexander Kuvaev.
#
#  This file is part of the MPRES-BLAS library.
#
#  MPRES-BLAS is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  MPRES-BLAS is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with MPRES-BLAS.  If not, see <https://www.gnu.org/licenses/>.

project(MPRES_BLAS_TEST LANGUAGES CUDA CXX)

if (NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif ()

#####################################
# OpenMP
#####################################
find_package(OpenMP REQUIRED)

#####################################
# Path to the installed MPACK (https://github.com/nakatamaho/mplapack) (for performance tests)
#####################################
set(MPACK_INC_PATH /usr/local/include/mpack)
include_directories(${MPACK_INC_PATH})

#########################
## Basic arith tests
#########################

# Test for checking the algorithms that calculate the interval evaluation of an RNS number
add_executable(test_rns_eval_verify arith/test_rns_eval_verify.cu)
target_compile_options(test_rns_eval_verify PRIVATE  $<${cuda_lang}:$<BUILD_INTERFACE:${GLOBAL_COMPILE_OPTIONS}>>)
target_link_libraries(test_rns_eval_verify gmp mpfr)

# Test for validating the mp_add routines
add_executable(test_mp_add_verify arith/test_mp_add_verify.cu)
target_compile_options(test_mp_add_verify PRIVATE $<${cuda_lang}:$<BUILD_INTERFACE:${GLOBAL_COMPILE_OPTIONS}>>)
target_link_libraries(test_mp_add_verify gmp mpfr OpenMP::OpenMP_CXX)

# Test for validating the mp_mul routines
add_executable(test_mp_mul_verify arith/test_mp_mul_verify.cu)
target_compile_options(test_mp_mul_verify PRIVATE $<${cuda_lang}:$<BUILD_INTERFACE:${GLOBAL_COMPILE_OPTIONS}>>)
target_link_libraries(test_mp_mul_verify gmp mpfr OpenMP::OpenMP_CXX)

# Test for checking the correctness of the extended-range floating-point routines
add_executable(test_extrange_verify arith/test_extrange_verify.cu)
target_compile_options(test_extrange_verify PRIVATE $<${cuda_lang}:$<BUILD_INTERFACE:${GLOBAL_COMPILE_OPTIONS}>>)

# Test for measure the performance of the RNS magnitude comparison algorithms
add_executable(test_rns_cmp_performance arith/test_rns_cmp_performance.cu)
target_compile_options(test_rns_cmp_performance PRIVATE  $<${cuda_lang}:$<BUILD_INTERFACE:${GLOBAL_COMPILE_OPTIONS}>>)
target_link_libraries(test_rns_cmp_performance gmp mpfr)

# Test for validating the mp_cmp routines
add_executable(test_mp_cmp_verify arith/test_mp_cmp_verify.cu)
target_compile_options(test_mp_cmp_verify PRIVATE  $<${cuda_lang}:$<BUILD_INTERFACE:${GLOBAL_COMPILE_OPTIONS}>>)
target_link_libraries(test_mp_cmp_verify gmp mpfr)

#########################
## Performance tests for BLAS routines
#########################

# Performance test for ASUM
add_executable(test_mpasum_performance blas/performance/test_mpasum_performance.cu)
target_compile_options(test_mpasum_performance PRIVATE  $<${cuda_lang}:$<BUILD_INTERFACE:${GLOBAL_COMPILE_OPTIONS}>>)
target_link_libraries(test_mpasum_performance gmp mpfr arprec openblas xblas gmpxx mpfrcxx mblas_mpfr mpc OpenMP::OpenMP_CXX cublas mpdec cump)

# Performance test for DOT
add_executable(test_mpdot_performance blas/performance/test_mpdot_performance.cu)
target_compile_options(test_mpdot_performance PRIVATE $<${cuda_lang}:$<BUILD_INTERFACE:${GLOBAL_COMPILE_OPTIONS}>>)
target_link_libraries(test_mpdot_performance gmp mpfr arprec openblas xblas gmpxx mpfrcxx mblas_mpfr mpc OpenMP::OpenMP_CXX cublas mpdec cump)

# Performance test for SCAL
add_executable(test_mpscal_performance blas/performance/test_mpscal_performance.cu)
target_compile_options(test_mpscal_performance PRIVATE $<${cuda_lang}:$<BUILD_INTERFACE:${GLOBAL_COMPILE_OPTIONS}>>)
target_link_libraries(test_mpscal_performance gmp mpfr arprec openblas gmpxx mpfrcxx mblas_mpfr mpc OpenMP::OpenMP_CXX cublas mpdec cump)

# Performance test for AXPY
add_executable(test_mpaxpy_performance blas/performance/test_mpaxpy_performance.cu)
target_compile_options(test_mpaxpy_performance PRIVATE $<${cuda_lang}:$<BUILD_INTERFACE:${GLOBAL_COMPILE_OPTIONS}>>)
target_link_libraries(test_mpaxpy_performance gmp mpfr arprec openblas xblas gmpxx mpfrcxx mblas_mpfr mpc OpenMP::OpenMP_CXX cublas mpdec cump)

# Performance test for NORM
add_executable(test_mpnorm_performance blas/performance/test_mpnorm_performance.cu)
target_compile_options(test_mpnorm_performance PRIVATE $<${cuda_lang}:$<BUILD_INTERFACE:${GLOBAL_COMPILE_OPTIONS}>>)
target_link_libraries(test_mpnorm_performance gmp mpfr OpenMP::OpenMP_CXX)

# Performance test for ROT
add_executable(test_mprot_performance blas/performance/test_mprot_performance.cu)
target_compile_options(test_mprot_performance PRIVATE  $<${cuda_lang}:$<BUILD_INTERFACE:${GLOBAL_COMPILE_OPTIONS}>>)
target_link_libraries(test_mprot_performance gmp mpfr arprec gmpxx mpfrcxx mblas_mpfr mpc OpenMP::OpenMP_CXX cump)

# Performance test for AXPY_DOT
add_executable(test_mpaxpydot_performance blas/performance/test_mpaxpydot_performance.cu)
target_compile_options(test_mpaxpydot_performance PRIVATE $<${cuda_lang}:$<BUILD_INTERFACE:${GLOBAL_COMPILE_OPTIONS}>>)
target_link_libraries(test_mpaxpydot_performance gmp mpfr arprec OpenMP::OpenMP_CXX cump)

# Performance test for GEMV
add_executable(test_mpgemv_performance blas/performance/test_mpgemv_performance.cu)
target_compile_options(test_mpgemv_performance PRIVATE  $<${cuda_lang}:$<BUILD_INTERFACE:${GLOBAL_COMPILE_OPTIONS}>>)
target_link_libraries(test_mpgemv_performance gmp mpfr arprec openblas gmpxx mpfrcxx mblas_mpfr mpc OpenMP::OpenMP_CXX cublas cump)

# Performance test for GER
add_executable(test_mpger_performance blas/performance/test_mpger_performance.cu)
target_compile_options(test_mpger_performance PRIVATE  $<${cuda_lang}:$<BUILD_INTERFACE:${GLOBAL_COMPILE_OPTIONS}>>)
target_link_libraries(test_mpger_performance gmp mpfr openblas gmpxx mpfrcxx mblas_mpfr mpc OpenMP::OpenMP_CXX cublas cump)

# Performance test for GEMM
add_executable(test_mpgemm_performance blas/performance/test_mpgemm_performance.cu)
target_compile_options(test_mpgemm_performance PRIVATE  $<${cuda_lang}:$<BUILD_INTERFACE:${GLOBAL_COMPILE_OPTIONS}>>)
target_link_libraries(test_mpgemm_performance gmp mpfr openblas gmpxx mpfrcxx mblas_mpfr mpc OpenMP::OpenMP_CXX cump)

# Performance test for GE_ADD
add_executable(test_mpgeadd_performance blas/performance/test_mpgeadd_performance.cu)
target_compile_options(test_mpgeadd_performance PRIVATE  $<${cuda_lang}:$<BUILD_INTERFACE:${GLOBAL_COMPILE_OPTIONS}>>)
target_link_libraries(test_mpgeadd_performance gmp mpfr OpenMP::OpenMP_CXX cump)

# Performance test for GE_ACC
add_executable(test_mpgeacc_performance blas/performance/test_mpgeacc_performance.cu)
target_compile_options(test_mpgeacc_performance PRIVATE  $<${cuda_lang}:$<BUILD_INTERFACE:${GLOBAL_COMPILE_OPTIONS}>>)
target_link_libraries(test_mpgeacc_performance gmp mpfr OpenMP::OpenMP_CXX cump)

# Performance test for GE_DIAG_SCALE
add_executable(test_mpgediagscale_performance blas/performance/test_mpgediagscale_performance.cu)
target_compile_options(test_mpgediagscale_performance PRIVATE  $<${cuda_lang}:$<BUILD_INTERFACE:${GLOBAL_COMPILE_OPTIONS}>>)
target_link_libraries(test_mpgediagscale_performance gmp mpfr OpenMP::OpenMP_CXX cump)

# Performance test for GE_LRSCALE
add_executable(test_mpgelrscale_performance blas/performance/test_mpgelrscale_performance.cu)
target_compile_options(test_mpgelrscale_performance PRIVATE  $<${cuda_lang}:$<BUILD_INTERFACE:${GLOBAL_COMPILE_OPTIONS}>>)
target_link_libraries(test_mpgelrscale_performance gmp mpfr OpenMP::OpenMP_CXX cump)

# Performance test for GE_NORM
add_executable(test_mpgenorm_performance blas/performance/test_mpgenorm_performance.cu)
target_compile_options(test_mpgenorm_performance PRIVATE  $<${cuda_lang}:$<BUILD_INTERFACE:${GLOBAL_COMPILE_OPTIONS}>>)
target_link_libraries(test_mpgenorm_performance gmp mpfr OpenMP::OpenMP_CXX)

#########################
## Accuracy tests for BLAS routines
#########################

# Accuracy test for ASUM
add_executable(test_mpasum_accuracy blas/accuracy/test_mpasum_accuracy.cu)
target_compile_options(test_mpasum_accuracy PRIVATE $<${cuda_lang}:$<BUILD_INTERFACE:${GLOBAL_COMPILE_OPTIONS}>>)
target_link_libraries(test_mpasum_accuracy gmp mpfr OpenMP::OpenMP_CXX)

# Accuracy test for DOT
add_executable(test_mpdot_accuracy blas/accuracy/test_mpdot_accuracy.cu)
target_compile_options(test_mpdot_accuracy PRIVATE $<${cuda_lang}:$<BUILD_INTERFACE:${GLOBAL_COMPILE_OPTIONS}>>)
target_link_libraries(test_mpdot_accuracy gmp mpfr OpenMP::OpenMP_CXX)

# Accuracy test for SCAL
add_executable(test_mpscal_accuracy blas/accuracy/test_mpscal_accuracy.cu)
target_compile_options(test_mpscal_accuracy PRIVATE  $<${cuda_lang}:$<BUILD_INTERFACE:${GLOBAL_COMPILE_OPTIONS}>>)
target_link_libraries(test_mpscal_accuracy gmp mpfr OpenMP::OpenMP_CXX)

# Accuracy test for AXPY
add_executable(test_mpaxpy_accuracy blas/accuracy/test_mpaxpy_accuracy.cu)
target_compile_options(test_mpaxpy_accuracy PRIVATE $<${cuda_lang}:$<BUILD_INTERFACE:${GLOBAL_COMPILE_OPTIONS}>>)
target_link_libraries(test_mpaxpy_accuracy gmp mpfr OpenMP::OpenMP_CXX)

#########################
## Performance tests for SpMV routines (double precision matrix)
#########################

# Performance test for SpMV using various sparse matrix storage formats
add_executable(test_mpdspmv_performance.cu sparse/performance/test_mpdspmv_performance.cu)
target_compile_options(test_mpdspmv_performance.cu PRIVATE  $<${cuda_lang}:$<BUILD_INTERFACE:${GLOBAL_COMPILE_OPTIONS}>>)
target_link_libraries(test_mpdspmv_performance.cu gmp mpfr OpenMP::OpenMP_CXX cump)

# Performance test for SpMV in CSR format
add_executable(test_mpdspmv_csr_performance sparse/performance/test_mpdspmv_csr_performance.cu)
target_compile_options(test_mpdspmv_csr_performance PRIVATE  $<${cuda_lang}:$<BUILD_INTERFACE:${GLOBAL_COMPILE_OPTIONS}>>)
target_link_libraries(test_mpdspmv_csr_performance gmp mpfr OpenMP::OpenMP_CXX cump taco)

# Performance test for SpMV in JAD (JDS) format
add_executable(test_mpdspmv_jad_performance sparse/performance/test_mpdspmv_jad_performance.cu)
target_compile_options(test_mpdspmv_jad_performance PRIVATE  $<${cuda_lang}:$<BUILD_INTERFACE:${GLOBAL_COMPILE_OPTIONS}>>)
target_link_libraries(test_mpdspmv_jad_performance gmp mpfr OpenMP::OpenMP_CXX cump taco)

# Performance test for SpMV in ELLPACK format
add_executable(test_mpdspmv_ell_performance sparse/performance/test_mpdspmv_ell_performance.cu)
target_compile_options(test_mpdspmv_ell_performance PRIVATE  $<${cuda_lang}:$<BUILD_INTERFACE:${GLOBAL_COMPILE_OPTIONS}>>)
target_link_libraries(test_mpdspmv_ell_performance gmp mpfr OpenMP::OpenMP_CXX cump taco)

# Performance test for SpMV in DIA format
add_executable(test_mpdspmv_dia_performance sparse/performance/test_mpdspmv_dia_performance.cu)
target_compile_options(test_mpdspmv_dia_performance PRIVATE  $<${cuda_lang}:$<BUILD_INTERFACE:${GLOBAL_COMPILE_OPTIONS}>>)
target_link_libraries(test_mpdspmv_dia_performance gmp mpfr OpenMP::OpenMP_CXX cump taco)

#########################
## Performance tests for SpMV routines with multiple-precision matrix
#########################

# Performance test for SpMV in CSR format
add_executable(test_mpspmv_mpmtx_csr_performance sparse/mpmtx/test_mpspmv_mpmtx_csr_performance.cu)
target_compile_options(test_mpspmv_mpmtx_csr_performance PRIVATE  $<${cuda_lang}:$<BUILD_INTERFACE:${GLOBAL_COMPILE_OPTIONS}>>)
target_link_libraries(test_mpspmv_mpmtx_csr_performance gmp mpfr OpenMP::OpenMP_CXX cump taco)

# Performance test for SpMV in ELLPACK format
add_executable(test_mpspmv_mpmtx_ell_performance sparse/mpmtx/test_mpspmv_mpmtx_ell_performance.cu)
target_compile_options(test_mpspmv_mpmtx_ell_performance PRIVATE  $<${cuda_lang}:$<BUILD_INTERFACE:${GLOBAL_COMPILE_OPTIONS}>>)
target_link_libraries(test_mpspmv_mpmtx_ell_performance gmp mpfr OpenMP::OpenMP_CXX cump taco)

# Performance test for SpMV in DIA format
add_executable(test_mpspmv_mpmtx_dia_performance sparse/mpmtx/test_mpspmv_mpmtx_dia_performance.cu)
target_compile_options(test_mpspmv_mpmtx_dia_performance PRIVATE  $<${cuda_lang}:$<BUILD_INTERFACE:${GLOBAL_COMPILE_OPTIONS}>>)
target_link_libraries(test_mpspmv_mpmtx_dia_performance gmp mpfr OpenMP::OpenMP_CXX cump taco)

# Performance test for SpMV in JAD (JDS) format
add_executable(test_mpspmv_mpmtx_jad_performance sparse/mpmtx/test_mpspmv_mpmtx_jad_performance.cu)
target_compile_options(test_mpspmv_mpmtx_jad_performance PRIVATE  $<${cuda_lang}:$<BUILD_INTERFACE:${GLOBAL_COMPILE_OPTIONS}>>)
target_link_libraries(test_mpspmv_mpmtx_jad_performance gmp mpfr OpenMP::OpenMP_CXX cump taco)