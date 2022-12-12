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

# Test for validating the mp_sub routines
add_executable(test_mp_sub_verify arith/test_mp_sub_verify.cu)
target_compile_options(test_mp_sub_verify PRIVATE $<${cuda_lang}:$<BUILD_INTERFACE:${GLOBAL_COMPILE_OPTIONS}>>)
target_link_libraries(test_mp_sub_verify gmp mpfr OpenMP::OpenMP_CXX)

# Test for validating the mp_mul routines
add_executable(test_mp_mul_verify arith/test_mp_mul_verify.cu)
target_compile_options(test_mp_mul_verify PRIVATE $<${cuda_lang}:$<BUILD_INTERFACE:${GLOBAL_COMPILE_OPTIONS}>>)
target_link_libraries(test_mp_mul_verify gmp mpfr OpenMP::OpenMP_CXX)

# Test for validating the mp_div routines
add_executable(test_mp_div_verify arith/test_mp_div_verify.cu)
target_compile_options(test_mp_div_verify PRIVATE $<${cuda_lang}:$<BUILD_INTERFACE:${GLOBAL_COMPILE_OPTIONS}>>)
target_link_libraries(test_mp_div_verify gmp mpfr OpenMP::OpenMP_CXX)

# Test for validating the mp_sqrt routines
add_executable(test_mp_sqrt_verify arith/test_mp_sqrt_verify.cu)
target_compile_options(test_mp_sqrt_verify PRIVATE $<${cuda_lang}:$<BUILD_INTERFACE:${GLOBAL_COMPILE_OPTIONS}>>)
target_link_libraries(test_mp_sqrt_verify gmp mpfr OpenMP::OpenMP_CXX)

# Test for validating the mp_inv_sqrt routines
add_executable(test_mp_inv_sqrt_verify arith/test_mp_inv_sqrt_verify.cu)
target_compile_options(test_mp_inv_sqrt_verify PRIVATE $<${cuda_lang}:$<BUILD_INTERFACE:${GLOBAL_COMPILE_OPTIONS}>>)
target_link_libraries(test_mp_inv_sqrt_verify gmp mpfr OpenMP::OpenMP_CXX)

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

# Test for validating the mp_cmp routines
add_executable(test_mp_arith_peak_performance arith/peak/test_mp_arith_peak_performance.cu)
target_compile_options(test_mp_arith_peak_performance PRIVATE  $<${cuda_lang}:$<BUILD_INTERFACE:${GLOBAL_COMPILE_OPTIONS}>>)
target_link_libraries(test_mp_arith_peak_performance gmp mpfr OpenMP::OpenMP_CXX)

#########################
## Performance tests for BLAS routines
#########################

# Performance test for ASUM
add_executable(test_asum_performance blas/performance/test_asum_performance.cu)
target_compile_options(test_asum_performance PRIVATE  $<${cuda_lang}:$<BUILD_INTERFACE:${GLOBAL_COMPILE_OPTIONS}>>)
target_link_libraries(test_asum_performance gmp mpfr arprec openblas xblas gmpxx mpfrcxx mblas_mpfr mpc OpenMP::OpenMP_CXX cublas mpdec cump)

# Performance test for DOT
add_executable(test_dot_performance blas/performance/test_dot_performance.cu)
target_compile_options(test_dot_performance PRIVATE $<${cuda_lang}:$<BUILD_INTERFACE:${GLOBAL_COMPILE_OPTIONS}>>)
target_link_libraries(test_dot_performance gmp mpfr arprec openblas xblas gmpxx mpfrcxx mblas_mpfr mpc OpenMP::OpenMP_CXX cublas mpdec cump)

# Performance test for SCAL
add_executable(test_scal_performance blas/performance/test_scal_performance.cu)
target_compile_options(test_scal_performance PRIVATE $<${cuda_lang}:$<BUILD_INTERFACE:${GLOBAL_COMPILE_OPTIONS}>>)
target_link_libraries(test_scal_performance gmp mpfr arprec openblas gmpxx mpfrcxx mblas_mpfr mpc OpenMP::OpenMP_CXX cublas mpdec cump)

# Performance test for AXPY
add_executable(test_axpy_performance blas/performance/test_axpy_performance.cu)
target_compile_options(test_axpy_performance PRIVATE $<${cuda_lang}:$<BUILD_INTERFACE:${GLOBAL_COMPILE_OPTIONS}>>)
target_link_libraries(test_axpy_performance gmp mpfr arprec openblas xblas gmpxx mpfrcxx mblas_mpfr mpc OpenMP::OpenMP_CXX cublas mpdec cump)

# Performance test for NORM
add_executable(test_norm_performance blas/performance/test_norm_performance.cu)
target_compile_options(test_norm_performance PRIVATE $<${cuda_lang}:$<BUILD_INTERFACE:${GLOBAL_COMPILE_OPTIONS}>>)
target_link_libraries(test_norm_performance gmp mpfr OpenMP::OpenMP_CXX)

# Performance test for ROT
add_executable(test_rot_performance blas/performance/test_rot_performance.cu)
target_compile_options(test_rot_performance PRIVATE  $<${cuda_lang}:$<BUILD_INTERFACE:${GLOBAL_COMPILE_OPTIONS}>>)
target_link_libraries(test_rot_performance gmp mpfr arprec gmpxx mpfrcxx mblas_mpfr mpc OpenMP::OpenMP_CXX cump)

# Performance test for AXPY_DOT
add_executable(test_axpydot_performance blas/performance/test_axpydot_performance.cu)
target_compile_options(test_axpydot_performance PRIVATE $<${cuda_lang}:$<BUILD_INTERFACE:${GLOBAL_COMPILE_OPTIONS}>>)
target_link_libraries(test_axpydot_performance gmp mpfr arprec OpenMP::OpenMP_CXX cump)

# Performance test for GEMV
add_executable(test_gemv_performance blas/performance/test_gemv_performance.cu)
target_compile_options(test_gemv_performance PRIVATE  $<${cuda_lang}:$<BUILD_INTERFACE:${GLOBAL_COMPILE_OPTIONS}>>)
target_link_libraries(test_gemv_performance gmp mpfr arprec openblas gmpxx mpfrcxx mblas_mpfr mpc OpenMP::OpenMP_CXX cublas cump)

# Performance test for GER
add_executable(test_ger_performance blas/performance/test_ger_performance.cu)
target_compile_options(test_ger_performance PRIVATE  $<${cuda_lang}:$<BUILD_INTERFACE:${GLOBAL_COMPILE_OPTIONS}>>)
target_link_libraries(test_ger_performance gmp mpfr openblas gmpxx mpfrcxx mblas_mpfr mpc OpenMP::OpenMP_CXX cublas cump)

# Performance test for GEMM
add_executable(test_gemm_performance blas/performance/test_gemm_performance.cu)
target_compile_options(test_gemm_performance PRIVATE  $<${cuda_lang}:$<BUILD_INTERFACE:${GLOBAL_COMPILE_OPTIONS}>>)
target_link_libraries(test_gemm_performance gmp mpfr openblas gmpxx mpfrcxx mblas_mpfr mpc OpenMP::OpenMP_CXX cump)

# Performance test for GE_ADD
add_executable(test_geadd_performance blas/performance/test_geadd_performance.cu)
target_compile_options(test_geadd_performance PRIVATE  $<${cuda_lang}:$<BUILD_INTERFACE:${GLOBAL_COMPILE_OPTIONS}>>)
target_link_libraries(test_geadd_performance gmp mpfr OpenMP::OpenMP_CXX cump)

# Performance test for GE_ACC
add_executable(test_geacc_performance blas/performance/test_geacc_performance.cu)
target_compile_options(test_geacc_performance PRIVATE  $<${cuda_lang}:$<BUILD_INTERFACE:${GLOBAL_COMPILE_OPTIONS}>>)
target_link_libraries(test_geacc_performance gmp mpfr OpenMP::OpenMP_CXX cump)

# Performance test for GE_DIAG_SCALE
add_executable(test_gediagscale_performance blas/performance/test_gediagscale_performance.cu)
target_compile_options(test_gediagscale_performance PRIVATE  $<${cuda_lang}:$<BUILD_INTERFACE:${GLOBAL_COMPILE_OPTIONS}>>)
target_link_libraries(test_gediagscale_performance gmp mpfr OpenMP::OpenMP_CXX cump)

# Performance test for GE_LRSCALE
add_executable(test_gelrscale_performance blas/performance/test_gelrscale_performance.cu)
target_compile_options(test_gelrscale_performance PRIVATE  $<${cuda_lang}:$<BUILD_INTERFACE:${GLOBAL_COMPILE_OPTIONS}>>)
target_link_libraries(test_gelrscale_performance gmp mpfr OpenMP::OpenMP_CXX cump)

# Performance test for GE_NORM
add_executable(test_genorm_performance blas/performance/test_genorm_performance.cu)
target_compile_options(test_genorm_performance PRIVATE  $<${cuda_lang}:$<BUILD_INTERFACE:${GLOBAL_COMPILE_OPTIONS}>>)
target_link_libraries(test_genorm_performance gmp mpfr OpenMP::OpenMP_CXX)

#########################
## Performance tests for BLAS v2 routines
#########################

# Performance test for SYMV
add_executable(test_symv_v2_performance blas/v2/symv/test_symv_v2_performance.cu)
target_compile_options(test_symv_v2_performance PRIVATE  $<${cuda_lang}:$<BUILD_INTERFACE:${GLOBAL_COMPILE_OPTIONS}>>)
target_link_libraries(test_symv_v2_performance gmp mpfr openblas OpenMP::OpenMP_CXX cublas)

# Performance test for DOT
add_executable(test_dot_v2_performance blas/v2/dot/test_dot_v2_performance.cu)
target_compile_options(test_dot_v2_performance PRIVATE  $<${cuda_lang}:$<BUILD_INTERFACE:${GLOBAL_COMPILE_OPTIONS}>>)
target_link_libraries(test_dot_v2_performance gmp mpfr openblas OpenMP::OpenMP_CXX)

# Performance test for SCAL
add_executable(test_scal_v2_performance blas/v2/scal/test_scal_v2_performance.cu)
target_compile_options(test_scal_v2_performance PRIVATE  $<${cuda_lang}:$<BUILD_INTERFACE:${GLOBAL_COMPILE_OPTIONS}>>)
target_link_libraries(test_scal_v2_performance gmp mpfr openblas OpenMP::OpenMP_CXX)

# Performance test for AXPY
add_executable(test_axpy_v2_performance blas/v2/axpy/test_axpy_v2_performance.cu)
target_compile_options(test_axpy_v2_performance PRIVATE  $<${cuda_lang}:$<BUILD_INTERFACE:${GLOBAL_COMPILE_OPTIONS}>>)
target_link_libraries(test_axpy_v2_performance gmp mpfr openblas OpenMP::OpenMP_CXX)

# Performance test for GER
add_executable(test_ger_v2_performance blas/v2/ger/test_ger_v2_performance.cu)
target_compile_options(test_ger_v2_performance PRIVATE  $<${cuda_lang}:$<BUILD_INTERFACE:${GLOBAL_COMPILE_OPTIONS}>>)
target_link_libraries(test_ger_v2_performance gmp mpfr openblas OpenMP::OpenMP_CXX cublas)

# Performance test for SYR
add_executable(test_syr_v2_performance blas/v2/syr/test_syr_v2_performance.cu)
target_compile_options(test_syr_v2_performance PRIVATE  $<${cuda_lang}:$<BUILD_INTERFACE:${GLOBAL_COMPILE_OPTIONS}>>)
target_link_libraries(test_syr_v2_performance gmp mpfr openblas OpenMP::OpenMP_CXX cublas)

# Performance test for SYR2
add_executable(test_syr2_v2_performance blas/v2/syr2/test_syr2_v2_performance.cu)
target_compile_options(test_syr2_v2_performance PRIVATE  $<${cuda_lang}:$<BUILD_INTERFACE:${GLOBAL_COMPILE_OPTIONS}>>)
target_link_libraries(test_syr2_v2_performance gmp mpfr openblas OpenMP::OpenMP_CXX cublas)

#########################
## Accuracy tests for BLAS routines
#########################

# Accuracy test for ASUM
add_executable(test_mpasum_accuracy blas/accuracy/test_asum_accuracy.cu)
target_compile_options(test_mpasum_accuracy PRIVATE $<${cuda_lang}:$<BUILD_INTERFACE:${GLOBAL_COMPILE_OPTIONS}>>)
target_link_libraries(test_mpasum_accuracy gmp mpfr OpenMP::OpenMP_CXX)

# Accuracy test for DOT
add_executable(test_mpdot_accuracy blas/accuracy/test_dot_accuracy.cu)
target_compile_options(test_mpdot_accuracy PRIVATE $<${cuda_lang}:$<BUILD_INTERFACE:${GLOBAL_COMPILE_OPTIONS}>>)
target_link_libraries(test_mpdot_accuracy gmp mpfr OpenMP::OpenMP_CXX)

# Accuracy test for SCAL
add_executable(test_mpscal_accuracy blas/accuracy/test_scal_accuracy.cu)
target_compile_options(test_mpscal_accuracy PRIVATE  $<${cuda_lang}:$<BUILD_INTERFACE:${GLOBAL_COMPILE_OPTIONS}>>)
target_link_libraries(test_mpscal_accuracy gmp mpfr OpenMP::OpenMP_CXX)

# Accuracy test for AXPY
add_executable(test_mpaxpy_accuracy blas/accuracy/test_axpy_accuracy.cu)
target_compile_options(test_mpaxpy_accuracy PRIVATE $<${cuda_lang}:$<BUILD_INTERFACE:${GLOBAL_COMPILE_OPTIONS}>>)
target_link_libraries(test_mpaxpy_accuracy gmp mpfr OpenMP::OpenMP_CXX)

#########################
## Benchmarks for SpMV routines (double precision matrix)
#########################

# Performance and memory consumption test for SpMV using various sparse matrix storage formats
add_executable(test_spmv sparse/test_spmv.cu)
target_compile_options(test_spmv PRIVATE  $<${cuda_lang}:$<BUILD_INTERFACE:${GLOBAL_COMPILE_OPTIONS}>>)
target_link_libraries(test_spmv gmp mpfr OpenMP::OpenMP_CXX cump)

# Performance test for SpMV in CSR format
add_executable(test_spmv_csr_performance sparse/test_spmv_csr_performance.cu)
target_compile_options(test_spmv_csr_performance PRIVATE  $<${cuda_lang}:$<BUILD_INTERFACE:${GLOBAL_COMPILE_OPTIONS}>>)
target_link_libraries(test_spmv_csr_performance gmp mpfr OpenMP::OpenMP_CXX cump taco)

# Performance test for SpMV in JAD (JDS) format
add_executable(test_spmv_jad_performance sparse/test_spmv_jad_performance.cu)
target_compile_options(test_spmv_jad_performance PRIVATE  $<${cuda_lang}:$<BUILD_INTERFACE:${GLOBAL_COMPILE_OPTIONS}>>)
target_link_libraries(test_spmv_jad_performance gmp mpfr OpenMP::OpenMP_CXX cump taco)

# Performance test for SpMV in ELLPACK format
add_executable(test_spmv_ell_performance sparse/test_spmv_ell_performance.cu)
target_compile_options(test_spmv_ell_performance PRIVATE  $<${cuda_lang}:$<BUILD_INTERFACE:${GLOBAL_COMPILE_OPTIONS}>>)
target_link_libraries(test_spmv_ell_performance gmp mpfr OpenMP::OpenMP_CXX cump taco)

# Performance test for SpMV in DIA format
add_executable(test_spmv_dia_performance sparse/test_spmv_dia_performance.cu)
target_compile_options(test_spmv_dia_performance PRIVATE  $<${cuda_lang}:$<BUILD_INTERFACE:${GLOBAL_COMPILE_OPTIONS}>>)
target_link_libraries(test_spmv_dia_performance gmp mpfr OpenMP::OpenMP_CXX cump taco)

# Accuracy test for SpMV in CSR format
add_executable(test_spmv_csr_accuracy sparse/test_spmv_csr_accuracy.cu)
target_compile_options(test_spmv_csr_accuracy PRIVATE  $<${cuda_lang}:$<BUILD_INTERFACE:${GLOBAL_COMPILE_OPTIONS}>>)
target_link_libraries(test_spmv_csr_accuracy gmp mpfr OpenMP::OpenMP_CXX)

# Performance test for SpMV vs GEMV
add_executable(test_spmv_gemv sparse/test_spmv_gemv.cu)
target_compile_options(test_spmv_gemv PRIVATE  $<${cuda_lang}:$<BUILD_INTERFACE:${GLOBAL_COMPILE_OPTIONS}>>)
target_link_libraries(test_spmv_gemv gmp mpfr OpenMP::OpenMP_CXX cump)


#########################
## Benchmarks for SpMV routines with multiple-precision matrix
#########################

# Performance test for SpMV in CSR format
add_executable(test_spmv_mpmtx_csr_performance sparse/mpmtx/test_spmv_mpmtx_csr_performance.cu)
target_compile_options(test_spmv_mpmtx_csr_performance PRIVATE  $<${cuda_lang}:$<BUILD_INTERFACE:${GLOBAL_COMPILE_OPTIONS}>>)
target_link_libraries(test_spmv_mpmtx_csr_performance gmp mpfr OpenMP::OpenMP_CXX cump taco)

# Performance test for SpMV in ELLPACK format
add_executable(test_spmv_mpmtx_ell_performance sparse/mpmtx/test_spmv_mpmtx_ell_performance.cu)
target_compile_options(test_spmv_mpmtx_ell_performance PRIVATE  $<${cuda_lang}:$<BUILD_INTERFACE:${GLOBAL_COMPILE_OPTIONS}>>)
target_link_libraries(test_spmv_mpmtx_ell_performance gmp mpfr OpenMP::OpenMP_CXX cump taco)

# Performance test for SpMV in DIA format
add_executable(test_spmv_mpmtx_dia_performance sparse/mpmtx/test_spmv_mpmtx_dia_performance.cu)
target_compile_options(test_spmv_mpmtx_dia_performance PRIVATE  $<${cuda_lang}:$<BUILD_INTERFACE:${GLOBAL_COMPILE_OPTIONS}>>)
target_link_libraries(test_spmv_mpmtx_dia_performance gmp mpfr OpenMP::OpenMP_CXX cump taco)

# Performance test for SpMV in JAD (JDS) format
add_executable(test_spmv_mpmtx_jad_performance sparse/mpmtx/test_spmv_mpmtx_jad_performance.cu)
target_compile_options(test_spmv_mpmtx_jad_performance PRIVATE  $<${cuda_lang}:$<BUILD_INTERFACE:${GLOBAL_COMPILE_OPTIONS}>>)
target_link_libraries(test_spmv_mpmtx_jad_performance gmp mpfr OpenMP::OpenMP_CXX cump taco)

#########################
## Benchmarks for iterative linear solvers
#########################
add_executable(test_solver_csr sparse/test_solver_csr.cu)
target_compile_options(test_solver_csr PRIVATE  $<${cuda_lang}:$<BUILD_INTERFACE:${GLOBAL_COMPILE_OPTIONS}>>)
target_link_libraries(test_solver_csr gmp mpfr OpenMP::OpenMP_CXX cublas)