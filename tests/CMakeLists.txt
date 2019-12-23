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
## Performance tests for BLAS routines
#########################

# Performance test for ASUM
add_executable(mp_asum_performance_test blas/performance/mp_asum_performance_test.cu)
target_compile_options(mp_asum_performance_test PRIVATE  $<${cuda_lang}:$<BUILD_INTERFACE:${GLOBAL_COMPILE_OPTIONS}>>)
target_link_libraries(mp_asum_performance_test gmp mpfr arprec openblas xblas gmpxx mpfrcxx mblas_mpfr mpc OpenMP::OpenMP_CXX cublas mpdec cump)

# Performance test for DOT
add_executable(mp_dot_performance_test blas/performance/mp_dot_performance_test.cu)
target_compile_options(mp_dot_performance_test PRIVATE $<${cuda_lang}:$<BUILD_INTERFACE:${GLOBAL_COMPILE_OPTIONS}>>)
target_link_libraries(mp_dot_performance_test gmp mpfr arprec openblas xblas gmpxx mpfrcxx mblas_mpfr mpc OpenMP::OpenMP_CXX cublas mpdec cump)

# Performance test for SCAL
add_executable(mp_scal_performance_test blas/performance/mp_scal_performance_test.cu)
target_compile_options(mp_scal_performance_test PRIVATE $<${cuda_lang}:$<BUILD_INTERFACE:${GLOBAL_COMPILE_OPTIONS}>>)
target_link_libraries(mp_scal_performance_test gmp mpfr arprec openblas gmpxx mpfrcxx mblas_mpfr mpc OpenMP::OpenMP_CXX cublas mpdec cump)

# Performance test for AXPY
add_executable(mp_axpy_performance_test blas/performance/mp_axpy_performance_test.cu)
target_compile_options(mp_axpy_performance_test PRIVATE $<${cuda_lang}:$<BUILD_INTERFACE:${GLOBAL_COMPILE_OPTIONS}>>)
target_link_libraries(mp_axpy_performance_test gmp mpfr arprec openblas xblas gmpxx mpfrcxx mblas_mpfr mpc OpenMP::OpenMP_CXX cublas mpdec cump)

# Performance test for ROT
add_executable(mp_rot_performance_test blas/performance/mp_rot_performance_test.cu)
target_compile_options(mp_rot_performance_test PRIVATE  $<${cuda_lang}:$<BUILD_INTERFACE:${GLOBAL_COMPILE_OPTIONS}>>)
target_link_libraries(mp_rot_performance_test gmp mpfr arprec gmpxx mpfrcxx mblas_mpfr mpc OpenMP::OpenMP_CXX cump)

# Performance test for AXPY_DOT
add_executable(mp_axpy_dot_performance_test blas/performance/mp_axpy_dot_performance_test.cu)
target_compile_options(mp_axpy_dot_performance_test PRIVATE $<${cuda_lang}:$<BUILD_INTERFACE:${GLOBAL_COMPILE_OPTIONS}>>)
target_link_libraries(mp_axpy_dot_performance_test gmp mpfr arprec OpenMP::OpenMP_CXX cump)

#########################
## Accuracy tests for BLAS routines
#########################

# Accuracy test for ASUM
add_executable(mp_asum_accuracy_test blas/accuracy/mp_asum_accuracy_test.cu)
target_compile_options(mp_asum_accuracy_test PRIVATE $<${cuda_lang}:$<BUILD_INTERFACE:${GLOBAL_COMPILE_OPTIONS}>>)
target_link_libraries(mp_asum_accuracy_test gmp mpfr OpenMP::OpenMP_CXX)

# Accuracy test for DOT
add_executable(mp_dot_accuracy_test blas/accuracy/mp_dot_accuracy_test.cu)
target_compile_options(mp_dot_accuracy_test PRIVATE $<${cuda_lang}:$<BUILD_INTERFACE:${GLOBAL_COMPILE_OPTIONS}>>)
target_link_libraries(mp_dot_accuracy_test gmp mpfr OpenMP::OpenMP_CXX)

# Accuracy test for SCAL
add_executable(mp_scal_accuracy_test blas/accuracy/mp_scal_accuracy_test.cu)
target_compile_options(mp_scal_accuracy_test PRIVATE  $<${cuda_lang}:$<BUILD_INTERFACE:${GLOBAL_COMPILE_OPTIONS}>>)
target_link_libraries(mp_scal_accuracy_test gmp mpfr OpenMP::OpenMP_CXX)

# Accuracy test for AXPY
add_executable(mp_axpy_accuracy_test blas/accuracy/mp_axpy_accuracy_test.cu)
target_compile_options(mp_axpy_accuracy_test PRIVATE $<${cuda_lang}:$<BUILD_INTERFACE:${GLOBAL_COMPILE_OPTIONS}>>)
target_link_libraries(mp_axpy_accuracy_test gmp mpfr OpenMP::OpenMP_CXX)


#########################
## Basic arith tests
#########################

# Test for checking the algorithms that calculate the interval evaluation of an RNS number
add_executable(rns_eval_accuracy_test arith/rns_eval_accuracy_test.cu)
target_compile_options(rns_eval_accuracy_test PRIVATE  $<${cuda_lang}:$<BUILD_INTERFACE:${GLOBAL_COMPILE_OPTIONS}>>)
target_link_libraries(rns_eval_accuracy_test gmp mpfr)

# Test for validating the mp_add routine
add_executable(mp_add_correctness_test arith/mp_add_correctness_test.cu)
target_compile_options(mp_add_correctness_test PRIVATE $<${cuda_lang}:$<BUILD_INTERFACE:${GLOBAL_COMPILE_OPTIONS}>>)
target_link_libraries(mp_add_correctness_test gmp mpfr)