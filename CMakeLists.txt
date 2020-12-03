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


cmake_minimum_required(VERSION 3.9)
project(MPRES_BLAS)

set(CMAKE_VERBOSE_MAKEFILE on)

#####################################
# Paths section
#####################################
set(MPRES_BLAS_HOME_DIR "${PROJECT_SOURCE_DIR}") # MPRES-BLAS home directory, should be fixed since PROJECT_SOURCE_DIR is changed
list(APPEND CMAKE_MODULE_PATH "${MPRES_BLAS_HOME_DIR}/cmake/") # CMake modules
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc") # Path to nvcc

#####################################
# CUDA
#####################################
set(cuda_lang "$<COMPILE_LANGUAGE:CUDA>")

#####################################
# Compile options
#####################################
#
set(GLOBAL_COMPILE_OPTIONS ${GLOBAL_COMPILE_OPTIONS} -Xcompiler=-fopenmp)

#####################################
# Directories
#####################################
include_directories(src/) # Source code
include_directories(tests/) # Benchmarks

#####################################
# Recurse into benchmarks directory
#####################################
add_subdirectory(tests/)