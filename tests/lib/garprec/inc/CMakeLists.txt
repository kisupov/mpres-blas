# Copyright (c) 2016 Vyatka State University
# All rights reserved.

cmake_minimum_required(VERSION 2.8)

project(garprec)
list(APPEND CMAKE_MODULE_PATH "cmake")   # CMake modules
find_package(CUDA REQUIRED)
find_package(ARPREC REQUIRED)

SET(CUDA_INCLUDE_DIRS "/usr/local/cuda/include" ${CUDA_INCLUDE_DIRS})
SET(CUDA_LIBRARIES ${CUDA_LIBRARIES} "/usr/local/cuda/lib64/libcudart.so" "/usr/lib64/libcuda.so" "/usr/local/cuda/lib64/libcublas.so")

#####################################
# Flags for CUDA library compilation
#####################################

if (CUDA_FOUND)
    #define for adding cuda functions in header files
    add_definitions(-DCUDA_COMPILE=true)
endif (CUDA_FOUND)
file(GLOB SOURCES "garprec.cu")

cuda_add_library(garprec STATIC garprec.cu garprec.cuh)
target_link_libraries(garprec arprec)


if(GENERATE_HEADERS)
    file(WRITE "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/mpres.cuh" "${ALL_FILES_CONTENT}")
endif()
