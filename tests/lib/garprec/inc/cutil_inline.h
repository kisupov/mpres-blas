
#include <cuda.h>
#include <cufft.h>
#include <cublas.h>
#include <cuda_runtime_api.h>
#include <math_constants.h>


#define cutilSafeCall(err)  __cudaSafeCall(err,__FILE__,__LINE__)
#define cufftSafeCall(err)  __cufftSafeCall(err,__FILE__,__LINE__)
#define cublasSafeCall(err) __cublasSafeCall(err,__FILE__,__LINE__)
#define cutilCheckMsg(msg)  __cutilCheckMsg(msg,__FILE__,__LINE__)

inline void __cudaSafeCall(cudaError err,
                           const char *file, const int line){
    if(cudaSuccess != err) {
        printf("%s(%i) : cutilSafeCall() Runtime API error : %s.\n",
               file, line, cudaGetErrorString(err) );
        exit(-1);
    }
}

inline void __cufftSafeCall(cufftResult err,
                            const char *file, const int line){
    if(CUFFT_SUCCESS != err) {
        printf("%s(%i) : cufftSafeCall() CUFFT error.\n", file, line);
        exit(-1);
    }
}

inline void __cublasSafeCall(cublasStatus err,
                             const char *file, const int line){
    if(CUBLAS_STATUS_SUCCESS != err) {
        printf("%s(%i) : cublasSafeCall() CUBLAS error.\n", file, line);
        exit(-1);
    }
}

inline void __cutilCheckMsg(const char *errorMessage,
                            const char *file, const int line) {
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err) {
        printf("%s(%i) : cutilCheckMsg() error : %s : %s.\n",
               file, line, errorMessage, cudaGetErrorString(err) );
        exit(-1);
    }
}

// this one is modified from its original form

inline void cutilDeviceInit(int argc, char **argv) {
    int            dev, deviceCount;
    cudaDeviceProp devProp;

    // optional selection of CUDA device dependent on Makefile

#ifdef CUDA_DEVICE
    printf("\n Setting CUDA device %d\n",CUDA_DEVICE);
  cudaSetDevice(CUDA_DEVICE);
#endif

    cutilSafeCall(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        printf("cutil error: no devices supporting CUDA\n");
        exit(-1);
    }

    cutilSafeCall(cudaGetDevice(&dev));
    cutilSafeCall(cudaGetDeviceProperties(&devProp,dev));
    printf("\n Using CUDA device %d: %s\n\n", dev,devProp.name);
}


//
// linux timing routine
//

#include <sys/time.h>

inline double elapsed_time(double *et) {
    struct timeval t;

    double old_time = *et;

    gettimeofday( &t, (struct timezone *)0 );
    *et = t.tv_sec + t.tv_usec*1.0e-6;

    return *et - old_time;
}

//
// Created by matroskinb on 22.11.17.
//

#ifndef MPRES_PROJECT_CUDA_HEADERS_H
/**
 * Copyright 1993-2013 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

////////////////////////////////////////////////////////////////////////////////
// These are CUDA Helper functions for initialization and error checking

#ifndef HELPER_CUDA_H
#define HELPER_CUDA_H

#pragma once

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "helper_string.h"

#ifndef EXIT_WAIVED
#define EXIT_WAIVED 2
#endif

// Note, it is required that your SDK sample to include the proper header files, please
// refer the CUDA examples for examples of the needed CUDA headers, which may change depending
// on which CUDA functions are used.

// CUDA Runtime error messages
#ifdef __DRIVER_TYPES_H__
static const char *_cudaGetErrorEnum(cudaError_t error)
{
    switch (error)
    {
        case cudaSuccess:
            return "cudaSuccess";

        case cudaErrorMissingConfiguration:
            return "cudaErrorMissingConfiguration";

        case cudaErrorMemoryAllocation:
            return "cudaErrorMemoryAllocation";

        case cudaErrorInitializationError:
            return "cudaErrorInitializationError";

        case cudaErrorLaunchFailure:
            return "cudaErrorLaunchFailure";

        case cudaErrorPriorLaunchFailure:
            return "cudaErrorPriorLaunchFailure";

        case cudaErrorLaunchTimeout:
            return "cudaErrorLaunchTimeout";

        case cudaErrorLaunchOutOfResources:
            return "cudaErrorLaunchOutOfResources";

        case cudaErrorInvalidDeviceFunction:
            return "cudaErrorInvalidDeviceFunction";

        case cudaErrorInvalidConfiguration:
            return "cudaErrorInvalidConfiguration";

        case cudaErrorInvalidDevice:
            return "cudaErrorInvalidDevice";

        case cudaErrorInvalidValue:
            return "cudaErrorInvalidValue";

        case cudaErrorInvalidPitchValue:
            return "cudaErrorInvalidPitchValue";

        case cudaErrorInvalidSymbol:
            return "cudaErrorInvalidSymbol";

        case cudaErrorMapBufferObjectFailed:
            return "cudaErrorMapBufferObjectFailed";

        case cudaErrorUnmapBufferObjectFailed:
            return "cudaErrorUnmapBufferObjectFailed";

        case cudaErrorInvalidHostPointer:
            return "cudaErrorInvalidHostPointer";

        case cudaErrorInvalidDevicePointer:
            return "cudaErrorInvalidDevicePointer";

        case cudaErrorInvalidTexture:
            return "cudaErrorInvalidTexture";

        case cudaErrorInvalidTextureBinding:
            return "cudaErrorInvalidTextureBinding";

        case cudaErrorInvalidChannelDescriptor:
            return "cudaErrorInvalidChannelDescriptor";

        case cudaErrorInvalidMemcpyDirection:
            return "cudaErrorInvalidMemcpyDirection";

        case cudaErrorAddressOfConstant:
            return "cudaErrorAddressOfConstant";

        case cudaErrorTextureFetchFailed:
            return "cudaErrorTextureFetchFailed";

        case cudaErrorTextureNotBound:
            return "cudaErrorTextureNotBound";

        case cudaErrorSynchronizationError:
            return "cudaErrorSynchronizationError";

        case cudaErrorInvalidFilterSetting:
            return "cudaErrorInvalidFilterSetting";

        case cudaErrorInvalidNormSetting:
            return "cudaErrorInvalidNormSetting";

        case cudaErrorMixedDeviceExecution:
            return "cudaErrorMixedDeviceExecution";

        case cudaErrorCudartUnloading:
            return "cudaErrorCudartUnloading";

        case cudaErrorUnknown:
            return "cudaErrorUnknown";

        case cudaErrorNotYetImplemented:
            return "cudaErrorNotYetImplemented";

        case cudaErrorMemoryValueTooLarge:
            return "cudaErrorMemoryValueTooLarge";

        case cudaErrorInvalidResourceHandle:
            return "cudaErrorInvalidResourceHandle";

        case cudaErrorNotReady:
            return "cudaErrorNotReady";

        case cudaErrorInsufficientDriver:
            return "cudaErrorInsufficientDriver";

        case cudaErrorSetOnActiveProcess:
            return "cudaErrorSetOnActiveProcess";

        case cudaErrorInvalidSurface:
            return "cudaErrorInvalidSurface";

        case cudaErrorNoDevice:
            return "cudaErrorNoDevice";

        case cudaErrorECCUncorrectable:
            return "cudaErrorECCUncorrectable";

        case cudaErrorSharedObjectSymbolNotFound:
            return "cudaErrorSharedObjectSymbolNotFound";

        case cudaErrorSharedObjectInitFailed:
            return "cudaErrorSharedObjectInitFailed";

        case cudaErrorUnsupportedLimit:
            return "cudaErrorUnsupportedLimit";

        case cudaErrorDuplicateVariableName:
            return "cudaErrorDuplicateVariableName";

        case cudaErrorDuplicateTextureName:
            return "cudaErrorDuplicateTextureName";

        case cudaErrorDuplicateSurfaceName:
            return "cudaErrorDuplicateSurfaceName";

        case cudaErrorDevicesUnavailable:
            return "cudaErrorDevicesUnavailable";

        case cudaErrorInvalidKernelImage:
            return "cudaErrorInvalidKernelImage";

        case cudaErrorNoKernelImageForDevice:
            return "cudaErrorNoKernelImageForDevice";

        case cudaErrorIncompatibleDriverContext:
            return "cudaErrorIncompatibleDriverContext";

        case cudaErrorPeerAccessAlreadyEnabled:
            return "cudaErrorPeerAccessAlreadyEnabled";

        case cudaErrorPeerAccessNotEnabled:
            return "cudaErrorPeerAccessNotEnabled";

        case cudaErrorDeviceAlreadyInUse:
            return "cudaErrorDeviceAlreadyInUse";

        case cudaErrorProfilerDisabled:
            return "cudaErrorProfilerDisabled";

        case cudaErrorProfilerNotInitialized:
            return "cudaErrorProfilerNotInitialized";

        case cudaErrorProfilerAlreadyStarted:
            return "cudaErrorProfilerAlreadyStarted";

        case cudaErrorProfilerAlreadyStopped:
            return "cudaErrorProfilerAlreadyStopped";

        /* Since CUDA 4.0*/
        case cudaErrorAssert:
            return "cudaErrorAssert";

        case cudaErrorTooManyPeers:
            return "cudaErrorTooManyPeers";

        case cudaErrorHostMemoryAlreadyRegistered:
            return "cudaErrorHostMemoryAlreadyRegistered";

        case cudaErrorHostMemoryNotRegistered:
            return "cudaErrorHostMemoryNotRegistered";

        /* Since CUDA 5.0 */
        case cudaErrorOperatingSystem:
            return "cudaErrorOperatingSystem";

        case cudaErrorPeerAccessUnsupported:
            return "cudaErrorPeerAccessUnsupported";

        case cudaErrorLaunchMaxDepthExceeded:
            return "cudaErrorLaunchMaxDepthExceeded";

        case cudaErrorLaunchFileScopedTex:
            return "cudaErrorLaunchFileScopedTex";

        case cudaErrorLaunchFileScopedSurf:
            return "cudaErrorLaunchFileScopedSurf";

        case cudaErrorSyncDepthExceeded:
            return "cudaErrorSyncDepthExceeded";

        case cudaErrorLaunchPendingCountExceeded:
            return "cudaErrorLaunchPendingCountExceeded";

        case cudaErrorNotPermitted:
            return "cudaErrorNotPermitted";

        case cudaErrorNotSupported:
            return "cudaErrorNotSupported";

        /* Since CUDA 6.0 */
        case cudaErrorHardwareStackError:
            return "cudaErrorHardwareStackError";

        case cudaErrorIllegalInstruction:
            return "cudaErrorIllegalInstruction";

        case cudaErrorMisalignedAddress:
            return "cudaErrorMisalignedAddress";

        case cudaErrorInvalidAddressSpace:
            return "cudaErrorInvalidAddressSpace";

        case cudaErrorInvalidPc:
            return "cudaErrorInvalidPc";

        case cudaErrorIllegalAddress:
            return "cudaErrorIllegalAddress";

        /* Since CUDA 6.5*/
        case cudaErrorInvalidPtx:
            return "cudaErrorInvalidPtx";

        case cudaErrorInvalidGraphicsContext:
            return "cudaErrorInvalidGraphicsContext";

        case cudaErrorStartupFailure:
            return "cudaErrorStartupFailure";

        case cudaErrorApiFailureBase:
            return "cudaErrorApiFailureBase";

        /* Since CUDA 8.0*/
        case cudaErrorNvlinkUncorrectable :
            return "cudaErrorNvlinkUncorrectable";

        /* Since CUDA 8.5*/
        case cudaErrorJitCompilerNotFound :
            return "cudaErrorJitCompilerNotFound";

        /* Since CUDA 9.0*/
        case cudaErrorCooperativeLaunchTooLarge :
            return "cudaErrorCooperativeLaunchTooLarge";

    }

    return "<unknown>";
}
#endif

#ifdef __cuda_cuda_h__
// CUDA Driver API errors
static const char *_cudaGetErrorEnum(CUresult error)
{
    switch (error)
    {
        case CUDA_SUCCESS:
            return "CUDA_SUCCESS";

        case CUDA_ERROR_INVALID_VALUE:
            return "CUDA_ERROR_INVALID_VALUE";

        case CUDA_ERROR_OUT_OF_MEMORY:
            return "CUDA_ERROR_OUT_OF_MEMORY";

        case CUDA_ERROR_NOT_INITIALIZED:
            return "CUDA_ERROR_NOT_INITIALIZED";

        case CUDA_ERROR_DEINITIALIZED:
            return "CUDA_ERROR_DEINITIALIZED";

        case CUDA_ERROR_PROFILER_DISABLED:
            return "CUDA_ERROR_PROFILER_DISABLED";

        case CUDA_ERROR_PROFILER_NOT_INITIALIZED:
            return "CUDA_ERROR_PROFILER_NOT_INITIALIZED";

        case CUDA_ERROR_PROFILER_ALREADY_STARTED:
            return "CUDA_ERROR_PROFILER_ALREADY_STARTED";

        case CUDA_ERROR_PROFILER_ALREADY_STOPPED:
            return "CUDA_ERROR_PROFILER_ALREADY_STOPPED";

        case CUDA_ERROR_NO_DEVICE:
            return "CUDA_ERROR_NO_DEVICE";

        case CUDA_ERROR_INVALID_DEVICE:
            return "CUDA_ERROR_INVALID_DEVICE";

        case CUDA_ERROR_INVALID_IMAGE:
            return "CUDA_ERROR_INVALID_IMAGE";

        case CUDA_ERROR_INVALID_CONTEXT:
            return "CUDA_ERROR_INVALID_CONTEXT";

        case CUDA_ERROR_CONTEXT_ALREADY_CURRENT:
            return "CUDA_ERROR_CONTEXT_ALREADY_CURRENT";

        case CUDA_ERROR_MAP_FAILED:
            return "CUDA_ERROR_MAP_FAILED";

        case CUDA_ERROR_UNMAP_FAILED:
            return "CUDA_ERROR_UNMAP_FAILED";

        case CUDA_ERROR_ARRAY_IS_MAPPED:
            return "CUDA_ERROR_ARRAY_IS_MAPPED";

        case CUDA_ERROR_ALREADY_MAPPED:
            return "CUDA_ERROR_ALREADY_MAPPED";

        case CUDA_ERROR_NO_BINARY_FOR_GPU:
            return "CUDA_ERROR_NO_BINARY_FOR_GPU";

        case CUDA_ERROR_ALREADY_ACQUIRED:
            return "CUDA_ERROR_ALREADY_ACQUIRED";

        case CUDA_ERROR_NOT_MAPPED:
            return "CUDA_ERROR_NOT_MAPPED";

        case CUDA_ERROR_NOT_MAPPED_AS_ARRAY:
            return "CUDA_ERROR_NOT_MAPPED_AS_ARRAY";

        case CUDA_ERROR_NOT_MAPPED_AS_POINTER:
            return "CUDA_ERROR_NOT_MAPPED_AS_POINTER";

        case CUDA_ERROR_ECC_UNCORRECTABLE:
            return "CUDA_ERROR_ECC_UNCORRECTABLE";

        case CUDA_ERROR_UNSUPPORTED_LIMIT:
            return "CUDA_ERROR_UNSUPPORTED_LIMIT";

        case CUDA_ERROR_CONTEXT_ALREADY_IN_USE:
            return "CUDA_ERROR_CONTEXT_ALREADY_IN_USE";

        case CUDA_ERROR_PEER_ACCESS_UNSUPPORTED:
            return "CUDA_ERROR_PEER_ACCESS_UNSUPPORTED";

        case CUDA_ERROR_INVALID_PTX:
            return "CUDA_ERROR_INVALID_PTX";

        case CUDA_ERROR_INVALID_GRAPHICS_CONTEXT:
            return "CUDA_ERROR_INVALID_GRAPHICS_CONTEXT";

        case CUDA_ERROR_NVLINK_UNCORRECTABLE:
            return "CUDA_ERROR_NVLINK_UNCORRECTABLE";

        case CUDA_ERROR_JIT_COMPILER_NOT_FOUND:
            return "CUDA_ERROR_JIT_COMPILER_NOT_FOUND";

        case CUDA_ERROR_INVALID_SOURCE:
            return "CUDA_ERROR_INVALID_SOURCE";

        case CUDA_ERROR_FILE_NOT_FOUND:
            return "CUDA_ERROR_FILE_NOT_FOUND";

        case CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND:
            return "CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND";

        case CUDA_ERROR_SHARED_OBJECT_INIT_FAILED:
            return "CUDA_ERROR_SHARED_OBJECT_INIT_FAILED";

        case CUDA_ERROR_OPERATING_SYSTEM:
            return "CUDA_ERROR_OPERATING_SYSTEM";

        case CUDA_ERROR_INVALID_HANDLE:
            return "CUDA_ERROR_INVALID_HANDLE";

        case CUDA_ERROR_NOT_FOUND:
            return "CUDA_ERROR_NOT_FOUND";

        case CUDA_ERROR_NOT_READY:
            return "CUDA_ERROR_NOT_READY";

        case CUDA_ERROR_ILLEGAL_ADDRESS:
            return "CUDA_ERROR_ILLEGAL_ADDRESS";

        case CUDA_ERROR_LAUNCH_FAILED:
            return "CUDA_ERROR_LAUNCH_FAILED";

        case CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES:
            return "CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES";

        case CUDA_ERROR_LAUNCH_TIMEOUT:
            return "CUDA_ERROR_LAUNCH_TIMEOUT";

        case CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING:
            return "CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING";

        case CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED:
            return "CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED";

        case CUDA_ERROR_PEER_ACCESS_NOT_ENABLED:
            return "CUDA_ERROR_PEER_ACCESS_NOT_ENABLED";

        case CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE:
            return "CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE";

        case CUDA_ERROR_CONTEXT_IS_DESTROYED:
            return "CUDA_ERROR_CONTEXT_IS_DESTROYED";

        case CUDA_ERROR_ASSERT:
            return "CUDA_ERROR_ASSERT";

        case CUDA_ERROR_TOO_MANY_PEERS:
            return "CUDA_ERROR_TOO_MANY_PEERS";

        case CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED:
            return "CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED";

        case CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED:
            return "CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED";

        case CUDA_ERROR_HARDWARE_STACK_ERROR:
            return "CUDA_ERROR_HARDWARE_STACK_ERROR";

        case CUDA_ERROR_ILLEGAL_INSTRUCTION:
            return "CUDA_ERROR_ILLEGAL_INSTRUCTION";

        case CUDA_ERROR_MISALIGNED_ADDRESS:
            return "CUDA_ERROR_MISALIGNED_ADDRESS";

        case CUDA_ERROR_INVALID_ADDRESS_SPACE:
            return "CUDA_ERROR_INVALID_ADDRESS_SPACE";

        case CUDA_ERROR_INVALID_PC:
            return "CUDA_ERROR_INVALID_PC";

        case CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE:
            return "CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE";

        case CUDA_ERROR_NOT_PERMITTED:
            return "CUDA_ERROR_NOT_PERMITTED";

        case CUDA_ERROR_NOT_SUPPORTED:
            return "CUDA_ERROR_NOT_SUPPORTED";

        case CUDA_ERROR_UNKNOWN:
            return "CUDA_ERROR_UNKNOWN";
    }

    return "<unknown>";
}
#endif

#ifdef CUBLAS_API_H_
// cuBLAS API errors
static const char *_cudaGetErrorEnum(cublasStatus_t error)
{
    switch (error)
    {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";

        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";

        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";

        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";

        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";

        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";

        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";

        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";

        case CUBLAS_STATUS_NOT_SUPPORTED:
            return "CUBLAS_STATUS_NOT_SUPPORTED";

        case CUBLAS_STATUS_LICENSE_ERROR:
            return "CUBLAS_STATUS_LICENSE_ERROR";
    }

    return "<unknown>";
}
#endif

#ifdef _CUFFT_H_
// cuFFT API errors
static const char *_cudaGetErrorEnum(cufftResult error)
{
    switch (error)
    {
        case CUFFT_SUCCESS:
            return "CUFFT_SUCCESS";

        case CUFFT_INVALID_PLAN:
            return "CUFFT_INVALID_PLAN";

        case CUFFT_ALLOC_FAILED:
            return "CUFFT_ALLOC_FAILED";

        case CUFFT_INVALID_TYPE:
            return "CUFFT_INVALID_TYPE";

        case CUFFT_INVALID_VALUE:
            return "CUFFT_INVALID_VALUE";

        case CUFFT_INTERNAL_ERROR:
            return "CUFFT_INTERNAL_ERROR";

        case CUFFT_EXEC_FAILED:
            return "CUFFT_EXEC_FAILED";

        case CUFFT_SETUP_FAILED:
            return "CUFFT_SETUP_FAILED";

        case CUFFT_INVALID_SIZE:
            return "CUFFT_INVALID_SIZE";

        case CUFFT_UNALIGNED_DATA:
            return "CUFFT_UNALIGNED_DATA";

        case CUFFT_INCOMPLETE_PARAMETER_LIST:
            return "CUFFT_INCOMPLETE_PARAMETER_LIST";

        case CUFFT_INVALID_DEVICE:
            return "CUFFT_INVALID_DEVICE";

        case CUFFT_PARSE_ERROR:
            return "CUFFT_PARSE_ERROR";

        case CUFFT_NO_WORKSPACE:
            return "CUFFT_NO_WORKSPACE";

        case CUFFT_NOT_IMPLEMENTED:
            return "CUFFT_NOT_IMPLEMENTED";

        case CUFFT_LICENSE_ERROR:
            return "CUFFT_LICENSE_ERROR";

        case CUFFT_NOT_SUPPORTED:
            return "CUFFT_NOT_SUPPORTED";
    }

    return "<unknown>";
}
#endif


#ifdef CUSPARSEAPI
// cuSPARSE API errors
static const char *_cudaGetErrorEnum(cusparseStatus_t error)
{
    switch (error)
    {
        case CUSPARSE_STATUS_SUCCESS:
            return "CUSPARSE_STATUS_SUCCESS";

        case CUSPARSE_STATUS_NOT_INITIALIZED:
            return "CUSPARSE_STATUS_NOT_INITIALIZED";

        case CUSPARSE_STATUS_ALLOC_FAILED:
            return "CUSPARSE_STATUS_ALLOC_FAILED";

        case CUSPARSE_STATUS_INVALID_VALUE:
            return "CUSPARSE_STATUS_INVALID_VALUE";

        case CUSPARSE_STATUS_ARCH_MISMATCH:
            return "CUSPARSE_STATUS_ARCH_MISMATCH";

        case CUSPARSE_STATUS_MAPPING_ERROR:
            return "CUSPARSE_STATUS_MAPPING_ERROR";

        case CUSPARSE_STATUS_EXECUTION_FAILED:
            return "CUSPARSE_STATUS_EXECUTION_FAILED";

        case CUSPARSE_STATUS_INTERNAL_ERROR:
            return "CUSPARSE_STATUS_INTERNAL_ERROR";

        case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
            return "CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
    }

    return "<unknown>";
}
#endif

#ifdef CUSOLVER_COMMON_H_
//cuSOLVER API errors
static const char *_cudaGetErrorEnum(cusolverStatus_t error)
{
   switch(error)
   {
       case CUSOLVER_STATUS_SUCCESS:
           return "CUSOLVER_STATUS_SUCCESS";
       case CUSOLVER_STATUS_NOT_INITIALIZED:
           return "CUSOLVER_STATUS_NOT_INITIALIZED";
       case CUSOLVER_STATUS_ALLOC_FAILED:
           return "CUSOLVER_STATUS_ALLOC_FAILED";
       case CUSOLVER_STATUS_INVALID_VALUE:
           return "CUSOLVER_STATUS_INVALID_VALUE";
       case CUSOLVER_STATUS_ARCH_MISMATCH:
           return "CUSOLVER_STATUS_ARCH_MISMATCH";
       case CUSOLVER_STATUS_MAPPING_ERROR:
           return "CUSOLVER_STATUS_MAPPING_ERROR";
       case CUSOLVER_STATUS_EXECUTION_FAILED:
           return "CUSOLVER_STATUS_EXECUTION_FAILED";
       case CUSOLVER_STATUS_INTERNAL_ERROR:
           return "CUSOLVER_STATUS_INTERNAL_ERROR";
       case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
           return "CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
       case CUSOLVER_STATUS_NOT_SUPPORTED :
           return "CUSOLVER_STATUS_NOT_SUPPORTED ";
       case CUSOLVER_STATUS_ZERO_PIVOT:
           return "CUSOLVER_STATUS_ZERO_PIVOT";
       case CUSOLVER_STATUS_INVALID_LICENSE:
           return "CUSOLVER_STATUS_INVALID_LICENSE";
    }

    return "<unknown>";

}
#endif

#ifdef CURAND_H_
// cuRAND API errors
static const char *_cudaGetErrorEnum(curandStatus_t error)
{
    switch (error)
    {
        case CURAND_STATUS_SUCCESS:
            return "CURAND_STATUS_SUCCESS";

        case CURAND_STATUS_VERSION_MISMATCH:
            return "CURAND_STATUS_VERSION_MISMATCH";

        case CURAND_STATUS_NOT_INITIALIZED:
            return "CURAND_STATUS_NOT_INITIALIZED";

        case CURAND_STATUS_ALLOCATION_FAILED:
            return "CURAND_STATUS_ALLOCATION_FAILED";

        case CURAND_STATUS_TYPE_ERROR:
            return "CURAND_STATUS_TYPE_ERROR";

        case CURAND_STATUS_OUT_OF_RANGE:
            return "CURAND_STATUS_OUT_OF_RANGE";

        case CURAND_STATUS_LENGTH_NOT_MULTIPLE:
            return "CURAND_STATUS_LENGTH_NOT_MULTIPLE";

        case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:
            return "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED";

        case CURAND_STATUS_LAUNCH_FAILURE:
            return "CURAND_STATUS_LAUNCH_FAILURE";

        case CURAND_STATUS_PREEXISTING_FAILURE:
            return "CURAND_STATUS_PREEXISTING_FAILURE";

        case CURAND_STATUS_INITIALIZATION_FAILED:
            return "CURAND_STATUS_INITIALIZATION_FAILED";

        case CURAND_STATUS_ARCH_MISMATCH:
            return "CURAND_STATUS_ARCH_MISMATCH";

        case CURAND_STATUS_INTERNAL_ERROR:
            return "CURAND_STATUS_INTERNAL_ERROR";
    }

    return "<unknown>";
}
#endif

#ifdef NV_NPPIDEFS_H
// NPP API errors
static const char *_cudaGetErrorEnum(NppStatus error)
{
    switch (error)
    {
        case NPP_NOT_SUPPORTED_MODE_ERROR:
            return "NPP_NOT_SUPPORTED_MODE_ERROR";

        case NPP_ROUND_MODE_NOT_SUPPORTED_ERROR:
            return "NPP_ROUND_MODE_NOT_SUPPORTED_ERROR";

        case NPP_RESIZE_NO_OPERATION_ERROR:
            return "NPP_RESIZE_NO_OPERATION_ERROR";

        case NPP_NOT_SUFFICIENT_COMPUTE_CAPABILITY:
            return "NPP_NOT_SUFFICIENT_COMPUTE_CAPABILITY";

#if ((NPP_VERSION_MAJOR << 12) + (NPP_VERSION_MINOR << 4)) <= 0x5000

        case NPP_BAD_ARG_ERROR:
            return "NPP_BAD_ARGUMENT_ERROR";

        case NPP_COEFF_ERROR:
            return "NPP_COEFFICIENT_ERROR";

        case NPP_RECT_ERROR:
            return "NPP_RECTANGLE_ERROR";

        case NPP_QUAD_ERROR:
            return "NPP_QUADRANGLE_ERROR";

        case NPP_MEM_ALLOC_ERR:
            return "NPP_MEMORY_ALLOCATION_ERROR";

        case NPP_HISTO_NUMBER_OF_LEVELS_ERROR:
            return "NPP_HISTOGRAM_NUMBER_OF_LEVELS_ERROR";

        case NPP_INVALID_INPUT:
            return "NPP_INVALID_INPUT";

        case NPP_POINTER_ERROR:
            return "NPP_POINTER_ERROR";

        case NPP_WARNING:
            return "NPP_WARNING";

        case NPP_ODD_ROI_WARNING:
            return "NPP_ODD_ROI_WARNING";
#else

            // These are for CUDA 5.5 or higher
        case NPP_BAD_ARGUMENT_ERROR:
            return "NPP_BAD_ARGUMENT_ERROR";

        case NPP_COEFFICIENT_ERROR:
            return "NPP_COEFFICIENT_ERROR";

        case NPP_RECTANGLE_ERROR:
            return "NPP_RECTANGLE_ERROR";

        case NPP_QUADRANGLE_ERROR:
            return "NPP_QUADRANGLE_ERROR";

        case NPP_MEMORY_ALLOCATION_ERR:
            return "NPP_MEMORY_ALLOCATION_ERROR";

        case NPP_HISTOGRAM_NUMBER_OF_LEVELS_ERROR:
            return "NPP_HISTOGRAM_NUMBER_OF_LEVELS_ERROR";

        case NPP_INVALID_HOST_POINTER_ERROR:
            return "NPP_INVALID_HOST_POINTER_ERROR";

        case NPP_INVALID_DEVICE_POINTER_ERROR:
            return "NPP_INVALID_DEVICE_POINTER_ERROR";
#endif

        case NPP_LUT_NUMBER_OF_LEVELS_ERROR:
            return "NPP_LUT_NUMBER_OF_LEVELS_ERROR";

        case NPP_TEXTURE_BIND_ERROR:
            return "NPP_TEXTURE_BIND_ERROR";

        case NPP_WRONG_INTERSECTION_ROI_ERROR:
            return "NPP_WRONG_INTERSECTION_ROI_ERROR";

        case NPP_NOT_EVEN_STEP_ERROR:
            return "NPP_NOT_EVEN_STEP_ERROR";

        case NPP_INTERPOLATION_ERROR:
            return "NPP_INTERPOLATION_ERROR";

        case NPP_RESIZE_FACTOR_ERROR:
            return "NPP_RESIZE_FACTOR_ERROR";

        case NPP_HAAR_CLASSIFIER_PIXEL_MATCH_ERROR:
            return "NPP_HAAR_CLASSIFIER_PIXEL_MATCH_ERROR";


#if ((NPP_VERSION_MAJOR << 12) + (NPP_VERSION_MINOR << 4)) <= 0x5000

        case NPP_MEMFREE_ERR:
            return "NPP_MEMFREE_ERR";

        case NPP_MEMSET_ERR:
            return "NPP_MEMSET_ERR";

        case NPP_MEMCPY_ERR:
            return "NPP_MEMCPY_ERROR";

        case NPP_MIRROR_FLIP_ERR:
            return "NPP_MIRROR_FLIP_ERR";
#else

        case NPP_MEMFREE_ERROR:
            return "NPP_MEMFREE_ERROR";

        case NPP_MEMSET_ERROR:
            return "NPP_MEMSET_ERROR";

        case NPP_MEMCPY_ERROR:
            return "NPP_MEMCPY_ERROR";

        case NPP_MIRROR_FLIP_ERROR:
            return "NPP_MIRROR_FLIP_ERROR";
#endif

        case NPP_ALIGNMENT_ERROR:
            return "NPP_ALIGNMENT_ERROR";

        case NPP_STEP_ERROR:
            return "NPP_STEP_ERROR";

        case NPP_SIZE_ERROR:
            return "NPP_SIZE_ERROR";

        case NPP_NULL_POINTER_ERROR:
            return "NPP_NULL_POINTER_ERROR";

        case NPP_CUDA_KERNEL_EXECUTION_ERROR:
            return "NPP_CUDA_KERNEL_EXECUTION_ERROR";

        case NPP_NOT_IMPLEMENTED_ERROR:
            return "NPP_NOT_IMPLEMENTED_ERROR";

        case NPP_ERROR:
            return "NPP_ERROR";

        case NPP_SUCCESS:
            return "NPP_SUCCESS";

        case NPP_WRONG_INTERSECTION_QUAD_WARNING:
            return "NPP_WRONG_INTERSECTION_QUAD_WARNING";

        case NPP_MISALIGNED_DST_ROI_WARNING:
            return "NPP_MISALIGNED_DST_ROI_WARNING";

        case NPP_AFFINE_QUAD_INCORRECT_WARNING:
            return "NPP_AFFINE_QUAD_INCORRECT_WARNING";

        case NPP_DOUBLE_SIZE_WARNING:
            return "NPP_DOUBLE_SIZE_WARNING";

        case NPP_WRONG_INTERSECTION_ROI_WARNING:
            return "NPP_WRONG_INTERSECTION_ROI_WARNING";

#if ((NPP_VERSION_MAJOR << 12) + (NPP_VERSION_MINOR << 4)) >= 0x6000
        /* These are 6.0 or higher */
        case NPP_LUT_PALETTE_BITSIZE_ERROR:
            return "NPP_LUT_PALETTE_BITSIZE_ERROR";

        case NPP_ZC_MODE_NOT_SUPPORTED_ERROR:
            return "NPP_ZC_MODE_NOT_SUPPORTED_ERROR";

        case NPP_QUALITY_INDEX_ERROR:
            return "NPP_QUALITY_INDEX_ERROR";

        case NPP_CHANNEL_ORDER_ERROR:
            return "NPP_CHANNEL_ORDER_ERROR";

        case NPP_ZERO_MASK_VALUE_ERROR:
            return "NPP_ZERO_MASK_VALUE_ERROR";

        case NPP_NUMBER_OF_CHANNELS_ERROR:
            return "NPP_NUMBER_OF_CHANNELS_ERROR";

        case NPP_COI_ERROR:
            return "NPP_COI_ERROR";

        case NPP_DIVISOR_ERROR:
            return "NPP_DIVISOR_ERROR";

        case NPP_CHANNEL_ERROR:
            return "NPP_CHANNEL_ERROR";

        case NPP_STRIDE_ERROR:
            return "NPP_STRIDE_ERROR";

        case NPP_ANCHOR_ERROR:
            return "NPP_ANCHOR_ERROR";

        case NPP_MASK_SIZE_ERROR:
            return "NPP_MASK_SIZE_ERROR";

        case NPP_MOMENT_00_ZERO_ERROR:
            return "NPP_MOMENT_00_ZERO_ERROR";

        case NPP_THRESHOLD_NEGATIVE_LEVEL_ERROR:
            return "NPP_THRESHOLD_NEGATIVE_LEVEL_ERROR";

        case NPP_THRESHOLD_ERROR:
            return "NPP_THRESHOLD_ERROR";

        case NPP_CONTEXT_MATCH_ERROR:
            return "NPP_CONTEXT_MATCH_ERROR";

        case NPP_FFT_FLAG_ERROR:
            return "NPP_FFT_FLAG_ERROR";

        case NPP_FFT_ORDER_ERROR:
            return "NPP_FFT_ORDER_ERROR";

        case NPP_SCALE_RANGE_ERROR:
            return "NPP_SCALE_RANGE_ERROR";

        case NPP_DATA_TYPE_ERROR:
            return "NPP_DATA_TYPE_ERROR";

        case NPP_OUT_OFF_RANGE_ERROR:
            return "NPP_OUT_OFF_RANGE_ERROR";

        case NPP_DIVIDE_BY_ZERO_ERROR:
            return "NPP_DIVIDE_BY_ZERO_ERROR";

        case NPP_RANGE_ERROR:
            return "NPP_RANGE_ERROR";

        case NPP_NO_MEMORY_ERROR:
            return "NPP_NO_MEMORY_ERROR";

        case NPP_ERROR_RESERVED:
            return "NPP_ERROR_RESERVED";

        case NPP_NO_OPERATION_WARNING:
            return "NPP_NO_OPERATION_WARNING";

        case NPP_DIVIDE_BY_ZERO_WARNING:
            return "NPP_DIVIDE_BY_ZERO_WARNING";
#endif

#if ((NPP_VERSION_MAJOR << 12) + (NPP_VERSION_MINOR << 4)) >= 0x7000
        /* These are 7.0 or higher */
        case NPP_OVERFLOW_ERROR:
            return "NPP_OVERFLOW_ERROR";

        case NPP_CORRUPTED_DATA_ERROR:
            return "NPP_CORRUPTED_DATA_ERROR";
#endif
    }

    return "<unknown>";
}
#endif

#ifdef __DRIVER_TYPES_H__
#ifndef DEVICE_RESET
#define DEVICE_RESET cudaDeviceReset();
#endif
#else
#ifndef DEVICE_RESET
#define DEVICE_RESET
#endif
#endif

template< typename T >
void check(T result, char const *const func, const char *const file, int const line)
{
    if (result)
    {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n",
                file, line, static_cast<unsigned int>(result), _cudaGetErrorEnum(result), func);
        DEVICE_RESET
        // Make sure we call CUDA Device Reset before exiting
        exit(EXIT_FAILURE);
    }
}

#ifdef __DRIVER_TYPES_H__
// This will output the proper CUDA error strings in the event that a CUDA host call returns an error
#define checkCudaErrors(val)           check ( (val), #val, __FILE__, __LINE__ )

// This will output the proper error string when calling cudaGetLastError
#define getLastCudaError(msg)      __getLastCudaError (msg, __FILE__, __LINE__)

inline void __getLastCudaError(const char *errorMessage, const char *file, const int line)
{
    cudaError_t err = cudaGetLastError();

    if (cudaSuccess != err)
    {
        fprintf(stderr, "%s(%i) : getLastCudaError() CUDA error : %s : (%d) %s.\n",
                file, line, errorMessage, (int)err, cudaGetErrorString(err));
        DEVICE_RESET
        exit(EXIT_FAILURE);
    }
}

// This will only print the proper error string when calling cudaGetLastError but not exit program incase error detected.
#define printLastCudaError(msg)      __printLastCudaError (msg, __FILE__, __LINE__)

inline void __printLastCudaError(const char *errorMessage, const char *file, const int line)
{
    cudaError_t err = cudaGetLastError();

    if (cudaSuccess != err)
    {
        fprintf(stderr, "%s(%i) : getLastCudaError() CUDA error : %s : (%d) %s.\n",
            file, line, errorMessage, (int)err, cudaGetErrorString(err));
    }
}
#endif

#ifndef MAX
#define MAX(a,b) (a > b ? a : b)
#endif

// Float To Int conversion
inline int ftoi(float value)
{
    return (value >= 0 ? (int)(value + 0.5) : (int)(value - 0.5));
}

// Beginning of GPU Architecture definitions
inline int _ConvertSMVer2Cores(int major, int minor)
{
    // Defines for GPU Architecture types (using the SM version to determine the # of cores per SM
    typedef struct
    {
        int SM; // 0xMm (hexidecimal notation), M = SM Major version, and m = SM minor version
        int Cores;
    } sSMtoCores;

    sSMtoCores nGpuArchCoresPerSM[] =
            {
                    { 0x30, 192}, // Kepler Generation (SM 3.0) GK10x class
                    { 0x32, 192}, // Kepler Generation (SM 3.2) GK10x class
                    { 0x35, 192}, // Kepler Generation (SM 3.5) GK11x class
                    { 0x37, 192}, // Kepler Generation (SM 3.7) GK21x class
                    { 0x50, 128}, // Maxwell Generation (SM 5.0) GM10x class
                    { 0x52, 128}, // Maxwell Generation (SM 5.2) GM20x class
                    { 0x53, 128}, // Maxwell Generation (SM 5.3) GM20x class
                    { 0x60, 64 }, // Pascal Generation (SM 6.0) GP100 class
                    { 0x61, 128}, // Pascal Generation (SM 6.1) GP10x class
                    { 0x62, 128}, // Pascal Generation (SM 6.2) GP10x class
                    { 0x70, 64 }, // Volta Generation (SM 7.0) GV100 class

                    {   -1, -1 }
            };

    int index = 0;

    while (nGpuArchCoresPerSM[index].SM != -1)
    {
        if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor))
        {
            return nGpuArchCoresPerSM[index].Cores;
        }

        index++;
    }

    // If we don't find the values, we default use the previous one to run properly
    printf("MapSMtoCores for SM %d.%d is undefined.  Default to use %d Cores/SM\n", major, minor, nGpuArchCoresPerSM[index-1].Cores);
    return nGpuArchCoresPerSM[index-1].Cores;
}
// end of GPU Architecture definitions

#ifdef __CUDA_RUNTIME_H__
// General GPU Device CUDA Initialization
inline int gpuDeviceInit(int devID)
{
    int device_count;
    checkCudaErrors(cudaGetDeviceCount(&device_count));

    if (device_count == 0)
    {
        fprintf(stderr, "gpuDeviceInit() CUDA error: no devices supporting CUDA.\n");
        exit(EXIT_FAILURE);
    }

    if (devID < 0)
    {
        devID = 0;
    }

    if (devID > device_count-1)
    {
        fprintf(stderr, "\n");
        fprintf(stderr, ">> %d CUDA capable GPU device(s) detected. <<\n", device_count);
        fprintf(stderr, ">> gpuDeviceInit (-device=%d) is not a valid GPU device. <<\n", devID);
        fprintf(stderr, "\n");
        return -devID;
    }

    cudaDeviceProp deviceProp;
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, devID));

    if (deviceProp.computeMode == cudaComputeModeProhibited)
    {
        fprintf(stderr, "Error: device is running in <Compute Mode Prohibited>, no threads can use ::cudaSetDevice().\n");
        return -1;
    }

    if (deviceProp.major < 1)
    {
        fprintf(stderr, "gpuDeviceInit(): GPU device does not support CUDA.\n");
        exit(EXIT_FAILURE);
    }

    checkCudaErrors(cudaSetDevice(devID));
    printf("gpuDeviceInit() CUDA Device [%d]: \"%s\n", devID, deviceProp.name);

    return devID;
}

// This function returns the best GPU (with maximum GFLOPS)
inline int gpuGetMaxGflopsDeviceId()
{
    int current_device     = 0, sm_per_multiproc  = 0;
    int max_perf_device    = 0;
    int device_count       = 0, best_SM_arch      = 0;
    int devices_prohibited = 0;

    unsigned long long max_compute_perf = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceCount(&device_count);

    checkCudaErrors(cudaGetDeviceCount(&device_count));

    if (device_count == 0)
    {
        fprintf(stderr, "gpuGetMaxGflopsDeviceId() CUDA error: no devices supporting CUDA.\n");
        exit(EXIT_FAILURE);
    }

    // Find the best major SM Architecture GPU device
    while (current_device < device_count)
    {
        cudaGetDeviceProperties(&deviceProp, current_device);

        // If this GPU is not running on Compute Mode prohibited, then we can add it to the list
        if (deviceProp.computeMode != cudaComputeModeProhibited)
        {
            if (deviceProp.major > 0 && deviceProp.major < 9999)
            {
                best_SM_arch = MAX(best_SM_arch, deviceProp.major);
            }
        }
        else
        {
            devices_prohibited++;
        }

        current_device++;
    }

    if (devices_prohibited == device_count)
    {
    	fprintf(stderr, "gpuGetMaxGflopsDeviceId() CUDA error: all devices have compute mode prohibited.\n");
    	exit(EXIT_FAILURE);
    }

    // Find the best CUDA capable GPU device
    current_device = 0;

    while (current_device < device_count)
    {
        cudaGetDeviceProperties(&deviceProp, current_device);

        // If this GPU is not running on Compute Mode prohibited, then we can add it to the list
        if (deviceProp.computeMode != cudaComputeModeProhibited)
        {
            if (deviceProp.major == 9999 && deviceProp.minor == 9999)
            {
                sm_per_multiproc = 1;
            }
            else
            {
                sm_per_multiproc = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor);
            }

            unsigned long long compute_perf  = (unsigned long long) deviceProp.multiProcessorCount * sm_per_multiproc * deviceProp.clockRate;

            if (compute_perf  > max_compute_perf)
            {
                // If we find GPU with SM major > 2, search only these
                if (best_SM_arch > 2)
                {
                    // If our device==dest_SM_arch, choose this, or else pass
                    if (deviceProp.major == best_SM_arch)
                    {
                        max_compute_perf  = compute_perf;
                        max_perf_device   = current_device;
                    }
                }
                else
                {
                    max_compute_perf  = compute_perf;
                    max_perf_device   = current_device;
                }
            }
        }

        ++current_device;
    }

    return max_perf_device;
}


// Initialization code to find the best CUDA Device
inline int findCudaDevice(int argc, const char **argv)
{
    cudaDeviceProp deviceProp;
    int devID = 0;

    // If the command-line has a device number specified, use it
    if (checkCmdLineFlag(argc, argv, "device"))
    {
        devID = getCmdLineArgumentInt(argc, argv, "device=");

        if (devID < 0)
        {
            printf("Invalid command line parameter\n ");
            exit(EXIT_FAILURE);
        }
        else
        {
            devID = gpuDeviceInit(devID);

            if (devID < 0)
            {
                printf("exiting...\n");
                exit(EXIT_FAILURE);
            }
        }
    }
    else
    {
        // Otherwise pick the device with highest Gflops/s
        devID = gpuGetMaxGflopsDeviceId();
        checkCudaErrors(cudaSetDevice(devID));
        checkCudaErrors(cudaGetDeviceProperties(&deviceProp, devID));
        printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", devID, deviceProp.name, deviceProp.major, deviceProp.minor);
    }

    return devID;
}

// General check for CUDA GPU SM Capabilities
inline bool checkCudaCapabilities(int major_version, int minor_version)
{
    cudaDeviceProp deviceProp;
    deviceProp.major = 0;
    deviceProp.minor = 0;
    int dev;

    checkCudaErrors(cudaGetDevice(&dev));
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, dev));

    if ((deviceProp.major > major_version) ||
        (deviceProp.major == major_version && deviceProp.minor >= minor_version))
    {
        printf("  Device %d: <%16s >, Compute SM %d.%d detected\n", dev, deviceProp.name, deviceProp.major, deviceProp.minor);
        return true;
    }
    else
    {
        printf("  No GPU device was found that can support CUDA compute capability %d.%d.\n", major_version, minor_version);
        return false;
    }
}
#endif

// end of CUDA Helper Functions


#endif

/**
 * Copyright 1993-2013 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#ifndef HELPER_CUDA_GL_H
#define HELPER_CUDA_GL_H

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

// includes, graphics
#if defined (__APPLE__) || defined(MACOSX)
#include <OpenGL/gl.h>
#else
#include <GL/gl.h>
#endif

#ifndef EXIT_WAIVED
#define EXIT_WAIVED 2
#endif

#ifdef __DRIVER_TYPES_H__
#ifndef DEVICE_RESET
#define DEVICE_RESET cudaDeviceReset()
#endif
#else
#ifndef DEVICE_RESET
#define DEVICE_RESET
#endif
#endif

#ifdef __CUDA_GL_INTEROP_H__
////////////////////////////////////////////////////////////////////////////////
// These are CUDA OpenGL Helper functions

inline int gpuGLDeviceInit(int ARGC, const char **ARGV)
{
    int deviceCount;
    checkCudaErrors(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0)
    {
        fprintf(stderr, "CUDA error: no devices supporting CUDA.\n");
        exit(EXIT_FAILURE);
    }

    int dev = 0;
    dev = getCmdLineArgumentInt(ARGC, ARGV, "device=");

    if (dev < 0)
    {
        dev = 0;
    }

    if (dev > deviceCount-1)
    {
        fprintf(stderr, "\n");
        fprintf(stderr, ">> %d CUDA capable GPU device(s) detected. <<\n", deviceCount);
        fprintf(stderr, ">> gpuGLDeviceInit (-device=%d) is not a valid GPU device. <<\n", dev);
        fprintf(stderr, "\n");
        return -dev;
    }

    cudaDeviceProp deviceProp;
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, dev));

    if (deviceProp.computeMode == cudaComputeModeProhibited)
    {
        fprintf(stderr, "Error: device is running in <Compute Mode Prohibited>, no threads can use ::cudaSetDevice().\n");
        return -1;
    }

    if (deviceProp.major < 1)
    {
        fprintf(stderr, "Error: device does not support CUDA.\n");
        exit(EXIT_FAILURE);
    }

    if (checkCmdLineFlag(ARGC, ARGV, "quiet") == false)
    {
        fprintf(stderr, "Using device %d: %s\n", dev, deviceProp.name);
    }

    checkCudaErrors(cudaGLSetGLDevice(dev));
    return dev;
}

// This function will pick the best CUDA device available with OpenGL interop
inline int findCudaGLDevice(int argc, const char **argv)
{
    int devID = 0;

    // If the command-line has a device number specified, use it
    if (checkCmdLineFlag(argc, (const char **)argv, "device"))
    {
        devID = gpuGLDeviceInit(argc, (const char **)argv);

        if (devID < 0)
        {
            printf("no CUDA capable devices found, exiting...\n");
            DEVICE_RESET
            exit(EXIT_SUCCESS);
        }
    }
    else
    {
        // Otherwise pick the device with highest Gflops/s
        devID = gpuGetMaxGflopsDeviceId();
        cudaGLSetGLDevice(devID);
    }

    return devID;
}

static inline const char* glErrorToString(GLenum err)
{
#define CASE_RETURN_MACRO(arg) case arg: return #arg
    switch(err)
    {
        CASE_RETURN_MACRO(GL_NO_ERROR);
        CASE_RETURN_MACRO(GL_INVALID_ENUM);
        CASE_RETURN_MACRO(GL_INVALID_VALUE);
        CASE_RETURN_MACRO(GL_INVALID_OPERATION);
        CASE_RETURN_MACRO(GL_OUT_OF_MEMORY);
        CASE_RETURN_MACRO(GL_STACK_UNDERFLOW);
        CASE_RETURN_MACRO(GL_STACK_OVERFLOW);
#ifdef GL_INVALID_FRAMEBUFFER_OPERATION
        CASE_RETURN_MACRO(GL_INVALID_FRAMEBUFFER_OPERATION);
#endif
        default: break;
    }
#undef CASE_RETURN_MACRO
    return "*UNKNOWN*";
}
////////////////////////////////////////////////////////////////////////////
//! Check for OpenGL error
//! @return bool if no GL error has been encountered, otherwise 0
//! @param file  __FILE__ macro
//! @param line  __LINE__ macro
//! @note The GL error is listed on stderr
//! @note This function should be used via the CHECK_ERROR_GL() macro
////////////////////////////////////////////////////////////////////////////
inline bool
sdkCheckErrorGL(const char *file, const int line)
{
    bool ret_val = true;

    // check for error
    GLenum gl_error = glGetError();

    if (gl_error != GL_NO_ERROR)
    {
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
        char tmpStr[512];
        // NOTE: "%s(%i) : " allows Visual Studio to directly jump to the file at the right line
        // when the user double clicks on the error line in the Output pane. Like any compile error.
        sprintf_s(tmpStr, 255, "\n%s(%i) : GL Error : %s\n\n", file, line, glErrorToString(gl_error));
        fprintf(stderr, "%s", tmpStr);
#endif
        fprintf(stderr, "GL Error in file '%s' in line %d :\n", file, line);
        fprintf(stderr, "%s\n", glErrorToString(gl_error));
        ret_val = false;
    }

    return ret_val;
}

#define SDK_CHECK_ERROR_GL()                                              \
    if( false == sdkCheckErrorGL( __FILE__, __LINE__)) {                  \
        DEVICE_RESET                                                      \
        exit(EXIT_FAILURE);                                               \
    }
#endif

#endif

/**
 * Copyright 1993-2013 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

// These are helper functions for the SDK samples (string parsing, timers, image helpers, etc)
#ifndef HELPER_FUNCTIONS_H
#define HELPER_FUNCTIONS_H

#ifdef WIN32
#pragma warning(disable:4996)
#endif

// includes, project
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <assert.h>
#include "exception.h"
#include <math.h>

#include <fstream>
#include <vector>
#include <iostream>
#include <algorithm>

// includes, timer, string parsing, image helpers
#include "helper_timer.h"   // helper functions for timers
#include "helper_string.h"  // helper functions for string parsing
#include "helper_image.h"   // helper functions for image compare, dump, data comparisons

#ifndef EXIT_WAIVED
#define EXIT_WAIVED 2
#endif

#endif //  HELPER_FUNCTIONS_H

/**
 * Copyright 1993-2013 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

// Helper Timing Functions
#ifndef HELPER_TIMER_H
#define HELPER_TIMER_H

#ifndef EXIT_WAIVED
#define EXIT_WAIVED 2
#endif

// includes, system
#include <vector>

// includes, project
#include <exception.h>

// Definition of the StopWatch Interface, this is used if we don't want to use the CUT functions
// But rather in a self contained class interface
class StopWatchInterface
{
    public:
        StopWatchInterface() {};
        virtual ~StopWatchInterface() {};

    public:
        //! Start time measurement
        virtual void start() = 0;

        //! Stop time measurement
        virtual void stop() = 0;

        //! Reset time counters to zero
        virtual void reset() = 0;

        //! Time in msec. after start. If the stop watch is still running (i.e. there
        //! was no call to stop()) then the elapsed time is returned, otherwise the
        //! time between the last start() and stop call is returned
        virtual float getTime() = 0;

        //! Mean time to date based on the number of times the stopwatch has been
        //! _stopped_ (ie finished sessions) and the current total time
        virtual float getAverageTime() = 0;
};


//////////////////////////////////////////////////////////////////
// Begin Stopwatch timer class definitions for all OS platforms //
//////////////////////////////////////////////////////////////////
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
// includes, system
#define WINDOWS_LEAN_AND_MEAN
#include <windows.h>
#undef min
#undef max

//! Windows specific implementation of StopWatch
class StopWatchWin : public StopWatchInterface
{
    public:
        //! Constructor, default
        StopWatchWin() :
            start_time(),     end_time(),
            diff_time(0.0f),  total_time(0.0f),
            running(false), clock_sessions(0), freq(0), freq_set(false)
        {
            if (! freq_set)
            {
                // helper variable
                LARGE_INTEGER temp;

                // get the tick frequency from the OS
                QueryPerformanceFrequency((LARGE_INTEGER *) &temp);

                // convert to type in which it is needed
                freq = ((double) temp.QuadPart) / 1000.0;

                // rememeber query
                freq_set = true;
            }
        };

        // Destructor
        ~StopWatchWin() { };

    public:
        //! Start time measurement
        inline void start();

        //! Stop time measurement
        inline void stop();

        //! Reset time counters to zero
        inline void reset();

        //! Time in msec. after start. If the stop watch is still running (i.e. there
        //! was no call to stop()) then the elapsed time is returned, otherwise the
        //! time between the last start() and stop call is returned
        inline float getTime();

        //! Mean time to date based on the number of times the stopwatch has been
        //! _stopped_ (ie finished sessions) and the current total time
        inline float getAverageTime();

    private:
        // member variables

        //! Start of measurement
        LARGE_INTEGER  start_time;
        //! End of measurement
        LARGE_INTEGER  end_time;

        //! Time difference between the last start and stop
        float  diff_time;

        //! TOTAL time difference between starts and stops
        float  total_time;

        //! flag if the stop watch is running
        bool running;

        //! Number of times clock has been started
        //! and stopped to allow averaging
        int clock_sessions;

        //! tick frequency
        double  freq;

        //! flag if the frequency has been set
        bool  freq_set;
};

// functions, inlined

////////////////////////////////////////////////////////////////////////////////
//! Start time measurement
////////////////////////////////////////////////////////////////////////////////
inline void
StopWatchWin::start()
{
    QueryPerformanceCounter((LARGE_INTEGER *) &start_time);
    running = true;
}

////////////////////////////////////////////////////////////////////////////////
//! Stop time measurement and increment add to the current diff_time summation
//! variable. Also increment the number of times this clock has been run.
////////////////////////////////////////////////////////////////////////////////
inline void
StopWatchWin::stop()
{
    QueryPerformanceCounter((LARGE_INTEGER *) &end_time);
    diff_time = (float)
                (((double) end_time.QuadPart - (double) start_time.QuadPart) / freq);

    total_time += diff_time;
    clock_sessions++;
    running = false;
}

////////////////////////////////////////////////////////////////////////////////
//! Reset the timer to 0. Does not change the timer running state but does
//! recapture this point in time as the current start time if it is running.
////////////////////////////////////////////////////////////////////////////////
inline void
StopWatchWin::reset()
{
    diff_time = 0;
    total_time = 0;
    clock_sessions = 0;

    if (running)
    {
        QueryPerformanceCounter((LARGE_INTEGER *) &start_time);
    }
}


////////////////////////////////////////////////////////////////////////////////
//! Time in msec. after start. If the stop watch is still running (i.e. there
//! was no call to stop()) then the elapsed time is returned added to the
//! current diff_time sum, otherwise the current summed time difference alone
//! is returned.
////////////////////////////////////////////////////////////////////////////////
inline float
StopWatchWin::getTime()
{
    // Return the TOTAL time to date
    float retval = total_time;

    if (running)
    {
        LARGE_INTEGER temp;
        QueryPerformanceCounter((LARGE_INTEGER *) &temp);
        retval += (float)
                  (((double)(temp.QuadPart - start_time.QuadPart)) / freq);
    }

    return retval;
}

////////////////////////////////////////////////////////////////////////////////
//! Time in msec. for a single run based on the total number of COMPLETED runs
//! and the total time.
////////////////////////////////////////////////////////////////////////////////
inline float
StopWatchWin::getAverageTime()
{
    return (clock_sessions > 0) ? (total_time/clock_sessions) : 0.0f;
}
#else
// Declarations for Stopwatch on Linux and Mac OSX
// includes, system
#include <ctime>
#include <sys/time.h>

//! Windows specific implementation of StopWatch
class StopWatchLinux : public StopWatchInterface
{
    public:
        //! Constructor, default
        StopWatchLinux() :
            start_time(), diff_time(0.0), total_time(0.0),
            running(false), clock_sessions(0)
        { };

        // Destructor
        virtual ~StopWatchLinux()
        { };

    public:
        //! Start time measurement
        inline void start();

        //! Stop time measurement
        inline void stop();

        //! Reset time counters to zero
        inline void reset();

        //! Time in msec. after start. If the stop watch is still running (i.e. there
        //! was no call to stop()) then the elapsed time is returned, otherwise the
        //! time between the last start() and stop call is returned
        inline float getTime();

        //! Mean time to date based on the number of times the stopwatch has been
        //! _stopped_ (ie finished sessions) and the current total time
        inline float getAverageTime();

    private:

        // helper functions

        //! Get difference between start time and current time
        inline float getDiffTime();

    private:

        // member variables

        //! Start of measurement
        struct timeval  start_time;

        //! Time difference between the last start and stop
        float  diff_time;

        //! TOTAL time difference between starts and stops
        float  total_time;

        //! flag if the stop watch is running
        bool running;

        //! Number of times clock has been started
        //! and stopped to allow averaging
        int clock_sessions;
};

// functions, inlined

////////////////////////////////////////////////////////////////////////////////
//! Start time measurement
////////////////////////////////////////////////////////////////////////////////
inline void
StopWatchLinux::start()
{
    gettimeofday(&start_time, 0);
    running = true;
}

////////////////////////////////////////////////////////////////////////////////
//! Stop time measurement and increment add to the current diff_time summation
//! variable. Also increment the number of times this clock has been run.
////////////////////////////////////////////////////////////////////////////////
inline void
StopWatchLinux::stop()
{
    diff_time = getDiffTime();
    total_time += diff_time;
    running = false;
    clock_sessions++;
}

////////////////////////////////////////////////////////////////////////////////
//! Reset the timer to 0. Does not change the timer running state but does
//! recapture this point in time as the current start time if it is running.
////////////////////////////////////////////////////////////////////////////////
inline void
StopWatchLinux::reset()
{
    diff_time = 0;
    total_time = 0;
    clock_sessions = 0;

    if (running)
    {
        gettimeofday(&start_time, 0);
    }
}

////////////////////////////////////////////////////////////////////////////////
//! Time in msec. after start. If the stop watch is still running (i.e. there
//! was no call to stop()) then the elapsed time is returned added to the
//! current diff_time sum, otherwise the current summed time difference alone
//! is returned.
////////////////////////////////////////////////////////////////////////////////
inline float
StopWatchLinux::getTime()
{
    // Return the TOTAL time to date
    float retval = total_time;

    if (running)
    {
        retval += getDiffTime();
    }

    return retval;
}

////////////////////////////////////////////////////////////////////////////////
//! Time in msec. for a single run based on the total number of COMPLETED runs
//! and the total time.
////////////////////////////////////////////////////////////////////////////////
inline float
StopWatchLinux::getAverageTime()
{
    return (clock_sessions > 0) ? (total_time/clock_sessions) : 0.0f;
}
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
inline float
StopWatchLinux::getDiffTime()
{
    struct timeval t_time;
    gettimeofday(&t_time, 0);

    // time difference in milli-seconds
    return (float)(1000.0 * (t_time.tv_sec - start_time.tv_sec)
                   + (0.001 * (t_time.tv_usec - start_time.tv_usec)));
}
#endif // WIN32

////////////////////////////////////////////////////////////////////////////////
//! Timer functionality exported

////////////////////////////////////////////////////////////////////////////////
//! Create a new timer
//! @return true if a time has been created, otherwise false
//! @param  name of the new timer, 0 if the creation failed
////////////////////////////////////////////////////////////////////////////////
inline bool
sdkCreateTimer(StopWatchInterface **timer_interface)
{
    //printf("sdkCreateTimer called object %08x\n", (void *)*timer_interface);
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
    *timer_interface = (StopWatchInterface *)new StopWatchWin();
#else
    *timer_interface = (StopWatchInterface *)new StopWatchLinux();
#endif
    return (*timer_interface != NULL) ? true : false;
}


////////////////////////////////////////////////////////////////////////////////
//! Delete a timer
//! @return true if a time has been deleted, otherwise false
//! @param  name of the timer to delete
////////////////////////////////////////////////////////////////////////////////
inline bool
sdkDeleteTimer(StopWatchInterface **timer_interface)
{
    //printf("sdkDeleteTimer called object %08x\n", (void *)*timer_interface);
    if (*timer_interface)
    {
        delete *timer_interface;
        *timer_interface = NULL;
    }

    return true;
}

////////////////////////////////////////////////////////////////////////////////
//! Start the time with name \a name
//! @param name  name of the timer to start
////////////////////////////////////////////////////////////////////////////////
inline bool
sdkStartTimer(StopWatchInterface **timer_interface)
{
    //printf("sdkStartTimer called object %08x\n", (void *)*timer_interface);
    if (*timer_interface)
    {
        (*timer_interface)->start();
    }

    return true;
}

////////////////////////////////////////////////////////////////////////////////
//! Stop the time with name \a name. Does not reset.
//! @param name  name of the timer to stop
////////////////////////////////////////////////////////////////////////////////
inline bool
sdkStopTimer(StopWatchInterface **timer_interface)
{
    // printf("sdkStopTimer called object %08x\n", (void *)*timer_interface);
    if (*timer_interface)
    {
        (*timer_interface)->stop();
    }

    return true;
}

////////////////////////////////////////////////////////////////////////////////
//! Resets the timer's counter.
//! @param name  name of the timer to reset.
////////////////////////////////////////////////////////////////////////////////
inline bool
sdkResetTimer(StopWatchInterface **timer_interface)
{
    // printf("sdkResetTimer called object %08x\n", (void *)*timer_interface);
    if (*timer_interface)
    {
        (*timer_interface)->reset();
    }

    return true;
}

////////////////////////////////////////////////////////////////////////////////
//! Return the average time for timer execution as the total time
//! for the timer dividied by the number of completed (stopped) runs the timer
//! has made.
//! Excludes the current running time if the timer is currently running.
//! @param name  name of the timer to return the time of
////////////////////////////////////////////////////////////////////////////////
inline float
sdkGetAverageTimerValue(StopWatchInterface **timer_interface)
{
    //  printf("sdkGetAverageTimerValue called object %08x\n", (void *)*timer_interface);
    if (*timer_interface)
    {
        return (*timer_interface)->getAverageTime();
    }
    else
    {
        return 0.0f;
    }
}

////////////////////////////////////////////////////////////////////////////////
//! Total execution time for the timer over all runs since the last reset
//! or timer creation.
//! @param name  name of the timer to obtain the value of.
////////////////////////////////////////////////////////////////////////////////
inline float
sdkGetTimerValue(StopWatchInterface **timer_interface)
{
    // printf("sdkGetTimerValue called object %08x\n", (void *)*timer_interface);
    if (*timer_interface)
    {
        return (*timer_interface)->getTime();
    }
    else
    {
        return 0.0f;
    }
}

#endif // HELPER_TIMER_H

#define MPRES_PROJECT_CUDA_HEADERS_H

#endif //MPRES_PROJECT_CUDA_HEADERS_H
