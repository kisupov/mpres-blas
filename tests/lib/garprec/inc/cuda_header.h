

#ifndef GARPREC_PROJECT_CUDA_HEADER_H
#define GARPREC_PROJECT_CUDA_HEADER_H
#include <stdio.h>
#include <stdlib.h>
#include "cutil_inline.h"
#include "helper_cuda.h"
#include "helper_cuda_gl.h"
#include "helper_functions.h"
#include "helper_timer.h"

/** compute capability */
#define MAX_SHARED_MEM_SIZE (16000) //byte


/* kernel macros */
#define NUM_TOTAL_THREAD (gridDim.x*blockDim.x)
#define GLOBAL_THREAD_OFFSET (blockDim.x*blockIdx.x + threadIdx.x)

/** macro utility */
#define GPUMALLOC(D_DATA, MEM_SIZE) cutilSafeCall(cudaMalloc(D_DATA, MEM_SIZE))
#define TOGPU(D_DATA, H_DATA, MEM_SIZE) cutilSafeCall(cudaMemcpy(D_DATA, H_DATA, MEM_SIZE, cudaMemcpyHostToDevice))
#define FROMGPU(H_DATA, D_DATA, MEM_SIZE) cutilSafeCall(cudaMemcpy( H_DATA, D_DATA, MEM_SIZE, cudaMemcpyDeviceToHost))
#define GPUTOGPU(DEST, SRC, MEM_SIZE) cutilSafeCall(cudaMemcpy( DEST, SRC, MEM_SIZE, cudaMemcpyDeviceToDevice ))
#define GPUFREE(MEM) cutilSafeCall(cudaFree(MEM));


void startTimer(StopWatchInterface *timer);

float endTimer(StopWatchInterface *timer, char *title);

#endif