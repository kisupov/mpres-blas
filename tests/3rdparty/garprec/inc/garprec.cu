#ifndef __GARPREC_CU__
#define __GARPREC_CU__

/**
*  the API file
*/

#define CUDA_DEVICE (1)

#include "helper_cuda.h"
#include "helper_cuda_gl.h"
#include "helper_functions.h"
#include "helper_timer.h"


#include "common.cu"
#include "gmp_small_inline.cu"
#include "garprec_init.cu"
#include "groun.cu"
#include "gnorm.cu"
#include "add.cu"
#include "sub.cu"
#include "mul.cu"
#include "geq.cu"

#endif /* __GARPREC_CU__ */

