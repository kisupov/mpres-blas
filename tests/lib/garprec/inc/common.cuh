#ifndef GARPREC_PROJECT_COMMON_H
#define GARPREC_PROJECT_COMMON_H

#include "gmp_array.h"

/** configure constants */
#define MAX_PREC_WORDS (145) //36: 500 digits 71: 1000 digits; 105: 1500 digits; 140: 2000 digits
#define MAX_N_WORDS (MAX_PREC_WORDS + 6)
#define ADD_D_SIZE (MAX_PREC_WORDS + 7)
#define MUL_D_SIZE (MAX_PREC_WORDS + 5 + FST_M) //FST_M = 3
#define DIV_D_SIZE (MAX_PREC_WORDS + 9)
#define MAX_D_SIZE DIV_D_SIZE
#define SIN_COS_TABLE_SIZE (129)



/** constant definition */
//#define FST_M (3)
#define ROUND_DIR 1 //round_to_nearest;
#define MPBBX (16777216.0)
#define MPBDX (MPBBX*MPBBX)
#define MPBX2 (MPBDX*MPBDX)
#define MPRDX (1.0 / MPBDX)
#define MPRX2 (MPRDX*MPRDX)
#define MPRXX (16.0 * MPRX2)
#define MPNPR (16)
#define MPNBT (48)

 __device__ __constant__ double _eps[2*MAX_N_WORDS];
 __device__ __constant__ double _eps2[2*MAX_N_WORDS];
 __device__ __constant__ double _pi[2*MAX_N_WORDS];
 __device__ __constant__ double _log2[2*MAX_N_WORDS];
 __device__ __constant__ double _log10[2*MAX_N_WORDS];
 __device__ __constant__ double _1[2*MAX_N_WORDS];
 __device__ __constant__ int gPrecWords[1];
gmp_array* gmp_sin_table = NULL;
gmp_array* gmp_cos_table = NULL;
double* d_sin_table = NULL;
double* d_cos_table = NULL;

#endif //MPRES_PROJECT_COMMON_H
