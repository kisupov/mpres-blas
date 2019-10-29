#ifndef __GARPREC_COMMON_CU__
#define __GARPREC_COMMON_CU__

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <iostream>
#include <arprec/mp_real.h>	//since nvcc can compile the arprec successfully, we merge this CPU library

#include "gmp_array.cu"


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

/** global constants */
///allocate for a higher precision
int prec_words = 0;
static __device__ __constant__ double _eps[2*MAX_N_WORDS];
static __device__ __constant__ double _eps2[2*MAX_N_WORDS];
static __device__ __constant__ double _pi[2*MAX_N_WORDS];
static __device__ __constant__ double _log2[2*MAX_N_WORDS];
static __device__ __constant__ double _log10[2*MAX_N_WORDS];
static __device__ __constant__ double _1[2*MAX_N_WORDS];
static __device__ __constant__ int gPrecWords[1];
gmp_array* gmp_sin_table = NULL;
gmp_array* gmp_cos_table = NULL;
double* d_sin_table = NULL;
double* d_cos_table = NULL;


/* macro utilities */
/*#define ERROR_EXIT exit(EXIT_FAILURE)
inline void errorExit(const char* func, const char* msg) {
	printf("!!!ERROR@%s: %s\n", func, msg);
	ERROR_EXIT;
}*/

/* mpr get and set functions */
__device__
double mpr_get( const double* mpr, const int interval, const int wordIdx ) {
	return mpr[interval*wordIdx];
}

__device__ 
void mpr_set( double* mpr, const int interval, const int wordIdx, const double value ) {
	mpr[interval*wordIdx] = value;
}

//+=
__device__ 
void mpr_compadd( double* mpr, const int interval, const int wordIdx, const double value ) {
	mpr[interval*wordIdx] += value;
}

//-=
__device__
void mpr_compsub( double* mpr, const int interval, const int wordIdx, const double value ) {
	mpr[interval*wordIdx] -= value;
}

//++
__device__
void mpr_inc( double* mpr, const int interval, const int wordIdx ) {
	mpr[interval*wordIdx]++;
}

//--
__device__
void mpr_dec( double* mpr, const int interval, const int wordIdx ) {
	mpr[interval*wordIdx]--;
}

//change the sign of one element
__device__
void mpr_sign( double* mpr, const int interval, const int wordIdx ) {
	mpr[interval*wordIdx] = -mpr[interval*wordIdx];
}


__device__
double sign(double a, double b) { return (b>=0 ? fabs(a) : -fabs(a)); }


__device__
int sign(int a, int b) { return (b>=0 ? std::abs(a) : -std::abs(a)); }


__device__
void zero(double* a, const int interval) { 
	//a.mpr[1] = a.mpr[2] = 0.; 
	mpr_set( a, interval, 1, 0.0 );
	mpr_set( a, interval, 2, 0.0 );
}



#endif /* __GARPREC_COMMON_CU__ */

