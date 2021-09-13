//
// Created by matroskinb on 26.11.17.
//

#ifndef GARPREC_PROJECT_GARPREC_H
#define GARPREC_PROJECT_GARPREC_H

#include "common.cuh"
#include "cuda_header.h"
#include "garprec_init.h"
#include "exp.cuh"
#include "sqrt.cuh"
/**
* perform the addition c = a + b
*/

/**
* @p d is a temperoral buffer, should be allocated outside with the size (prec_words+7)
* @p a, @p b and @p c are all INTERVAL access, but @p d is SEQUENTIAL access
*/
__device__
void gmpadd( const double* a, const int interval_a,
             const double* b, const int interval_b,
             double* c, const int interval_c,
             const int prec_words,
             double* d, const int interval_d = 1 );


/**
* @p d is a temperoral buffer, should be allocated outside with the size (prec_words+7)
* @p a, @p b and @p c are all INTERVAL access, but @p d is SEQUENTIAL access
*/
__device__
void gmpmul( const double* a, const int interval_a,
             const double* b, const int interval_b,
             double* c, const int interval_c,
             const int prec_words,
             double* d, const int interval_d = 1 );


/**
* @p d is a temperoral buffer, should be allocated outside with the size (prec_words+9)
* @p a, @p b and @p c are all INTERVAL access, but @p d is SEQUENTIAL access
*/
__device__
void gmpdiv( const double* a, const int interval_a,
             const double* b, const int interval_b,
             double* c, const int interval_c,
             const int prec_words,
             double* d );

__device__
void gmpdiv( const double* a, const int interval_a,
             const double* b, const int interval_b,
             double* c, const int interval_c,
             const int prec_words,
             double* d, const int interval_d );

__device__
void gmplog( const double* a, const int interval_a,
             double* b, const int interval_b,
             int prec_words, double* d );
__device__
void gmplog( const double* a, const int interval_a,
             double* b, const int interval_b,
             int prec_words, double* d, const int interval_d,
             double* sk0, double* sk1, double* sk2, double* sk3, double* sk4,
             const int interval_sk );

__device__
void gmpsin( const double* a, const int interval_a,
             double* c, const int interval_c,
             const int prec_words,
             double* d, const int interval_d,
             const double* sin_table,
             const double* cos_table,
             double* sk0, double* sk1, double* sk2, double* sk3,
             double* sk4, double* sk5, double* sk6, const int interval_sk );

__device__
void gmpcos( const double* a, const int interval_a,
             double* c, const int interval_c,
             const int prec_words,
             double* d, const int interval_d,
             const double* sin_table,
             const double* cos_table,
             double* sk0, double* sk1, double* sk2, double* sk3,
             double* sk4, double* sk5, double* sk6, const int interval_sk );


/**
* the most complicated function in the library!!!!
* calculate the sin and cos of an input number
* x = cos
* y = sin
*/


__device__
void gmpcssn( const double* a, const int interval_a,
              double* x, const int interval_x,
              double* y, const int interval_y,
              int prec_words,
              double* d, const int interval_d,
              const double* d_pi_over_256_sine_table,
              const double* d_pi_over_256_cosine_table,
              double* sk0, double* sk1, double* sk2,
              double* sk3, double* sk4, double* sk5,
              double* sk6, const int interval_sk );
__global__
void gmp_init_kernel( double* d_mpr, const int numElement, const int n_words );
#endif //MPRES_PROJECT_GARPREC_H
