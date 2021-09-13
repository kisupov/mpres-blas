#ifndef GARPREC_SQRT_CUH
#define GARPREC_SQRT_CUH

#define SQRT_TMP_SIZE (MAX_PREC_WORDS + 7)
__device__
void gmpsqrt( const double* a, const int interval_a,
              double* b, const int interval_b,
              int prec_words,
              double* d, const int interval_d = 1 );

__device__
void gmpsqrt( const double* a, const int interval_a,
              double* b, const int interval_b,
              int prec_words,
              double* d, const int interval_d,
              double* sk0, const int interval_sk0,
              double* sk1, const int interval_sk1 );
#endif