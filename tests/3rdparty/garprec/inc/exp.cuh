#ifndef GARPREC_EXP_H
#define GARPREC_EXP_H
    #define EXP_TMP_SIZE (MAX_PREC_WORDS + 6)

__device__
void gmpexp( const double* a, const int interval_a,
             double* b, const int interval_b,
             int prec_words,
             double* d );

__device__
void gmpexp( const double* a, const int interval_a,
             double* b, const int interval_b,
             int prec_words,
             double* d, const int interval_d,
             double* sk0, double* sk1, double* sk2, double* sk3, const int interval_sk );


__device__
void gmpexp( const double* a, const int interval_a,
             double* b, const int interval_b,
             int prec_words,
             double* d, const int interval_d,
             double* sk0, const int interval_sk0,
             double* sk1, const int interval_sk1,
             double* sk2, const int interval_sk2,
             double* sk3, const int interval_sk3 );
#endif