#ifndef _GMP_SMALL_INLINE_CU_
#define _GMP_SMALL_INLINE_CU_


#define _SPLITTER 134217729.0               /* = 2^27 + 1 */
#define TWO_TO_THE_52 4503599627370496.0

#define TWO_TO_THE_52 4503599627370496.0

__device__
double SLOPPY_ANINT_POSITIVE(double a) {
    // performs anint, with sometimes incorrect rounding.
    // Assumes that the input is in [0, 2^52).
    return (a + TWO_TO_THE_52) - TWO_TO_THE_52;
}

__device__
double SLOPPY_ANINT(double a) {
    // this one is correct most of the time.
    // performs anint, with possible rounding errors.
    // Assumes that the input is in (-2^52, 2^52).
    a = (a + TWO_TO_THE_52) - TWO_TO_THE_52; //for positives.
    a = (a - TWO_TO_THE_52) + TWO_TO_THE_52; //for negatives.
    return a;
}

__device__
void garprec_split(double a, double &hi, double &lo) {
    double temp;
    temp = _SPLITTER * a;
    hi = temp - (temp - a);
    lo = a - hi;
}

__device__  double mp_two_prod_positive(double a, double b, double &err) {
    double s1;

    double a_hi, a_lo, b_hi, b_lo;
    double p = a * b;
    garprec_split(a, a_hi, a_lo);
    garprec_split(b, b_hi, b_lo);
    err = ((a_hi * b_hi - p) + a_hi * b_lo + a_lo * b_hi) + a_lo * b_lo;
    s1 = p;

    double t = s1 * MPRDX; //mprdx;
    s1 = SLOPPY_ANINT_POSITIVE(t); // ok since inputs always positive
    err += MPBDX * (t - s1);

    return s1;
}

__device__
double AINT(double a) {
    // performs aint, correctly.
    // Assumes that the input is in (-2^52, 2^52).
    double b = a;
    if (a >= 0) {
        b = (b + TWO_TO_THE_52) - TWO_TO_THE_52;
        if (b > a) return b - 1.0; else return b;
    } else {
        b = (b - TWO_TO_THE_52) + TWO_TO_THE_52;
        if (b < a) return b + 1.0; else return b;
    }
}


__device__
double mp_two_prod(double a, double b, double &err) {
    double s1;

    double a_hi, a_lo, b_hi, b_lo;
    double p = a * b;
    garprec_split(a, a_hi, a_lo);
    garprec_split(b, b_hi, b_lo);
    err = ((a_hi * b_hi - p) + a_hi * b_lo + a_lo * b_hi) + a_lo * b_lo;
    s1 = p;

    double t = s1 * MPRDX;
    s1 = SLOPPY_ANINT(t);
    err += MPBDX * (t - s1);

    return s1;
}


__device__
double gpu_sqr(double t) {
    return t * t;
}


__device__
double FLOOR_POSITIVE(double a) {
    // performs floor, correctly.
    // Assumes that the input is in [0, 2^52).
    double b = (a + TWO_TO_THE_52) - TWO_TO_THE_52;
    if (b > a) return b - 1.0; else return b;
}


__device__
double d_sqr(double t) {
    return t * t;
}


__device__ double gaint(double a) { return a > 0 ? floor(a) : ceil(a); }

__device__ double ganint(double a) { return a > 0 ? ceil(a - 0.5) : floor(a + 0.5); }

#endif

