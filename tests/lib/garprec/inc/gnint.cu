
#ifndef _GMP_NINT_CU_
#define _GMP_NINT_CU_

/**
* round to nearest integer
* b = gmpnint(a)
*/
__device__
void gmpnint( const double* a, const int interval_a, 
			  double* b, const int interval_b, 
			  const int prec_words, 
			  double* d, const int interval_d = 1 ) {
  ///**int na = int(std::abs(a[1]));
  ///**int nb = std::min(int(b[0])-5, std::min(prec_words, na));
  int na = int(fabs(mpr_get(a, interval_a, 1)));
  int nb = min(int(mpr_get(b, interval_b, 0))-5, min(prec_words, na));
  double mag_up = 0.0;
  ///**double exp = a[2];
  double exp = mpr_get(a, interval_a, 2);
  ///**double sgn = sign(1.0, a[1]);
  double sgn = sign(1.0, mpr_get(a, interval_a, 1));

  if(na == 0 || nb == 0) {
    zero(b, interval_b);
    return;
  }
  
  if(exp - na + 1 < 0) {
    // we must truncate to exp+1 words
    int nout = min(nb, int(exp)+1);
    if(nout < 0) {
      // magnitude far less than 1.
      zero(b, interval_b);
      return;
    }
    if(!nout) {
      // might round up to one.
      ///**if(a[FST_M] >= mpbdx/2.0) {
      if(mpr_get(a, interval_a, FST_M) >= MPBDX/2.0) {
        ///**b[1] = sgn * 1.0;
        mpr_set(b, interval_b, 1, sgn * 1.0);
        ///**b[2] = 0.0;
        mpr_set(b, interval_b, 2, 0.0);
        ///**b[3] = 1.0;
        mpr_set(b, interval_b, 3, 1.0);
        ///**b[4] = 0.0;
        mpr_set(b, interval_b, 4, 0.0);
        return;
      } else {
        zero(b, interval_b);
        return;
      }
    }
    if(na > nout) {
      // rounding required
      ///**if(a[FST_M + nout] >= mpbdx/2.0) {
      if(mpr_get(a, interval_a, FST_M + nout) >= MPBDX/2.0) {
        mag_up = 1.0;
      }
    }
    nb = nout;
  }
  ///**b[nb+FST_M] = b[nb + FST_M+1] = 0.0;
  mpr_set(b, interval_b, nb+FST_M, 0.0);
  mpr_set(b, interval_b, nb + FST_M+1, 0.0);
  ///**for(int i = nb+FST_M-1; i >= FST_M; i--) b[i] = a[i];
  for(int i = nb+FST_M-1; i >= FST_M; i--) { 
	///**b[i] = a[i];
	mpr_set(b, interval_b, i, mpr_get(a, interval_a, i));
  }
  ///**b[1] = sgn * nb;
  mpr_set(b, interval_b, 1, sgn * nb);
  ///**b[2] = a[2];
  mpr_set(b, interval_b, 2, mpr_get(a, interval_a, 2));
  if(mag_up == 1.0) {
    // add one (or subtract one if negative).
    ///***mp_real sk0,f;
	double sk0[MAX_PREC_WORDS + 5];
	double f[MAX_PREC_WORDS + 5];
	new_gmp( sk0, prec_words );
	new_gmp( f, prec_words );
    f[1] = 1;
    f[2] = 0;
    f[3] = 1;
    f[4] = 0;
    if(sgn > 0) {
      ///**mpadd(b, f, sk0, prec_words);
      gmpadd(b, interval_b, f, 1, sk0, 1, prec_words, d, interval_d);
	}
    else {
      ///**mpsub(b, f, sk0, prec_words);
      gmpsub(b, interval_b, f, 1, sk0, 1, prec_words, d, interval_d);
	}	
    ///**mpeq(sk0, b, prec_words);
	gmpeq(sk0, 1, b, interval_b, prec_words);
  }
  return;
}


#endif

