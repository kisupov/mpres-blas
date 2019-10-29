#ifndef _GMP_SQRT_CU_
#define _GMP_SQRT_CU_

/**
* perform square root
* b = sqrt(a)
*/

#define SQRT_TMP_SIZE (MAX_PREC_WORDS + 7)

/**
* @p a, @p b are both INTERVAL access
* @p d should be allocated with the max possible buffer size, it is SEQUENTIAL access
*/
__device__
void gmpsqrt( const double* a, const int interval_a, 
			 double* b, const int interval_b, 
			 int prec_words, 
			 double* d, const int interval_d = 1 ) {
  const double cl2 = 1.4426950408889633; // == log_2(e) ==  1/log(2)
  const int nit = 3;

  /**
  if(error_no != 0) {
    if(error_no == 00) mpabrt();
    zero(b);
    return; 
  }
  if(debug_level >= 7 ) {
    cerr << "Runnung mpsqrt" << endl;
  }*/
  
  //**int ia = sign(1, int(a[1]));
  int ia = sign(1, int(mpr_get(a, interval_a, 1)));
  //**int na = std::min(int(std::abs(a[1])), prec_words);
  int na = min(int(fabs(mpr_get(a, interval_a, 1))), prec_words);
  
  if(na == 0) {
    zero(b, interval_b);
    return;
  }
  if(ia < 0.0) {//negative radicand!
    /**if(MPKER[70] != 0) {
      cerr << "*** MPSQRT: Argument is negative." << endl;
      error_no = 70;
      if(MPKER[error_no] == 2)
        mpabrt();
    }*/
    return;
  } // end negative radicand check

  int nws = prec_words;
  int k, mq, n, n2, iq=0;
  double t1, t2;
  int nw1, nw2, prec_change=0;
  int n7 = prec_words+7; 
  //**mp_real sk0(0.0, n7), sk1(0.0, n7);
  double sk0[SQRT_TMP_SIZE]; //TO DO !!!
  double sk1[SQRT_TMP_SIZE];
  new_gmp( sk0, 0.0, n7 );
  new_gmp( sk1, 0.0, n7 );  

  // Determine the least integer MQ such that 2 ^ MQ >= prec_words.

  t1 = prec_words;
  mq = int(cl2 * log(t1) + 1.0 - MPRXX);
  
  //        Compute the initial approximation of Sqrt(A) using double
  //        precision.

  gmpmdc(a, interval_a, t1, n, prec_words);

  n2 = n / 2;
  t2 = sqrt((n2*2 == n) ? t1 : t1 * 2.0);
  t1 = t2;
  gmpdmc(t1, n2, b, interval_b, prec_words);

  nw1 = nw2 = prec_words = 3;
  iq = 0;
  
  //         Perform the Newton-Raphson iteration described above by
  //         changing the precision level MPNW (one greater than powers of two).
  for(k=2;k <= mq;k++) {
    if(prec_change) {
      nw1 = prec_words;
      prec_words = min(2*prec_words-2, nws)+1; 
      nw2 = prec_words;
    } else {
      k--;
      prec_change = 1;
    }
    gmpmul(b, interval_b, b, interval_b, sk0, 1, prec_words, d, interval_d);
    gmpsub(a, interval_a, sk0, 1, sk1, 1, prec_words, d, interval_d);
    prec_words = nw1;
    gmpdiv(sk1, 1, b, interval_b, sk0, 1, prec_words, d, interval_d);
    gmpmuld(sk0, 1, 0.5, 0, sk1, 1, prec_words, d, interval_d);
    prec_words = nw2;
    gmpadd(b, interval_b, sk1, 1, b, interval_b, prec_words, d, interval_d);
    //the above line needs to change if mpadd is not safe for 
    // same variable input/output.

    if(k == mq - nit && iq == 0) {
      iq = 1;
      prec_change = 0;
    }
  } // end for

  // Restore original precision level
  prec_words = nws;
  gmproun(b, interval_b); 
/*
  */
}

__device__
void gmpsqrt( const double* a, const int interval_a, 
		double* b, const int interval_b, 
		int prec_words, 
		double* d, const int interval_d, 
		double* sk0, const int interval_sk0,
		double* sk1, const int interval_sk1 ) {
  const double cl2 = 1.4426950408889633; // == log_2(e) ==  1/log(2)
  const int nit = 3;

  /**
  if(error_no != 0) {
    if(error_no == 00) mpabrt();
    zero(b);
    return; 
  }
  if(debug_level >= 7 ) {
    cerr << "Runnung mpsqrt" << endl;
  }*/
  
  //**int ia = sign(1, int(a[1]));
  int ia = sign(1, int(mpr_get(a, interval_a, 1)));
  //**int na = std::min(int(std::abs(a[1])), prec_words);
  int na = min(int(fabs(mpr_get(a, interval_a, 1))), prec_words);
  
  if(na == 0) {
    zero(b, interval_b);
    return;
  }
  if(ia < 0.0) {//negative radicand!
    /**if(MPKER[70] != 0) {
      cerr << "*** MPSQRT: Argument is negative." << endl;
      error_no = 70;
      if(MPKER[error_no] == 2)
        mpabrt();
    }*/
    return;
  } // end negative radicand check

  int nws = prec_words;
  int k, mq, n, n2, iq=0;
  double t1, t2;
  int nw1, nw2, prec_change=0;
  int n7 = prec_words+7; 
  //**mp_real sk0(0.0, n7), sk1(0.0, n7);
  //double sk0[SQRT_TMP_SIZE]; //TO DO !!!
  //double sk1[SQRT_TMP_SIZE];
  new_gmp( sk0, interval_sk0, 0.0, n7 );
  new_gmp( sk1, interval_sk1, 0.0, n7 );  

  // Determine the least integer MQ such that 2 ^ MQ >= prec_words.

  t1 = prec_words;
  mq = int(cl2 * log(t1) + 1.0 - MPRXX);
  
  //        Compute the initial approximation of Sqrt(A) using double
  //        precision.

  gmpmdc(a, interval_a, t1, n, prec_words);

  n2 = n / 2;
  t2 = sqrt((n2*2 == n) ? t1 : t1 * 2.0);
  t1 = t2;
  gmpdmc(t1, n2, b, interval_b, prec_words);

  nw1 = nw2 = prec_words = 3;
  iq = 0;
  
  //         Perform the Newton-Raphson iteration described above by
  //         changing the precision level MPNW (one greater than powers of two).
  for(k=2;k <= mq;k++) {
    if(prec_change) {
      nw1 = prec_words;
      prec_words = min(2*prec_words-2, nws)+1; 
      nw2 = prec_words;
    } else {
      k--;
      prec_change = 1;
    }
    gmpmul(b, interval_b, b, interval_b, sk0, interval_sk0, prec_words, d, interval_d);
    gmpsub(a, interval_a, sk0, interval_sk0, sk1, interval_sk1, prec_words, d, interval_d);
    prec_words = nw1;
    gmpdiv(sk1, interval_sk1, b, interval_b, sk0, interval_sk0, prec_words, d, interval_d);
    gmpmuld(sk0, interval_sk0, 0.5, 0, sk1, interval_sk1, prec_words, d, interval_d);
    prec_words = nw2;
    gmpadd(b, interval_b, sk1, interval_sk1, b, interval_b, prec_words, d, interval_d);
    //the above line needs to change if mpadd is not safe for 
    // same variable input/output.

    if(k == mq - nit && iq == 0) {
      iq = 1;
      prec_change = 0;
    }
  } // end for

  // Restore original precision level
  prec_words = nws;
  gmproun(b, interval_b); 
/*
  */
}


#endif
