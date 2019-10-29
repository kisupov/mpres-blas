#ifndef _GMP_LOG_CU_
#define _GMP_LOG_CU_

/**
* b = log(a)
* @note this routine use the second strategy in the library (logx)
*/

#define LOG_TMP_SIZE (MAX_PREC_WORDS + 6)

__device__
void gmplog( const double* a, const int interval_a, 
			double* b, const int interval_b, 
			int prec_words, double* d ) {
  double t1, t2, tn;
  const double alt = 0.693147180559945309, cpi = 3.141592653589793;
  const int mzl = -5;
  int it2, n1;
  const double* pi = _pi;
  const double* al2 = _log2;
  
  /**
  if(error_no != 0) {
    if (error_no == 99) mpabrt();
    zero(b);
    return;
  }
  if (debug_level >= 6) cerr << "MPLOGX I" << endl;
  */

  ///**int ia = sign(1, int(a[1]));
  int ia = sign(1, int(mpr_get(a, interval_a, 1)));
  ///**int na = std::min(std::abs(int(a[1])), prec_words);
  int na = min(abs(int(mpr_get(a, interval_a, 1))), prec_words);
  ///**int ncr = 1 << (mpmcrx-2); //This version of log is faster at
			// smaller number of words.
  int n2;

  // Check if precision level is too low to justify the advanced routine.
	/**
  if (prec_words <= ncr) {
    mplog(a, al2, b, prec_words);
    return;
  }*/
  
  if(ia < 0 || na == 0) {
    //input is less than or equal to zero.
    /**if(MPKER[52] != 0) {
      cerr <<"*** MPLOGX: Argument is less than or equal to zero." << endl;
      error_no = 52;
      if(MPKER[error_no] == 2) mpabrt();
    }*/
    return;
  }

  // check if Pi has been precomputed.
  ///**mpmdc(pi, t1, n1, prec_words);
  gmpmdc(pi, 1, t1, n1, prec_words);
  ///**if(n1 != 0 || std::abs(t1 - cpi) > mprx2) {
  if(n1 != 0 || fabs(t1 - cpi) > MPRX2) {
    /**if(MPKER[53] != 0) {
      cerr << "*** MPLOGX: PI must be precomputed." << endl;
      error_no = 53;
      if(MPKER[error_no] == 2) mpabrt();      
    }*/
    return;
  }

  // Unless the input is 2, Log(2) must have been precomputed.
 
  ///**if(a[1] != 1.0 || a[2] != 0.0 || a[3] != 2.0) {
  if(mpr_get(a, interval_a, 1) != 1.0 
	|| mpr_get(a, interval_a, 2) != 0.0 
	|| mpr_get(a, interval_a, 3) != 2.0) {
    it2 = 0;
    ///**mpmdc(al2, t2, n2, prec_words);
    gmpmdc(al2, 1, t2, n2, prec_words);
    
	///**if(n2 != -mpnbt || std::abs(t2 * mprdx - alt) > mprx2) {
	if(n2 != -MPNBT || fabs(t2 * MPRDX - alt) > MPRX2) {
      /**
	  if(MPKER[54] != 0) {
		cerr << "*** MPLOGX: Log (2) must be precomputed." << endl;
		error_no = 54;
		if(MPKER[error_no] == 2) mpabrt();
      }
	  */
      return;
    }
  } else {
    it2 = 1;
  }
  
  int nws = prec_words;
  prec_words++;

  int n6 = prec_words + 6;
  ///**mp_real sk0(0.0, n6), sk1(0.0, n6), sk2(0.0, n6);
  ///**mp_real sk3(0.0, n6), f1(1.0, 8), f4(4.0, 8);
  double sk0[LOG_TMP_SIZE];
  double sk1[LOG_TMP_SIZE];
  double sk2[LOG_TMP_SIZE];
  double sk3[LOG_TMP_SIZE];
  double f1[8];
  double f4[8];
  new_gmp( sk0, 0.0, n6 );
  new_gmp( sk1, 0.0, n6 );
  new_gmp( sk2, 0.0, n6 );
  new_gmp( sk3, 0.0, n6 );
  new_gmp( f1, 1.0, 8 );
  new_gmp( f4, 4.0, 8 );

  // If argument is 1 the result is zero.  If the argement is
  //extremeley close to 1, employ  a Taylor's series instead.

  ///**mpsub(a, f1, sk0, prec_words);
  gmpsub( a, interval_a, f1, 1, sk0, 1, prec_words, d );
  if(sk0[1] == 0.0) {
    zero(b, interval_b);
    prec_words = nws;
    return;
  } else if(sk0[2] < mzl) {
    ///**mp_real sk4(0.0, n6);
	double sk4[LOG_TMP_SIZE];
	new_gmp( sk4, 0.0, n6 );
    ///**mpeq(sk0, sk1, prec_words);
	gmpeq( sk0, 1, sk1, 1, prec_words );
    ///**mpeq(sk1, sk2, prec_words);
	gmpeq( sk1, 1, sk2, 1, prec_words );
    int i1 = 1;
    int tl = int(sk0[2] - prec_words - 1);
    double st, is = 1.0;
    ///**if(debug_level >= 6) cerr <<"Using Taylor series in MPLOGX." << endl;
    do {
      i1++;
      is = -is;
      st  = is * i1;
      ///*mpmulx(sk1, sk2, sk3, prec_words);
	  gmpmul( sk1, 1, sk2, 1, sk3, 1, prec_words, d );
      ///**mpeq(sk3, sk2, prec_words);
	  gmpeq( sk3, 1, sk2, 1, prec_words );
      ///**mpdivd(sk3, st, 0, sk4, prec_words);
	  gmpdivd( sk3, 1, st, 0, sk4, 1, prec_words, d );
      ///**mpadd(sk0, sk4, sk3, prec_words);
	  gmpadd( sk0, 1, sk4, 1, sk3, 1, prec_words, d );
      ///**mpeq(sk3, sk0, prec_words);
	  gmpeq( sk3, 1, sk0, 1, prec_words );
    } while(sk2[2] >= tl);

    prec_words = nws;
    ///**mpeq(sk0, b, prec_words);
	gmpeq( sk0, 1, b, interval_b, prec_words );
    return;
  }
  
  // If input is exactly 2, set the exponent to a large value. Otherwise,
  // multiply the input by a large power of two.

  ///**mpmdc(a, t1, n1, prec_words);
  gmpmdc(a, interval_a, t1, n1, prec_words);
  n2 = MPNBT * (prec_words / 2 + 2) - n1;
  tn = n2;
  if(it2 == 1) {
    ///**mpdmc(1.0, n2, sk0, prec_words);
    gmpdmc(1.0, n2, sk0, 1, prec_words);
  }
  else {
    ///**mpmuld(a, 1.0, n2, sk0, prec_words);
    gmpmuld(a, interval_a, 1.0, n2, sk0, 1, prec_words, d);
   }

  // Perform AGM iterations.
  ///**mpeq(f1, sk1, prec_words);
  gmpeq(f1, 1, sk1, 1, prec_words);
  ///**mpdivx(f4, sk0, sk2, prec_words);
  gmpdiv(f4, 1, sk0, 1, sk2, 1, prec_words, d);
  ///**mpagmx(sk1, sk2, prec_words);
  gmpagm(sk1, 1, sk2, 1, prec_words, d);
  
  // Compute B = Pi / (2 * A), where A is the limit of the AGM iterations.

  ///**mpmuld(sk1, 2.0, 0, sk0, prec_words);
  gmpmuld(sk1, 1, 2.0, 0, sk0, 1, prec_words, d);
  ///**mpdivx(pi, sk0, sk1, prec_words);
  gmpdiv(pi, 1, sk0, 1, sk1, 1, prec_words, d);
  // If the input was exactly 2, divide by TN.  Otherwise,
  // subtract TN * Log(2).
  
  if(it2 == 1) {
    ///**mpdivd(sk1, tn, 0, b, prec_words);
    gmpdivd(sk1, 1, tn, 0, b, interval_b, prec_words, d);
  } else {
    ///**mpmuld(al2, tn, 0, sk2, prec_words);
    gmpmuld(al2, 1, tn, 0, sk2, 1, prec_words, d);
    ///**mpsub(sk1, sk2, b, prec_words);
    gmpsub(sk1, 1, sk2, 1, b, interval_b, prec_words, d);
  }
  prec_words = nws;

  /*
  if(debug_level >= 6) cerr << "MPLOGX 0" << endl;
  */
}


__device__
void gmplog( const double* a, const int interval_a, 
		double* b, const int interval_b, 
		int prec_words, double* d, const int interval_d,  
		double* sk0, double* sk1, double* sk2, double* sk3, double* sk4,
		const int interval_sk ) {
  double t1, t2, tn;
  const double alt = 0.693147180559945309, cpi = 3.141592653589793;
  const int mzl = -5;
  int it2, n1;
  const double* pi = _pi;
  const double* al2 = _log2;
  
  /**
  if(error_no != 0) {
    if (error_no == 99) mpabrt();
    zero(b);
    return;
  }
  if (debug_level >= 6) cerr << "MPLOGX I" << endl;
  */

  ///**int ia = sign(1, int(a[1]));
  int ia = sign(1, int(mpr_get(a, interval_a, 1)));
  ///**int na = std::min(std::abs(int(a[1])), prec_words);
  int na = min(abs(int(mpr_get(a, interval_a, 1))), prec_words);
  ///**int ncr = 1 << (mpmcrx-2); //This version of log is faster at
			// smaller number of words.
  int n2;

  // Check if precision level is too low to justify the advanced routine.
	/**
  if (prec_words <= ncr) {
    mplog(a, al2, b, prec_words);
    return;
  }*/
  
  if(ia < 0 || na == 0) {
    //input is less than or equal to zero.
    /**if(MPKER[52] != 0) {
      cerr <<"*** MPLOGX: Argument is less than or equal to zero." << endl;
      error_no = 52;
      if(MPKER[error_no] == 2) mpabrt();
    }*/
    return;
  }

  // check if Pi has been precomputed.
  ///**mpmdc(pi, t1, n1, prec_words);
  gmpmdc(pi, 1, t1, n1, prec_words);
  ///**if(n1 != 0 || std::abs(t1 - cpi) > mprx2) {
  if(n1 != 0 || fabs(t1 - cpi) > MPRX2) {
    /**if(MPKER[53] != 0) {
      cerr << "*** MPLOGX: PI must be precomputed." << endl;
      error_no = 53;
      if(MPKER[error_no] == 2) mpabrt();      
    }*/
    return;
  }

  // Unless the input is 2, Log(2) must have been precomputed.
 
  ///**if(a[1] != 1.0 || a[2] != 0.0 || a[3] != 2.0) {
  if(mpr_get(a, interval_a, 1) != 1.0 
	|| mpr_get(a, interval_a, 2) != 0.0 
	|| mpr_get(a, interval_a, 3) != 2.0) {
    it2 = 0;
    ///**mpmdc(al2, t2, n2, prec_words);
    gmpmdc(al2, 1, t2, n2, prec_words);
    
	///**if(n2 != -mpnbt || std::abs(t2 * mprdx - alt) > mprx2) {
	if(n2 != -MPNBT || fabs(t2 * MPRDX - alt) > MPRX2) {
      /**
	  if(MPKER[54] != 0) {
		cerr << "*** MPLOGX: Log (2) must be precomputed." << endl;
		error_no = 54;
		if(MPKER[error_no] == 2) mpabrt();
      }
	  */
      return;
    }
  } else {
    it2 = 1;
  }
  
  int nws = prec_words;
  prec_words++;

  int n6 = prec_words + 6;
  ///**mp_real sk0(0.0, n6), sk1(0.0, n6), sk2(0.0, n6);
  ///**mp_real sk3(0.0, n6), f1(1.0, 8), f4(4.0, 8);
  //double sk0[LOG_TMP_SIZE];
  //double sk1[LOG_TMP_SIZE];
  //double sk2[LOG_TMP_SIZE];
  //double sk3[LOG_TMP_SIZE];
  double f1[8];
  double f4[8];
  new_gmp( sk0, interval_sk, 0.0, n6 );
  new_gmp( sk1, interval_sk, 0.0, n6 );
  new_gmp( sk2, interval_sk, 0.0, n6 );
  new_gmp( sk3, interval_sk, 0.0, n6 );
  new_gmp( f1, 1.0, 8 );
  new_gmp( f4, 4.0, 8 );

  // If argument is 1 the result is zero.  If the argement is
  //extremeley close to 1, employ  a Taylor's series instead.

  ///**mpsub(a, f1, sk0, prec_words);
  gmpsub( a, interval_a, f1, 1, sk0, interval_sk, prec_words, d, interval_d );
  if(mpr_get(sk0, interval_sk, 1) == 0.0) {
    zero(b, interval_b);
    prec_words = nws;
    return;
  } else if(mpr_get(sk0, interval_sk, 2) < mzl) {
    ///**mp_real sk4(0.0, n6);
	//double sk4[LOG_TMP_SIZE];
	new_gmp( sk4, interval_sk, 0.0, n6 );
    ///**mpeq(sk0, sk1, prec_words);
	gmpeq( sk0, interval_sk, sk1, interval_sk, prec_words );
    ///**mpeq(sk1, sk2, prec_words);
	gmpeq( sk1, interval_sk, sk2, interval_sk, prec_words );
    int i1 = 1;
    int tl = int(mpr_get(sk0, interval_sk, 2) - prec_words - 1);
    double st, is = 1.0;
    ///**if(debug_level >= 6) cerr <<"Using Taylor series in MPLOGX." << endl;
    do {
      i1++;
      is = -is;
      st  = is * i1;
      ///*mpmulx(sk1, sk2, sk3, prec_words);
	  gmpmul( sk1, interval_sk, sk2, interval_sk, sk3, interval_sk, prec_words, d, interval_d );
      ///**mpeq(sk3, sk2, prec_words);
	  gmpeq( sk3, interval_sk, sk2, interval_sk, prec_words );
      ///**mpdivd(sk3, st, 0, sk4, prec_words);
	  gmpdivd( sk3, interval_sk, st, 0, sk4, interval_sk, prec_words, d, interval_d );
      ///**mpadd(sk0, sk4, sk3, prec_words);
	  gmpadd( sk0, interval_sk, sk4, interval_sk, sk3, interval_sk, prec_words, d, interval_d );
      ///**mpeq(sk3, sk0, prec_words);
	  gmpeq( sk3, interval_sk, sk0, interval_sk, prec_words );
    } while(mpr_get(sk2, interval_sk, 2) >= tl);

    prec_words = nws;
    ///**mpeq(sk0, b, prec_words);
	gmpeq( sk0, interval_sk, b, interval_b, prec_words );
    return;
  }
  
  // If input is exactly 2, set the exponent to a large value. Otherwise,
  // multiply the input by a large power of two.

  ///**mpmdc(a, t1, n1, prec_words);
  gmpmdc(a, interval_a, t1, n1, prec_words);
  n2 = MPNBT * (prec_words / 2 + 2) - n1;
  tn = n2;
  if(it2 == 1) {
    ///**mpdmc(1.0, n2, sk0, prec_words);
    gmpdmc(1.0, n2, sk0, interval_sk, prec_words);
  }
  else {
    ///**mpmuld(a, 1.0, n2, sk0, prec_words);
    gmpmuld(a, interval_a, 1.0, n2, sk0, interval_sk, prec_words, d, interval_d);
   }

  // Perform AGM iterations.
  ///**mpeq(f1, sk1, prec_words);
  gmpeq(f1, 1, sk1, interval_sk, prec_words);
  ///**mpdivx(f4, sk0, sk2, prec_words);
  gmpdiv(f4, 1, sk0, interval_sk, sk2, interval_sk, prec_words, d, interval_d);
  ///**mpagmx(sk1, sk2, prec_words);
  gmpagm(sk1, interval_sk, sk2, interval_sk, prec_words, d, interval_d);
  
  // Compute B = Pi / (2 * A), where A is the limit of the AGM iterations.

  ///**mpmuld(sk1, 2.0, 0, sk0, prec_words);
  gmpmuld(sk1, interval_sk, 2.0, 0, sk0, interval_sk, prec_words, d, interval_d);
  ///**mpdivx(pi, sk0, sk1, prec_words);
  gmpdiv(pi, 1, sk0, interval_sk, sk1, interval_sk, prec_words, d, interval_d);
  // If the input was exactly 2, divide by TN.  Otherwise,
  // subtract TN * Log(2).
  
  if(it2 == 1) {
    ///**mpdivd(sk1, tn, 0, b, prec_words);
    gmpdivd(sk1, interval_sk, tn, 0, b, interval_b, prec_words, d, interval_d);
  } else {
    ///**mpmuld(al2, tn, 0, sk2, prec_words);
    gmpmuld(al2, 1, tn, 0, sk2, interval_sk, prec_words, d, interval_d);
    ///**mpsub(sk1, sk2, b, prec_words);
    gmpsub(sk1, interval_sk, sk2, interval_sk, b, interval_b, prec_words, d, interval_d);
  }
  prec_words = nws;

  /*
  if(debug_level >= 6) cerr << "MPLOGX 0" << endl;
  */
}


#endif

