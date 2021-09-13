
#ifndef _GMP_EXP_CU_
#define _GMP_EXP_CU_

/**
* perform b = exp(a)
*/


#define EXP_TMP_SIZE (MAX_PREC_WORDS + 6)

__device__
void gmpexp( const double* a, const int interval_a, 
			 double* b, const int interval_b, 
			 int prec_words, 
			 double* d ) {
  const double* al2 = _log2;
  const double alt = 0.693147180559945309;
  const int nq = 8;
  double t1, t2;
  int n1, n2;
  
  /**
  if(error_no != 0) {
    if(error_no == 99) mpabrt();
    zero(b);
    return;
  }
  if(debug_level >= 7) cerr << "MPEXP I "<< a << endl;
  */
  
  //**mpmdc(a, t1, n1, prec_words);
  gmpmdc(a, interval_a, t1, n1, prec_words);
  if(n1 > 25) { // argument has very large magnitude
    if(t1 > 0.0) {
      /**if(MPKER[35] != 0) {
        cerr <<"*** MPEXP : Argument is too large : "<< a << endl;
        error_no = 35;
        if(MPKER[error_no] == 2) mpabrt();
      }
		*/
    } else {
      // large negative, just return 0.0
      zero(b, interval_b);
    }
    return;
  }
  if(n1 != 0)
    t1 = ldexp(t1, n1);
  
  // Unless the argument is near log(2), log(2) must be precomputed. This
  // Exception is necessary because MPLOG calls MPEXP to initialize Log(2).
  
  ///**if(abs(t1 - alt) > mprdx) {
  if( fabs(t1 - alt) > MPRDX ) {
    ///** mpmdc(al2, t2, n2, prec_words);
	gmpmdc(al2, 1, t2, n2, prec_words);
    ///**if(n2 != -mpnbt || (std::abs(t2 * mprdx - alt) > mprx2)) {
    if(n2 != -MPNBT || (fabs(t2 * MPRDX - alt) > MPRX2)) {
      /**if(MPKER[34] != 0) {
        cerr << "*** MPEXP: LOG (2) must be precomputed." << endl;
        error_no = 34;
        if(MPKER[error_no] == 2) mpabrt();
		*/
		return;
    }
  }
  
  // Check for overflows and underflows.
  // The constant is roughly (2^26 * mpnbt)* log(2),
  // Which allows a result with (word) exponent up to 2^26.
  if(fabs(t1) > 2325815993.0) {
    if(t1 > 0.0) {
      /**if(MPKER[35] != 0) {
        cerr << "*** MPEXP : Argument is too large : " << a << endl;
        error_no = 35;
        if(MPKER[error_no] == 2) mpabrt();
      }*/
      return;
    } else {
      // argument is very negative, the result would be
      // too small, so just return zero.
      zero(b, interval_b);
      return;
    }
  }
  
  int n6 = prec_words + 6;
  ///**mp_real sk0(0.0, n6), sk1(0.0, n6), sk2(0.0, n6), sk3(0.0, n6);
  ///!!!!!!!!! TO DO
  double sk0[EXP_TMP_SIZE];
  double sk1[EXP_TMP_SIZE];
  double sk2[EXP_TMP_SIZE];
  double sk3[EXP_TMP_SIZE];
  new_gmp( sk0, 0.0, n6 );
  new_gmp( sk1, 0.0, n6 );
  new_gmp( sk2, 0.0, n6 );
  new_gmp( sk3, 0.0, n6 );
  int nws = prec_words, nz, tl;
  prec_words++;
  ///**mp_real f(1.0, 8);
  double f[8];
  new_gmp( f, 1.0, 8 );
  
  //        Compute the reduced argument A' = A - Log(2) * nint(A / Log(2)).
  //        Save NZ = nint(A / Log(2)) for correcting the exponent of the final
  //         result.

  ///**if(std::abs(t1 - alt) > mprdx) {
  if(fabs(t1 - alt) > MPRDX) {
    //It is unnecessary to compute much of fractional part of the following
    //division.
    ///** prec_words = std::min(prec_words, int(a[2]) - int(al2[2]) + 3);
    prec_words = min(prec_words, int(mpr_get(a, interval_a, 2)) - int(al2[2]) + 3);
    prec_words = max(prec_words, 1);
    ///***mpdiv(a, al2, sk0, prec_words);
	gmpdiv( a, interval_a, al2, 1, sk0, 1, 
			 prec_words, d );
    prec_words = nws+1;
    ///**mpnint(sk0, sk1, prec_words);
	gmpnint( sk0, 1, sk1, 1, prec_words, d );
    ///**mpmdc(sk1, t1, n1, prec_words);
    gmpmdc(sk1, 1, t1, n1, prec_words);

    nz = int(ldexp(t1, n1) + sign(MPRXX, t1));
    ///**mpmul(sk1, al2, sk2, prec_words);
    gmpmul( sk1, 1, al2, 1, sk2, 1, prec_words, d );
	///**mpsub(a, sk2, sk0, prec_words);
	gmpsub( a, interval_a, sk2, 1, sk0, 1, prec_words, d );
  } else {
    ///**mpeq(a, sk0, prec_words);
	gmpeq(a, interval_a, sk0, 1, prec_words);
    nz = 0;
  }
  
  tl = int(sk0[2]) - prec_words;


  // Check if the reduced argument is zero

  if(sk0[1] == 0.0) {
    sk0[1] = 1.0;
    sk0[2] = 0.0;
    sk0[3] = 1.0;
    ///**mpmuld(sk0, 1.0, nz, sk1, prec_words);
	gmpmuld( sk0, 1, 1.0, nz, 
			 sk1, 1, prec_words, d );
	
	///**mpeq(sk1, b, prec_words);
    gmpeq(sk1, 1, b, interval_b, prec_words);
	
	prec_words = nws;
    gmproun(b, interval_b);
    return;
  }

  // Divide the reduced argument by 2^nq.

  ///**mpdivd(sk0, 1.0, nq, sk1, prec_words);
   gmpdivd(sk0, 1, 1.0, nq, sk1, 1, prec_words, d );
  
  // Compute Exp using the usual Taylor series.
  
  ///**mpeq(f, sk2, prec_words);
  gmpeq(f, 1, sk2, 1, prec_words);
  ///**mpeq(f, sk3, prec_words);
  gmpeq( f, 1, sk3, 1, prec_words );
  const int max_iteration_count = 10000;
  int l1 = 0; //iteration number.
  int not_there_yet = 1;
  int term_prec;
  int i;

  while(not_there_yet && l1 < max_iteration_count) {
    l1++;
    t2 = l1;
    //get term precision from exponents of components of term, subtracting
    // exponent of current sum
    term_prec = min(nws+1, nws+int(sk2[2]+sk1[2]-sk3[2])+2);
    term_prec = max(term_prec, 0);
    if(term_prec <= 0) {
      prec_words = nws+1;
      break;
    }
    prec_words = term_prec;
    ///**mpmul(sk2, sk1, sk0, prec_words);
    gmpmul(sk2, 1, sk1, 1, sk0, 1, prec_words, d);
    ///**mpdivd(sk0, t2, 0, sk2, prec_words);
    gmpdivd(sk0, 1, t2, 0, sk2, 1, prec_words, d );
	prec_words = nws+1; // full precision to add term in.
    ///**mpadd(sk3, sk2, sk3, prec_words);
    gmpadd(sk3, 1, sk2, 1, sk3, 1, prec_words, d);
    //the above line needs relies on mpadd being safe 
    //for use when the first and third arguments are the same object.

    // Check for convergence of the series.
    if(sk2[1] != 0.0 && sk2[2] > tl) {
      //keep going.
    } else {
      not_there_yet = 0;
    }
  }
  //check if exceeded iteration bound.
  if(l1 > max_iteration_count) {
    /**if(MPKER[36] != 0) {
      cerr <<"*** MPEXP: Iteration limit exceeded." << endl;
      error_no = 36;
      if(MPKER[error_no] == 2) mpabrt();
      prec_words = nws;
      return;
    }*/
	return;
  }
  //Result of taylor series stored in sk3.

  //         Raise to the (2^NQ)-th power.

  for(i=0;i<nq;i++) {
    ///**mpmul(sk3, sk3, sk3, prec_words);
    gmpmul(sk3, 1, sk3, 1, sk3, 1, prec_words, d);
  }
  
  // Multiply by 2^NZ.
  if(nz) {
    ///**mpmuld(sk3, 1.0, nz, b, prec_words);
	gmpmuld( sk3, 1, 1.0, nz, b, interval_b, prec_words, d ) ;
  } else {
    ///**mpeq(sk3, b, prec_words);
	gmpeq(sk3, 1, b, interval_b, prec_words);
  }
  //restore original precision level.
  prec_words = nws;
  gmproun(b, interval_b);
  
  return;
  /*
  */
  
}


__device__
void gmpexp( const double* a, const int interval_a, 
	     double* b, const int interval_b, 
	     int prec_words, 
	     double* d, const int interval_d, 
	     double* sk0, double* sk1, double* sk2, double* sk3, const int interval_sk ) {
  const double* al2 = _log2;
  const double alt = 0.693147180559945309;
  const int nq = 8;
  double t1, t2;
  int n1, n2;
  
  /**
  if(error_no != 0) {
    if(error_no == 99) mpabrt();
    zero(b);
    return;
  }
  if(debug_level >= 7) cerr << "MPEXP I "<< a << endl;
  */
  
  //**mpmdc(a, t1, n1, prec_words);
  gmpmdc(a, interval_a, t1, n1, prec_words);
  if(n1 > 25) { // argument has very large magnitude
    if(t1 > 0.0) {
      /**if(MPKER[35] != 0) {
        cerr <<"*** MPEXP : Argument is too large : "<< a << endl;
        error_no = 35;
        if(MPKER[error_no] == 2) mpabrt();
      }
		*/
    } else {
      // large negative, just return 0.0
      zero(b, interval_b);
    }
    return;
  }
  if(n1 != 0)
    t1 = ldexp(t1, n1);
  
  // Unless the argument is near log(2), log(2) must be precomputed. This
  // Exception is necessary because MPLOG calls MPEXP to initialize Log(2).
  
  ///**if(abs(t1 - alt) > mprdx) {
  if( fabs(t1 - alt) > MPRDX ) {
    ///** mpmdc(al2, t2, n2, prec_words);
	gmpmdc(al2, 1, t2, n2, prec_words);
    ///**if(n2 != -mpnbt || (std::abs(t2 * mprdx - alt) > mprx2)) {
    if(n2 != -MPNBT || (fabs(t2 * MPRDX - alt) > MPRX2)) {
      /**if(MPKER[34] != 0) {
        cerr << "*** MPEXP: LOG (2) must be precomputed." << endl;
        error_no = 34;
        if(MPKER[error_no] == 2) mpabrt();
		*/
		return;
    }
  }
  
  // Check for overflows and underflows.
  // The constant is roughly (2^26 * mpnbt)* log(2),
  // Which allows a result with (word) exponent up to 2^26.
  if(fabs(t1) > 2325815993.0) {
    if(t1 > 0.0) {
      /**if(MPKER[35] != 0) {
        cerr << "*** MPEXP : Argument is too large : " << a << endl;
        error_no = 35;
        if(MPKER[error_no] == 2) mpabrt();
      }*/
      return;
    } else {
      // argument is very negative, the result would be
      // too small, so just return zero.
      zero(b, interval_b);
      return;
    }
  }
  
  int n6 = prec_words + 6;
  ///**mp_real sk0(0.0, n6), sk1(0.0, n6), sk2(0.0, n6), sk3(0.0, n6);
  ///!!!!!!!!! TO DO
  //double sk0[EXP_TMP_SIZE];
  //double sk1[EXP_TMP_SIZE];
  //double sk2[EXP_TMP_SIZE];
  //double sk3[EXP_TMP_SIZE];
  //new_gmp( sk0, 0.0, n6 );
  //new_gmp( sk1, 0.0, n6 );
  //new_gmp( sk2, 0.0, n6 );
  //new_gmp( sk3, 0.0, n6 );
	new_gmp( sk0, interval_sk, 0.0, n6 );
  	new_gmp( sk1, interval_sk, 0.0, n6 );
	new_gmp( sk2, interval_sk, 0.0, n6 );
	new_gmp( sk3, interval_sk, 0.0, n6 );

  int nws = prec_words, nz, tl;
  prec_words++;
  ///**mp_real f(1.0, 8);
  double f[8];
  new_gmp( f, 1.0, 8 );
  
  //        Compute the reduced argument A' = A - Log(2) * nint(A / Log(2)).
  //        Save NZ = nint(A / Log(2)) for correcting the exponent of the final
  //         result.

  ///**if(std::abs(t1 - alt) > mprdx) {
  if(fabs(t1 - alt) > MPRDX) {
    //It is unnecessary to compute much of fractional part of the following
    //division.
    ///** prec_words = std::min(prec_words, int(a[2]) - int(al2[2]) + 3);
    prec_words = min(prec_words, int(mpr_get(a, interval_a, 2)) - int(al2[2]) + 3);
    prec_words = max(prec_words, 1);
    ///***mpdiv(a, al2, sk0, prec_words);
	gmpdiv( a, interval_a, al2, 1, sk0, interval_sk, 
			 prec_words, d, interval_d );
    prec_words = nws+1;
    ///**mpnint(sk0, sk1, prec_words);
	gmpnint( sk0, interval_sk, sk1, interval_sk, prec_words, d, interval_d );
    ///**mpmdc(sk1, t1, n1, prec_words);
    gmpmdc(sk1, interval_sk, t1, n1, prec_words);

    nz = int(ldexp(t1, n1) + sign(MPRXX, t1));
    ///**mpmul(sk1, al2, sk2, prec_words);
    gmpmul( sk1, interval_sk, al2, 1, sk2, interval_sk, prec_words, d, interval_d );
	///**mpsub(a, sk2, sk0, prec_words);
	gmpsub( a, interval_a, sk2, interval_sk, sk0, interval_sk, prec_words, d, interval_d );
  } else {
    ///**mpeq(a, sk0, prec_words);
	gmpeq(a, interval_a, sk0, interval_sk, prec_words);
    nz = 0;
  }
  
  tl = int(mpr_get(sk0, interval_sk, 2)) - prec_words;


  // Check if the reduced argument is zero

  if(mpr_get(sk0, interval_sk, 1) == 0.0) {
    mpr_set(sk0, interval_sk, 1, 1.0);
    mpr_set(sk0, interval_sk, 2, 0.0);
    mpr_set(sk0, interval_sk, 3, 1.0);
    ///**mpmuld(sk0, 1.0, nz, sk1, prec_words);
	gmpmuld( sk0, interval_sk, 1.0, nz, 
			 sk1, interval_sk, prec_words, d, interval_d );
	
	///**mpeq(sk1, b, prec_words);
    gmpeq(sk1, interval_sk, b, interval_b, prec_words);
	
	prec_words = nws;
    gmproun(b, interval_b);
    return;
  }

  // Divide the reduced argument by 2^nq.

  ///**mpdivd(sk0, 1.0, nq, sk1, prec_words);
   gmpdivd(sk0, interval_sk, 1.0, nq, sk1, interval_sk, prec_words, d, interval_d );
  
  // Compute Exp using the usual Taylor series.
  
  ///**mpeq(f, sk2, prec_words);
  gmpeq(f, 1, sk2, interval_sk, prec_words);
  ///**mpeq(f, sk3, prec_words);
  gmpeq( f, 1, sk3, interval_sk, prec_words );
  const int max_iteration_count = 10000;
  int l1 = 0; //iteration number.
  int not_there_yet = 1;
  int term_prec;
  int i;

  while(not_there_yet && l1 < max_iteration_count) {
    l1++;
    t2 = l1;
    //get term precision from exponents of components of term, subtracting
    // exponent of current sum
    term_prec = min(nws+1, nws+int(mpr_get(sk2, interval_sk, 2)+mpr_get(sk1, interval_sk, 2)-mpr_get(sk3, interval_sk, 2))+2);
    term_prec = max(term_prec, 0);
    if(term_prec <= 0) {
      prec_words = nws+1;
      break;
    }
    prec_words = term_prec;
    ///**mpmul(sk2, sk1, sk0, prec_words);
    gmpmul(sk2, interval_sk, sk1, interval_sk, sk0, interval_sk, prec_words, d, interval_d);
    ///**mpdivd(sk0, t2, 0, sk2, prec_words);
    gmpdivd(sk0, interval_sk, t2, 0, sk2, interval_sk, prec_words, d, interval_d );
	prec_words = nws+1; // full precision to add term in.
    ///**mpadd(sk3, sk2, sk3, prec_words);
    gmpadd(sk3, interval_sk, sk2, interval_sk, sk3, interval_sk, prec_words, d, interval_d);
    //the above line needs relies on mpadd being safe 
    //for use when the first and third arguments are the same object.

    // Check for convergence of the series.
    if(mpr_get(sk2, interval_sk, 1) != 0.0 && mpr_get(sk2, interval_sk, 2) > tl) {
      //keep going.
    } else {
      not_there_yet = 0;
    }
  }
  //check if exceeded iteration bound.
  if(l1 > max_iteration_count) {
    /**if(MPKER[36] != 0) {
      cerr <<"*** MPEXP: Iteration limit exceeded." << endl;
      error_no = 36;
      if(MPKER[error_no] == 2) mpabrt();
      prec_words = nws;
      return;
    }*/
	return;
  }
  //Result of taylor series stored in sk3.

  //         Raise to the (2^NQ)-th power.

  for(i=0;i<nq;i++) {
    ///**mpmul(sk3, sk3, sk3, prec_words);
    gmpmul(sk3, interval_sk, sk3, interval_sk, sk3, interval_sk, prec_words, d, interval_d);
  }
  
  // Multiply by 2^NZ.
  if(nz) {
    ///**mpmuld(sk3, 1.0, nz, b, prec_words);
	gmpmuld( sk3, interval_sk, 1.0, nz, b, interval_b, prec_words, d, interval_d ) ;
  } else {
    ///**mpeq(sk3, b, prec_words);
	gmpeq(sk3, interval_sk, b, interval_b, prec_words);
  }
  //restore original precision level.
  prec_words = nws;
  gmproun(b, interval_b);
  
  return;
  /*
  */
  
}



__device__
void gmpexp( const double* a, const int interval_a, 
			double* b, const int interval_b, 
			int prec_words, 
			double* d, const int interval_d, 
			double* sk0, const int interval_sk0,
			double* sk1, const int interval_sk1,
			double* sk2, const int interval_sk2,
			double* sk3, const int interval_sk3 ) 
{
	const double* al2 = _log2;
	const double alt = 0.693147180559945309;
	const int nq = 8;
	double t1, t2;
	int n1, n2;

	//**mpmdc(a, t1, n1, prec_words);
	gmpmdc(a, interval_a, t1, n1, prec_words);
	if(n1 > 25) { // argument has very large magnitude
		if(t1 > 0.0) {
		} else {
			// large negative, just return 0.0
			zero(b, interval_b);
		}
		return;
	}
	if(n1 != 0)
		t1 = ldexp(t1, n1);

	if( fabs(t1 - alt) > MPRDX ) {
		gmpmdc(al2, 1, t2, n2, prec_words);
		if(n2 != -MPNBT || (fabs(t2 * MPRDX - alt) > MPRX2)) {
			return;
		}
	}

	if(fabs(t1) > 2325815993.0) {
		if(t1 > 0.0) {

			return;
		} else {

			zero(b, interval_b);
			return;
		}
	}

	int n6 = prec_words + 6;
	new_gmp( sk0, interval_sk0, 0.0, n6 );
	new_gmp( sk1, interval_sk1, 0.0, n6 );
	new_gmp( sk2, interval_sk2, 0.0, n6 );
	new_gmp( sk3, interval_sk3, 0.0, n6 );

	int nws = prec_words, nz, tl;
	prec_words++;
	///**mp_real f(1.0, 8);
	double f[8];
	new_gmp( f, 1.0, 8 );

	if(fabs(t1 - alt) > MPRDX) {
		prec_words = min(prec_words, int(mpr_get(a, interval_a, 2)) - int(al2[2]) + 3);
		prec_words = max(prec_words, 1);
		gmpdiv( a, interval_a, al2, 1, sk0, interval_sk0, 
			prec_words, d, interval_d );
		prec_words = nws+1;
		gmpnint( sk0, interval_sk0, sk1, interval_sk1, prec_words, d, interval_d );
		gmpmdc(sk1, interval_sk1, t1, n1, prec_words);

		nz = int(ldexp(t1, n1) + sign(MPRXX, t1));
		gmpmul( sk1, interval_sk1, al2, 1, sk2, interval_sk2, prec_words, d, interval_d );
		gmpsub( a, interval_a, sk2, interval_sk2, sk0, interval_sk0, prec_words, d, interval_d );
	} else {
		gmpeq(a, interval_a, sk0, interval_sk0, prec_words);
		nz = 0;
	}

	tl = int(mpr_get(sk0, interval_sk0, 2)) - prec_words;


	if(mpr_get(sk0, interval_sk0, 1) == 0.0) {
		mpr_set(sk0, interval_sk0, 1, 1.0);
		mpr_set(sk0, interval_sk0, 2, 0.0);
		mpr_set(sk0, interval_sk0, 3, 1.0);
		gmpmuld( sk0, interval_sk0, 1.0, nz, 
			sk1, interval_sk1, prec_words, d, interval_d );

		gmpeq(sk1, interval_sk1, b, interval_b, prec_words);

		prec_words = nws;
		gmproun(b, interval_b);
		return;
	}

	gmpdivd(sk0, interval_sk0, 1.0, nq, sk1, interval_sk1, prec_words, d, interval_d );

	gmpeq(f, 1, sk2, interval_sk2, prec_words);
	gmpeq( f, 1, sk3, interval_sk3, prec_words );
	const int max_iteration_count = 10000;
	int l1 = 0; //iteration number.
	int not_there_yet = 1;
	int term_prec;
	int i;

	while(not_there_yet && l1 < max_iteration_count) {
		l1++;
		t2 = l1;
		term_prec = min(nws+1, nws+int(mpr_get(sk2, interval_sk2, 2)+mpr_get(sk1, interval_sk1, 2)-mpr_get(sk3, interval_sk3, 2))+2);
		term_prec = max(term_prec, 0);
		if(term_prec <= 0) {
			prec_words = nws+1;
			break;
		}
		prec_words = term_prec;
		gmpmul(sk2, interval_sk2, sk1, interval_sk1, sk0, interval_sk0, prec_words, d, interval_d);
		gmpdivd(sk0, interval_sk0, t2, 0, sk2, interval_sk2, prec_words, d, interval_d );
		prec_words = nws+1; // full precision to add term in.
		gmpadd(sk3, interval_sk3, sk2, interval_sk2, sk3, interval_sk3, prec_words, d, interval_d);

		if(mpr_get(sk2, interval_sk2, 1) != 0.0 && mpr_get(sk2, interval_sk2, 2) > tl) {
			//keep going.
		} else {
			not_there_yet = 0;
		}
	}
	if(l1 > max_iteration_count) {
		return;
	}

	for(i=0;i<nq;i++) {
		gmpmul(sk3, interval_sk3, sk3, interval_sk3, sk3, interval_sk3, prec_words, d, interval_d);
	}

	// Multiply by 2^NZ.
	if(nz) {
		gmpmuld( sk3, interval_sk3, 1.0, nz, b, interval_b, prec_words, d, interval_d ) ;
	} else {
		gmpeq(sk3, interval_sk3, b, interval_b, prec_words);
	}
	//restore original precision level.
	prec_words = nws;
	gmproun(b, interval_b);

	return;
}



#endif
