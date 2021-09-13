#ifndef _GMP_AGM_CU_
#define _GMP_AGM_CU_

#define LOG_TMP_SIZE (MAX_PREC_WORDS + 6)

  /**
   * This performs the arithmetic-geometric mean (AGM) iterations.
   * this routine is called by MPLOGX.  It is not intended to be 
   * called directly by the user.
   */
__device__
void gmpagm(double* a, const int interval_a, 
			double* b, const int interval_b, 
			const int prec_words,
			double* d )
{
	/**
  if(error_no != 0) {
    if(error_no == 99) mpabrt();
    zero(a); zero(b);
    return;
  }
  */
  
  int n6 = prec_words + 6;
  ///**mp_real sk0(0.0, n6), sk1(0.0, n6);
  double sk0[LOG_TMP_SIZE];
  double sk1[LOG_TMP_SIZE];
  new_gmp( sk0, 0.0, n6 );
  new_gmp( sk1, 0.0, n6 );
  int l1 = 0;
  double s1;
  sk0[2] = 10000000.0;// high value to force second iteration.

  do {
    l1++;
    if (l1 == 50) {
      /**
	  if(MPKER[5] != 0) {
		cerr <<"*** MPAGMX: Iteration limit exceeded." << endl;
		error_no = 5;
		if(MPKER[error_no] == 2) mpabrt();
      }
	  */
      break;
    }
    
    s1 = sk0[2];
    ///**mpadd(a, b, sk0, prec_words);
    gmpadd(a, interval_a, b, interval_b, sk0, 1, prec_words, d);
    ///**mpmuld(sk0, 0.5, 0, sk1, prec_words);
    gmpmuld(sk0, 1, 0.5, 0, sk1, 1, prec_words, d);
    ///**mpmulx(a, b, sk0, prec_words);
    gmpmul(a, interval_a, b, interval_b, sk0, 1, prec_words, d);
    ///**mpsqrtx(sk0, b, prec_words);
    gmpsqrt(sk0, 1, b, interval_b, prec_words, d);
    ///**mpeq(sk1, a, prec_words);
    gmpeq(sk1, 1, a, interval_a, prec_words);
    ///**mpsub(a, b, sk0, prec_words);
    gmpsub(a, interval_a, b, interval_b, sk0, 1, prec_words, d);
    
    // Check for convergence.
  }
  while(sk0[1] != 0.0 && (sk0[2] < s1 || sk0[2] >= -2));
  ///**if (debug_level >= 6) 
  ///**  cerr << "MPAGMX: Iteration = " << l1 << ", Tol. Achieved = " << sk0[2] << endl;
  return;
}


__device__
void gmpagm(double* a, const int interval_a, 
			double* b, const int interval_b, 
			const int prec_words,
			double* d, const int interval_d )
{
	/**
  if(error_no != 0) {
    if(error_no == 99) mpabrt();
    zero(a); zero(b);
    return;
  }
  */
  
  int n6 = prec_words + 6;
  ///**mp_real sk0(0.0, n6), sk1(0.0, n6);
  double sk0[LOG_TMP_SIZE];
  double sk1[LOG_TMP_SIZE];
  new_gmp( sk0, 0.0, n6 );
  new_gmp( sk1, 0.0, n6 );
  int l1 = 0;
  double s1;
  sk0[2] = 10000000.0;// high value to force second iteration.

  do {
    l1++;
    if (l1 == 50) {
      /**
	  if(MPKER[5] != 0) {
		cerr <<"*** MPAGMX: Iteration limit exceeded." << endl;
		error_no = 5;
		if(MPKER[error_no] == 2) mpabrt();
      }
	  */
      break;
    }
    
    s1 = sk0[2];
    ///**mpadd(a, b, sk0, prec_words);
    gmpadd(a, interval_a, b, interval_b, sk0, 1, prec_words, d, interval_d);
    ///**mpmuld(sk0, 0.5, 0, sk1, prec_words);
    gmpmuld(sk0, 1, 0.5, 0, sk1, 1, prec_words, d, interval_d);
    ///**mpmulx(a, b, sk0, prec_words);
    gmpmul(a, interval_a, b, interval_b, sk0, 1, prec_words, d, interval_d);
    ///**mpsqrtx(sk0, b, prec_words);
    gmpsqrt(sk0, 1, b, interval_b, prec_words, d, interval_d);
    ///**mpeq(sk1, a, prec_words);
    gmpeq(sk1, 1, a, interval_a, prec_words);
    ///**mpsub(a, b, sk0, prec_words);
    gmpsub(a, interval_a, b, interval_b, sk0, 1, prec_words, d, interval_d);
    
    // Check for convergence.
  }
  while(sk0[1] != 0.0 && (sk0[2] < s1 || sk0[2] >= -2));
  ///**if (debug_level >= 6) 
  ///**  cerr << "MPAGMX: Iteration = " << l1 << ", Tol. Achieved = " << sk0[2] << endl;
  return;
}


#endif

