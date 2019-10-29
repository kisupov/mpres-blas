
#ifndef _GMP_MULD_CU_
#define _GMP_MULD_CU_

  /**
   * This routine multiplies the MP number A by the DPE number (B, N) to yield
   * the MP product C.  Debug output starts with debug_level = 9.
   *
   * Here, DPE means double precision + exponent, so (B, N) = b * 2^n
   */


/**
	@p a, @p c are both INTERVAL access
	@p d is SEQUENTIAL access
	the size of @p d is (prec_words + 6)
*/
__device__
void gmpmuld( const double* a, const int interval_a, 
			 double b, int n, 
			 double* c, const int interval_c, 
			 const int prec_words, 
			 double* d, const int interval_d = 1 ) {

  int i, ia, ib, k, na, n1, n2, d_add;
  double bb, t[2];
  //**double* d;
  
  /**
  if (error_no != 0) {
    if (error_no == 99)  mpabrt();
    zero(c);
    return;
  }
  
  if (debug_level >= 8) {
    print_mpreal("MPMULD a ", a);
    cerr << " MPMULD b " << b << " " << n << endl; 
  }*/
  
  // Check for zero inputs.
  //**ia = int(sign(1.0, a[1]));
  ia = int(sign(1.0, mpr_get(a, interval_a, 1)));
  //**na = std::min (int(std::abs(a[1])), prec_words);
  na = min (int(fabs(mpr_get(a, interval_a, 1))), prec_words);
  ib = int(sign(1.0, b));
  if (na == 0 || b == 0.0) {
    zero(c, interval_c);
    //**if (debug_level >= 9) print_mpreal("MPMULD O ", c);
    return;
  }
  if(n) {
    n1 = n / MPNBT;      // n = mpnbt*n1+n2
    n2 = n - MPNBT * n1;
    bb = ldexp(fabs(b), n2);
  } else {
    n1 = n2 = 0;
    bb = fabs(b);
  }

  // Reduce BB to within 1 and MPBDX.
  if (bb >= MPBDX) {
    for (k = 1; k <= 100; ++k) {
      bb = MPRDX * bb;
      if (bb < MPBDX) {
        n1 = n1 + k;
        break;
      }
    }
  } else if (bb < 1.0) {
    for (k = 1; k <= 100; ++k) {
      bb = MPBDX * bb;
      if (bb >= 1.0) {
        n1 = n1 - k;
        break;
      }
    }
  }
/**#if 0
  printf("mpmuld[1]: bb = %22.18e, na = %d\n", bb, na);
#endif*/
  // BB is now between 1 and MPBDX (and positive)
  // If BB cannot be represented exactly in a single mantissa word, use MPMUL.
  if (bb != floor(bb)) {
    //**mp_real f(0.0, 9);
	double f[9];
	new_gmp( f, 0.0, 9 );
    gmpdmc(b, n, f, 1, prec_words);
	double new_d[MUL_D_SIZE];
    gmpmul(f, 1, a, interval_a, c, interval_c, prec_words, new_d);
    //**if (debug_level >= 9) print_mpreal("MPMULD O ", c);
    return; 
  }
  
  //**d = new double[prec_words+6];
  d_add = 0;

  // Perform short multiply operation.
  mpr_set(d, interval_d, 2, 0.0); //d[2] = 0.;
  for (i = FST_M; i < na + FST_M; ++i) {

    //**t[0] = mp_two_prod_positive(a[i], bb, t[1]); 
    t[0] = mp_two_prod_positive(mpr_get(a, interval_a, i), bb, t[1]); 
        // t[0], t[1] non-overlap, <= 2^mpnbt-1

    //d[i-1] += t[0];  // exact, <= 2^53-2
    //d[i] = t[1];

    mpr_compadd(d, interval_d, i-1, t[0]);  // exact, <= 2^53-2
    mpr_set(d, interval_d, i, t[1]);
  }
  
  // If carry is nonzero, shift the result one word right.
  if (mpr_get(d, interval_d, 2) != 0.0) {
    ++n1;  // exponent
    ++na;//number of words

    d = d - interval_d; //d--;
    d_add++;// "shift" the array one to the right.
    // This has the same effect as the following commented out loop:
    //for (i = na + FST_M; i >= FST_M; --i) d[i] = d[i-1];
  }

  // Set the exponent and fix up the result.
  //d[1] = ia+ib ? na : -na;//same as sign (na, ia * ib);
  //**d[2] = a[2] + n1;
  //d[2] = mpr_get(a, interval_a, 2) + n1;
  //d[na+3] = 0.0;
  //d[na+4] = 0.0;
  
  mpr_set(d, interval_d, 1, ia+ib ? na : -na);//same as sign (na, ia * ib);
  //**d[2] = a[2] + n1;
  mpr_set(d, interval_d, 2, mpr_get(a, interval_a, 2) + n1);
  mpr_set(d, interval_d, na+3, 0.0);
  mpr_set(d, interval_d, na+4, 0.0);

  //  Fix up result, since some words may be negative or exceed MPBDX.
  gmpnorm(d, interval_d, c, interval_c, prec_words);
  //**delete [] (d + d_add);

  d = d + interval_d; //we need to reset d!!!

  //**if (debug_level >= 8) print_mpreal("MPMULD O ", c);
  return;
 /*
  */
}


#endif

