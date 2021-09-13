

#ifndef _GMP_DMC_CU_
#define _GMP_DMC_CU_

  /**
   * This routine converts the DPE number (A, N) to MP form in B.  All bits of
   * A are recovered in B.  However, note for example that if A = 0.1D0 and N
   * is 0, then B will NOT be the multiprecision equivalent of 1/10.  Debug
   * output starts with debug_level = 9.
   *
   */

__device__
void gmpdmc( double a, int n, double* b, const int interval_b, 
			const int prec_words ) {
  int i, k, n1, n2;
  double aa;

	/**
  if (error_no != 0) {
    if (error_no == 99)  mpabrt();
    zero(b);
    return;
  }
  
  if (debug_level >= 8) {
    cerr << " MPDMC I: a = " << a << endl;
    cerr << "n = " << n << endl;
    cerr << "mpdmc[1] prec_words " << prec_words << endl;
  }*/
  
  //  Check for zero.
  if (a == 0.0) {
    zero(b, interval_b);
    //**if (debug_level >= 9) print_mpreal("MPDMC O ", b);
    return;
  }
  
  if(n) {
    n1 = n / MPNBT;      // n = mpnbt*n1+n2
    n2 = n - MPNBT * n1;
    aa = ldexp(fabs(a), n2);
  } else {
    n1 = n2 = 0;
    aa = fabs(a);
  }

  //  Reduce AA to within 1 and MPBDX.
  if (aa >= MPBDX) {
    for (k = 1; k <= 100; ++k) {
      aa = MPRDX * aa;
      if (aa < MPBDX) {
        n1 = n1 + k;
        break;
      }
    }
  } else if (aa < 1.0) {
    for (k = 1; k <= 100; ++k) {
      aa = MPBDX * aa;
      if (aa >= 1.0) {
        n1 = n1 - k;
        break;
      }
    } 
  }
  
  //  Store successive sections of AA into B.
  double d[8];

  d[2] = n1;
  d[3] = FLOOR_POSITIVE(aa);
  aa = MPBDX * (aa - d[3]);
  d[4] = FLOOR_POSITIVE(aa);
  aa = MPBDX * (aa - d[4]);
  d[5] = FLOOR_POSITIVE(aa);
  d[6] = 0.;
  d[7] = 0.;

  for (i = 5; i >= FST_M; --i)
    if (d[i] != 0.) break;
    
  aa = i - 2;
  //**aa = std::min(aa, (b[0])-5);
  aa = min(aa, (mpr_get(b, interval_b, 0))-5);
  //**b[1] = sign(aa, a);
  mpr_set(b, interval_b, 1, sign(aa, a)); 
  for(i=2;i<int(aa)+FST_M;i++) {
    //**b[i] = d[i];
    mpr_set(b, interval_b, i, d[i]);
  }
    
  //**if (debug_level >= 8) print_mpreal("MPDMC O ", b);
  /*
  */
}



/**
* allocate a new mp number with SEQUENTIAL access
* the initialized value is a
*/
__device__
void new_gmp( double* mpr, double a, int sz ) {
	mpr[0] = (double)sz;
	mpr[1] = 0.0;
	
	if( sz > 0 ) {
		//gmpdmc( double a, int n, double* b, const int interval_b, const int prec_words )
		gmpdmc( a, 0, mpr, 1, *gPrecWords );
	}
}

/**
* allocate a new gmp number with INTERVAL access
*/
__device__
void new_gmp( double* mpr, const int interval, double a, int sz ) {
	mpr[0] = (double)sz;
	mpr[interval] = 0.0;

	if( sz > 0 ) {
		gmpdmc( a, 0, mpr, interval, *gPrecWords );
	}
}

__device__
void new_gmp( double* mpr, const int interval, double a, int sz, const int prec_words ) {
	mpr[0] = (double)sz;
	mpr[interval] = 0.0;

	if( sz > 0 ) {
		gmpdmc( a, 0, mpr, interval, prec_words );
	}
}


__device__
void new_gmp( double* mpr, const int prec_words ) {
	mpr[0] = (double)(prec_words + 5);
	mpr[1] = 0.0;
}

#endif

