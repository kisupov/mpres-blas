
#ifndef _GMP_GEQ_CU_
#define _GMP_GEQ_CU_

/**
* direct copy n_words from a to b, the caller should guarantee the security
*/
__device__
void gmpcopy( const double* a, const int interval_a, 
		 double* b, const int interval_b, 
		 const int n_words ) {
	 for( int i = 0; i < n_words; i++ ) {
		//**b[i] = a[i];
		mpr_set( b, interval_b, i, mpr_get(a, interval_a, i) );
	 }
}

  // This routine sets the MP number B equal to the MP number A.  Debug output
  // starts with debug_level = 10.
  //
  // Max DP space for B: MPNW + 3 cells.
  //
  // The fact that only MPNW + 3 cells, and not MPNW + 4 cells, are copied is
  // important in some routines that increase the precision level by one.
__device__
void gmpeq(const double* a, const int interval_a, 
			double* b, const int interval_b, int prec_words)
{ 
  int i, ia, na, nb;
  /**
  if (error_no != 0) {
    if (error_no == 99) mpabrt();
    zero(b);
    return;
  }
  if (debug_level >= 10) cerr << "MPEQ" << endl;
  */

  ///**ia = int(sign(1.0, a[1]));
  ia = int(sign(1.0, mpr_get(a, interval_a, 1)));
  ///**na = std::min(int(std::abs(a[1])), prec_words);
  na = min(int(fabs(mpr_get(a, interval_a, 1))), prec_words);
  ///**nb = std::min(na, int(b[0])-FST_M-1);
  nb = min(na, int(mpr_get(b, interval_b, 0))-FST_M-1);
  if (na == 0) {
    zero(b, interval_b);
    return;
  }

  ///**b[1] = sign(nb, ia);
  mpr_set(b, interval_b, 1, sign(nb, ia));
  ///**for (i = 2; i < nb + FST_M; ++i) b[i] = a[i];
  for (i = 2; i < nb + FST_M; ++i) {
	mpr_set(b, interval_b, i, mpr_get(a, interval_a, i));
  }
  ///**nb = std::min(nb + FST_M + 1, int(b[0]));
  nb = min(nb + FST_M + 1, int(mpr_get(b, interval_b, 0)));
  ///**for (; i < nb; i++) b[i] = 0.0;
  for (; i < nb; i++) mpr_set(b, interval_b, i, 0.0);
}


#endif

