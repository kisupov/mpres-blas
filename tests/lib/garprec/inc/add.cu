
#ifndef _GMP_ADD_CU_
#define _GMP_ADD_CU_

/**
* perform the addition c = a + b
*/

/**
* @p d is a temperoral buffer, should be allocated outside with the size (prec_words+7)
* @p a, @p b and @p c are all INTERVAL access, but @p d is SEQUENTIAL access
*/
__device__
void gmpadd( const double* a, const int interval_a, 
			 const double* b, const int interval_b, 
			 double* c, const int interval_c, 
			 const int prec_words, 
			 double* d, const int interval_d = 1 )
{
  double db;
  int i, ia, ib, ish, ixa, ixb, ixd, na, nb;
  int m1, m2, m3, m4, m5, nsh;
  int nd; // number of actual words in d[]

  
  //**ia = a[1] >= 0 ? 1 : -1;
  //**ib = b[1] >= 0 ? 1 : -1;
  ia = mpr_get( a, interval_a, 1 ) >= 0 ? 1 : -1;
  ib = mpr_get( b, interval_b, 1 ) >= 0 ? 1 : -1;
  //**na = min (static_cast<int>(std::abs(a[1])), prec_words); // number of words in A
  //**nb = min (static_cast<int>(gmpaddstd::abs(b[1])), prec_words); // number of words in B
  na = min((int)(fabs(mpr_get( a, interval_a, 1 ))), prec_words); // number of words in A
  nb = min((int)(fabs(mpr_get( b, interval_b, 1 ))), prec_words); // number of words in B


  if (na == 0) {
    //**int num_words = min(nb, (int)(c[0])-FST_M);
    int num_words = min(nb, (int)(mpr_get(c, interval_c, 0))-FST_M);
    //**c[1] = ib > 0 ? num_words : -num_words;
    mpr_set( c, interval_c, 1, ib > 0 ? num_words : -num_words );
    for (i = 2; i < num_words + FST_M; ++i) {
		//**c[i] = b[i];
		mpr_set( c, interval_c, i, mpr_get( b, interval_b, i ) );
	}
    return;
  } else if (nb == 0) {
    //**int num_words = min(na, int(c[0])-FST_M);
    int num_words = min(na, int(mpr_get( c, interval_c, 0 ))-FST_M);
    //**c[1] = ia >= 0 ? num_words : -num_words; 
    mpr_set( c, interval_c, 1, ia >= 0 ? num_words : -num_words );
    for (i = 2; i < num_words + FST_M; ++i) {
		//**c[i] = a[i];
		mpr_set( c, interval_c, i, mpr_get( a, interval_a, i ) );
	}
    return;
  }

  if (ia == ib) db = 1.0; //same signs - add
  else db = -1.0; // different signs - subtract

  //**ixa = static_cast<int>(a[2]);
  ixa = (int)(mpr_get(a, interval_a, 2));
  //**ixb = static_cast<int>(b[2]);
  ixb = (int)(mpr_get(b, interval_b, 2));
  ish = ixa - ixb;

  //**d[1] = 0.0;
  mpr_set(d, interval_d, 1, 0.0);
  //**d[2] = 0.0;
  mpr_set(d, interval_d, 2, 0.0);

  if (ish >= 0) { // |A| >= |B|
    
    m1 = min (na, ish);
    m2 = min (na, nb + ish);
    m3 = na;
    m4 = min (max (na, ish), prec_words + 1);
    m5 = min (max (na, nb + ish), prec_words + 1);
    
    for (i = FST_M; i < m1 + FST_M; ++i) {
		//**d[i] = a[i];
		mpr_set(d, interval_d, i, mpr_get(a, interval_a, i)); //a[i];
	}
    
    if(db > 0) {//Addition
      for (i = m1 + FST_M; i < m2 + FST_M; ++i)
         mpr_set(d, interval_d, i, mpr_get(a, interval_a, i) + mpr_get(b, interval_b, i-ish)); //**d[i] = a[i] + b[i-ish];
      
      for (i = m2 + FST_M; i < m3 + FST_M; ++i)
        mpr_set(d, interval_d, i, mpr_get( a, interval_a, i ));//**d[i] = a[i];
    
      for (i = m3 + FST_M; i < m4 + FST_M; ++i)
        mpr_set(d, interval_d, i, 0.0);
      
      for (i = m4 + FST_M; i < m5 + FST_M; ++i)
        mpr_set(d, interval_d, i, mpr_get( b, interval_b, i-ish ));//**d[i] = b[i-ish];
    } else {//Subtraction
      for (i = m1 + FST_M; i < m2 + FST_M; ++i)
        mpr_set(d, interval_d, i, mpr_get(a, interval_a, i) - mpr_get(b, interval_b, i-ish));//**d[i] = a[i] - b[i-ish];
      
      for (i = m2 + FST_M; i < m3 + FST_M; ++i)
        mpr_set(d, interval_d, i, mpr_get( a, interval_a, i ));//**d[i] = a[i];
    
      for (i = m3 + FST_M; i < m4 + FST_M; ++i)
        mpr_set(d, interval_d, i, 0.0);
      
      for (i = m4 + FST_M; i < m5 + FST_M; ++i)
        mpr_set(d, interval_d, i, - mpr_get( b, interval_b, i-ish ));//**d[i] = - b[i-ish];
    }
    nd = m5;
    ixd = ixa;
    mpr_set(d, interval_d, nd+3, 0.0);
    mpr_set(d, interval_d, nd+4, 0.0);

  } else {
    
    nsh = -ish;
    m1 = min (nb, nsh);
    m2 = min (nb, na + nsh);
    m3 = nb;
    m4 = min (max (nb, nsh), prec_words + 1);
    m5 = min (max (nb, na + nsh), prec_words + 1);
    
    if(db > 0) {//Addition
      for (i = FST_M; i < m1 + FST_M; ++i)
        mpr_set(d, interval_d, i, mpr_get( b, interval_b, i ));//**d[i] = b[i];
      
      for (i = m1 + FST_M; i < m2 + FST_M; ++i)
        mpr_set(d, interval_d, i, mpr_get(a, interval_a, i-nsh) + mpr_get(b, interval_b, i));//d[i] = a[i-nsh] + b[i];
      
      for (i = m2 + FST_M; i < m3 + FST_M; ++i)
        mpr_set(d, interval_d, i, mpr_get( b, interval_b, i ));//**d[i] = b[i];

    } else {//Subtraction
      for (i = FST_M; i < m1 + FST_M; ++i)
        mpr_set(d, interval_d, i,  -mpr_get( b, interval_b, i ));//**d[i] = - b[i];
      
      for (i = m1 + FST_M; i < m2 + FST_M; ++i)
        mpr_set(d, interval_d, i, mpr_get(a, interval_a, i-nsh) - mpr_get(b, interval_b, i));//**d[i] = a[i-nsh]  - b[i];
      
      for (i = m2 + FST_M; i < m3 + FST_M; ++i)
        mpr_set(d, interval_d, i, -mpr_get(b, interval_b, i));//**d[i] = - b[i];
    }

    for (i = m3 + FST_M; i < m4 + FST_M; ++i)
      mpr_set(d, interval_d, i, 0.0);
    
    for (i = m4 + FST_M; i < m5 + FST_M; ++i)
      mpr_set(d, interval_d, i, mpr_get( a, interval_a, i-nsh ));//**d[i] = a[i-nsh];
    
    nd = m5;
    ixd = ixb;
    mpr_set(d, interval_d, nd+3, 0.0);
    mpr_set(d, interval_d, nd+4, 0.0);
  }

  // Call mpnorm to fix up result and store in c.
  mpr_set(d, interval_d, 1, ia >= 0 ? nd : -nd);
  mpr_set(d, interval_d, 2, ixd);
  gmpnorm(d, interval_d, c, interval_c, prec_words);

  return;
/*
  */
}

#endif
