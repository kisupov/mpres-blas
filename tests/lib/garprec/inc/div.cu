
#ifndef _GMP_DIV_CU_
#define _GMP_DIV_CU_

/**
* perform multiplication c = a / b
*/

/**
* @p d is a temperoral buffer, should be allocated outside with the size (prec_words+9)
* @p a, @p b and @p c are all INTERVAL access, but @p d is SEQUENTIAL access
*/
__device__
void gmpdiv( const double* a, const int interval_a, 
			 const double* b, const int interval_b, 
			 double* c, const int interval_c, 
			 const int prec_words, 
			 double* d ) {

  int i, ia, ib, ij, is, i2, i3=0, j, j3, na, nb, nc, BreakLoop;
  double rb, ss, t0, t1, t2, t[2];
  //**double* d;
  
  /**
  if (error_no != 0) {
    if (error_no == 99) mpabrt();
    zero(c);
    return;
  }
  
  if (debug_level >= 8) {
    print_mpreal("MPDIV a ", a);
    print_mpreal("MPDIV b ", b);
  }*/
  
  
  
  //**ia = (a[1] >= 0 ? 1 : -1); 
  ia = (mpr_get(a, interval_a, 1) >= 0 ? 1 : -1); 
  //**ib = (b[1] >= 0 ? 1 : -1); 
  ib = (mpr_get(b, interval_b, 1) >= 0 ? 1 : -1); 
  //**na = std::min (int(std::abs(a[1])), prec_words);
  na = min (int(fabs(mpr_get(a, interval_a, 1))), prec_words);
  //**nb = std::min (int(std::abs(b[1])), prec_words);
  nb = min (int(fabs(mpr_get(b, interval_b, 1))), prec_words);
  
  //  Check if dividend is zero.
  if (na == 0) {
    zero(c, interval_c);
    //**if (debug_level >= 8) print_mpreal("MPDIV O ", c);
    return;
  }
  
  //**if (nb == 1 && b[FST_M] == 1.) {
  if ((nb == 1) && (mpr_get(b, interval_b, FST_M) == 1.)) {
    // Divisor is 1 or -1 -- result is A or -A.
    //**c[1] = sign(na, ia * ib);
    mpr_set(c, interval_c, 1, sign(na, ia * ib));
    //**c[2] = a[2] - b[2];
    mpr_set(c, interval_c, 2, mpr_get(a, interval_a, 2) - mpr_get(b, interval_b, 2));
    for (i = FST_M; i < na+FST_M; ++i) { 
		//**c[i] = a[i];
		mpr_set( c, interval_c, i, mpr_get(a, interval_a, i) );
	}
    
    //**if (debug_level >= 8) print_mpreal("MPDIV O ", c);
    return;
  }
  
  //  Check if divisor is zero.
  /**if (nb == 0) {
    if (MPKER[31] != 0) {
      cerr << "*** MPDIV: Divisor is zero." << endl;
      error_no = 31;
      if (MPKER[error_no] == 2) mpabrt();
    }
    return;
  }*/
  
  //need the scratch space now...
  //**d = new double[prec_words+9];
  int d_add=0;
  d++; d_add--;

  // Initialize trial divisor and trial dividend.
  //**t0 = mpbdx * b[3];
  t0 = MPBDX * mpr_get(b, interval_b, 3);
  //**if (nb >= 2) t0 = t0 + b[4];
  if (nb >= 2) t0 = t0 + mpr_get(b, interval_b, 4);
  //**if (nb >= 3) t0 = t0 + mprdx * b[5];
  if (nb >= 3) t0 = t0 + MPRDX * mpr_get(b, interval_b, 5);
  rb = 1.0 / t0;
  d[0]  = d[1] = 0.0;
  
  //**for (i = 2; i < na+2; ++i) d[i] = a[i+1];
  for (i = 2; i < na+2; ++i) d[i] = mpr_get(a, interval_a, i+1);
  for (; i <= prec_words+7; ++i) d[i] = 0.0;

  // Perform ordinary long division algorithm.  First compute only the first
  // NA words of the quotient.
  for (j = 2; j <= na+1; ++j) {
    t1 = MPBX2 * d[j-1] + MPBDX * d[j] + d[j+1];
    t0 = AINT (rb * t1); // trial quotient, approx is ok.
    j3 = j - 3;
    i2 = min (nb, prec_words + 2 - j3) + 2;
    ij = i2 + j3;
    for (i = 3; i <= i2; ++i) {
      i3 = i + j3;
      //**t[0] = mp_two_prod(t0, b[i], t[1]);
      t[0] = mp_two_prod(t0, mpr_get(b, interval_b, i), t[1]);
      d[i3-1] -= t[0];   // >= -(2^mpnbt-1), <= 2^mpnbt-1
      d[i3] -= t[1];
    }
    
      // Release carry to avoid overflowing the exact integer capacity
      // (2^52-1) of a floating point word in D.
    //**if(!(j & (mp::mpnpr-1))) { // assume mpnpr is power of two
    if(!(j & (MPNPR-1))) { // assume mpnpr is power of two
      t2 = 0.0;
      for(i=i3;i>j+1;i--) {
        t1 = t2 + d[i];
        t2 = int (t1 * MPRDX);     // carry <= 1
        d[i] = t1 - t2 * MPBDX;   // remainder of t1 * 2^(-mpnbt)
      }
      d[i] += t2;
    }
    
    d[j] += MPBDX * d[j-1];
    d[j-1] = t0; // quotient
  }
  
  // Compute additional words of the quotient, as long as the remainder
  // is nonzero.  
  BreakLoop = 0;
  for (j = na+2; j <= prec_words+3; ++j) {
    t1 = MPBX2 * d[j-1] + MPBDX * d[j];
    if (j < prec_words + 3) t1 += d[j+1];
    t0 = AINT (rb * t1); // trial quotient, approx is ok.
    j3 = j - 3;
    i2 = min (nb, prec_words + 2 - j3) + 2;
    ij = i2 + j3;
    ss = 0.0;
    
    for (i = 3; i <= i2; ++i) {
      i3 = i + j3;
      //**t[0] = mp_two_prod(t0, b[i], t[1]); b
      t[0] = mp_two_prod(t0, mpr_get(b, interval_b, i), t[1]);
      d[i3-1] -= t[0];   // >= -(2^mpnbt-1), <= 2^mpnbt-1
      d[i3] -= t[1];
      
      //square to avoid cancellation when d[i3] or d[i3-1] are negative
      ss += d_sqr (d[i3-1]) + d_sqr (d[i3]); 
    }
      // Release carry to avoid overflowing the exact integer capacity
      // (2^mpnbt-1) of a floating point word in D.
    if(!(j & (MPNPR-1))) { // assume mpnpr is power of two
      t2 = 0.0;
      for(i=i3;i>j+1;i--) {
        t1 = t2 + d[i];
        t2 = int (t1 * MPRDX);     // carry <= 1
        d[i] = t1 - t2 * MPBDX;   // remainder of t1 * 2^(-mpnbt)
      }
      d[i] += t2;
    }

    d[j] += MPBDX * d[j-1];
    d[j-1] = t0;
    if (ss == 0.0) {
      BreakLoop = 1;
      break;
    }
    if (ij <= prec_words+1) d[ij+3] = 0.0;
  } 
	

  // Set sign and exponent, and fix up result.
  if(!BreakLoop) j--;
  d[j] = 0.0;
  
  if (d[1] == 0.0) {
    is = 1;
    d--; d_add++;
  } else {
    is = 2;
    d-=2; d_add+=2;
    //for (i = j+1; i >= 3; --i) d[i] =  d[i-2];
  }

  //**nc = std::min( (int(c[0])-FST_M-2), std::min (j-1, prec_words));
  nc = min( (int(mpr_get(c, interval_c, 0))-FST_M-2), min (j-1, prec_words));
  

  d[1] = ia+ib ? nc : -nc;//sign(nc, ia * ib);
  //**d[2] = a[2] - b[2] + is - 2;  
  d[2] = mpr_get(a, interval_a, 2) - mpr_get(b, interval_b, 2) + is - 2;  
  
  gmpnorm(d, c, interval_c, prec_words);
  //**delete [] (d+d_add);
  
  //**if (debug_level >= 8) print_mpreal("MPDIV O ", c);
  return;
  
}

__device__
void gmpdiv( const double* a, const int interval_a, 
			 const double* b, const int interval_b, 
			 double* c, const int interval_c, 
			 const int prec_words, 
			 double* d, const int interval_d ) {

  int i, ia, ib, ij, is, i2, i3=0, j, j3, na, nb, nc, BreakLoop;
  double rb, ss, t0, t1, t2, t[2];
  //**double* d;
  
  /**
  if (error_no != 0) {
    if (error_no == 99) mpabrt();
    zero(c);
    return;
  }
  
  if (debug_level >= 8) {
    print_mpreal("MPDIV a ", a);
    print_mpreal("MPDIV b ", b);
  }*/
  
  
  
  //**ia = (a[1] >= 0 ? 1 : -1); 
  ia = (mpr_get(a, interval_a, 1) >= 0 ? 1 : -1); 
  //**ib = (b[1] >= 0 ? 1 : -1); 
  ib = (mpr_get(b, interval_b, 1) >= 0 ? 1 : -1); 
  //**na = std::min (int(std::abs(a[1])), prec_words);
  na = min (int(fabs(mpr_get(a, interval_a, 1))), prec_words);
  //**nb = std::min (int(std::abs(b[1])), prec_words);
  nb = min (int(fabs(mpr_get(b, interval_b, 1))), prec_words);
  
  //  Check if dividend is zero.
  if (na == 0) {
    zero(c, interval_c);
    //**if (debug_level >= 8) print_mpreal("MPDIV O ", c);
    return;
  }
  
  //**if (nb == 1 && b[FST_M] == 1.) {
  if ((nb == 1) && (mpr_get(b, interval_b, FST_M) == 1.)) {
    // Divisor is 1 or -1 -- result is A or -A.
    //**c[1] = sign(na, ia * ib);
    mpr_set(c, interval_c, 1, sign(na, ia * ib));
    //**c[2] = a[2] - b[2];
    mpr_set(c, interval_c, 2, mpr_get(a, interval_a, 2) - mpr_get(b, interval_b, 2));
    for (i = FST_M; i < na+FST_M; ++i) { 
		//**c[i] = a[i];
		mpr_set( c, interval_c, i, mpr_get(a, interval_a, i) );
	}
    
    //**if (debug_level >= 8) print_mpreal("MPDIV O ", c);
    return;
  }
  
  //  Check if divisor is zero.
  /**if (nb == 0) {
    if (MPKER[31] != 0) {
      cerr << "*** MPDIV: Divisor is zero." << endl;
      error_no = 31;
      if (MPKER[error_no] == 2) mpabrt();
    }
    return;
  }*/
  
  //need the scratch space now...
  //**d = new double[prec_words+9];
  int d_add=0;
  d = d + interval_d;//d++; 
  d_add--;

  // Initialize trial divisor and trial dividend.
  //**t0 = mpbdx * b[3];
  t0 = MPBDX * mpr_get(b, interval_b, 3);
  //**if (nb >= 2) t0 = t0 + b[4];
  if (nb >= 2) t0 = t0 + mpr_get(b, interval_b, 4);
  //**if (nb >= 3) t0 = t0 + mprdx * b[5];
  if (nb >= 3) t0 = t0 + MPRDX * mpr_get(b, interval_b, 5);
  rb = 1.0 / t0;
  //d[0]  = d[1] = 0.0;
  mpr_set(d, interval_d, 0, 0.0);
  mpr_set(d, interval_d, 1, 0.0);
  
  //**for (i = 2; i < na+2; ++i) d[i] = a[i+1];
  for (i = 2; i < na+2; ++i) mpr_set(d, interval_d, i, mpr_get(a, interval_a, i+1));
  for (; i <= prec_words+7; ++i) mpr_set(d, interval_d, i, 0.0);

  // Perform ordinary long division algorithm.  First compute only the first
  // NA words of the quotient.
  for (j = 2; j <= na+1; ++j) {
    t1 = MPBX2 * mpr_get(d, interval_d, j-1) + MPBDX * mpr_get(d, interval_d, j) + mpr_get(d, interval_d, j+1);
    t0 = AINT (rb * t1); // trial quotient, approx is ok.
    j3 = j - 3;
    i2 = min (nb, prec_words + 2 - j3) + 2;
    ij = i2 + j3;
    for (i = 3; i <= i2; ++i) {
      i3 = i + j3;
      //**t[0] = mp_two_prod(t0, b[i], t[1]);
      t[0] = mp_two_prod(t0, mpr_get(b, interval_b, i), t[1]);
      mpr_compsub(d, interval_d, i3-1, t[0]);   // >= -(2^mpnbt-1), <= 2^mpnbt-1
      mpr_compsub(d, interval_d, i3, t[1]);
    }
    
      // Release carry to avoid overflowing the exact integer capacity
      // (2^52-1) of a floating point word in D.
    //**if(!(j & (mp::mpnpr-1))) { // assume mpnpr is power of two
    if(!(j & (MPNPR-1))) { // assume mpnpr is power of two
      t2 = 0.0;
      for(i=i3;i>j+1;i--) {
        t1 = t2 + mpr_get(d, interval_d, i);
        t2 = int (t1 * MPRDX);     // carry <= 1
        mpr_set(d, interval_d, i, t1 - t2 * MPBDX);   // remainder of t1 * 2^(-mpnbt)
      }
      mpr_compadd(d, interval_d, i, t2);
    }
    
    mpr_compadd(d, interval_d, j, MPBDX * mpr_get(d, interval_d, j-1));
    mpr_set(d, interval_d, j-1, t0); // quotient
  }
  
  // Compute additional words of the quotient, as long as the remainder
  // is nonzero.  
  BreakLoop = 0;
  for (j = na+2; j <= prec_words+3; ++j) {
    t1 = MPBX2 * mpr_get(d, interval_d, j-1) + MPBDX * mpr_get(d, interval_d, j);
    if (j < prec_words + 3) t1 += mpr_get(d, interval_d, j+1);
    t0 = AINT (rb * t1); // trial quotient, approx is ok.
    j3 = j - 3;
    i2 = min (nb, prec_words + 2 - j3) + 2;
    ij = i2 + j3;
    ss = 0.0;
    
    for (i = 3; i <= i2; ++i) {
      i3 = i + j3;
      //**t[0] = mp_two_prod(t0, b[i], t[1]); b
      t[0] = mp_two_prod(t0, mpr_get(b, interval_b, i), t[1]);
      mpr_compsub(d, interval_d, i3-1, t[0]);   // >= -(2^mpnbt-1), <= 2^mpnbt-1
      mpr_compsub(d, interval_d, i3, t[1]);
      
      //square to avoid cancellation when d[i3] or d[i3-1] are negative
      ss += d_sqr (mpr_get(d, interval_d, i3-1)) + d_sqr (mpr_get(d, interval_d, i3)); 
    }
      // Release carry to avoid overflowing the exact integer capacity
      // (2^mpnbt-1) of a floating point word in D.
    if(!(j & (MPNPR-1))) { // assume mpnpr is power of two
      t2 = 0.0;
      for(i=i3;i>j+1;i--) {
        t1 = t2 + mpr_get(d, interval_d, i);
        t2 = int (t1 * MPRDX);     // carry <= 1
        mpr_set(d, interval_d, i, t1 - t2 * MPBDX);   // remainder of t1 * 2^(-mpnbt)
      }
      mpr_compadd(d, interval_d, i, t2);
    }

    mpr_compadd(d, interval_d, j, MPBDX*mpr_get(d, interval_d, j-1));
    mpr_set(d, interval_d, j-1, t0);
    if (ss == 0.0) {
      BreakLoop = 1;
      break;
    }
    if (ij <= prec_words+1) mpr_set(d, interval_d, ij+3, 0.0);
  } 
	

  // Set sign and exponent, and fix up result.
  if(!BreakLoop) j--;
  mpr_set(d, interval_d, j, 0.0);
  
  if (mpr_get(d, interval_d, 1) == 0.0) {
    is = 1;
    d = d - interval_d; 
    d_add++;
  } else {
    is = 2;
    //d-=2; 
    d = d - 2*interval_d;
    d_add+=2;
    //for (i = j+1; i >= 3; --i) d[i] =  d[i-2];
  }

  //**nc = std::min( (int(c[0])-FST_M-2), std::min (j-1, prec_words));
  nc = min( (int(mpr_get(c, interval_c, 0))-FST_M-2), min (j-1, prec_words));
  

  mpr_set(d, interval_d, 1, ia+ib ? nc : -nc);//sign(nc, ia * ib);
  //**d[2] = a[2] - b[2] + is - 2;  
  mpr_set(d, interval_d, 2, mpr_get(a, interval_a, 2) - mpr_get(b, interval_b, 2) + is - 2);  
  
  gmpnorm(d, interval_d, c, interval_c, prec_words);
  //**delete [] (d+d_add);
  
  //**if (debug_level >= 8) print_mpreal("MPDIV O ", c);
  return;
  
}


#endif
