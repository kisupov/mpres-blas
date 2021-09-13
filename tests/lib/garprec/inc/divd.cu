
#ifndef _GMP_DIVD_CU_
#define _GMP_DIVD_CU_
  
  
  /**
   * This routine divides the MP number A by the DPE number (B, N) to yield
   * the MP quotient C.   Debug output starts with debug_level = 9.
   *
   */
__device__   
void gmpdivd(const double* a, const int interval_a, 
			 const double b, const int n, 
			 double* c, const int interval_c, 
			 const int prec_words,
			 double* d, const int interval_d = 1 )
{

  int ia, ib, i, j, k, na, nc, n1, n2, d_add;
  double bb, br, dd, t0, t[2];
  ///**double* d;
  
  /***
  if (error_no != 0) {
    if (error_no == 99)  mpabrt();
    zero(c);
    return;
  }
  if (debug_level >= 9) {
    print_mpreal("MPDIVD a ", a);
    cerr << " MIDIVD b " << b << endl;
    cerr << "        n " << n << endl;
  }*/

  ///**ia = (a[1] >= 0 ? 1 : -1);
  ia = (mpr_get(a, interval_a, 1) >= 0 ? 1 : -1);
  ///**na = std::min (int(std::abs(a[1])), prec_words);
  na = min (int(fabs(mpr_get(a, interval_a, 1))), prec_words);
  ib = (b >= 0 ? 1 : -1);
    
  // Check if divisor is zero.
  if (b == 0.0) {
    /**if (MPKER[32] != 0) {
      cerr << "*** MPDIVD: Divisor is zero.\b.n";     
      error_no = 32;
      if (MPKER[error_no] == 2) mpabrt();
    }*/
    return;
  }


  // Check if dividend is zero.
  if (na == 0) {
    zero(c, interval_c);
    ///**if (debug_level >= 9) print_mpreal("MPDIVD O ", c);
    return;
  }

  if(n) {
    n1 = n / MPNBT;
    n2 = n - MPNBT * n1;   // n = mpnbt*n1+n2
    bb = ldexp(fabs(b), n2);
  } else {
    n1 = n2 = 0;
    bb = fabs(b);
  }

  //  Reduce BB to within 1 and MPBDX.
  if (bb >= MPBDX) {
    for (k = 1; k <= 100; ++k) {
      bb = MPRDX * bb;
      if (bb < MPBDX) {
        n1 += k;
        break;
      }
    }
  } else if (bb < 1.0) {
    for (k = 1; k <= 100; ++k) {
      bb = MPBDX * bb;
      if (bb >= 1.0) {
        n1 -= k;
        break;
      }
    }
  }
  
  // If B cannot be represented exactly in a single mantissa word, use MPDIV.
  if (bb != floor(bb)) {
    ///**mp_real f(0.0, 9);
	double f[9];
	new_gmp(f, 0.0, 9);
    bb = sign(bb, b);
    ///**mpdmc(bb , n1*mpnbt, f, prec_words);
    gmpdmc( bb, n1*MPNBT, f, 1, prec_words );
	gmpdiv(a, interval_a, f, 1, c, interval_c, prec_words, d, interval_d);
    ///**if (debug_level >= 9) print_mpreal("MPDIVD O ", c);
    return;
  }
  
  //Allocate scratch space.
  ///**d  = new double[prec_words+6];  
  d_add = 0;

  br = 1.0 / bb;
  for (i = FST_M; i < na + FST_M; ++i) {
	///**d[i] = a[i];
	mpr_set(d, interval_d, i, mpr_get(a, interval_a, i));
  }
  for (/*i = na+FST_M*/; i <= prec_words+FST_M+2 ; i++) {
    mpr_set(d, interval_d, i, 0.0);
  }
  mpr_set(d, interval_d, 2, 0.);

  // Perform short division (not vectorizable at present).
  // Continue as long as the remainder remains nonzero.
  for (j = FST_M; j <= prec_words+FST_M+1; ++j) {
    dd = MPBDX * mpr_get(d, interval_d, j-1);
    if (j < na + FST_M)
      dd += mpr_get(d, interval_d, j);
    else {
      if (dd == 0.0) {
               break;
      }
    }
    t0 = AINT (br * dd); // [0, 2^mpnbt-1], trial quotient.
    t[0] = mp_two_prod(t0, bb, t[1]); // t[0], t[1] non-overlap, <= 2^mpnbt-1
    
    //d[j-1] -= t[0];
    //d[j] -= t[1];
    //d[j] += MPBDX * d[j-1];
    //d[j-1] = t0; // quotient
  
    mpr_compsub(d, interval_d, j-1, t[0]);
    mpr_compsub(d, interval_d, j, t[1]);
    mpr_compadd(d, interval_d, j,  MPBDX * mpr_get(d, interval_d, j-1));
    mpr_set(d, interval_d, j-1, t0); // quotient
  
  }
  
  //  Set sign and exponent of result.
  j--;
  //if(AINT(d[j] * br)  != 0.0) {
  //  if(AINT(d[j] * br) >= 0.0)
  //    d[j-1] += 1.0;
  //  else
  //    d[j-1] -= 1.0;
  //}

  if(AINT(mpr_get(d, interval_d, j) * br)  != 0.0) {
    if(AINT(mpr_get(d, interval_d, j) * br) >= 0.0){
      //d[j-1] += 1.0;
      mpr_compadd(d, interval_d, j-1, 1.0);
    }
    else {
      //d[j-1] -= 1.0;
      mpr_compsub(d, interval_d, j-1, 1.0);
    }
  }

  mpr_set(d, interval_d, j, 0.);
  if ( mpr_get(d, interval_d, 2) != 0. ) {
    --n1;
    d = d - interval_d;//d--;
    d_add++;
    j++;
    //roughly equivelent slower version is commented out:
    //for (i = j; i >= FST_M; --i) d[i] = d[i-1];
  }
  
  nc = j-FST_M;
  //Quotient result is negative if exactly one of {ia, ib} is -1
  mpr_set(d, interval_d, 1, ia+ib ? nc : -nc); 
  ///**d[2] = a[2] - n1 -1;
  mpr_set(d, interval_d, 2, mpr_get(a, interval_a, 2) - n1 -1);
  
  ///**mpnorm(d, c, prec_words);
  gmpnorm( d, interval_d, c, interval_c, prec_words );
  
  ///**delete [] (d+d_add);
  ///**if (debug_level >= 9) print_mpreal("MPDIVD O ", c);
  
  d = d + interval_d; //we need to reset the d!!!

  return; 
}


#endif

