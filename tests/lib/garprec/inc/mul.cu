
#ifndef _GMP_MUL_CU_
#define _GMP_MUL_CU_

/**
* perform multiplication c = a x b
*/

/**
* @p d is a temperoral buffer, should be allocated outside with the size (prec_words+7)
* @p a, @p b and @p c are all INTERVAL access, but @p d is SEQUENTIAL access
*/
__device__
void gmpmul( const double* a, const int interval_a, 
			 const double* b, const int interval_b, 
			 double* c, const int interval_c, 
			 const int prec_words, 
			 double* d, const int interval_d = 1 ) {
	int i, j, j3, jd, ia, ib, na, nb, nc, n2;
	double d2, t1, t2, t[2], a_val;
	//**double* d;


	/***
	**ia = int(sign(1.0, a[1]));
	**ib = int(sign(1.0, b[1]));
	**na = std::min (int(std::abs(a[1])), prec_words);
	**nb = std::min (int(std::abs(b[1])), prec_words);
	*/
	ia = int(sign(1.0, mpr_get(a, interval_a, 1)));
	ib = int(sign(1.0, mpr_get(b, interval_b, 1)));
	na = min (int(fabs(mpr_get(a, interval_a, 1))), prec_words);
	nb = min (int(fabs(mpr_get(b, interval_b, 1))), prec_words);

	// One of the inputs is zero -- result is zero.
	if (na == 0 || nb == 0) {
	 zero(c, interval_c);
	 //**if (debug_level >= 8) print_mpreal("MPMUL O ", c);
	 return;
	}

	// One of the inputs is 1 or -1.
	//!!!!!! TO DO
	
	// ok. ready for main part of routine
	//**d = new double[prec_words+5+FST_M];  // accumulator

	//**nc = std::min(int(c.mpr[0])-5, std::min (na + nb, prec_words));
	nc = min(int(mpr_get(c, interval_c, 0))-5, min (na + nb, prec_words));
	//**d2 = a[2] + b[2]; // exponent
	d2 = mpr_get(a, interval_a, 2) + mpr_get(b, interval_b, 2); // exponent
	for (i = 1; i < prec_words+4+FST_M; ++i) {
		//d[i] = 0.0;
		mpr_set(d, interval_d, i, 0.0);
	}


	//  MPNW+5-FST+1 == (max j possible)-FST_M+1
	// mantissa words of the product.

	for (j = FST_M; j < na + FST_M; ++j) {
	 //**a_val = a[j];
	 a_val = mpr_get(a, interval_a, j);
	 j3 = j - FST_M;
	 n2 = min (nb + FST_M, prec_words + 5 - j3);

	 jd = j;
	 for(i = FST_M; i < n2; ++i) {
		 //**t[0] = mp_two_prod_positive(a_val, b[i], t[1]); 
		 t[0] = mp_two_prod_positive(a_val, mpr_get(b, interval_b, i), t[1]); 
		 // t[0], t[1] non-overlap, <= 2^mpnbt-1
		 //d[jd-1] += t[0];
		 //d[jd] += t[1];
		// +=  mpr_compadd( double* mpr, const int interval, const int wordIdx, const double value );
                 mpr_compadd(d, interval_d, jd-1, t[0]);
                 mpr_compadd(d, interval_d, jd, t[1]);

		 ++jd;
	 }

	 // Release carry to avoid overflowing the exact integer capacity
	 // (2^mpnbt-1) of a floating point word in D.
	 //**if(!((j-2) & (mp::mpnpr-1))) { // assume mpnpr is power of two
	 if(!((j-2) & (MPNPR-1))) { // assume mpnpr is power of two
		 for(i= jd-1;i>=j;i--) {
			 t1 = mpr_get(d, interval_d, i);//d[i];
			 t2 = int (t1 * MPRDX);     // carry <= 1
			 mpr_set(d, interval_d, i, t1 - t2 * MPBDX); //d[i] = t1 - t2 * MPBDX;   // remainder of t1 * 2^(-mpnbt)
			 mpr_compadd(d, interval_d, i-1, t2); //d[i-1] += t2;
		 }
	 }
	}
	int d_add = 0;

	// If D[1] is nonzero, shift the result two words right.
	if (mpr_get(d, interval_d, 1) != 0.0) {
	 // this case shouldn't really happen.
	 //**assert(0);
	 d2 += 2.0;
	 for (i = nc + 4; i >= FST_M; --i)
		 mpr_set(d, interval_d, i, mpr_get(d, interval_d, i-2)); //d[i] = d[i-2];    
	} else if (mpr_get(d, interval_d, 2) != 0.0 || 
		(mpr_get(d, interval_d, 3) >= MPBDX && (mpr_set(d, interval_d, 2, 0.0), 1))) {
	 // If D[2] is nonzero, shift the result one word right.
	 d2 += 1.0;  // exponent
	 d = d - interval_d; //d--; 
	 d_add++;
	}
	// Result is negative if one of {ia, ib} is negative.
	mpr_set(d, interval_d, 1, ia+ib ? nc : -nc);
	mpr_set(d, interval_d, 2, d2);

	//  Fix up result, since some words may be negative or exceed MPBDX.
	gmpnorm(d, interval_d, c, interval_c, prec_words);
	//**delete [] (d +  d_add);

	//**if (debug_level >= 8) print_mpreal("MPMUL O ", c);
	return;
	/*
	*/
}

#endif

