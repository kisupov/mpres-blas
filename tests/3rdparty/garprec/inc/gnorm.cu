
#ifndef _GNORM_CU_
#define _GNORM_CU_

/**
* normalization
*/


/**
* @param d the temperal buffer, SEQUENTIAL access
* @param a the normalized result, INTERVAL access
* @param interval_a
* @param prec_words
*/

__device__
void gmpnorm( double* d, 
			 double* a, const int interval_a, 
			 const int prec_words ) {
	double a2, t1, t2, t3;
	int i, ia, na, nd, n4;

	ia = (int)(sign(1.0, d[1]));
	nd = abs((int)(d[1]));
	na = min(nd, prec_words);
	//na = min(na, ((int)(a[0]) - 5)); // do not exceed the allocated memory
	na = min(na, ((int)(mpr_get(a, interval_a, 0)) - 5)); // do not exceed the allocated memory
	if (na == 0) {
		zero(a, interval_a);
		return;
	}
	n4 = na + 4; 
	a2 = d[2];

	
	// Carry release loop.
	t1 = 0.0;
	for (i = n4; i >= FST_M; --i) {
	 t3 = t1 + d[i];  // exact 
	 //**t2 = t3 * mprdx;
	 t2 = t3 * MPRDX;
	 t1 = int (t2);   // carry <= 1
	 if ( (t2 < 0.0) && (t1 != t2) ) // make sure d[i] will be >= 0.0
		 t1 -= 1.0;
	 //**a[i] = t3 - t1 * mpbdx;
	 mpr_set( a, interval_a, i, t3 - t1*MPBDX );
	}
	//**a[2] = t1;
	mpr_set( a, interval_a, 2, t1 );

	//**if ( a[2] < 0.0 ) {
	if( mpr_get(a, interval_a, 2) < 0.0 ) {
		ia = -ia;
		
		for (i = 2; i <= n4; ++i) {
			//**a[i] = -a[i];
			mpr_set( a, interval_a, i, -(mpr_get(a, interval_a, i)) );
		}
		
		for (i = n4; i >= FST_M; --i) {
			 //**if ( a[i] < 0 ) {
			 if( mpr_get(a, interval_a, i) < 0 ) {
				//**a[i] = mpbdx + a[i]; // <= 2^50-1
				//**a[i-1] -= 1.0;
				
				mpr_set( a, interval_a, i, MPBDX + mpr_get(a, interval_a, i) );
				mpr_set( a, interval_a, i - 1, mpr_get( a, interval_a, i - 1 ) - 1.0 );
			 }
		}
	}

	//**if ( a[2] > 0.0 ) {
	if( mpr_get(a, interval_a, 2) > 0.0 ) {
	 //**if (na != prec_words && na < (int)(a[0]) - 5) {
	 if ((na != prec_words) && (na < (int)(mpr_get( a, interval_a, 0 )) - 5)) {
		 for (i = n4+1; i >= FST_M; --i) {
			//**a[i] = a[i-1];
			mpr_set( a, interval_a, i, mpr_get(a, interval_a, i - 1) );
		}
		 na = min(na+1, prec_words);
		 a2 += 1;
	 } else {
		 for (i = n4; i >= FST_M; --i) {
			//**a[i] = a[i-1];
			mpr_set( a, interval_a, i, mpr_get(a, interval_a, i - 1) );
		}
		 a2 += 1;
	 }
	}

	// Perform rounding and truncation.
	//**a[1] = ia >= 0 ? na : -na;
	//**a[2] = a2;
	mpr_set( a, interval_a, 1, ia >= 0 ? na : -na );
	mpr_set( a, interval_a, 2, a2 );
	
	gmproun(a, interval_a);
	return;
}


__device__
void gmpnorm( double* d, const int interval_d, double* a, const int interval_a, const int prec_words ) {
	double a2, t1, t2, t3;
	int i, ia, na, nd, n4;

	ia = (int)(sign(1.0, mpr_get(d, interval_d, 1)));
	nd = abs((int)(mpr_get(d, interval_d, 1)));
	na = min(nd, prec_words);
	//na = min(na, ((int)(a[0]) - 5)); // do not exceed the allocated memory
	na = min(na, ((int)(mpr_get(a, interval_a, 0)) - 5)); // do not exceed the allocated memory
	if (na == 0) {
		zero(a, interval_a);
		return;
	}
	n4 = na + 4; 
	a2 = mpr_get(d, interval_d, 2);

	
	// Carry release loop.
	t1 = 0.0;
	for (i = n4; i >= FST_M; --i) {
	 t3 = t1 + mpr_get(d, interval_d, i);  // exact 
	 //**t2 = t3 * mprdx;
	 t2 = t3 * MPRDX;
	 t1 = int (t2);   // carry <= 1
	 if ( (t2 < 0.0) && (t1 != t2) ) // make sure d[i] will be >= 0.0
		 t1 -= 1.0;
	 //**a[i] = t3 - t1 * mpbdx;
	 mpr_set( a, interval_a, i, t3 - t1*MPBDX );
	}
	//**a[2] = t1;
	mpr_set( a, interval_a, 2, t1 );

	//**if ( a[2] < 0.0 ) {
	if( mpr_get(a, interval_a, 2) < 0.0 ) {
		ia = -ia;
		
		for (i = 2; i <= n4; ++i) {
			//**a[i] = -a[i];
			mpr_set( a, interval_a, i, -(mpr_get(a, interval_a, i)) );
		}
		
		for (i = n4; i >= FST_M; --i) {
			 //**if ( a[i] < 0 ) {
			 if( mpr_get(a, interval_a, i) < 0 ) {
				//**a[i] = mpbdx + a[i]; // <= 2^50-1
				//**a[i-1] -= 1.0;
				
				mpr_set( a, interval_a, i, MPBDX + mpr_get(a, interval_a, i) );
				mpr_set( a, interval_a, i - 1, mpr_get( a, interval_a, i - 1 ) - 1.0 );
			 }
		}
	}

	//**if ( a[2] > 0.0 ) {
	if( mpr_get(a, interval_a, 2) > 0.0 ) {
	 //**if (na != prec_words && na < (int)(a[0]) - 5) {
	 if ((na != prec_words) && (na < (int)(mpr_get( a, interval_a, 0 )) - 5)) {
		 for (i = n4+1; i >= FST_M; --i) {
			//**a[i] = a[i-1];
			mpr_set( a, interval_a, i, mpr_get(a, interval_a, i - 1) );
		}
		 na = min(na+1, prec_words);
		 a2 += 1;
	 } else {
		 for (i = n4; i >= FST_M; --i) {
			//**a[i] = a[i-1];
			mpr_set( a, interval_a, i, mpr_get(a, interval_a, i - 1) );
		}
		 a2 += 1;
	 }
	}

	// Perform rounding and truncation.
	//**a[1] = ia >= 0 ? na : -na;
	//**a[2] = a2;
	mpr_set( a, interval_a, 1, ia >= 0 ? na : -na );
	mpr_set( a, interval_a, 2, a2 );
	
	gmproun(a, interval_a);
	return;
}

#endif

