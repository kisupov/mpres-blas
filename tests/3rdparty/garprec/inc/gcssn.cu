
#ifndef _GMP_CSSN_CU_
#define _GMP_CSSN_CU_


/** SIN-COS table */
//double* g_sin_table; 
//double* g_cos_table;

/**
* the most complicated function in the library!!!!
* calculate the sin and cos of an input number
* x = cos
* y = sin
*/
__device__
void gmpcssn( const double* a, const int interval_a, 
			  double* x, const int interval_x, 
			  double* y, const int interval_y, 
			  int prec_words, 
			  double* d, 
			  const double* d_pi_over_256_sine_table, 
			  const double* d_pi_over_256_cosine_table) {
	const double cpi = 3.141592653589793;
	int nq = 4;
	double t1, t2;
	int n1, na, neg=0;

	/**
	if (error_no != 0) {
		if (error_no == 99) mpabrt();
		zero(x); zero(y); return;
	}
	if (debug_level >= 6) cout << "\nMPVSSN I";
	*/

	
	//na = std::min(int(std::abs(a[1])), prec_words);
	na = min(int(fabs(mpr_get(a, interval_a, 1))), prec_words);
	if(na == 0) {
		//x = 1.0, y = zero.
		//x[1] = 1.0; 
		
		if( x != NULL ) {
			mpr_set(x, interval_x, 1, 1.0); 
			//x[2] = 0.0; 
			mpr_set(x, interval_x, 2, 0.0); 
			//x[3] = 1.0;
			mpr_set(x, interval_x, 3, 1.0);
		}

		if( y != NULL ) {
			zero(y, interval_y);
		}

		return;
	}
	
	

	// Check is Pi has been precomputed.

	//mpmdc (pi, t1, n1, prec_words);
	gmpmdc(_pi, 1, t1, n1, prec_words);
	//if(n1 != 0 || std::abs(t1 - cpi) > mprx2) {
	if(n1 != 0 || fabs(t1 - cpi) > MPRX2) {
		return;
	}

	int nws = prec_words;
	prec_words++;
	int n5 = prec_words+5;
	int ka, kb, kc;
	
	double sk0[MAX_N_WORDS];
	double sk1[MAX_N_WORDS];
	double sk2[MAX_N_WORDS];
	double sk3[MAX_N_WORDS];
	double sk4[MAX_N_WORDS];
	double sk5[MAX_N_WORDS];
	double sk6[MAX_N_WORDS];
	double f[6];
	new_gmp( sk0, 0.0, n5 );
	new_gmp( sk1, 0.0, n5 );
	new_gmp( sk2, 0.0, n5 );
	new_gmp( sk3, 0.0, n5 );
	new_gmp( sk4, 0.0, n5 );
	new_gmp( sk5, 0.0, n5 );
	new_gmp( sk6, 0.0, n5 );
	new_gmp( f, 1.0, 6 ) ;
	

	//   Reduce to between -Pi and Pi.
	
	
	//mpmuld(pi, 2.0, 0, sk0, prec_words);
	gmpmuld(_pi, 1, 2.0, 0, sk0, 1, prec_words, d);
	//mpdiv(a, sk0, sk1, prec_words);
	gmpdiv(a, interval_a, sk0, 1, sk1, 1, prec_words, d);
	//mpnint(sk1, sk2, prec_words);
	gmpnint(sk1, 1, sk2, 1, prec_words, d);
	//mpsub(sk1, sk2, sk3, prec_words);
	gmpsub(sk1, 1, sk2, 1, sk3, 1, prec_words, d);

	

	// Determine rearest multiple of Pi / 2, and within a quadrant, the
	// nearest multiple of Pi / 256.  Through most of the rest of this
	// subroutine, KA and KB are the integers a and b of the algorithm
	// above.

	//mpmdc(sk3, t1, n1, prec_words);
	gmpmdc(sk3, 1, t1, n1, prec_words);
	if(n1 >= -MPNBT) {
		t1 = ldexp(t1, n1);
		t2 = 4.0 * t1;
		t1 = ganint (double(t2));
		ka = int(t1);
		kb = int(ganint (128.0 * (t2 - ka)));
	} else {
		ka = 0;
		kb = 0;
	}
	t1 = (128 * ka + kb) / 512.0;
	//mpdmc(t1, 0, sk1, prec_words);
	gmpdmc(t1, 0, sk1, 1, prec_words);
	//mpsub(sk3, sk1, sk2, prec_words);
	gmpsub(sk3, 1, sk1, 1, sk2, 1, prec_words, d);
	//mpmul(sk2, sk0, sk1, prec_words);
	gmpmul(sk2, 1, sk0, 1, sk1, 1, prec_words, d);


	// compute consine and sine of the reduced argument s.

	if(sk1[1] == 0.0) { // if sk1 == zero
		zero(sk0, 1);
		nq = 0;
	} else {
		//Divide by 2^nq (possibly), fix after series has converged.
		//if(sk1.mpr[2] < -1 || sk1.mpr[3] < mpbdx/4096.0) {
		if(sk1[2] < -1 || sk1[3] < MPBDX/4096.0) {
			nq = 0;
		} else {
			//mpdivd(sk1, 1.0, nq, sk1, prec_words);
			gmpdivd(sk1, 1, 1.0, nq, sk1, 1, prec_words, d);
		}

		// Compute the Taylor's series now.
		//mpeq(sk1, sk0, prec_words);
		gmpeq(sk1, 1, sk0, 1, prec_words);
		//mpmul(sk0, sk0, sk2, prec_words);
		gmpmul(sk0, 1, sk0, 1, sk2, 1, prec_words, d);

		int l1=0; //iteration count.
		int term_prec;

		neg = sk1[1] < 0.0 ? 1 : 0;
		do {
			l1++;
			t2 = - (2.0 * l1) * (2.0 * l1 + 1.0);

			// compute this term with term_prec words of precision only.
			term_prec = min(nws+1, nws+int(sk2[2]+sk1[2]-sk0[2])+2);
			prec_words = max(0, term_prec); 
			//mpmul(sk1, sk2, sk3, prec_words);
			gmpmul(sk1, 1, sk2, 1, sk3, 1, prec_words, d);
			//mpdivd(sk3, t2, 0, sk1, prec_words);
			gmpdivd(sk3, 1, t2, 0, sk1, 1, prec_words, d);
			prec_words = nws+1; // full precision to add term in.
			gmpadd(sk1, 1, sk0, 1, sk0, 1, prec_words, d);
			//the above line needs to change if mpadd is not safe for 
			// same variable input/output.

			// Check for convergence of the series in the loop condition
		} while(l1 < 10000 &&
			(sk1[1] != 0.0 && sk1[2] >= sk0[2] - prec_words));

		//answer needs to end up in sk0.
		if(nq) {
			// Perform double angle formulas, 
			// Cos (s) = 1 - 2 * Sin^2(s/2) = 2 * Cos^2(s/2) - 1 
			//mpmul(sk0, sk0, sk1, prec_words);
			gmpmul(sk0, 1, sk0, 1, sk1, 1, prec_words, d);
			//mpmuld(sk1, 2.0, 0, sk2, prec_words);
			gmpmuld(sk1, 1, 2.0, 0, sk2, 1, prec_words, d);
			//mpsub(f, sk2, sk0, prec_words);
			gmpsub(f, 1, sk2, 1, sk0, 1, prec_words, d);
			for(int i=1;i<nq;i++) {
				//mpmul(sk0, sk0, sk1, prec_words);
				gmpmul(sk0, 1, sk0, 1, sk1, 1, prec_words, d);
				//mpmuld(sk1, 2.0, 0, sk2, prec_words);
				gmpmuld(sk1, 1, 2.0, 0, sk2, 1, prec_words, d);
				//mpsub(sk2, f, sk0, prec_words);
				gmpsub(sk2, 1, f, 1, sk0, 1, prec_words, d);
			}
		}      
	}



	if(nq) {
		gmpmul(sk0, 1, sk0, 1, sk2, 1, prec_words, d);
		gmpsub(f, 1, sk2, 1, sk3, 1, prec_words, d);
		gmpsqrt(sk3, 1, sk1, 1, prec_words, d);
		if(neg) {
			sk1[1] = -sk1[1];
		}
	} else {
		gmpeq(sk0, 1, sk1, 1, prec_words);
		gmpmul(sk0, 1, sk0, 1, sk2, 1, prec_words, d);
		gmpsub(f, 1, sk2, 1, sk3, 1, prec_words, d);
		gmpsqrt(sk3, 1, sk0, 1, prec_words, d);
	}


	// Now sk0 holds Cos(s), sk1 holds Sin(s).

	// Compute cosine and sine of b * Pi / 512; or, 
	//   get it from the table.  

	kc = abs(kb);
	

	//mpeq(*pi_over_256_cosine_table[kc], sk2, prec_words);
	gmpeq(d_pi_over_256_cosine_table + kc, SIN_COS_TABLE_SIZE, sk2, 1, prec_words);
	//mpeq(*pi_over_256_sine_table[kc], sk3, prec_words);
	gmpeq(d_pi_over_256_sine_table + kc, SIN_COS_TABLE_SIZE, sk3, 1, prec_words);
	
	if (kb < 0) sk3[1] = -sk3[1];
	// Now sk2 holds Cos (b * Pi / 256), 
	// sk3 holds Cos (b * Pi / 256).


	// Apply the trig summation identities to compute cosine and sine
	// of s + b * Pi / 256;  

	//mpmul(sk0, sk2, sk4, prec_words);
	gmpmul(sk0, 1, sk2, 1, sk4, 1, prec_words, d);
	//mpmul(sk1, sk3, sk5, prec_words);
	gmpmul(sk1, 1, sk3, 1, sk5, 1, prec_words, d);
	//mpsub(sk4, sk5, sk6, prec_words);
	gmpsub(sk4, 1, sk5, 1, sk6, 1, prec_words, d);
	//mpmul(sk1, sk2, sk4, prec_words);
	gmpmul(sk1, 1, sk2, 1, sk4, 1, prec_words, d);
	//mpmul(sk0, sk3, sk5, prec_words);
	gmpmul(sk0, 1, sk3, 1, sk5, 1, prec_words, d);
	//mpadd(sk4, sk5, sk1, prec_words);
	gmpadd(sk4, 1, sk5, 1, sk1, 1, prec_words, d);
	//mpeq(sk6, sk0, prec_words);
	gmpeq(sk6, 1, sk0, 1, prec_words);


	// This code in effect applies the trig summation identities for
	// (s + b * Pi / 256) + a * Pi / 2.

	switch(ka) {
  case 0: 
	  //mpeq(sk0, x, prec_words);
	  if( x != NULL ) {
		gmpeq(sk0, 1, x, interval_x, prec_words);
	  }
	  //mpeq(sk1, y, prec_words);
	  if( y != NULL ) {
		gmpeq(sk1, 1, y, interval_y, prec_words);
	  }
	  break;
  case 1:
	  if( x != NULL ) {
		  //mpeq(sk1, x, prec_words);
		  gmpeq(sk1, 1, x, interval_x, prec_words);
		  //x[1] = - x[1];
		  mpr_set(x, interval_x, 1, - mpr_get(x, interval_x, 1));
	  }
	  if( y != NULL ) {
		//mpeq(sk0, y, prec_words);
		gmpeq(sk0, 1, y, interval_y, prec_words);
	  }
	  break;
  case -1:
		if( x != NULL ) {
		  //mpeq(sk1, x, prec_words);
		  gmpeq(sk1, 1, x, interval_x, prec_words);
	  }
	  
	  if( y != NULL ) {
		  //mpeq(sk0, y, prec_words);
		  gmpeq(sk0, 1, y, interval_y, prec_words);
		  //y[1] = -y[1];
		  mpr_set(y, interval_y, 1, -mpr_get(y, interval_y, 1));
	  }
	  break;
  case 2:
  case -2:
		if( x != NULL ) {
		  //mpeq(sk0, x, prec_words);
		  gmpeq(sk0, 1, x, interval_x, prec_words);
		  //x[1] = -x[1];
		  mpr_set(x, interval_x, 1, -mpr_get(x, interval_x, 1));
	  }
	  
	  if( y != NULL ) {
		  //mpeq(sk1, y, prec_words);
		  gmpeq(sk1, 1, y, interval_y, prec_words);
		  //y[1] = -y[1];
		  mpr_set(y, interval_y, 1, -mpr_get(y, interval_y, 1));
	  }
	  break;
	}

	// Restore original precision level.

	prec_words = nws;
	if( x != NULL ) {
		gmproun(x, interval_x);
	}
	
	if( y != NULL ) {
		gmproun(y, interval_y);
	}
	//if(debug_level >= 6) cout << "\nMPCSSN done : sin = "<<x<<"\t cos = "<<y;

	return;
}


__device__
void gmpsin( const double* a, const int interval_a, 
			 double* c, const int interval_c, 
			 const int prec_words, 
			 double* d, 
			 const double* d_sin_table, const double* d_cos_table ) {
	gmpcssn( a, interval_a, NULL, 0, c, interval_c, prec_words, d, 
		d_sin_table, d_cos_table );
}

__device__
void gmpcos( const double* a, const int interval_a, 
			 double* c, const int interval_c, 
			 const int prec_words, 
			 double* d, 
			 const double* d_sin_table, const double* d_cos_table ) {
	gmpcssn( a, interval_a, c, interval_c, NULL, 0, prec_words, d, 
		d_sin_table, d_cos_table );
}



#endif

