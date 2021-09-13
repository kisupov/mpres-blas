
#ifndef _GROUN_CU_
#define _GROUN_CU_

/**
	rounding routine
*/

//__device__
//void gmproun(double* a, const int interval_a, const int gPrecWords) {
__device__
void gmproun(double* a, const int interval_a) {
  int i, ia, k, na, n4, AllZero, LoopBreak;
  double a2;

  //a2 = a[2]; // exponent
  a2 = mpr_get( a, interval_a, 2 );
  //**a[2] = 0.0;
  mpr_set( a, interval_a, 2, 0.0 );
  //**ia = a[1] >= 0 ? 1 : -1; // sign (1., a(1))
  ia = mpr_get( a, interval_a, 1 ) >= 0 ? 1 : -1; // sign (1., a(1))
  //**na = min(int (abs (a[1])), gPrecWords);
  na = min((int)(fabs(mpr_get( a, interval_a, 1 ))), *gPrecWords);
  //**na = min(na, int(a[0])-5);
  na = min(na, (int)(mpr_get(a, interval_a, 0))-5);
  n4 = na + 4; //index of last addressable word.
     
  //**if (a[FST_M] == 0.) {
  if( mpr_get( a, interval_a, FST_M ) == 0. ) {
    AllZero = 1;
    for (i = 4; i <= n4; ++i) {
      //**if (a[i] != 0.0) {
	  if( mpr_get( a, interval_a, i ) != 0.0 ) {
        AllZero = 0;
        break;
      }
    }
       
    if ( AllZero ) {
      zero(a, interval_a);
      return;
    }

    k = i - FST_M; // number of leading zeros

    // !dir$ ivdep
    for (i = FST_M; i <= n4 - k; ++i){
		//**a[i] = a[i+k];
		mpr_set( a, interval_a, i, mpr_get( a, interval_a, i + k ) );
	}

    a2 = a2 - k;
    na -= max (k - 2, 0);
    if (k == 2) {
		//**a[na + FST_M] = 0.0; // BUG FIX
		mpr_set( a, interval_a, na + FST_M, 0.0 ) ;
	}
  }

  if (na == (*gPrecWords) && ROUND_DIR >= 1) {
    /**if ( (ROUND_DIR == 1 && a[na+3] >= 0.5 * mpbdx) || 
			(ROUND_DIR == 2 && a[na+3] >= 1) ) {
      a[na+2] += 1.0;
	}*/
	if ( (ROUND_DIR == 1 && mpr_get( a, interval_a, na+3 ) >= 0.5 * MPBDX) || 
         (ROUND_DIR == 2 && mpr_get( a, interval_a, na+3 ) >= 1) ) {
      //**a[na+2] += 1.0;
	  mpr_set( a, interval_a, na + 2, mpr_get( a, interval_a, na + 2 ) + 1.0 );
	}

    //**a[na+3] = a[na+4] = 0.0;
	mpr_set( a, interval_a, na + 3, 0.0 );
	mpr_set( a, interval_a, na + 4, 0.0 );

    LoopBreak = 0;
    for (i = na + 2; i >= FST_M; --i) {
      //**if (a[i] < mpbdx) {
	  if( mpr_get( a, interval_a, i ) < MPBDX ) {
        LoopBreak = 1; // goto 140
        break;
      }
      //**a[i] -= mpbdx;
	  mpr_compsub( a, interval_a, i, MPBDX );
      //**++a[i-1];
	  mpr_inc( a, interval_a, i - 1 );
    }

    if ( !LoopBreak ) {
      //**a[3] = a[2];
	  mpr_set( a, interval_a, 3, mpr_get( a, interval_a, 2 ) );
      na = 1;
      a2++;
    }
  }

  //**if (a[na+2] == 0.) {
  if( 0.0 == mpr_get( a, interval_a, na + 2 ) ) {
    AllZero = 1;
    for (i = na + 1; i >= FST_M; --i) {
      //**if (a[i] != 0.) {
	  if( 0.0 != mpr_get( a, interval_a, i ) ) {
        AllZero = 0;
        break; // goto 160
      }
    }
    if ( AllZero ) {
      zero(a, interval_a);
      return;
    }

    // 160
    na = i - 2;
    //**a[na+4] = 0.0;
	mpr_set( a, interval_a, na + 4, 0.0 );
  }

  //**if (a[FST_M] == 0.) {
  if( 0.0 == mpr_get( a, interval_a, FST_M ) ) {
    zero(a, interval_a);
  } else {
    //**a[1] = ia >= 0 ? na : -na; // sign (na, ia)
    //**a[2] = a2;
    
	mpr_set( a, interval_a, 1, ia >= 0 ? na : -na ); // sign (na, ia)
	mpr_set( a, interval_a, 2, a2 );
  }
/*
  */
}

#endif

