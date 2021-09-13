

#ifndef _GMP_SUB_CU_
#define _GMP_SUB_CU_



/**
* perform the addition c = a - b
*/

/**
* @p d is a temperoral buffer, should be allocated outside with the size (prec_words+9)
* @p a, @p b and @p c are all INTERVAL access, but @p d is SEQUENTIAL access
*/
__device__
void gmpsub( const double* a, const int interval_a, 
			 double* b, const int interval_b, 
			 double* c, const int interval_c, 
			 const int prec_words, 
			 double* d, const int interval_d = 1 ) {
  int i, BreakLoop;
  double b1;

  /*
  if (error_no != 0) {
    if (error_no == 99)  mpabrt();
    zero(c);
    return;
  }
  
  if (debug_level >= 9) cerr << " MPSUB" << endl;*/

  // Check if A = B.  This is necessary because A and B might be same array,
  // in which case negating B below won't work.

  // check if A == B points to the same object 
  //**if(&a == &b) {
  if(&(a[0]) == &(b[0])) {
    zero(c, interval_c);
    //**if(debug_level >= 9) print_mpreal("MPSUB O ", c);
    return;
  }
  
  // check if their exponent and mantissas are the same
  //**if (a[1] == b[1]) {
  if (mpr_get(a, interval_a, 1) == mpr_get(b, interval_b, 1)) {
    BreakLoop = 0;
    //**for (i = 2; i < int(std::abs(a[1])) + FST_M; ++i) {
    for (i = 2; i < int(fabs(mpr_get(a, interval_a, 1))) + FST_M; ++i) {
      //**if (a[i] != b[i]) {
      if (mpr_get(a, interval_a, i) != mpr_get(b, interval_b, i)) {
        BreakLoop = 1;
        break;
      }
    }
    if (!BreakLoop) {
      zero(c, interval_c);
      //**if(debug_level >= 9) print_mpreal("MPSUB O ", c);
      return;
    }
  }
  
  // Save the sign of B, and then negate B.
  //**b1 = b[1];
  b1 = mpr_get(b, interval_b, 1);
  double *temp; // use temp to keep const modifier
  temp = b;
  //**temp[1] = -b1;
  mpr_set(temp, interval_b, 1, -b1);
  

  // Perform addition and restore the sign of B.
  gmpadd(a, interval_a, b, interval_b, c, interval_c, prec_words, d, interval_d);
  
  // When restoring the sign of b, we must make sure that
  // b and c were not the same object.  if they were,
  // then b was overwriten, and c already contains the correct
  // result.
  if(&(b[0]) != &(c[0])) {
     //**temp[1] = b1;
     mpr_set(temp, interval_b, 1, b1);
  }

  return;	
}

#endif
