#ifndef _GMP_MDC_CU_
#define _GMP_MDC_CU_


__device__
void gmpmdc(const double* a, const int interval_a, double &b, int &n, int prec_words)
{
  /**
   * This procedure takes the mp_real A, and splits it into 
   * a double, b, and a exponent, n. 
   *
   * On exit, the following should be roughly true: 
   *
   *       a ==(roughly) b*2^n
   */

  double aa;

  /**
  if(error_no != 0) {
    if(error_no == 99) mpabrt();
    b = 0.0;
    n = 0;
    return;
  }
  if(debug_level >= 9) {
    int no = std::min(int(std::abs(a[1])), debug_words) + 2;
    cerr << "MPMDC I " << no << endl;
  }*/


  //**if(a[1] == 0.0) {
  if(mpr_get(a, interval_a, 1) == 0.0) {
    b = 0.0;
    n = 0;
    return;
  }

 //** int na = int(std::abs(a[1]));
  int na = int(fabs(mpr_get(a, interval_a, 1)));
  //**aa = a[FST_M];
  aa = mpr_get(a, interval_a, FST_M);
  //**if(na >= 2) aa += mprdx * a[FST_M+1];
  if(na >= 2) aa += MPRDX * mpr_get(a, interval_a, FST_M+1);
  //**if(na >= 3) aa += mprx2 * a[FST_M+2];
  if(na >= 3) aa += MPRX2 * mpr_get(a, interval_a, FST_M+2);
  //**if(na >= 4) aa += mprx2 * mprdx * a[FST_M+3];
  if(na >= 4) aa += MPRX2 * MPRDX * mpr_get(a, interval_a, FST_M+3);

  //**n = int(mpnbt * a[2]); 
  n = int(MPNBT * mpr_get(a, interval_a, 2)); 
  //**b = sign(aa, a[1]);
  b = sign(aa, mpr_get(a, interval_a, 1));
  
  //**if(debug_level >= 9) cerr << "MPMDC 0 " << b << ", " << n << endl;
}

#endif

