#ifndef __GARPREC_INIT_CU__
#define __GARPREC_INIT_CU__


#include "garprec_init.h"
#include "common.cu"

void initConstants();
void initCssn(gmp_array** gmp_sin_table, gmp_array** gmp_cos_table);


int garprecInit(const unsigned int numDigit, const int device) {
	mp_real::mp_init(numDigit);

	prec_words = mp_real::prec_words;

	//init device
	//cudaSetDevice(device);

	//init constant
	initConstants();

	//init sin and cos table	
	initCssn(&gmp_sin_table, &gmp_cos_table);

	return MAX_PREC_WORDS;
}

void garprecFinalize() {
	mp_real::mp_finalize();
    cudaDeviceReset();
    //cudaThreadExit();
}


void initCssn(gmp_array** gmp_sin_table, gmp_array** gmp_cos_table) {
	int prec_words = mp_real::prec_words;

	mp_real* pi_over_256_sine_table = new mp_real[SIN_COS_TABLE_SIZE];
	mp_real* pi_over_256_cosine_table = new mp_real[SIN_COS_TABLE_SIZE];
	
	prec_words++;
	int n5 = prec_words+5;
	mp_real sk0(0.0, n5), sk1(0.0, n5), sk2(0.0, n5), sk3(0.0, n5);
	mp_real sk4(0.0, n5), sk5(0.0, n5), sk6(0.0, n5), f(1.0, 6);

	for( int kc = 0; kc < SIN_COS_TABLE_SIZE; kc++ ) {
		f[FST_M] = 2.0; // f = 2.0

		if(kc == 0) {
			sk2[1] = 1.0; sk2[2] = 0.0; sk2[3] = 1.0;// sk2 = 1.0;
			mp_real::zero(sk3);
		} else {
			switch(kc % 8) {
	  case 0:
		  //sk4 = 2.0 == 2*cos(0) 
		  sk4[1] = 1.0; sk4[2] = 0.0; sk4[3] = 2.0;
		  break;
	  case 7:
	  case 1:
		  mp_real::mpsqrt(f, sk4, prec_words);
		  mp_real::mpadd(f, sk4, sk5, prec_words);
		  mp_real::mpsqrt(sk5, sk4, prec_words);
		  break;
	  case 6:
	  case 2:
		  mp_real::mpsqrt(f, sk4, prec_words);
		  break;
	  case 5: 
	  case 3:
		  mp_real::mpsqrt(f, sk4, prec_words);
		  mp_real::mpsub(f, sk4, sk5, prec_words);
		  mp_real::mpsqrt(sk5, sk4, prec_words);
		  break;
	  case 4:
		  mp_real::zero(sk4);
			}
			// if kc * Pi/8 is on the negative half of the unit circle...
			if(((kc+4)/8) & 0x1) sk4[1] = -sk4[1];
			// now sk4 holds 2 * Cos (kc * Pi / 8)
			mp_real::mpadd(f, sk4, sk5, prec_words);
			mp_real::mpsqrt(sk5, sk4, prec_words);
			if(((kc+8)/16) & 0x1) sk4[1] = -sk4[1];
			// now sk4 holds 2 * Cos (kc * Pi / 16)
			mp_real::mpadd(f, sk4, sk5, prec_words);
			mp_real::mpsqrt(sk5, sk4, prec_words);
			if(((kc+16)/32) & 0x1) sk4[1] = -sk4[1];
			// now sk4 holds 2 * Cos (kc * Pi / 32)
			mp_real::mpadd(f, sk4, sk5, prec_words);
			mp_real::mpsqrt(sk5, sk4, prec_words);
			if(((kc+32)/64) & 0x1) sk4[1] = -sk4[1];
			// now sk4 holds 2 * Cos (kc * Pi / 64)

			mp_real::mpadd(f, sk4, sk5, prec_words);
			mp_real::mpsqrt(sk5, sk4, prec_words);
			// now sk4 holds 2 * Cos (kc * Pi / 128)

			// do for all kc != 0 
			mp_real::mpadd(f, sk4, sk5, prec_words);
			mp_real::mpsqrt(sk5, sk3, prec_words);
			mp_real::mpmuld(sk3, 0.5, 0, sk2, prec_words);
			mp_real::mpsub(f, sk4, sk5, prec_words);
			mp_real::mpsqrt(sk5, sk4, prec_words);
			mp_real::mpmuld(sk4, 0.5, 0, sk3, prec_words);
		}
		pi_over_256_sine_table[kc] = sk3;
		pi_over_256_cosine_table[kc] = sk2;
	}
	
	/** copy to the device */
	(*gmp_sin_table) = new gmp_array( pi_over_256_sine_table, SIN_COS_TABLE_SIZE );
	(*gmp_cos_table) = new gmp_array( pi_over_256_cosine_table, SIN_COS_TABLE_SIZE );
	d_sin_table = (*gmp_sin_table)->d_mpr;
	d_cos_table = (*gmp_cos_table)->d_mpr;
}


void initConstants() {
        int size = 0;
	const int prec_words = mp_real::prec_words;

        //eps
	double* eps = mp_real::_eps.mpr;
        size = (int)eps[0];
        assert( size < 2*MAX_N_WORDS );
        cudaMemcpyToSymbol( _eps, eps, sizeof(double)*size );

        //1.0
	double* num_1 = mp_real(1.0).mpr;
        size = (int)num_1[0];
        assert( size < 2*MAX_N_WORDS );
        cudaMemcpyToSymbol( _1, num_1, sizeof(double)*size );

        //0.5*_eps
	double* eps2 = (0.5*mp_real::_eps).mpr;
        size = (int)eps2[0];
        assert( size < 2*MAX_N_WORDS );
        cudaMemcpyToSymbol( _eps2, eps2, sizeof(double)*size );

        //pi
	double* pi = mp_real::_pi.mpr;
        size = (int)pi[0];
        assert( size < 2*MAX_N_WORDS );
        cudaMemcpyToSymbol( _pi, pi, sizeof(double)*size );

        //log2
	double* log2 = mp_real::_log2.mpr;
        size = (int)log2[0];
        assert( size < 2*MAX_N_WORDS );
        cudaMemcpyToSymbol( _log2, log2, sizeof(double)*size );

        //log10
	double* log10 = mp_real::_log10.mpr;
        size = (int)log10[0];
        assert( size < 2*MAX_N_WORDS );
        cudaMemcpyToSymbol( _log10, log10, sizeof(double)*size );

        //the global prec_words
        cudaMemcpyToSymbol( gPrecWords, &(prec_words), sizeof(int) );
}


#endif /* __GARPREC_INIT_CU__ */


