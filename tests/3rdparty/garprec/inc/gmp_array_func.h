

#ifndef __GMP_ARRAY_FUNC_H__
#define __GMP_ARRAY_FUNC_H__

#include "gmp_array.h"
#include "garprec_init.h"


void gmparray_add(gmp_array *d_a, gmp_array *d_b, gmp_array *d_c, unsigned int numBlock = 2400,
                  unsigned int numThread = 128);

void gmparray_mul(gmp_array *d_a, gmp_array *d_b, gmp_array *d_c, unsigned int numBlock = 2400,
                  unsigned int numThread = 128);

void gmparray_exp(gmp_array *gmp_in, gmp_array *gmp_out, unsigned int numBlock = 2400, unsigned int numThread = 128);

void gmparray_log(gmp_array *d_a, gmp_array *d_b, unsigned int numBlock, unsigned int numThread);

void gmparray_log_alldevice(gmp_array *d_a, gmp_array *d_b, unsigned int numBlock, unsigned int numThread);

void gmparray_sqrt(gmp_array *d_a, gmp_array *d_b, unsigned int numBlock = 2400, unsigned int numThread = 128);

void gmparray_div(gmp_array *d_a, gmp_array *d_b, gmp_array *d_c, unsigned int numBlock = 2400,
                  unsigned int numThread = 128);

void gmparray_div_device(gmp_array *d_a, gmp_array *d_b, gmp_array *d_c, unsigned int numBlock = 2400,
                         unsigned int numThread = 128);

void gmparray_sin(gmp_array *gmp_in, gmp_array *gmp_out, unsigned int numBlock = 2400, unsigned int numThread = 64);

void gmparray_sin_alldevice(gmp_array *d_a, gmp_array *d_b, unsigned int numBlock, unsigned int numThread);

void gmparray_cos_alldevice(gmp_array *d_a, gmp_array *d_b, unsigned int numBlock, unsigned int numThread);

void gmparray_tan_alldevice(gmp_array *d_a, gmp_array *d_b, unsigned int numBlock, unsigned int numThread);

///////////////////////////
// test code
//

void gmparray_sqrt_registers(gmp_array *d_a, gmp_array *d_b, unsigned int numBlock, unsigned int numThread);

void gmparray_sqrt_shared(gmp_array *d_a, gmp_array *d_b, unsigned int numBlock, unsigned int numThread);

void gmparray_sqrt_alldevice(gmp_array *d_a, gmp_array *d_b, unsigned int numBlock, unsigned int numThread);

void gmparray_exp_alldevice(gmp_array *d_a, gmp_array *d_b, unsigned int numBlock = 2400, unsigned int numThread = 64);

void gmparray_exp_shared(gmp_array *d_a, gmp_array *d_b, unsigned int numBlock = 2400, unsigned int numThread = 64);

void gmparray_exp_registers(gmp_array *d_a, gmp_array *d_b, unsigned int numBlock, unsigned int numThread);

#endif /* __GMP_ARRAY_FUNC_H__ */ 

