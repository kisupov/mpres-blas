/*
 *  Global precomputed constants abd helper routines for multiple-precision arithmetic
 *
 *  Copyright 2019, 2020 by Konstantin Isupov and Ivan Babeshko.
 *
 *  This file is part of the MPRES-BLAS library.
 *
 *  MPRES-BLAS is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  MPRES-BLAS is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with MPRES-BLAS.  If not, see <https://www.gnu.org/licenses/>.
 */

#ifndef MPRES_ARITH_UTILS_CUH
#define MPRES_ARITH_UTILS_CUH

#include <iostream>
#include "rns.cuh"

/********************* Global precomputed constants *********************/

int MP_PRECISION; // The precision of arithmetic operations
int MP_H; // An integer such that 2^MP_H <= SQRT(M) / M < 2^{MP_H +1}. Used in rounding
int MP_J; // An integer such that 2^MP_J <= (M - SQRT(M)) / M < 2^{MP_J +1}. Used in exponent alignment
mp_float_t MP_ZERO; //Zero in the used multiple-precision representation
mp_float_t MP_MIN; //The smallest number in MP format (largest negative value)

//Constants for GPU
namespace cuda {
    __device__ __constant__ int MP_H;
    __device__ __constant__ int MP_J;
    __device__ __constant__ mp_float_t MP_ZERO;
    __device__ __constant__ mp_float_t MP_MIN;
}

void mp_const_init() {
    //Computating MP_PRECISION
    mpfr_t mpfr_tmp;
    mpfr_init2(mpfr_tmp, 20000);
    mpfr_sqrt(mpfr_tmp, RNS_MODULI_PRODUCT_MPFR, MPFR_RNDD);
    mpfr_log2(mpfr_tmp, mpfr_tmp, MPFR_RNDD);
    MP_PRECISION = floor(mpfr_get_ui(mpfr_tmp, MPFR_RNDD)) - 1;
    // Сomputing MP_H
    mpfr_sqrt(mpfr_tmp, RNS_MODULI_PRODUCT_MPFR, MPFR_RNDD);
    mpfr_div(mpfr_tmp, mpfr_tmp, RNS_MODULI_PRODUCT_MPFR, MPFR_RNDD);
    mpfr_log2(mpfr_tmp, mpfr_tmp, MPFR_RNDD);
    MP_H = mpfr_get_si(mpfr_tmp, MPFR_RNDD);
    //Сomputing MP_J
    mpfr_sqrt(mpfr_tmp, RNS_MODULI_PRODUCT_MPFR, MPFR_RNDD);
    mpfr_sub(mpfr_tmp, RNS_MODULI_PRODUCT_MPFR, mpfr_tmp, MPFR_RNDD);
    mpfr_div(mpfr_tmp, mpfr_tmp, RNS_MODULI_PRODUCT_MPFR, MPFR_RNDD);
    mpfr_log2(mpfr_tmp, mpfr_tmp, MPFR_RNDD);
    MP_J = mpfr_get_si(mpfr_tmp, MPFR_RNDD);
    mpfr_clear(mpfr_tmp);
    //Setting MP_ZERO
    MP_ZERO.sign = 0;
    MP_ZERO.exp = 0;
    MP_ZERO.eval[0].exp = 0;
    MP_ZERO.eval[1].exp = 0;
    MP_ZERO.eval[0].frac = 0.0;
    MP_ZERO.eval[1].frac = 0.0;
    for (int i = 0; i < RNS_MODULI_SIZE; ++i) {
        MP_ZERO.digits[i] = 0;
    }
    //Setting MP_MIN
    MP_MIN.sign = 1;
    MP_MIN.exp = MP_EXP_MAX;
    for (int i = 0; i < RNS_MODULI_SIZE; ++i) {
        MP_MIN.digits[i] = RNS_MODULI[i] - 1;
    }
    rns_eval_compute(&MP_MIN.eval[0], &MP_MIN.eval[1], MP_MIN.digits);
    //Copying constants to the GPU memory
    cudaMemcpyToSymbol(cuda::MP_H, &MP_H, sizeof(int));
    cudaMemcpyToSymbol(cuda::MP_J, &MP_J, sizeof(int));
    cudaMemcpyToSymbol(cuda::MP_ZERO, &MP_ZERO, sizeof(mp_float_t));
    cudaMemcpyToSymbol(cuda::MP_MIN, &MP_MIN, sizeof(mp_float_t));
}

/*
 * Print main constants of the MP arithmetic
 */
void mp_const_print(){
    std::cout << "Constants of the RNS-based floating-point arithmetic:" << std::endl;
    printf("- MP_PRECISION: %i\n", MP_PRECISION);
    printf("- MP_H: %i\n", MP_H);
    printf("- MP_J: %i\n", MP_J);
}


/*!
 * Print the parts (fields) of a multiple-precision number
 */
void mp_print(mp_float_ptr x) {
    printf("\nMultiple-precision value:\n");
    printf("- Sign: %d\n", x->sign);
    printf("- Significand: <");
    for (int i = 0; i < RNS_MODULI_SIZE; i++)
        printf("%d ", x->digits[i]);
    printf("\b>\n");
    printf("- Exp: %i\n", x->exp);
    printf("- Eval.low: ");
    er_print(x->eval[0]);
    printf("\n- Eval.exact: ");
    er_float_t er_temp;
    rns_fractional(&er_temp, x->digits);
    er_print(er_temp);
    printf("\n- Eval.upp: ");
    er_print(x->eval[1]);
    printf("\n\n");
}


/********************* Helper routines for multiple-precision arithmetic *********************/

/*
 * Returns the number of rounding bits for x
 */
GCC_FORCEINLINE int mp_get_rnd_bits(mp_float_ptr x) {
#if defined(DEBUG) || defined(_DEBUG)
    if(x->eval[1].exp != x->eval[0].exp){
        printf("\n [CPU WARNING] Possible loss of accuracy");
    }
#endif
    return x->eval[1].exp - MP_H + 1;
}

/*
 * Rounding x by n bits
 */
GCC_FORCEINLINE void mp_round(mp_float_ptr x, int n) {
    if (n > 0) {
        x->exp = x->exp + n;
        rns_scale2pow(x->digits, x->digits, (unsigned) n);
        //After rounding, the significand will be small enough,
        //so the interval evaluation can be computed faster.
        rns_eval_compute_fast(&x->eval[0], &x->eval[1], x->digits);
    }
}

/*
 * Sign identification in addition and subtraction operations when an ambiguous case occurs.
 * To compare the aligned significands, the mixed-radix conversion is used.
 * */
GCC_FORCEINLINE int sign_estimate(const int * digx, const int * digy, const int sx, const int sy,
                                  const int gamma, const int theta,  const unsigned char nzx, const unsigned char nzy){
    int lx[RNS_MODULI_SIZE];
    int ly[RNS_MODULI_SIZE];
    for(int i = 0; i < RNS_MODULI_SIZE; i++){
        lx[i] = mod_mul(digx[i], RNS_POW2[gamma][i] * nzx, RNS_MODULI[i]);
        ly[i] = mod_mul(digy[i], RNS_POW2[theta][i] * nzy, RNS_MODULI[i]);
    }
    int cmp = mrc_compare_rns(lx, ly);
    return (cmp < 0 ? sy : sx) * (cmp != 0);
}

/*
 * GPU functions
 */
namespace cuda {

    /*
   * Returns the number of rounding bits for x
   */
    DEVICE_CUDA_FORCEINLINE int mp_get_rnd_bits(mp_float_ptr x) {
        #if defined(DEBUG) || defined(_DEBUG)
        if(x->eval[1].exp != x->eval[0].exp){
            printf("\n [CUDA WARNING] Possible loss of accuracy");
         }
        #endif
        return x->eval[1].exp - cuda::MP_H + 1;
    }

    /*
     * Rounding x by n bits
     */
    DEVICE_CUDA_FORCEINLINE void mp_round(mp_float_ptr x, int n) {
        while (n > 0) {
            x->exp += n;
            cuda::rns_scale2pow(x->digits, x->digits, (unsigned int) n);
            //After rounding, the significand will be small enough,
            //so the interval evaluation can be computed faster.
            cuda::rns_eval_compute_fast(&x->eval[0], &x->eval[1], x->digits);
            n = -1;
        }
    }

    /*
     * Sign identification in addition and subtraction operations when an ambiguous case occurs.
     * To compare the aligned significands, the mixed-radix conversion is used.
     */
    DEVICE_CUDA_FORCEINLINE int sign_estimate(const int * digx, const int * digy, const int sx, const int sy,
                                              const int gamma, const int theta,  const int nzx, const int nzy){
        int lx[RNS_MODULI_SIZE];
        int ly[RNS_MODULI_SIZE];
        cuda::rns_mul_c(lx, digx, cuda::RNS_POW2[gamma], nzx);
        cuda::rns_mul_c(ly, digy, cuda::RNS_POW2[theta], nzy);
        int cmp = cuda::mrc_compare_rns(lx, ly);
        return (cmp < 0 ? sy : sx) * (cmp != 0);
    }

} //namespace cuda

#endif //MPRES_ARITH_UTILS_CUH
