/*
 *  Inter-modulo (non-modular) computations in the Residue Number System.
 *  Algorithms for CPU and GPU are presented for computing the interval evaluation of the fractional representation
 *  of an RNS number, power-of-two RNS scaling, magnitude comparison, and mixed-radix conversion. For details, see
 *  https://dx.doi.org/10.1109/ACCESS.2020.2982365 (Interval evaluation in RNS, Magnitude comparison)
 *  http://dx.doi.org/10.1109/ICEnT.2017.36 (Power-of-two RNS scaling)
 *  Szabo, Tanaka, Residue Arithmetic and its Application to Computer Technology (Mixed-radix conversion)
 *
 *  Copyright 2018, 2019 by Konstantin Isupov and Alexander Kuvaev.
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

#ifndef MPRES_RNS_CUH
#define MPRES_RNS_CUH

#include <algorithm>
#include "gmp.h"
#include "mpfr.h"
#include "extrange.cuh"
#include "modular.cuh"

#define RNS_P2_SCALING_FACTOR 1 << RNS_P2_SCALING_THRESHOLD // RNS power-of-two scaling factor

/********************* Global precomputed constants *********************/

mpz_t  RNS_MODULI_PRODUCT; // Product of all RNS moduli, M = m_1 * m_2 * ... * m_n
mpfr_t RNS_MODULI_PRODUCT_MPFR; // Product of all RNS moduli in the MPFR type, M = m_1 * m_2 * ... * m_n
mpz_t  RNS_PART_MODULI_PRODUCT[RNS_MODULI_SIZE]; // Partial products of moduli, M_i = M / m_i (i = 1,...,n)
int    RNS_PART_MODULI_PRODUCT_INVERSE[RNS_MODULI_SIZE]; // Modulo m_i multiplicative inverses of M_i (i = 1,...,n)
mpz_t  RNS_ORTHOGONAL_BASE[RNS_MODULI_SIZE]; // Orthogonal bases of the RNS, B_i = M_i * RNS_PART_MODULI_PRODUCT_INVERSE[i] (i = 1,...,n)

/*
 * Residue codes of 2^j (j = 0,....,RNS_MODULI_PRODUCT_LOG2).
 * Each j-th row contains ( 2^j mod m_1, 2^j mod m_2, 2^j mod m_3 ... )
 * Memory addressing is as follows:
 *      RNS_POW2[0] -> residue code of 2^0
 *      RNS_POW2[1] -> residue code of 2^1
 *      RNS_POW2[RNS_MODULI_PRODUCT_LOG2] -> residue code of 2^RNS_MODULI_PRODUCT_LOG2
 */
int RNS_POW2[RNS_MODULI_PRODUCT_LOG2+1][RNS_MODULI_SIZE];

/*
 * Residues of the RNS moduli product (M) modulo 2^j (j = 1,...,RNS_P2_SCALING_THRESHOLD)
 * This constant is used to power-of-two RNS scaling.
 * Memory addressing is as follows:
 *      RNS_MODULI_PRODUCT_POW2_RESIDUES[0] -> M mod 2^1
 *      RNS_MODULI_PRODUCT_POW2_RESIDUES[1] -> M mod 2^2
 *      RNS_MODULI_PRODUCT_POW2_RESIDUES[RNS_P2_SCALING_THRESHOLD-1] -> M mod 2^RNS_P2_SCALING_THRESHOLD
 */
int RNS_MODULI_PRODUCT_POW2_RESIDUES[RNS_P2_SCALING_THRESHOLD];

/*
 * Matrix of residues of the partial RNS moduli products (M_i) modulo 2^j (j = 1,...,RNS_P2_SCALING_THRESHOLD)
 * Each j-th row contains ( M_1 mod 2^j, M_2 mod 2^j, M_3 mod 2^j ... )
 * This constant is used to power-of-two RNS scaling.
 * Memory addressing is as follows:
 *      RNS_PART_MODULI_PRODUCT_POW2_RESIDUES[0] -> M_i mod 2^1
 *      RNS_PART_MODULI_PRODUCT_POW2_RESIDUES[1] -> M_i mod 2^2
 *      RNS_PART_MODULI_PRODUCT_POW2_RESIDUES[RNS_P2_SCALING_THRESHOLD-1] -> M_i mod 2^RNS_P2_SCALING_THRESHOLD
 */
int RNS_PART_MODULI_PRODUCT_POW2_RESIDUES[RNS_P2_SCALING_THRESHOLD][RNS_MODULI_SIZE];

/*
 * Array of multiplicative inverses of 2^1, 2^2, ..., 2^RNS_P2_SCALING_THRESHOLD modulo m_i (i = 1,...,n)
 * Each j-th row contains ( (2^j)^-1 mod m_1, (2^j)^-1 mod m_2, (2^j)^-1 mod m_3 ... )
 * This constant is used to power-of-two RNS scaling.
 * Memory addressing is as follows:
 *      RNS_POW2_INVERSE[0] -> 2^-1 mod m_i
 *      RNS_POW2_INVERSE[1] -> 4^-1 mod m_i
 *      RNS_POW2_INVERSE[RNS_P2_SCALING_THRESHOLD-1] -> (2^RNS_P2_SCALING_THRESHOLD)^{-1} mod m_i
 */
int RNS_POW2_INVERSE[RNS_P2_SCALING_THRESHOLD][RNS_MODULI_SIZE];

/*
 * Moduli reciprocals, rounded directly
 */
double RNS_MODULI_RECIP_RD[RNS_MODULI_SIZE];
double RNS_MODULI_RECIP_RU[RNS_MODULI_SIZE];

/*
 * Constants for computing the interval evaluation of an RNS number
 */
double RNS_EVAL_ACCURACY; // Accuracy constant for computing the RNS interval evaluation. RNS_EVAL_ACCURACY = 4*u*n*log_2(n)*(1+RNS_EVAL_RELATIVE_ERROR/2)/RNS_EVAL_RELATIVE_ERROR, where u is the unit roundoff,
int RNS_EVAL_REF_FACTOR; // Refinement factor for computing the RNS interval evaluation
interval_t RNS_EVAL_UNIT; // Interval approximation of 1 / M
interval_t RNS_EVAL_INV_UNIT; // Interval approximation of (M - 1) / M
er_float_t RNS_EVAL_ZERO_BOUND = (er_float_t) {0.0, 0}; // To set the zero interval evaluation

/*
 * Mixed-radix conversion (MRC) constants
 */
int MRC_MULT_INV[RNS_MODULI_SIZE][RNS_MODULI_SIZE]; // Triangle matrix with elements | mult.inv(m_i) |m_j

/*
 * Constants for GPU
 */
namespace cuda {
    __device__ __constant__ int RNS_PART_MODULI_PRODUCT_INVERSE[RNS_MODULI_SIZE];
    __device__ int RNS_POW2[RNS_MODULI_PRODUCT_LOG2+1][RNS_MODULI_SIZE];
    __device__ int RNS_MODULI_PRODUCT_POW2_RESIDUES[RNS_P2_SCALING_THRESHOLD];
    __device__ int RNS_PART_MODULI_PRODUCT_POW2_RESIDUES[RNS_P2_SCALING_THRESHOLD][RNS_MODULI_SIZE];
    __device__ int RNS_POW2_INVERSE[RNS_P2_SCALING_THRESHOLD][RNS_MODULI_SIZE];
    __device__ __constant__ double RNS_EVAL_ACCURACY;
    __device__ __constant__ int RNS_EVAL_REF_FACTOR;
    __device__ __constant__  interval_t RNS_EVAL_UNIT;
    __device__ __constant__  interval_t RNS_EVAL_INV_UNIT;
    __device__ __constant__ er_float_t RNS_EVAL_ZERO_BOUND;
    __device__ int MRC_MULT_INV[RNS_MODULI_SIZE][RNS_MODULI_SIZE];
    __device__ double RNS_MODULI_RECIP_RD[RNS_MODULI_SIZE];
    __device__ double RNS_MODULI_RECIP_RU[RNS_MODULI_SIZE];
}


/********************* Helper functions *********************/


/*
 * Computes 2^p
 */
static unsigned int pow2i(int p) {
    unsigned int pow = 1;
    for (int i = 0; i < p; i++)
        pow = (pow * 2);
    return pow;
}

/*
 * Finds modulo m inverse of x using non-recursive extended Euclidean Algorithm
 */
static int mod_inverse(long x, long m){
    long a = x;
    long b = m;
    long result = 0;
    long temp1 = 1;
    long temp2 = 0;
    long temp3 = 0;
    long temp4 = 1;
    long q;

    while(1) {
        q = a / b;
        a = a % b;
        temp1 = temp1 - q * temp2;
        temp3 = temp3 - q * temp4;
        if (a == 0) {
            result = temp2;
            break;
        }
        q = b / a;
        b = b % a;
        temp2 = temp2 - q * temp1;
        temp4 = temp4 - q * temp3;
        if (b == 0) {
            result = temp1;
            break;
        }
    }
    result = (result % m + m) % m;
    return (int) result;
}

/*
 *  Finds modulo m inverse of x using non-recursive extended Euclidean Algorithm,
 *  multiple-precision version
 */
static int mod_inverse_mpz(mpz_t x, int m) {
    mpz_t a;
    mpz_init(a);
    mpz_set(a, x);

    mpz_t b;
    mpz_init(b);
    mpz_set_si(b, m);

    mpz_t result;
    mpz_init(result);
    mpz_set_ui(result, 0);

    mpz_t temp0;
    mpz_init(temp0);

    mpz_t temp1;
    mpz_init(temp1);
    mpz_set_ui(temp1, 1);

    mpz_t temp2;
    mpz_init(temp2);
    mpz_set_ui(temp2, 0);

    mpz_t temp3;
    mpz_init(temp3);
    mpz_set_ui(temp3, 0);

    mpz_t temp4;
    mpz_init(temp4);
    mpz_set_ui(temp4, 1);

    mpz_t q;
    mpz_init(q);

    while(1) {
        mpz_fdiv_q(q, a, b);
        mpz_mod(a, a, b);

        // temp1 = temp1 - q * temp2
        mpz_mul(temp0, q, temp2);
        mpz_sub(temp1, temp1, temp0);

        //temp3 = temp3 - q * temp4;
        mpz_mul(temp0, q, temp4);
        mpz_sub(temp3, temp3, temp0);

        if (mpz_cmp_ui(a, 0) == 0){
            mpz_set(result, temp2);
            break;
        }
        mpz_fdiv_q(q, b, a);
        mpz_mod(b, b, a);

        //temp2 = temp2 - q * temp1;
        mpz_mul(temp0, q, temp1);
        mpz_sub(temp2, temp2, temp0);

        //temp4 = temp4 - q * temp3;
        mpz_mul(temp0, q, temp3);
        mpz_sub(temp4, temp4, temp0);

        if (mpz_cmp_ui(b, 0) == 0){
            mpz_set(result, temp1);
            break;
        }
    }
    mpz_mod_ui(result, result, m);
    mpz_add_ui(result, result, m);
    mpz_mod_ui(result, result, m);
    long inverse = mpz_get_ui(result);

    mpz_clear(a);
    mpz_clear(b);
    mpz_clear(result);
    mpz_clear(temp0);
    mpz_clear(temp1);
    mpz_clear(temp2);
    mpz_clear(temp3);
    mpz_clear(temp4);
    mpz_clear(q);
    return (int) inverse;
}

/*
 * Set RNS number target from another RNS number x
 */
GCC_FORCEINLINE void rns_set(int *target, int * x) {
    memcpy(target, x, sizeof(int) * RNS_MODULI_SIZE);
}

/*
 * Converts x from binary system to RNS
 * The result is stored in target
 */
GCC_FORCEINLINE void rns_from_binary(int * target, mpz_t x) {
    mpz_t residue;
    mpz_init(residue);
    for (int i = 0; i < RNS_MODULI_SIZE; i++) {
        mpz_mod_ui(residue, x, RNS_MODULI[i]);
        target[i] = mpz_get_ui(residue);
    }
    mpz_clear(residue);
}

/*
 * Converts x from RNS to binary system using Chinese remainder theorem.
 * The result is stored in target
 */
GCC_FORCEINLINE void rns_to_binary(mpz_t target, int * x) {
    mpz_t term;
    mpz_init(term);
    mpz_set_ui(target, 0);
    for (int i = 0; i < RNS_MODULI_SIZE; i++) {
        mpz_mul_ui(term, RNS_ORTHOGONAL_BASE[i], x[i]);
        mpz_add(target, target, term);
    }
    mpz_mod(target, target, RNS_MODULI_PRODUCT);
    mpz_clear(term);
}

/*
 * Computing the EXACT fractional representation of x (i.e. x/M) using CRT
 */
GCC_FORCEINLINE void rns_fractional(er_float_ptr result, int * x) {
    mpfr_t mpfr;
    mpfr_init(mpfr);
    mpz_t m;
    mpz_init(m);
    rns_to_binary(m, x);
    mpfr_set_z(mpfr, m, MPFR_RNDN);
    mpfr_div(mpfr, mpfr, RNS_MODULI_PRODUCT_MPFR, MPFR_RNDN);
    result->frac = mpfr_get_d_2exp((long *) &result->exp, mpfr, MPFR_RNDN);
    mpfr_clear(mpfr);
    mpz_clear(m);
}


/********************* Functions for calculating and printing the RNS constants *********************/


/*
 * Init precomputed RNS data
 */
void rns_const_init(){
    mpz_init(RNS_MODULI_PRODUCT);
    mpfr_init(RNS_MODULI_PRODUCT_MPFR);
    mpz_set_ui(RNS_MODULI_PRODUCT, 1);
    for (int i = 0; i < RNS_MODULI_SIZE; i++) {
        mpz_init(RNS_PART_MODULI_PRODUCT[i]);
        mpz_init(RNS_ORTHOGONAL_BASE[i]);
    }
    //Computing moduli product, M
    for (int i = 0; i < RNS_MODULI_SIZE; i++) {
        mpz_mul_ui(RNS_MODULI_PRODUCT, RNS_MODULI_PRODUCT, RNS_MODULI[i]);
    }
    mpfr_set_z(RNS_MODULI_PRODUCT_MPFR, RNS_MODULI_PRODUCT, MPFR_RNDD);

    //Computing partial products, M_i = M / m_i
    for (int i = 0; i < RNS_MODULI_SIZE; i++) {
        mpz_div_ui(RNS_PART_MODULI_PRODUCT[i], RNS_MODULI_PRODUCT, RNS_MODULI[i]);
    }
    //Computing multiplicative inverse of M_i modulo m_i
    for (int i = 0; i < RNS_MODULI_SIZE; i++) {
        RNS_PART_MODULI_PRODUCT_INVERSE[i] = mod_inverse_mpz(RNS_PART_MODULI_PRODUCT[i], RNS_MODULI[i]);

    }
    //Computing orthogonal bases
    for (int i = 0; i < RNS_MODULI_SIZE; i++) {
        mpz_mul_si(RNS_ORTHOGONAL_BASE[i], RNS_PART_MODULI_PRODUCT[i], RNS_PART_MODULI_PRODUCT_INVERSE[i]);
    }
    //Computing RNS representations for powers of two
    for (int i = 0; i <= RNS_MODULI_PRODUCT_LOG2; i++) {
        for (int j = 0; j < RNS_MODULI_SIZE; j++) {
            RNS_POW2[i][j] = 1;
            for (int k = 0; k < i; k++)
                RNS_POW2[i][j] = mod_mul(RNS_POW2[i][j], 2, RNS_MODULI[j]); //(RNS_POW2[i][j] * 2) % RNS_MODULI[j];
        }
    }
    //Computing the residues of moduli product (M) modulo 2^j (j = 1,...,RNS_P2_SCALING_THRESHOLD)
    mpz_t residue;
    mpz_init(residue);
    for (int i = 0; i < RNS_P2_SCALING_THRESHOLD; i++) {
        mpz_mod_ui(residue, RNS_MODULI_PRODUCT, pow2i(i+1));
        RNS_MODULI_PRODUCT_POW2_RESIDUES[i] = (int)mpz_get_si(residue);
    }
    //Computing the matrix of residues of the partial RNS moduli products (M_i) modulo 2^j (the H matrix)
    for (int i = 0; i < RNS_P2_SCALING_THRESHOLD; i++) {
        for(int j = 0; j < RNS_MODULI_SIZE; j++){
            mpz_mod_ui(residue, RNS_PART_MODULI_PRODUCT[j], pow2i(i+1));
            RNS_PART_MODULI_PRODUCT_POW2_RESIDUES[i][j] = (int)mpz_get_si(residue);
        }
    }
    //Computing multiplicative inverses of 2^1,2^2,...,2^RNS_P2_SCALING_THRESHOLD modulo m_i
    for (int i = 0; i < RNS_P2_SCALING_THRESHOLD; i++) {
        for (int j = 0; j < RNS_MODULI_SIZE; j++) {
            int pow = 1;
            for (int k = 0; k <= i; k++){
                pow = mod_mul(pow, 2, RNS_MODULI[j]);     //(pow * 2) % RNS_MODULI[j];
            }
            RNS_POW2_INVERSE[i][j] = mod_inverse((long)pow, (long)RNS_MODULI[j]);
        }
    }
    mpz_clear(residue);
    //Computing accuracy constant for RNS interval evaluation
    RNS_EVAL_ACCURACY  =  4 * pow(2.0, 1 - DBL_PRECISION) * RNS_MODULI_SIZE * log2((double)RNS_MODULI_SIZE) * (1 + RNS_EVAL_RELATIVE_ERROR / 2)  / RNS_EVAL_RELATIVE_ERROR;
    // Computing refinement coefficient for RNS interval evaluation
    RNS_EVAL_REF_FACTOR  =  floor(log2(1/(2*RNS_EVAL_ACCURACY)));
    mpfr_t mpfr_tmp, mpfr_one;
    mpfr_init2(mpfr_tmp, 10000);
    mpfr_init2(mpfr_one, 10000);
    mpfr_set_ui(mpfr_one, 1, MPFR_RNDN);
    //round_up_mode();
    //Computing upper bound for 1 / M
    mpfr_set_ui(mpfr_tmp, 1, MPFR_RNDN);
    mpfr_div(mpfr_tmp, mpfr_tmp, RNS_MODULI_PRODUCT_MPFR, MPFR_RNDU);
    RNS_EVAL_UNIT.upp.frac = mpfr_get_d_2exp(&RNS_EVAL_UNIT.upp.exp, mpfr_tmp, MPFR_RNDU);
    //Computing upper bound for (M - 1) / M = 1 - 1 / M
    mpfr_sub(mpfr_tmp, mpfr_one, mpfr_tmp, MPFR_RNDU);
    RNS_EVAL_INV_UNIT.upp.frac = mpfr_get_d_2exp(&RNS_EVAL_INV_UNIT.upp.exp, mpfr_tmp, MPFR_RNDU);
    //round_down_mode();
    //Computing lower bound for 1 / M
    mpfr_set_ui(mpfr_tmp, 1, MPFR_RNDN);
    mpfr_div(mpfr_tmp, mpfr_tmp, RNS_MODULI_PRODUCT_MPFR, MPFR_RNDD);
    RNS_EVAL_UNIT.low.frac = mpfr_get_d_2exp(&RNS_EVAL_UNIT.low.exp, mpfr_tmp, MPFR_RNDD);
    //Computing lower bound for (M - 1) / M = 1 - 1 / M
    mpfr_sub(mpfr_tmp, mpfr_one, mpfr_tmp, MPFR_RNDD);
    RNS_EVAL_INV_UNIT.low.frac = mpfr_get_d_2exp(&RNS_EVAL_INV_UNIT.low.exp, mpfr_tmp, MPFR_RNDD);
    //round_nearest_mode();
    mpfr_clear(mpfr_tmp);
    mpfr_clear(mpfr_one);
    //Computing reciprocals
    for(int i = 0; i < RNS_MODULI_SIZE; i++){
        RNS_MODULI_RECIP_RD[i] = ddiv_rd(1.0, RNS_MODULI[i]);
        RNS_MODULI_RECIP_RU[i] = ddiv_ru(1.0, RNS_MODULI[i]);
    }
    //Init the MRC constants
    for (int i = 0; i < RNS_MODULI_SIZE; ++i) {
        for (int j = 0; j < RNS_MODULI_SIZE; ++j) {
            MRC_MULT_INV[i][j] = 0;
        }
    }
    for (int i = 0; i < RNS_MODULI_SIZE; ++i) {
        for (int j = i + 1; j < RNS_MODULI_SIZE; ++j) {
            MRC_MULT_INV[i][j] = mod_inverse((long)RNS_MODULI[i], (long)RNS_MODULI[j]);
        }
    }
    //Copying constants to the GPU memory
    cudaMemcpyToSymbol(cuda::RNS_MODULI, &RNS_MODULI, RNS_MODULI_SIZE * sizeof(int)); // Declared in modular.cuh
    cudaMemcpyToSymbol(cuda::RNS_PART_MODULI_PRODUCT_INVERSE, &RNS_PART_MODULI_PRODUCT_INVERSE, RNS_MODULI_SIZE * sizeof(int));
    cudaMemcpyToSymbol(cuda::RNS_POW2, &RNS_POW2, (RNS_MODULI_PRODUCT_LOG2+1) * RNS_MODULI_SIZE * sizeof(int));
    cudaMemcpyToSymbol(cuda::RNS_MODULI_PRODUCT_POW2_RESIDUES, &RNS_MODULI_PRODUCT_POW2_RESIDUES, RNS_P2_SCALING_THRESHOLD * sizeof(int));
    cudaMemcpyToSymbol(cuda::RNS_PART_MODULI_PRODUCT_POW2_RESIDUES, &RNS_PART_MODULI_PRODUCT_POW2_RESIDUES, RNS_P2_SCALING_THRESHOLD * RNS_MODULI_SIZE * sizeof(int));
    cudaMemcpyToSymbol(cuda::RNS_POW2_INVERSE, &RNS_POW2_INVERSE, RNS_P2_SCALING_THRESHOLD * RNS_MODULI_SIZE * sizeof(int));
    cudaMemcpyToSymbol(cuda::RNS_EVAL_ACCURACY, &::RNS_EVAL_ACCURACY, sizeof(double));
    cudaMemcpyToSymbol(cuda::RNS_EVAL_REF_FACTOR, &::RNS_EVAL_REF_FACTOR, sizeof(int));
    cudaMemcpyToSymbol(cuda::RNS_EVAL_UNIT, &RNS_EVAL_UNIT, sizeof(interval_t));
    cudaMemcpyToSymbol(cuda::RNS_EVAL_INV_UNIT, &RNS_EVAL_INV_UNIT, sizeof(interval_t));
    cudaMemcpyToSymbol(cuda::RNS_EVAL_ZERO_BOUND, &RNS_EVAL_ZERO_BOUND, sizeof(er_float_t));
    cudaMemcpyToSymbol(cuda::MRC_MULT_INV, &MRC_MULT_INV, sizeof(int) * RNS_MODULI_SIZE * RNS_MODULI_SIZE);
    cudaMemcpyToSymbol(cuda::RNS_MODULI_RECIP_RD, &RNS_MODULI_RECIP_RD, RNS_MODULI_SIZE * sizeof(double));
    cudaMemcpyToSymbol(cuda::RNS_MODULI_RECIP_RU, &RNS_MODULI_RECIP_RU, RNS_MODULI_SIZE * sizeof(double));
}

/*
 * Printing the constants of the RNS
 */
void rns_const_print(bool briefly) {
    std::cout << "Constants of the RNS system:" << std::endl;
    std::cout << "- RNS_MODULI_SIZE, n: " << RNS_MODULI_SIZE << std::endl;
    std::cout << "- RNS_MODULI_PRODUCT, M: " << mpz_get_str(NULL, 10, RNS_MODULI_PRODUCT) << std::endl;
    mpfr_t log2;
    mpfr_init(log2);
    mpfr_log2(log2, RNS_MODULI_PRODUCT_MPFR, MPFR_RNDD);
    std::cout << "- BIT-SIZE OF MODULI PRODUCT, LOG2(M): " << mpfr_get_d(log2, MPFR_RNDN) << std::endl;
    mpfr_clear(log2);
    std::cout << "- RNS_P2_SCALING_THRESHOLD, T: " << RNS_P2_SCALING_THRESHOLD << std::endl;
    if (!briefly) {
        std::cout << "- RNS_MODULI, m_i: ";
        for (int i = 0; i < RNS_MODULI_SIZE; i++)
            std::cout << std::endl << std::tab << std::tab << RNS_MODULI[i];
        std::cout << std::endl;
        std::cout << "- RNS_PART_MODULI_PRODUCT, M_i: ";
        for (int i = 0; i < RNS_MODULI_SIZE; i++)
            std::cout << std::endl << std::tab << std::tab << mpz_get_str(NULL, 10, RNS_PART_MODULI_PRODUCT[i]);
        std::cout << std::endl;
        std::cout << "- RNS_PART_MODULI_PRODUCT_INVERSE, (M_i)^-1: ";
        for (int i = 0; i < RNS_MODULI_SIZE; i++)
            std::cout << std::endl << std::tab << std::tab << RNS_PART_MODULI_PRODUCT_INVERSE[i];
        std::cout << std::endl;
        std::cout <<  "- RNS_ORTHOGONAL_BASE, M_i * (M_i)^-1: ";
        for (int i = 0; i < RNS_MODULI_SIZE; i++)
            std::cout << std::endl << std::tab << std::tab << mpz_get_str(NULL, 10, RNS_ORTHOGONAL_BASE[i]);
        std::cout << std::endl;
    }
}

/*
 * Printing the constants for the RNS interval evaluation
 */
void rns_eval_const_print() {
    printf("Constants of the RNS interval evaluation for %i moduli:\n", RNS_MODULI_SIZE);
    printf("- RNS_EVAL_RELATIVE_ERROR: %.10f\n", RNS_EVAL_RELATIVE_ERROR);
    printf("- RNS_EVAL_ACCURACY: %.17g\n", RNS_EVAL_ACCURACY);
    printf("- RNS_EVAL_REF_FACTOR: %i\n", RNS_EVAL_REF_FACTOR);
}


/********************* Mixed-radix conversion functions *********************/

/*!
 * Computes the mixed-radix representation for a given RNS number
 * @param mr - pointer to the result mixed-radix representation
 * @param x - pointer to the input RNS number
 */
GCC_FORCEINLINE void mrc(int * mr, int * x) {
    for (int i = 0; i < RNS_MODULI_SIZE; i++) {
        mr[i] = x[i];
        for (int j = 0; j < i; j++) {
            if (mr[i] < mr[j]) {
                long tmp = (long)RNS_MODULI[i] - (long)mr[j] + (long)mr[i];
                mr[i] = (int)tmp;
            } else {
                mr[i] = mr[i] - mr[j];
            }
            mr[i] = mod_mul(mr[i], MRC_MULT_INV[j][i], RNS_MODULI[i]);
        }
    }
}

/*
 * Pairwise comparison of the mixed-radix digits
 */
GCC_FORCEINLINE static int mrs_cmp(int * x, int * y) {
    for (int i = RNS_MODULI_SIZE - 1; i >= 0; i--) {
        if (x[i] > y[i]) {
            return 1;
        } else if (y[i] > x[i]) {
            return -1;
        }
    }
    return 0;
}

/*!
 * Compares RNS numbers using mixed-radix conversion
 * @return 1, if x > y; -1, if x < y; 0, if x = y
 */
GCC_FORCEINLINE int mrc_compare_rns(int * x, int * y) {
    int mx[RNS_MODULI_SIZE];
    int my[RNS_MODULI_SIZE];
    mrc(mx, x);
    mrc(my, y);
    return mrs_cmp(mx, my);
}


/*
 * GPU functions
 */
namespace cuda{

    /*!
     * Computes the mixed-radix representation for a given RNS number
     * @param mr - pointer to the result mixed-radix representation
     * @param x - pointer to the input RNS number
     */
    DEVICE_CUDA_FORCEINLINE void mrc(int * mr, int * x) {
        for (int i = 0; i < RNS_MODULI_SIZE; i++) {
            mr[i] = x[i];
            for (int j = 0; j < i; j++) {
                if (mr[i] < mr[j]) {
                    mr[i] = cuda::mod_psub(mr[i], mr[j], cuda::RNS_MODULI[i]);
                } else {
                    mr[i] = mr[i] - mr[j];
                }
                mr[i] = cuda::mod_mul(mr[i], cuda::MRC_MULT_INV[j][i], cuda::RNS_MODULI[i]);
            }
        }
    }

    /*
     * Pairwise comparison of the mixed-radix digits
     */
    DEVICE_CUDA_FORCEINLINE static int mrs_cmp(int * x, int * y) {
        for (int i = RNS_MODULI_SIZE - 1; i >= 0; i--) {
            if (x[i] > y[i]) {
                return 1;
            } else if (y[i] > x[i]) {
                return -1;
            }
        }
        return 0;
    }

    /*!
     * Compares two RNS numbers using mixed-radix conversion
     * @return 1, if x > y; -1, if x < y; 0, if x = y
     */
    DEVICE_CUDA_FORCEINLINE int mrc_compare_rns(int * x, int * y) {
        int mx[RNS_MODULI_SIZE];
        int my[RNS_MODULI_SIZE];
        cuda::mrc(mx, x);
        cuda::mrc(my, y);
        return cuda::mrs_cmp(mx, my);
    }

} //end of namespace


/********************* Functions for calculating the interval evaluation of an RNS number *********************/


/*!
 * Computes the interval evaluation for a given RNS number
 * This is an improved version of the algorithm from IEEE Access paper (for reference, see README.md)
 * @param low - pointer to the lower bound of the result interval evaluation
 * @param upp - pointer to the upper bound of the result interval evaluation
 * @param x - pointer to the input RNS number
 */
GCC_FORCEINLINE void rns_eval_compute(er_float_ptr low, er_float_ptr upp, int * x) {
    int s[RNS_MODULI_SIZE]; //Array of x_i * w_i (mod m_i)
    double fracl[RNS_MODULI_SIZE];   //Array of x_i * w_i (mod m_i) / m_i, rounding down
    double fracu[RNS_MODULI_SIZE];   //Array of x_i * w_i (mod m_i) / m_i, rounding up
    double suml = 0.0; //Rounded downward sum
    double sumu = 0.0; //Rounded upward sum
    int mrd[RNS_MODULI_SIZE];
    int mr = -1;
    //Checking for zero
    if(rns_check_zero(x)){
        er_set(low, &RNS_EVAL_ZERO_BOUND);
        er_set(upp, &RNS_EVAL_ZERO_BOUND);
        return;
    }
    //Computing the products x_i * w_i (mod m_i) and the corresponding fractions (lower and upper)
    rns_mul(s, x, RNS_PART_MODULI_PRODUCT_INVERSE);
    for (int i = 0; i < RNS_MODULI_SIZE; i++) {
        fracl[i] = dmul_rd(s[i], RNS_MODULI_RECIP_RD[i]);
        fracu[i] = dmul_ru(s[i], RNS_MODULI_RECIP_RU[i]);
        //ddiv_rdu(&fracl[i], &fracu[i], (double)s[i], (double)RNS_MODULI[i]);
    }
    //Pairwise summation of the fractions
    suml = psum_rd<RNS_MODULI_SIZE>(fracl);
    sumu = psum_ru<RNS_MODULI_SIZE>(fracu);
    //Splitting into whole and fractional parts
    auto whl = (unsigned int) suml; // Whole part
    auto whu = (unsigned int) sumu; // Whole part
    suml = suml - whl;    // Fractional part
    sumu = sumu - whu;    // Fractional part
    //Assign the computed values to the result
    er_set_d(low, suml);
    er_set_d(upp, sumu);
    //Check for ambiguity
    if(whl != whu) {
        mrc(mrd, x); //Computing the mixed-radix representation of x
        mr = mrd[RNS_MODULI_SIZE - 1];
    }
    //Adjust if ambiguity was found
    if(mr > 0){
        er_set(upp, &RNS_EVAL_INV_UNIT.upp);
        return;
    }
    if(mr == 0){
        er_set(low, &RNS_EVAL_UNIT.low);
    }
    // Refinement is not required
    if(sumu >= RNS_EVAL_ACCURACY){
        return;
    }
    //Need more accuracy. Performing a refinement loop with stepwise calculation of the shifted upper bound
    int K = 0;
    while (sumu < RNS_EVAL_ACCURACY) {
        //The improvement is that the refinement factor depends on the value of X
        int k = MAX(-(ceil(log2(sumu))+1), RNS_EVAL_REF_FACTOR);
        rns_mul(s, s, RNS_POW2[k]);
        for(int i = 0; i < RNS_MODULI_SIZE; i++) {
            fracu[i] = ddiv_ru((double)s[i], (double)RNS_MODULI[i]);
        }
        sumu = psum_ru<RNS_MODULI_SIZE>(fracu);
        sumu -= (unsigned int) sumu;
        K += k;
    }
    //Computing the shifted lower bound
    for (int i = 0; i < RNS_MODULI_SIZE; i++) {
        fracl[i] = ddiv_rd((double)s[i], (double)RNS_MODULI[i]);
    }
    suml = psum_rd<RNS_MODULI_SIZE>(fracl);
    suml -= (unsigned int) suml;
    //Setting the result lower and upper bounds of eval with appropriate correction (scaling by a power of two)
    er_set_d(low, suml);
    er_set_d(upp, sumu);
    low->exp -= K;
    upp->exp -= K;
}


/*!
 * For a given RNS number, which is guaranteed not to be too large,
 * this function computes the interval evaluation faster than the previous common function.
 * @param low - pointer to the lower bound of the result interval evaluation
 * @param upp - pointer to the upper bound of the result interval evaluation
 * @param x - pointer to the input RNS number
 */
GCC_FORCEINLINE void rns_eval_compute_fast(er_float_ptr low, er_float_ptr upp, int * x) {
    int s[RNS_MODULI_SIZE];
    double fracl[RNS_MODULI_SIZE];
    double fracu[RNS_MODULI_SIZE];
    double suml = 0.0;
    double sumu = 0.0;
    //Checking for zero
    if(rns_check_zero(x)){
        er_set(low, &RNS_EVAL_ZERO_BOUND);
        er_set(upp, &RNS_EVAL_ZERO_BOUND);
        return;
    }
    //Computing the products x_i * w_i (mod m_i) and the corresponding fractions (lower and upper)
    rns_mul(s, x, RNS_PART_MODULI_PRODUCT_INVERSE);
    for (int i = 0; i < RNS_MODULI_SIZE; i++) {
        fracl[i] = dmul_rd(s[i], RNS_MODULI_RECIP_RD[i]);
        fracu[i] = dmul_ru(s[i], RNS_MODULI_RECIP_RU[i]);
        //ddiv_rdu(&fracl[i], &fracu[i], (double)s[i], (double)RNS_MODULI[i]);
    }
    //Pairwise summation of the fractions
    suml = psum_rd<RNS_MODULI_SIZE>(fracl);
    sumu = psum_ru<RNS_MODULI_SIZE>(fracu);
    //Dropping integer parts
    suml -= (unsigned int) suml;  //Lower bound
    sumu -= (unsigned int) sumu;  //Upper bound
    //Accuracy checking
    if (sumu >= RNS_EVAL_ACCURACY) {
        er_set_d(low, suml);
        er_set_d(upp, sumu);
        return;
    }
    //Need more accuracy. Performing a refinement loop with stepwise calculation of the shifted upper bound
    int K = 0;
    while (sumu < RNS_EVAL_ACCURACY) {
        //The improvement is that the refinement factor depends on the value of X
        int k = MAX(-(ceil(log2(sumu))+1), RNS_EVAL_REF_FACTOR);
        rns_mul(s, s, RNS_POW2[k]);
        for(int i = 0; i < RNS_MODULI_SIZE; i++){
            fracu[i] = ddiv_ru((double)s[i], (double)RNS_MODULI[i]);
        }
        sumu = psum_ru<RNS_MODULI_SIZE>(fracu);
        sumu -= (unsigned int) sumu;
        K += k;
    }
    //Computing the shifted lower bound
    for (int i = 0; i < RNS_MODULI_SIZE; i++) {
        fracl[i] = ddiv_rd((double)s[i], (double)RNS_MODULI[i]);
    }
    suml = psum_rd<RNS_MODULI_SIZE>(fracl);
    suml -= (unsigned int) suml;
    //Setting the result lower and upper bounds of eval with appropriate correction (scaling by a power of two)
    er_set_d(low, suml);
    er_set_d(upp, sumu);
    low->exp -= K;
    upp->exp -= K;
}



/*
 * GPU functions
 */
namespace cuda{

    /*!
     * Computes the interval evaluation for a given RNS number
     * This is an improved version of the algorithm from IEEE Access paper (for reference, see README.md)
     * @param low - pointer to the lower bound of the result interval evaluation
     * @param upp - pointer to the upper bound of the result interval evaluation
     * @param x - pointer to the input RNS number
     */
    DEVICE_CUDA_FORCEINLINE void rns_eval_compute(er_float_ptr low, er_float_ptr upp, int * x) {
        const double accuracy_constant = cuda::RNS_EVAL_ACCURACY;
        int  s[RNS_MODULI_SIZE];
        double fracl[RNS_MODULI_SIZE];
        double fracu[RNS_MODULI_SIZE];
        double suml = 0.0;
        double sumu = 0.0;
        int mrd[RNS_MODULI_SIZE];
        int mr = -1;
        //Computing the products x_i * w_i (mod m_i) and the corresponding fractions (lower and upper)
        cuda::rns_mul(s, x, cuda::RNS_PART_MODULI_PRODUCT_INVERSE);
        for (int i = 0; i < RNS_MODULI_SIZE; i++) {
            fracl[i] = __dmul_rd(s[i], cuda::RNS_MODULI_RECIP_RD[i]);
            fracu[i] = __dmul_ru(s[i], cuda::RNS_MODULI_RECIP_RU[i]);
        }
        //Pairwise summation of the fractions
        suml = cuda::psum_rd<RNS_MODULI_SIZE>(fracl);
        sumu = cuda::psum_ru<RNS_MODULI_SIZE>(fracu);
        //Checking for zero
        if (suml == 0 && sumu == 0) {
            cuda::er_set(low, &cuda::RNS_EVAL_ZERO_BOUND);
            cuda::er_set(upp, &cuda::RNS_EVAL_ZERO_BOUND);
            return;
        }
        //Splitting into whole and fractional parts
        auto whl = (unsigned int) (suml);
        auto whu = (unsigned int) (sumu);
        suml = __dsub_rd(suml, whl);    // lower bound
        sumu = __dsub_ru(sumu, whu);    // upper bound
        //Assign the computed values to the result
        cuda::er_set_d(low, suml);
        cuda::er_set_d(upp, sumu);
        //Check for ambiguity
        if(whl != whu) {
            cuda::mrc(mrd, x); //Computing the mixed-radix representation of x
            mr = mrd[RNS_MODULI_SIZE - 1];
        }
        //Adjust if ambiguity was found
        if(mr > 0){
            cuda::er_set(upp, &cuda::RNS_EVAL_INV_UNIT.upp);
            return;
        }
        if(mr == 0){
            cuda::er_set(low, &cuda::RNS_EVAL_UNIT.low);
        }
        // Refinement is not required
        if(sumu >= accuracy_constant){
            return;
        }
        //Need more accuracy. Performing a refinement loop with stepwise calculation of the shifted upper bound
        int K = 0;
        while (sumu < accuracy_constant) {
            //The improvement is that the refinement factor depends on the value of X
            int k = MAX(-(ceil(log2(sumu))+1), cuda::RNS_EVAL_REF_FACTOR);
            cuda::rns_mul(s, s, cuda::RNS_POW2[k]);
            for(int i = 0; i < RNS_MODULI_SIZE; i++) {
                fracu[i] = __dmul_ru(s[i], cuda::RNS_MODULI_RECIP_RU[i]);
            }
            sumu = cuda::psum_ru<RNS_MODULI_SIZE>(fracu);
            sumu = __dsub_ru(sumu, (unsigned int) sumu);
            K += k;
        }
        // Computing the shifted lower bound
        for (int i = 0; i < RNS_MODULI_SIZE; i++) {
            fracl[i] = __dmul_rd(s[i], cuda::RNS_MODULI_RECIP_RD[i]);
        }
        suml = cuda::psum_rd<RNS_MODULI_SIZE>(fracl);
        suml = __dsub_rd(suml, (unsigned int) suml);
        //Setting the result lower and upper bounds of eval with appropriate correction (scaling by a power of two)
        cuda::er_set_d(low, suml);
        cuda::er_set_d(upp, sumu);
        low->exp -= K;
        upp->exp -= K;
    }


    /*!
     * For a given RNS number, which is guaranteed not to be too large,
     * this function computes the interval evaluation faster than the previous common function.
     * @param low - pointer to the lower bound of the result interval evaluation
     * @param upp - pointer to the upper bound of the result interval evaluation
     * @param x - pointer to the input RNS number
     */
    DEVICE_CUDA_FORCEINLINE void rns_eval_compute_fast(er_float_ptr low, er_float_ptr upp, int * x) {
        const double accuracy_constant = cuda::RNS_EVAL_ACCURACY;
        int s[RNS_MODULI_SIZE];
        double fracl[RNS_MODULI_SIZE];
        double fracu[RNS_MODULI_SIZE];
        double suml = 0.0;
        double sumu = 0.0;
        //Computing the products x_i * w_i (mod m_i) and the corresponding fractions (lower and upper)
        cuda::rns_mul(s, x, cuda::RNS_PART_MODULI_PRODUCT_INVERSE);
        for (int i = 0; i < RNS_MODULI_SIZE; i++) {
            fracl[i] = __dmul_rd(s[i], cuda::RNS_MODULI_RECIP_RD[i]);
            fracu[i] = __dmul_ru(s[i], cuda::RNS_MODULI_RECIP_RU[i]);
        }
        //Pairwise summation of the fractions
        suml = cuda::psum_rd<RNS_MODULI_SIZE>(fracl);
        sumu = cuda::psum_ru<RNS_MODULI_SIZE>(fracu);
        //Checking for zero
        if (suml == 0 && sumu == 0) {
            cuda::er_set(low, &cuda::RNS_EVAL_ZERO_BOUND);
            cuda::er_set(upp, &cuda::RNS_EVAL_ZERO_BOUND);
            return;
        }
        //Dropping integer parts
        suml = __dsub_rd(suml, (unsigned int) suml); //Lower bound
        sumu = __dsub_ru(sumu, (unsigned int) sumu); //Upper bound
        //Accuracy checking
        if (sumu >= accuracy_constant) {
            cuda::er_set_d(low, suml);
            cuda::er_set_d(upp, sumu);
            return;
        }
        //Need more accuracy. Performing a refinement loop with stepwise calculation of the shifted upper bound
        int K = 0;
        while (sumu < accuracy_constant) {
            //The improvement is that the refinement factor depends on the value of X
            int k = MAX(-(ceil(log2(sumu))+1), cuda::RNS_EVAL_REF_FACTOR);
            cuda::rns_mul(s, s, cuda::RNS_POW2[k]);
            for(int i = 0; i < RNS_MODULI_SIZE; i++) {
                fracu[i] = __dmul_ru(s[i], cuda::RNS_MODULI_RECIP_RU[i]);
            }
            sumu = cuda::psum_ru<RNS_MODULI_SIZE>(fracu);
            sumu = __dsub_ru(sumu, (unsigned int) sumu);
            K += k;
        }
        // Computing the shifted lower bound
        for (int i = 0; i < RNS_MODULI_SIZE; i++) {
            fracl[i] = __dmul_rd(s[i], cuda::RNS_MODULI_RECIP_RD[i]);
        }
        suml = cuda::psum_rd<RNS_MODULI_SIZE>(fracl);
        suml = __dsub_rd(suml, (unsigned int) suml);
        //Setting the result lower and upper bounds of eval with appropriate correction (scaling by a power of two)
        cuda::er_set_d(low, suml);
        cuda::er_set_d(upp, sumu);
        low->exp -= K;
        upp->exp -= K;
    }


} //end of namespace


/********************* Power-of-two RNS scaling functions *********************/

/*
 * For a given RNS number x = (x0,...,xn), this helper function computes r such that
 * X = sum( Mi * x_i * w_i (mod m_i) ) - r * M. Array s stores the computed values (xi * w_i) mod mi
 * where w_i is the modulo mi multiplicative inverse of Mi = M / mi.
 */
GCC_FORCEINLINE static int rns_rank_compute(int * x, int * s) {
    double fracl[RNS_MODULI_SIZE];   //Array of x_i * w_i (mod m_i) / m_i, rounding down
    double fracu[RNS_MODULI_SIZE];   //Array of x_i * w_i (mod m_i) / m_i, rounding up
    double suml = 0.0;
    double sumu = 0.0;
    int mrd[RNS_MODULI_SIZE];
    int mr = -1;

    //Computing ( (x_i * w_i) mod m_i ) / mi
    for (int i = 0; i < RNS_MODULI_SIZE; i++) {
        fracl[i] = dmul_rd(s[i], RNS_MODULI_RECIP_RD[i]);
        fracu[i] = dmul_ru(s[i], RNS_MODULI_RECIP_RU[i]);
        //ddiv_rdu(&fracl[i], &fracu[i], (double)s[i], (double)RNS_MODULI[i]);
    }
    //Pairwise summation of the fractions
    suml = psum_rd<RNS_MODULI_SIZE>(fracl);
    sumu = psum_ru<RNS_MODULI_SIZE>(fracu);
    //Discarding the fractional part
    auto whl = (unsigned int) suml; // Whole part
    auto whu = (unsigned int) sumu; // Whole part
    //Checking for quick return
    if(whl == whu) {
        return whl;
    } else {
        mrc(mrd, x); //Computing the mixed-radix representation of x
        mr = mrd[RNS_MODULI_SIZE - 1];
        return mr == 0 ? whu : whl;
    }
}

/*
 * For a given RNS number x = (x0,...,xn), which is guaranteed not to be too large,
 * this helper function computes r such that X = sum( Mi * x_i * w_i (mod m_i) ) - r * M.
 * Array s stores the computed values (xi * w_i) mod mi, where w_i is the modulo mi multiplicative inverse of Mi = M / mi.
 * This function performs faster than the previous one.
 */
GCC_FORCEINLINE static int rns_rank_compute_fast(int * s) {
    double fracu[RNS_MODULI_SIZE];   //Array of x_i * w_i (mod m_i) / m_i, rounding up
    double sumu = 0.0;
    for (int i = 0; i < RNS_MODULI_SIZE; i++) {
        fracu[i] = dmul_ru(s[i], RNS_MODULI_RECIP_RU[i]);
        //fracu[i] = ddiv_ru((double)s[i], (double)RNS_MODULI[i]);
    }
    sumu = psum_ru<RNS_MODULI_SIZE>(fracu);
    return (int) sumu;
}

/*
 * This helper function performs one step of scaling by a power of two
 */
GCC_FORCEINLINE static void make_scaling_step(int *y, int k, unsigned int j, int pow2j, int * x, int * c) {
    long residue = 0; //X mod 2^j
    int multiple[RNS_MODULI_SIZE]; //X - (X mod pow2j)
    for (int i = 0; i < RNS_MODULI_SIZE; i++) {
        //RNS_PART_MODULI_PRODUCT_POW2_RESIDUES[j-1][i] ->  M_i mod 2^j
        residue += (long)mod_mul(RNS_PART_MODULI_PRODUCT_POW2_RESIDUES[j - 1][i], c[i], pow2j);
    }
    //RNS_MODULI_PRODUCT_POW2_RESIDUES[j-1] ->  M mod 2^j
    long temp = (long)k * (long)RNS_MODULI_PRODUCT_POW2_RESIDUES[j - 1];
    residue = (residue - temp) % pow2j;
    if(residue < 0)
        residue += pow2j;
    for (int i = 0; i < RNS_MODULI_SIZE; i++) { // multiple[i] <- (remainder when X is divided by pow2j) mod m_i
        multiple[i] = residue % RNS_MODULI[i];
    }
    rns_sub(multiple, x, multiple); //multiple <- X - remainder
    //RNS_POW2_INVERSE[j-1][i] -> (2^j )^{-1} mod m_i
    rns_mul(y, multiple, RNS_POW2_INVERSE[j - 1]);
}

/*!
  * Scaling an RNS number by a power of 2: result = x / 2^D
  * @param result - pointer to the result (scaled number)
  * @param x - pointer to the RNS number to be scaled
  * @param D - exponent of the scaling factor
  */
GCC_FORCEINLINE void rns_scale2pow(int * result, int * x, unsigned int D) {
    rns_set(result, x); // result <- x
    int t = D / RNS_P2_SCALING_THRESHOLD;
    int k = 0;
    int c[RNS_MODULI_SIZE];
    //first step
    if (t > 0) {
        rns_mul(c, x, RNS_PART_MODULI_PRODUCT_INVERSE);
        k = rns_rank_compute(x, c);
        make_scaling_step(result, k, RNS_P2_SCALING_THRESHOLD, RNS_P2_SCALING_FACTOR, x, c);
        t -= 1;
    }
    //second step
    while (t > 0) {
        rns_mul(c, result, RNS_PART_MODULI_PRODUCT_INVERSE);
        k = rns_rank_compute_fast(c);
        make_scaling_step(result, k, RNS_P2_SCALING_THRESHOLD, RNS_P2_SCALING_FACTOR, result, c);
        t -= 1;
    }
    //third step
    unsigned int d = D % RNS_P2_SCALING_THRESHOLD;
    if (d > 0) {
        rns_mul(c, result, RNS_PART_MODULI_PRODUCT_INVERSE);
        k = d < D ? rns_rank_compute_fast(c) : rns_rank_compute(result, c);
        make_scaling_step(result, k, d, 1 << d, result, c);
    }
}


/*
 * GPU functions
 */
namespace cuda{

    /*
     * For a given RNS number x = (x0,...,xn), this helper function computes r such that
     * X = sum( Mi * x_i * w_i (mod m_i) ) - r * M. Array s stores the computed values (xi * w_i) mod mi
     * where w_i is the modulo mi multiplicative inverse of Mi = M / mi.
     */
    DEVICE_CUDA_FORCEINLINE static int rns_rank_compute(int * x, int * s) {
        double fracl[RNS_MODULI_SIZE];
        double fracu[RNS_MODULI_SIZE];
        double suml = 0.0;
        double sumu = 0.0;
        int mrd[RNS_MODULI_SIZE];
        int mr = -1;
        //Computing ( (x_i * w_i) mod m_i ) / mi
        for (int i = 0; i < RNS_MODULI_SIZE; i++) {
            fracl[i] = __dmul_rd(s[i], cuda::RNS_MODULI_RECIP_RD[i]);
            fracu[i] = __dmul_ru(s[i], cuda::RNS_MODULI_RECIP_RU[i]);
        }
        //Pairwise summation of the fractions
        suml = cuda::psum_rd<RNS_MODULI_SIZE>(fracl);
        sumu = cuda::psum_ru<RNS_MODULI_SIZE>(fracu);
        //Discarding the fractional part
        auto whl = (unsigned int) (suml);
        auto whu = (unsigned int) (sumu);
        //Checking for quick return
        if(whl == whu) {
            return whl;
        } else {
            cuda::mrc(mrd, x);
            mr = mrd[RNS_MODULI_SIZE - 1];
            return mr == 0 ? whu : whl;
        }
    }

    /*
     * For a given RNS number x = (x0,...,xn), which is guaranteed not to be too large,
     * this helper function computes r such that X = sum( Mi * x_i * w_i (mod m_i) ) - r * M.
     * Array s stores the computed values (xi * w_i) mod mi, where w_i is the modulo mi multiplicative inverse of Mi = M / mi.
     * This function performs faster than the previous one.
     */
    DEVICE_CUDA_FORCEINLINE static int rns_rank_compute_fast(int * x, int * s) {
        double fracu[RNS_MODULI_SIZE];
        double sumu = 0.0;
        for (int i = 0; i < RNS_MODULI_SIZE; i++) {
            fracu[i] = __dmul_ru(s[i], cuda::RNS_MODULI_RECIP_RU[i]);
        }
        sumu = cuda::psum_ru<RNS_MODULI_SIZE>(fracu);
        return (int) sumu;
    }

    /*
     * This helper function performs one step of scaling by a power of two
     */
    DEVICE_CUDA_FORCEINLINE static void make_scaling_step(int * y, int k, unsigned int j, int pow2j, int * x, int * c) {
        long residue = 0; // X mod 2^j
        int multiple[RNS_MODULI_SIZE]; // X - (X mod pow2j)
        for (int i = 0; i < RNS_MODULI_SIZE; i++) {
            //RNS_PART_MODULI_PRODUCT_POW2_RESIDUES[j-1][i] ->  M_i mod 2^j
            residue += (long)cuda::mod_mul(cuda::RNS_PART_MODULI_PRODUCT_POW2_RESIDUES[j - 1][i], c[i], pow2j);
        }
        //RNS_MODULI_PRODUCT_POW2_RESIDUES[j-1] ->  M mod 2^j
        long temp = (long)k * (long)cuda::RNS_MODULI_PRODUCT_POW2_RESIDUES[j - 1];
        residue = (residue - temp) % pow2j;
        if(residue < 0)
            residue += pow2j;
        for (int i = 0; i < RNS_MODULI_SIZE; i++) { // multiple[i] <- (remainder when X is divided by pow2j) mod m_i
            multiple[i] = residue % cuda::RNS_MODULI[i];
        }
        cuda::rns_sub(multiple, x, multiple); //multiple <- X - remainder
        //RNS_POW2_INVERSE[j-1][i] -> (2^j )^{-1} mod m_i
        cuda::rns_mul(y, multiple, cuda::RNS_POW2_INVERSE[j - 1]);
    }

    /*!
      * Scaling an RNS number by a power of 2: result = x / 2^D
      * @param result - pointer to the result (scaled number)
      * @param x - pointer to the RNS number to be scaled
      * @param D - exponent of the scaling factor
      */
    DEVICE_CUDA_FORCEINLINE void rns_scale2pow(int *result, int * x, unsigned int D) {
        for (int i = 0; i < RNS_MODULI_SIZE; i++){
            result[i] = x[i];
        }
        int t = D / RNS_P2_SCALING_THRESHOLD;
        int k = 0;
        int c[RNS_MODULI_SIZE];
        //first step
        if (t > 0) {
            cuda::rns_mul(c, x, cuda::RNS_PART_MODULI_PRODUCT_INVERSE);
            k = cuda::rns_rank_compute(x, c);
            cuda::make_scaling_step(result, k, RNS_P2_SCALING_THRESHOLD, RNS_P2_SCALING_FACTOR, x, c);
            t -= 1;
        }
        //second step
        while (t > 0) {
            cuda::rns_mul(c, result, cuda::RNS_PART_MODULI_PRODUCT_INVERSE);
            k = cuda::rns_rank_compute_fast(result, c);
            cuda::make_scaling_step(result, k, RNS_P2_SCALING_THRESHOLD, RNS_P2_SCALING_FACTOR, result, c);
            t -= 1;
        }
        //third step
        unsigned int d = D % RNS_P2_SCALING_THRESHOLD;
        if (d > 0) {
            cuda::rns_mul(c, result, cuda::RNS_PART_MODULI_PRODUCT_INVERSE);
            k = d < D ? cuda::rns_rank_compute(result, c) : cuda::rns_rank_compute_fast(result, c);
            cuda::make_scaling_step(result, k, d, 1 << d, result, c);
        }
    }

} //end of namespace


/********************* RNS magnitude comparison functions *********************/

/*!
 * Given two integers x and y such that 0 \le x,y < M, represented as x = (x1 ,x2,...,xn) and y = (y1 ,y2,...,yn),
 * this routine returns:
 *  0, if x = y
 *  1, if x > y
 * -1, if x < y
 * @param x - pointer to the number in the RNS
 * @param y - pointer to the number in the RNS
 */
GCC_FORCEINLINE int rns_cmp(int *x, int *y) {
    interval_t ex; //Interval evaluation of x
    interval_t ey; //Interval evaluation of y
    rns_eval_compute(&ex.low, &ex.upp, x);
    rns_eval_compute(&ey.low, &ey.upp, y);
    if(er_ucmp(&ex.low, &ey.upp) > 0){
        return 1;
    }
    if(er_ucmp(&ey.low, &ex.upp) > 0){
        return -1;
    }
    bool equals = true;
    for(int i = 0; i < RNS_MODULI_SIZE; i++){
        if(x[i] != y[i]){
            equals = false;
            break;
        }
    }
    return equals ? 0 : mrc_compare_rns(x, y);
}

/*!
 * Given two integers x and y such that 0 \le x,y < M, represented as x = (x1 ,x2,...,xn) and y = (y1 ,y2,...,yn),
 * and their interval evaluations I(X/M) = [exl, exu], I(Y/M) = [eyl, eyu], this routine returns:
 *  0, if x = y
 *  1, if x > y
 * -1, if x < y
 * @param x - pointer to the number in the RNS
 * @param exl - pointer to the lower bound of the interval evaluation of x
 * @param exu - pointer to the upper bound of the interval evaluation of x
 * @param y - pointer to the number in the RNS
 * @param eyl - pointer to the lower bound of the interval evaluation of y
 * @param eyu - pointer to the upper bound of the interval evaluation of y
 */
GCC_FORCEINLINE int rns_cmp(int *x, er_float_ptr exl, er_float_ptr exu, int *y, er_float_ptr eyl, er_float_ptr eyu) {
    if(er_ucmp(exl, eyu) > 0){
        return 1;
    }
    if(er_ucmp(eyl, exu) > 0){
        return -1;
    }
    bool equals = true;
    for(int i = 0; i < RNS_MODULI_SIZE; i++){
        if(x[i] != y[i]){
            equals = false;
            break;
        }
    }
    return equals ? 0 : mrc_compare_rns(x, y);
}


/*
 * GPU functions
 */
namespace cuda {

    /*!
     * Given two integers x and y such that 0 \le x,y < M, represented as x = (x1 ,x2,...,xn) and y = (y1 ,y2,...,yn),
     * this routine returns:
     *  0, if x = y
     *  1, if x > y
     * -1, if x < y
     * @param x - pointer to the number in the RNS
     * @param y - pointer to the number in the RNS
     */
    DEVICE_CUDA_FORCEINLINE int rns_cmp(int *x, int *y) {
        interval_t ex; //Interval evaluation of x
        interval_t ey; //Interval evaluation of y
        cuda::rns_eval_compute(&ex.low, &ex.upp, x);
        cuda::rns_eval_compute(&ey.low, &ey.upp, y);
        if(cuda::er_ucmp(&ex.low, &ey.upp) > 0){
            return 1;
        }
        if(cuda::er_ucmp(&ey.low, &ex.upp) > 0){
            return -1;
        }
        bool equals = true;
        for(int i = 0; i < RNS_MODULI_SIZE; i++){
            if(x[i] != y[i]){
                equals = false;
                break;
            }
        }
        return equals ? 0 : cuda::mrc_compare_rns(x, y);
    }

    /*!
     * Given two integers x and y such that 0 \le x,y < M, represented as x = (x1 ,x2,...,xn) and y = (y1 ,y2,...,yn),
     * and their interval evaluations I(X/M) = [exl, exu], I(Y/M) = [eyl, eyu], this routine returns:
     *  0, if x = y
     *  1, if x > y
     * -1, if x < y
     * @param x - pointer to the number in the RNS
     * @param exl - pointer to the lower bound of the interval evaluation of x
     * @param exu - pointer to the upper bound of the interval evaluation of x
     * @param y - pointer to the number in the RNS
     * @param eyl - pointer to the lower bound of the interval evaluation of y
     * @param eyu - pointer to the upper bound of the interval evaluation of y
     */
    DEVICE_CUDA_FORCEINLINE int rns_cmp(int *x, er_float_ptr exl, er_float_ptr exu, int *y, er_float_ptr eyl, er_float_ptr eyu) {
        if(cuda::er_ucmp(exl, eyu) > 0){
            return 1;
        }
        if(cuda::er_ucmp(eyl, exu) > 0){
            return -1;
        }
        bool equals = true;
        for(int i = 0; i < RNS_MODULI_SIZE; i++){
            if(x[i] != y[i]){
                equals = false;
                break;
            }
        }
        return equals ? 0 : cuda::mrc_compare_rns(x, y);
    }

} //end of namespace


#endif //MPRES_RNS_CUH
