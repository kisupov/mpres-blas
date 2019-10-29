/*
 *  Inter-modulo (non-modular) computations in the Residue Number System.
 *  The CPU and GPU algorithms are presented for computing the interval evaluation
 *  of the fractional representation of an RNS number, power-of-two RNS scaling,
 *  and mixed-radix conversion. For details, see
 *  http://dx.doi.org/10.1142/S0218126618500044 (Interval evaluation in RNS)
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
double RNS_MODULI_RECIPROCAL[RNS_MODULI_SIZE]; // Array of 1 / RNS_MODULI[i]

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
 * Constants for computing the interval evaluation of an RNS number
 */
int RNS_EVAL_OFFSET_VEC[RNS_EVAL_OFFSET_VEC_SIZE]; // Offset vector. This vector contains only the power-of-two exponents
int RNS_EVAL_OFFSET_MATR[RNS_EVAL_OFFSET_VEC_SIZE][RNS_MODULI_SIZE]; //Offset multiplicative inverses matrix: m(i,j) = mult_inverse[i] * 2^i (i = 1,...,g; j = 1,...,n)
interval_t RNS_EVAL_INV_UNIT; // Interval approximation of (M - 1) / M
interval_t RNS_EVAL_UNIT; // Interval approximation of 1 / M
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
    __device__ double RNS_MODULI_RECIPROCAL[RNS_MODULI_SIZE];
    __device__ int RNS_POW2[RNS_MODULI_PRODUCT_LOG2+1][RNS_MODULI_SIZE];
    __device__ int RNS_MODULI_PRODUCT_POW2_RESIDUES[RNS_P2_SCALING_THRESHOLD];
    __device__ int RNS_PART_MODULI_PRODUCT_POW2_RESIDUES[RNS_P2_SCALING_THRESHOLD][RNS_MODULI_SIZE];
    __device__ int RNS_POW2_INVERSE[RNS_P2_SCALING_THRESHOLD][RNS_MODULI_SIZE];
    __device__ __constant__ int RNS_EVAL_OFFSET_VEC[RNS_EVAL_OFFSET_VEC_SIZE];
    __device__ int RNS_EVAL_OFFSET_MATR[RNS_EVAL_OFFSET_VEC_SIZE][RNS_MODULI_SIZE];
    __device__ __constant__  interval_t RNS_EVAL_INV_UNIT;
    __device__ __constant__  interval_t RNS_EVAL_UNIT;
    __device__ __constant__ er_float_t RNS_EVAL_ZERO_BOUND;
    __device__ int MRC_MULT_INV[RNS_MODULI_SIZE][RNS_MODULI_SIZE];
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
inline void rns_set(int *target, int * x) {
    memcpy(target, x, sizeof(int) * RNS_MODULI_SIZE);
}

/*
 * Converts x from RNS to binary system using Chinese remainder theorem.
 * The result is stored in target
 */
void rns_to_binary(mpz_t target, int * x) {
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
void rns_fractional(er_float_ptr result, int * x) {
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
    //Computing recipocals for moduli
    for (int i = 0; i < RNS_MODULI_SIZE; ++i) {
        RNS_MODULI_RECIPROCAL[i] = (double) 1 / RNS_MODULI[i];
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
    //Computing offset vector for RNS interval evaluation
    for (int i = 0; i < RNS_EVAL_OFFSET_VEC_SIZE; i++) {
        RNS_EVAL_OFFSET_VEC[i] = (int) ceil(-2*(i+1) - (i+1)*log(RNS_EVAL_MIN_LIMIT)/log(2.0));
    }
    //Computing offset matrix for RNS interval evaluation
    for (int i = 0; i < RNS_EVAL_OFFSET_VEC_SIZE; i++) {
        for (int j = 0; j < RNS_MODULI_SIZE; j++) {
            int pow2_mod = 1;
            for (int k = 0; k < RNS_EVAL_OFFSET_VEC[i]; k++) //Overflow-safe compute 2^i mod p_i
                pow2_mod = mod_mul(pow2_mod, 2, RNS_MODULI[j]); //(pow2_mod * 2) % RNS_MODULI[j];
            RNS_EVAL_OFFSET_MATR[i][j] = mod_mul(RNS_PART_MODULI_PRODUCT_INVERSE[j], pow2_mod,  RNS_MODULI[j]); //(RNS_PART_MODULI_PRODUCT_INVERSE[j] * pow2_mod) % RNS_MODULI[j];
        }
    }
    mpfr_t mpfr_tmp, mpfr_one;
    mpfr_init2(mpfr_tmp, 10000);
    mpfr_init2(mpfr_one, 10000);
    mpfr_set_ui(mpfr_one, 1, MPFR_RNDN);
    round_up_mode();
    //Computing upper bound for 1 / M
    mpfr_set_ui(mpfr_tmp, 1, MPFR_RNDN);
    mpfr_div(mpfr_tmp, mpfr_tmp, RNS_MODULI_PRODUCT_MPFR, MPFR_RNDU);
    RNS_EVAL_UNIT.upp.frac = mpfr_get_d_2exp(&RNS_EVAL_UNIT.upp.exp, mpfr_tmp, MPFR_RNDU);
    //Computing upper bound for (M - 1) / M = 1 - 1 / M
    mpfr_sub(mpfr_tmp, mpfr_one, mpfr_tmp, MPFR_RNDU);
    RNS_EVAL_INV_UNIT.upp.frac = mpfr_get_d_2exp(&RNS_EVAL_INV_UNIT.upp.exp, mpfr_tmp, MPFR_RNDU);
    round_down_mode();
    //Computing lower bound for 1 / M
    mpfr_set_ui(mpfr_tmp, 1, MPFR_RNDN);
    mpfr_div(mpfr_tmp, mpfr_tmp, RNS_MODULI_PRODUCT_MPFR, MPFR_RNDD);
    RNS_EVAL_UNIT.low.frac = mpfr_get_d_2exp(&RNS_EVAL_UNIT.low.exp, mpfr_tmp, MPFR_RNDD);
    //Computing lower bound for (M - 1) / M = 1 - 1 / M
    mpfr_sub(mpfr_tmp, mpfr_one, mpfr_tmp, MPFR_RNDD);
    RNS_EVAL_INV_UNIT.low.frac = mpfr_get_d_2exp(&RNS_EVAL_INV_UNIT.low.exp, mpfr_tmp, MPFR_RNDD);
    round_nearest_mode();
    mpfr_clear(mpfr_tmp);
    mpfr_clear(mpfr_one);
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
    cudaMemcpyToSymbol(cuda::RNS_MODULI_RECIPROCAL, &RNS_MODULI_RECIPROCAL, RNS_MODULI_SIZE * sizeof(double));
    cudaMemcpyToSymbol(cuda::RNS_POW2, &RNS_POW2, (RNS_MODULI_PRODUCT_LOG2+1) * RNS_MODULI_SIZE * sizeof(int));
    cudaMemcpyToSymbol(cuda::RNS_MODULI_PRODUCT_POW2_RESIDUES, &RNS_MODULI_PRODUCT_POW2_RESIDUES, RNS_P2_SCALING_THRESHOLD * sizeof(int));
    cudaMemcpyToSymbol(cuda::RNS_PART_MODULI_PRODUCT_POW2_RESIDUES, &RNS_PART_MODULI_PRODUCT_POW2_RESIDUES, RNS_P2_SCALING_THRESHOLD * RNS_MODULI_SIZE * sizeof(int));
    cudaMemcpyToSymbol(cuda::RNS_POW2_INVERSE, &RNS_POW2_INVERSE, RNS_P2_SCALING_THRESHOLD * RNS_MODULI_SIZE * sizeof(int));
    cudaMemcpyToSymbol(cuda::RNS_EVAL_OFFSET_VEC, &RNS_EVAL_OFFSET_VEC, RNS_EVAL_OFFSET_VEC_SIZE * sizeof(int));
    cudaMemcpyToSymbol(cuda::RNS_EVAL_OFFSET_MATR, &RNS_EVAL_OFFSET_MATR, RNS_MODULI_SIZE * RNS_EVAL_OFFSET_VEC_SIZE * sizeof(int));
    cudaMemcpyToSymbol(cuda::RNS_EVAL_INV_UNIT, &RNS_EVAL_INV_UNIT, sizeof(interval_t));
    cudaMemcpyToSymbol(cuda::RNS_EVAL_UNIT, &RNS_EVAL_UNIT, sizeof(interval_t));
    cudaMemcpyToSymbol(cuda::RNS_EVAL_ZERO_BOUND, &RNS_EVAL_ZERO_BOUND, sizeof(er_float_t));
    cudaMemcpyToSymbol(cuda::MRC_MULT_INV, &MRC_MULT_INV, sizeof(int) * RNS_MODULI_SIZE * RNS_MODULI_SIZE);
}

/*
 * Print main constants of the RNS
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
    std::cout << "- RNS_PARALLEL_REDUCTION_IDX: " << RNS_PARALLEL_REDUCTION_IDX << std::endl;
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
 * Calculates and prints the actual values of the interval evaluation constants
 * in accordance with the parameters from params.h
 */
void rns_eval_const_calc() {
    double offset_vec_size, eval_min_value;
    eval_min_value = RNS_MODULI_SIZE * RNS_MODULI_SIZE * pow(2.0, 1 - 53) / RNS_EVAL_ACCURACY;
    mpfr_t log2mpfr;
    mpfr_init(log2mpfr);
    mpfr_log2(log2mpfr, RNS_MODULI_PRODUCT_MPFR, MPFR_RNDN);
    double logM = mpfr_get_d(log2mpfr, MPFR_RNDN);
    double logPsi = log(eval_min_value) / log(2.0);
    offset_vec_size = (unsigned int) ceil(-(logM + logPsi) / (2 + logPsi));
    printf("Constants of the RNS interval evaluation for %i moduli:\n", RNS_MODULI_SIZE);
    printf("- RNS_EVAL_MIN_LIMIT = %.17g\n", eval_min_value);
    printf("- RNS_EVAL_OFFSET_VEC_SIZE = %.17g\n", offset_vec_size);
    mpfr_clear(log2mpfr);
}


/********************* Mixed-radix conversion functions *********************/

/*!
 * Computes the mixed-radix representation for a given RNS number
 * @param mr - pointer to the result mixed-radix representation
 * @param x - pointer to the input RNS number
 */
GCC_FORCEINLINE void perform_mrc(int *mr, int * x) {
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
 * GPU functions
 */
namespace cuda{

    /*!
     * Computes the mixed-radix representation for a given RNS number
     * @param mr - pointer to the result mixed-radix representation
     * @param x - pointer to the input RNS number
     */
    DEVICE_CUDA_FORCEINLINE void perform_mrc(int *mr, int * x) {
        for (int i = 0; i < RNS_MODULI_SIZE; i++) {
            mr[i] = x[i];
            for (int j = 0; j < i; j++) {
                if (mr[i] < mr[j]) {
                    long tmp = (long)cuda::RNS_MODULI[i] - (long)mr[j] + (long)mr[i];
                    mr[i] = (int)tmp;
                } else {
                    mr[i] = mr[i] - mr[j];
                }
                mr[i] = cuda::mod_mul(mr[i], cuda::MRC_MULT_INV[j][i], cuda::RNS_MODULI[i]);
            }
        }
    }

} //end of namespace


/********************* Functions for calculating the interval evaluation of an RNS number *********************/


/*!
 * Computes the interval evaluation for a given RNS number
 * @param eval - pointer to the result interval evaluation
 * @param x - pointer to the input RNS number
 */
__attribute__ ((optimize("O2"))) //Do not use -O3 here
void rns_eval_compute(interval_ptr eval, int * x) {
    double low = 0.0;
    double upp = 0.0;
    int s[RNS_MODULI_SIZE];

    //Calculations for the lower bound
    round_down_mode();
    for (int i = 0; i < RNS_MODULI_SIZE; i++) {
        s[i] = mod_mul(x[i], RNS_PART_MODULI_PRODUCT_INVERSE[i], RNS_MODULI[i]);
        low += s[i] / (double) RNS_MODULI[i];
    }
    //Calculations for the upper bound
    round_up_mode();
    for (int i = 0; i < RNS_MODULI_SIZE; i++) {
        upp += s[i] / (double) RNS_MODULI[i];
    }
    //Checking for zero
    round_nearest_mode();
    if (low == 0 && upp == 0) {
        er_set(&eval->low, &RNS_EVAL_ZERO_BOUND);
        er_set(&eval->upp, &RNS_EVAL_ZERO_BOUND);
        return;
    }
    //Splitting into whole and fractional parts
    unsigned int whl = (unsigned int) low; // Whole part
    unsigned int whu = (unsigned int) upp; // Whole part
    low = low - whl;    // Fractional part
    upp = upp - whu;    // Fractional part
    //Checking correctness and adjust
    bool BIG = false;
    bool TINY = false;
    if (whl != whu) { //Interval evaluation is wrong
        perform_mrc(s, x); //Mixed-radix representation of x
        if (s[RNS_MODULI_SIZE - 1] == 0) {
            TINY = true; //Number is too small, the lower bound is incorrect
            er_set(&eval->low, &RNS_EVAL_UNIT.low);
        } else{
            BIG = true; //Number is too large, the upper bound is incorrect
            er_set(&eval->upp, &RNS_EVAL_INV_UNIT.upp);
        }
    }
    /*
     * Accuracy checking
     * If the lower bound is incorrectly calculated (the number is too small), then refinement may be required;
     * If the upper bound is incorrectly calculated (the number is too large), no refinement is required.
    */
    if (BIG || upp >= RNS_EVAL_MIN_LIMIT) {
        if (!TINY)  er_set_d(&eval->low, low);
        if (!BIG)   er_set_d(&eval->upp, upp);
        return;
    }
    //Need more accuracy. Performing a refinement loop with stepwise calculation of the shifted upper bound
    round_up_mode();
    int offset = -1;
    do {
        offset++;
        upp = 0.0;
        for (int i = 0; i < RNS_MODULI_SIZE; i++) {
            s[i] = mod_mul(x[i], RNS_EVAL_OFFSET_MATR[offset][i], RNS_MODULI[i]);
            upp += s[i] / (double) RNS_MODULI[i];
        }
        upp -= (unsigned int) upp;
    } while( upp < RNS_EVAL_MIN_LIMIT && offset < (RNS_EVAL_OFFSET_VEC_SIZE-1) );
    //Setting the upper bound of eval with appropriate correction
    er_set_d(&eval->upp, upp);
    eval->upp.exp -= RNS_EVAL_OFFSET_VEC[offset];
    //Computing the shifted lower bound
    round_down_mode();
    low = 0.0;
    for (int i = 0; i < RNS_MODULI_SIZE; i++) {
        low += s[i] / (double) RNS_MODULI[i];
    }
    low -= (unsigned int) low;
    //Setting the lower bound of eval with appropriate correction
    er_set_d(&eval->low, low);
    eval->low.exp -= RNS_EVAL_OFFSET_VEC[offset];
    round_nearest_mode();
}

/*!
 * For a given RNS number, which is guaranteed not to be too large,
 * this function computes the interval evaluation faster than the previous common function.
 * @param eval - pointer to the result interval evaluation
 * @param x - pointer to the input RNS number
 */
__attribute__ ((optimize("O2"))) //Do not use -O3 here
void rns_eval_compute_fast(interval_ptr eval, int * x) {
    double low = 0.0;
    double upp = 0.0;
    int s[RNS_MODULI_SIZE];

    //Calculations for the upper bound
    round_up_mode();
    for (int i = 0; i < RNS_MODULI_SIZE; i++) {
        s[i] = mod_mul(x[i], RNS_PART_MODULI_PRODUCT_INVERSE[i], RNS_MODULI[i]);
        upp += s[i] / (double) RNS_MODULI[i];
    }
    //Calculations for the lower bound
    round_down_mode();
    for (int i = 0; i < RNS_MODULI_SIZE; i++) {
        low += s[i] / (double) RNS_MODULI[i];
    }
    //Checking for zero
    round_nearest_mode();
    if(low == 0 && upp == 0){
        er_set(&eval->low, &RNS_EVAL_ZERO_BOUND);
        er_set(&eval->upp, &RNS_EVAL_ZERO_BOUND);
        return;
    }
    //Dropping whole parts
    upp = upp - (unsigned int) upp;  //Upper bound
    low = low - (unsigned int) low;  //Lower bound
    //Accuracy checking
    if (upp >= RNS_EVAL_MIN_LIMIT) {
        er_set_d(&eval->low, low);
        er_set_d(&eval->upp, upp);
        return;
    }
    //Need more accuracy. Performing a refinement loop with stepwise calculation of the shifted upper bound
    round_up_mode();
    int offset = -1;
    do {
        offset++;
        upp = 0.0;
        for (int i = 0; i < RNS_MODULI_SIZE; i++) {
            s[i] = mod_mul(x[i], RNS_EVAL_OFFSET_MATR[offset][i], RNS_MODULI[i]);
            upp += s[i] / (double) RNS_MODULI[i];
        }
        upp -= (unsigned int) upp;
    } while( upp < RNS_EVAL_MIN_LIMIT && (offset < RNS_EVAL_OFFSET_VEC_SIZE-1) );
    //Setting the upper bound of eval with appropriate correction
    er_set_d(&eval->upp, upp);
    eval->upp.exp -= RNS_EVAL_OFFSET_VEC[offset];
    //Computing the shifted lower bound
    round_down_mode();
    low = 0.0;
    for (int i = 0; i < RNS_MODULI_SIZE; i++) {
        low += s[i] / (double) RNS_MODULI[i];
    }
    low -= (unsigned int) low;
    //Setting the lower bound of eval with appropriate correction
    er_set_d(&eval->low, low);
    eval->low.exp -= RNS_EVAL_OFFSET_VEC[offset];
    round_nearest_mode();
}


/*
 * GPU functions
 */
namespace cuda{

    /*!
     * Computes the interval evaluation for a given RNS number
     * @param eval - pointer to the result interval evaluation
     * @param x - pointer to the input RNS number
     */
    DEVICE_CUDA_FORCEINLINE void rns_eval_compute(interval_ptr eval, int * x) {
        double low = 0.0;
        double upp = 0.0;
        int s[RNS_MODULI_SIZE];

        //Straightforward computations
        for (int i = 0; i < RNS_MODULI_SIZE; i++) {
            s[i] = cuda::mod_mul(x[i], cuda::RNS_PART_MODULI_PRODUCT_INVERSE[i], cuda::RNS_MODULI[i]);
            low = __dadd_rd(low, __ddiv_rd(s[i], (double) cuda::RNS_MODULI[i]));
            upp = __dadd_ru(upp, __ddiv_ru(s[i], (double) cuda::RNS_MODULI[i]));
        }
        //Checking for zero
        if (low == 0 && upp == 0) {
            cuda::er_set(&eval->low, &cuda::RNS_EVAL_ZERO_BOUND);
            cuda::er_set(&eval->upp, &cuda::RNS_EVAL_ZERO_BOUND);
            return;
        }
        //Splitting into whole and fractional parts
        unsigned int whl = (unsigned int) (low);
        unsigned int whu = (unsigned int) (upp);
        low = __dsub_rd(low, whl);    // lower bound
        upp = __dsub_ru(upp, whu);    // upper bound

        //Checking correcntess and adjust
        bool BIG = false;
        bool TINY = false;
        if (whl != whu) { // Interval evaluation is wrong
            cuda::perform_mrc(s, x); //Mixed-radix representation of x
            if (s[RNS_MODULI_SIZE - 1] == 0) {
                TINY = true; //Number is too small, the lower bound is incorrect
                cuda::er_set(&eval->low, &cuda::RNS_EVAL_UNIT.low);
            } else{ //Number is too large, the upper bound is incorrect
                BIG = true;
                cuda::er_set(&eval->upp, &cuda::RNS_EVAL_INV_UNIT.upp);
            }
        }
        /*
         * Accuracy checking
         * If the lower bound is incorrectly calculated (the number is too small), then refinement may be required;
         * If the upper bound is incorrectly calculated (the number is too large), no refinement is required.
        */
        if (BIG || upp >= RNS_EVAL_MIN_LIMIT) {
            if (!TINY)  cuda::er_set_d(&eval->low, low);
            if (!BIG)   cuda::er_set_d(&eval->upp, upp);
            return;
        }
        //Need more accuracy. Performing a refinement loop with stepwise calculation of the shifted upper bound
        int offset = -1;
        do {
            offset++;
            upp = 0.0;
            for (int i = 0; i < RNS_MODULI_SIZE; i++) {
                s[i] = cuda::mod_mul(x[i], cuda::RNS_EVAL_OFFSET_MATR[offset][i], cuda::RNS_MODULI[i]);
                upp = __dadd_ru(upp, __ddiv_ru(s[i], (double) cuda::RNS_MODULI[i]));
            }
            upp -= (unsigned int) upp;
        } while( upp < RNS_EVAL_MIN_LIMIT && offset < (RNS_EVAL_OFFSET_VEC_SIZE-1) );
        //Setting the upper bound of eval with appropriate correction
        cuda::er_set_d(&eval->upp, upp);
        eval->upp.exp -= cuda::RNS_EVAL_OFFSET_VEC[offset];
        //Computing the shifted lower bound
        low = 0.0;
        for (int i = 0; i < RNS_MODULI_SIZE; i++) {
            low = __dadd_rd(low, __ddiv_rd(s[i], (double) cuda::RNS_MODULI[i]));
        }
        low -= (unsigned int) low;
        //Setting the lower bound of eval with appropriate correction
        cuda::er_set_d(&eval->low, low);
        eval->low.exp -= cuda::RNS_EVAL_OFFSET_VEC[offset];
    }

   /*!
    * For a given RNS number, which is guaranteed not to be too large,
    * this function computes the interval evaluation faster than the previous common function.
    * @param eval - pointer to the result interval evaluation
    * @param x - pointer to the input RNS number
    */
    DEVICE_CUDA_FORCEINLINE void rns_eval_compute_fast(interval_ptr result, int * r_numb) {
        double low = 0.0;
        double upp = 0.0;
        int s[RNS_MODULI_SIZE];

       //Straightforward computations
       for (int i = 0; i < RNS_MODULI_SIZE; i++) {
            s[i] = cuda::mod_mul(r_numb[i], cuda::RNS_PART_MODULI_PRODUCT_INVERSE[i], cuda::RNS_MODULI[i]);
            upp = __dadd_ru(upp, __ddiv_ru(s[i], (double) cuda::RNS_MODULI[i]));
            low = __dadd_rd(low, __ddiv_rd(s[i], (double) cuda::RNS_MODULI[i]));
        }
        //Checking for zero
        if (low == 0 && upp == 0) {
            cuda::er_set(&result->low, &cuda::RNS_EVAL_ZERO_BOUND);
            cuda::er_set(&result->upp, &cuda::RNS_EVAL_ZERO_BOUND);
            return;
        }
       //Dropping whole parts
        upp = upp - (unsigned int) upp;  //Upper bound
        low = low - (unsigned int) low;  //Lower bound
       //Accuracy checking
        if (upp >= RNS_EVAL_MIN_LIMIT) {
            cuda::er_set_d(&result->low, low);
            cuda::er_set_d(&result->upp, upp);
            return;
        }
       //Need more accuracy. Performing a refinement loop with stepwise calculation of the shifted upper bound
       int offset = -1;
       do {
           offset++;
           upp = 0.0;
           for (int i = 0; i < RNS_MODULI_SIZE; i++) {
               s[i] = cuda::mod_mul(r_numb[i], cuda::RNS_EVAL_OFFSET_MATR[offset][i], cuda::RNS_MODULI[i]);
               upp = __dadd_ru(upp, __ddiv_ru(s[i], (double) cuda::RNS_MODULI[i]));
           }
           upp -= (unsigned int) upp;
       } while( upp < RNS_EVAL_MIN_LIMIT && offset < (RNS_EVAL_OFFSET_VEC_SIZE-1) );
       //Setting the upper bound of eval with appropriate correction
        cuda::er_set_d(&result->upp, upp);
        result->upp.exp -= cuda::RNS_EVAL_OFFSET_VEC[offset];
       //Computing the shifted lower bound
        low = 0.0;
        for (int i = 0; i < RNS_MODULI_SIZE; i++) {
            low = __dadd_rd(low, __ddiv_rd(s[i], (double) cuda::RNS_MODULI[i]));
        }
        low -= (unsigned int) low;
       //Setting the lower bound of eval with appropriate correction
        cuda::er_set_d(&result->low, low);
        result->low.exp -= cuda::RNS_EVAL_OFFSET_VEC[offset];
    }

} //end of namespace


/********************* Power-of-two RNS scaling functions *********************/

/*
 * For a given RNS number x = (x0,...,xn), this helper function computes k such that
 * X = sum( Mi * |xi * mult.inv(Mi)|_mi ) - k * M. Array c stores the computed values
 * |xi * mult.inv(Mi)|_mi, where mult.inv(Mi) is the modulo mi multiplicative inverse of Mi, i.e. M_i^{-1} mod mi
 */
__attribute__ ((optimize("O2"))) //Do not use -O3 here
static int compute_k(int * x, int * c) {
    double s_upp = 0.0;
    double s_low = 0.0;
    int k_low, k_upp;
    round_down_mode();
    for (int i = 0; i < RNS_MODULI_SIZE; i++) {
        s_low += c[i] / (double) RNS_MODULI[i];
    }
    round_up_mode();
    for (int i = 0; i < RNS_MODULI_SIZE; i++) {
        s_upp += c[i] / (double) RNS_MODULI[i];
    }
    round_nearest_mode();
    k_low = (int) s_low;
    k_upp = (int) s_upp;
    if (k_low == k_upp) {
        return k_low;
    } else {
        int mr[RNS_MODULI_SIZE];
        perform_mrc(mr, x);
        if (mr[RNS_MODULI_SIZE - 1] == 0) {
            return k_upp;
        } else{
            return k_low;
        }
    }
}

/*
 * For a given RNS number x = (x0,...,xn), which is guaranteed not to be too large, this helper function computes k such that
 * X = sum( Mi * |xi * mult.inv(Mi)|_mi ) - k * M. Array c stores the computed values
 * |xi * mult.inv(Mi)|_mi, where mult.inv(Mi) is the modulo mi multiplicative inverse of Mi, i.e. M_i^{-1} mod mi
 * This function performs faster than the previous common function.
 */
__attribute__ ((optimize("O2"))) //Do not use -O3 here
static int compute_k_fast(int * x, int * c) {
    double s = 0.0;
    round_up_mode();
    for (int i = 0; i < RNS_MODULI_SIZE; i++) {
        s += c[i] / (double) RNS_MODULI[i];
    }
    round_nearest_mode();
    return (int) s;
}

/*
 * This helper function performs one step of scaling by a power of two
 */
static void scale2powj(int *y, int k, unsigned int j, int pow2j, int * x, int * c) {
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
void rns_scale2pow(int * result, int * x, unsigned int D) {
    rns_set(result, x); // result <- x
    int t = D / RNS_P2_SCALING_THRESHOLD;
    int k = 0;
    int c[RNS_MODULI_SIZE];
    //first step
    if (t > 0) {
        rns_mul(c, x, RNS_PART_MODULI_PRODUCT_INVERSE);
        k = compute_k(x, c);
        scale2powj(result, k, RNS_P2_SCALING_THRESHOLD, RNS_P2_SCALING_FACTOR, x, c);
        t -= 1;
    }
    //second step
    while (t > 0) {
        rns_mul(c, result, RNS_PART_MODULI_PRODUCT_INVERSE);
        k = compute_k_fast(result, c);
        scale2powj(result, k, RNS_P2_SCALING_THRESHOLD, RNS_P2_SCALING_FACTOR, result, c);
        t -= 1;
    }
    //third step
    unsigned int d = D % RNS_P2_SCALING_THRESHOLD;
    if (d > 0) {
        rns_mul(c, result, RNS_PART_MODULI_PRODUCT_INVERSE);
        k = d < D ? compute_k_fast(result, c) : compute_k(result, c);
        scale2powj(result, k, d, 1 << d, result, c);
    }
}


/*
 * GPU functions
 */
namespace cuda{

    ///// Single-threaded functions /////

    /*
     * For a given RNS number x = (x0,...,xn), this helper function computes k such that
     * X = sum( Mi * |xi * mult.inv(Mi)|_mi ) - k * M. Array c stores the computed values
     * |xi * mult.inv(Mi)|_mi, where mult.inv(Mi) is the modulo mi multiplicative inverse of Mi, i.e. M_i^{-1} mod mi
     */
    DEVICE_CUDA_FORCEINLINE static int compute_k(int * x, int * c) {
        double s_upp = 0.0;
        double s_low = 0.0;
        int k_low, k_upp;
        int mr[RNS_MODULI_SIZE];
        for (int i = 0; i < RNS_MODULI_SIZE; i++) {
            s_low = __dadd_rd(s_low, __ddiv_rd((double) c[i], (double) cuda::RNS_MODULI[i]));
            s_upp = __dadd_ru(s_upp, __ddiv_ru((double) c[i], (double) cuda::RNS_MODULI[i]));
        }
        k_low = (int) s_low;
        k_upp = (int) s_upp;
        if (k_low == k_upp) {
            return k_low;
        } else {
            cuda::perform_mrc(mr, x);
            if (mr[RNS_MODULI_SIZE - 1] == 0) {
                return k_upp;  //Number is too small
            } else{
                return k_low; //Number is too large
            }
        }
    }

    /*
     * For a given RNS number x = (x0,...,xn), which is guaranteed not to be too large, this helper function computes k such that
     * X = sum( Mi * |xi * mult.inv(Mi)|_mi ) - k * M. Array c stores the computed values
     * |xi * mult.inv(Mi)|_mi, where mult.inv(Mi) is the modulo mi multiplicative inverse of Mi, i.e. M_i^{-1} mod mi
     * This function performs faster than the previous common function.
     */
    DEVICE_CUDA_FORCEINLINE static int compute_k_fast(int * x, int * c) {
        double s = 0.0;
        for (int i = 0; i < RNS_MODULI_SIZE; i++) {
            s = __dadd_ru(s, __ddiv_ru(c[i], (double) cuda::RNS_MODULI[i]));
        }
        return (int) s;
    }

    /*
     * This helper function performs one step of scaling by a power of two
     */
    DEVICE_CUDA_FORCEINLINE static void scale2powj(int * y, int k, unsigned int j, int pow2j, int * x, int * c) {
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
            k = cuda::compute_k(x, c);
            cuda::scale2powj(result, k, RNS_P2_SCALING_THRESHOLD, RNS_P2_SCALING_FACTOR, x, c);
            t -= 1;
        }
        //second step
        while (t > 0) {
            cuda::rns_mul(c, result, cuda::RNS_PART_MODULI_PRODUCT_INVERSE);
            k = cuda::compute_k_fast(result, c);
            cuda::scale2powj(result, k, RNS_P2_SCALING_THRESHOLD, RNS_P2_SCALING_FACTOR, result, c);
            t -= 1;
        }
        //third step
        unsigned int d = D % RNS_P2_SCALING_THRESHOLD;
        if (d > 0) {
            cuda::rns_mul(c, result, cuda::RNS_PART_MODULI_PRODUCT_INVERSE);
            k = d < D ? cuda::compute_k_fast(result, c) : cuda::compute_k(result, c);
            cuda::scale2powj(result, k, d, 1 << d, result, c);
        }
    }

    ///// Multi-threaded functions /////

    /*
     * For a given RNS number x = (x0,...,xn), this helper function computes k such that
     * X = sum( Mi * |xi * mult.inv(Mi)|_mi ) - k * M. Array c stores the computed values
     * |xi * mult.inv(Mi)|_mi, where mult.inv(Mi) is the modulo mi multiplicative inverse of Mi, i.e. M_i^{-1} mod mi
     * This function must be performed by n threads simultaneously within a single thread block.
     */
    DEVICE_CUDA_FORCEINLINE static void compute_k_thread(int * result, int * x, int c) {
        int residueId = threadIdx.x;
        int modulus = cuda::RNS_MODULI[residueId];
        __shared__ double s_upp[RNS_MODULI_SIZE];
        __shared__ double s_low[RNS_MODULI_SIZE];
        __shared__ bool return_flag;
        return_flag = false;
        int k_low, k_upp;
        s_low[residueId] = __ddiv_rd(c, (double) modulus);
        s_upp[residueId] = __ddiv_ru(c, (double) modulus);
        __syncthreads();
        for (unsigned int s = RNS_PARALLEL_REDUCTION_IDX; s > 0; s >>= 1) {
            if (residueId < s && residueId + s < RNS_MODULI_SIZE) {
                s_low[residueId] = __dadd_rd((double) s_low[residueId], (double) s_low[residueId + s]);
                s_upp[residueId] = __dadd_ru((double) s_upp[residueId], (double) s_upp[residueId + s]);
            }
            __syncthreads();
        }
        k_low = (int) s_low[residueId];
        k_upp = (int) s_upp[residueId];
        if (residueId == 0) {
            if (k_low == k_upp) {
                *result = k_low;
                return_flag = true;
            }
        }
        __syncthreads();
        if (return_flag) {
            return;
        } else {
            if (residueId == 0) {
                int mr[RNS_MODULI_SIZE];
                cuda::perform_mrc(mr, x); // parallel MRC should be used, see http://dx.doi.org/10.1109/ISCAS.2009.5117800
                if (mr[RNS_MODULI_SIZE - 1] == 0) {
                    *result = k_upp;
                } else{
                    *result = k_low;
                }
            }
        }
        __syncthreads();
    }

    /*
     * For a given RNS number x = (x0,...,xn), which is guaranteed not to be too large, this helper function computes k such that
     * X = sum( Mi * |xi * mult.inv(Mi)|_mi ) - k * M. Array c stores the computed values
     * |xi * mult.inv(Mi)|_mi, where mult.inv(Mi) is the modulo mi multiplicative inverse of Mi, i.e. M_i^{-1} mod mi
     * This function performs faster than the previous common function.
     * This function must be performed by n threads simultaneously within a single thread block.
     */
    DEVICE_CUDA_FORCEINLINE static void compute_k_fast_thread(int * result, int * x, int c){
        int residueId = threadIdx.x;
        __shared__ double S[RNS_MODULI_SIZE];
        S[residueId] = __ddiv_ru((double) c, (double) cuda::RNS_MODULI[residueId]);
        __syncthreads();
        for (unsigned int s = RNS_PARALLEL_REDUCTION_IDX; s > 0; s >>= 1) {
            if (residueId < s && residueId + s < RNS_MODULI_SIZE) {
                S[residueId] = __dadd_ru(S[residueId], S[residueId + s]);
            }
            __syncthreads();
        }
        if (residueId == 0)
            *result = (int) S[0];
        __syncthreads();
    }

    /*
     * This helper function performs one step of scaling by a power of two
     * This function must be performed by n threads simultaneously within a single thread block.
     */
    DEVICE_CUDA_FORCEINLINE static void scale2powj_thread(int * y, int k, unsigned int j, int pow2j, int * x, int c) {
        int residueId = threadIdx.x;
        int modulus = cuda::RNS_MODULI[residueId];
        int multiple;
        __shared__ int residue[RNS_MODULI_SIZE]; // X mod 2^j
        //RNS_PART_MODULI_PRODUCT_POW2_RESIDUES[j-1][i] ->  M_i mod 2^j
        residue[residueId] = cuda::mod_mul(cuda::RNS_PART_MODULI_PRODUCT_POW2_RESIDUES[j - 1][residueId], c, pow2j);// (cuda::RNS_PART_MODULI_PRODUCT_POW2_RESIDUES[j - 1][residueId] * terms) % pow2j;
        __syncthreads();
        for (unsigned int s = RNS_PARALLEL_REDUCTION_IDX; s > 0; s >>= 1) {
            if (residueId < s && residueId + s < RNS_MODULI_SIZE) {
                residue[residueId] = residue[residueId] + residue[residueId + s];
            }
            __syncthreads();
        }
        //RNS_MODULI_PRODUCT_POW2_RESIDUES[j-1] ->  M mod 2^j
        if(residueId == 0){
            residue[0] = (residue[0] - k * cuda::RNS_MODULI_PRODUCT_POW2_RESIDUES[j - 1]) % pow2j;
            if(residue[0] < 0){
                residue[0] += pow2j;
            }
        }
        __syncthreads();
        residue[residueId] = residue[0];
        multiple = residue[residueId] % modulus;
        multiple = x[residueId] - multiple;
        if (multiple < 0) {
            multiple += modulus;
        }
        //RNS_POW2_INVERSE[j-1][i] -> (2^j )^{-1} mod m_i
        y[residueId] = cuda::mod_mul(multiple, cuda::RNS_POW2_INVERSE[j - 1][residueId], modulus);   //( multiple * cuda::RNS_POW2_INVERSE[j - 1][residueId] ) % modulus;
    }

    /*!
     * Parallel (n threads) scaling an RNS number by a power of 2: result = x / 2^D.
     * This function must be performed by n threads simultaneously within a single thread block.
     * @param result - pointer to the result (scaled number)
     * @param x - pointer to the RNS number to be scaled
     * @param D - exponent of the scaling factor
     */
    DEVICE_CUDA_FORCEINLINE void rns_scale2pow_thread(int * result, int * x, unsigned int D) {
        result[threadIdx.x] = x[threadIdx.x];
        __shared__ int k;
        int t;
        int c;
        int residueId = threadIdx.x;
        int modulus = cuda::RNS_MODULI[residueId];
        int inverse = cuda::RNS_PART_MODULI_PRODUCT_INVERSE[residueId];

        t = D / RNS_P2_SCALING_THRESHOLD;
        __syncthreads();
        //first step
        if (t > 0) {
            c = (x[residueId] * inverse) % modulus;
            cuda::compute_k_thread(&k, x, c);
            cuda::scale2powj_thread(result, k, RNS_P2_SCALING_THRESHOLD, RNS_P2_SCALING_FACTOR, x, c);
            t -= 1;
        }
        __syncthreads();
        //second step
        while (t > 0) {
            c = (result[residueId] * inverse) % modulus;
            cuda::compute_k_fast_thread(&k, result, c);
            cuda::scale2powj_thread(result, k, RNS_P2_SCALING_THRESHOLD, RNS_P2_SCALING_FACTOR, result, c);
            t -= 1;
            __syncthreads();
        }
        //third step
        unsigned int d = D % RNS_P2_SCALING_THRESHOLD;
        if (d > 0) {
            c =  cuda::mod_mul(result[residueId], inverse, modulus);
            if (d < D) {
                cuda::compute_k_fast_thread(&k, result, c);
            } else {
                cuda::compute_k_thread(&k, result, c);
            }
            cuda::scale2powj_thread(result, k, d, 1 << d, result, c);
            __syncthreads();
        }
    }

} //end of namespace

#endif //MPRES_RNS_CUH
