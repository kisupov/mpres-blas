/*
 *  Data generation and conversion functions
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


#ifndef MPRES_TEST_TSTHELPER_CUH
#define MPRES_TEST_TSTHELPER_CUH

#include "omp.h"
#include <random>
#include <chrono>
#include <timers.cuh>
#include "../src/arith/mpassign.cuh"
#include "../src/arith/mpadd.cuh"


/*
 * Creates an array of random multiple-precision floating-point numbers
 */
mpfr_t * create_random_array(unsigned long size, int bits){
    waitFor(5);
    srand(time(NULL));

    gmp_randstate_t state;                           // Random generator state object
    gmp_randinit_default(state);                     // Initialize state for a Mersenne Twister algorithm
    gmp_randseed_ui(state, (unsigned) time(NULL));   // Call gmp_randseed_ui to set initial seed value into state

    std::uniform_real_distribution<double> unif(-1, 1);
    std::default_random_engine re;
    re.seed(std::chrono::system_clock::now().time_since_epoch().count());

    mpz_t random_number;
    mpz_init2(random_number, bits);
    mpfr_t pow_bits;
    mpfr_init2(pow_bits, bits);
    mpfr_set_d(pow_bits, 2, MPFR_RNDN);
    mpfr_pow_si(pow_bits, pow_bits, -1 * bits, MPFR_RNDN);

    mpfr_t* array = new mpfr_t[size];

    for(int i = 0; i < size; i ++){
        mpfr_init2(array[i], bits);
    }

    for (int i = 0; i < size; i++) {
        mpz_urandomb(random_number, state, bits);
        //Generate a uniformly distributed random double x
        mpfr_set_z(array[i], random_number, MPFR_RNDD);
        mpfr_mul_d(array[i], array[i], unif(re), MPFR_RNDN);
        mpfr_mul(array[i], array[i], pow_bits, MPFR_RNDN);
    }
    return array;
}


/*
 * Converts a mpfr_t number to a string of digits in scientific notation, e.g. -0.12345e13.
 * ndigits is the number of significant digits output in the string
 */
std::string convert_to_string_sci(mpfr_t number, int ndigits){
    char * significand;
    long exp = 0;
    //Convert number to a string of digits in base 10
    significand = mpfr_get_str(NULL, &exp, 10, ndigits, number, MPFR_RNDN);
    //Convert to std::string
    std::string number_string(significand);
    //Set decimal point
    if(number_string.compare(0, 1, "-") == 0){
        number_string.insert(1, "0.");
    }else {
        number_string.insert(0, "0.");
    }
    //Add the exponent
    number_string += "e";
    number_string += std::to_string(exp);
    //Cleanup
    mpfr_free_str(significand);
    return number_string;
}


/*
 * Converts a mpfr_t number to a string of digits in fixed-point notation, e.g. -0.1234567.
 * ndigits is the number of significant digits output in the string
 */
std::string convert_to_string_fix(mpfr_t number, int ndigits){
    char * significand;
    long exp = 0;
    //Convert number to a string of digits in base 10
    significand = mpfr_get_str(NULL, &exp, 10, ndigits, number, MPFR_RNDN);
    std::string number_string(significand);
    std::string zeroes = "";
    for(int i = 0; i < abs(exp); i++){
        zeroes.insert(0, "0");
    }
    int insert_offset = number_string.compare(0, 1, "-") == 0;
    if(exp > 0){
        //Add zeroes to the end of string
        number_string.append(zeroes);
    } else{
        //Add zeroes to the start of string
        number_string.insert(insert_offset, zeroes);

    }
    number_string.insert(insert_offset, "0.");
    mpfr_free_str(significand);
    return number_string;
}

/*
 * Prints the sum of array elements
 */
void print_double_sum(double *arr, int n) {
    double sum = 0;
    for (int i = 0; i < n; i++) {
        sum += arr[i];
    }
    printf("result: %.70f\n", sum);
}

/*
 * Prints the sum of array elements
 */
void print_mp_sum(mp_float_ptr arr, int n) {
    mp_float_t print_result;
    print_result = MP_ZERO;
    mpfr_t sum;
    mpfr_init2(sum, MP_PRECISION * 10);
    mpfr_set_d(sum, 0.0, MPFR_RNDN);
    for (int i = 0; i < n; i+= 1) {
        mp_add(&print_result, &print_result, &arr[i]);
    }
    mp_get_mpfr(sum, &print_result);
    mpfr_printf("result: %.70Rf \n", sum);
    mpfr_clear(sum);
}

/*
 * Prints the sum of array elements
 */
void print_mpfr_sum(mpfr_t *arr, int n) {
    mpfr_t sum;
    mpfr_init2(sum, MP_PRECISION * 10);
    mpfr_set_d(sum, 0.0, MPFR_RNDN);
    for (int i = 0; i < n; i++) {
        mpfr_add(sum, sum, arr[i], MPFR_RNDN);
    }
    mpfr_printf("result: %.70Rf\n", sum);
    mpfr_clear(sum);
}

/*
 * Converts a multiple precision mpfr_t vector to a double precision vector
 */
void convert_vector(double * dest, const mpfr_t *source, int width){
    #pragma omp parallel for
    for( int i = 0; i < width; i++ ){
        dest[i] = mpfr_get_d(source[i], MPFR_RNDN);
    }
}

/*
 * Converts a multiple-precision mpfr_t vector to a multiple precision mp_float_ptr vector
 */
void convert_vector(mp_float_ptr dest, const mpfr_t *source, int width){
    #pragma omp parallel for
    for( int i = 0; i < width; i++ ){
        mp_set_mpfr(&dest[i], source[i]);
    }
}

/*
 * Converts a double precision vector to a multiple precision mp_float_ptr vector
 */
void convert_vector(mp_float_ptr dest, const double *source, int width){
    #pragma omp parallel for
    for( int i = 0; i < width; i++ ){
        mp_set_d(&dest[i], source[i]);
    }
}

static void convert_matrix(mp_float_ptr dest, mpfr_t *source, int rows, int cols){
    int width = rows * cols;
    #pragma omp parallel for
    for( int i = 0; i < width; i++ ){
        mp_set_mpfr(&dest[i], source[i]);
    }
}

/*
 * Return the size in MB of n-element double precision array
 */
double get_double_array_size_in_mb(size_t n){
    return double(sizeof(double)) * n / double(1024 * 1024);
}

/*
 * Return the size in MB of n-element integer array
 */
double get_int_array_size_in_mb(size_t n){
    return double(sizeof(int)) * n / double(1024 * 1024);
}

/*
 * Return the size in MB of n-element mp_float_t array
 */
double get_mp_float_array_size_in_mb(size_t n){
    return double(sizeof(mp_float_t)) * n / double(1024 * 1024);
}

/***************************************************/
/* Helper methods for sparse matrices              */
/***************************************************/

/*
 * Displays the memory consumption of a double precision CSR structure and returns the total size of the structure in MB
 */
double print_dbl_csr_memory_consumption(const int m, const int nnz){
    double sizeOfAs = get_double_array_size_in_mb(nnz);
    double sizeOfCsr = sizeOfAs + get_int_array_size_in_mb(nnz) + get_int_array_size_in_mb(m + 1);
    printf("\tMatrix (AS array) size (MB): %lf\n", sizeOfAs);
    printf("\tCSR structure size (MB): %lf\n", sizeOfCsr);
    return sizeOfCsr;
}

/*
 * Displays the memory consumption of a double precision ELLPACK structure and returns the total size of the structure in MB
 */
double print_dbl_ell_memory_consumption(const int m, const int maxnzr){
    double sizeOfAs = get_double_array_size_in_mb(m * maxnzr);
    double sizeOfEll = sizeOfAs + get_int_array_size_in_mb(m * maxnzr);
    printf("\tMatrix (AS array) size (MB): %lf\n", sizeOfAs);
    printf("\tELL structure size (MB): %lf\n", sizeOfEll);
    return sizeOfEll;
}

/*
 * Displays the memory consumption of a double precision JAD structure and returns the total size of the structure in MB
 */
double print_dbl_jad_memory_consumption(const int m, const int n, const int nnz, const int maxnzr){
    double sizeOfAs = get_double_array_size_in_mb(nnz);
    double sizeOfJad = sizeOfAs + get_int_array_size_in_mb(nnz) + get_int_array_size_in_mb(maxnzr + 1)+ get_int_array_size_in_mb(m);
    printf("\tMatrix (AS array) size (MB): %lf\n", sizeOfAs);
    printf("\tJAD structure size (MB): %lf\n", sizeOfJad);
    return sizeOfJad;
}

/*
 * Displays the memory consumption of a double precision DIA structure and returns the total size of the structure in MB
 */
double print_dbl_dia_memory_consumption(const int m, const int ndiag){
    double sizeOfAs = get_double_array_size_in_mb(m * ndiag);
    double sizeOfDia = sizeOfAs + get_int_array_size_in_mb(ndiag);
    printf("\tMatrix (AS array) size (MB): %lf\n", sizeOfAs);
    printf("\tDIA structure size (MB): %lf\n", sizeOfDia);
    return sizeOfDia;
}

#endif //MPRES_TEST_TSTHELPER_CUH
