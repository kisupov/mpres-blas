/*
 *  Accuracy test for the DOT (mp_array_dot) routine
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

#include "../../tsthelper.cuh"
#include "../../logger.cuh"
#include "../../../src/blas/mpdot.cuh"

#define SIZE 1000000 //Operation size
//Execution configuration for mp_array_dot
#define MPRES_CUDA_BLOCKS_FIELDS_ROUND   512
#define MPRES_CUDA_THREADS_FIELDS_ROUND  128
#define MPRES_CUDA_BLOCKS_RESIDUES       8192
#define MPRES_CUDA_BLOCKS_REDUCE         64
#define MPRES_CUDA_THREADS_REDUCE        64
int REFERENCE_PRECISION; //Precision for reference results

/*
 * Computes the error bounds
 */
static void compute_bounds(mpfr_t *x, mpfr_t *y){
    //Compute unit roundoff
    mpfr_t u;
    mpfr_init2(u, REFERENCE_PRECISION);
    mpfr_sqrt(u, RNS_MODULI_PRODUCT_MPFR, MPFR_RNDN);
    mpfr_ui_div(u, 4, u, MPFR_RNDN);
    //Compute error bounds
    mpfr_t rel_bound;
    mpfr_t abs_bound;
    mpfr_t tmp;
    mpfr_t cond_upp;
    mpfr_t cond_down;
    mpfr_init2(rel_bound, REFERENCE_PRECISION);
    mpfr_init2(abs_bound, REFERENCE_PRECISION);
    mpfr_init2(tmp, REFERENCE_PRECISION);
    mpfr_init2(cond_upp, REFERENCE_PRECISION);
    mpfr_init2(cond_down, REFERENCE_PRECISION);
    mpfr_set_d(cond_upp, 0, MPFR_RNDN);
    mpfr_set_d(cond_down, 0, MPFR_RNDN);
    //Compute the parts of the condition number: cond_upp = summ of |x*y| and cond_down = |summ of x * y|
    for(int i = 0; i < SIZE; i ++){
        mpfr_mul(tmp, x[i], y[i], MPFR_RNDN);
        mpfr_add(cond_down, cond_down, tmp, MPFR_RNDN);
        mpfr_abs(tmp, tmp, MPFR_RNDN);
        mpfr_add(cond_upp, cond_upp, tmp, MPFR_RNDN);
    }
    mpfr_abs(cond_down, cond_down, MPFR_RNDN);
    mpfr_mul_ui(tmp, u, SIZE, MPFR_RNDN); // u * N
    mpfr_ui_sub(tmp, 1, tmp, MPFR_RNDN);  // 1 -  u * N
    mpfr_mul_ui(rel_bound, u, SIZE, MPFR_RNDN); // u * N
    mpfr_div(rel_bound, rel_bound, tmp, MPFR_RNDN); // u * N / (  1 -  u * N )
    mpfr_mul(abs_bound, cond_upp, rel_bound, MPFR_RNDN); //  u * N / (  1 -  u * N ) * sum of |x * y|
    mpfr_div(cond_upp, cond_upp, cond_down, MPFR_RNDN); // condition number
    mpfr_mul(rel_bound, rel_bound, cond_upp, MPFR_RNDN); // u * N / (  1 -  u * N ) * condition number

    Logger::printDash();
    mpfr_printf("Unit Roundoff: %.25Re\n", u);
    Logger::printDash();
    mpfr_printf("Computed absolute error bound (MPRES-BLAS): %.25Re\n", abs_bound);
    mpfr_printf("Computed relative error bound (MPRES-BLAS): %.25Re\n", rel_bound);
    Logger::printDash();

    mpfr_clear(u);
    mpfr_clear(rel_bound);
    mpfr_clear(abs_bound);
    mpfr_clear(tmp);
    mpfr_clear(cond_upp);
    mpfr_clear(cond_down);
}

/*
 * Computes the actual errors
 */
static void evaluate_accuracy(mpfr_t reference, mp_float_t mp_result, mpfr_t mpfr_result){

    mpfr_t converted;
    mpfr_t rel_error;
    mpfr_t abs_error;

    mpfr_init2(converted, REFERENCE_PRECISION);
    mpfr_init2(rel_error, REFERENCE_PRECISION);
    mpfr_init2(abs_error, REFERENCE_PRECISION);
    mpfr_set_d(abs_error, 0, MPFR_RNDN);
    mpfr_set_d(rel_error, 0, MPFR_RNDN);

    mp_get_mpfr(converted, &mp_result);
    mpfr_sub(abs_error, reference, converted, MPFR_RNDN);
    mpfr_abs(abs_error, abs_error, MPFR_RNDN);
    mpfr_div(rel_error, abs_error, reference, MPFR_RNDN);
    mpfr_printf("Actual absolute error (MPRES-BLAS): %.25Re\n", abs_error);
    mpfr_printf("Actual relative error (MPRES-BLAS): %.25Re\n", rel_error);

    mpfr_set_d(abs_error, 0, MPFR_RNDN);
    mpfr_set_d(rel_error, 0, MPFR_RNDN);
    mpfr_sub(abs_error, reference, mpfr_result, MPFR_RNDN);
    mpfr_abs(abs_error, abs_error, MPFR_RNDN);
    mpfr_div(rel_error, abs_error, reference, MPFR_RNDN);
    mpfr_printf("Actual absolute error (MPFR): %.25Re\n", abs_error);
    mpfr_printf("Actual relative error (MPFR): %.25Re\n", rel_error);

    mpfr_clear(converted);
    mpfr_clear(abs_error);
    mpfr_clear(rel_error);
}

/*
 * Computes dot product using the MPFR library
 */
static void mpfr_dot(mpfr_t result, mpfr_t * x, mpfr_t * y, int n, int precision){
    mpfr_t prod;
    mpfr_init2(prod, precision);
    mpfr_set_d(result, 0.0, MPFR_RNDN);
    for(int i = 0; i < n; i++){
        mpfr_mul(prod, x[i], y[i], MPFR_RNDN);
        mpfr_add(result, result, prod, MPFR_RNDN);
    }
    mpfr_clear(prod);
}


int main(){
    cudaDeviceReset();
    rns_const_init();
    mp_const_init();

    //Start logging
    Logger::beginTestDescription(Logger::BLAS_DOT_ACCURACY_TEST);
    Logger::printTestParameters(SIZE, 1, MP_PRECISION, (int)(MP_PRECISION / 3.32 + 1));
    Logger::beginSection("Additional info:");
    Logger::printParam("RNS_MODULI_SIZE", RNS_MODULI_SIZE);
    Logger::printParam("MPRES_CUDA_BLOCKS_FIELDS_ROUND", MPRES_CUDA_BLOCKS_FIELDS_ROUND);
    Logger::printParam("MPRES_CUDA_THREADS_FIELDS_ROUND", MPRES_CUDA_THREADS_FIELDS_ROUND);
    Logger::printParam("MPRES_CUDA_BLOCKS_RESIDUES", MPRES_CUDA_BLOCKS_RESIDUES);
    Logger::printParam("MPRES_CUDA_BLOCKS_REDUCE", MPRES_CUDA_BLOCKS_REDUCE);
    Logger::printParam("MPRES_CUDA_THREADS_REDUCE", MPRES_CUDA_THREADS_REDUCE);
    Logger::printDash();
    rns_const_print(true);
    Logger::printDash();
    rns_eval_const_calc();
    Logger::printDash();
    mp_const_print();
    Logger::endSection(true);
    Logger::printSpace();

    //Reference inputs
    REFERENCE_PRECISION = MP_PRECISION * 5;
    mpfr_t * reference_x  = new mpfr_t[SIZE];
    mpfr_t * reference_y = new mpfr_t[SIZE];

    //Inputs
    mpfr_t * mpfr_x = create_random_array(SIZE, MP_PRECISION);
    mpfr_t * mpfr_y = create_random_array(SIZE, MP_PRECISION);

    //Setting the reference inputs
    for(int i = 0; i < SIZE; i ++){
        mpfr_init2(reference_x[i], REFERENCE_PRECISION);
        mpfr_init2(reference_y[i], REFERENCE_PRECISION);
        mpfr_set(reference_x[i], mpfr_x[i], MPFR_RNDN);
        mpfr_set(reference_y[i], mpfr_y[i], MPFR_RNDN);
    }

    //Computing the error bounds
    compute_bounds(reference_x, reference_y);

    //Reference
    mpfr_t reference_res;
    mpfr_init2(reference_res, REFERENCE_PRECISION);
    mpfr_dot(reference_res, reference_x, reference_y, SIZE, REFERENCE_PRECISION);

    //Free the reference inputs
    for(int i = 0; i < SIZE; i++){
        mpfr_clear(reference_x[i]);
        mpfr_clear(reference_y[i]);
    }
    delete [] reference_x;
    delete [] reference_y;

    //MPFR with MP_PRECISION bits of precision
    mpfr_t mpfr_result;
    mpfr_init2(mpfr_result, MP_PRECISION);
    mpfr_dot(mpfr_result, mpfr_x, mpfr_y, SIZE, MP_PRECISION);

    //MPRES-BLAS inputs
    mp_float_t mp_result = MP_ZERO;
    mp_float_ptr mp_x = new mp_float_t[SIZE];
    mp_float_ptr mp_y = new mp_float_t[SIZE];

    //Setting the MPRES-BLAS inputs
    for(int i = 0; i < SIZE; i ++){
        mp_set_mpfr(&mp_x[i], mpfr_x[i]);
        mp_set_mpfr(&mp_y[i], mpfr_y[i]);
    }

    //Free the MPFR inputs
    for(int i = 0; i < SIZE; i++){
        mpfr_clear(mpfr_x[i]);
        mpfr_clear(mpfr_y[i]);
    }
    delete [] mpfr_x;
    delete [] mpfr_y;

    //GPU data
    mp_array_t mp_dev_x;
    mp_array_t mp_dev_y;
    mp_array_t mp_dev_buffer;
    mp_array_t mp_dev_product;

    cuda::mp_array_init(mp_dev_x, SIZE);
    cuda::mp_array_init(mp_dev_y, SIZE);
    cuda::mp_array_init(mp_dev_buffer, SIZE);
    cuda::mp_array_init(mp_dev_product, 1);

    //Data transfer
    cuda::mp_array_host2device(mp_dev_x, mp_x, SIZE);
    cuda::mp_array_host2device(mp_dev_y, mp_y, SIZE);

    //Free the MPRES-BLAS HOST inputs
    delete [] mp_x;
    delete [] mp_y;

    //Computing the MPRES-BLAS dot product
    cuda::mp_array_dot<
            MPRES_CUDA_BLOCKS_FIELDS_ROUND,
            MPRES_CUDA_THREADS_FIELDS_ROUND,
            MPRES_CUDA_BLOCKS_RESIDUES,
            MPRES_CUDA_BLOCKS_REDUCE,
            MPRES_CUDA_THREADS_REDUCE>
            (SIZE, mp_dev_x, 1, mp_dev_y, 1, mp_dev_product, mp_dev_buffer);

    cudaDeviceSynchronize();
    cuda::mp_array_device2host(&mp_result, mp_dev_product, 1);

    //Free the MPRES-BLAS DEVICE inputs
    cuda::mp_array_clear(mp_dev_x);
    cuda::mp_array_clear(mp_dev_y);
    cuda::mp_array_clear(mp_dev_product);
    cuda::mp_array_clear(mp_dev_buffer);

    //Accuracy evaluation
    evaluate_accuracy(reference_res, mp_result, mpfr_result);

    //Free remain results
    mpfr_clear(reference_res);
    mpfr_clear(mpfr_result);

    //End logging
    Logger::endTestDescription();
}
