/*
 *  Accuracy test for the ASUM (mp_asum) routine
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

#include "../../../src/mparray.cuh"
#include "../../../src/blas/asum.cuh"
#include "../../tsthelper.cuh"
#include "../../logger.cuh"

#define SIZE 1000000 //Operation size
//Execution configuration for mp_asum
#define MPRES_CUDA_BLOCKS_REDUCE   256
#define MPRES_CUDA_THREADS_REDUCE  128
int REFERENCE_PRECISION; //Precision for reference results

/*
 * Computes the error bounds
 */
static void compute_bounds(mpfr_t sum) {
    //Compute unit roundoff
    mpfr_t u;
    mpfr_init2(u, REFERENCE_PRECISION);
    mpfr_sqrt(u, RNS_MODULI_PRODUCT_MPFR, MPFR_RNDN);
    mpfr_ui_div(u, 4, u, MPFR_RNDN);
    //Compute error bounds
    mpfr_t rel_bound;
    mpfr_t abs_bound;
    mpfr_t tmp;
    mpfr_init2(rel_bound, REFERENCE_PRECISION);
    mpfr_init2(abs_bound, REFERENCE_PRECISION);
    mpfr_init2(tmp, REFERENCE_PRECISION);
    mpfr_mul_ui(tmp, u, (SIZE-1), MPFR_RNDN); // u *  (N-1)
    mpfr_ui_sub(tmp, 1, tmp, MPFR_RNDN);  // 1 - u * (N-1)
    mpfr_mul_ui(rel_bound, u, (SIZE-1), MPFR_RNDN); // u * (N-1)
    mpfr_div(rel_bound, rel_bound, tmp, MPFR_RNDN); //  rel_bound = u * (N-1) / (1 - u * (N-1))
    mpfr_mul(abs_bound, sum, rel_bound, MPFR_RNDN); //  rel_bound * asum

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


int main() {
    cudaDeviceReset();
    rns_const_init();
    mp_const_init();

    //Start logging
    Logger::beginTestDescription(Logger::BLAS_ASUM_ACCURACY_TEST);
    Logger::printTestParameters(SIZE, 1, MP_PRECISION, (int)(MP_PRECISION / 3.32 + 1));
    Logger::beginSection("Additional info:");
    Logger::printParam("RNS_MODULI_SIZE", RNS_MODULI_SIZE);
    Logger::printParam("MPRES_CUDA_BLOCKS_REDUCE", MPRES_CUDA_BLOCKS_REDUCE);
    Logger::printParam("MPRES_CUDA_THREADS_REDUCE", MPRES_CUDA_THREADS_REDUCE);
    Logger::printDash();
    rns_const_print(true);
    Logger::printDash();
    rns_eval_const_print();
    Logger::printDash();
    mp_const_print();
    Logger::endSection(true);
    Logger::printSpace();

    //Reference data
    REFERENCE_PRECISION = MP_PRECISION * 10;
    mpfr_t *reference_x = new mpfr_t[SIZE];
    mpfr_t reference_res;

    //MPFR results with MP_PRECISION bits of precision
    mpfr_t *mpfr_x = create_random_array(SIZE, MP_PRECISION);
    mpfr_t mpfr_result;

    //MPRES-BLAS results with MP_PRECISION bits of precision
    mp_float_ptr mp_x = new mp_float_t[SIZE];
    mp_float_t mp_result = MP_ZERO;
    mp_array_t mp_dev_x;
    mp_array_t mp_dev_result;

    cuda::mp_array_init(mp_dev_x, SIZE);
    cuda::mp_array_init(mp_dev_result, 1);

    mpfr_init2(reference_res, REFERENCE_PRECISION);
    mpfr_init2(mpfr_result, MP_PRECISION);
    mpfr_set_d(reference_res, 0, MPFR_RNDN);
    mpfr_set_d(mpfr_result, 0, MPFR_RNDN);

    for (int i = 0; i < SIZE; i++) {
        mpfr_init2(reference_x[i], REFERENCE_PRECISION);
        mpfr_set(reference_x[i], mpfr_x[i], MPFR_RNDN);
        mp_set_mpfr(&mp_x[i], mpfr_x[i]);
    }
    cuda::mp_array_host2device(mp_dev_x, mp_x, SIZE);

    //Calculation of the reference result
    for (int i = 0; i < SIZE; i++) {
        mpfr_abs(reference_x[i], reference_x[i], MPFR_RNDN);
        mpfr_add(reference_res, reference_res, reference_x[i], MPFR_RNDN);
    }
    //Calculation of the MPFR result with MP_PRECISION bits of precision
    for (int i = 0; i < SIZE; i++) {
        mpfr_abs(mpfr_x[i], mpfr_x[i], MPFR_RNDN);
        mpfr_add(mpfr_result, mpfr_result, mpfr_x[i], MPFR_RNDN);
    }
    //Calculation of the MPRES-BLAS result with MP_PRECISION bits of precision
    cuda::mp_asum<MPRES_CUDA_BLOCKS_REDUCE, MPRES_CUDA_THREADS_REDUCE>(SIZE, mp_dev_x, 1, mp_dev_result);
    cuda::mp_array_device2host(&mp_result, mp_dev_result, 1);
    cudaDeviceSynchronize();

    //Error bounds
    compute_bounds(reference_res);

    //Accuracy evaluation
    evaluate_accuracy(reference_res, mp_result, mpfr_result);

    //Cleanup
    for (int i = 0; i < SIZE; i++) {
        mpfr_clear(reference_x[i]);
        mpfr_clear(mpfr_x[i]);
    }
    mpfr_clear(reference_res);
    mpfr_clear(mpfr_result);
    delete[] mp_x;
    cuda::mp_array_clear(mp_dev_x);
    cuda::mp_array_clear(mp_dev_result);

    //End logging
    Logger::endTestDescription();
}