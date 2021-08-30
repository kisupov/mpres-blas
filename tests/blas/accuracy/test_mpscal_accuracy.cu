/*
 *  Accuracy test for the SCAL (mpscal) routine
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
#include "../../../src/blas/scal.cuh"
#include "../../tsthelper.cuh"
#include "../../logger.cuh"

#define SIZE 1000000 //Operation size
//Execution configuration for mpscal
#define MPRES_CUDA_BLOCKS_FIELDS_ROUND   512
#define MPRES_CUDA_THREADS_FIELDS_ROUND  128
#define MPRES_CUDA_BLOCKS_RESIDUES       8192
int REFERENCE_PRECISION; //Precision for reference results

/*
 * Computes the actual errors and error bounds
 */
static void evaluate_accuracy(mpfr_t *reference, mpfr_t *result_mpfr, mp_float_ptr result_mpres, unsigned long size) {

    mpfr_t temp;         // Temporary storage for abs ref value
    mpfr_t unitRoundoff; // Unit Roundoff
    mpfr_t normRef;      // Reference vector norm ||ax|| = |ax1| + |ax2| .... + |axn|
    mpfr_t absBound;     // Absolute forward error bound
    mpfr_t relBound;     // Relative forward error bound
    mpfr_t *vector = new mpfr_t[size]; // Vector for the converted MPRES-BLAS results
    mpfr_t absErrorMpres; // Absolute forward error for MPRES-BLAS
    mpfr_t absErrorMpfr;  // Absolute forward error for MPFR
    mpfr_t relErrorMpres; // Relative forward error for MPRES-BLAS
    mpfr_t relErrorMpfr;  // Relative forward error for MPFR

    // Init
    mpfr_init2(temp, REFERENCE_PRECISION);
    mpfr_init2(unitRoundoff, REFERENCE_PRECISION);
    mpfr_init2(normRef, REFERENCE_PRECISION);
    mpfr_init2(absBound, REFERENCE_PRECISION);
    mpfr_init2(relBound, REFERENCE_PRECISION);
    mpfr_init2(absErrorMpres, REFERENCE_PRECISION);
    mpfr_init2(absErrorMpfr, REFERENCE_PRECISION);
    mpfr_init2(relErrorMpres, REFERENCE_PRECISION);
    mpfr_init2(relErrorMpfr, REFERENCE_PRECISION);

    // Computing unit roundoff
    mpfr_set_d(unitRoundoff, 0, MPFR_RNDN);
    mpfr_sqrt(unitRoundoff, RNS_MODULI_PRODUCT_MPFR, MPFR_RNDN);
    mpfr_ui_div(unitRoundoff, 4, unitRoundoff, MPFR_RNDN);

    // Computing norm of reference vector
    mpfr_set_d(normRef, 0, MPFR_RNDN);
    mpfr_set_d(temp, 0, MPFR_RNDN);
    for (int i = 0; i < size; i++) {
        mpfr_abs(temp, reference[i], MPFR_RNDN);
        mpfr_add(normRef, normRef, temp, MPFR_RNDN);
    }

    // Computing error bounds
    mpfr_set_d(absBound, 0, MPFR_RNDN);
    mpfr_set_d(relBound, 0, MPFR_RNDN);
    mpfr_mul(absBound, unitRoundoff, normRef, MPFR_RNDN); // Absolute error
    mpfr_set(relBound, unitRoundoff, MPFR_RNDN); // Relative error

    // Converting the MPRES-BLAS result to MPFR
    for (int i = 0; i < size; i++) {
        mpfr_init2(vector[i], REFERENCE_PRECISION);
        mp_get_mpfr(vector[i], &result_mpres[i]);
    }

    // Actual SCAL errors:
    // absolute error = |ref_1 - x_1| + ... + |ref_n - x_n|, where ref - exact vector, x - computed vector
    // relative error = absolute error / norm of exact vector

    // Computing actual absolute forward errors
    mpfr_set_d(absErrorMpres, 0, MPFR_RNDN);
    mpfr_set_d(absErrorMpfr, 0, MPFR_RNDN);
    for (int i = 0; i < size; i++) {
        // MPRES-BLAS
        mpfr_sub(temp, reference[i], vector[i], MPFR_RNDN);
        mpfr_abs(temp, temp, MPFR_RNDN);
        mpfr_add(absErrorMpres, absErrorMpres, temp, MPFR_RNDN);

        //MPFR
        mpfr_sub(temp, reference[i], result_mpfr[i], MPFR_RNDN);
        mpfr_abs(temp, temp, MPFR_RNDN);
        mpfr_add(absErrorMpfr, absErrorMpfr, temp, MPFR_RNDN);
    }

    // Computing actual relative forward errors
    mpfr_div(relErrorMpres, absErrorMpres, normRef, MPFR_RNDN);
    mpfr_div(relErrorMpfr, absErrorMpfr, normRef, MPFR_RNDN);

    // Print evaluation results
    Logger::printDash();
    mpfr_printf("Unit Roundoff: %.25Re\n", unitRoundoff);
    Logger::printDash();
    mpfr_printf("Computed absolute error bound (MPRES-BLAS): %.25Re\n", absBound);
    mpfr_printf("Computed relative error bound (MPRES-BLAS): %.25Re\n", relBound);
    Logger::printDash();
    mpfr_printf("Actual absolute error (MPRES-BLAS): %.25Re\n", absErrorMpres);
    mpfr_printf("Actual relative error (MPRES-BLAS): %.25Re\n", relErrorMpres);
    mpfr_printf("Actual absolute error (MPFR): %.25Re\n", absErrorMpfr);
    mpfr_printf("Actual relative error (MPFR): %.25Re\n", relErrorMpfr);

    // Cleanup
    mpfr_clear(temp);
    mpfr_clear(unitRoundoff);
    mpfr_clear(normRef);
    mpfr_clear(absBound);
    mpfr_clear(relBound);
    for (int i = 0; i < size; i++) {
        mpfr_clear(vector[i]);
    }
    delete[] vector;
    mpfr_clear(absErrorMpres);
    mpfr_clear(relErrorMpres);
    mpfr_clear(absErrorMpfr);
    mpfr_clear(relErrorMpfr);
}

int main() {
    cudaDeviceReset();
    rns_const_init();
    mp_const_init();

    //Start logging
    Logger::beginTestDescription(Logger::BLAS_SCAL_ACCURACY_TEST);
    Logger::printTestParameters(SIZE, 1, MP_PRECISION, (int)(MP_PRECISION / 3.32 + 1));
    Logger::beginSection("Additional info:");
    Logger::printParam("RNS_MODULI_SIZE", RNS_MODULI_SIZE);
    Logger::printParam("MPRES_CUDA_BLOCKS_FIELDS_ROUND", MPRES_CUDA_BLOCKS_FIELDS_ROUND);
    Logger::printParam("MPRES_CUDA_THREADS_FIELDS_ROUND", MPRES_CUDA_THREADS_FIELDS_ROUND);
    Logger::printParam("MPRES_CUDA_BLOCKS_RESIDUES", MPRES_CUDA_BLOCKS_RESIDUES);
    Logger::printDash();
    rns_const_print(true);
    Logger::printDash();
    rns_eval_const_print();
    Logger::printDash();
    mp_const_print();
    Logger::endSection(true);
    Logger::printSpace();

    REFERENCE_PRECISION = MP_PRECISION * 5;
    // Reference alpha
    mpfr_t reference_alpha;
    mpfr_init2(reference_alpha, REFERENCE_PRECISION);
    // Reference x for error bounds
    mpfr_t *reference_x = new mpfr_t[SIZE];
    for (int i = 0; i < SIZE; i++) {
        mpfr_init2(reference_x[i], REFERENCE_PRECISION);
    }

    mpfr_t *mpfr_alpha; //MPFR alpha
    mpfr_t *mpfr_x; //MPFR x
    mp_float_t mp_alpha;  //MPRES-BLAS alpha
    mp_float_ptr mp_x = new mp_float_t[SIZE];  //MPRES-BLAS x
    mp_array_t mp_dev_x, mp_dev_alpha;

    cuda::mp_array_init(mp_dev_x, SIZE);
    cuda::mp_array_init(mp_dev_alpha, 1);

    //Data generation
    mpfr_alpha = create_random_array(1, MP_PRECISION);
    mpfr_x = create_random_array(SIZE, MP_PRECISION);

    //Setting inputs
    mpfr_set(reference_alpha, mpfr_alpha[0], MPFR_RNDN);
    mp_set_mpfr(&mp_alpha, mpfr_alpha[0]);
    for (int i = 0; i < SIZE; i++) {
        mpfr_set(reference_x[i], mpfr_x[i], MPFR_RNDN);
        mp_set_mpfr(&mp_x[i], mpfr_x[i]);
    }
    //Data transfer
    cuda::mp_array_host2device(mp_dev_x, mp_x, SIZE);
    cuda::mp_array_host2device(mp_dev_alpha, &mp_alpha, 1);

    //Computing reference
    for (int i = 0; i < SIZE; i++) {
        mpfr_mul(reference_x[i], reference_x[i], reference_alpha, MPFR_RNDN);
    }

    //Computing MPFR results
    for (int i = 0; i < SIZE; i++) {
        mpfr_mul(mpfr_x[i], mpfr_x[i], mpfr_alpha[0], MPFR_RNDN);
    }

    //Call to mpscal
    cuda::mpscal<
            MPRES_CUDA_BLOCKS_FIELDS_ROUND,
            MPRES_CUDA_THREADS_FIELDS_ROUND,
            MPRES_CUDA_BLOCKS_RESIDUES>
            (SIZE, mp_dev_alpha, mp_dev_x, 1);

    cuda::mp_array_device2host(mp_x, mp_dev_x, SIZE);

    //Accuracy evaluation
    evaluate_accuracy(reference_x, mpfr_x, mp_x, SIZE);

    //Cleanup
    for (int i = 0; i < SIZE; i++) {
        mpfr_clear(reference_x[i]);
        mpfr_clear(mpfr_x[i]);
    }
    mpfr_clear(reference_alpha);
    mpfr_clear(mpfr_alpha[0]);
    delete[] reference_x;
    delete[] mpfr_x;
    delete[] mpfr_alpha;
    delete[] mp_x;
    cuda::mp_array_clear(mp_dev_x);
    cuda::mp_array_clear(mp_dev_alpha);

    //End logging
    Logger::endTestDescription();
}
