/*
 *  Performance test for BLAS DOT routines
 *
 *  Copyright 2021 by Konstantin Isupov.
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

#include "logger.cuh"
#include "tsthelper.cuh"
#include "test_mpres_dot.cuh"
#include "test_mpfr_dot.cuh"
#include "test_openblas_dot.cuh"

#define N 10000000  // Number of matrix rows / column and the vectors dimension
#define REPEAT_TEST 1 //Number of repeats

int MP_PRECISION_DEC;
int INP_BITS;
int INP_DIGITS;

void setPrecisions() {
    MP_PRECISION_DEC = (int) (MP_PRECISION / 3.32 + 1);
    INP_BITS = (int) (MP_PRECISION / 4);
    INP_DIGITS = (int) (INP_BITS / 3.32 + 1);
}

void initialize() {
    cudaDeviceReset();
    rns_const_init();
    mp_const_init();
    setPrecisions();
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();
}

void finalize() {
    cudaDeviceReset();
}

void test() {
    //Inputs
    mpfr_t *vectorX = create_random_array(N, INP_BITS);
    mpfr_t *vectorY = create_random_array(N, INP_BITS);
    //Launch tests
    test_openblas(N, vectorX, vectorY, REPEAT_TEST);
    test_mpfr(N, vectorX, vectorY, REPEAT_TEST);
    test_mpres(N, vectorX, vectorY, REPEAT_TEST);
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();
    //Cleanup
    for (int i = 0; i < N; i++) {
        mpfr_clear(vectorX[i]);
        mpfr_clear(vectorY[i]);
    }
    delete[] vectorX;
    delete[] vectorY;
    cudaDeviceReset();
}
int main() {
    initialize();
    Logger::beginTestDescription(Logger::BLAS_SYMV_PERFORMANCE_TEST);
    Logger::printTestParameters(N, REPEAT_TEST, MP_PRECISION, MP_PRECISION_DEC);
  //  Logger::beginSection("Operation info:");
  //  Logger::printParam("Operation size, N", N);
  //  Logger::printParam("Repeats", REPEAT_TEST);
   // Logger::printDash();
    Logger::beginSection("Additional info:");
    Logger::printParam("RNS_MODULI_SIZE", RNS_MODULI_SIZE);
//    Logger::printParam("CAMPARY_PRECISION (n-double)", CAMPARY_PRECISION);
    Logger::endSection(true);
    test();
    finalize();
    Logger::endTestDescription();
    return 0;
}