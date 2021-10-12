/*
 *  Microbenchmark for evaluating the peak performance of multiple-precision arithmetic (addition and multiplication)
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
#include "test_mp_arith_peak.cuh"

int INP_BITS;

void setPrecisions() {
    INP_BITS = (int) (MP_PRECISION / 4);
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

int main() {
    initialize();
    Logger::beginTestDescription(Logger::ARITH_PEAK_PERFORMANCE_TEST);
    Logger::printTestParameters(N, 1, MP_PRECISION, 0);
    Logger::beginSection("Additional info:");
    Logger::printParam("RNS_MODULI_SIZE", RNS_MODULI_SIZE);
    Logger::endSection(true);
    test_mp_peak_performance(INP_BITS);
    finalize();
    Logger::endTestDescription();
    return 0;
}