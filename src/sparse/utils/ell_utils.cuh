/*
 *  Helper routines for ELLPACK
 *
 *  Copyright 2021 by Konstantin Isupov and Ivan Babeshko.
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


#ifndef ELL_UTILS_CUH
#define ELL_UTILS_CUH

#include <types.cuh>
#include <common.cuh>

void ell_init(ell_t &ell, const int m, const int maxnzr) {
    ell.as = new double[m * maxnzr]();
    ell.ja = new int[m * maxnzr]();
}

void ell_clear(ell_t &ell) {
    delete [] ell.as;
    delete [] ell.ja;
}

namespace cuda {

    void ell_init(ell_t &ell, const int m, const int maxnzr) {
        checkDeviceHasErrors(cudaMalloc(&ell.as, m * maxnzr * sizeof(double)));
        checkDeviceHasErrors(cudaMalloc(&ell.ja, m * maxnzr * sizeof(int)));
    }

    void ell_clear(ell_t &ell) {
        checkDeviceHasErrors(cudaFree(ell.as));
        checkDeviceHasErrors(cudaFree(ell.ja));
        checkDeviceHasErrors(cudaDeviceSynchronize());
        cudaCheckErrors();
    }

    void ell_host2device(ell_t &ell_dev, const ell_t &ell_host, const int m, const int maxnzr) {
        checkDeviceHasErrors(cudaMemcpy(ell_dev.as, ell_host.as, m * maxnzr * sizeof(double), cudaMemcpyHostToDevice));
        checkDeviceHasErrors(cudaMemcpy(ell_dev.ja, ell_host.ja, m * maxnzr * sizeof(int), cudaMemcpyHostToDevice));
        checkDeviceHasErrors(cudaDeviceSynchronize());
        cudaCheckErrors();
    }
}

#endif //ELL_UTILS_CUH
