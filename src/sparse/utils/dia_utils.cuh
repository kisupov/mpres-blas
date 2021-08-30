/*
 *  Helper routines for DIA
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


#ifndef MPRES_DIA_UTILS_CUH
#define MPRES_DIA_UTILS_CUH

#include <types.cuh>
#include <common.cuh>

void dia_init(dia_t &dia, const int m, const int ndiag) {
    dia.as = new double[m * ndiag]();
    dia.offset = new int[ndiag]();
}

void dia_clear(dia_t &dia) {
    delete [] dia.as;
    delete [] dia.offset;
}

namespace cuda {

    void dia_init(dia_t &dia, const int m, const int ndiag) {
        checkDeviceHasErrors(cudaMalloc(&dia.as, m * ndiag * sizeof(double)));
        checkDeviceHasErrors(cudaMalloc(&dia.offset, ndiag * sizeof(int)));
    }

    void dia_clear(dia_t &dia) {
        checkDeviceHasErrors(cudaFree(dia.as));
        checkDeviceHasErrors(cudaFree(dia.offset));
        checkDeviceHasErrors(cudaDeviceSynchronize());
        cudaCheckErrors();
    }

    void dia_host2device(dia_t &dia_dev, const dia_t &dia_host, const int m, const int ndiag) {
        checkDeviceHasErrors(cudaMemcpy(dia_dev.as, dia_host.as, m * ndiag * sizeof(double), cudaMemcpyHostToDevice));
        checkDeviceHasErrors(cudaMemcpy(dia_dev.offset, dia_host.offset, ndiag * sizeof(int), cudaMemcpyHostToDevice));
        checkDeviceHasErrors(cudaDeviceSynchronize());
        cudaCheckErrors();
    }
}

#endif //MPRES_DIA_UTILS_CUH
