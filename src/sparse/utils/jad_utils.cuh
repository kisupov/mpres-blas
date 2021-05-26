/*
 *  Helper routines for JAD
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


#ifndef JAD_UTILS_CUH
#define JAD_UTILS_CUH

#include <types.cuh>
#include <common.cuh>

void jad_init(jad_t &jad, const int m, const int maxnzr, const int nnz) {
    jad.as = new double[nnz]();
    jad.ja = new int[nnz]();
    jad.jcp = new int[maxnzr + 1]();
    jad.perm = new int[m]();
}

void jad_clear(jad_t &jad) {
    delete [] jad.as;
    delete [] jad.ja;
    delete [] jad.jcp;
    delete [] jad.perm;
}

namespace cuda {

    void jad_init(jad_t &jad, const int m, const int maxnzr, const int nnz) {
        checkDeviceHasErrors(cudaMalloc(&jad.as, nnz * sizeof(double)));
        checkDeviceHasErrors(cudaMalloc(&jad.ja, nnz * sizeof(int)));
        checkDeviceHasErrors(cudaMalloc(&jad.jcp, (maxnzr + 1) * sizeof(int)));
        checkDeviceHasErrors(cudaMalloc(&jad.perm, m * sizeof(int)));
    }

    void jad_clear(jad_t &jad) {
        checkDeviceHasErrors(cudaFree(jad.as));
        checkDeviceHasErrors(cudaFree(jad.ja));
        checkDeviceHasErrors(cudaFree(jad.jcp));
        checkDeviceHasErrors(cudaFree(jad.perm));
        checkDeviceHasErrors(cudaDeviceSynchronize());
        cudaCheckErrors();
    }

    void jad_host2device(jad_t &jad_dev, const jad_t &jad_host, const int m, const int maxnzr, const int nnz) {
        checkDeviceHasErrors(cudaMemcpy(jad_dev.as, jad_host.as, nnz * sizeof(double), cudaMemcpyHostToDevice));
        checkDeviceHasErrors(cudaMemcpy(jad_dev.ja, jad_host.ja, nnz * sizeof(int), cudaMemcpyHostToDevice));
        checkDeviceHasErrors(cudaMemcpy(jad_dev.jcp, jad_host.jcp, (maxnzr + 1) * sizeof(int), cudaMemcpyHostToDevice));
        checkDeviceHasErrors(cudaMemcpy(jad_dev.perm, jad_host.perm, m * sizeof(int), cudaMemcpyHostToDevice));
        checkDeviceHasErrors(cudaDeviceSynchronize());
        cudaCheckErrors();
    }
}

#endif //JAD_UTILS_CUH
