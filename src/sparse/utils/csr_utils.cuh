/*
 *  Helper routines for CSR
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


#ifndef MPRES_CSR_UTILS_CUH
#define MPRES_CSR_UTILS_CUH

#include <types.cuh>
#include <common.cuh>

void csr_init(csr_t &csr, const int m, const int nnz) {
    csr.as = new double[nnz]();
    csr.ja = new int[nnz]();
    csr.irp = new int[m + 1](); //last element of irp equals to nnz (number of nonzeros in matrix)
}

void csr_clear(csr_t &csr) {
    delete [] csr.as;
    delete [] csr.ja;
    delete [] csr.irp;
}

//calculate the main diagonal of the matrix
void csr_mdiag(const csr_t &csr, const int m, double *md){
    for(int row = 0; row < m; row++){
        md[row] = 0;
        int row_start = csr.irp[row];
        int row_end = csr.irp[row+1];
        for (int i = row_start; i < row_end; i++) {
            if(csr.ja[i] == row){
                md[row] = csr.as[i];
                break;
            }
        }
    }
}

namespace cuda {

    void csr_init(csr_t &csr, const int m, const int nnz) {
        checkDeviceHasErrors(cudaMalloc(&csr.as, nnz * sizeof(double)));
        checkDeviceHasErrors(cudaMalloc(&csr.ja, nnz * sizeof(int)));
        checkDeviceHasErrors(cudaMalloc(&csr.irp, (m + 1) * sizeof(int)));
    }

    void csr_clear(csr_t &csr) {
        checkDeviceHasErrors(cudaFree(csr.as));
        checkDeviceHasErrors(cudaFree(csr.ja));
        checkDeviceHasErrors(cudaFree(csr.irp));
        checkDeviceHasErrors(cudaDeviceSynchronize());
        cudaCheckErrors();
    }

    void csr_host2device(csr_t &csr_dev, const csr_t &csr_host, const int m, const int nnz) {
        checkDeviceHasErrors(cudaMemcpy(csr_dev.as, csr_host.as, nnz * sizeof(double), cudaMemcpyHostToDevice));
        checkDeviceHasErrors(cudaMemcpy(csr_dev.ja, csr_host.ja, nnz * sizeof(int), cudaMemcpyHostToDevice));
        checkDeviceHasErrors(cudaMemcpy(csr_dev.irp, csr_host.irp, (m + 1) * sizeof(int), cudaMemcpyHostToDevice));
        checkDeviceHasErrors(cudaDeviceSynchronize());
        cudaCheckErrors();
    }
}

#endif //MPRES_CSR_UTILS_CUH
