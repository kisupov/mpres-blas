/*
 *  Performance test for the MPRES-BLAS library SpMV routine mpspmv_mpmtx_csr_2stage (multiple precision matrix)
 *
 *  Copyright 2020 by Konstantin Isupov.
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

#ifndef TEST_MPRES_MPSPMV_MPMTX_CSR_2STAGE_CUH
#define TEST_MPRES_MPSPMV_MPMTX_CSR_2STAGE_CUH

#include "../../../tsthelper.cuh"
#include "../../../logger.cuh"
#include "../../../timers.cuh"
#include "../../../../src/mparray.cuh"
#include "../../../../src/mpcollection.cuh"
#include "../../../../src/sparse/mpmtx/mpspmv_mpmtx_csr_2stage.cuh"

/////////
// SpMV CSR two-stage implementation
/////////
void test_mpres_mpspmv_mpmtx_csr_2stage(const int m, const int n, const int nnz, const csr_t &csr, const mpfr_t * x){
    Logger::printDash();
    InitCudaTimer();
    PrintTimerName("[GPU] MPRES-BLAS mpspmv_mpmtx_csr_2stage");

    //Execution configuration
    const int gridDim1 = 512;   //blocks for fields
    const int blockDim1 = 128;  //threads for fields
    const int gridDim2 = 32768; //blocks for residues
    const int blockDim3 = 64; //threads for reduce
    printf("\tExec. config: gridDim1 = %i, blockDim1 = %i, gridDim2 = %i, blockDim3 = %i\n", gridDim1, blockDim1, gridDim2, blockDim3);
    printf("\tMatrix (AS array) size (MB): %lf\n", get_mp_float_array_size_in_mb(nnz));

    //Host data
    auto hx = new mp_float_t[n];
    auto hy = new mp_float_t[m];
    auto has = new mp_float_t[nnz];

    //GPU data
    mp_array_t dx;
    mp_array_t dy;
    mp_collection_t das;
    mp_collection_t dbuf;
    int *dirp;
    int *dja;

    //Init data
    cuda::mp_array_init(dx, n);
    cuda::mp_array_init(dy, m);
    cuda::mp_collection_init(das, nnz);
    cuda::mp_collection_init(dbuf, nnz);
    cudaMalloc(&dirp, sizeof(int) * (m + 1));
    cudaMalloc(&dja, sizeof(int) * nnz);

    // Convert from MPFR and double
    convert_vector(hx, x, n);
    convert_vector(has, csr.as, nnz);

    //Copying to the GPU
    cuda::mp_array_host2device(dx, hx, n);
    cuda::mp_collection_host2device(das, has, nnz);
    cudaMemcpy(dirp, csr.irp, sizeof(int) * (m + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(dja, csr.ja, sizeof(int) * nnz, cudaMemcpyHostToDevice);
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    //Launch
    StartCudaTimer();
    cuda::mpspmv_mpmtx_csr_2stage<gridDim1, blockDim1, gridDim2, blockDim3>(m, n, nnz, dirp, dja, das, dx, dy, dbuf);
    EndCudaTimer();
    PrintCudaTimer("took");
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    //Copying to the host
    cuda::mp_array_device2host(hy, dy, m);
    print_mp_sum(hy, m);

    //Cleanup
    delete [] hx;
    delete [] hy;
    delete [] has;
    cuda::mp_array_clear(dx);
    cuda::mp_array_clear(dy);
    cuda::mp_collection_clear(das);
    cuda::mp_collection_clear(dbuf);
    cudaFree(dirp);
    cudaFree(dja);
}


#endif //TEST_MPRES_MPSPMV_MPMTX_CSR_2STAGE_CUH