/*
 *  Multiple-precision sparse linear algebra kernels using CUMP as well as corresponding performance benchmarks.
 *
 *  Copyright 2020 by Konstantin Isupov and Ivan Babeshko.
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

#ifndef MPRES_TEST_CUMP_SPARSE_CUH
#define MPRES_TEST_CUMP_SPARSE_CUH

#include "../../tsthelper.cuh"
#include "../../logger.cuh"
#include "../../timers.cuh"
#include "3rdparty/cump_common.cuh"

/********************* Computational kernels *********************/

/*
 * Performs the matrix-vector operation y = A * x
 * where x and y are dense vectors and A is a sparse matrix.
 * The matrix should be stored in the ELLPACK format: entries are stored in a dense array in column major order and explicit zeros are stored if necessary (zero padding)
 */
__global__ void cump_spmv_ell_kernel(const int m, const int nzr, const int *ja, mpf_array_t as, mpf_array_t x, mpf_array_t y, mpf_array_t buf) {
    using namespace cump;
    unsigned int row = threadIdx.x + blockIdx.x * blockDim.x;
    if( row < m ) {
        for (int col = 0; col < nzr; col++) {
            int index = ja[col * m + row];
            if(index >= 0){
                mpf_mul(buf[row], x[index], as[col * m + row]);
                mpf_add(y[row], y[row], buf[row]);
            }
        }
    }
}

/*
 * Performs the matrix-vector operation y = A * x
 * where x and y are dense vectors and A is a sparse matrix.
 * The matrix should be stored in the CSR format
 */
__global__ void cump_spmv_csr_kernel(const int m, const int *irp,  const int *ja, mpf_array_t as, mpf_array_t x, mpf_array_t y, mpf_array_t buf) {
    using namespace cump;
    unsigned int row = threadIdx.x + blockIdx.x * blockDim.x;
    if( row < m ) {
        int row_start = irp[row];
        int row_end = irp[row+1];
        for (int i = row_start; i < row_end; i++) {
            mpf_mul(buf[row], x[ja[i]], as[i]);
            mpf_add(y[row], y[row], buf[row]);
        }
    }
}

/*
 * Performs the matrix-vector operation y = A * x
 * where x and y are dense vectors and A is a sparse matrix.
 * The matrix should be stored in the DIA format: entries are stored in a dense array in column major order and explicit zeros are stored if necessary (zero padding)
 */
__global__ void cump_spmv_dia_kernel(const int m, const int n, const int ndiag, const int *offset, mpf_array_t as, mpf_array_t x, mpf_array_t y, mpf_array_t buf) {
    using namespace cump;
    unsigned int row = threadIdx.x + blockIdx.x * blockDim.x;
    if (row < m) {
        for (int i = 0; i < ndiag; i++) {
            int col = row + offset[i];
            if(col  >= 0 && col < n) {
                mpf_mul(buf[row], x[col], as[m * i + row]);
                mpf_add(y[row], y[row], buf[row]);
            }
        }
    }
}

/********************* Benchmarks *********************/

/*
 * SpMV ELLPACK test
 */
void cump_spmv_ell_test(const int m, const int n, const int nzr, const int *ja, const double *as, mpfr_t *x, const int prec, const int convert_digits){
    Logger::printDash();
    InitCudaTimer();
    PrintTimerName("[GPU] CUMP SpMV ELLPACK");

    //Set precision
    mpf_set_default_prec(prec);
    cumpf_set_default_prec(prec);

    //Execution configuration
    int threads = 32;
    int blocks = m / threads + 1;
    printf("(exec. config: blocks = %i, threads = %i)\n", blocks, threads);

    //Host data
    mpf_t *hx = new mpf_t[n];
    mpf_t *hy = new mpf_t[m];
    mpf_t *has = new mpf_t[m * nzr];

    //GPU data
    cumpf_array_t dx;
    cumpf_array_t dy;
    cumpf_array_t das;
    cumpf_array_t dbuf;
    int *dja = new int[m * nzr];

    cumpf_array_init2(dx, n, prec);
    cumpf_array_init2(dy, m, prec);
    cumpf_array_init2(das, m * nzr, prec);
    cumpf_array_init2(dbuf, m, prec);
    cudaMalloc(&dja, sizeof(int) * m * nzr);

    //Convert from MPFR
    for(int i = 0; i < n; i++){
        mpf_init2(hx[i], prec);
        mpf_set_str(hx[i], convert_to_string_sci(x[i], convert_digits).c_str(), 10);
    }
    for(int i = 0; i < m; i++){
        mpf_init2(hy[i], prec);
        mpf_set_d(hy[i], 0.0);
    }
    //Convert from double
    for(int i = 0; i < m * nzr; i++){
        mpf_init2(has[i], prec);
        mpf_set_d(has[i], as[i]);
    }

    //Copying to the GPU
    cumpf_array_set_mpf(dx, hx, n);
    cumpf_array_set_mpf(dy, hy, m);
    cumpf_array_set_mpf(das, has, m * nzr);
    cudaMemcpy(dja, ja, sizeof(int) * m * nzr, cudaMemcpyHostToDevice);

    //Launch
    StartCudaTimer();
    cump_spmv_ell_kernel<<<blocks, threads>>>(m, nzr, dja, das, dx, dy, dbuf);
    EndCudaTimer();
    PrintCudaTimer("took");

    //Copying to the host
    mpf_array_set_cumpf(hy, dy, m);
    for(int i = 1; i < m; i++){
        mpf_add(hy[0], hy[i], hy[0]);
    }
    gmp_printf ("result: %.70Ff \n", hy[0]);

    //Cleanup
    for(int i = 0; i < n; i++){
        mpf_clear(hx[i]);
    }
    for(int i = 0; i < m; i++){
        mpf_clear(hy[i]);
    }
    for(int i = 0; i < m * nzr; i++){
        mpf_clear(has[i]);
    }
    delete [] hx;
    delete [] hy;
    delete [] has;
    cumpf_array_clear(dx);
    cumpf_array_clear(dy);
    cumpf_array_clear(das);
    cumpf_array_clear(dbuf);
    cudaFree(dja);
}

/*
 * SpMV CSR test
 */
void cump_spmv_csr_test(const int m, const int n, const int nnz, const int *irp, const int *ja, const double *as,  mpfr_t *x, const int prec, const int convert_digits){
    Logger::printDash();
    InitCudaTimer();
    PrintTimerName("[GPU] CUMP SpMV CSR");

    //Set precision
    mpf_set_default_prec(prec);
    cumpf_set_default_prec(prec);

    //Execution configuration
    int threads = 32;
    int blocks = m / threads + 1;
    printf("(exec. config: blocks = %i, threads = %i)\n", blocks, threads);

    //Host data
    mpf_t *hx = new mpf_t[n];
    mpf_t *hy = new mpf_t[m];
    mpf_t *has = new mpf_t[nnz];

    //GPU data
    cumpf_array_t dx;
    cumpf_array_t dy;
    cumpf_array_t das;
    cumpf_array_t dbuf;
    int *dirp = new int[m + 1];
    int *dja = new int[nnz];

    cumpf_array_init2(dx, n, prec);
    cumpf_array_init2(dy, m, prec);
    cumpf_array_init2(das, nnz, prec);
    cumpf_array_init2(dbuf, m, prec);
    cudaMalloc(&dirp, sizeof(int) * (m + 1));
    cudaMalloc(&dja, sizeof(int) * nnz);

    //Convert from MPFR
    for(int i = 0; i < n; i++){
        mpf_init2(hx[i], prec);
        mpf_set_str(hx[i], convert_to_string_sci(x[i], convert_digits).c_str(), 10);
    }
    for(int i = 0; i < m; i++){
        mpf_init2(hy[i], prec);
        mpf_set_d(hy[i], 0.0);
    }
    //Convert from double
    for(int i = 0; i < nnz; i++){
        mpf_init2(has[i], prec);
        mpf_set_d(has[i], as[i]);
    }

    //Copying to the GPU
    cumpf_array_set_mpf(dx, hx, n);
    cumpf_array_set_mpf(dy, hy, m);
    cumpf_array_set_mpf(das, has, nnz);
    cudaMemcpy(dirp, irp, sizeof(int) * (m+1), cudaMemcpyHostToDevice);
    cudaMemcpy(dja, ja, sizeof(int) * nnz, cudaMemcpyHostToDevice);

    //Launch
    StartCudaTimer();
    cump_spmv_csr_kernel<<<blocks, threads>>>(m, dirp, dja, das, dx, dy, dbuf);
    EndCudaTimer();
    PrintCudaTimer("took");

    //Copying to the host
    mpf_array_set_cumpf(hy, dy, m);
    for(int i = 1; i < m; i++){
        mpf_add(hy[0], hy[i], hy[0]);
    }
    gmp_printf ("result: %.70Ff \n", hy[0]);

    //Cleanup
    for(int i = 0; i < n; i++){
        mpf_clear(hx[i]);
    }
    for(int i = 0; i < m; i++){
        mpf_clear(hy[i]);
    }
    for(int i = 0; i < nnz; i++){
        mpf_clear(has[i]);
    }
    delete [] hx;
    delete [] hy;
    delete [] has;
    cumpf_array_clear(dx);
    cumpf_array_clear(dy);
    cumpf_array_clear(das);
    cumpf_array_clear(dbuf);
    cudaFree(dirp);
    cudaFree(dja);
}

/*
 * SpMV DIA test
 */
void cump_spmv_dia_test(const int m, const int n, const int ndiag, const int *offset, const double *as, mpfr_t *x, const int prec, const int convert_digits){
    Logger::printDash();
    InitCudaTimer();
    PrintTimerName("[GPU] CUMP SpMV DIA");

    //Set precision
    mpf_set_default_prec(prec);
    cumpf_set_default_prec(prec);

    //Execution configuration
    int threads = 32;
    int blocks = m / threads + 1;
    printf("(exec. config: blocks = %i, threads = %i)\n", blocks, threads);

    //Host data
    mpf_t *hx = new mpf_t[n];
    mpf_t *hy = new mpf_t[m];
    mpf_t *has = new mpf_t[m * ndiag];

    //GPU data
    cumpf_array_t dx;
    cumpf_array_t dy;
    cumpf_array_t das;
    cumpf_array_t dbuf;
    int *doffset = new int[ndiag];

    cumpf_array_init2(dx, n, prec);
    cumpf_array_init2(dy, m, prec);
    cumpf_array_init2(das, m * ndiag, prec);
    cumpf_array_init2(dbuf, m, prec);
    cudaMalloc(&doffset, sizeof(int) * ndiag);

    //Convert from MPFR
    for(int i = 0; i < n; i++){
        mpf_init2(hx[i], prec);
        mpf_set_str(hx[i], convert_to_string_sci(x[i], convert_digits).c_str(), 10);
    }
    for(int i = 0; i < m; i++){
        mpf_init2(hy[i], prec);
        mpf_set_d(hy[i], 0.0);
    }
    //Convert from double
    for(int i = 0; i < m * ndiag; i++){
        mpf_init2(has[i], prec);
        mpf_set_d(has[i], as[i]);
    }

    //Copying to the GPU
    cumpf_array_set_mpf(dx, hx, n);
    cumpf_array_set_mpf(dy, hy, m);
    cumpf_array_set_mpf(das, has, m * ndiag);
    cudaMemcpy(doffset, offset, sizeof(int) * ndiag, cudaMemcpyHostToDevice);

    //Launch
    StartCudaTimer();
    cump_spmv_dia_kernel<<<blocks, threads>>>(m, n, ndiag, doffset, das, dx, dy, dbuf);
    EndCudaTimer();
    PrintCudaTimer("took");

    //Copying to the host
    mpf_array_set_cumpf(hy, dy, m);
    for(int i = 1; i < m; i++){
        mpf_add(hy[0], hy[i], hy[0]);
    }
    gmp_printf ("result: %.70Ff \n", hy[0]);

    //Cleanup
    for(int i = 0; i < n; i++){
        mpf_clear(hx[i]);
    }
    for(int i = 0; i < m; i++){
        mpf_clear(hy[i]);
    }
    for(int i = 0; i < m * ndiag; i++){
        mpf_clear(has[i]);
    }
    delete [] hx;
    delete [] hy;
    delete [] has;
    cumpf_array_clear(dx);
    cumpf_array_clear(dy);
    cumpf_array_clear(das);
    cumpf_array_clear(dbuf);
    cudaFree(doffset);
}

#endif //MPRES_TEST_CUMP_SPARSE_CUH