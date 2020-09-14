/*
 *  Performance test for BLAS GEMV routines
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

#include "omp.h"
#include "../../logger.cuh"
#include "../../timers.cuh"
#include "../../tsthelper.cuh"
#include "../../../src/sparse/mpspmvell.cuh"
#include "../../blas/performance/3rdparty.cuh"


#define M 300  // Number of matrix rows and the vector Y dimension
#define N 300 // Number of matrix columns and the vector X dimension
#define LDA (M) // Specifies the leading dimension of A as declared in the calling (sub)program.
#define TRANS "N" // Specifies the operation: if trans = 'N' or 'n', then y := alpha*A*x + beta*y; if trans = 'T' or 't' or 'C' or 'c' then y = alpha*A**T*x + beta*y (transposed matrix).
#define INCX 1 // Specifies the increment for the elements of x.
#define INCY 1 // Specifies the increment for the elements of y.
#define REPEAT_TEST 1 //Number of repeats

//Execution configuration for mpgemv
#define MPRES_CUDA_BLOCKS_FIELDS_ROUND 256
#define MPRES_CUDA_THREADS_FIELDS_ROUND 128
#define MPRES_CUDA_BLOCKS_RESIDUES 256
#define MPRES_CUDA_THREADS_REDUCE 32

#define OPENBLAS_THREADS 4

int MP_PRECISION_DEC; //in decimal digits
int INP_BITS; //in bits
int INP_DIGITS; //in decimal digits

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
    mp_real::mp_init(MP_PRECISION_DEC);
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();
}

void finalize() {
    mp_real::mp_finalize();
}

void convert_vector(mp_float_ptr dest, mpfr_t *source, int width) {
    for (int i = 0; i < width; i++) {
        mp_set_mpfr(&dest[i], source[i]);
    }
}

void convert_matrix(mp_float_ptr dest, mpfr_t *source, int rows, int cols) {
    int width = rows * cols;
    for (int i = 0; i < width; i++) {
        mp_set_mpfr(&dest[i], source[i]);
    }
}



/********************* GEMV implementations and benchmarks *********************/

/////////
// MPRES-BLAS (structure of arrays)
/////////
void mpres_test(enum mblas_trans_type trans, int m, int n, int maxNonZeros, int lenx, int leny, mp_float_t *A, int *indices,
                mp_float_t *x, mp_float_t *y) {
    InitCudaTimer();
    Logger::printDash();
    PrintTimerName("[GPU] MPRES-BLAS gemv");

    // Host data
    mp_float_ptr hx = x;
    mp_float_ptr hy = y;
    mp_float_ptr hA = A;

    //GPU data
    mp_array_t dx;
    mp_array_t dy;
    mp_array_t dA;
    mp_array_t dbuf1;
    int *dindices;

    //Init data
    cuda::mp_array_init(dx, lenx);
    cuda::mp_array_init(dy, leny);
    cuda::mp_array_init(dA, m * maxNonZeros);
    cuda::mp_array_init(dbuf1, m * n);

    cudaMalloc(&dindices, sizeof(int) * m * maxNonZeros);

    //Copying to the GPU
    cuda::mp_array_host2device(dx, hx, lenx);
    cuda::mp_array_host2device(dA, hA, m * maxNonZeros);
    cudaMemcpy(dindices, indices, sizeof(int) * m * maxNonZeros, cudaMemcpyHostToDevice);

    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();
    //Launch
    for (int i = 0; i < REPEAT_TEST; i++) {
        cuda::mp_array_host2device(dy, hy, leny);StartCudaTimer();

        cuda::spmv<
                MPRES_CUDA_BLOCKS_FIELDS_ROUND,
                MPRES_CUDA_THREADS_FIELDS_ROUND,
                MPRES_CUDA_BLOCKS_RESIDUES,
                MPRES_CUDA_THREADS_REDUCE>
                (trans, m, n, maxNonZeros, dA, dindices, dx, dy, dbuf1);
        EndCudaTimer();
    }
    PrintCudaTimer("took");
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();

    //Copying to the host
    cuda::mp_array_device2host(hy, dy, leny);
    print_mp_sum(hy, leny);

    //Cleanup
    delete[] hx;
    delete[] hy;
    delete[] hA;
    cuda::mp_array_clear(dx);
    cuda::mp_array_clear(dy);
    cuda::mp_array_clear(dA);
    cuda::mp_array_clear(dbuf1);
    cudaFree(dindices);
}

void create_ellpack_matrices(char filename[], mp_float_t *&data, int *&indices, int &m, int &n, int &maxNonZeros) {

    std::ifstream file(filename);
    int num_lines = 0;

// Ignore comments headers
    while (file.peek() == '%') file.ignore(2048, '\n');

// Read number of rows and columns
    file >> m >> n >> num_lines;

// Create 2D array and fill with zeros
    int *nonZeros;
    nonZeros = new int[m]();

// fill the matrix with data
    for (int l = 0; l < num_lines; l++) {
        double fileData = 0.0;
        int row = 0, col = 0;
        file >> row >> col >> fileData;
        nonZeros[(row - 1)] = nonZeros[(row - 1)] + 1;
    }


    for (int i = 0; i < m; i++) {
        std::cout << nonZeros[i] << " ";
    }
    std::cout << std::endl;

    maxNonZeros = *std::max_element(nonZeros, nonZeros + m);

    data = new mp_float_t[m * (maxNonZeros)];
    indices = new int[m * (maxNonZeros)]();

    //курсор в начало
    file.seekg(0, ios::beg);

    // Ignore comments headers
    while (file.peek() == '%') file.ignore(2048, '\n');

    // Read number of rows and columns
    file >> m >> n >> num_lines;

    int * colNum = new int[m]();

    //разобраться как заново считывать файл
    for (int l = 0; l < num_lines; l++) {
        double fileData = 0.0;
        int row = 0, col = 0;
        file >> row >> col >> fileData;
        mp_set_d(&data[colNum[(row - 1)] * m + (row - 1)], fileData);
        indices[colNum[(row - 1)] * m + (row - 1)] = (col-1);
        colNum[row - 1]++;
    }

    file.close();

    std::cout << "data" << std::endl;
    for (int j = 0; j < m; ++j) {
        for (int i = 0; i < maxNonZeros; ++i) {
            std::cout << mp_get_d(&data[j + m * i]) << " ";
        }
        std::cout << std::endl;
    }

/*
    std::cout << "data inline" << std::endl;
    for (int i = 0; i < m * maxNonZeros; ++i) {
        std::cout << i << " = " << mp_get_d(&data[i]) << std::endl;
    }
*/

    std::cout << std::endl;
    std::cout << "indices" << std::endl;
    for (int j = 0; j < m; ++j) {
        for (int i = 0; i < (maxNonZeros); ++i) {
            std::cout << indices[j + m * i] << " ";
        }
        std::cout << std::endl;
    }

/*    std::cout << "indices inline" << std::endl;
    for (int i = 0; i < m * maxNonZeros; ++i) {
        std::cout << i << " = " << indices[i] << std::endl;
    }*/
}
/********************* Main test *********************/

/*
 * Test for non-transposed matrix
 * x is of size n
 * y is of size m
 * a is of size lda * n, where the value of lda must be at least max(1, m).
 */
void testNoTrans() {
    //Actual length of the vectors

    int m = 0, n = 0, maxNonZeros = 0;
    mp_float_t *matrixA;
    int *indices = new int;

    create_ellpack_matrices("/home/ivan/Загрузки/matrixes/5x5 16-not-null.mtx", matrixA, indices, m, n, maxNonZeros);

    int lenx = (1 + (n - 1) * abs(INCX));
    int leny = (1 + (m - 1) * abs(INCY));

    //Inputs
    mp_float_t *vectorX = new mp_float_t[lenx];
    mp_float_t *vectorY = new mp_float_t[leny]();

    for (int i = 0; i < lenx; ++i) {
        mp_set_d(&vectorX[i], (i + 1));
    }

    //Launch tests

    mpres_test(mblas_no_trans, m, n, maxNonZeros, lenx, leny, matrixA, indices, vectorX, vectorY);

    checkDeviceHasErrors(cudaDeviceSynchronize());
    // cudaCheckErrors(); //CUMP gives failure

    //Cleanup
    delete[] vectorX;
    delete[] vectorY;
    //delete[] matrixA;

    cudaDeviceReset();
}

int main() {

    initialize();

    //Start logging
    Logger::beginTestDescription(Logger::BLAS_SPMV_PERFORMANCE_TEST);
    Logger::printTestParameters(N * M, REPEAT_TEST, MP_PRECISION, MP_PRECISION_DEC);
    Logger::beginSection("Operation info:");
    Logger::printParam("Matrix rows, M", M);
    Logger::printParam("Matrix columns, N", N);
    Logger::printParam("LDA", LDA);
    Logger::printParam("TRANS", TRANS);
    Logger::printDash();
    Logger::beginSection("Additional info:");
    Logger::printParam("RNS_MODULI_SIZE", RNS_MODULI_SIZE);
    Logger::printParam("MPRES_CUDA_BLOCKS_FIELDS_ROUND", MPRES_CUDA_BLOCKS_FIELDS_ROUND);
    Logger::printParam("MPRES_CUDA_THREADS_FIELDS_ROUND", MPRES_CUDA_THREADS_FIELDS_ROUND);
    Logger::printParam("MPRES_CUDA_BLOCKS_RESIDUES", MPRES_CUDA_BLOCKS_RESIDUES);
    Logger::printParam("MPRES_CUDA_THREADS_REDUCE", MPRES_CUDA_THREADS_REDUCE);
    Logger::printParam("CAMPARY_PRECISION (n-double)", CAMPARY_PRECISION);
    Logger::endSection(true);

    //Run the test
    testNoTrans();


    //Finalize
    finalize();

    //End logging
    Logger::endTestDescription();

    return 0;
}