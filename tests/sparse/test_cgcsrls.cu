/*
 *  Performance test for conjugate gradient linear solver using CSR
 *  Path to the matrix must be given as a command line argument, e.g., ../../tests/sparse/matrices/t3dl.mtx
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

#include "mpfr.h"
#include "logger.cuh"
#include "sparse/utils/mtx_reader.cuh"
#include "sparse/utils/csr_utils.cuh"
#include "sparse/cgcsrls/test_double_cgcsrls.cuh"
#include "sparse/cgcsrls/test_double_pcgcsrls_jacobi.cuh"

int INP_BITS; //in bits
int INP_DIGITS; //in decimal digits

void setPrecisions() {
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
}

void test(const char * MATRIX_PATH, const int N, const int LINES, const int NNZ, const bool SYMM, const string DATATYPE, const double TOL, const int MAXIT) {
    //Inputs
    auto *rhs = new double[N]; //right-hand-side vector
    for(int i = 0; i < N; i++){
        rhs[i] = 1;
    }
    csr_t CSR;
    csr_init(CSR, N, NNZ);
    //Convert a sparse matrix to the double-precision CSR format
    build_csr(MATRIX_PATH, N, NNZ, LINES, SYMM, CSR);
    //Launch tests
    test_double_cgcsrls(N, NNZ, CSR, rhs, TOL, MAXIT);
    test_double_pcgcsrls_jacobi(N, NNZ, CSR, rhs, TOL, MAXIT);
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors(); //CUMP gives failure
    //Cleanup
    delete[] rhs;
    csr_clear(CSR);
    cudaDeviceReset();
}

int main(int argc, char *argv[]) {
    //The operation parameters. Read from an input file that contains a sparse matrix
    int M = 0; //number of rows
    int N = 0; //number of columns
    int NNZ = 0; //number of nonzeros in matrix
    int MAXNZR = 0; //maximum number of nonzeros per row in the matrix A
    int NZMD = 0; //number of nonzeros in the main diagonal of the matrix
    int LINES = 0; //number of lines in the input matrix file
    bool SYMM = false; //true if the input matrix is to be treated as symmetrical; otherwise false
    string DATATYPE; //defines type of data in MatrixMarket: real, integer, binary
    const double TOL = 1e-12;
    const int MAXIT = 40000;

    initialize();

    //Start logging
    Logger::beginTestDescription(Logger::CGCSRLS_TEST);
    if(argc<=1) {
        printf("Matrix is not specified in command line arguments.");
        Logger::printSpace();
        Logger::endTestDescription();
        exit(1);
    }
    const char * MATRIX_PATH = argv[1];

    Logger::beginSection("Matrix properties:");
    Logger::printParam("Path", MATRIX_PATH);
    collect_mtx_stats(MATRIX_PATH, M, N, LINES, MAXNZR, NZMD, SYMM, DATATYPE);
    NNZ = SYMM ? ( (LINES - NZMD) * 2 + NZMD) : LINES;
    Logger::printParam("Number of rows/columns, N", N);
    Logger::printParam("Number of nonzeros, NNZ", NNZ);
    Logger::printParam("Maximum number of nonzeros per row, MAXNZR", MAXNZR);
    Logger::printParam("Average number of nonzeros per row, AVGNZR", (double)NNZ/M);
    Logger::printParam("Symmetry, SYMM", SYMM);
    Logger::printParam("Data type, DATATYPE", DATATYPE);
    Logger::printDash();
    Logger::beginSection("Additional info:");
    Logger::printParam("RNS_MODULI_SIZE", RNS_MODULI_SIZE);
    Logger::printParam("MP_PRECISION", MP_PRECISION);
    //Logger::printParam("CAMPARY_PRECISION (n-double)", CAMPARY_PRECISION);
    Logger::printParam("TOLERANCE", TOL);
    Logger::printParam("MAXIT", MAXIT);
    Logger::endSection(true);

    //Run the test
    test(MATRIX_PATH, N, LINES, NNZ, SYMM, DATATYPE, TOL, MAXIT);

    //Finalize
    finalize();

    //End logging
    Logger::endTestDescription();

    return 0;
}