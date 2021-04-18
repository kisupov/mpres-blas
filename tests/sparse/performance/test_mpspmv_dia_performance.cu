/*
 *  Performance test for SpMV routines using the DIA matrix format (multiple precision matrix)
 *  Path to the matrix must be given as a command line argument, e.g., ../../tests/sparse/matrices/t3dl.mtx

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

#include "logger.cuh"
#include "tsthelper.cuh"
#include "sparse/matrix_converter.cuh"
#include "sparse/performance/dia/mp/test_mpres_mpspmv_dia.cuh"
#include "sparse/performance/dia/mp/test_campary_mpspmv_dia.cuh"
#include "sparse/performance/dia/mp/test_cump_mpspmv_dia.cuh"
#include "sparse/performance/dia/dbl/test_double_spmv_dia.cuh"
#include "sparse/performance/csr/dbl/test_taco_spmv_csr.cuh"

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

void test(const char * MATRIX_PATH, const int M, const int N, const int LINES, const bool SYMM, const string DATATYPE) {

    //Input arrays
    mpfr_t *vectorX = create_random_array(N, INP_BITS);
    double *AS;
    int *OFFSET;
    int NDIAG;

    //Convert a sparse matrix to the double-precision DIA format
    convert_to_dia(MATRIX_PATH, M, LINES, SYMM, NDIAG, AS, OFFSET);

    //Launch tests
    test_double_spmv_dia(M, N, NDIAG, OFFSET, AS, vectorX);
    //test_taco_spmv_csr(MATRIX_PATH, vectorX, DATATYPE);
    test_mpres_mpspmv_dia(M, N, NDIAG, OFFSET, AS, vectorX);
    test_campary_mpspmv_dia<CAMPARY_PRECISION>(M, N, NDIAG, OFFSET, AS, vectorX, INP_DIGITS);
    test_cump_mpspmv_dia(M, N, NDIAG, OFFSET, AS, vectorX, MP_PRECISION, INP_DIGITS);
    checkDeviceHasErrors(cudaDeviceSynchronize());
    // cudaCheckErrors(); //CUMP gives failure

    //Cleanup
    for(int i = 0; i < N; i++){
        mpfr_clear(vectorX[i]);
    }
    delete[] vectorX;
    delete[] AS;
    delete[] OFFSET;
    cudaDeviceReset();
}

int main(int argc, char *argv[]) {

    //The operation parameters. Read from an input file that contains a sparse matrix
    int M = 0; //number of rows
    int N = 0; //number of columns
    int MAXNZ = 0; //Maximum number of nonzeros per row in the matrix A
    int NZMD = 0; //number of nonzeros in the main diagonal of the matrix
    int LINES = 0; //number of lines in the input matrix file
    bool SYMM = false; //true if the input matrix is to be treated as symmetrical; otherwise false
    string DATATYPE; //defines type of data in MatrixMarket: real, integer, binary

    initialize();

    //Start logging
    Logger::beginTestDescription(Logger::SPMV_MP_DIA_PERFORMANCE_TEST);
    if(argc<=1) {
        printf("Matrix is not specified in command line arguments.");
        Logger::printSpace();
        Logger::endTestDescription();
        exit(1);
    }
    const char * MATRIX_PATH = argv[1];

    Logger::beginSection("Matrix properties:");
    Logger::printParam("Path", MATRIX_PATH);
    read_matrix_properties(MATRIX_PATH, M, N, LINES, MAXNZ, NZMD, SYMM, DATATYPE);
    int NNZ = SYMM ? ( (LINES - NZMD) * 2 + NZMD) : LINES;
    Logger::printParam("Number of rows, M", M);
    Logger::printParam("Number of column, N", N);
    Logger::printParam("Number of nonzeros, NNZ", NNZ);
    Logger::printParam("Maximum number of nonzeros per row, MAXNZ", MAXNZ);
    Logger::printParam("Average number of nonzeros per row, AVGNZ", (double)NNZ/M);
    Logger::printParam("Symmetry, SYMM", SYMM);
    Logger::printParam("Data type, DATATYPE", DATATYPE);
    Logger::printDash();
    Logger::beginSection("Additional info:");
    Logger::printParam("RNS_MODULI_SIZE", RNS_MODULI_SIZE);
    Logger::printParam("MP_PRECISION", MP_PRECISION);
    Logger::printParam("CAMPARY_PRECISION (n-double)", CAMPARY_PRECISION);
    Logger::endSection(true);

    //Run the test
    test(MATRIX_PATH, M, N, LINES, SYMM, DATATYPE);

    //Finalize
    finalize();

    //End logging
    Logger::endTestDescription();

    return 0;
}