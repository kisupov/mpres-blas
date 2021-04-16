/*
 *  Performance test for SpMV routines using the JAD (JDS) matrix format (double precision matrix)
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
#include "sparse/performance/jad/mpd/test_mpres_mpdspmv_jad.cuh"
#include "sparse/performance/jad/mpd/test_campary_mpdspmv_jad.cuh"
#include "sparse/performance/jad/mp/test_cump_mpspmv_jad.cuh"
#include "sparse/performance/jad/dbl/test_double_spmv_jad.cuh"
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


void test(const char * MATRIX_PATH, const int M, const int N, const int LINES, const int NZR, const int NNZ, const bool SYMM, const string DATATYPE) {

    //Input arrays
    mpfr_t *vectorX = create_random_array(N, INP_BITS);
    auto *AS = new double [NNZ]();
    auto *JA = new int[NNZ]();
    auto *JCP = new int[NZR + 1]();
    auto *PERM_ROWS = new int[M]();

    //Convert a sparse matrix to the double-precision JAD (JDS) format
    convert_to_jad(MATRIX_PATH, M, NZR, NNZ, LINES, SYMM, AS, JCP, JA, PERM_ROWS);

    //Launch tests
    test_double_spmv_jad(M, N, NZR, NNZ, JA, JCP, AS, PERM_ROWS, vectorX);
    //test_taco_spmv_csr(MATRIX_PATH, vectorX, DATATYPE);
    test_mpres_mpdspmv_jad(M, N, NZR, NNZ, JA, JCP, AS, PERM_ROWS, vectorX);
    test_campary_mpdspmv_jad<CAMPARY_PRECISION>(M, N, NZR, NNZ, JA, JCP, AS, PERM_ROWS, vectorX, INP_DIGITS);
    test_cump_mpspmv_jad(M, N, NZR, NNZ, JA, JCP, AS, PERM_ROWS, vectorX, MP_PRECISION, INP_DIGITS);
    checkDeviceHasErrors(cudaDeviceSynchronize());
    // cudaCheckErrors(); //CUMP gives failure

    //Cleanup
    for(int i = 0; i < N; i++){
        mpfr_clear(vectorX[i]);
    }
    delete[] vectorX;
    delete[] AS;
    delete[] JA;
    delete[] JCP;
    delete[] PERM_ROWS;
    cudaDeviceReset();
}

int main(int argc, char *argv[]) {

    //The operation parameters. Read from an input file that contains a sparse matrix
    int M = 0; //number of rows
    int N = 0; //number of columns
    int NNZ = 0; //number of nonzeros in matrix
    int NZR = 0; //number of nonzeros per row array (maximum number of nonzeros per row in the matrix A)
    int NZMD = 0; //number of nonzeros in main diagonal of the matrix
    int LINES = 0; //number of lines in the input matrix file
    bool SYMM = false; //true if the input matrix is to be treated as symmetrical; otherwise false
    string DATATYPE; //defines type of data in MatrixMarket: real, integer, binary

    initialize();

    //Start logging
    Logger::beginTestDescription(Logger::SPMV_MPD_JAD_PERFORMANCE_TEST);
    if(argc<=1) {
        printf("Matrix is not specified in command line arguments.");
        Logger::printSpace();
        Logger::endTestDescription();
        exit(1);
    }
    const char * MATRIX_PATH = argv[1];

    Logger::beginSection("Operation info:");
    Logger::printParam("Matrix path", MATRIX_PATH);
    read_matrix_properties(MATRIX_PATH, M, N, LINES, NZR, NZMD, SYMM, DATATYPE);
    NNZ = SYMM ? ( (LINES - NZMD) * 2 + NZMD) : LINES;
    Logger::printParam("Number of rows in matrix, M", M);
    Logger::printParam("Number of column in matrix, N", N);
    Logger::printParam("Number of nonzeros in matrix, NNZ", NNZ);
    Logger::printParam("Number of nonzeros per row array, NZR", NZR);
    Logger::printParam("Symmetry of matrix, SYMM", SYMM);
    Logger::printParam("Data type, DATATYPE", DATATYPE);
    Logger::printDash();
    Logger::beginSection("Additional info:");
    Logger::printParam("RNS_MODULI_SIZE", RNS_MODULI_SIZE);
    Logger::printParam("MP_PRECISION", MP_PRECISION);
    Logger::printParam("CAMPARY_PRECISION (n-double)", CAMPARY_PRECISION);
    Logger::endSection(true);

    //Run the test
    test(MATRIX_PATH, M, N, LINES, NZR, NNZ, SYMM, DATATYPE);

    //Finalize
    finalize();

    //End logging
    Logger::endTestDescription();

    return 0;
}