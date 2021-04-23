/*
 *  Performance test for SpMV routines using various sparse matrix storage formats (double precision matrix)
 *  Path to the matrix must be given as a command line argument, e.g., ../../tests/sparse/matrices/t3dl.mtx
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

#include "mpfr.h"
#include "logger.cuh"
#include "sparse/matrix_converter.cuh"
#include "sparse/performance/csr/dbl/test_double_spmv_csr.cuh"
#include "sparse/performance/csr/mpd/test_mpfr_mpdspmv_csr.cuh"
#include "sparse/performance/csr/mpd/test_mpres_mpdspmv_csr_scalar.cuh"
#include "sparse/performance/csr/mpd/test_mpres_mpdspmv_csr_vector.cuh"
#include "sparse/performance/jad/mpd/test_mpres_mpdspmv_jad.cuh"
#include "sparse/performance/ellpack/mpd/test_mpres_mpdspmv_ellpack_scalar.cuh"
#include "sparse/performance/dia/mpd/test_mpres_mpdspmv_dia.cuh"
#include "sparse/performance/csr/mpd/test_campary_mpdspmv_csr_scalar.cuh"
#include "sparse/performance/jad/mpd/test_campary_mpdspmv_jad.cuh"
#include "sparse/performance/ellpack//mpd/test_campary_mpdspmv_ellpack.cuh"
#include "sparse/performance/dia/mpd/test_campary_mpdspmv_dia.cuh"
#include "sparse/performance/csr/mp/test_cump_mpspmv_csr_scalar.cuh"
#include "sparse/performance/jad/mp/test_cump_mpspmv_jad.cuh"
#include "sparse/performance/ellpack/mp/test_cump_mpspmv_ellpack.cuh"
#include "sparse/performance/dia/mp/test_cump_mpspmv_dia.cuh"


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

void test(const char * MATRIX_PATH, const int M, const int N, const int LINES, const int NNZ, const int MAXNZR, const bool SYMM, const string DATATYPE) {
    //TODO: CSR, JAD, ELL, DIA
    //Input vector
    mpfr_t *vectorX = create_random_array(N, INP_BITS);

    //Matrix storage arrays
    double *AS;
    int *JA;
    int *IRP;
    int *JCP;
    int *PERM_ROWS;
    int *OFFSET;
    int NDIAG;

    /*
     * Double and MPFR CSR tests
     */
    Logger::printSpace();
    Logger::printDash();
    Logger::beginSection("***** Double precision and MPFR Tests *****");

    AS = new double [NNZ]();
    JA = new int[NNZ]();
    IRP = new int[M + 1]();
    convert_to_csr(MATRIX_PATH, M, NNZ, LINES, SYMM, AS, IRP, JA);
    test_double_spmv_csr(M, N, NNZ, IRP, JA, AS, vectorX);
    test_mpfr_mpdspmv_csr(M, N, NNZ, IRP, JA, AS, vectorX, MP_PRECISION);
    Logger::printDash();


    /*
     * MPRES-BLAS tests
     */
    Logger::printSpace();
    Logger::printSpace();
    Logger::printDash();
    Logger::beginSection("***** MPRES-BLAS Tests *****");

    //MPRES
    test_mpres_mpdspmv_csr_scalar(M, N, NNZ, IRP, JA, AS, vectorX);
    test_mpres_mpdspmv_csr_vector<8>(M, N, NNZ, IRP, JA, AS, vectorX);
    test_mpres_mpdspmv_csr_vector<16>(M, N, NNZ, IRP, JA, AS, vectorX);
    test_mpres_mpdspmv_csr_vector<32>(M, N, NNZ, IRP, JA, AS, vectorX);
    delete[] AS;
    delete[] JA;
    delete[] IRP;

    //JAD
    AS = new double [NNZ]();
    JA = new int[NNZ]();
    JCP = new int[MAXNZR + 1]();
    PERM_ROWS = new int[M]();
    convert_to_jad(MATRIX_PATH, M, MAXNZR, NNZ, LINES, SYMM, AS, JCP, JA, PERM_ROWS);
    test_mpres_mpdspmv_jad(M, N, MAXNZR, NNZ, JA, JCP, AS, PERM_ROWS, vectorX);
    delete[] AS;
    delete[] JA;
    delete[] JCP;
    delete[] PERM_ROWS;

    //ELLPACK
    AS = new double [M * MAXNZR]();
    JA = new int[M * MAXNZR]();
    convert_to_ellpack(MATRIX_PATH, M, MAXNZR, LINES, SYMM, AS, JA);
    test_mpres_mpdspmv_ellpack_scalar(M, N, MAXNZR, JA, AS, vectorX);
    delete[] AS;
    delete[] JA;

    //DIA
    convert_to_dia(MATRIX_PATH, M, LINES, SYMM, NDIAG, AS, OFFSET);
    test_mpres_mpdspmv_dia(M, N, NDIAG, OFFSET, AS, vectorX);
    delete[] AS;
    delete[] OFFSET;
    Logger::printDash();

    /*
     * CAMPARY tests
     */
    //TODO Print Memory consumption

    Logger::printSpace();
    Logger::printSpace();
    Logger::printDash();
    Logger::beginSection("***** CAMPARY Tests *****");

    //CSR
    AS = new double [NNZ]();
    JA = new int[NNZ]();
    IRP = new int[M + 1]();
    convert_to_csr(MATRIX_PATH, M, NNZ, LINES, SYMM, AS, IRP, JA);
    test_campary_mpdspmv_csr_scalar<CAMPARY_PRECISION>(M, N, NNZ, IRP, JA, AS, vectorX, INP_DIGITS);
    delete[] AS;
    delete[] JA;
    delete[] IRP;

    //JAD
    AS = new double [NNZ]();
    JA = new int[NNZ]();
    JCP = new int[MAXNZR + 1]();
    PERM_ROWS = new int[M]();
    convert_to_jad(MATRIX_PATH, M, MAXNZR, NNZ, LINES, SYMM, AS, JCP, JA, PERM_ROWS);
    test_campary_mpdspmv_jad<CAMPARY_PRECISION>(M, N, MAXNZR, NNZ, JA, JCP, AS, PERM_ROWS, vectorX, INP_DIGITS);
    delete[] AS;
    delete[] JA;
    delete[] JCP;
    delete[] PERM_ROWS;

    //ELLPACK
    AS = new double [M * MAXNZR]();
    JA = new int[M * MAXNZR]();
    convert_to_ellpack(MATRIX_PATH, M, MAXNZR, LINES, SYMM, AS, JA);
    test_campary_mpdspmv_ellpack<CAMPARY_PRECISION>(M, N, MAXNZR, JA, AS, vectorX, INP_DIGITS);
    delete[] AS;
    delete[] JA;

    //DIA
    convert_to_dia(MATRIX_PATH, M, LINES, SYMM, NDIAG, AS, OFFSET);
    test_campary_mpdspmv_dia<CAMPARY_PRECISION>(M, N, NDIAG, OFFSET, AS, vectorX, INP_DIGITS);
    delete[] AS;
    delete[] OFFSET;
    Logger::printDash();

    /*
     * CUMP tests
     */
    Logger::printSpace();
    Logger::printSpace();
    Logger::printDash();
    Logger::beginSection("***** CUMP Tests *****");

    //CSR
    AS = new double [NNZ]();
    JA = new int[NNZ]();
    IRP = new int[M + 1]();
    convert_to_csr(MATRIX_PATH, M, NNZ, LINES, SYMM, AS, IRP, JA);
    test_cump_mpspmv_csr_scalar(M, N, NNZ, IRP, JA, AS, vectorX, MP_PRECISION, INP_DIGITS);
    delete[] AS;
    delete[] JA;
    delete[] IRP;

    //JAD
    AS = new double [NNZ]();
    JA = new int[NNZ]();
    JCP = new int[MAXNZR + 1]();
    PERM_ROWS = new int[M]();
    convert_to_jad(MATRIX_PATH, M, MAXNZR, NNZ, LINES, SYMM, AS, JCP, JA, PERM_ROWS);
    test_cump_mpspmv_jad(M, N, MAXNZR, NNZ, JA, JCP, AS, PERM_ROWS, vectorX, MP_PRECISION, INP_DIGITS);
    delete[] AS;
    delete[] JA;
    delete[] JCP;
    delete[] PERM_ROWS;

    //ELLPACK
    AS = new double [M * MAXNZR]();
    JA = new int[M * MAXNZR]();
    convert_to_ellpack(MATRIX_PATH, M, MAXNZR, LINES, SYMM, AS, JA);
    test_cump_mpspmv_ellpack(M, N, MAXNZR, JA, AS, vectorX, MP_PRECISION, INP_DIGITS);
    delete[] AS;
    delete[] JA;

    //DIA
    convert_to_dia(MATRIX_PATH, M, LINES, SYMM, NDIAG, AS, OFFSET);
    test_cump_mpspmv_dia(M, N, NDIAG, OFFSET, AS, vectorX, MP_PRECISION, INP_DIGITS);
    delete[] AS;
    delete[] OFFSET;

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

    initialize();

    //Start logging
    Logger::beginTestDescription(Logger::SPMV_MPD_PERFORMANCE_TEST);
    if(argc<=1) {
        printf("Matrix is not specified in command line arguments.");
        Logger::printSpace();
        Logger::endTestDescription();
        exit(1);
    }
    const char * MATRIX_PATH = argv[1];

    Logger::beginSection("Matrix properties:");
    Logger::printParam("Path", MATRIX_PATH);
    read_matrix_properties(MATRIX_PATH, M, N, LINES, MAXNZR, NZMD, SYMM, DATATYPE);
    NNZ = SYMM ? ( (LINES - NZMD) * 2 + NZMD) : LINES;
    Logger::printParam("Number of rows, M", M);
    Logger::printParam("Number of column, N", N);
    Logger::printParam("Number of nonzeros, NNZ", NNZ);
    Logger::printParam("Maximum number of nonzeros per row, MAXNZR", MAXNZR);
    Logger::printParam("Average number of nonzeros per row, AVGNZR", (double)NNZ/M);
    Logger::printParam("Symmetry", SYMM);
    Logger::printParam("Data type", DATATYPE);
    Logger::printDash();
    Logger::beginSection("Additional info:");
    Logger::printParam("RNS_MODULI_SIZE", RNS_MODULI_SIZE);
    Logger::printParam("MP_PRECISION", MP_PRECISION);
    Logger::printParam("CAMPARY_PRECISION (n-double)", CAMPARY_PRECISION);
    Logger::endSection(true);

    //Run the test
    test(MATRIX_PATH, M, N, LINES, NNZ, MAXNZR, SYMM, DATATYPE);

    //Finalize
    finalize();

    //End logging
    Logger::endTestDescription();

    return 0;
}