/*
 *  Performance test for SpMV routines using the CSR matrix format (multiple precision matrix)
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
#include "sparse/utils/mtx_reader.cuh"
#include "sparse/csr/test_mpfr_spmv_csr.cuh"
#include "sparse/mpmtx/csr/test_mpres_spmv_mpmtx_csr2st.cuh"
#include "sparse/mpmtx/csr/test_mpres_spmv_mpmtx_csr.cuh"
#include "sparse/mpmtx/csr/test_mpres_spmv_mpmtx_csrv.cuh"
#include "sparse/mpmtx/csr/test_campary_spmv_mpmtx_csr.cuh"
#include "sparse/mpmtx/csr/test_campary_spmv_mpmtx_csrv.cuh"
#include "sparse/csr/test_cump_spmv_csr.cuh"
#include "sparse/csr/test_double_spmv_csr.cuh"
#include "sparse/csr/test_taco_spmv_csr.cuh"

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

void test(const char * MATRIX_PATH, const int M, const int N, const int LINES, const int NNZ, const bool SYMM, const string DATATYPE) {
    //Inputs
    mpfr_t *vectorX = create_random_array(N, INP_BITS);
    csr_t CSR;
    csr_init(CSR,M,NNZ);
    build_csr(MATRIX_PATH, M, NNZ, LINES, SYMM, CSR);
    //Launch tests
    test_mpfr_spmv_csr(M, N, NNZ, CSR, vectorX, MP_PRECISION);
    test_double_spmv_csr(M, N, NNZ, CSR, vectorX);
    //test_taco_spmv_csr(MATRIX_PATH, vectorX, DATATYPE);
    Logger::printStars();
    test_mpres_spmv_mpmtx_csrv<2>(M, N, NNZ, CSR, vectorX);
    test_mpres_spmv_mpmtx_csrv<4>(M, N, NNZ, CSR, vectorX);
    test_mpres_spmv_mpmtx_csrv<8>(M, N, NNZ, CSR, vectorX);
    test_mpres_spmv_mpmtx_csrv<16>(M, N, NNZ, CSR, vectorX);
    test_mpres_spmv_mpmtx_csrv<32>(M, N, NNZ, CSR, vectorX);
    test_mpres_spmv_mpmtx_csr(M, N, NNZ, CSR, vectorX);
    test_mpres_spmv_mpmtx_csr2st(M, N, NNZ, CSR, vectorX);
    Logger::printStars();
    test_campary_spmv_mpmtx_csrv<CAMPARY_PRECISION, 2>(M, N, NNZ, CSR, vectorX, INP_DIGITS);
    test_campary_spmv_mpmtx_csrv<CAMPARY_PRECISION, 4>(M, N, NNZ, CSR, vectorX, INP_DIGITS);
    test_campary_spmv_mpmtx_csrv<CAMPARY_PRECISION, 8>(M, N, NNZ, CSR, vectorX, INP_DIGITS);
    test_campary_spmv_mpmtx_csrv<CAMPARY_PRECISION, 16>(M, N, NNZ, CSR, vectorX, INP_DIGITS);
    test_campary_spmv_mpmtx_csrv<CAMPARY_PRECISION, 32>(M, N, NNZ, CSR, vectorX, INP_DIGITS);
    test_campary_spmv_mpmtx_csr<CAMPARY_PRECISION>(M, N, NNZ, CSR, vectorX, INP_DIGITS);
    Logger::printStars();
    test_cump_spmv_csr(M, N, NNZ, CSR, vectorX, MP_PRECISION, INP_DIGITS);
    checkDeviceHasErrors(cudaDeviceSynchronize());
    // cudaCheckErrors(); //CUMP gives failure
    //Cleanup
    for(int i = 0; i < N; i++){
        mpfr_clear(vectorX[i]);
    }
    delete[] vectorX;
    csr_clear(CSR);
    cudaDeviceReset();
}

int main(int argc, char *argv[]) {

    //The operation parameters. Read from an input file that contains a sparse matrix
    int M = 0; //number of rows
    int N = 0; //number of columns
    int NNZ = 0; //number of nonzeros in matrix
    int MAXNZR = 0; //Maximum number of nonzeros per row in the matrix A
    int NZMD = 0; //number of nonzeros in the main diagonal of the matrix
    int LINES = 0; //number of lines in the input matrix file
    bool SYMM = false; //true if the input matrix is to be treated as symmetrical; otherwise false
    string DATATYPE; //defines type of data in MatrixMarket: real, integer, binary

    initialize();

    //Start logging
    Logger::beginTestDescription(Logger::SPMV_MPMTX_CSR_TEST);
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
    Logger::printParam("Number of rows, M", M);
    Logger::printParam("Number of column, N", N);
    Logger::printParam("Number of nonzeros, NNZ", NNZ);
    Logger::printParam("Maximum number of nonzeros per row, MAXNZR", MAXNZR);
    Logger::printParam("Average number of nonzeros per row, AVGNZR", (double)NNZ/M);
    Logger::printParam("Symmetry, SYMM", SYMM);
    Logger::printParam("Data type, DATATYPE", DATATYPE);
    Logger::printDash();
    Logger::beginSection("Additional info:");
    Logger::printParam("RNS_MODULI_SIZE", RNS_MODULI_SIZE);
    Logger::printParam("MP_PRECISION", MP_PRECISION);
    Logger::printParam("CAMPARY_PRECISION (n-double)", CAMPARY_PRECISION);
    Logger::endSection(true);

    //Run the test
    test(MATRIX_PATH, M, N, LINES, NNZ, SYMM, DATATYPE);

    //Finalize
    finalize();

    //End logging
    Logger::endTestDescription();

    return 0;
}