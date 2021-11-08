/*
 *  Performance and memory consumption test for SpMV routines using various sparse matrix storage formats (double precision matrix)
 *  Path to the matrix must be given as a command line argument, e.g., ../../tests/sparse/matrices/t3dl.mtx
 *  The second argument (0 or 1) determines whether or not to test the DIA kernels.
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
#include "sparse/utils/csr_utils.cuh"
#include "sparse/utils/jad_utils.cuh"
#include "sparse/utils/ell_utils.cuh"
#include "sparse/utils/dia_utils.cuh"
#include "sparse/csr/test_double_spmv_csr.cuh"
#include "sparse/csr/test_mpfr_spmv_csr.cuh"
#include "sparse/csr/test_mpres_spmv_csr.cuh"
#include "sparse/csr/test_mpres_spmv_csrv.cuh"
#include "sparse/csr/test_campary_spmv_csr.cuh"
#include "sparse/csr/test_campary_spmv_csrv.cuh"
#include "sparse/csr/test_cump_spmv_csr.cuh"
#include "sparse/jad/test_mpres_spmv_jad.cuh"
#include "sparse/jad/test_mpres_spmv_jadv.cuh"
#include "sparse/jad/test_campary_spmv_jad.cuh"
#include "sparse/jad/test_cump_spmv_jad.cuh"
#include "sparse/ell/test_mpres_spmv_ell.cuh"
#include "sparse/ell/test_campary_spmv_ell.cuh"
#include "sparse/ell/test_cump_spmv_ell.cuh"
#include "sparse/dia/test_mpres_spmv_dia.cuh"
#include "sparse/dia/test_campary_spmv_dia.cuh"
#include "sparse/dia/test_cump_spmv_dia.cuh"



bool TEST_DIA = false; // Whether or not dia should be tested

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

/*
 * Memory evaluation
 */
void printMemoryConsumption(const char *dataType, double sizeOfMatrix, double sizeOfVectors){
    printf("%s\n", dataType);
    printf("\t- matrix storage format (MB): %lf\n", sizeOfMatrix);
    printf("\t- vectors x and y (MB): %lf\n", sizeOfVectors);
    printf("\t- total (vector + matrix) consumption (MB): %lf\n", sizeOfMatrix + sizeOfVectors);
}

void evaluateMemoryConsumption(const char * MATRIX_PATH, const int M, const int N, const int LINES, const int NNZ, const int MAXNZR, const bool SYMM){
    Logger::printDDash();
    Logger::beginSection("MEMORY EVALUATION");
    Logger::printDDash();
    int ndiag = 0;
    //Double
    printMemoryConsumption("Double CSR", get_dbl_csr_memory_consumption(M, NNZ), get_double_array_size_in_mb(M+N));
    Logger::printDash();
    printMemoryConsumption("Double JAD", get_dbl_jad_memory_consumption(M, N, NNZ, MAXNZR), get_double_array_size_in_mb(M+N));
    Logger::printDash();
    printMemoryConsumption("Double ELLPACK", get_dbl_ell_memory_consumption(M, MAXNZR), get_double_array_size_in_mb(M+N));
    if(TEST_DIA){
        Logger::printDash();
        ndiag = calc_ndiag(MATRIX_PATH, LINES, SYMM);
        printMemoryConsumption("Double DIA", get_dbl_dia_memory_consumption(M, ndiag), get_double_array_size_in_mb(M+N));
    }
    //MPRES-BLAS
    Logger::printDDash();
    printMemoryConsumption("MPRES-BLAS CSR", get_dbl_csr_memory_consumption(M, NNZ), get_mp_float_array_size_in_mb(M+N));
    Logger::printDash();
    printMemoryConsumption("MPRES-BLAS JAD", get_dbl_jad_memory_consumption(M, N, NNZ, MAXNZR), get_mp_float_array_size_in_mb(M+N));
    Logger::printDash();
    printMemoryConsumption("MPRES-BLAS ELLPACK", get_dbl_ell_memory_consumption(M, MAXNZR), get_mp_float_array_size_in_mb(M+N));
    if(TEST_DIA) {
        Logger::printDash();
        printMemoryConsumption("MPRES-BLAS DIA", get_dbl_dia_memory_consumption(M, ndiag), get_mp_float_array_size_in_mb(M + N));
    }
    //CAMPARY
    Logger::printDDash();
    printMemoryConsumption("CAMPARY CSR", get_dbl_csr_memory_consumption(M, NNZ), get_campary_array_size_in_mb<CAMPARY_PRECISION>(M+N));
    Logger::printDash();
    printMemoryConsumption("CAMPARY JAD", get_dbl_jad_memory_consumption(M, N, NNZ, MAXNZR), get_campary_array_size_in_mb<CAMPARY_PRECISION>(M+N));
    Logger::printDash();
    printMemoryConsumption("CAMPARY ELLPACK", get_dbl_ell_memory_consumption(M, MAXNZR), get_campary_array_size_in_mb<CAMPARY_PRECISION>(M+N));
    /*if(TEST_DIA) {
        Logger::printDash();
        printMemoryConsumption("CAMPARY DIA", get_dbl_dia_memory_consumption(M, ndiag),get_campary_array_size_in_mb<CAMPARY_PRECISION>(M + N));
    }*/
    //CUMP
    Logger::printDDash();
    printMemoryConsumption("CUMP CSR", get_cump_csr_memory_consumption(M, NNZ, MP_PRECISION), get_cump_array_size_in_mb(M+N+M, MP_PRECISION));
    Logger::printDash();
    printMemoryConsumption("CUMP JAD", get_cump_jad_memory_consumption(M, N, NNZ, MAXNZR, MP_PRECISION), get_cump_array_size_in_mb(M+N+M, MP_PRECISION));
    Logger::printDash();
    printMemoryConsumption("CUMP ELLPACK", get_cump_ell_memory_consumption(M, MAXNZR, MP_PRECISION), get_cump_array_size_in_mb(M+N+M, MP_PRECISION));
    /*if(TEST_DIA) {
        Logger::printDash();
        printMemoryConsumption("CUMP DIA", get_cump_dia_memory_consumption(M, ndiag, MP_PRECISION), get_cump_array_size_in_mb(M+N+M, MP_PRECISION));
    }*/
    Logger::printDDash();
    Logger::printSpace();
}


/*
 * Performance evaluation
 */
void evaluatePerformance(const char * MATRIX_PATH, const int M, const int N, const int LINES, const int NNZ, const int MAXNZR, const bool SYMM, const string DATATYPE) {
    Logger::printDDash();
    Logger::beginSection("PERFORMANCE EVALUATION");
    Logger::printDDash();

    //Input vector
    mpfr_t *vectorX = create_random_array(N, INP_BITS);

    //Matrix storage structures
    csr_t CSR;
    jad_t JAD;
    ell_t ELL;
    dia_t DIA;
    int NDIAG;

    if(TEST_DIA) {
        NDIAG = calc_ndiag(MATRIX_PATH, LINES, SYMM);
    }
    /*
     * Double and MPFR CSR tests
     */
    Logger::beginSection("***** Double precision and MPFR Tests *****");

    csr_init(CSR,M,NNZ);
    build_csr(MATRIX_PATH, M, NNZ, LINES, SYMM, CSR);
    test_double_spmv_csr(M, N, NNZ, CSR, vectorX);
    test_mpfr_spmv_csr(M, N, NNZ, CSR, vectorX, MP_PRECISION);
    Logger::printDash();


    /*
     * MPRES-BLAS tests
     */
    Logger::printSpace();
    Logger::printDash();
    Logger::beginSection("***** MPRES-BLAS Tests *****");

    //CSR
    test_mpres_spmv_csr(M, N, NNZ, CSR, vectorX);
    test_mpres_spmv_csrv<4>(M, N, NNZ, CSR, vectorX);
    test_mpres_spmv_csrv<16>(M, N, NNZ, CSR, vectorX);
    test_mpres_spmv_csrv<32>(M, N, NNZ, CSR, vectorX);
    csr_clear(CSR);

    //JAD
    jad_init(JAD, M, MAXNZR, NNZ);
    build_jad(MATRIX_PATH, M, MAXNZR, NNZ, LINES, SYMM, JAD);
    test_mpres_spmv_jad(M, N, MAXNZR, NNZ, JAD, vectorX);
/*  test_mpres_spmv_jadv<2>(M, N, MAXNZR, NNZ, JAD, vectorX);
    test_mpres_spmv_jadv<8>(M, N, MAXNZR, NNZ, JAD, vectorX);
    test_mpres_spmv_jadv<32>(M, N, MAXNZR, NNZ, JAD, vectorX);  */
    jad_clear(JAD);

    //ELLPACK
    ell_init(ELL, M, MAXNZR);
    build_ell(MATRIX_PATH, M, MAXNZR, LINES, SYMM, ELL);
    test_mpres_spmv_ell(M, N, MAXNZR, ELL, vectorX);
    ell_clear(ELL);

    //DIA
    if(TEST_DIA) {
        dia_init(DIA, M, NDIAG);
        build_dia(MATRIX_PATH, M, LINES, SYMM, DIA);
        test_mpres_spmv_dia(M, N, NDIAG, DIA, vectorX);
        dia_clear(DIA);
    }
    Logger::printDash();

    /*
     * CAMPARY tests
     */

    Logger::printSpace();
    Logger::printDash();
    Logger::beginSection("***** CAMPARY Tests *****");

    //CSR
    csr_init(CSR,M,NNZ);
    build_csr(MATRIX_PATH, M, NNZ, LINES, SYMM, CSR);
    test_campary_spmv_csr<CAMPARY_PRECISION>(M, N, NNZ, CSR, vectorX, INP_DIGITS);
    /*test_campary_spmv_csrv<CAMPARY_PRECISION, 4>(M, N, NNZ, CSR, vectorX, INP_DIGITS);
    test_campary_spmv_csrv<CAMPARY_PRECISION, 8>(M, N, NNZ, CSR, vectorX, INP_DIGITS);*/
    csr_clear(CSR);

    //JAD
    jad_init(JAD, M, MAXNZR, NNZ);
    build_jad(MATRIX_PATH, M, MAXNZR, NNZ, LINES, SYMM, JAD);
    test_campary_spmv_jad<CAMPARY_PRECISION>(M, N, MAXNZR, NNZ, JAD, vectorX, INP_DIGITS);
    jad_clear(JAD);

    //ELLPACK
    ell_init(ELL, M, MAXNZR);
    build_ell(MATRIX_PATH, M, MAXNZR, LINES, SYMM, ELL);
    test_campary_spmv_ell<CAMPARY_PRECISION>(M, N, MAXNZR, ELL, vectorX, INP_DIGITS);
    ell_clear(ELL);

    //DIA
    /*if(TEST_DIA) {
        dia_init(DIA, M, NDIAG);
        build_dia(MATRIX_PATH, M, LINES, SYMM, DIA);
        test_campary_spmv_dia<CAMPARY_PRECISION>(M, N, NDIAG, DIA, vectorX, INP_DIGITS);
        dia_clear(DIA);
    }*/

    Logger::printDash();

    /*
     * CUMP tests
     */
    Logger::printSpace();
    Logger::printDash();
    Logger::beginSection("***** CUMP Tests *****");

    //CSR
    csr_init(CSR,M,NNZ);
    build_csr(MATRIX_PATH, M, NNZ, LINES, SYMM, CSR);
    test_cump_spmv_csr(M, N, NNZ, CSR, vectorX, MP_PRECISION, INP_DIGITS);
    csr_clear(CSR);

    //JAD
    jad_init(JAD, M, MAXNZR, NNZ);
    build_jad(MATRIX_PATH, M, MAXNZR, NNZ, LINES, SYMM, JAD);
    test_cump_spmv_jad(M, N, MAXNZR, NNZ, JAD, vectorX, MP_PRECISION, INP_DIGITS);
    jad_clear(JAD);

    //ELLPACK
    ell_init(ELL, M, MAXNZR);
    build_ell(MATRIX_PATH, M, MAXNZR, LINES, SYMM, ELL);
    test_cump_spmv_ell(M, N, MAXNZR, ELL, vectorX, MP_PRECISION, INP_DIGITS);
    ell_clear(ELL);

    //DIA
    /*if(TEST_DIA) {
        dia_init(DIA, M, NDIAG);
        build_dia(MATRIX_PATH, M, LINES, SYMM, DIA);
        test_cump_spmv_dia(M, N, NDIAG, DIA, vectorX, MP_PRECISION, INP_DIGITS);
        dia_clear(DIA);
    }*/
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
    Logger::beginTestDescription(Logger::SPMV_TEST);
    if(argc< 2) {
        printf("Matrix is not specified in command line arguments.");
        Logger::printSpace();
        Logger::endTestDescription();
        exit(1);
    }
    const char * MATRIX_PATH = argv[1];
    if(argc > 2){
        TEST_DIA = atoi(argv[2]);
    }

    Logger::beginSection("Matrix properties:");
    Logger::printParam("Path", MATRIX_PATH);
    collect_mtx_stats(MATRIX_PATH, M, N, LINES, MAXNZR, NZMD, SYMM, DATATYPE);
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

    //Memory Evaluation
    evaluateMemoryConsumption(MATRIX_PATH, M, N, LINES, NNZ, MAXNZR, SYMM);

    //Performance evaluation
    evaluatePerformance(MATRIX_PATH, M, N, LINES, NNZ, MAXNZR, SYMM, DATATYPE);

    //Finalize
    finalize();

    //End logging
    Logger::endTestDescription();

    return 0;
}