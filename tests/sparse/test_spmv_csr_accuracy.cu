/*
 *  Accuracy test for SpMV routines using the CSR matrix format (double precision matrix)
 *  Path to the matrix and vector must be given as a command line argument,
 *  e.g., ../../tests/sparse/matrices/t3dl.mtx ../../tests/sparse/matrices/t3dl_vector_x.mtx
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
#include "sparse/accuracy/spmv_csr_accuracy_utils.cuh"
#include "sparse/accuracy/test_mpres_spmv_csr_accuracy.cuh"
#include "sparse/accuracy/test_double_spmv_csr_accuracy.cuh"

void initialize() {
    cudaDeviceReset();
    rns_const_init();
    mp_const_init();
    checkDeviceHasErrors(cudaDeviceSynchronize());
    cudaCheckErrors();
}

void test(const char * MATRIX_PATH, const char * VECTOR_PATH, const int M, const int N, const int LINES, const int NNZ, const int MAXNZR, const bool SYMM, const string DATATYPE) {
    //Matrix
    csr_t CSR;
    csr_init(CSR, M, NNZ);
    build_csr(MATRIX_PATH, M, NNZ, LINES, SYMM, CSR);
    //Input vector
    double * vectorX = new double [N];
    read_vector(VECTOR_PATH, N, vectorX);

    //Allocate additional memory
    mpfr_t * exact = new mpfr_t[M];
    mpfr_t * vectorY = new mpfr_t[M];
    mpfr_t exact_norm, cond, u, g, residual;
    for(int i = 0; i < M; i++){
        mpfr_init2(exact[i], REFERENCE_PRECISION);
        mpfr_init2(vectorY[i], REFERENCE_PRECISION);
    }
    mpfr_init2(exact_norm, REFERENCE_PRECISION);
    mpfr_init2(cond, REFERENCE_PRECISION);
    mpfr_init2(u, REFERENCE_PRECISION);
    mpfr_init2(g, REFERENCE_PRECISION);
    mpfr_init2(residual, REFERENCE_PRECISION);

    //Compute exact result and its norm
    exact_spmv(M, CSR.irp, CSR.ja, CSR.as, vectorX, exact);
    norm2(exact_norm, exact, M);
    //Compute condition number of the SpMV
    spmv_condition_number(cond, M, CSR.irp, CSR.ja, CSR.as, vectorX, exact_norm);
    mpfr_printf("SpMV condition number: %.25Re\n", cond);

    /**
     * Test MPRES-BLAS
     */

    Logger::printDash();
    PrintTimerName("[GPU] MPRES-BLAS CSR (mp_spmv_csr)");

    test_mpres_spmv_csr_accuracy(M, N, NNZ, MAXNZR, CSR, vectorX, vectorY);
    //print_mpfr_sum(vectorY, M);

    //Compute unit roundoff
    unit_roundoff_mpres(u);
    mpfr_printf("- Unit roundoff:\t\t\t %.25Re\n", u);

    //Compute error bound = gamma * cond
    gamma(g, u, MAXNZR);
    mpfr_mul(g, g, cond, MPFR_RNDN);
    mpfr_printf("- Relative error bound:\t\t %.25Re\n", g);

    //Compute actual relative error
    spmv_relative_residual(residual, M, vectorY, exact, exact_norm);
    mpfr_printf("- Actual relative residual:\t %.25Re\n", residual);

    /**
     * Test double
     */
    Logger::printDash();
    PrintTimerName("[GPU] double CSR");
    for(int i = 0; i < M; i++){
        mpfr_set_d(vectorY[i], 0, MPFR_RNDN);
    }
    test_double_spmv_csr_accuracy(M, N, NNZ, MAXNZR, CSR, vectorX, vectorY);
    //print_mpfr_sum(vectorY, M);

    //Compute unit roundoff
    unit_roundoff_double(u);
    mpfr_printf("- Unit roundoff:\t\t\t %.25Re\n", u);

    //Compute error bound = gamma * cond
    gamma(g, u, MAXNZR);
    mpfr_mul(g, g, cond, MPFR_RNDN);
    mpfr_printf("- Relative error bound:\t\t %.25Re\n", g);

    //Compute actual relative error
    spmv_relative_residual(residual, M, vectorY, exact, exact_norm);
    mpfr_printf("- Actual relative residual:\t %.25Re\n", residual);

    //Cleanup
    mpfr_clear(exact_norm);
    mpfr_clear(cond);
    mpfr_clear(u);
    mpfr_clear(g);
    mpfr_clear(residual);
    for(int i = 0; i < M; i++){
        mpfr_clear(vectorY[i]);
        mpfr_clear(exact[i]);
    }
    delete[] vectorX;
    delete[] vectorY;
    delete[] exact;
    csr_clear(CSR);
    cudaDeviceReset();

}

int main(int argc, char *argv[]) {

    initialize();

    //The operation parameters. Read from an input file that contains a sparse matrix
    int M = 0; //number of rows
    int N = 0; //number of columns
    int NNZ = 0; //number of nonzeros in matrix
    int MAXNZR = 0; //maximum number of nonzeros per row in the matrix A
    int NZMD = 0; //number of nonzeros in the main diagonal of the matrix
    int LINES = 0; //number of lines in the input matrix file
    bool SYMM = false; //true if the input matrix is to be treated as symmetrical; otherwise false
    string DATATYPE; //defines type of data in MatrixMarket: real, integer, binary

    //Start logging
    Logger::beginTestDescription(Logger::SPMV_CSR_ACCURACY_TEST);
    if(argc<=1) {
        printf("Matrix is not specified in command line arguments.");
        Logger::printSpace();
        Logger::endTestDescription();
        exit(1);
    }
    const char * MATRIX_PATH = argv[1];
    const char * VECTOR_PATH = argv[2];

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
    Logger::printParam("REFERENCE_PRECISION", REFERENCE_PRECISION);
    Logger::endSection(true);

    //Run the test
    test(MATRIX_PATH, VECTOR_PATH, M, N, LINES, NNZ, MAXNZR, SYMM, DATATYPE);

    Logger::endTestDescription();
    return 0;
}