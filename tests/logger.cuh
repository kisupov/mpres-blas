/*
 *  Logging functions
 *
 *  Copyright 2018, 2019 by Konstantin Isupov and Alexander Kuvaev.
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

#ifndef MPRES_TEST_LOGGER_CUH
#define MPRES_TEST_LOGGER_CUH

#include <iostream>

namespace Logger {

    // Tests headers
    enum TestHeader {
        BLAS_ASUM_PERFORMANCE_TEST,
        BLAS_DOT_PERFORMANCE_TEST,
        BLAS_SCAL_PERFORMANCE_TEST,
        BLAS_AXPY_PERFORMANCE_TEST,
        BLAS_NORM_PERFORMANCE_TEST,
        BLAS_AXPY_DOT_PERFORMANCE_TEST,
        BLAS_ROT_PERFORMANCE_TEST,
        BLAS_GEMV_PERFORMANCE_TEST,
        BLAS_SYMV_PERFORMANCE_TEST,
        BLAS_GER_PERFORMANCE_TEST,
        BLAS_SYR_PERFORMANCE_TEST,
        BLAS_SYR2_PERFORMANCE_TEST,
        BLAS_SYRK_PERFORMANCE_TEST,
        BLAS_GE_ADD_PERFORMANCE_TEST,
        BLAS_GE_ACC_PERFORMANCE_TEST,
        BLAS_GE_DIAG_SCALE_PERFORMANCE_TEST,
        BLAS_GE_LRSCALE_PERFORMANCE_TEST,
        BLAS_GEMM_PERFORMANCE_TEST,
        BLAS_GE_NORM_PERFORMANCE_TEST,
        BLAS_ASUM_ACCURACY_TEST,
        BLAS_DOT_ACCURACY_TEST,
        BLAS_SCAL_ACCURACY_TEST,
        BLAS_AXPY_ACCURACY_TEST,
        RNS_EVAL_ACCURACY_TEST,
        EXTRANGE_CORRECTNESS_TEST,
        RNS_CMP_PERFORMANCE_TEST,
        ARITH_PEAK_PERFORMANCE_TEST,
        SPMV_MPMTX_CSR_TEST,
        SPMV_CSR_TEST,
        SPMV_MPMTX_ELL_TEST,
        SPMV_ELL_TEST,
        SPMV_MPMTX_DIA_TEST,
        SPMV_DIA_TEST,
        SPMV_MPMTX_JAD_TEST,
        SPMV_JAD_TEST,
        SPMV_TEST,
        SPMV_CSR_ACCURACY_TEST,
        SPMV_VS_GEMV_TEST,
        CG_CSR_TEST
    };

    const char *testHeaderAsString(enum TestHeader header) {
        switch (header) {
            case BLAS_ASUM_PERFORMANCE_TEST:
                return "Performance test for BLAS ASUM routines";
            case BLAS_DOT_PERFORMANCE_TEST:
                return "Performance test for BLAS DOT routines";
            case BLAS_SCAL_PERFORMANCE_TEST:
                return "Performance test for BLAS SCAL routines";
            case BLAS_AXPY_PERFORMANCE_TEST:
                return "Performance test for BLAS AXPY routines";
            case BLAS_NORM_PERFORMANCE_TEST:
                return "Performance test for BLAS NORM routines";
            case BLAS_AXPY_DOT_PERFORMANCE_TEST:
                return "Performance test for BLAS AXPY_DOT routines";
            case BLAS_ROT_PERFORMANCE_TEST:
                return "Performance test for BLAS ROT routines";
            case BLAS_GEMV_PERFORMANCE_TEST:
                return "Performance test for BLAS GEMV routines";
            case BLAS_SYMV_PERFORMANCE_TEST:
                return "Performance test for BLAS SYMV routines";
            case BLAS_GER_PERFORMANCE_TEST:
                return "Performance test for BLAS GER routines";
            case BLAS_SYR_PERFORMANCE_TEST:
                return "Performance test for BLAS SYR routines";
            case BLAS_SYR2_PERFORMANCE_TEST:
                return "Performance test for BLAS SYR2 routines";
            case BLAS_SYRK_PERFORMANCE_TEST:
                return "Performance test for BLAS SYRK routines";
            case BLAS_GE_ADD_PERFORMANCE_TEST:
                return "Performance test for BLAS GE_ADD routines";
            case BLAS_GE_ACC_PERFORMANCE_TEST:
                return "Performance test for BLAS GE_ACC routines";
            case BLAS_GE_DIAG_SCALE_PERFORMANCE_TEST:
                return "Performance test for BLAS GE_DIAG_SCALE routines";
            case BLAS_GE_LRSCALE_PERFORMANCE_TEST:
                return "Performance test for BLAS GE_LRSCALE routines";
            case BLAS_GEMM_PERFORMANCE_TEST:
                return "Performance test for BLAS GEMM routines";
            case BLAS_GE_NORM_PERFORMANCE_TEST:
                return "Performance test for BLAS GE_NORM routines";
            case BLAS_ASUM_ACCURACY_TEST:
                return "Accuracy test for MPRES-BLAS ASUM routine";
            case BLAS_DOT_ACCURACY_TEST:
                return "Accuracy test for MPRES-BLAS DOT routine";
            case BLAS_SCAL_ACCURACY_TEST:
                return "Accuracy test for MPRES-BLAS SCAL routine";
            case BLAS_AXPY_ACCURACY_TEST:
                return "Accuracy test for MPRES-BLAS AXPY routine";
            case RNS_EVAL_ACCURACY_TEST:
                return "Test for checking the correctness and accuracy of the algorithms that calculate the RNS interval evaluation";
            case EXTRANGE_CORRECTNESS_TEST:
                return "Test for checking the correctness of the extended-range floating-point routines";
            case RNS_CMP_PERFORMANCE_TEST:
                return "Test for measure the performance of the RNS magnitude comparison algorithms";
            case ARITH_PEAK_PERFORMANCE_TEST:
                return " Microbenchmark for evaluating the peak performance of multiple-precision arithmetic (addition and multiplication)";
            case SPMV_TEST:
                return "Performance and memory consumption test for SpMV routines using various matrix storage formats (double precision matrix)";
            case SPMV_CSR_TEST:
                return "Performance test for SpMV routines using the CSR matrix storage format (double precision matrix)";
            case SPMV_JAD_TEST:
                return "Performance test for SpMV routines using the JAD matrix storage format (double precision matrix)";
            case SPMV_ELL_TEST:
                return "Performance test for SpMV routines using the ELLPACK matrix storage format (double precision matrix)";
            case SPMV_DIA_TEST:
                return "Performance test for SpMV routines using the DIA matrix storage format (double precision matrix)";
            case SPMV_MPMTX_CSR_TEST:
                return "Performance test for SpMV routines using the CSR matrix storage format (multiple precision matrix)";
            case SPMV_MPMTX_JAD_TEST:
                return "Performance test for SpMV routines using the JAD matrix storage format (multiple precision matrix)";
            case SPMV_MPMTX_ELL_TEST:
                return "Performance test for SpMV routines using the ELLPACK matrix storage format (multiple precision matrix)";
            case SPMV_MPMTX_DIA_TEST:
                return "Performance test for SpMV routines using the DIA matrix storage format (multiple precision matrix)";
            case SPMV_CSR_ACCURACY_TEST:
                return "Accuracy test for SpMV routines using the CSR matrix storage format (double precision matrix)";
            case SPMV_VS_GEMV_TEST:
                return "Performance test for SpMV vs GEMV";
            case CG_CSR_TEST:
                return "Test for CG and PCG iterative solvers using the CSR matrix storage format";
        }
        return "";
    }

    // Служебные методы печати
    static void printSysdate() {
        time_t t = time(NULL);
        struct tm *tm = localtime(&t);
        std::cout << "Date: " << asctime(tm);
    }

    void printDash() {
        std::cout << "---------------------------------------------------" << std::endl;
    }

    static void printDDash() {
        std::cout << "===================================================" << std::endl;
    }

    static void printStars() {
        std::cout << "***************************************************" << std::endl;
    }

    static void printSpace() {
        std::cout << std::endl;
    }

    static void printString(const char *string) {
        std::cout << string << std::endl;
    }

    void printParam(const char *param, const char *value) {
        std::cout << "- " << param << ": " << value << std::endl;
    }

    void printParam(const char *param, const std::string value) {
        std::cout << "- " << param << ": " << value << std::endl;
    }

    void printParam(const char *param, const int value) {
        std::cout << "- " << param << ": " << value << std::endl;
    }

    void printParam(const char *param, const long value) {
        std::cout << "- " << param << ": " << value << std::endl;
    }

    void printParam(const char *param, const double value) {
        std::cout << "- " << param << ": " << value << std::endl;
    }

    void printKernelExecutionConfig1D(unsigned int threads, unsigned int blocks){
        printf("\tExec. config: threads = %i, blocks = %i\n", threads, blocks);
    }

    void printKernelExecutionConfig2D(unsigned int threadsX, unsigned int threadsY, unsigned int blocksX, unsigned int blocksY){
        printf("\tExec. config: threads.x = %i, threads.y = %i, blocks.x = %i, blocks.y = %i\n", threadsX, threadsY, blocksX, blocksY);
    }

    void beginTestDescription(TestHeader header) {
        printSpace();
        printDDash();
        std::cout << testHeaderAsString(header) << std::endl;
        printSpace();
        printSysdate();
        printDDash();
    }

    void endTestDescription() {
        printDDash();
        printSpace();
        printSpace();
    }

    void printTestParameters(
            long operationSize,
            int numberOfRepeats,
            int precisionInBits,
            int precisionInDecs) {
        printString("Parameters:");
        printParam("Operation size", operationSize);
        printParam("Number of repeats", numberOfRepeats);
        printParam("Precision (in bits)", precisionInBits);
        printParam("Precision (in decimals)", precisionInDecs);
        printDash();
    }

    void beginSection(const char *sectionName) {
        printString(sectionName);
    }

    void endSection(bool lastBeforeResults) {
        if (lastBeforeResults) {
            printDDash();
            printSpace();
        } else {
            printDash();
        }
    }
} // end of namespace Logger


#endif //MPRES_TEST_LOGGER_CUH