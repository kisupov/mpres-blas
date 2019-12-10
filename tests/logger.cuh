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
        BLAS_ROT_PERFORMANCE_TEST,
        BLAS_ASUM_ACCURACY_TEST,
        BLAS_DOT_ACCURACY_TEST,
        BLAS_SCAL_ACCURACY_TEST,
        BLAS_AXPY_ACCURACY_TEST,
        RNS_EVAL_ACCURACY_TEST
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
            case BLAS_ROT_PERFORMANCE_TEST:
                return "Performance test for BLAS ROT routines";
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

    void printParam(const char *param, const int value) {
        std::cout << "- " << param << ": " << value << std::endl;
    }

    void printParam(const char *param, const long value) {
        std::cout << "- " << param << ": " << value << std::endl;
    }

    void printParam(const char *param, const double value) {
        std::cout << "- " << param << ": " << value << std::endl;
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