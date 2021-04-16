/*
 *  Performance test for the TACO library SpMV routine (double precision matrix and vectors)
 *  https://github.com/tensor-compiler/taco
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

#ifndef TEST_TACO_SPMV_CSR_CUH
#define TEST_TACO_SPMV_CSR_CUH

#include "../../../../tsthelper.cuh"
#include "../../../../logger.cuh"
#include "../../../../timers.cuh"
#include "../../../../../../../../../../usr/local/include/taco.h"

void test_taco_spmv_csr(const char * matrix_path, const mpfr_t * vectorX, const string matrix_type){
    if (matrix_type == "real") {
        using namespace taco;
        InitCpuTimer();
        Logger::printDash();
        PrintTimerName("[CPU] TACO SpMV CSR");

        Format csr({Dense,Sparse});
        Format  dv({Dense});

        Tensor<double> A = read(matrix_path, csr, false);

        Tensor<double> x({A.getDimension(1)}, dv);
        for (int i = 0; i < x.getDimension(0); ++i) {
            x.insert({i}, mpfr_get_d(vectorX[i], MPFR_RNDN));
        }
        x.pack();
        Tensor<double> result({A.getDimension(0)}, dv);

        IndexVar i, j;
        result(i) = (A(i,j) * x(j));

        StartCpuTimer();
        result.compile();
        result.assemble();
        result.compute();
        EndCpuTimer();

        double sum = 0.0;
        for (int k = 0; k < result.getDimension(0); k++) {
            sum += result(k);
        }
        PrintCpuTimer("took");
        printf("result: %.70f\n", sum);
    }
}

#endif //TEST_TACO_SPMV_CSR_CUH
