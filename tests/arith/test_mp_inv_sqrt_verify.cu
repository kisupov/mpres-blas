/*
 *  Test for validating the mp_inv_sqrt routines
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


#include "arith/invsqrt.cuh"
#include <cmath>

int main() {
    rns_const_init();
    mp_const_init();
    rns_const_print(true);
    rns_eval_const_print();

    mp_float_t x, y;
    mp_set_d(&x, 1e15);

    printf("\nARG X       = %.160f", mp_get_d(x));
    printf("\nREFERENCE   = %.160f", 1.0 / sqrt(mp_get_d(x)));
    mp_inv_sqrt(&y, &x);
    printf("\nCPU RESULT  = %.160f", mp_get_d(y));

    mp_float_ptr dx;
    mp_float_ptr dy;
    cudaMalloc(&dx, sizeof(mp_float_t));
    cudaMalloc(&dy, sizeof(mp_float_t));
    cudaMemcpy(dx, &x, sizeof(mp_float_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dy, &y, sizeof(mp_float_t), cudaMemcpyHostToDevice);
    cuda::mp_inv_sqrt(dy, dx);
    mp_set_d(&y, 0.0);
    cudaMemcpy(&y, dy, sizeof(mp_float_t), cudaMemcpyDeviceToHost);
    printf("\nCUDA RESULT = %.160f", mp_get_d(y));
    return 0;
}