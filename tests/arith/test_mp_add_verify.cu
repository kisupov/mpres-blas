/*
 *  Test for validating the mp_add routine
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


#include "../../src/mpfloat.cuh"

int main() {
    rns_const_init();
    mp_const_init();
    rns_const_print(true);
    rns_eval_const_print();

    mp_float_t x, y, res;
    mp_set_d(&x, 0.001);
    mp_set_d(&y, -0.0333);
    mp_add(&res, &x, &y);
    printf("\nResult = %lf", mp_get_d(&res));
    mp_print(&res);

    return 0;
}