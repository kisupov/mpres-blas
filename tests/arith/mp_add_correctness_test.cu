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
    rns_eval_const_calc();

    mp_float_t x, y, res;
    int x_r[] = {22652,6605,5584,10364,36578,11297,9551,15551};
    int y_r[] = {5370,1156,35997,27573,23361,2301,37162,28742};
    interval_t i_x, i_y;

    x.sign = 0;
    *x.digits = *x_r;
    x.exp = -55;
    y.eval[0].frac = 1.999278880646115919;
    y.eval[0].exp = -63;
    y.eval[1].frac =1.999278881030258415 ;
    y.eval[1].exp = -64;


    y.sign = 0;
    *y.digits = *y_r;
    y.exp = -31;
    y.eval[0].frac = 1.999278880646115919;
    y.eval[0].exp = -96;
    y.eval[1].frac =1.202608979215716190 ;
    y.eval[1].exp = -96;
    y.eval[0] = i_x.low;
    y.eval[1] = i_y.upp;

    mp_add(&res, &x, &y);

    mp_print(&res);

    return 0;
}