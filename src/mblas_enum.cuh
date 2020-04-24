/*
 *  Enumerated types (compliant with XBLAS).
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

#ifndef MPRES_BLAS_ENUM_CUH
#define MPRES_BLAS_ENUM_CUH

enum mblas_trans_type {
    mblas_no_trans   = 111,
    mblas_trans      = 112,
    mblas_conj_trans = 113
};

enum mblas_side_type {
    mblas_left_side  = 141,
    mblas_right_side = 142
};



#endif //MPRES_BLAS_ENUM_CUH
