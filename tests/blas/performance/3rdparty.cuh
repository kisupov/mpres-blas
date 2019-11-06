/*
 *  Inclusions for testing third-party multiple-precision software
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

#ifndef MPRES_TEST_3RDPARTY_CUH
#define MPRES_TEST_3RDPARTY_CUH

//MPFR - https://mpfr.org
#include "mpfr.h"
//OpenBLAS - https://www.openblas.net/
#include "cblas.h"
//XBLAS - https://www.netlib.org/xblas/
#include "../../3rdparty/xblas/inc/blas_extended.h"
//ARPREC - https://www.davidhbailey.com/dhbsoftware/
#include "arprec/mp_real.h"
//MPACK - http://mplapack.sourceforge.net/
#include "mpack/mpreal.h"
#include "mpack/mblas_mpfr.h"
//libmpdec - https://www.bytereef.org/mpdecimal/
#include "mpdecimal.h"
//CUMP - https://github.com/skystar0227/CUMP
#include "../lib/cump_blas.cuh"
//GARPREC - https://code.google.com/archive/p/gpuprec/downloads
#include "../lib/garprec_blas.cuh"
//CAMPARY - http://homepages.laas.fr/mmjoldes/campary/
#include "../lib/campary_blas.cuh"
//cuBLAS - https://developer.nvidia.com/cublas
#include "cublas_v2.h"

#endif //MPRES_TEST_3RDPARTY_CUH
