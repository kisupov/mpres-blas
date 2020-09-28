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
#ifndef EXCLUDE_MPFR
#include "mpfr.h"
#endif

//OpenBLAS - https://www.openblas.net/
#ifndef EXCLUDE_OPENBLAS
#include "cblas.h"
#endif

//ARPREC - https://www.davidhbailey.com/dhbsoftware/
#ifndef EXCLUDE_ARPREC
#include "arprec/mp_real.h"
#endif

//MPACK - http://mplapack.sourceforge.net/
#ifndef EXCLUDE_MPACK
#include "mpack/mpreal.h"
#include "mpack/mblas_mpfr.h"
#endif

//libmpdec - https://www.bytereef.org/mpdecimal/
#ifndef EXCLUDE_MPDECIMAL
#include "mpdecimal.h"
#endif

//CUMP - https://github.com/skystar0227/CUMP
#ifndef EXCLUDE_CUMP
#include "../lib/cump_sparse.cuh"
#endif

//CAMPARY - http://homepages.laas.fr/mmjoldes/campary/
#ifndef EXCLUDE_CAMPARY
#include "../lib/campary_sparse.cuh"
#endif

#endif //MPRES_TEST_3RDPARTY_CUH
