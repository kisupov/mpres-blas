/*
 *  Optimized routines for the pairwise summation of sets of floating-point numbers.
 *  Two directed rounding modes are also supported, namely rounding downward and rounding upward.
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


#ifndef MPRES_PAIRWISE_CUH
#define MPRES_PAIRWISE_CUH

#include "common.cuh"
#include "dinterval.cuh"

/*
 * Below are the internal CPU routines for pairwise summation
 */

/********************* Current rounding mode *********************/

//Serial pairwise summation of an array of SIZE floating-point numbers, for 1 <= SIZE <= 2
template<int LENGTH>
GCC_FORCEINLINE static double psum2(const double *x) {
    double s2 = (LENGTH > 1) ? x[1] : 0;
    return x[0] + s2;
}

//Serial pairwise summation of an array of SIZE floating-point numbers, for 1 <= SIZE <= 4
template<int LENGTH>
GCC_FORCEINLINE static double psum4(const double *x) {
    double s1 = psum2<LENGTH>(x);
    double s2 = (LENGTH > 2) ? psum2<LENGTH - 2>(&x[2]) : 0;
    return s1 + s2;
}

//Serial pairwise summation of an array of SIZE floating-point numbers, for 1 <= SIZE <= 8
template<int LENGTH>
GCC_FORCEINLINE static double psum8(const double *x) {
    double s1 = psum4<LENGTH>(x);
    double s2 = (LENGTH > 4) ? psum4<LENGTH - 4>(&x[4]) : 0;
    return s1 + s2;
}

//Serial pairwise summation of an array of SIZE floating-point numbers, for 1 <= SIZE <= 16
template<int LENGTH>
GCC_FORCEINLINE static double psum16(const double *x) {
    double s1 = psum8<LENGTH>(x);
    double s2 = (LENGTH > 8) ? psum8<LENGTH - 8>(&x[8]) : 0;
    return s1 + s2;
}

//Serial pairwise summation of an array of SIZE floating-point numbers, for 1 <= SIZE <= 32
template<int LENGTH>
GCC_FORCEINLINE static double psum32(const double *x) {
    double s1 = psum16<LENGTH>(x);
    double s2 = (LENGTH > 16) ? psum16<LENGTH - 16>(&x[16]) : 0;
    return s1 + s2;
}

//Serial pairwise summation of an array of SIZE floating-point numbers, for 1 <= SIZE <= 64
template<int LENGTH>
GCC_FORCEINLINE static double psum64(const double *x) {
    double s1 = psum32<LENGTH>(x);
    double s2 = (LENGTH > 32) ? psum32<LENGTH - 32>(&x[32]) : 0;
    return s1 + s2;
}

//Serial pairwise summation of an array of SIZE floating-point numbers, for 1 <= SIZE <= 128
template<int LENGTH>
GCC_FORCEINLINE static double psum128(const double *x) {
    double s1 = psum64<LENGTH>(x);
    double s2 = (LENGTH > 64) ? psum64<LENGTH - 64>(&x[64]) : 0;
    return s1 + s2;
}

//Serial pairwise summation of an array of SIZE floating-point numbers, for 1 <= SIZE <= 256
template<int LENGTH>
GCC_FORCEINLINE static double psum256(const double *x) {
    double s1 = psum128<LENGTH>(x);
    double s2 = (LENGTH > 128) ? psum128<LENGTH - 128>(&x[128]) : 0;
    return s1 + s2;
}

//Serial pairwise summation of an array of SIZE floating-point numbers, for 1 <= SIZE <= 512
template<int LENGTH>
GCC_FORCEINLINE static double psum512(const double *x) {
    double s1 = psum256<LENGTH>(x);
    double s2 = (LENGTH > 256) ? psum256<LENGTH - 256>(&x[256]) : 0;
    return s1 + s2;
}

//Serial pairwise summation of an array of SIZE floating-point numbers, for 1 <= SIZE <= 1024
template<int LENGTH>
GCC_FORCEINLINE static double psum1024(const double *x) {
    double s1 = psum512<LENGTH>(x);
    double s2 = (LENGTH > 512) ? psum512<LENGTH - 512>(&x[512]) : 0;
    return s1 + s2;
}

//Serial pairwise summation of an array of SIZE floating-point numbers, for 1 <= SIZE <= 2048
template<int LENGTH>
GCC_FORCEINLINE static double psum2048(const double *x) {
    double s1 = psum1024<LENGTH>(x);
    double s2 = (LENGTH > 1024) ? psum1024<LENGTH - 1024>(&x[1024]) : 0;
    return s1 + s2;
}

//Serial pairwise summation of an array of SIZE floating-point numbers, for 1 <= SIZE <= 4096
template<int LENGTH>
GCC_FORCEINLINE static double psum4096(const double *x) {
    double s1 = psum2048<LENGTH>(x);
    double s2 = (LENGTH > 2048) ? psum2048<LENGTH - 2048>(&x[2048]) : 0;
    return s1 + s2;
}

//Serial pairwise summation of an array of SIZE floating-point numbers, for 1 <= SIZE <= 8192
template<int LENGTH>
GCC_FORCEINLINE static double psum8192(const double *x) {
    double s1 = psum4096<LENGTH>(x);
    double s2 = (LENGTH > 4096) ? psum4096<LENGTH - 4096>(&x[4096]) : 0;
    return s1 + s2;
}

/********************* Rounding upwards *********************/

//Serial round-up pairwise summation of an array of SIZE floating-point numbers, for 1 <= SIZE <= 2
template<int LENGTH>
GCC_FORCEINLINE static double psum2_ru(const double *x) {
    return dadd_ru(x[0], (LENGTH > 1) ? x[1] : 0);
}

//Serial round-up pairwise summation of an array of SIZE floating-point numbers, for 1 <= SIZE <= 4
template<int LENGTH>
GCC_FORCEINLINE static double psum4_ru(const double *x) {
    double s1 = psum2_ru<LENGTH>(x);
    double s2 = (LENGTH > 2) ? psum2_ru<LENGTH - 2>(&x[2]) : 0;
    return dadd_ru(s1, s2);
}

//Serial round-up pairwise summation of an array of SIZE floating-point numbers, for 1 <= SIZE <= 8
template<int LENGTH>
GCC_FORCEINLINE static double psum8_ru(const double *x) {
    double s1 = psum4_ru<LENGTH>(x);
    double s2 = (LENGTH > 4) ? psum4_ru<LENGTH - 4>(&x[4]) : 0;
    return dadd_ru(s1, s2);
}

//Serial round-up pairwise summation of an array of SIZE floating-point numbers, for 1 <= SIZE <= 16
template<int LENGTH>
GCC_FORCEINLINE static double psum16_ru(const double *x) {
    double s1 = psum8_ru<LENGTH>(x);
    double s2 = (LENGTH > 8) ? psum8_ru<LENGTH - 8>(&x[8]) : 0;
    return dadd_ru(s1, s2);
}

//Serial round-up pairwise summation of an array of SIZE floating-point numbers, for 1 <= SIZE <= 32
template<int LENGTH>
GCC_FORCEINLINE static double psum32_ru(const double *x) {
    double s1 = psum16_ru<LENGTH>(x);
    double s2 = (LENGTH > 16) ? psum16_ru<LENGTH - 16>(&x[16]) : 0;
    return dadd_ru(s1, s2);
}

//Serial round-up pairwise summation of an array of SIZE floating-point numbers, for 1 <= SIZE <= 64
template<int LENGTH>
GCC_FORCEINLINE static double psum64_ru(const double *x) {
    double s1 = psum32_ru<LENGTH>(x);
    double s2 = (LENGTH > 32) ? psum32_ru<LENGTH - 32>(&x[32]) : 0;
    return dadd_ru(s1, s2);
}

//Serial round-up pairwise summation of an array of SIZE floating-point numbers, for 1 <= SIZE <= 128
template<int LENGTH>
GCC_FORCEINLINE static double psum128_ru(const double *x) {
    double s1 = psum64_ru<LENGTH>(x);
    double s2 = (LENGTH > 64) ? psum64_ru<LENGTH - 64>(&x[64]) : 0;
    return dadd_ru(s1, s2);
}

//Serial round-up pairwise summation of an array of SIZE floating-point numbers, for 1 <= SIZE <= 256
template<int LENGTH>
GCC_FORCEINLINE static double psum256_ru(const double *x) {
    double s1 = psum128_ru<LENGTH>(x);
    double s2 = (LENGTH > 128) ? psum128_ru<LENGTH - 128>(&x[128]) : 0;
    return dadd_ru(s1, s2);
}

//Serial round-up pairwise summation of an array of SIZE floating-point numbers, for 1 <= SIZE <= 512
template<int LENGTH>
GCC_FORCEINLINE static double psum512_ru(const double *x) {
    double s1 = psum256_ru<LENGTH>(x);
    double s2 = (LENGTH > 256) ? psum256_ru<LENGTH - 256>(&x[256]) : 0;
    return dadd_ru(s1, s2);
}

//Serial round-up pairwise summation of an array of SIZE floating-point numbers, for 1 <= SIZE <= 1024
template<int LENGTH>
GCC_FORCEINLINE static double psum1024_ru(const double *x) {
    double s1 = psum512_ru<LENGTH>(x);
    double s2 = (LENGTH > 512) ? psum512_ru<LENGTH - 512>(&x[512]) : 0;
    return dadd_ru(s1, s2);
}

//Serial round-up pairwise summation of an array of SIZE floating-point numbers, for 1 <= SIZE <= 2048
template<int LENGTH>
GCC_FORCEINLINE static double psum2048_ru(const double *x) {
    double s1 = psum1024_ru<LENGTH>(x);
    double s2 = (LENGTH > 1024) ? psum1024_ru<LENGTH - 1024>(&x[1024]) : 0;
    return dadd_ru(s1, s2);
}

//Serial round-up pairwise summation of an array of SIZE floating-point numbers, for 1 <= SIZE <= 4096
template<int LENGTH>
GCC_FORCEINLINE static double psum4096_ru(const double *x) {
    double s1 = psum2048_ru<LENGTH>(x);
    double s2 = (LENGTH > 2048) ? psum2048_ru<LENGTH - 2048>(&x[2048]) : 0;
    return dadd_ru(s1, s2);
}

//Serial round-up pairwise summation of an array of SIZE floating-point numbers, for 1 <= SIZE <= 8192
template<int LENGTH>
GCC_FORCEINLINE static double psum8192_ru(const double *x) {
    double s1 = psum4096_ru<LENGTH>(x);
    double s2 = (LENGTH > 4096) ? psum4096_ru<LENGTH - 4096>(&x[4096]) : 0;
    return dadd_ru(s1, s2);
}

/********************* Rounding downwards *********************/

//Serial round-down pairwise summation of an array of SIZE floating-point numbers, for 1 <= SIZE <= 2
template<int LENGTH>
GCC_FORCEINLINE static double psum2_rd(const double *x) {
    return dadd_rd(x[0], (LENGTH > 1) ? x[1] : 0);
}

//Serial round-down pairwise summation of an array of SIZE floating-point numbers, for 1 <= SIZE <= 4
template<int LENGTH>
GCC_FORCEINLINE static double psum4_rd(const double *x) {
    double s1 = psum2_rd<LENGTH>(x);
    double s2 = (LENGTH > 2) ? psum2_rd<LENGTH - 2>(&x[2]) : 0;
    return dadd_rd(s1, s2);
}

//Serial round-down pairwise summation of an array of SIZE floating-point numbers, for 1 <= SIZE <= 8
template<int LENGTH>
GCC_FORCEINLINE static double psum8_rd(const double *x) {
    double s1 = psum4_rd<LENGTH>(x);
    double s2 = (LENGTH > 4) ? psum4_rd<LENGTH - 4>(&x[4]) : 0;
    return dadd_rd(s1, s2);
}

//Serial round-down pairwise summation of an array of SIZE floating-point numbers, for 1 <= SIZE <= 16
template<int LENGTH>
GCC_FORCEINLINE static double psum16_rd(const double *x) {
    double s1 = psum8_rd<LENGTH>(x);
    double s2 = (LENGTH > 8) ? psum8_rd<LENGTH - 8>(&x[8]) : 0;
    return dadd_rd(s1, s2);
}

//Serial round-down pairwise summation of an array of SIZE floating-point numbers, for 1 <= SIZE <= 32
template<int LENGTH>
GCC_FORCEINLINE static double psum32_rd(const double *x) {
    double s1 = psum16_rd<LENGTH>(x);
    double s2 = (LENGTH > 16) ? psum16_rd<LENGTH - 16>(&x[16]) : 0;
    return dadd_rd(s1, s2);
}

//Serial round-down pairwise summation of an array of SIZE floating-point numbers, for 1 <= SIZE <= 64
template<int LENGTH>
GCC_FORCEINLINE static double psum64_rd(const double *x) {
    double s1 = psum32_rd<LENGTH>(x);
    double s2 = (LENGTH > 32) ? psum32_rd<LENGTH - 32>(&x[32]) : 0;
    return dadd_rd(s1, s2);
}

//Serial round-down pairwise summation of an array of SIZE floating-point numbers, for 1 <= SIZE <= 128
template<int LENGTH>
GCC_FORCEINLINE static double psum128_rd(const double *x) {
    double s1 = psum64_rd<LENGTH>(x);
    double s2 = (LENGTH > 64) ? psum64_rd<LENGTH - 64>(&x[64]) : 0;
    return dadd_rd(s1, s2);
}

//Serial round-down pairwise summation of an array of SIZE floating-point numbers, for 1 <= SIZE <= 256
template<int LENGTH>
GCC_FORCEINLINE static double psum256_rd(const double *x) {
    double s1 = psum128_rd<LENGTH>(x);
    double s2 = (LENGTH > 128) ? psum128_rd<LENGTH - 128>(&x[128]) : 0;
    return dadd_rd(s1, s2);
}

//Serial round-down pairwise summation of an array of SIZE floating-point numbers, for 1 <= SIZE <= 512
template<int LENGTH>
GCC_FORCEINLINE static double psum512_rd(const double *x) {
    double s1 = psum256_rd<LENGTH>(x);
    double s2 = (LENGTH > 256) ? psum256_rd<LENGTH - 256>(&x[256]) : 0;
    return dadd_rd(s1, s2);
}

//Serial round-down pairwise summation of an array of SIZE floating-point numbers, for 1 <= SIZE <= 1024
template<int LENGTH>
GCC_FORCEINLINE static double psum1024_rd(const double *x) {
    double s1 = psum512_rd<LENGTH>(x);
    double s2 = (LENGTH > 512) ? psum512_rd<LENGTH - 512>(&x[512]) : 0;
    return dadd_rd(s1, s2);
}

//Serial round-down pairwise summation of an array of SIZE floating-point numbers, for 1 <= SIZE <= 2048
template<int LENGTH>
GCC_FORCEINLINE static double psum2048_rd(const double *x) {
    double s1 = psum1024_rd<LENGTH>(x);
    double s2 = (LENGTH > 1024) ? psum1024_rd<LENGTH - 1024>(&x[1024]) : 0;
    return dadd_rd(s1, s2);
}

//Serial round-down pairwise summation of an array of SIZE floating-point numbers, for 1 <= SIZE <= 4096
template<int LENGTH>
GCC_FORCEINLINE static double psum4096_rd(const double *x) {
    double s1 = psum2048_rd<LENGTH>(x);
    double s2 = (LENGTH > 2048) ? psum2048_rd<LENGTH - 2048>(&x[2048]) : 0;
    return dadd_rd(s1, s2);
}

//Serial round-down pairwise summation of an array of SIZE floating-point numbers, for 1 <= SIZE <= 8192
template<int LENGTH>
GCC_FORCEINLINE static double psum8192_rd(const double *x) {
    double s1 = psum4096_rd<LENGTH>(x);
    double s2 = (LENGTH > 4096) ? psum4096_rd<LENGTH - 4096>(&x[4096]) : 0;
    return dadd_rd(s1, s2);
}


/*
 * Below are the global CPU routines for pairwise summation
 */

/*!
 * Fully unrolled CPU-tuned pairwise summation of an array of SIZE floating-point numbers in a current rounding mode.
 * We perform additions with zeros if SIZE is not a power of two.
 * Array size restrictions: 0 <= SIZE <= 8192
 * @param x - array of SIZE floating-point numbers
 * @return - sum of array elements
 */
template<int LENGTH>
GCC_FORCEINLINE double psum(const double *x) {
    if (LENGTH >= 4096) {               //For 4096 ... 8192
        return psum8192<LENGTH>(x);
    } else if (LENGTH >= 2048) {        //For 2048 ... 4095
        return psum4096<LENGTH>(x);
    } else if (LENGTH >= 1024) {        //For 1024 ... 2047
        return psum2048<LENGTH>(x);
    } else if (LENGTH >= 512) {         //For 512 ... 1023
        return psum1024<LENGTH>(x);
    } else if (LENGTH >= 256) {         //For 256 ... 511
        return psum512<LENGTH>(x);
    } else if (LENGTH >= 128) {         //For 128 ... 255
        return psum256<LENGTH>(x);
    } else if (LENGTH >= 64) {          //For 64 ... 127
        return psum128<LENGTH>(x);
    } else if (LENGTH >= 32) {          //For 32 ... 63
        return psum64<LENGTH>(x);
    } else if (LENGTH >= 16) {          //For 16 ... 31
        return psum32<LENGTH>(x);
    } else if (LENGTH >= 8) {           //For 8 ... 15
        return psum16<LENGTH>(x);
    } else if (LENGTH >= 4) {           //For 4 ... 7
        return psum8<LENGTH>(x);
    } else if (LENGTH == 3) {           //For 3
        return x[0] + x[1] + x[2];
    } else if (LENGTH == 2) {           //For 2
        return x[0] + x[1];
    } else if (LENGTH == 1) {           //For 1
        return x[0];
    } else {
        return 0;
    }
}

/*!
 * Fully unrolled CPU-tuned round-up pairwise summation of an array of SIZE floating-point numbers.
 * We perform additions with zeros if SIZE is not a power of two.
 * Array size restrictions: 0 <= SIZE <= 8192
 * @param x - array of SIZE floating-point numbers
 * @return - sum of array elements
 */
template<int LENGTH>
GCC_FORCEINLINE double psum_ru(const double *x) {
    if (LENGTH >= 4096) {               //For 4096 ... 8192
        return psum8192_ru<LENGTH>(x);
    } else if (LENGTH >= 2048) {        //For 2048 ... 4095
        return psum4096_ru<LENGTH>(x);
    } else if (LENGTH >= 1024) {        //For 1024 ... 2047
        return psum2048_ru<LENGTH>(x);
    } else if (LENGTH >= 512) {         //For 512 ... 1023
        return psum1024_ru<LENGTH>(x);
    } else if (LENGTH >= 256) {         //For 256 ... 511
        return psum512_ru<LENGTH>(x);
    } else if (LENGTH >= 128) {         //For 128 ... 255
        return psum256_ru<LENGTH>(x);
    } else if (LENGTH >= 64) {          //For 64 ... 127
        return psum128_ru<LENGTH>(x);
    } else if (LENGTH >= 32) {          //For 32 ... 63
        return psum64_ru<LENGTH>(x);
    } else if (LENGTH >= 16) {          //For 16 ... 31
        return psum32_ru<LENGTH>(x);
    } else if (LENGTH >= 8) {           //For 8 ... 15
        return psum16_ru<LENGTH>(x);
    } else if (LENGTH >= 4) {           //For 4 ... 7
        return psum8_ru<LENGTH>(x);
    } else if (LENGTH == 3) {           //For 3
        return dadd_ru(dadd_ru(x[0], x[1]), x[2]);
    } else if (LENGTH == 2) {           //For 2
        return dadd_ru(x[0], x[1]);
    } else if (LENGTH == 1) {           //For 1
        return x[0];
    } else {
        return 0;
    }
}

/*!
 * Fully unrolled CPU-tuned round-down pairwise summation of an array of SIZE floating-point numbers.
 * We perform additions with zeros if SIZE is not a power of two.
 * Array size restrictions: 0 <= SIZE <= 8192
 * @param x - array of SIZE floating-point numbers
 * @return - sum of array elements
 */
template<int LENGTH>
GCC_FORCEINLINE double psum_rd(const double *x) {
    if (LENGTH >= 4096) {               //For 4096 ... 8192
        return psum8192_rd<LENGTH>(x);
    } else if (LENGTH >= 2048) {        //For 2048 ... 4095
        return psum4096_rd<LENGTH>(x);
    } else if (LENGTH >= 1024) {        //For 1024 ... 2047
        return psum2048_rd<LENGTH>(x);
    } else if (LENGTH >= 512) {         //For 512 ... 1023
        return psum1024_rd<LENGTH>(x);
    } else if (LENGTH >= 256) {         //For 256 ... 511
        return psum512_rd<LENGTH>(x);
    } else if (LENGTH >= 128) {         //For 128 ... 255
        return psum256_rd<LENGTH>(x);
    } else if (LENGTH >= 64) {          //For 64 ... 127
        return psum128_rd<LENGTH>(x);
    } else if (LENGTH >= 32) {          //For 32 ... 63
        return psum64_rd<LENGTH>(x);
    } else if (LENGTH >= 16) {          //For 16 ... 31
        return psum32_rd<LENGTH>(x);
    } else if (LENGTH >= 8) {           //For 8 ... 15
        return psum16_rd<LENGTH>(x);
    } else if (LENGTH >= 4) {           //For 4 ... 7
        return psum8_rd<LENGTH>(x);
    } else if (LENGTH == 3) {           //For 3
        return dadd_rd(dadd_rd(x[0], x[1]), x[2]);
    } else if (LENGTH == 2) {           //For 2
        return dadd_rd(x[0], x[1]);
    } else if (LENGTH == 1) {           //For 1
        return x[0];
    } else {
        return 0;
    }
}

/*
 * GPU functions
 */
namespace cuda {

    /*
     * Below are the internal CUDA routines for pairwise summation
     */

    /********************* Current rounding mode *********************/

    //Serial pairwise summation of an array of SIZE floating-point numbers, for 1 <= SIZE <= 2
    template<int LENGTH>
    DEVICE_CUDA_FORCEINLINE static double psum2(const double *x) {
        double s2 = (LENGTH > 1) ? x[1] : 0;
        return x[0] + s2;
    }

    //Serial pairwise summation of an array of SIZE floating-point numbers, for 1 <= SIZE <= 4
    template<int LENGTH>
    DEVICE_CUDA_FORCEINLINE static double psum4(const double *x) {
        double s1 = cuda::psum2<LENGTH>(x);
        double s2 = (LENGTH > 2) ? cuda::psum2<LENGTH - 2>(&x[2]) : 0;
        return s1 + s2;
    }

    //Serial pairwise summation of an array of SIZE floating-point numbers, for 1 <= SIZE <= 8
    template<int LENGTH>
    DEVICE_CUDA_FORCEINLINE static double psum8(const double *x) {
        double s1 = cuda::psum4<LENGTH>(x);
        double s2 = (LENGTH > 4) ? cuda::psum4<LENGTH - 4>(&x[4]) : 0;
        return s1 + s2;
    }

    //Serial pairwise summation of an array of SIZE floating-point numbers, for 1 <= SIZE <= 16
    template<int LENGTH>
    DEVICE_CUDA_FORCEINLINE static double psum16(const double *x) {
        double s1 = cuda::psum8<LENGTH>(x);
        double s2 = (LENGTH > 8) ? cuda::psum8<LENGTH - 8>(&x[8]) : 0;
        return s1 + s2;
    }

    //Serial pairwise summation of an array of SIZE floating-point numbers, for 1 <= SIZE <= 32
    template<int LENGTH>
    DEVICE_CUDA_FORCEINLINE static double psum32(const double *x) {
        double s1 = cuda::psum16<LENGTH>(x);
        double s2 = (LENGTH > 16) ? cuda::psum16<LENGTH - 16>(&x[16]) : 0;
        return s1 + s2;
    }

    //Serial pairwise summation of an array of SIZE floating-point numbers, for 1 <= SIZE <= 64
    template<int LENGTH>
    DEVICE_CUDA_FORCEINLINE static double psum64(const double *x) {
        double s1 = cuda::psum32<LENGTH>(x);
        double s2 = (LENGTH > 32) ? cuda::psum32<LENGTH - 32>(&x[32]) : 0;
        return s1 + s2;
    }

    //Serial pairwise summation of an array of SIZE floating-point numbers, for 1 <= SIZE <= 128
    template<int LENGTH>
    DEVICE_CUDA_FORCEINLINE static double psum128(const double *x) {
        double s1 = cuda::psum64<LENGTH>(x);
        double s2 = (LENGTH > 64) ? cuda::psum64<LENGTH - 64>(&x[64]) : 0;
        return s1 + s2;
    }

    //Serial pairwise summation of an array of SIZE floating-point numbers, for 1 <= SIZE <= 256
    template<int LENGTH>
    DEVICE_CUDA_FORCEINLINE static double psum256(const double *x) {
        double s1 = cuda::psum128<LENGTH>(x);
        double s2 = (LENGTH > 128) ? cuda::psum128<LENGTH - 128>(&x[128]) : 0;
        return s1 + s2;
    }

    //Serial pairwise summation of an array of SIZE floating-point numbers, for 1 <= SIZE <= 512
    template<int LENGTH>
    DEVICE_CUDA_FORCEINLINE static double psum512(const double *x) {
        double s1 = cuda::psum256<LENGTH>(x);
        double s2 = (LENGTH > 256) ? cuda::psum256<LENGTH - 256>(&x[256]) : 0;
        return s1 + s2;
    }

    //Serial pairwise summation of an array of SIZE floating-point numbers, for 1 <= SIZE <= 1024
    template<int LENGTH>
    DEVICE_CUDA_FORCEINLINE static double psum1024(const double *x) {
        double s1 = cuda::psum512<LENGTH>(x);
        double s2 = (LENGTH > 512) ? cuda::psum512<LENGTH - 512>(&x[512]) : 0;
        return s1 + s2;
    }

    //Serial pairwise summation of an array of SIZE floating-point numbers, for 1 <= SIZE <= 2048
    template<int LENGTH>
    DEVICE_CUDA_FORCEINLINE static double psum2048(const double *x) {
        double s1 = cuda::psum1024<LENGTH>(x);
        double s2 = (LENGTH > 1024) ? cuda::psum1024<LENGTH - 1024>(&x[1024]) : 0;
        return s1 + s2;
    }

    //Serial pairwise summation of an array of SIZE floating-point numbers, for 1 <= SIZE <= 4096
    template<int LENGTH>
    DEVICE_CUDA_FORCEINLINE static double psum4096(const double *x) {
        double s1 = cuda::psum2048<LENGTH>(x);
        double s2 = (LENGTH > 2048) ? cuda::psum2048<LENGTH - 2048>(&x[2048]) : 0;
        return s1 + s2;
    }

    //Serial pairwise summation of an array of SIZE floating-point numbers, for 1 <= SIZE <= 8192
    template<int LENGTH>
    DEVICE_CUDA_FORCEINLINE static double psum8192(const double *x) {
        double s1 = cuda::psum4096<LENGTH>(x);
        double s2 = (LENGTH > 4096) ? cuda::psum4096<LENGTH - 4096>(&x[4096]) : 0;
        return s1 + s2;
    }

    /********************* Rounding upwards *********************/

    //Serial round-up pairwise summation of an array of SIZE floating-point numbers, for 1 <= SIZE <= 2
    template<int LENGTH>
    DEVICE_CUDA_FORCEINLINE static double psum2_ru(const double *x) {
        return __dadd_ru(x[0], (LENGTH > 1) ? x[1] : 0);
    }

    //Serial round-up pairwise summation of an array of SIZE floating-point numbers, for 1 <= SIZE <= 4
    template<int LENGTH>
    DEVICE_CUDA_FORCEINLINE static double psum4_ru(const double *x) {
        double s1 = cuda::psum2_ru<LENGTH>(x);
        double s2 = (LENGTH > 2) ? cuda::psum2_ru<LENGTH - 2>(&x[2]) : 0;
        return __dadd_ru(s1, s2);
    }

    //Serial round-up pairwise summation of an array of SIZE floating-point numbers, for 1 <= SIZE <= 8
    template<int LENGTH>
    DEVICE_CUDA_FORCEINLINE static double psum8_ru(const double *x) {
        double s1 = cuda::psum4_ru<LENGTH>(x);
        double s2 = (LENGTH > 4) ? cuda::psum4_ru<LENGTH - 4>(&x[4]) : 0;
        return __dadd_ru(s1, s2);
    }

    //Serial round-up pairwise summation of an array of SIZE floating-point numbers, for 1 <= SIZE <= 16
    template<int LENGTH>
    DEVICE_CUDA_FORCEINLINE static double psum16_ru(const double *x) {
        double s1 = cuda::psum8_ru<LENGTH>(x);
        double s2 = (LENGTH > 8) ? cuda::psum8_ru<LENGTH - 8>(&x[8]) : 0;
        return __dadd_ru(s1, s2);
    }

    //Serial round-up pairwise summation of an array of SIZE floating-point numbers, for 1 <= SIZE <= 32
    template<int LENGTH>
    DEVICE_CUDA_FORCEINLINE static double psum32_ru(const double *x) {
        double s1 = cuda::psum16_ru<LENGTH>(x);
        double s2 = (LENGTH > 16) ? cuda::psum16_ru<LENGTH - 16>(&x[16]) : 0;
        return __dadd_ru(s1, s2);
    }

    //Serial round-up pairwise summation of an array of SIZE floating-point numbers, for 1 <= SIZE <= 64
    template<int LENGTH>
    DEVICE_CUDA_FORCEINLINE static double psum64_ru(const double *x) {
        double s1 = cuda::psum32_ru<LENGTH>(x);
        double s2 = (LENGTH > 32) ? cuda::psum32_ru<LENGTH - 32>(&x[32]) : 0;
        return __dadd_ru(s1, s2);
    }

    //Serial round-up pairwise summation of an array of SIZE floating-point numbers, for 1 <= SIZE <= 128
    template<int LENGTH>
    DEVICE_CUDA_FORCEINLINE static double psum128_ru(const double *x) {
        double s1 = cuda::psum64_ru<LENGTH>(x);
        double s2 = (LENGTH > 64) ? cuda::psum64_ru<LENGTH - 64>(&x[64]) : 0;
        return __dadd_ru(s1, s2);
    }

    //Serial round-up pairwise summation of an array of SIZE floating-point numbers, for 1 <= SIZE <= 256
    template<int LENGTH>
    DEVICE_CUDA_FORCEINLINE static double psum256_ru(const double *x) {
        double s1 = cuda::psum128_ru<LENGTH>(x);
        double s2 = (LENGTH > 128) ? cuda::psum128_ru<LENGTH - 128>(&x[128]) : 0;
        return __dadd_ru(s1, s2);
    }

    //Serial round-up pairwise summation of an array of SIZE floating-point numbers, for 1 <= SIZE <= 512
    template<int LENGTH>
    DEVICE_CUDA_FORCEINLINE static double psum512_ru(const double *x) {
        double s1 = cuda::psum256_ru<LENGTH>(x);
        double s2 = (LENGTH > 256) ? cuda::psum256_ru<LENGTH - 256>(&x[256]) : 0;
        return __dadd_ru(s1, s2);
    }

    //Serial round-up pairwise summation of an array of SIZE floating-point numbers, for 1 <= SIZE <= 1024
    template<int LENGTH>
    DEVICE_CUDA_FORCEINLINE static double psum1024_ru(const double *x) {
        double s1 = cuda::psum512_ru<LENGTH>(x);
        double s2 = (LENGTH > 512) ? cuda::psum512_ru<LENGTH - 512>(&x[512]) : 0;
        return __dadd_ru(s1, s2);
    }

    //Serial round-up pairwise summation of an array of SIZE floating-point numbers, for 1 <= SIZE <= 2048
    template<int LENGTH>
    DEVICE_CUDA_FORCEINLINE static double psum2048_ru(const double *x) {
        double s1 = cuda::psum1024_ru<LENGTH>(x);
        double s2 = (LENGTH > 1024) ? cuda::psum1024_ru<LENGTH - 1024>(&x[1024]) : 0;
        return __dadd_ru(s1, s2);
    }

    //Serial round-up pairwise summation of an array of SIZE floating-point numbers, for 1 <= SIZE <= 4096
    template<int LENGTH>
    DEVICE_CUDA_FORCEINLINE static double psum4096_ru(const double *x) {
        double s1 = cuda::psum2048_ru<LENGTH>(x);
        double s2 = (LENGTH > 2048) ? cuda::psum2048_ru<LENGTH - 2048>(&x[2048]) : 0;
        return __dadd_ru(s1, s2);
    }

    //Serial round-up pairwise summation of an array of SIZE floating-point numbers, for 1 <= SIZE <= 8192
    template<int LENGTH>
    DEVICE_CUDA_FORCEINLINE static double psum8192_ru(const double *x) {
        double s1 = cuda::psum4096_ru<LENGTH>(x);
        double s2 = (LENGTH > 4096) ? cuda::psum4096_ru<LENGTH - 4096>(&x[4096]) : 0;
        return __dadd_ru(s1, s2);
    }

    /********************* Rounding downwards *********************/

    //Serial round-down pairwise summation of an array of SIZE floating-point numbers, for 1 <= SIZE <= 2
    template<int LENGTH>
    DEVICE_CUDA_FORCEINLINE static double psum2_rd(const double *x) {
        return __dadd_rd(x[0], (LENGTH > 1) ? x[1] : 0);
    }

    //Serial round-down pairwise summation of an array of SIZE floating-point numbers, for 1 <= SIZE <= 4
    template<int LENGTH>
    DEVICE_CUDA_FORCEINLINE static double psum4_rd(const double *x) {
        double s1 = cuda::psum2_rd<LENGTH>(x);
        double s2 = (LENGTH > 2) ? cuda::psum2_rd<LENGTH - 2>(&x[2]) : 0;
        return __dadd_rd(s1, s2);
    }

    //Serial round-down pairwise summation of an array of SIZE floating-point numbers, for 1 <= SIZE <= 8
    template<int LENGTH>
    DEVICE_CUDA_FORCEINLINE static double psum8_rd(const double *x) {
        double s1 = cuda::psum4_rd<LENGTH>(x);
        double s2 = (LENGTH > 4) ? cuda::psum4_rd<LENGTH - 4>(&x[4]) : 0;
        return __dadd_rd(s1, s2);
    }

    //Serial round-down pairwise summation of an array of SIZE floating-point numbers, for 1 <= SIZE <= 16
    template<int LENGTH>
    DEVICE_CUDA_FORCEINLINE static double psum16_rd(const double *x) {
        double s1 = cuda::psum8_rd<LENGTH>(x);
        double s2 = (LENGTH > 8) ? cuda::psum8_rd<LENGTH - 8>(&x[8]) : 0;
        return __dadd_rd(s1, s2);
    }

    //Serial round-down pairwise summation of an array of SIZE floating-point numbers, for 1 <= SIZE <= 32
    template<int LENGTH>
    DEVICE_CUDA_FORCEINLINE static double psum32_rd(const double *x) {
        double s1 = cuda::psum16_rd<LENGTH>(x);
        double s2 = (LENGTH > 16) ? cuda::psum16_rd<LENGTH - 16>(&x[16]) : 0;
        return __dadd_rd(s1, s2);
    }

    //Serial round-down pairwise summation of an array of SIZE floating-point numbers, for 1 <= SIZE <= 64
    template<int LENGTH>
    DEVICE_CUDA_FORCEINLINE static double psum64_rd(const double *x) {
        double s1 = cuda::psum32_rd<LENGTH>(x);
        double s2 = (LENGTH > 32) ? cuda::psum32_rd<LENGTH - 32>(&x[32]) : 0;
        return __dadd_rd(s1, s2);
    }

    //Serial round-down pairwise summation of an array of SIZE floating-point numbers, for 1 <= SIZE <= 128
    template<int LENGTH>
    DEVICE_CUDA_FORCEINLINE static double psum128_rd(const double *x) {
        double s1 = cuda::psum64_rd<LENGTH>(x);
        double s2 = (LENGTH > 64) ? cuda::psum64_rd<LENGTH - 64>(&x[64]) : 0;
        return __dadd_rd(s1, s2);
    }

    //Serial round-down pairwise summation of an array of SIZE floating-point numbers, for 1 <= SIZE <= 256
    template<int LENGTH>
    DEVICE_CUDA_FORCEINLINE static double psum256_rd(const double *x) {
        double s1 = cuda::psum128_rd<LENGTH>(x);
        double s2 = (LENGTH > 128) ? cuda::psum128_rd<LENGTH - 128>(&x[128]) : 0;
        return __dadd_rd(s1, s2);
    }

    //Serial round-down pairwise summation of an array of SIZE floating-point numbers, for 1 <= SIZE <= 512
    template<int LENGTH>
    DEVICE_CUDA_FORCEINLINE static double psum512_rd(const double *x) {
        double s1 = cuda::psum256_rd<LENGTH>(x);
        double s2 = (LENGTH > 256) ? cuda::psum256_rd<LENGTH - 256>(&x[256]) : 0;
        return __dadd_rd(s1, s2);
    }

    //Serial round-down pairwise summation of an array of SIZE floating-point numbers, for 1 <= SIZE <= 1024
    template<int LENGTH>
    DEVICE_CUDA_FORCEINLINE static double psum1024_rd(const double *x) {
        double s1 = cuda::psum512_rd<LENGTH>(x);
        double s2 = (LENGTH > 512) ? cuda::psum512_rd<LENGTH - 512>(&x[512]) : 0;
        return __dadd_rd(s1, s2);
    }

    //Serial round-down pairwise summation of an array of SIZE floating-point numbers, for 1 <= SIZE <= 2048
    template<int LENGTH>
    DEVICE_CUDA_FORCEINLINE static double psum2048_rd(const double *x) {
        double s1 = cuda::psum1024_rd<LENGTH>(x);
        double s2 = (LENGTH > 1024) ? cuda::psum1024_rd<LENGTH - 1024>(&x[1024]) : 0;
        return __dadd_rd(s1, s2);
    }

    //Serial round-down pairwise summation of an array of SIZE floating-point numbers, for 1 <= SIZE <= 4096
    template<int LENGTH>
    DEVICE_CUDA_FORCEINLINE static double psum4096_rd(const double *x) {
        double s1 = cuda::psum2048_rd<LENGTH>(x);
        double s2 = (LENGTH > 2048) ? cuda::psum2048_rd<LENGTH - 2048>(&x[2048]) : 0;
        return __dadd_rd(s1, s2);
    }

    //Serial round-down pairwise summation of an array of SIZE floating-point numbers, for 1 <= SIZE <= 8192
    template<int LENGTH>
    DEVICE_CUDA_FORCEINLINE static double psum8192_rd(const double *x) {
        double s1 = cuda::psum4096_rd<LENGTH>(x);
        double s2 = (LENGTH > 4096) ? cuda::psum4096_rd<LENGTH - 4096>(&x[4096]) : 0;
        return __dadd_rd(s1, s2);
    }


    /*
     * Below are the global CUDA routines for pairwise summation
     */

    /*!
     * Fully unrolled CUDA-tuned pairwise summation of an array of SIZE floating-point numbers in a current rounding mode.
     * We perform additions with zeros if SIZE is not a power of two.
     * Array size restrictions: 0 <= SIZE <= 8192
     * @param x - array of SIZE floating-point numbers
     * @return - sum of array elements
     */
    template<int LENGTH>
    DEVICE_CUDA_FORCEINLINE double psum(const double *x) {
        if (LENGTH >= 4096) {               //For 4096 ... 8192
            return cuda::psum8192<LENGTH>(x);
        } else if (LENGTH >= 2048) {        //For 2048 ... 4095
            return cuda::psum4096<LENGTH>(x);
        } else if (LENGTH >= 1024) {        //For 1024 ... 2047
            return cuda::psum2048<LENGTH>(x);
        } else if (LENGTH >= 512) {         //For 512 ... 1023
            return cuda::psum1024<LENGTH>(x);
        } else if (LENGTH >= 256) {         //For 256 ... 511
            return cuda::psum512<LENGTH>(x);
        } else if (LENGTH >= 128) {         //For 128 ... 255
            return cuda::psum256<LENGTH>(x);
        } else if (LENGTH >= 64) {          //For 64 ... 127
            return cuda::psum128<LENGTH>(x);
        } else if (LENGTH >= 32) {          //For 32 ... 63
            return cuda::psum64<LENGTH>(x);
        } else if (LENGTH >= 16) {          //For 16 ... 31
            return cuda::psum32<LENGTH>(x);
        } else if (LENGTH >= 8) {           //For 8 ... 15
            return cuda::psum16<LENGTH>(x);
        } else if (LENGTH >= 4) {           //For 4 ... 7
            return cuda::psum8<LENGTH>(x);
        } else if (LENGTH == 3) {           //For 3
            return x[0] + x[1] + x[2];
        } else if (LENGTH == 2) {           //For 2
            return x[0] + x[1];
        } else if (LENGTH == 1) {           //For 1
            return x[0];
        } else {
            return 0;
        }
    }

    /*!
     * Fully unrolled CUDA-tuned round-up pairwise summation of an array of SIZE floating-point numbers.
     * We perform additions with zeros if SIZE is not a power of two.
     * Array size restrictions: 0 <= SIZE <= 8192
     * @param x - array of SIZE floating-point numbers
     * @return - sum of array elements
     */
    template<int LENGTH>
    DEVICE_CUDA_FORCEINLINE double psum_ru(const double *x) {
        if (LENGTH >= 4096) {               //For 4096 ... 8192
            return cuda::psum8192_ru<LENGTH>(x);
        } else if (LENGTH >= 2048) {        //For 2048 ... 4095
            return cuda::psum4096_ru<LENGTH>(x);
        } else if (LENGTH >= 1024) {        //For 1024 ... 2047
            return cuda::psum2048_ru<LENGTH>(x);
        } else if (LENGTH >= 512) {         //For 512 ... 1023
            return cuda::psum1024_ru<LENGTH>(x);
        } else if (LENGTH >= 256) {         //For 256 ... 511
            return cuda::psum512_ru<LENGTH>(x);
        } else if (LENGTH >= 128) {         //For 128 ... 255
            return cuda::psum256_ru<LENGTH>(x);
        } else if (LENGTH >= 64) {          //For 64 ... 127
            return cuda::psum128_ru<LENGTH>(x);
        } else if (LENGTH >= 32) {          //For 32 ... 63
            return cuda::psum64_ru<LENGTH>(x);
        } else if (LENGTH >= 16) {          //For 16 ... 31
            return cuda::psum32_ru<LENGTH>(x);
        } else if (LENGTH >= 8) {           //For 8 ... 15
            return cuda::psum16_ru<LENGTH>(x);
        } else if (LENGTH >= 4) {           //For 4 ... 7
            return cuda::psum8_ru<LENGTH>(x);
        } else if (LENGTH == 3) {           //For 3
            return __dadd_ru(__dadd_ru(x[0], x[1]), x[2]);
        } else if (LENGTH == 2) {           //For 2
            return __dadd_ru(x[0], x[1]);
        } else if (LENGTH == 1) {           //For 1
            return x[0];
        } else {
            return 0;
        }
    }

    /*!
     * Fully unrolled CUDA-tuned round-down pairwise summation of an array of SIZE floating-point numbers.
     * We perform additions with zeros if SIZE is not a power of two.
     * Array size restrictions: 0 <= SIZE <= 8192
     * @param x - array of SIZE floating-point numbers
     * @return - sum of array elements
     */
    template<int LENGTH>
    DEVICE_CUDA_FORCEINLINE double psum_rd(const double *x) {
        if (LENGTH >= 4096) {               //For 4096 ... 8192
            return cuda::psum8192_rd<LENGTH>(x);
        } else if (LENGTH >= 2048) {        //For 2048 ... 4095
            return cuda::psum4096_rd<LENGTH>(x);
        } else if (LENGTH >= 1024) {        //For 1024 ... 2047
            return cuda::psum2048_rd<LENGTH>(x);
        } else if (LENGTH >= 512) {         //For 512 ... 1023
            return cuda::psum1024_rd<LENGTH>(x);
        } else if (LENGTH >= 256) {         //For 256 ... 511
            return cuda::psum512_rd<LENGTH>(x);
        } else if (LENGTH >= 128) {         //For 128 ... 255
            return cuda::psum256_rd<LENGTH>(x);
        } else if (LENGTH >= 64) {          //For 64 ... 127
            return cuda::psum128_rd<LENGTH>(x);
        } else if (LENGTH >= 32) {          //For 32 ... 63
            return cuda::psum64_rd<LENGTH>(x);
        } else if (LENGTH >= 16) {          //For 16 ... 31
            return cuda::psum32_rd<LENGTH>(x);
        } else if (LENGTH >= 8) {           //For 8 ... 15
            return cuda::psum16_rd<LENGTH>(x);
        } else if (LENGTH >= 4) {           //For 4 ... 7
            return cuda::psum8_rd<LENGTH>(x);
        } else if (LENGTH == 3) {           //For 3
            return __dadd_rd(__dadd_rd(x[0], x[1]), x[2]);
        } else if (LENGTH == 2) {           //For 2
            return __dadd_rd(x[0], x[1]);
        } else if (LENGTH == 1) {           //For 1
            return x[0];
        } else {
            return 0;
        }
    }

} //end of namespace

#endif //MPRES_PAIRWISE_CUH
