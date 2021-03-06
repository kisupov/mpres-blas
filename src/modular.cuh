/*
 *  Modular (modulo m) integer operations, as well as unrolled addition,
 *  subtraction and multiplication in the Residue Number System.
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

#ifndef MPRES_MODULAR_CUH
#define MPRES_MODULAR_CUH

#include "params.h"
#include "common.cuh"

/**
 * Constants that are computed  or copied to the device memory in rns.cuh
 */
namespace cuda {
    __device__  __constant__ int RNS_MODULI[RNS_MODULI_SIZE]; // The set of RNS moduli for GPU computing
}


/********************* Integer modulo m operations *********************/

/*!
 * Modulo m addition of x and y using the long data type
 * for intermediate result to avoid overflow.
 * In order to speedup computations, the modulo operation is replaced
 * by multiplication by d = 1 / m.
*/
GCC_FORCEINLINE int mod_add(const int x, const int y, const int m, const double d){
    long r = (long)x + (long)y;
    double quotient = (double) r * d;
    int i = (int) quotient;
    return (int) (r - (long) i * (long) m);
}

/*!
 * Modulo m subtraction of x and y using the long data type
 * for intermediate result to avoid overflow.
 * The subtraction result is not adjusted and may be negative
 */
GCC_FORCEINLINE int mod_sub(const int x, const int y, const int m){
    long r = (long)x - (long)y;
    r = r % (long)m;
    return (int) r;
}

/*!
 * Modulo m subtraction of x and y using the long data type
 * for intermediate result to avoid overflow.
 * Returns the adjusted (non-negative) result.
 */
GCC_FORCEINLINE int mod_psub(const int x, const int y, const int m){
    long r = ((long)x - (long)y + (long)m);
    r = r % (long)m;
    return (int) r;
}

/*!
 * Modulo m multiplication of x and y using the long data type
 * for intermediate result to avoid overflow.
 */
GCC_FORCEINLINE int mod_mul(const int x, const int y, const int m){
    long r = (long)x * (long)y;
    r = r % (long)m;
    return (int)r;
}

/*!
 * Modulo m multiplication of x and y using the long data type
 * for intermediate result to avoid overflow.
 * In order to speedup computations, the modulo operation is replaced
 * by multiplication by d = 1 / m.
*/
GCC_FORCEINLINE int mod_mul(const int x, const int y, const int m, const double d){
    long r = (long)x * (long)y;
    double quotient = (double) r * d;
    int i = (int) quotient;
    return (int) (r - (long) i * (long) m);
}

/*!
 * Modulo m addition of a * x and b * y using the long data type
 * for intermediate result to avoid overflow.
 * Returned result is (a * x + b * y) mod m
 */
GCC_FORCEINLINE int mod_axby(const int a, const int x, const int b, const int y, const int m){
    long r1 = (long)a * (long)x;
    long r2 = (long)b * (long)y;
    long r = (r1 + r2) % (long)m;
    return (int)r;
}

/*
 * GPU functions
 */
namespace cuda {

    /*!
     * Modulo m addition of x and y using the long data type
     * for intermediate result to avoid overflow.
     */
    DEVICE_CUDA_FORCEINLINE int mod_add(const int x, const int y, const int m){
        long r = (long)x + (long)y;
        r = r % (long)m;
        return (int)r;
    }

    /*!
     * Modulo m subtraction of x and y using the long data type
     * for intermediate result to avoid overflow.
     * The subtraction result is not adjusted and may be negative
     */
    DEVICE_CUDA_FORCEINLINE int mod_sub(const int x, const int y, const int m){
        long r = (long)x - (long)y;
        r = r % (long)m;
        return (int)r;
    }

    /*!
     * Modulo m subtraction of x and y using the long data type
     * for intermediate result to avoid overflow.
     * Returns the adjusted (non-negative) result.
     */
    DEVICE_CUDA_FORCEINLINE int mod_psub(const int x, const int y, const int m){
        long r = ((long)x - (long)y + (long)m);
        r = r % (long)m;
        return (int) r;
    }

    /*!
     * Modulo m multiplication of x and y using the long data type
     * for intermediate result to avoid overflow.
     */
    DEVICE_CUDA_FORCEINLINE int mod_mul(const int x, const int y, const int m){
        long r = (long)x * (long)y;
        r = r % (long)m;
        return (int)r;
    }

    /*!
     * Modulo m addition of a*x and b*y using the long data type
     * for intermediate result to avoid overflow.
     * Returned result is (a * x + b * y) mod m
     * In order to speedup computations, the modulo operation is replaced
     * by multiplication by d = 1 / m.
     */
    DEVICE_CUDA_FORCEINLINE int mod_axby(const int a, const int x, const int b, const int y, const int m, const double d){
        long r = (((long) a * (long) x + (long) b * (long) y));
        double quotient = (double) r * d;
        int i = (int) quotient;
        return (int) (r - (long) i * (long) m);
    }

    DEVICE_CUDA_FORCEINLINE int mod_axby(const int a, const int x, const int b, const int y, const int m){
        long r = ((long) a * (long) x + (long) b * (long) y);
        r = r % (long)m;
        return (int)r;
    }

} //end of namespace

/********************* Common RNS functions *********************/


/*!
 * Returns true if the RNS number is zero
 */
GCC_FORCEINLINE bool rns_check_zero(const int *x) {
    for (int i = 0; i < RNS_MODULI_SIZE; i++) {
      if(x[i] != 0){
          return false;
      }
    }
    return true;
}

/*
 * GPU functions
 */
namespace cuda {

    /*!
     * Returns true if the RNS number is zero
     */
    DEVICE_CUDA_FORCEINLINE bool rns_check_zero(const int *x) {
        for (int i = 0; i < RNS_MODULI_SIZE; i++) {
            if(x[i] != 0){
                return false;
            }
        }
        return true;
    }

} //end of namespace

/********************* Unrolled modular arithmetic over RNS numbers *********************/


/*!
 * Multiplication of two RNS numbers.
 */
GCC_FORCEINLINE void rns_mul(int * result, int * x, int * y){
    for(int i = 0; i < RNS_MODULI_SIZE; i++){
        result[i] = mod_mul(x[i], y[i], RNS_MODULI[i], 1.0/RNS_MODULI[i]);
    }
}

/*!
 * Addition of two RNS numbers.
 */
GCC_FORCEINLINE void rns_add(int * result, int * x, int * y){
    for(int i = 0; i < RNS_MODULI_SIZE; i++){
        result[i] = mod_add(x[i], y[i], RNS_MODULI[i], 1.0/RNS_MODULI[i]);
    }
}

/*!
 * Subtraction of two RNS numbers.
 */
GCC_FORCEINLINE void rns_sub(int * result, int * x, int * y){
    for(int i = 0; i < RNS_MODULI_SIZE; i++){
        result[i] = mod_psub(x[i], y[i], RNS_MODULI[i]);
    }
}

/*
 * GPU functions
 */
namespace cuda {

    /*!
     * Multiplication of two RNS numbers.
     */
    DEVICE_CUDA_FORCEINLINE void rns_mul(int * result, const int * x, const int * y){
        constexpr int moduli[ RNS_MODULI_SIZE ] = RNS_MODULI_VALUES;
        #pragma unroll
        for(int i = 0; i < RNS_MODULI_SIZE; i++){
            result[i] = cuda::mod_mul(x[i], y[i], moduli[i]);
        }
    }

    /*!
     * Multiplication of an RNS number by an integer.
     */
    DEVICE_CUDA_FORCEINLINE void rns_mul_l(int * result, const int * x, const long y){
        constexpr int moduli[ RNS_MODULI_SIZE ] = RNS_MODULI_VALUES;
        #pragma unroll
        for(int i = 0; i < RNS_MODULI_SIZE; i++){
            result[i] = cuda::mod_mul(x[i], (y % moduli[i]), moduli[i]);
        }
    }

    /*!
      * Addition of two RNS numbers.
      */
    DEVICE_CUDA_FORCEINLINE void rns_add(int * result, int * x, int * y){
        constexpr int moduli[ RNS_MODULI_SIZE ] = RNS_MODULI_VALUES;
        #pragma unroll
        for(int i = 0; i < RNS_MODULI_SIZE; i++){
            result[i] = cuda::mod_add(x[i], y[i], moduli[i]);
        }
    }

    /*!
     * Subtraction of two RNS numbers.
     */
    DEVICE_CUDA_FORCEINLINE void rns_sub(int * result, int * x, int * y){
        constexpr int moduli[ RNS_MODULI_SIZE ] = RNS_MODULI_VALUES;
        #pragma unroll
        for(int i = 0; i < RNS_MODULI_SIZE; i++){
            result[i] = cuda::mod_psub(x[i], y[i], moduli[i]);
        }
    }

    /*!
     * Multiplication of two RNS numbers and an integer constant
     */
    DEVICE_CUDA_FORCEINLINE void rns_mul_c(int * result, const int * x, const int * y, const int c){
        constexpr int moduli[ RNS_MODULI_SIZE ] = RNS_MODULI_VALUES;
        #pragma unroll
        for(int i = 0; i < RNS_MODULI_SIZE; i++){
            result[i] = c * cuda::mod_mul(x[i], y[i], moduli[i]);
        }
    }

    /*
     * Computes r[i] = (a[i] * x[i] * c) + (b[i] * y[i] * d) mod m
     * If the result is less than zero then it is adjusted by adding m
     */
    DEVICE_CUDA_FORCEINLINE void rns_axby_cd(int *r, const int * a, const int * x, const int c, const int * b, const int * y, const int d){
        constexpr int moduli[ RNS_MODULI_SIZE ] = RNS_MODULI_VALUES;
        #pragma unroll
        for(int i = 0; i < RNS_MODULI_SIZE; i++){
            long res = (long)a[i] * (long)x[i] * (long)c + (long)b[i] * (long)y[i] * (long)d;
            res = res % (long)moduli[i];
            r[i] = (int)res + (res < 0) * moduli[i];
        }
    }

} //end of namespace


#endif //MPRES_MODULAR_CUH
