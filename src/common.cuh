/*
 *  Common useful routines and macros
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

#ifndef MPRES_COMMON_CUH
#define MPRES_COMMON_CUH

#include <stdio.h>
#include <string>
#include <sstream>

/*
 * Macros that define inline specifiers for gcc and nvcc
 */
#define GCC_FORCEINLINE __attribute__((always_inline)) inline
#define DEVICE_CUDA_FORCEINLINE __device__ __forceinline__

/*
 * Checking CUDA results
 */
#define checkDeviceHasErrors(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = false) {
    if (code != cudaSuccess) {
        fprintf(stderr, "%s %s:%d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

#define cudaCheckErrors() {                                        \
 cudaError_t e=cudaGetLastError();                                 \
 if(e!=cudaSuccess) {                                              \
   printf("CUDA Runtime Error %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e)); \
   exit(0); \
 }                                                                 \
}

/*
 * Converts int to std::string
 */
std::string toString(int i) {
    std::string valueAsString;
    std::stringstream strstream;
    strstream << i;
    strstream >> valueAsString;
    return valueAsString;
}

/*
 * Tabulation
 */
namespace std {
    template<typename _CharT, typename _Traits>
    inline basic_ostream <_CharT, _Traits> &
    tab(basic_ostream <_CharT, _Traits> &__os) {
        return __os.put(__os.widen('\t'));
    }
}

/*
 * Round up v to the next highest power of 2
 * https://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2
 */
unsigned int nextPow2(unsigned int v){
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;
    return v;
}

#endif //MPRES_COMMON_CUH
