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

/*
 * The set of RNS moduli for GPU computing.
 * They are copied to the GPU memory in rns.cuh
 */
namespace cuda {
    __device__  int RNS_MODULI[RNS_MODULI_SIZE];
}


/********************* Integer modulo m operations *********************/


/*!
 * Modulo m addition of x and y using the long data type
 * for intermediate result to avoid overflow.
 */
GCC_FORCEINLINE int mod_add(int x, int y, int m){
    long r = (long)x + (long)y;
    r = r % (long)m;
    return (int) r;
}

/*!
 * Modulo m subtraction of x and y using the long data type
 * for intermediate result to avoid overflow.
 * The subtraction result is not adjusted and may be negative
 */
GCC_FORCEINLINE int mod_sub(int x, int y, int m){
    long r = (long)x - (long)y;
    r = r % (long)m;
    return (int) r;
}

/*!
 * Modulo m subtraction of x and y using the long data type
 * for intermediate result to avoid overflow.
 * Returns the adjusted (non-negative) result.
 */
GCC_FORCEINLINE int mod_psub(int x, int y, int m){
    long r = ((long)x - (long)y + (long)m);
    r = r % (long)m;
    return (int) r;
}

/*!
 * Modulo m multiplication of x and y using the long data type
 * for intermediate result to avoid overflow.
 */
GCC_FORCEINLINE int mod_mul(int x, int y, int m){
    long r = (long)x * (long)y;
    r = r % (long)m;
    return (int)r;
}

/*!
 * Modulo m addition of a * x and b * y using the long data type
 * for intermediate result to avoid overflow.
 * Returned result is (a * x + b * y) mod m
 */
GCC_FORCEINLINE int mod_axby(int a, int x, int b, int y, int m){
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
    DEVICE_CUDA_FORCEINLINE int mod_add(int x, int y, int m){
        long r = (long)x + (long)y;
        r = r % (long)m;
        return (int)r;
    }

    /*!
     * Modulo m subtraction of x and y using the long data type
     * for intermediate result to avoid overflow.
     * The subtraction result is not adjusted and may be negative
     */
    DEVICE_CUDA_FORCEINLINE int mod_sub(int x, int y, int m){
        long r = (long)x - (long)y;
        r = r % (long)m;
        return (int)r;
    }

    /*!
     * Modulo m subtraction of x and y using the long data type
     * for intermediate result to avoid overflow.
     * Returns the adjusted (non-negative) result.
     */
    DEVICE_CUDA_FORCEINLINE int mod_psub(int x, int y, int m){
        long r = ((long)x - (long)y + (long)m);
        r = r % (long)m;
        return (int) r;
    }

    /*!
     * Modulo m multiplication of x and y using the long data type
     * for intermediate result to avoid overflow.
     */
    DEVICE_CUDA_FORCEINLINE int mod_mul(int x, int y, int m){
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

} //end of namespace


/********************* Unrolled modular arithmetic over RNS numbers *********************/


/*!
 * Unrolled multiplication of two RNS numbers.
 */
GCC_FORCEINLINE void rns_mul(int * result, int * x, int * y){
    #if RNS_MODULI_SIZE > 0
        result[0] = mod_mul(x[0], y[0], RNS_MODULI[0]);
    #endif
    #if RNS_MODULI_SIZE > 1
        result[1] = mod_mul(x[1], y[1], RNS_MODULI[1]);
    #endif
    #if RNS_MODULI_SIZE > 2
        result[2] = mod_mul(x[2], y[2], RNS_MODULI[2]);
    #endif
    #if RNS_MODULI_SIZE > 3
        result[3] = mod_mul(x[3], y[3], RNS_MODULI[3]);
    #endif
    #if RNS_MODULI_SIZE > 4
        result[4] = mod_mul(x[4], y[4], RNS_MODULI[4]);
    #endif
    #if RNS_MODULI_SIZE > 5
        result[5] = mod_mul(x[5], y[5], RNS_MODULI[5]);
    #endif
    #if RNS_MODULI_SIZE > 6
        result[6] = mod_mul(x[6], y[6], RNS_MODULI[6]);
    #endif
    #if RNS_MODULI_SIZE > 7
        result[7] = mod_mul(x[7], y[7], RNS_MODULI[7]);
    #endif
    #if RNS_MODULI_SIZE > 8
        result[8] = mod_mul(x[8], y[8], RNS_MODULI[8]);
    #endif
    #if RNS_MODULI_SIZE > 9
        result[9] = mod_mul(x[9], y[9], RNS_MODULI[9]);
    #endif
    #if RNS_MODULI_SIZE > 10
        result[10] = mod_mul(x[10], y[10], RNS_MODULI[10]);
    #endif
    #if RNS_MODULI_SIZE > 11
        result[11] = mod_mul(x[11], y[11], RNS_MODULI[11]);
    #endif
    #if RNS_MODULI_SIZE > 12
        result[12] = mod_mul(x[12], y[12], RNS_MODULI[12]);
    #endif
    #if RNS_MODULI_SIZE > 13
        result[13] = mod_mul(x[13], y[13], RNS_MODULI[13]);
    #endif
    #if RNS_MODULI_SIZE > 14
        result[14] = mod_mul(x[14], y[14], RNS_MODULI[14]);
    #endif
    #if RNS_MODULI_SIZE > 15
        result[15] = mod_mul(x[15], y[15], RNS_MODULI[15]);
    #endif
    #if RNS_MODULI_SIZE > 16
        result[16] = mod_mul(x[16], y[16], RNS_MODULI[16]);
    #endif
    #if RNS_MODULI_SIZE > 17
        result[17] = mod_mul(x[17], y[17], RNS_MODULI[17]);
    #endif
    #if RNS_MODULI_SIZE > 18
        result[18] = mod_mul(x[18], y[18], RNS_MODULI[18]);
    #endif
    #if RNS_MODULI_SIZE > 19
        result[19] = mod_mul(x[19], y[19], RNS_MODULI[19]);
    #endif
    #if RNS_MODULI_SIZE > 20
        result[20] = mod_mul(x[20], y[20], RNS_MODULI[20]);
    #endif
    #if RNS_MODULI_SIZE > 21
        result[21] = mod_mul(x[21], y[21], RNS_MODULI[21]);
    #endif
    #if RNS_MODULI_SIZE > 22
        result[22] = mod_mul(x[22], y[22], RNS_MODULI[22]);
    #endif
    #if RNS_MODULI_SIZE > 23
        result[23] = mod_mul(x[23], y[23], RNS_MODULI[23]);
    #endif
    #if RNS_MODULI_SIZE > 24
        result[24] = mod_mul(x[24], y[24], RNS_MODULI[24]);
    #endif
    #if RNS_MODULI_SIZE > 25
        result[25] = mod_mul(x[25], y[25], RNS_MODULI[25]);
    #endif
    #if RNS_MODULI_SIZE > 26
        result[26] = mod_mul(x[26], y[26], RNS_MODULI[26]);
    #endif
    #if RNS_MODULI_SIZE > 27
        result[27] = mod_mul(x[27], y[27], RNS_MODULI[27]);
    #endif
    #if RNS_MODULI_SIZE > 28
        result[28] = mod_mul(x[28], y[28], RNS_MODULI[28]);
    #endif
    #if RNS_MODULI_SIZE > 29
        result[29] = mod_mul(x[29], y[29], RNS_MODULI[29]);
    #endif
    #if RNS_MODULI_SIZE > 30
        result[30] = mod_mul(x[30], y[30], RNS_MODULI[30]);
    #endif
    #if RNS_MODULI_SIZE > 31
        result[31] = mod_mul(x[31], y[31], RNS_MODULI[31]);
    #endif
    #if RNS_MODULI_SIZE > 32
        result[32] = mod_mul(x[32], y[32], RNS_MODULI[32]);
    #endif
    #if RNS_MODULI_SIZE > 33
        result[33] = mod_mul(x[33], y[33], RNS_MODULI[33]);
    #endif
    #if RNS_MODULI_SIZE > 34
        result[34] = mod_mul(x[34], y[34], RNS_MODULI[34]);
    #endif
    #if RNS_MODULI_SIZE > 35
        result[35] = mod_mul(x[35], y[35], RNS_MODULI[35]);
    #endif
    #if RNS_MODULI_SIZE > 36
        result[36] = mod_mul(x[36], y[36], RNS_MODULI[36]);
    #endif
    #if RNS_MODULI_SIZE > 37
        result[37] = mod_mul(x[37], y[37], RNS_MODULI[37]);
    #endif
    #if RNS_MODULI_SIZE > 38
        result[38] = mod_mul(x[38], y[38], RNS_MODULI[38]);
    #endif
    #if RNS_MODULI_SIZE > 39
        result[39] = mod_mul(x[39], y[39], RNS_MODULI[39]);
    #endif
    #if RNS_MODULI_SIZE > 40
        result[40] = mod_mul(x[40], y[40], RNS_MODULI[40]);
    #endif
    #if RNS_MODULI_SIZE > 41
        result[41] = mod_mul(x[41], y[41], RNS_MODULI[41]);
    #endif
    #if RNS_MODULI_SIZE > 42
        result[42] = mod_mul(x[42], y[42], RNS_MODULI[42]);
    #endif
    #if RNS_MODULI_SIZE > 43
        result[43] = mod_mul(x[43], y[43], RNS_MODULI[43]);
    #endif
    #if RNS_MODULI_SIZE > 44
        result[44] = mod_mul(x[44], y[44], RNS_MODULI[44]);
    #endif
    #if RNS_MODULI_SIZE > 45
        result[45] = mod_mul(x[45], y[45], RNS_MODULI[45]);
    #endif
    #if RNS_MODULI_SIZE > 46
        result[46] = mod_mul(x[46], y[46], RNS_MODULI[46]);
    #endif
    #if RNS_MODULI_SIZE > 47
        result[47] = mod_mul(x[47], y[47], RNS_MODULI[47]);
    #endif
    #if RNS_MODULI_SIZE > 48
        result[48] = mod_mul(x[48], y[48], RNS_MODULI[48]);
    #endif
    #if RNS_MODULI_SIZE > 49
        result[49] = mod_mul(x[49], y[49], RNS_MODULI[49]);
    #endif
    #if RNS_MODULI_SIZE > 50
        result[50] = mod_mul(x[50], y[50], RNS_MODULI[50]);
    #endif
    #if RNS_MODULI_SIZE > 51
        result[51] = mod_mul(x[51], y[51], RNS_MODULI[51]);
    #endif
    #if RNS_MODULI_SIZE > 52
        result[52] = mod_mul(x[52], y[52], RNS_MODULI[52]);
    #endif
    #if RNS_MODULI_SIZE > 53
        result[53] = mod_mul(x[53], y[53], RNS_MODULI[53]);
    #endif
    #if RNS_MODULI_SIZE > 54
        result[54] = mod_mul(x[54], y[54], RNS_MODULI[54]);
    #endif
    #if RNS_MODULI_SIZE > 55
        result[55] = mod_mul(x[55], y[55], RNS_MODULI[55]);
    #endif
    #if RNS_MODULI_SIZE > 56
        result[56] = mod_mul(x[56], y[56], RNS_MODULI[56]);
    #endif
    #if RNS_MODULI_SIZE > 57
        result[57] = mod_mul(x[57], y[57], RNS_MODULI[57]);
    #endif
    #if RNS_MODULI_SIZE > 58
        result[58] = mod_mul(x[58], y[58], RNS_MODULI[58]);
    #endif
    #if RNS_MODULI_SIZE > 59
        result[59] = mod_mul(x[59], y[59], RNS_MODULI[59]);
    #endif
    #if RNS_MODULI_SIZE > 60
        result[60] = mod_mul(x[60], y[60], RNS_MODULI[60]);
    #endif
    #if RNS_MODULI_SIZE > 61
        result[61] = mod_mul(x[61], y[61], RNS_MODULI[61]);
    #endif
    #if RNS_MODULI_SIZE > 62
        result[62] = mod_mul(x[62], y[62], RNS_MODULI[62]);
    #endif
    #if RNS_MODULI_SIZE > 63
        result[63] = mod_mul(x[63], y[63], RNS_MODULI[63]);
    #endif
    #if RNS_MODULI_SIZE > 64
        result[64] = mod_mul(x[64], y[64], RNS_MODULI[64]);
    #endif
    #if RNS_MODULI_SIZE > 65
        result[65] = mod_mul(x[65], y[65], RNS_MODULI[65]);
    #endif
    #if RNS_MODULI_SIZE > 66
        result[66] = mod_mul(x[66], y[66], RNS_MODULI[66]);
    #endif
    #if RNS_MODULI_SIZE > 67
        result[67] = mod_mul(x[67], y[67], RNS_MODULI[67]);
    #endif
    #if RNS_MODULI_SIZE > 68
        result[68] = mod_mul(x[68], y[68], RNS_MODULI[68]);
    #endif
    #if RNS_MODULI_SIZE > 69
        result[69] = mod_mul(x[69], y[69], RNS_MODULI[69]);
    #endif
    #if RNS_MODULI_SIZE > 70
        result[70] = mod_mul(x[70], y[70], RNS_MODULI[70]);
    #endif
    #if RNS_MODULI_SIZE > 71
        result[71] = mod_mul(x[71], y[71], RNS_MODULI[71]);
    #endif
    #if RNS_MODULI_SIZE > 72
        result[72] = mod_mul(x[72], y[72], RNS_MODULI[72]);
    #endif
    #if RNS_MODULI_SIZE > 73
        result[73] = mod_mul(x[73], y[73], RNS_MODULI[73]);
    #endif
    #if RNS_MODULI_SIZE > 74
        result[74] = mod_mul(x[74], y[74], RNS_MODULI[74]);
    #endif
    #if RNS_MODULI_SIZE > 75
        result[75] = mod_mul(x[75], y[75], RNS_MODULI[75]);
    #endif
    #if RNS_MODULI_SIZE > 76
        result[76] = mod_mul(x[76], y[76], RNS_MODULI[76]);
    #endif
    #if RNS_MODULI_SIZE > 77
        result[77] = mod_mul(x[77], y[77], RNS_MODULI[77]);
    #endif
    #if RNS_MODULI_SIZE > 78
        result[78] = mod_mul(x[78], y[78], RNS_MODULI[78]);
    #endif
    #if RNS_MODULI_SIZE > 79
        result[79] = mod_mul(x[79], y[79], RNS_MODULI[79]);
    #endif
    #if RNS_MODULI_SIZE > 80
        result[80] = mod_mul(x[80], y[80], RNS_MODULI[80]);
    #endif
    #if RNS_MODULI_SIZE > 81
        result[81] = mod_mul(x[81], y[81], RNS_MODULI[81]);
    #endif
    #if RNS_MODULI_SIZE > 82
        result[82] = mod_mul(x[82], y[82], RNS_MODULI[82]);
    #endif
    #if RNS_MODULI_SIZE > 83
        result[83] = mod_mul(x[83], y[83], RNS_MODULI[83]);
    #endif
    #if RNS_MODULI_SIZE > 84
        result[84] = mod_mul(x[84], y[84], RNS_MODULI[84]);
    #endif
    #if RNS_MODULI_SIZE > 85
        result[85] = mod_mul(x[85], y[85], RNS_MODULI[85]);
    #endif
    #if RNS_MODULI_SIZE > 86
        result[86] = mod_mul(x[86], y[86], RNS_MODULI[86]);
    #endif
    #if RNS_MODULI_SIZE > 87
        result[87] = mod_mul(x[87], y[87], RNS_MODULI[87]);
    #endif
    #if RNS_MODULI_SIZE > 88
        result[88] = mod_mul(x[88], y[88], RNS_MODULI[88]);
    #endif
    #if RNS_MODULI_SIZE > 89
        result[89] = mod_mul(x[89], y[89], RNS_MODULI[89]);
    #endif
    #if RNS_MODULI_SIZE > 90
        result[90] = mod_mul(x[90], y[90], RNS_MODULI[90]);
    #endif
    #if RNS_MODULI_SIZE > 91
        result[91] = mod_mul(x[91], y[91], RNS_MODULI[91]);
    #endif
    #if RNS_MODULI_SIZE > 92
        result[92] = mod_mul(x[92], y[92], RNS_MODULI[92]);
    #endif
    #if RNS_MODULI_SIZE > 93
        result[93] = mod_mul(x[93], y[93], RNS_MODULI[93]);
    #endif
    #if RNS_MODULI_SIZE > 94
        result[94] = mod_mul(x[94], y[94], RNS_MODULI[94]);
    #endif
    #if RNS_MODULI_SIZE > 95
        result[95] = mod_mul(x[95], y[95], RNS_MODULI[95]);
    #endif
    #if RNS_MODULI_SIZE > 96
        result[96] = mod_mul(x[96], y[96], RNS_MODULI[96]);
    #endif
    #if RNS_MODULI_SIZE > 97
        result[97] = mod_mul(x[97], y[97], RNS_MODULI[97]);
    #endif
    #if RNS_MODULI_SIZE > 98
        result[98] = mod_mul(x[98], y[98], RNS_MODULI[98]);
    #endif
    #if RNS_MODULI_SIZE > 99
        result[99] = mod_mul(x[99], y[99], RNS_MODULI[99]);
    #endif
    #if RNS_MODULI_SIZE > 100
        result[100] = mod_mul(x[100], y[100], RNS_MODULI[100]);
    #endif
    #if RNS_MODULI_SIZE > 101
        result[101] = mod_mul(x[101], y[101], RNS_MODULI[101]);
    #endif
    #if RNS_MODULI_SIZE > 102
        result[102] = mod_mul(x[102], y[102], RNS_MODULI[102]);
    #endif
    #if RNS_MODULI_SIZE > 103
        result[103] = mod_mul(x[103], y[103], RNS_MODULI[103]);
    #endif
    #if RNS_MODULI_SIZE > 104
        result[104] = mod_mul(x[104], y[104], RNS_MODULI[104]);
    #endif
    #if RNS_MODULI_SIZE > 105
        result[105] = mod_mul(x[105], y[105], RNS_MODULI[105]);
    #endif
    #if RNS_MODULI_SIZE > 106
        result[106] = mod_mul(x[106], y[106], RNS_MODULI[106]);
    #endif
    #if RNS_MODULI_SIZE > 107
        result[107] = mod_mul(x[107], y[107], RNS_MODULI[107]);
    #endif
    #if RNS_MODULI_SIZE > 108
        result[108] = mod_mul(x[108], y[108], RNS_MODULI[108]);
    #endif
    #if RNS_MODULI_SIZE > 109
        result[109] = mod_mul(x[109], y[109], RNS_MODULI[109]);
    #endif
    #if RNS_MODULI_SIZE > 110
        result[110] = mod_mul(x[110], y[110], RNS_MODULI[110]);
    #endif
    #if RNS_MODULI_SIZE > 111
        result[111] = mod_mul(x[111], y[111], RNS_MODULI[111]);
    #endif
    #if RNS_MODULI_SIZE > 112
        result[112] = mod_mul(x[112], y[112], RNS_MODULI[112]);
    #endif
    #if RNS_MODULI_SIZE > 113
        result[113] = mod_mul(x[113], y[113], RNS_MODULI[113]);
    #endif
    #if RNS_MODULI_SIZE > 114
        result[114] = mod_mul(x[114], y[114], RNS_MODULI[114]);
    #endif
    #if RNS_MODULI_SIZE > 115
        result[115] = mod_mul(x[115], y[115], RNS_MODULI[115]);
    #endif
    #if RNS_MODULI_SIZE > 116
        result[116] = mod_mul(x[116], y[116], RNS_MODULI[116]);
    #endif
    #if RNS_MODULI_SIZE > 117
        result[117] = mod_mul(x[117], y[117], RNS_MODULI[117]);
    #endif
    #if RNS_MODULI_SIZE > 118
        result[118] = mod_mul(x[118], y[118], RNS_MODULI[118]);
    #endif
    #if RNS_MODULI_SIZE > 119
        result[119] = mod_mul(x[119], y[119], RNS_MODULI[119]);
    #endif
    #if RNS_MODULI_SIZE > 120
        result[120] = mod_mul(x[120], y[120], RNS_MODULI[120]);
    #endif
    #if RNS_MODULI_SIZE > 121
        result[121] = mod_mul(x[121], y[121], RNS_MODULI[121]);
    #endif
    #if RNS_MODULI_SIZE > 122
        result[122] = mod_mul(x[122], y[122], RNS_MODULI[122]);
    #endif
    #if RNS_MODULI_SIZE > 123
        result[123] = mod_mul(x[123], y[123], RNS_MODULI[123]);
    #endif
    #if RNS_MODULI_SIZE > 124
        result[124] = mod_mul(x[124], y[124], RNS_MODULI[124]);
    #endif
    #if RNS_MODULI_SIZE > 125
        result[125] = mod_mul(x[125], y[125], RNS_MODULI[125]);
    #endif
    #if RNS_MODULI_SIZE > 126
        result[126] = mod_mul(x[126], y[126], RNS_MODULI[126]);
    #endif
    #if RNS_MODULI_SIZE > 127
        result[127] = mod_mul(x[127], y[127], RNS_MODULI[127]);
    #endif
    #if RNS_MODULI_SIZE > 128
        result[128] = mod_mul(x[128], y[128], RNS_MODULI[128]);
    #endif
    #if RNS_MODULI_SIZE > 129
        result[129] = mod_mul(x[129], y[129], RNS_MODULI[129]);
    #endif
    #if RNS_MODULI_SIZE > 130
        result[130] = mod_mul(x[130], y[130], RNS_MODULI[130]);
    #endif
    #if RNS_MODULI_SIZE > 131
        result[131] = mod_mul(x[131], y[131], RNS_MODULI[131]);
    #endif
    #if RNS_MODULI_SIZE > 132
        result[132] = mod_mul(x[132], y[132], RNS_MODULI[132]);
    #endif
    #if RNS_MODULI_SIZE > 133
        result[133] = mod_mul(x[133], y[133], RNS_MODULI[133]);
    #endif
    #if RNS_MODULI_SIZE > 134
        result[134] = mod_mul(x[134], y[134], RNS_MODULI[134]);
    #endif
    #if RNS_MODULI_SIZE > 135
        result[135] = mod_mul(x[135], y[135], RNS_MODULI[135]);
    #endif
    #if RNS_MODULI_SIZE > 136
        result[136] = mod_mul(x[136], y[136], RNS_MODULI[136]);
    #endif
    #if RNS_MODULI_SIZE > 137
        result[137] = mod_mul(x[137], y[137], RNS_MODULI[137]);
    #endif
    #if RNS_MODULI_SIZE > 138
        result[138] = mod_mul(x[138], y[138], RNS_MODULI[138]);
    #endif
    #if RNS_MODULI_SIZE > 139
        result[139] = mod_mul(x[139], y[139], RNS_MODULI[139]);
    #endif
    #if RNS_MODULI_SIZE > 140
        result[140] = mod_mul(x[140], y[140], RNS_MODULI[140]);
    #endif
    #if RNS_MODULI_SIZE > 141
        result[141] = mod_mul(x[141], y[141], RNS_MODULI[141]);
    #endif
    #if RNS_MODULI_SIZE > 142
        result[142] = mod_mul(x[142], y[142], RNS_MODULI[142]);
    #endif
    #if RNS_MODULI_SIZE > 143
        result[143] = mod_mul(x[143], y[143], RNS_MODULI[143]);
    #endif
    #if RNS_MODULI_SIZE > 144
        result[144] = mod_mul(x[144], y[144], RNS_MODULI[144]);
    #endif
    #if RNS_MODULI_SIZE > 145
        result[145] = mod_mul(x[145], y[145], RNS_MODULI[145]);
    #endif
    #if RNS_MODULI_SIZE > 146
        result[146] = mod_mul(x[146], y[146], RNS_MODULI[146]);
    #endif
    #if RNS_MODULI_SIZE > 147
        result[147] = mod_mul(x[147], y[147], RNS_MODULI[147]);
    #endif
    #if RNS_MODULI_SIZE > 148
        result[148] = mod_mul(x[148], y[148], RNS_MODULI[148]);
    #endif
    #if RNS_MODULI_SIZE > 149
        result[149] = mod_mul(x[149], y[149], RNS_MODULI[149]);
    #endif
    #if RNS_MODULI_SIZE > 150
        result[150] = mod_mul(x[150], y[150], RNS_MODULI[150]);
    #endif
    #if RNS_MODULI_SIZE > 151
        result[151] = mod_mul(x[151], y[151], RNS_MODULI[151]);
    #endif
    #if RNS_MODULI_SIZE > 152
        result[152] = mod_mul(x[152], y[152], RNS_MODULI[152]);
    #endif
    #if RNS_MODULI_SIZE > 153
        result[153] = mod_mul(x[153], y[153], RNS_MODULI[153]);
    #endif
    #if RNS_MODULI_SIZE > 154
        result[154] = mod_mul(x[154], y[154], RNS_MODULI[154]);
    #endif
    #if RNS_MODULI_SIZE > 155
        result[155] = mod_mul(x[155], y[155], RNS_MODULI[155]);
    #endif
    #if RNS_MODULI_SIZE > 156
        result[156] = mod_mul(x[156], y[156], RNS_MODULI[156]);
    #endif
    #if RNS_MODULI_SIZE > 157
        result[157] = mod_mul(x[157], y[157], RNS_MODULI[157]);
    #endif
    #if RNS_MODULI_SIZE > 158
        result[158] = mod_mul(x[158], y[158], RNS_MODULI[158]);
    #endif
    #if RNS_MODULI_SIZE > 159
        result[159] = mod_mul(x[159], y[159], RNS_MODULI[159]);
    #endif
    #if RNS_MODULI_SIZE > 160
        result[160] = mod_mul(x[160], y[160], RNS_MODULI[160]);
    #endif
    #if RNS_MODULI_SIZE > 161
        result[161] = mod_mul(x[161], y[161], RNS_MODULI[161]);
    #endif
    #if RNS_MODULI_SIZE > 162
        result[162] = mod_mul(x[162], y[162], RNS_MODULI[162]);
    #endif
    #if RNS_MODULI_SIZE > 163
        result[163] = mod_mul(x[163], y[163], RNS_MODULI[163]);
    #endif
    #if RNS_MODULI_SIZE > 164
        result[164] = mod_mul(x[164], y[164], RNS_MODULI[164]);
    #endif
    #if RNS_MODULI_SIZE > 165
        result[165] = mod_mul(x[165], y[165], RNS_MODULI[165]);
    #endif
    #if RNS_MODULI_SIZE > 166
        result[166] = mod_mul(x[166], y[166], RNS_MODULI[166]);
    #endif
    #if RNS_MODULI_SIZE > 167
        result[167] = mod_mul(x[167], y[167], RNS_MODULI[167]);
    #endif
    #if RNS_MODULI_SIZE > 168
        result[168] = mod_mul(x[168], y[168], RNS_MODULI[168]);
    #endif
    #if RNS_MODULI_SIZE > 169
        result[169] = mod_mul(x[169], y[169], RNS_MODULI[169]);
    #endif
    #if RNS_MODULI_SIZE > 170
        result[170] = mod_mul(x[170], y[170], RNS_MODULI[170]);
    #endif
    #if RNS_MODULI_SIZE > 171
        result[171] = mod_mul(x[171], y[171], RNS_MODULI[171]);
    #endif
    #if RNS_MODULI_SIZE > 172
        result[172] = mod_mul(x[172], y[172], RNS_MODULI[172]);
    #endif
    #if RNS_MODULI_SIZE > 173
        result[173] = mod_mul(x[173], y[173], RNS_MODULI[173]);
    #endif
    #if RNS_MODULI_SIZE > 174
        result[174] = mod_mul(x[174], y[174], RNS_MODULI[174]);
    #endif
    #if RNS_MODULI_SIZE > 175
        result[175] = mod_mul(x[175], y[175], RNS_MODULI[175]);
    #endif
    #if RNS_MODULI_SIZE > 176
        result[176] = mod_mul(x[176], y[176], RNS_MODULI[176]);
    #endif
    #if RNS_MODULI_SIZE > 177
        result[177] = mod_mul(x[177], y[177], RNS_MODULI[177]);
    #endif
    #if RNS_MODULI_SIZE > 178
        result[178] = mod_mul(x[178], y[178], RNS_MODULI[178]);
    #endif
    #if RNS_MODULI_SIZE > 179
        result[179] = mod_mul(x[179], y[179], RNS_MODULI[179]);
    #endif
    #if RNS_MODULI_SIZE > 180
        result[180] = mod_mul(x[180], y[180], RNS_MODULI[180]);
    #endif
    #if RNS_MODULI_SIZE > 181
        result[181] = mod_mul(x[181], y[181], RNS_MODULI[181]);
    #endif
    #if RNS_MODULI_SIZE > 182
        result[182] = mod_mul(x[182], y[182], RNS_MODULI[182]);
    #endif
    #if RNS_MODULI_SIZE > 183
        result[183] = mod_mul(x[183], y[183], RNS_MODULI[183]);
    #endif
    #if RNS_MODULI_SIZE > 184
        result[184] = mod_mul(x[184], y[184], RNS_MODULI[184]);
    #endif
    #if RNS_MODULI_SIZE > 185
        result[185] = mod_mul(x[185], y[185], RNS_MODULI[185]);
    #endif
    #if RNS_MODULI_SIZE > 186
        result[186] = mod_mul(x[186], y[186], RNS_MODULI[186]);
    #endif
    #if RNS_MODULI_SIZE > 187
        result[187] = mod_mul(x[187], y[187], RNS_MODULI[187]);
    #endif
    #if RNS_MODULI_SIZE > 188
        result[188] = mod_mul(x[188], y[188], RNS_MODULI[188]);
    #endif
    #if RNS_MODULI_SIZE > 189
        result[189] = mod_mul(x[189], y[189], RNS_MODULI[189]);
    #endif
    #if RNS_MODULI_SIZE > 190
        result[190] = mod_mul(x[190], y[190], RNS_MODULI[190]);
    #endif
    #if RNS_MODULI_SIZE > 191
        result[191] = mod_mul(x[191], y[191], RNS_MODULI[191]);
    #endif
    #if RNS_MODULI_SIZE > 192
        result[192] = mod_mul(x[192], y[192], RNS_MODULI[192]);
    #endif
    #if RNS_MODULI_SIZE > 193
        result[193] = mod_mul(x[193], y[193], RNS_MODULI[193]);
    #endif
    #if RNS_MODULI_SIZE > 194
        result[194] = mod_mul(x[194], y[194], RNS_MODULI[194]);
    #endif
    #if RNS_MODULI_SIZE > 195
        result[195] = mod_mul(x[195], y[195], RNS_MODULI[195]);
    #endif
    #if RNS_MODULI_SIZE > 196
        result[196] = mod_mul(x[196], y[196], RNS_MODULI[196]);
    #endif
    #if RNS_MODULI_SIZE > 197
        result[197] = mod_mul(x[197], y[197], RNS_MODULI[197]);
    #endif
    #if RNS_MODULI_SIZE > 198
        result[198] = mod_mul(x[198], y[198], RNS_MODULI[198]);
    #endif
    #if RNS_MODULI_SIZE > 199
        result[199] = mod_mul(x[199], y[199], RNS_MODULI[199]);
    #endif
}

/*!
 * Unrolled addition of two RNS numbers.
 */
GCC_FORCEINLINE void rns_add(int * result, int * x, int * y){
    #if RNS_MODULI_SIZE > 0
        result[0] = mod_add(x[0], y[0], RNS_MODULI[0]);
    #endif
    #if RNS_MODULI_SIZE > 1
        result[1] = mod_add(x[1], y[1], RNS_MODULI[1]);
    #endif
    #if RNS_MODULI_SIZE > 2
        result[2] = mod_add(x[2], y[2], RNS_MODULI[2]);
    #endif
    #if RNS_MODULI_SIZE > 3
        result[3] = mod_add(x[3], y[3], RNS_MODULI[3]);
    #endif
    #if RNS_MODULI_SIZE > 4
        result[4] = mod_add(x[4], y[4], RNS_MODULI[4]);
    #endif
    #if RNS_MODULI_SIZE > 5
        result[5] = mod_add(x[5], y[5], RNS_MODULI[5]);
    #endif
    #if RNS_MODULI_SIZE > 6
        result[6] = mod_add(x[6], y[6], RNS_MODULI[6]);
    #endif
    #if RNS_MODULI_SIZE > 7
        result[7] = mod_add(x[7], y[7], RNS_MODULI[7]);
    #endif
    #if RNS_MODULI_SIZE > 8
        result[8] = mod_add(x[8], y[8], RNS_MODULI[8]);
    #endif
    #if RNS_MODULI_SIZE > 9
        result[9] = mod_add(x[9], y[9], RNS_MODULI[9]);
    #endif
    #if RNS_MODULI_SIZE > 10
        result[10] = mod_add(x[10], y[10], RNS_MODULI[10]);
    #endif
    #if RNS_MODULI_SIZE > 11
        result[11] = mod_add(x[11], y[11], RNS_MODULI[11]);
    #endif
    #if RNS_MODULI_SIZE > 12
        result[12] = mod_add(x[12], y[12], RNS_MODULI[12]);
    #endif
    #if RNS_MODULI_SIZE > 13
        result[13] = mod_add(x[13], y[13], RNS_MODULI[13]);
    #endif
    #if RNS_MODULI_SIZE > 14
        result[14] = mod_add(x[14], y[14], RNS_MODULI[14]);
    #endif
    #if RNS_MODULI_SIZE > 15
        result[15] = mod_add(x[15], y[15], RNS_MODULI[15]);
    #endif
    #if RNS_MODULI_SIZE > 16
        result[16] = mod_add(x[16], y[16], RNS_MODULI[16]);
    #endif
    #if RNS_MODULI_SIZE > 17
        result[17] = mod_add(x[17], y[17], RNS_MODULI[17]);
    #endif
    #if RNS_MODULI_SIZE > 18
        result[18] = mod_add(x[18], y[18], RNS_MODULI[18]);
    #endif
    #if RNS_MODULI_SIZE > 19
        result[19] = mod_add(x[19], y[19], RNS_MODULI[19]);
    #endif
    #if RNS_MODULI_SIZE > 20
        result[20] = mod_add(x[20], y[20], RNS_MODULI[20]);
    #endif
    #if RNS_MODULI_SIZE > 21
        result[21] = mod_add(x[21], y[21], RNS_MODULI[21]);
    #endif
    #if RNS_MODULI_SIZE > 22
        result[22] = mod_add(x[22], y[22], RNS_MODULI[22]);
    #endif
    #if RNS_MODULI_SIZE > 23
        result[23] = mod_add(x[23], y[23], RNS_MODULI[23]);
    #endif
    #if RNS_MODULI_SIZE > 24
        result[24] = mod_add(x[24], y[24], RNS_MODULI[24]);
    #endif
    #if RNS_MODULI_SIZE > 25
        result[25] = mod_add(x[25], y[25], RNS_MODULI[25]);
    #endif
    #if RNS_MODULI_SIZE > 26
        result[26] = mod_add(x[26], y[26], RNS_MODULI[26]);
    #endif
    #if RNS_MODULI_SIZE > 27
        result[27] = mod_add(x[27], y[27], RNS_MODULI[27]);
    #endif
    #if RNS_MODULI_SIZE > 28
        result[28] = mod_add(x[28], y[28], RNS_MODULI[28]);
    #endif
    #if RNS_MODULI_SIZE > 29
        result[29] = mod_add(x[29], y[29], RNS_MODULI[29]);
    #endif
    #if RNS_MODULI_SIZE > 30
        result[30] = mod_add(x[30], y[30], RNS_MODULI[30]);
    #endif
    #if RNS_MODULI_SIZE > 31
        result[31] = mod_add(x[31], y[31], RNS_MODULI[31]);
    #endif
    #if RNS_MODULI_SIZE > 32
        result[32] = mod_add(x[32], y[32], RNS_MODULI[32]);
    #endif
    #if RNS_MODULI_SIZE > 33
        result[33] = mod_add(x[33], y[33], RNS_MODULI[33]);
    #endif
    #if RNS_MODULI_SIZE > 34
        result[34] = mod_add(x[34], y[34], RNS_MODULI[34]);
    #endif
    #if RNS_MODULI_SIZE > 35
        result[35] = mod_add(x[35], y[35], RNS_MODULI[35]);
    #endif
    #if RNS_MODULI_SIZE > 36
        result[36] = mod_add(x[36], y[36], RNS_MODULI[36]);
    #endif
    #if RNS_MODULI_SIZE > 37
        result[37] = mod_add(x[37], y[37], RNS_MODULI[37]);
    #endif
    #if RNS_MODULI_SIZE > 38
        result[38] = mod_add(x[38], y[38], RNS_MODULI[38]);
    #endif
    #if RNS_MODULI_SIZE > 39
        result[39] = mod_add(x[39], y[39], RNS_MODULI[39]);
    #endif
    #if RNS_MODULI_SIZE > 40
        result[40] = mod_add(x[40], y[40], RNS_MODULI[40]);
    #endif
    #if RNS_MODULI_SIZE > 41
        result[41] = mod_add(x[41], y[41], RNS_MODULI[41]);
    #endif
    #if RNS_MODULI_SIZE > 42
        result[42] = mod_add(x[42], y[42], RNS_MODULI[42]);
    #endif
    #if RNS_MODULI_SIZE > 43
        result[43] = mod_add(x[43], y[43], RNS_MODULI[43]);
    #endif
    #if RNS_MODULI_SIZE > 44
        result[44] = mod_add(x[44], y[44], RNS_MODULI[44]);
    #endif
    #if RNS_MODULI_SIZE > 45
        result[45] = mod_add(x[45], y[45], RNS_MODULI[45]);
    #endif
    #if RNS_MODULI_SIZE > 46
        result[46] = mod_add(x[46], y[46], RNS_MODULI[46]);
    #endif
    #if RNS_MODULI_SIZE > 47
        result[47] = mod_add(x[47], y[47], RNS_MODULI[47]);
    #endif
    #if RNS_MODULI_SIZE > 48
        result[48] = mod_add(x[48], y[48], RNS_MODULI[48]);
    #endif
    #if RNS_MODULI_SIZE > 49
        result[49] = mod_add(x[49], y[49], RNS_MODULI[49]);
    #endif
    #if RNS_MODULI_SIZE > 50
        result[50] = mod_add(x[50], y[50], RNS_MODULI[50]);
    #endif
    #if RNS_MODULI_SIZE > 51
        result[51] = mod_add(x[51], y[51], RNS_MODULI[51]);
    #endif
    #if RNS_MODULI_SIZE > 52
        result[52] = mod_add(x[52], y[52], RNS_MODULI[52]);
    #endif
    #if RNS_MODULI_SIZE > 53
        result[53] = mod_add(x[53], y[53], RNS_MODULI[53]);
    #endif
    #if RNS_MODULI_SIZE > 54
        result[54] = mod_add(x[54], y[54], RNS_MODULI[54]);
    #endif
    #if RNS_MODULI_SIZE > 55
        result[55] = mod_add(x[55], y[55], RNS_MODULI[55]);
    #endif
    #if RNS_MODULI_SIZE > 56
        result[56] = mod_add(x[56], y[56], RNS_MODULI[56]);
    #endif
    #if RNS_MODULI_SIZE > 57
        result[57] = mod_add(x[57], y[57], RNS_MODULI[57]);
    #endif
    #if RNS_MODULI_SIZE > 58
        result[58] = mod_add(x[58], y[58], RNS_MODULI[58]);
    #endif
    #if RNS_MODULI_SIZE > 59
        result[59] = mod_add(x[59], y[59], RNS_MODULI[59]);
    #endif
    #if RNS_MODULI_SIZE > 60
        result[60] = mod_add(x[60], y[60], RNS_MODULI[60]);
    #endif
    #if RNS_MODULI_SIZE > 61
        result[61] = mod_add(x[61], y[61], RNS_MODULI[61]);
    #endif
    #if RNS_MODULI_SIZE > 62
        result[62] = mod_add(x[62], y[62], RNS_MODULI[62]);
    #endif
    #if RNS_MODULI_SIZE > 63
        result[63] = mod_add(x[63], y[63], RNS_MODULI[63]);
    #endif
    #if RNS_MODULI_SIZE > 64
        result[64] = mod_add(x[64], y[64], RNS_MODULI[64]);
    #endif
    #if RNS_MODULI_SIZE > 65
        result[65] = mod_add(x[65], y[65], RNS_MODULI[65]);
    #endif
    #if RNS_MODULI_SIZE > 66
        result[66] = mod_add(x[66], y[66], RNS_MODULI[66]);
    #endif
    #if RNS_MODULI_SIZE > 67
        result[67] = mod_add(x[67], y[67], RNS_MODULI[67]);
    #endif
    #if RNS_MODULI_SIZE > 68
        result[68] = mod_add(x[68], y[68], RNS_MODULI[68]);
    #endif
    #if RNS_MODULI_SIZE > 69
        result[69] = mod_add(x[69], y[69], RNS_MODULI[69]);
    #endif
    #if RNS_MODULI_SIZE > 70
        result[70] = mod_add(x[70], y[70], RNS_MODULI[70]);
    #endif
    #if RNS_MODULI_SIZE > 71
        result[71] = mod_add(x[71], y[71], RNS_MODULI[71]);
    #endif
    #if RNS_MODULI_SIZE > 72
        result[72] = mod_add(x[72], y[72], RNS_MODULI[72]);
    #endif
    #if RNS_MODULI_SIZE > 73
        result[73] = mod_add(x[73], y[73], RNS_MODULI[73]);
    #endif
    #if RNS_MODULI_SIZE > 74
        result[74] = mod_add(x[74], y[74], RNS_MODULI[74]);
    #endif
    #if RNS_MODULI_SIZE > 75
        result[75] = mod_add(x[75], y[75], RNS_MODULI[75]);
    #endif
    #if RNS_MODULI_SIZE > 76
        result[76] = mod_add(x[76], y[76], RNS_MODULI[76]);
    #endif
    #if RNS_MODULI_SIZE > 77
        result[77] = mod_add(x[77], y[77], RNS_MODULI[77]);
    #endif
    #if RNS_MODULI_SIZE > 78
        result[78] = mod_add(x[78], y[78], RNS_MODULI[78]);
    #endif
    #if RNS_MODULI_SIZE > 79
        result[79] = mod_add(x[79], y[79], RNS_MODULI[79]);
    #endif
    #if RNS_MODULI_SIZE > 80
        result[80] = mod_add(x[80], y[80], RNS_MODULI[80]);
    #endif
    #if RNS_MODULI_SIZE > 81
        result[81] = mod_add(x[81], y[81], RNS_MODULI[81]);
    #endif
    #if RNS_MODULI_SIZE > 82
        result[82] = mod_add(x[82], y[82], RNS_MODULI[82]);
    #endif
    #if RNS_MODULI_SIZE > 83
        result[83] = mod_add(x[83], y[83], RNS_MODULI[83]);
    #endif
    #if RNS_MODULI_SIZE > 84
        result[84] = mod_add(x[84], y[84], RNS_MODULI[84]);
    #endif
    #if RNS_MODULI_SIZE > 85
        result[85] = mod_add(x[85], y[85], RNS_MODULI[85]);
    #endif
    #if RNS_MODULI_SIZE > 86
        result[86] = mod_add(x[86], y[86], RNS_MODULI[86]);
    #endif
    #if RNS_MODULI_SIZE > 87
        result[87] = mod_add(x[87], y[87], RNS_MODULI[87]);
    #endif
    #if RNS_MODULI_SIZE > 88
        result[88] = mod_add(x[88], y[88], RNS_MODULI[88]);
    #endif
    #if RNS_MODULI_SIZE > 89
        result[89] = mod_add(x[89], y[89], RNS_MODULI[89]);
    #endif
    #if RNS_MODULI_SIZE > 90
        result[90] = mod_add(x[90], y[90], RNS_MODULI[90]);
    #endif
    #if RNS_MODULI_SIZE > 91
        result[91] = mod_add(x[91], y[91], RNS_MODULI[91]);
    #endif
    #if RNS_MODULI_SIZE > 92
        result[92] = mod_add(x[92], y[92], RNS_MODULI[92]);
    #endif
    #if RNS_MODULI_SIZE > 93
        result[93] = mod_add(x[93], y[93], RNS_MODULI[93]);
    #endif
    #if RNS_MODULI_SIZE > 94
        result[94] = mod_add(x[94], y[94], RNS_MODULI[94]);
    #endif
    #if RNS_MODULI_SIZE > 95
        result[95] = mod_add(x[95], y[95], RNS_MODULI[95]);
    #endif
    #if RNS_MODULI_SIZE > 96
        result[96] = mod_add(x[96], y[96], RNS_MODULI[96]);
    #endif
    #if RNS_MODULI_SIZE > 97
        result[97] = mod_add(x[97], y[97], RNS_MODULI[97]);
    #endif
    #if RNS_MODULI_SIZE > 98
        result[98] = mod_add(x[98], y[98], RNS_MODULI[98]);
    #endif
    #if RNS_MODULI_SIZE > 99
        result[99] = mod_add(x[99], y[99], RNS_MODULI[99]);
    #endif
    #if RNS_MODULI_SIZE > 100
        result[100] = mod_add(x[100], y[100], RNS_MODULI[100]);
    #endif
    #if RNS_MODULI_SIZE > 101
        result[101] = mod_add(x[101], y[101], RNS_MODULI[101]);
    #endif
    #if RNS_MODULI_SIZE > 102
        result[102] = mod_add(x[102], y[102], RNS_MODULI[102]);
    #endif
    #if RNS_MODULI_SIZE > 103
        result[103] = mod_add(x[103], y[103], RNS_MODULI[103]);
    #endif
    #if RNS_MODULI_SIZE > 104
        result[104] = mod_add(x[104], y[104], RNS_MODULI[104]);
    #endif
    #if RNS_MODULI_SIZE > 105
        result[105] = mod_add(x[105], y[105], RNS_MODULI[105]);
    #endif
    #if RNS_MODULI_SIZE > 106
        result[106] = mod_add(x[106], y[106], RNS_MODULI[106]);
    #endif
    #if RNS_MODULI_SIZE > 107
        result[107] = mod_add(x[107], y[107], RNS_MODULI[107]);
    #endif
    #if RNS_MODULI_SIZE > 108
        result[108] = mod_add(x[108], y[108], RNS_MODULI[108]);
    #endif
    #if RNS_MODULI_SIZE > 109
        result[109] = mod_add(x[109], y[109], RNS_MODULI[109]);
    #endif
    #if RNS_MODULI_SIZE > 110
        result[110] = mod_add(x[110], y[110], RNS_MODULI[110]);
    #endif
    #if RNS_MODULI_SIZE > 111
        result[111] = mod_add(x[111], y[111], RNS_MODULI[111]);
    #endif
    #if RNS_MODULI_SIZE > 112
        result[112] = mod_add(x[112], y[112], RNS_MODULI[112]);
    #endif
    #if RNS_MODULI_SIZE > 113
        result[113] = mod_add(x[113], y[113], RNS_MODULI[113]);
    #endif
    #if RNS_MODULI_SIZE > 114
        result[114] = mod_add(x[114], y[114], RNS_MODULI[114]);
    #endif
    #if RNS_MODULI_SIZE > 115
        result[115] = mod_add(x[115], y[115], RNS_MODULI[115]);
    #endif
    #if RNS_MODULI_SIZE > 116
        result[116] = mod_add(x[116], y[116], RNS_MODULI[116]);
    #endif
    #if RNS_MODULI_SIZE > 117
        result[117] = mod_add(x[117], y[117], RNS_MODULI[117]);
    #endif
    #if RNS_MODULI_SIZE > 118
        result[118] = mod_add(x[118], y[118], RNS_MODULI[118]);
    #endif
    #if RNS_MODULI_SIZE > 119
        result[119] = mod_add(x[119], y[119], RNS_MODULI[119]);
    #endif
    #if RNS_MODULI_SIZE > 120
        result[120] = mod_add(x[120], y[120], RNS_MODULI[120]);
    #endif
    #if RNS_MODULI_SIZE > 121
        result[121] = mod_add(x[121], y[121], RNS_MODULI[121]);
    #endif
    #if RNS_MODULI_SIZE > 122
        result[122] = mod_add(x[122], y[122], RNS_MODULI[122]);
    #endif
    #if RNS_MODULI_SIZE > 123
        result[123] = mod_add(x[123], y[123], RNS_MODULI[123]);
    #endif
    #if RNS_MODULI_SIZE > 124
        result[124] = mod_add(x[124], y[124], RNS_MODULI[124]);
    #endif
    #if RNS_MODULI_SIZE > 125
        result[125] = mod_add(x[125], y[125], RNS_MODULI[125]);
    #endif
    #if RNS_MODULI_SIZE > 126
        result[126] = mod_add(x[126], y[126], RNS_MODULI[126]);
    #endif
    #if RNS_MODULI_SIZE > 127
        result[127] = mod_add(x[127], y[127], RNS_MODULI[127]);
    #endif
    #if RNS_MODULI_SIZE > 128
        result[128] = mod_add(x[128], y[128], RNS_MODULI[128]);
    #endif
    #if RNS_MODULI_SIZE > 129
        result[129] = mod_add(x[129], y[129], RNS_MODULI[129]);
    #endif
    #if RNS_MODULI_SIZE > 130
        result[130] = mod_add(x[130], y[130], RNS_MODULI[130]);
    #endif
    #if RNS_MODULI_SIZE > 131
        result[131] = mod_add(x[131], y[131], RNS_MODULI[131]);
    #endif
    #if RNS_MODULI_SIZE > 132
        result[132] = mod_add(x[132], y[132], RNS_MODULI[132]);
    #endif
    #if RNS_MODULI_SIZE > 133
        result[133] = mod_add(x[133], y[133], RNS_MODULI[133]);
    #endif
    #if RNS_MODULI_SIZE > 134
        result[134] = mod_add(x[134], y[134], RNS_MODULI[134]);
    #endif
    #if RNS_MODULI_SIZE > 135
        result[135] = mod_add(x[135], y[135], RNS_MODULI[135]);
    #endif
    #if RNS_MODULI_SIZE > 136
        result[136] = mod_add(x[136], y[136], RNS_MODULI[136]);
    #endif
    #if RNS_MODULI_SIZE > 137
        result[137] = mod_add(x[137], y[137], RNS_MODULI[137]);
    #endif
    #if RNS_MODULI_SIZE > 138
        result[138] = mod_add(x[138], y[138], RNS_MODULI[138]);
    #endif
    #if RNS_MODULI_SIZE > 139
        result[139] = mod_add(x[139], y[139], RNS_MODULI[139]);
    #endif
    #if RNS_MODULI_SIZE > 140
        result[140] = mod_add(x[140], y[140], RNS_MODULI[140]);
    #endif
    #if RNS_MODULI_SIZE > 141
        result[141] = mod_add(x[141], y[141], RNS_MODULI[141]);
    #endif
    #if RNS_MODULI_SIZE > 142
        result[142] = mod_add(x[142], y[142], RNS_MODULI[142]);
    #endif
    #if RNS_MODULI_SIZE > 143
        result[143] = mod_add(x[143], y[143], RNS_MODULI[143]);
    #endif
    #if RNS_MODULI_SIZE > 144
        result[144] = mod_add(x[144], y[144], RNS_MODULI[144]);
    #endif
    #if RNS_MODULI_SIZE > 145
        result[145] = mod_add(x[145], y[145], RNS_MODULI[145]);
    #endif
    #if RNS_MODULI_SIZE > 146
        result[146] = mod_add(x[146], y[146], RNS_MODULI[146]);
    #endif
    #if RNS_MODULI_SIZE > 147
        result[147] = mod_add(x[147], y[147], RNS_MODULI[147]);
    #endif
    #if RNS_MODULI_SIZE > 148
        result[148] = mod_add(x[148], y[148], RNS_MODULI[148]);
    #endif
    #if RNS_MODULI_SIZE > 149
        result[149] = mod_add(x[149], y[149], RNS_MODULI[149]);
    #endif
    #if RNS_MODULI_SIZE > 150
        result[150] = mod_add(x[150], y[150], RNS_MODULI[150]);
    #endif
    #if RNS_MODULI_SIZE > 151
        result[151] = mod_add(x[151], y[151], RNS_MODULI[151]);
    #endif
    #if RNS_MODULI_SIZE > 152
        result[152] = mod_add(x[152], y[152], RNS_MODULI[152]);
    #endif
    #if RNS_MODULI_SIZE > 153
        result[153] = mod_add(x[153], y[153], RNS_MODULI[153]);
    #endif
    #if RNS_MODULI_SIZE > 154
        result[154] = mod_add(x[154], y[154], RNS_MODULI[154]);
    #endif
    #if RNS_MODULI_SIZE > 155
        result[155] = mod_add(x[155], y[155], RNS_MODULI[155]);
    #endif
    #if RNS_MODULI_SIZE > 156
        result[156] = mod_add(x[156], y[156], RNS_MODULI[156]);
    #endif
    #if RNS_MODULI_SIZE > 157
        result[157] = mod_add(x[157], y[157], RNS_MODULI[157]);
    #endif
    #if RNS_MODULI_SIZE > 158
        result[158] = mod_add(x[158], y[158], RNS_MODULI[158]);
    #endif
    #if RNS_MODULI_SIZE > 159
        result[159] = mod_add(x[159], y[159], RNS_MODULI[159]);
    #endif
    #if RNS_MODULI_SIZE > 160
        result[160] = mod_add(x[160], y[160], RNS_MODULI[160]);
    #endif
    #if RNS_MODULI_SIZE > 161
        result[161] = mod_add(x[161], y[161], RNS_MODULI[161]);
    #endif
    #if RNS_MODULI_SIZE > 162
        result[162] = mod_add(x[162], y[162], RNS_MODULI[162]);
    #endif
    #if RNS_MODULI_SIZE > 163
        result[163] = mod_add(x[163], y[163], RNS_MODULI[163]);
    #endif
    #if RNS_MODULI_SIZE > 164
        result[164] = mod_add(x[164], y[164], RNS_MODULI[164]);
    #endif
    #if RNS_MODULI_SIZE > 165
        result[165] = mod_add(x[165], y[165], RNS_MODULI[165]);
    #endif
    #if RNS_MODULI_SIZE > 166
        result[166] = mod_add(x[166], y[166], RNS_MODULI[166]);
    #endif
    #if RNS_MODULI_SIZE > 167
        result[167] = mod_add(x[167], y[167], RNS_MODULI[167]);
    #endif
    #if RNS_MODULI_SIZE > 168
        result[168] = mod_add(x[168], y[168], RNS_MODULI[168]);
    #endif
    #if RNS_MODULI_SIZE > 169
        result[169] = mod_add(x[169], y[169], RNS_MODULI[169]);
    #endif
    #if RNS_MODULI_SIZE > 170
        result[170] = mod_add(x[170], y[170], RNS_MODULI[170]);
    #endif
    #if RNS_MODULI_SIZE > 171
        result[171] = mod_add(x[171], y[171], RNS_MODULI[171]);
    #endif
    #if RNS_MODULI_SIZE > 172
        result[172] = mod_add(x[172], y[172], RNS_MODULI[172]);
    #endif
    #if RNS_MODULI_SIZE > 173
        result[173] = mod_add(x[173], y[173], RNS_MODULI[173]);
    #endif
    #if RNS_MODULI_SIZE > 174
        result[174] = mod_add(x[174], y[174], RNS_MODULI[174]);
    #endif
    #if RNS_MODULI_SIZE > 175
        result[175] = mod_add(x[175], y[175], RNS_MODULI[175]);
    #endif
    #if RNS_MODULI_SIZE > 176
        result[176] = mod_add(x[176], y[176], RNS_MODULI[176]);
    #endif
    #if RNS_MODULI_SIZE > 177
        result[177] = mod_add(x[177], y[177], RNS_MODULI[177]);
    #endif
    #if RNS_MODULI_SIZE > 178
        result[178] = mod_add(x[178], y[178], RNS_MODULI[178]);
    #endif
    #if RNS_MODULI_SIZE > 179
        result[179] = mod_add(x[179], y[179], RNS_MODULI[179]);
    #endif
    #if RNS_MODULI_SIZE > 180
        result[180] = mod_add(x[180], y[180], RNS_MODULI[180]);
    #endif
    #if RNS_MODULI_SIZE > 181
        result[181] = mod_add(x[181], y[181], RNS_MODULI[181]);
    #endif
    #if RNS_MODULI_SIZE > 182
        result[182] = mod_add(x[182], y[182], RNS_MODULI[182]);
    #endif
    #if RNS_MODULI_SIZE > 183
        result[183] = mod_add(x[183], y[183], RNS_MODULI[183]);
    #endif
    #if RNS_MODULI_SIZE > 184
        result[184] = mod_add(x[184], y[184], RNS_MODULI[184]);
    #endif
    #if RNS_MODULI_SIZE > 185
        result[185] = mod_add(x[185], y[185], RNS_MODULI[185]);
    #endif
    #if RNS_MODULI_SIZE > 186
        result[186] = mod_add(x[186], y[186], RNS_MODULI[186]);
    #endif
    #if RNS_MODULI_SIZE > 187
        result[187] = mod_add(x[187], y[187], RNS_MODULI[187]);
    #endif
    #if RNS_MODULI_SIZE > 188
        result[188] = mod_add(x[188], y[188], RNS_MODULI[188]);
    #endif
    #if RNS_MODULI_SIZE > 189
        result[189] = mod_add(x[189], y[189], RNS_MODULI[189]);
    #endif
    #if RNS_MODULI_SIZE > 190
        result[190] = mod_add(x[190], y[190], RNS_MODULI[190]);
    #endif
    #if RNS_MODULI_SIZE > 191
        result[191] = mod_add(x[191], y[191], RNS_MODULI[191]);
    #endif
    #if RNS_MODULI_SIZE > 192
        result[192] = mod_add(x[192], y[192], RNS_MODULI[192]);
    #endif
    #if RNS_MODULI_SIZE > 193
        result[193] = mod_add(x[193], y[193], RNS_MODULI[193]);
    #endif
    #if RNS_MODULI_SIZE > 194
        result[194] = mod_add(x[194], y[194], RNS_MODULI[194]);
    #endif
    #if RNS_MODULI_SIZE > 195
        result[195] = mod_add(x[195], y[195], RNS_MODULI[195]);
    #endif
    #if RNS_MODULI_SIZE > 196
        result[196] = mod_add(x[196], y[196], RNS_MODULI[196]);
    #endif
    #if RNS_MODULI_SIZE > 197
        result[197] = mod_add(x[197], y[197], RNS_MODULI[197]);
    #endif
    #if RNS_MODULI_SIZE > 198
        result[198] = mod_add(x[198], y[198], RNS_MODULI[198]);
    #endif
    #if RNS_MODULI_SIZE > 199
        result[199] = mod_add(x[199], y[199], RNS_MODULI[199]);
    #endif
}

/*!
 * Unrolled subtraction of two RNS numbers.
 */
GCC_FORCEINLINE void rns_sub(int * result, int * x, int * y){
    #if RNS_MODULI_SIZE > 0
        result[0] = mod_psub(x[0], y[0], RNS_MODULI[0]);
    #endif
    #if RNS_MODULI_SIZE > 1
        result[1] = mod_psub(x[1], y[1], RNS_MODULI[1]);
    #endif
    #if RNS_MODULI_SIZE > 2
        result[2] = mod_psub(x[2], y[2], RNS_MODULI[2]);
    #endif
    #if RNS_MODULI_SIZE > 3
        result[3] = mod_psub(x[3], y[3], RNS_MODULI[3]);
    #endif
    #if RNS_MODULI_SIZE > 4
        result[4] = mod_psub(x[4], y[4], RNS_MODULI[4]);
    #endif
    #if RNS_MODULI_SIZE > 5
        result[5] = mod_psub(x[5], y[5], RNS_MODULI[5]);
    #endif
    #if RNS_MODULI_SIZE > 6
        result[6] = mod_psub(x[6], y[6], RNS_MODULI[6]);
    #endif
    #if RNS_MODULI_SIZE > 7
        result[7] = mod_psub(x[7], y[7], RNS_MODULI[7]);
    #endif
    #if RNS_MODULI_SIZE > 8
        result[8] = mod_psub(x[8], y[8], RNS_MODULI[8]);
    #endif
    #if RNS_MODULI_SIZE > 9
        result[9] = mod_psub(x[9], y[9], RNS_MODULI[9]);
    #endif
    #if RNS_MODULI_SIZE > 10
        result[10] = mod_psub(x[10], y[10], RNS_MODULI[10]);
    #endif
    #if RNS_MODULI_SIZE > 11
        result[11] = mod_psub(x[11], y[11], RNS_MODULI[11]);
    #endif
    #if RNS_MODULI_SIZE > 12
        result[12] = mod_psub(x[12], y[12], RNS_MODULI[12]);
    #endif
    #if RNS_MODULI_SIZE > 13
        result[13] = mod_psub(x[13], y[13], RNS_MODULI[13]);
    #endif
    #if RNS_MODULI_SIZE > 14
        result[14] = mod_psub(x[14], y[14], RNS_MODULI[14]);
    #endif
    #if RNS_MODULI_SIZE > 15
        result[15] = mod_psub(x[15], y[15], RNS_MODULI[15]);
    #endif
    #if RNS_MODULI_SIZE > 16
        result[16] = mod_psub(x[16], y[16], RNS_MODULI[16]);
    #endif
    #if RNS_MODULI_SIZE > 17
        result[17] = mod_psub(x[17], y[17], RNS_MODULI[17]);
    #endif
    #if RNS_MODULI_SIZE > 18
        result[18] = mod_psub(x[18], y[18], RNS_MODULI[18]);
    #endif
    #if RNS_MODULI_SIZE > 19
        result[19] = mod_psub(x[19], y[19], RNS_MODULI[19]);
    #endif
    #if RNS_MODULI_SIZE > 20
        result[20] = mod_psub(x[20], y[20], RNS_MODULI[20]);
    #endif
    #if RNS_MODULI_SIZE > 21
        result[21] = mod_psub(x[21], y[21], RNS_MODULI[21]);
    #endif
    #if RNS_MODULI_SIZE > 22
        result[22] = mod_psub(x[22], y[22], RNS_MODULI[22]);
    #endif
    #if RNS_MODULI_SIZE > 23
        result[23] = mod_psub(x[23], y[23], RNS_MODULI[23]);
    #endif
    #if RNS_MODULI_SIZE > 24
        result[24] = mod_psub(x[24], y[24], RNS_MODULI[24]);
    #endif
    #if RNS_MODULI_SIZE > 25
        result[25] = mod_psub(x[25], y[25], RNS_MODULI[25]);
    #endif
    #if RNS_MODULI_SIZE > 26
        result[26] = mod_psub(x[26], y[26], RNS_MODULI[26]);
    #endif
    #if RNS_MODULI_SIZE > 27
        result[27] = mod_psub(x[27], y[27], RNS_MODULI[27]);
    #endif
    #if RNS_MODULI_SIZE > 28
        result[28] = mod_psub(x[28], y[28], RNS_MODULI[28]);
    #endif
    #if RNS_MODULI_SIZE > 29
        result[29] = mod_psub(x[29], y[29], RNS_MODULI[29]);
    #endif
    #if RNS_MODULI_SIZE > 30
        result[30] = mod_psub(x[30], y[30], RNS_MODULI[30]);
    #endif
    #if RNS_MODULI_SIZE > 31
        result[31] = mod_psub(x[31], y[31], RNS_MODULI[31]);
    #endif
    #if RNS_MODULI_SIZE > 32
        result[32] = mod_psub(x[32], y[32], RNS_MODULI[32]);
    #endif
    #if RNS_MODULI_SIZE > 33
        result[33] = mod_psub(x[33], y[33], RNS_MODULI[33]);
    #endif
    #if RNS_MODULI_SIZE > 34
        result[34] = mod_psub(x[34], y[34], RNS_MODULI[34]);
    #endif
    #if RNS_MODULI_SIZE > 35
        result[35] = mod_psub(x[35], y[35], RNS_MODULI[35]);
    #endif
    #if RNS_MODULI_SIZE > 36
        result[36] = mod_psub(x[36], y[36], RNS_MODULI[36]);
    #endif
    #if RNS_MODULI_SIZE > 37
        result[37] = mod_psub(x[37], y[37], RNS_MODULI[37]);
    #endif
    #if RNS_MODULI_SIZE > 38
        result[38] = mod_psub(x[38], y[38], RNS_MODULI[38]);
    #endif
    #if RNS_MODULI_SIZE > 39
        result[39] = mod_psub(x[39], y[39], RNS_MODULI[39]);
    #endif
    #if RNS_MODULI_SIZE > 40
        result[40] = mod_psub(x[40], y[40], RNS_MODULI[40]);
    #endif
    #if RNS_MODULI_SIZE > 41
        result[41] = mod_psub(x[41], y[41], RNS_MODULI[41]);
    #endif
    #if RNS_MODULI_SIZE > 42
        result[42] = mod_psub(x[42], y[42], RNS_MODULI[42]);
    #endif
    #if RNS_MODULI_SIZE > 43
        result[43] = mod_psub(x[43], y[43], RNS_MODULI[43]);
    #endif
    #if RNS_MODULI_SIZE > 44
        result[44] = mod_psub(x[44], y[44], RNS_MODULI[44]);
    #endif
    #if RNS_MODULI_SIZE > 45
        result[45] = mod_psub(x[45], y[45], RNS_MODULI[45]);
    #endif
    #if RNS_MODULI_SIZE > 46
        result[46] = mod_psub(x[46], y[46], RNS_MODULI[46]);
    #endif
    #if RNS_MODULI_SIZE > 47
        result[47] = mod_psub(x[47], y[47], RNS_MODULI[47]);
    #endif
    #if RNS_MODULI_SIZE > 48
        result[48] = mod_psub(x[48], y[48], RNS_MODULI[48]);
    #endif
    #if RNS_MODULI_SIZE > 49
        result[49] = mod_psub(x[49], y[49], RNS_MODULI[49]);
    #endif
    #if RNS_MODULI_SIZE > 50
        result[50] = mod_psub(x[50], y[50], RNS_MODULI[50]);
    #endif
    #if RNS_MODULI_SIZE > 51
        result[51] = mod_psub(x[51], y[51], RNS_MODULI[51]);
    #endif
    #if RNS_MODULI_SIZE > 52
        result[52] = mod_psub(x[52], y[52], RNS_MODULI[52]);
    #endif
    #if RNS_MODULI_SIZE > 53
        result[53] = mod_psub(x[53], y[53], RNS_MODULI[53]);
    #endif
    #if RNS_MODULI_SIZE > 54
        result[54] = mod_psub(x[54], y[54], RNS_MODULI[54]);
    #endif
    #if RNS_MODULI_SIZE > 55
        result[55] = mod_psub(x[55], y[55], RNS_MODULI[55]);
    #endif
    #if RNS_MODULI_SIZE > 56
        result[56] = mod_psub(x[56], y[56], RNS_MODULI[56]);
    #endif
    #if RNS_MODULI_SIZE > 57
        result[57] = mod_psub(x[57], y[57], RNS_MODULI[57]);
    #endif
    #if RNS_MODULI_SIZE > 58
        result[58] = mod_psub(x[58], y[58], RNS_MODULI[58]);
    #endif
    #if RNS_MODULI_SIZE > 59
        result[59] = mod_psub(x[59], y[59], RNS_MODULI[59]);
    #endif
    #if RNS_MODULI_SIZE > 60
        result[60] = mod_psub(x[60], y[60], RNS_MODULI[60]);
    #endif
    #if RNS_MODULI_SIZE > 61
        result[61] = mod_psub(x[61], y[61], RNS_MODULI[61]);
    #endif
    #if RNS_MODULI_SIZE > 62
        result[62] = mod_psub(x[62], y[62], RNS_MODULI[62]);
    #endif
    #if RNS_MODULI_SIZE > 63
        result[63] = mod_psub(x[63], y[63], RNS_MODULI[63]);
    #endif
    #if RNS_MODULI_SIZE > 64
        result[64] = mod_psub(x[64], y[64], RNS_MODULI[64]);
    #endif
    #if RNS_MODULI_SIZE > 65
        result[65] = mod_psub(x[65], y[65], RNS_MODULI[65]);
    #endif
    #if RNS_MODULI_SIZE > 66
        result[66] = mod_psub(x[66], y[66], RNS_MODULI[66]);
    #endif
    #if RNS_MODULI_SIZE > 67
        result[67] = mod_psub(x[67], y[67], RNS_MODULI[67]);
    #endif
    #if RNS_MODULI_SIZE > 68
        result[68] = mod_psub(x[68], y[68], RNS_MODULI[68]);
    #endif
    #if RNS_MODULI_SIZE > 69
        result[69] = mod_psub(x[69], y[69], RNS_MODULI[69]);
    #endif
    #if RNS_MODULI_SIZE > 70
        result[70] = mod_psub(x[70], y[70], RNS_MODULI[70]);
    #endif
    #if RNS_MODULI_SIZE > 71
        result[71] = mod_psub(x[71], y[71], RNS_MODULI[71]);
    #endif
    #if RNS_MODULI_SIZE > 72
        result[72] = mod_psub(x[72], y[72], RNS_MODULI[72]);
    #endif
    #if RNS_MODULI_SIZE > 73
        result[73] = mod_psub(x[73], y[73], RNS_MODULI[73]);
    #endif
    #if RNS_MODULI_SIZE > 74
        result[74] = mod_psub(x[74], y[74], RNS_MODULI[74]);
    #endif
    #if RNS_MODULI_SIZE > 75
        result[75] = mod_psub(x[75], y[75], RNS_MODULI[75]);
    #endif
    #if RNS_MODULI_SIZE > 76
        result[76] = mod_psub(x[76], y[76], RNS_MODULI[76]);
    #endif
    #if RNS_MODULI_SIZE > 77
        result[77] = mod_psub(x[77], y[77], RNS_MODULI[77]);
    #endif
    #if RNS_MODULI_SIZE > 78
        result[78] = mod_psub(x[78], y[78], RNS_MODULI[78]);
    #endif
    #if RNS_MODULI_SIZE > 79
        result[79] = mod_psub(x[79], y[79], RNS_MODULI[79]);
    #endif
    #if RNS_MODULI_SIZE > 80
        result[80] = mod_psub(x[80], y[80], RNS_MODULI[80]);
    #endif
    #if RNS_MODULI_SIZE > 81
        result[81] = mod_psub(x[81], y[81], RNS_MODULI[81]);
    #endif
    #if RNS_MODULI_SIZE > 82
        result[82] = mod_psub(x[82], y[82], RNS_MODULI[82]);
    #endif
    #if RNS_MODULI_SIZE > 83
        result[83] = mod_psub(x[83], y[83], RNS_MODULI[83]);
    #endif
    #if RNS_MODULI_SIZE > 84
        result[84] = mod_psub(x[84], y[84], RNS_MODULI[84]);
    #endif
    #if RNS_MODULI_SIZE > 85
        result[85] = mod_psub(x[85], y[85], RNS_MODULI[85]);
    #endif
    #if RNS_MODULI_SIZE > 86
        result[86] = mod_psub(x[86], y[86], RNS_MODULI[86]);
    #endif
    #if RNS_MODULI_SIZE > 87
        result[87] = mod_psub(x[87], y[87], RNS_MODULI[87]);
    #endif
    #if RNS_MODULI_SIZE > 88
        result[88] = mod_psub(x[88], y[88], RNS_MODULI[88]);
    #endif
    #if RNS_MODULI_SIZE > 89
        result[89] = mod_psub(x[89], y[89], RNS_MODULI[89]);
    #endif
    #if RNS_MODULI_SIZE > 90
        result[90] = mod_psub(x[90], y[90], RNS_MODULI[90]);
    #endif
    #if RNS_MODULI_SIZE > 91
        result[91] = mod_psub(x[91], y[91], RNS_MODULI[91]);
    #endif
    #if RNS_MODULI_SIZE > 92
        result[92] = mod_psub(x[92], y[92], RNS_MODULI[92]);
    #endif
    #if RNS_MODULI_SIZE > 93
        result[93] = mod_psub(x[93], y[93], RNS_MODULI[93]);
    #endif
    #if RNS_MODULI_SIZE > 94
        result[94] = mod_psub(x[94], y[94], RNS_MODULI[94]);
    #endif
    #if RNS_MODULI_SIZE > 95
        result[95] = mod_psub(x[95], y[95], RNS_MODULI[95]);
    #endif
    #if RNS_MODULI_SIZE > 96
        result[96] = mod_psub(x[96], y[96], RNS_MODULI[96]);
    #endif
    #if RNS_MODULI_SIZE > 97
        result[97] = mod_psub(x[97], y[97], RNS_MODULI[97]);
    #endif
    #if RNS_MODULI_SIZE > 98
        result[98] = mod_psub(x[98], y[98], RNS_MODULI[98]);
    #endif
    #if RNS_MODULI_SIZE > 99
        result[99] = mod_psub(x[99], y[99], RNS_MODULI[99]);
    #endif
    #if RNS_MODULI_SIZE > 100
        result[100] = mod_psub(x[100], y[100], RNS_MODULI[100]);
    #endif
    #if RNS_MODULI_SIZE > 101
        result[101] = mod_psub(x[101], y[101], RNS_MODULI[101]);
    #endif
    #if RNS_MODULI_SIZE > 102
        result[102] = mod_psub(x[102], y[102], RNS_MODULI[102]);
    #endif
    #if RNS_MODULI_SIZE > 103
        result[103] = mod_psub(x[103], y[103], RNS_MODULI[103]);
    #endif
    #if RNS_MODULI_SIZE > 104
        result[104] = mod_psub(x[104], y[104], RNS_MODULI[104]);
    #endif
    #if RNS_MODULI_SIZE > 105
        result[105] = mod_psub(x[105], y[105], RNS_MODULI[105]);
    #endif
    #if RNS_MODULI_SIZE > 106
        result[106] = mod_psub(x[106], y[106], RNS_MODULI[106]);
    #endif
    #if RNS_MODULI_SIZE > 107
        result[107] = mod_psub(x[107], y[107], RNS_MODULI[107]);
    #endif
    #if RNS_MODULI_SIZE > 108
        result[108] = mod_psub(x[108], y[108], RNS_MODULI[108]);
    #endif
    #if RNS_MODULI_SIZE > 109
        result[109] = mod_psub(x[109], y[109], RNS_MODULI[109]);
    #endif
    #if RNS_MODULI_SIZE > 110
        result[110] = mod_psub(x[110], y[110], RNS_MODULI[110]);
    #endif
    #if RNS_MODULI_SIZE > 111
        result[111] = mod_psub(x[111], y[111], RNS_MODULI[111]);
    #endif
    #if RNS_MODULI_SIZE > 112
        result[112] = mod_psub(x[112], y[112], RNS_MODULI[112]);
    #endif
    #if RNS_MODULI_SIZE > 113
        result[113] = mod_psub(x[113], y[113], RNS_MODULI[113]);
    #endif
    #if RNS_MODULI_SIZE > 114
        result[114] = mod_psub(x[114], y[114], RNS_MODULI[114]);
    #endif
    #if RNS_MODULI_SIZE > 115
        result[115] = mod_psub(x[115], y[115], RNS_MODULI[115]);
    #endif
    #if RNS_MODULI_SIZE > 116
        result[116] = mod_psub(x[116], y[116], RNS_MODULI[116]);
    #endif
    #if RNS_MODULI_SIZE > 117
        result[117] = mod_psub(x[117], y[117], RNS_MODULI[117]);
    #endif
    #if RNS_MODULI_SIZE > 118
        result[118] = mod_psub(x[118], y[118], RNS_MODULI[118]);
    #endif
    #if RNS_MODULI_SIZE > 119
        result[119] = mod_psub(x[119], y[119], RNS_MODULI[119]);
    #endif
    #if RNS_MODULI_SIZE > 120
        result[120] = mod_psub(x[120], y[120], RNS_MODULI[120]);
    #endif
    #if RNS_MODULI_SIZE > 121
        result[121] = mod_psub(x[121], y[121], RNS_MODULI[121]);
    #endif
    #if RNS_MODULI_SIZE > 122
        result[122] = mod_psub(x[122], y[122], RNS_MODULI[122]);
    #endif
    #if RNS_MODULI_SIZE > 123
        result[123] = mod_psub(x[123], y[123], RNS_MODULI[123]);
    #endif
    #if RNS_MODULI_SIZE > 124
        result[124] = mod_psub(x[124], y[124], RNS_MODULI[124]);
    #endif
    #if RNS_MODULI_SIZE > 125
        result[125] = mod_psub(x[125], y[125], RNS_MODULI[125]);
    #endif
    #if RNS_MODULI_SIZE > 126
        result[126] = mod_psub(x[126], y[126], RNS_MODULI[126]);
    #endif
    #if RNS_MODULI_SIZE > 127
        result[127] = mod_psub(x[127], y[127], RNS_MODULI[127]);
    #endif
    #if RNS_MODULI_SIZE > 128
        result[128] = mod_psub(x[128], y[128], RNS_MODULI[128]);
    #endif
    #if RNS_MODULI_SIZE > 129
        result[129] = mod_psub(x[129], y[129], RNS_MODULI[129]);
    #endif
    #if RNS_MODULI_SIZE > 130
        result[130] = mod_psub(x[130], y[130], RNS_MODULI[130]);
    #endif
    #if RNS_MODULI_SIZE > 131
        result[131] = mod_psub(x[131], y[131], RNS_MODULI[131]);
    #endif
    #if RNS_MODULI_SIZE > 132
        result[132] = mod_psub(x[132], y[132], RNS_MODULI[132]);
    #endif
    #if RNS_MODULI_SIZE > 133
        result[133] = mod_psub(x[133], y[133], RNS_MODULI[133]);
    #endif
    #if RNS_MODULI_SIZE > 134
        result[134] = mod_psub(x[134], y[134], RNS_MODULI[134]);
    #endif
    #if RNS_MODULI_SIZE > 135
        result[135] = mod_psub(x[135], y[135], RNS_MODULI[135]);
    #endif
    #if RNS_MODULI_SIZE > 136
        result[136] = mod_psub(x[136], y[136], RNS_MODULI[136]);
    #endif
    #if RNS_MODULI_SIZE > 137
        result[137] = mod_psub(x[137], y[137], RNS_MODULI[137]);
    #endif
    #if RNS_MODULI_SIZE > 138
        result[138] = mod_psub(x[138], y[138], RNS_MODULI[138]);
    #endif
    #if RNS_MODULI_SIZE > 139
        result[139] = mod_psub(x[139], y[139], RNS_MODULI[139]);
    #endif
    #if RNS_MODULI_SIZE > 140
        result[140] = mod_psub(x[140], y[140], RNS_MODULI[140]);
    #endif
    #if RNS_MODULI_SIZE > 141
        result[141] = mod_psub(x[141], y[141], RNS_MODULI[141]);
    #endif
    #if RNS_MODULI_SIZE > 142
        result[142] = mod_psub(x[142], y[142], RNS_MODULI[142]);
    #endif
    #if RNS_MODULI_SIZE > 143
        result[143] = mod_psub(x[143], y[143], RNS_MODULI[143]);
    #endif
    #if RNS_MODULI_SIZE > 144
        result[144] = mod_psub(x[144], y[144], RNS_MODULI[144]);
    #endif
    #if RNS_MODULI_SIZE > 145
        result[145] = mod_psub(x[145], y[145], RNS_MODULI[145]);
    #endif
    #if RNS_MODULI_SIZE > 146
        result[146] = mod_psub(x[146], y[146], RNS_MODULI[146]);
    #endif
    #if RNS_MODULI_SIZE > 147
        result[147] = mod_psub(x[147], y[147], RNS_MODULI[147]);
    #endif
    #if RNS_MODULI_SIZE > 148
        result[148] = mod_psub(x[148], y[148], RNS_MODULI[148]);
    #endif
    #if RNS_MODULI_SIZE > 149
        result[149] = mod_psub(x[149], y[149], RNS_MODULI[149]);
    #endif
    #if RNS_MODULI_SIZE > 150
        result[150] = mod_psub(x[150], y[150], RNS_MODULI[150]);
    #endif
    #if RNS_MODULI_SIZE > 151
        result[151] = mod_psub(x[151], y[151], RNS_MODULI[151]);
    #endif
    #if RNS_MODULI_SIZE > 152
        result[152] = mod_psub(x[152], y[152], RNS_MODULI[152]);
    #endif
    #if RNS_MODULI_SIZE > 153
        result[153] = mod_psub(x[153], y[153], RNS_MODULI[153]);
    #endif
    #if RNS_MODULI_SIZE > 154
        result[154] = mod_psub(x[154], y[154], RNS_MODULI[154]);
    #endif
    #if RNS_MODULI_SIZE > 155
        result[155] = mod_psub(x[155], y[155], RNS_MODULI[155]);
    #endif
    #if RNS_MODULI_SIZE > 156
        result[156] = mod_psub(x[156], y[156], RNS_MODULI[156]);
    #endif
    #if RNS_MODULI_SIZE > 157
        result[157] = mod_psub(x[157], y[157], RNS_MODULI[157]);
    #endif
    #if RNS_MODULI_SIZE > 158
        result[158] = mod_psub(x[158], y[158], RNS_MODULI[158]);
    #endif
    #if RNS_MODULI_SIZE > 159
        result[159] = mod_psub(x[159], y[159], RNS_MODULI[159]);
    #endif
    #if RNS_MODULI_SIZE > 160
        result[160] = mod_psub(x[160], y[160], RNS_MODULI[160]);
    #endif
    #if RNS_MODULI_SIZE > 161
        result[161] = mod_psub(x[161], y[161], RNS_MODULI[161]);
    #endif
    #if RNS_MODULI_SIZE > 162
        result[162] = mod_psub(x[162], y[162], RNS_MODULI[162]);
    #endif
    #if RNS_MODULI_SIZE > 163
        result[163] = mod_psub(x[163], y[163], RNS_MODULI[163]);
    #endif
    #if RNS_MODULI_SIZE > 164
        result[164] = mod_psub(x[164], y[164], RNS_MODULI[164]);
    #endif
    #if RNS_MODULI_SIZE > 165
        result[165] = mod_psub(x[165], y[165], RNS_MODULI[165]);
    #endif
    #if RNS_MODULI_SIZE > 166
        result[166] = mod_psub(x[166], y[166], RNS_MODULI[166]);
    #endif
    #if RNS_MODULI_SIZE > 167
        result[167] = mod_psub(x[167], y[167], RNS_MODULI[167]);
    #endif
    #if RNS_MODULI_SIZE > 168
        result[168] = mod_psub(x[168], y[168], RNS_MODULI[168]);
    #endif
    #if RNS_MODULI_SIZE > 169
        result[169] = mod_psub(x[169], y[169], RNS_MODULI[169]);
    #endif
    #if RNS_MODULI_SIZE > 170
        result[170] = mod_psub(x[170], y[170], RNS_MODULI[170]);
    #endif
    #if RNS_MODULI_SIZE > 171
        result[171] = mod_psub(x[171], y[171], RNS_MODULI[171]);
    #endif
    #if RNS_MODULI_SIZE > 172
        result[172] = mod_psub(x[172], y[172], RNS_MODULI[172]);
    #endif
    #if RNS_MODULI_SIZE > 173
        result[173] = mod_psub(x[173], y[173], RNS_MODULI[173]);
    #endif
    #if RNS_MODULI_SIZE > 174
        result[174] = mod_psub(x[174], y[174], RNS_MODULI[174]);
    #endif
    #if RNS_MODULI_SIZE > 175
        result[175] = mod_psub(x[175], y[175], RNS_MODULI[175]);
    #endif
    #if RNS_MODULI_SIZE > 176
        result[176] = mod_psub(x[176], y[176], RNS_MODULI[176]);
    #endif
    #if RNS_MODULI_SIZE > 177
        result[177] = mod_psub(x[177], y[177], RNS_MODULI[177]);
    #endif
    #if RNS_MODULI_SIZE > 178
        result[178] = mod_psub(x[178], y[178], RNS_MODULI[178]);
    #endif
    #if RNS_MODULI_SIZE > 179
        result[179] = mod_psub(x[179], y[179], RNS_MODULI[179]);
    #endif
    #if RNS_MODULI_SIZE > 180
        result[180] = mod_psub(x[180], y[180], RNS_MODULI[180]);
    #endif
    #if RNS_MODULI_SIZE > 181
        result[181] = mod_psub(x[181], y[181], RNS_MODULI[181]);
    #endif
    #if RNS_MODULI_SIZE > 182
        result[182] = mod_psub(x[182], y[182], RNS_MODULI[182]);
    #endif
    #if RNS_MODULI_SIZE > 183
        result[183] = mod_psub(x[183], y[183], RNS_MODULI[183]);
    #endif
    #if RNS_MODULI_SIZE > 184
        result[184] = mod_psub(x[184], y[184], RNS_MODULI[184]);
    #endif
    #if RNS_MODULI_SIZE > 185
        result[185] = mod_psub(x[185], y[185], RNS_MODULI[185]);
    #endif
    #if RNS_MODULI_SIZE > 186
        result[186] = mod_psub(x[186], y[186], RNS_MODULI[186]);
    #endif
    #if RNS_MODULI_SIZE > 187
        result[187] = mod_psub(x[187], y[187], RNS_MODULI[187]);
    #endif
    #if RNS_MODULI_SIZE > 188
        result[188] = mod_psub(x[188], y[188], RNS_MODULI[188]);
    #endif
    #if RNS_MODULI_SIZE > 189
        result[189] = mod_psub(x[189], y[189], RNS_MODULI[189]);
    #endif
    #if RNS_MODULI_SIZE > 190
        result[190] = mod_psub(x[190], y[190], RNS_MODULI[190]);
    #endif
    #if RNS_MODULI_SIZE > 191
        result[191] = mod_psub(x[191], y[191], RNS_MODULI[191]);
    #endif
    #if RNS_MODULI_SIZE > 192
        result[192] = mod_psub(x[192], y[192], RNS_MODULI[192]);
    #endif
    #if RNS_MODULI_SIZE > 193
        result[193] = mod_psub(x[193], y[193], RNS_MODULI[193]);
    #endif
    #if RNS_MODULI_SIZE > 194
        result[194] = mod_psub(x[194], y[194], RNS_MODULI[194]);
    #endif
    #if RNS_MODULI_SIZE > 195
        result[195] = mod_psub(x[195], y[195], RNS_MODULI[195]);
    #endif
    #if RNS_MODULI_SIZE > 196
        result[196] = mod_psub(x[196], y[196], RNS_MODULI[196]);
    #endif
    #if RNS_MODULI_SIZE > 197
        result[197] = mod_psub(x[197], y[197], RNS_MODULI[197]);
    #endif
    #if RNS_MODULI_SIZE > 198
        result[198] = mod_psub(x[198], y[198], RNS_MODULI[198]);
    #endif
    #if RNS_MODULI_SIZE > 199
        result[199] = mod_psub(x[199], y[199], RNS_MODULI[199]);
    #endif
}

/*
 * GPU functions
 */
namespace cuda {

    /*!
     * Unrolled multiplication of two RNS numbers.
     */
    DEVICE_CUDA_FORCEINLINE void rns_mul(int * result, int * x, int * y){
        #if RNS_MODULI_SIZE > 0
                result[0] = cuda::mod_mul(x[0], y[0], cuda::RNS_MODULI[0]);
        #endif
        #if RNS_MODULI_SIZE > 1
                result[1] = cuda::mod_mul(x[1], y[1], cuda::RNS_MODULI[1]);
        #endif
        #if RNS_MODULI_SIZE > 2
                result[2] = cuda::mod_mul(x[2], y[2], cuda::RNS_MODULI[2]);
        #endif
        #if RNS_MODULI_SIZE > 3
                result[3] = cuda::mod_mul(x[3], y[3], cuda::RNS_MODULI[3]);
        #endif
        #if RNS_MODULI_SIZE > 4
                result[4] = cuda::mod_mul(x[4], y[4], cuda::RNS_MODULI[4]);
        #endif
        #if RNS_MODULI_SIZE > 5
                result[5] = cuda::mod_mul(x[5], y[5], cuda::RNS_MODULI[5]);
        #endif
        #if RNS_MODULI_SIZE > 6
                result[6] = cuda::mod_mul(x[6], y[6], cuda::RNS_MODULI[6]);
        #endif
        #if RNS_MODULI_SIZE > 7
                result[7] = cuda::mod_mul(x[7], y[7], cuda::RNS_MODULI[7]);
        #endif
        #if RNS_MODULI_SIZE > 8
                result[8] = cuda::mod_mul(x[8], y[8], cuda::RNS_MODULI[8]);
        #endif
        #if RNS_MODULI_SIZE > 9
                result[9] = cuda::mod_mul(x[9], y[9], cuda::RNS_MODULI[9]);
        #endif
        #if RNS_MODULI_SIZE > 10
                result[10] = cuda::mod_mul(x[10], y[10], cuda::RNS_MODULI[10]);
        #endif
        #if RNS_MODULI_SIZE > 11
                result[11] = cuda::mod_mul(x[11], y[11], cuda::RNS_MODULI[11]);
        #endif
        #if RNS_MODULI_SIZE > 12
                result[12] = cuda::mod_mul(x[12], y[12], cuda::RNS_MODULI[12]);
        #endif
        #if RNS_MODULI_SIZE > 13
                result[13] = cuda::mod_mul(x[13], y[13], cuda::RNS_MODULI[13]);
        #endif
        #if RNS_MODULI_SIZE > 14
                result[14] = cuda::mod_mul(x[14], y[14], cuda::RNS_MODULI[14]);
        #endif
        #if RNS_MODULI_SIZE > 15
                result[15] = cuda::mod_mul(x[15], y[15], cuda::RNS_MODULI[15]);
        #endif
        #if RNS_MODULI_SIZE > 16
                result[16] = cuda::mod_mul(x[16], y[16], cuda::RNS_MODULI[16]);
        #endif
        #if RNS_MODULI_SIZE > 17
                result[17] = cuda::mod_mul(x[17], y[17], cuda::RNS_MODULI[17]);
        #endif
        #if RNS_MODULI_SIZE > 18
                result[18] = cuda::mod_mul(x[18], y[18], cuda::RNS_MODULI[18]);
        #endif
        #if RNS_MODULI_SIZE > 19
                result[19] = cuda::mod_mul(x[19], y[19], cuda::RNS_MODULI[19]);
        #endif
        #if RNS_MODULI_SIZE > 20
                result[20] = cuda::mod_mul(x[20], y[20], cuda::RNS_MODULI[20]);
        #endif
        #if RNS_MODULI_SIZE > 21
                result[21] = cuda::mod_mul(x[21], y[21], cuda::RNS_MODULI[21]);
        #endif
        #if RNS_MODULI_SIZE > 22
                result[22] = cuda::mod_mul(x[22], y[22], cuda::RNS_MODULI[22]);
        #endif
        #if RNS_MODULI_SIZE > 23
                result[23] = cuda::mod_mul(x[23], y[23], cuda::RNS_MODULI[23]);
        #endif
        #if RNS_MODULI_SIZE > 24
                result[24] = cuda::mod_mul(x[24], y[24], cuda::RNS_MODULI[24]);
        #endif
        #if RNS_MODULI_SIZE > 25
                result[25] = cuda::mod_mul(x[25], y[25], cuda::RNS_MODULI[25]);
        #endif
        #if RNS_MODULI_SIZE > 26
                result[26] = cuda::mod_mul(x[26], y[26], cuda::RNS_MODULI[26]);
        #endif
        #if RNS_MODULI_SIZE > 27
                result[27] = cuda::mod_mul(x[27], y[27], cuda::RNS_MODULI[27]);
        #endif
        #if RNS_MODULI_SIZE > 28
                result[28] = cuda::mod_mul(x[28], y[28], cuda::RNS_MODULI[28]);
        #endif
        #if RNS_MODULI_SIZE > 29
                result[29] = cuda::mod_mul(x[29], y[29], cuda::RNS_MODULI[29]);
        #endif
        #if RNS_MODULI_SIZE > 30
                result[30] = cuda::mod_mul(x[30], y[30], cuda::RNS_MODULI[30]);
        #endif
        #if RNS_MODULI_SIZE > 31
                result[31] = cuda::mod_mul(x[31], y[31], cuda::RNS_MODULI[31]);
        #endif
        #if RNS_MODULI_SIZE > 32
                result[32] = cuda::mod_mul(x[32], y[32], cuda::RNS_MODULI[32]);
        #endif
        #if RNS_MODULI_SIZE > 33
                result[33] = cuda::mod_mul(x[33], y[33], cuda::RNS_MODULI[33]);
        #endif
        #if RNS_MODULI_SIZE > 34
                result[34] = cuda::mod_mul(x[34], y[34], cuda::RNS_MODULI[34]);
        #endif
        #if RNS_MODULI_SIZE > 35
                result[35] = cuda::mod_mul(x[35], y[35], cuda::RNS_MODULI[35]);
        #endif
        #if RNS_MODULI_SIZE > 36
                result[36] = cuda::mod_mul(x[36], y[36], cuda::RNS_MODULI[36]);
        #endif
        #if RNS_MODULI_SIZE > 37
                result[37] = cuda::mod_mul(x[37], y[37], cuda::RNS_MODULI[37]);
        #endif
        #if RNS_MODULI_SIZE > 38
                result[38] = cuda::mod_mul(x[38], y[38], cuda::RNS_MODULI[38]);
        #endif
        #if RNS_MODULI_SIZE > 39
                result[39] = cuda::mod_mul(x[39], y[39], cuda::RNS_MODULI[39]);
        #endif
        #if RNS_MODULI_SIZE > 40
                result[40] = cuda::mod_mul(x[40], y[40], cuda::RNS_MODULI[40]);
        #endif
        #if RNS_MODULI_SIZE > 41
                result[41] = cuda::mod_mul(x[41], y[41], cuda::RNS_MODULI[41]);
        #endif
        #if RNS_MODULI_SIZE > 42
                result[42] = cuda::mod_mul(x[42], y[42], cuda::RNS_MODULI[42]);
        #endif
        #if RNS_MODULI_SIZE > 43
                result[43] = cuda::mod_mul(x[43], y[43], cuda::RNS_MODULI[43]);
        #endif
        #if RNS_MODULI_SIZE > 44
                result[44] = cuda::mod_mul(x[44], y[44], cuda::RNS_MODULI[44]);
        #endif
        #if RNS_MODULI_SIZE > 45
                result[45] = cuda::mod_mul(x[45], y[45], cuda::RNS_MODULI[45]);
        #endif
        #if RNS_MODULI_SIZE > 46
                result[46] = cuda::mod_mul(x[46], y[46], cuda::RNS_MODULI[46]);
        #endif
        #if RNS_MODULI_SIZE > 47
                result[47] = cuda::mod_mul(x[47], y[47], cuda::RNS_MODULI[47]);
        #endif
        #if RNS_MODULI_SIZE > 48
                result[48] = cuda::mod_mul(x[48], y[48], cuda::RNS_MODULI[48]);
        #endif
        #if RNS_MODULI_SIZE > 49
                result[49] = cuda::mod_mul(x[49], y[49], cuda::RNS_MODULI[49]);
        #endif
        #if RNS_MODULI_SIZE > 50
                result[50] = cuda::mod_mul(x[50], y[50], cuda::RNS_MODULI[50]);
        #endif
        #if RNS_MODULI_SIZE > 51
                result[51] = cuda::mod_mul(x[51], y[51], cuda::RNS_MODULI[51]);
        #endif
        #if RNS_MODULI_SIZE > 52
                result[52] = cuda::mod_mul(x[52], y[52], cuda::RNS_MODULI[52]);
        #endif
        #if RNS_MODULI_SIZE > 53
                result[53] = cuda::mod_mul(x[53], y[53], cuda::RNS_MODULI[53]);
        #endif
        #if RNS_MODULI_SIZE > 54
                result[54] = cuda::mod_mul(x[54], y[54], cuda::RNS_MODULI[54]);
        #endif
        #if RNS_MODULI_SIZE > 55
                result[55] = cuda::mod_mul(x[55], y[55], cuda::RNS_MODULI[55]);
        #endif
        #if RNS_MODULI_SIZE > 56
                result[56] = cuda::mod_mul(x[56], y[56], cuda::RNS_MODULI[56]);
        #endif
        #if RNS_MODULI_SIZE > 57
                result[57] = cuda::mod_mul(x[57], y[57], cuda::RNS_MODULI[57]);
        #endif
        #if RNS_MODULI_SIZE > 58
                result[58] = cuda::mod_mul(x[58], y[58], cuda::RNS_MODULI[58]);
        #endif
        #if RNS_MODULI_SIZE > 59
                result[59] = cuda::mod_mul(x[59], y[59], cuda::RNS_MODULI[59]);
        #endif
        #if RNS_MODULI_SIZE > 60
                result[60] = cuda::mod_mul(x[60], y[60], cuda::RNS_MODULI[60]);
        #endif
        #if RNS_MODULI_SIZE > 61
                result[61] = cuda::mod_mul(x[61], y[61], cuda::RNS_MODULI[61]);
        #endif
        #if RNS_MODULI_SIZE > 62
                result[62] = cuda::mod_mul(x[62], y[62], cuda::RNS_MODULI[62]);
        #endif
        #if RNS_MODULI_SIZE > 63
                result[63] = cuda::mod_mul(x[63], y[63], cuda::RNS_MODULI[63]);
        #endif
        #if RNS_MODULI_SIZE > 64
                result[64] = cuda::mod_mul(x[64], y[64], cuda::RNS_MODULI[64]);
        #endif
        #if RNS_MODULI_SIZE > 65
                result[65] = cuda::mod_mul(x[65], y[65], cuda::RNS_MODULI[65]);
        #endif
        #if RNS_MODULI_SIZE > 66
                result[66] = cuda::mod_mul(x[66], y[66], cuda::RNS_MODULI[66]);
        #endif
        #if RNS_MODULI_SIZE > 67
                result[67] = cuda::mod_mul(x[67], y[67], cuda::RNS_MODULI[67]);
        #endif
        #if RNS_MODULI_SIZE > 68
                result[68] = cuda::mod_mul(x[68], y[68], cuda::RNS_MODULI[68]);
        #endif
        #if RNS_MODULI_SIZE > 69
                result[69] = cuda::mod_mul(x[69], y[69], cuda::RNS_MODULI[69]);
        #endif
        #if RNS_MODULI_SIZE > 70
                result[70] = cuda::mod_mul(x[70], y[70], cuda::RNS_MODULI[70]);
        #endif
        #if RNS_MODULI_SIZE > 71
                result[71] = cuda::mod_mul(x[71], y[71], cuda::RNS_MODULI[71]);
        #endif
        #if RNS_MODULI_SIZE > 72
                result[72] = cuda::mod_mul(x[72], y[72], cuda::RNS_MODULI[72]);
        #endif
        #if RNS_MODULI_SIZE > 73
                result[73] = cuda::mod_mul(x[73], y[73], cuda::RNS_MODULI[73]);
        #endif
        #if RNS_MODULI_SIZE > 74
                result[74] = cuda::mod_mul(x[74], y[74], cuda::RNS_MODULI[74]);
        #endif
        #if RNS_MODULI_SIZE > 75
                result[75] = cuda::mod_mul(x[75], y[75], cuda::RNS_MODULI[75]);
        #endif
        #if RNS_MODULI_SIZE > 76
                result[76] = cuda::mod_mul(x[76], y[76], cuda::RNS_MODULI[76]);
        #endif
        #if RNS_MODULI_SIZE > 77
                result[77] = cuda::mod_mul(x[77], y[77], cuda::RNS_MODULI[77]);
        #endif
        #if RNS_MODULI_SIZE > 78
                result[78] = cuda::mod_mul(x[78], y[78], cuda::RNS_MODULI[78]);
        #endif
        #if RNS_MODULI_SIZE > 79
                result[79] = cuda::mod_mul(x[79], y[79], cuda::RNS_MODULI[79]);
        #endif
        #if RNS_MODULI_SIZE > 80
                result[80] = cuda::mod_mul(x[80], y[80], cuda::RNS_MODULI[80]);
        #endif
        #if RNS_MODULI_SIZE > 81
                result[81] = cuda::mod_mul(x[81], y[81], cuda::RNS_MODULI[81]);
        #endif
        #if RNS_MODULI_SIZE > 82
                result[82] = cuda::mod_mul(x[82], y[82], cuda::RNS_MODULI[82]);
        #endif
        #if RNS_MODULI_SIZE > 83
                result[83] = cuda::mod_mul(x[83], y[83], cuda::RNS_MODULI[83]);
        #endif
        #if RNS_MODULI_SIZE > 84
                result[84] = cuda::mod_mul(x[84], y[84], cuda::RNS_MODULI[84]);
        #endif
        #if RNS_MODULI_SIZE > 85
                result[85] = cuda::mod_mul(x[85], y[85], cuda::RNS_MODULI[85]);
        #endif
        #if RNS_MODULI_SIZE > 86
                result[86] = cuda::mod_mul(x[86], y[86], cuda::RNS_MODULI[86]);
        #endif
        #if RNS_MODULI_SIZE > 87
                result[87] = cuda::mod_mul(x[87], y[87], cuda::RNS_MODULI[87]);
        #endif
        #if RNS_MODULI_SIZE > 88
                result[88] = cuda::mod_mul(x[88], y[88], cuda::RNS_MODULI[88]);
        #endif
        #if RNS_MODULI_SIZE > 89
                result[89] = cuda::mod_mul(x[89], y[89], cuda::RNS_MODULI[89]);
        #endif
        #if RNS_MODULI_SIZE > 90
                result[90] = cuda::mod_mul(x[90], y[90], cuda::RNS_MODULI[90]);
        #endif
        #if RNS_MODULI_SIZE > 91
                result[91] = cuda::mod_mul(x[91], y[91], cuda::RNS_MODULI[91]);
        #endif
        #if RNS_MODULI_SIZE > 92
                result[92] = cuda::mod_mul(x[92], y[92], cuda::RNS_MODULI[92]);
        #endif
        #if RNS_MODULI_SIZE > 93
                result[93] = cuda::mod_mul(x[93], y[93], cuda::RNS_MODULI[93]);
        #endif
        #if RNS_MODULI_SIZE > 94
                result[94] = cuda::mod_mul(x[94], y[94], cuda::RNS_MODULI[94]);
        #endif
        #if RNS_MODULI_SIZE > 95
                result[95] = cuda::mod_mul(x[95], y[95], cuda::RNS_MODULI[95]);
        #endif
        #if RNS_MODULI_SIZE > 96
                result[96] = cuda::mod_mul(x[96], y[96], cuda::RNS_MODULI[96]);
        #endif
        #if RNS_MODULI_SIZE > 97
                result[97] = cuda::mod_mul(x[97], y[97], cuda::RNS_MODULI[97]);
        #endif
        #if RNS_MODULI_SIZE > 98
                result[98] = cuda::mod_mul(x[98], y[98], cuda::RNS_MODULI[98]);
        #endif
        #if RNS_MODULI_SIZE > 99
                result[99] = cuda::mod_mul(x[99], y[99], cuda::RNS_MODULI[99]);
        #endif
        #if RNS_MODULI_SIZE > 100
                result[100] = cuda::mod_mul(x[100], y[100], cuda::RNS_MODULI[100]);
        #endif
        #if RNS_MODULI_SIZE > 101
                result[101] = cuda::mod_mul(x[101], y[101], cuda::RNS_MODULI[101]);
        #endif
        #if RNS_MODULI_SIZE > 102
                result[102] = cuda::mod_mul(x[102], y[102], cuda::RNS_MODULI[102]);
        #endif
        #if RNS_MODULI_SIZE > 103
                result[103] = cuda::mod_mul(x[103], y[103], cuda::RNS_MODULI[103]);
        #endif
        #if RNS_MODULI_SIZE > 104
                result[104] = cuda::mod_mul(x[104], y[104], cuda::RNS_MODULI[104]);
        #endif
        #if RNS_MODULI_SIZE > 105
                result[105] = cuda::mod_mul(x[105], y[105], cuda::RNS_MODULI[105]);
        #endif
        #if RNS_MODULI_SIZE > 106
                result[106] = cuda::mod_mul(x[106], y[106], cuda::RNS_MODULI[106]);
        #endif
        #if RNS_MODULI_SIZE > 107
                result[107] = cuda::mod_mul(x[107], y[107], cuda::RNS_MODULI[107]);
        #endif
        #if RNS_MODULI_SIZE > 108
                result[108] = cuda::mod_mul(x[108], y[108], cuda::RNS_MODULI[108]);
        #endif
        #if RNS_MODULI_SIZE > 109
                result[109] = cuda::mod_mul(x[109], y[109], cuda::RNS_MODULI[109]);
        #endif
        #if RNS_MODULI_SIZE > 110
                result[110] = cuda::mod_mul(x[110], y[110], cuda::RNS_MODULI[110]);
        #endif
        #if RNS_MODULI_SIZE > 111
                result[111] = cuda::mod_mul(x[111], y[111], cuda::RNS_MODULI[111]);
        #endif
        #if RNS_MODULI_SIZE > 112
                result[112] = cuda::mod_mul(x[112], y[112], cuda::RNS_MODULI[112]);
        #endif
        #if RNS_MODULI_SIZE > 113
                result[113] = cuda::mod_mul(x[113], y[113], cuda::RNS_MODULI[113]);
        #endif
        #if RNS_MODULI_SIZE > 114
                result[114] = cuda::mod_mul(x[114], y[114], cuda::RNS_MODULI[114]);
        #endif
        #if RNS_MODULI_SIZE > 115
                result[115] = cuda::mod_mul(x[115], y[115], cuda::RNS_MODULI[115]);
        #endif
        #if RNS_MODULI_SIZE > 116
                result[116] = cuda::mod_mul(x[116], y[116], cuda::RNS_MODULI[116]);
        #endif
        #if RNS_MODULI_SIZE > 117
                result[117] = cuda::mod_mul(x[117], y[117], cuda::RNS_MODULI[117]);
        #endif
        #if RNS_MODULI_SIZE > 118
                result[118] = cuda::mod_mul(x[118], y[118], cuda::RNS_MODULI[118]);
        #endif
        #if RNS_MODULI_SIZE > 119
                result[119] = cuda::mod_mul(x[119], y[119], cuda::RNS_MODULI[119]);
        #endif
        #if RNS_MODULI_SIZE > 120
                result[120] = cuda::mod_mul(x[120], y[120], cuda::RNS_MODULI[120]);
        #endif
        #if RNS_MODULI_SIZE > 121
                result[121] = cuda::mod_mul(x[121], y[121], cuda::RNS_MODULI[121]);
        #endif
        #if RNS_MODULI_SIZE > 122
                result[122] = cuda::mod_mul(x[122], y[122], cuda::RNS_MODULI[122]);
        #endif
        #if RNS_MODULI_SIZE > 123
                result[123] = cuda::mod_mul(x[123], y[123], cuda::RNS_MODULI[123]);
        #endif
        #if RNS_MODULI_SIZE > 124
                result[124] = cuda::mod_mul(x[124], y[124], cuda::RNS_MODULI[124]);
        #endif
        #if RNS_MODULI_SIZE > 125
                result[125] = cuda::mod_mul(x[125], y[125], cuda::RNS_MODULI[125]);
        #endif
        #if RNS_MODULI_SIZE > 126
                result[126] = cuda::mod_mul(x[126], y[126], cuda::RNS_MODULI[126]);
        #endif
        #if RNS_MODULI_SIZE > 127
                result[127] = cuda::mod_mul(x[127], y[127], cuda::RNS_MODULI[127]);
        #endif
        #if RNS_MODULI_SIZE > 128
                result[128] = cuda::mod_mul(x[128], y[128], cuda::RNS_MODULI[128]);
        #endif
        #if RNS_MODULI_SIZE > 129
                result[129] = cuda::mod_mul(x[129], y[129], cuda::RNS_MODULI[129]);
        #endif
        #if RNS_MODULI_SIZE > 130
                result[130] = cuda::mod_mul(x[130], y[130], cuda::RNS_MODULI[130]);
        #endif
        #if RNS_MODULI_SIZE > 131
                result[131] = cuda::mod_mul(x[131], y[131], cuda::RNS_MODULI[131]);
        #endif
        #if RNS_MODULI_SIZE > 132
                result[132] = cuda::mod_mul(x[132], y[132], cuda::RNS_MODULI[132]);
        #endif
        #if RNS_MODULI_SIZE > 133
                result[133] = cuda::mod_mul(x[133], y[133], cuda::RNS_MODULI[133]);
        #endif
        #if RNS_MODULI_SIZE > 134
                result[134] = cuda::mod_mul(x[134], y[134], cuda::RNS_MODULI[134]);
        #endif
        #if RNS_MODULI_SIZE > 135
                result[135] = cuda::mod_mul(x[135], y[135], cuda::RNS_MODULI[135]);
        #endif
        #if RNS_MODULI_SIZE > 136
                result[136] = cuda::mod_mul(x[136], y[136], cuda::RNS_MODULI[136]);
        #endif
        #if RNS_MODULI_SIZE > 137
                result[137] = cuda::mod_mul(x[137], y[137], cuda::RNS_MODULI[137]);
        #endif
        #if RNS_MODULI_SIZE > 138
                result[138] = cuda::mod_mul(x[138], y[138], cuda::RNS_MODULI[138]);
        #endif
        #if RNS_MODULI_SIZE > 139
                result[139] = cuda::mod_mul(x[139], y[139], cuda::RNS_MODULI[139]);
        #endif
        #if RNS_MODULI_SIZE > 140
                result[140] = cuda::mod_mul(x[140], y[140], cuda::RNS_MODULI[140]);
        #endif
        #if RNS_MODULI_SIZE > 141
                result[141] = cuda::mod_mul(x[141], y[141], cuda::RNS_MODULI[141]);
        #endif
        #if RNS_MODULI_SIZE > 142
                result[142] = cuda::mod_mul(x[142], y[142], cuda::RNS_MODULI[142]);
        #endif
        #if RNS_MODULI_SIZE > 143
                result[143] = cuda::mod_mul(x[143], y[143], cuda::RNS_MODULI[143]);
        #endif
        #if RNS_MODULI_SIZE > 144
                result[144] = cuda::mod_mul(x[144], y[144], cuda::RNS_MODULI[144]);
        #endif
        #if RNS_MODULI_SIZE > 145
                result[145] = cuda::mod_mul(x[145], y[145], cuda::RNS_MODULI[145]);
        #endif
        #if RNS_MODULI_SIZE > 146
                result[146] = cuda::mod_mul(x[146], y[146], cuda::RNS_MODULI[146]);
        #endif
        #if RNS_MODULI_SIZE > 147
                result[147] = cuda::mod_mul(x[147], y[147], cuda::RNS_MODULI[147]);
        #endif
        #if RNS_MODULI_SIZE > 148
                result[148] = cuda::mod_mul(x[148], y[148], cuda::RNS_MODULI[148]);
        #endif
        #if RNS_MODULI_SIZE > 149
                result[149] = cuda::mod_mul(x[149], y[149], cuda::RNS_MODULI[149]);
        #endif
        #if RNS_MODULI_SIZE > 150
                result[150] = cuda::mod_mul(x[150], y[150], cuda::RNS_MODULI[150]);
        #endif
        #if RNS_MODULI_SIZE > 151
                result[151] = cuda::mod_mul(x[151], y[151], cuda::RNS_MODULI[151]);
        #endif
        #if RNS_MODULI_SIZE > 152
                result[152] = cuda::mod_mul(x[152], y[152], cuda::RNS_MODULI[152]);
        #endif
        #if RNS_MODULI_SIZE > 153
                result[153] = cuda::mod_mul(x[153], y[153], cuda::RNS_MODULI[153]);
        #endif
        #if RNS_MODULI_SIZE > 154
                result[154] = cuda::mod_mul(x[154], y[154], cuda::RNS_MODULI[154]);
        #endif
        #if RNS_MODULI_SIZE > 155
                result[155] = cuda::mod_mul(x[155], y[155], cuda::RNS_MODULI[155]);
        #endif
        #if RNS_MODULI_SIZE > 156
                result[156] = cuda::mod_mul(x[156], y[156], cuda::RNS_MODULI[156]);
        #endif
        #if RNS_MODULI_SIZE > 157
                result[157] = cuda::mod_mul(x[157], y[157], cuda::RNS_MODULI[157]);
        #endif
        #if RNS_MODULI_SIZE > 158
                result[158] = cuda::mod_mul(x[158], y[158], cuda::RNS_MODULI[158]);
        #endif
        #if RNS_MODULI_SIZE > 159
                result[159] = cuda::mod_mul(x[159], y[159], cuda::RNS_MODULI[159]);
        #endif
        #if RNS_MODULI_SIZE > 160
                result[160] = cuda::mod_mul(x[160], y[160], cuda::RNS_MODULI[160]);
        #endif
        #if RNS_MODULI_SIZE > 161
                result[161] = cuda::mod_mul(x[161], y[161], cuda::RNS_MODULI[161]);
        #endif
        #if RNS_MODULI_SIZE > 162
                result[162] = cuda::mod_mul(x[162], y[162], cuda::RNS_MODULI[162]);
        #endif
        #if RNS_MODULI_SIZE > 163
                result[163] = cuda::mod_mul(x[163], y[163], cuda::RNS_MODULI[163]);
        #endif
        #if RNS_MODULI_SIZE > 164
                result[164] = cuda::mod_mul(x[164], y[164], cuda::RNS_MODULI[164]);
        #endif
        #if RNS_MODULI_SIZE > 165
                result[165] = cuda::mod_mul(x[165], y[165], cuda::RNS_MODULI[165]);
        #endif
        #if RNS_MODULI_SIZE > 166
                result[166] = cuda::mod_mul(x[166], y[166], cuda::RNS_MODULI[166]);
        #endif
        #if RNS_MODULI_SIZE > 167
                result[167] = cuda::mod_mul(x[167], y[167], cuda::RNS_MODULI[167]);
        #endif
        #if RNS_MODULI_SIZE > 168
                result[168] = cuda::mod_mul(x[168], y[168], cuda::RNS_MODULI[168]);
        #endif
        #if RNS_MODULI_SIZE > 169
                result[169] = cuda::mod_mul(x[169], y[169], cuda::RNS_MODULI[169]);
        #endif
        #if RNS_MODULI_SIZE > 170
                result[170] = cuda::mod_mul(x[170], y[170], cuda::RNS_MODULI[170]);
        #endif
        #if RNS_MODULI_SIZE > 171
                result[171] = cuda::mod_mul(x[171], y[171], cuda::RNS_MODULI[171]);
        #endif
        #if RNS_MODULI_SIZE > 172
                result[172] = cuda::mod_mul(x[172], y[172], cuda::RNS_MODULI[172]);
        #endif
        #if RNS_MODULI_SIZE > 173
                result[173] = cuda::mod_mul(x[173], y[173], cuda::RNS_MODULI[173]);
        #endif
        #if RNS_MODULI_SIZE > 174
                result[174] = cuda::mod_mul(x[174], y[174], cuda::RNS_MODULI[174]);
        #endif
        #if RNS_MODULI_SIZE > 175
                result[175] = cuda::mod_mul(x[175], y[175], cuda::RNS_MODULI[175]);
        #endif
        #if RNS_MODULI_SIZE > 176
                result[176] = cuda::mod_mul(x[176], y[176], cuda::RNS_MODULI[176]);
        #endif
        #if RNS_MODULI_SIZE > 177
                result[177] = cuda::mod_mul(x[177], y[177], cuda::RNS_MODULI[177]);
        #endif
        #if RNS_MODULI_SIZE > 178
                result[178] = cuda::mod_mul(x[178], y[178], cuda::RNS_MODULI[178]);
        #endif
        #if RNS_MODULI_SIZE > 179
                result[179] = cuda::mod_mul(x[179], y[179], cuda::RNS_MODULI[179]);
        #endif
        #if RNS_MODULI_SIZE > 180
                result[180] = cuda::mod_mul(x[180], y[180], cuda::RNS_MODULI[180]);
        #endif
        #if RNS_MODULI_SIZE > 181
                result[181] = cuda::mod_mul(x[181], y[181], cuda::RNS_MODULI[181]);
        #endif
        #if RNS_MODULI_SIZE > 182
                result[182] = cuda::mod_mul(x[182], y[182], cuda::RNS_MODULI[182]);
        #endif
        #if RNS_MODULI_SIZE > 183
                result[183] = cuda::mod_mul(x[183], y[183], cuda::RNS_MODULI[183]);
        #endif
        #if RNS_MODULI_SIZE > 184
                result[184] = cuda::mod_mul(x[184], y[184], cuda::RNS_MODULI[184]);
        #endif
        #if RNS_MODULI_SIZE > 185
                result[185] = cuda::mod_mul(x[185], y[185], cuda::RNS_MODULI[185]);
        #endif
        #if RNS_MODULI_SIZE > 186
                result[186] = cuda::mod_mul(x[186], y[186], cuda::RNS_MODULI[186]);
        #endif
        #if RNS_MODULI_SIZE > 187
                result[187] = cuda::mod_mul(x[187], y[187], cuda::RNS_MODULI[187]);
        #endif
        #if RNS_MODULI_SIZE > 188
                result[188] = cuda::mod_mul(x[188], y[188], cuda::RNS_MODULI[188]);
        #endif
        #if RNS_MODULI_SIZE > 189
                result[189] = cuda::mod_mul(x[189], y[189], cuda::RNS_MODULI[189]);
        #endif
        #if RNS_MODULI_SIZE > 190
                result[190] = cuda::mod_mul(x[190], y[190], cuda::RNS_MODULI[190]);
        #endif
        #if RNS_MODULI_SIZE > 191
                result[191] = cuda::mod_mul(x[191], y[191], cuda::RNS_MODULI[191]);
        #endif
        #if RNS_MODULI_SIZE > 192
                result[192] = cuda::mod_mul(x[192], y[192], cuda::RNS_MODULI[192]);
        #endif
        #if RNS_MODULI_SIZE > 193
                result[193] = cuda::mod_mul(x[193], y[193], cuda::RNS_MODULI[193]);
        #endif
        #if RNS_MODULI_SIZE > 194
                result[194] = cuda::mod_mul(x[194], y[194], cuda::RNS_MODULI[194]);
        #endif
        #if RNS_MODULI_SIZE > 195
                result[195] = cuda::mod_mul(x[195], y[195], cuda::RNS_MODULI[195]);
        #endif
        #if RNS_MODULI_SIZE > 196
                result[196] = cuda::mod_mul(x[196], y[196], cuda::RNS_MODULI[196]);
        #endif
        #if RNS_MODULI_SIZE > 197
                result[197] = cuda::mod_mul(x[197], y[197], cuda::RNS_MODULI[197]);
        #endif
        #if RNS_MODULI_SIZE > 198
                result[198] = cuda::mod_mul(x[198], y[198], cuda::RNS_MODULI[198]);
        #endif
        #if RNS_MODULI_SIZE > 199
                result[199] = cuda::mod_mul(x[199], y[199], cuda::RNS_MODULI[199]);
        #endif
    }

    /*!
     * Unrolled addition of two RNS numbers.
     */
    DEVICE_CUDA_FORCEINLINE void rns_add(int * result, int * x, int * y){
        #if RNS_MODULI_SIZE > 0
                result[0] = cuda::mod_add(x[0], y[0], cuda::RNS_MODULI[0]);
        #endif
        #if RNS_MODULI_SIZE > 1
                result[1] = cuda::mod_add(x[1], y[1], cuda::RNS_MODULI[1]);
        #endif
        #if RNS_MODULI_SIZE > 2
                result[2] = cuda::mod_add(x[2], y[2], cuda::RNS_MODULI[2]);
        #endif
        #if RNS_MODULI_SIZE > 3
                result[3] = cuda::mod_add(x[3], y[3], cuda::RNS_MODULI[3]);
        #endif
        #if RNS_MODULI_SIZE > 4
                result[4] = cuda::mod_add(x[4], y[4], cuda::RNS_MODULI[4]);
        #endif
        #if RNS_MODULI_SIZE > 5
                result[5] = cuda::mod_add(x[5], y[5], cuda::RNS_MODULI[5]);
        #endif
        #if RNS_MODULI_SIZE > 6
                result[6] = cuda::mod_add(x[6], y[6], cuda::RNS_MODULI[6]);
        #endif
        #if RNS_MODULI_SIZE > 7
                result[7] = cuda::mod_add(x[7], y[7], cuda::RNS_MODULI[7]);
        #endif
        #if RNS_MODULI_SIZE > 8
                result[8] = cuda::mod_add(x[8], y[8], cuda::RNS_MODULI[8]);
        #endif
        #if RNS_MODULI_SIZE > 9
                result[9] = cuda::mod_add(x[9], y[9], cuda::RNS_MODULI[9]);
        #endif
        #if RNS_MODULI_SIZE > 10
                result[10] = cuda::mod_add(x[10], y[10], cuda::RNS_MODULI[10]);
        #endif
        #if RNS_MODULI_SIZE > 11
                result[11] = cuda::mod_add(x[11], y[11], cuda::RNS_MODULI[11]);
        #endif
        #if RNS_MODULI_SIZE > 12
                result[12] = cuda::mod_add(x[12], y[12], cuda::RNS_MODULI[12]);
        #endif
        #if RNS_MODULI_SIZE > 13
                result[13] = cuda::mod_add(x[13], y[13], cuda::RNS_MODULI[13]);
        #endif
        #if RNS_MODULI_SIZE > 14
                result[14] = cuda::mod_add(x[14], y[14], cuda::RNS_MODULI[14]);
        #endif
        #if RNS_MODULI_SIZE > 15
                result[15] = cuda::mod_add(x[15], y[15], cuda::RNS_MODULI[15]);
        #endif
        #if RNS_MODULI_SIZE > 16
                result[16] = cuda::mod_add(x[16], y[16], cuda::RNS_MODULI[16]);
        #endif
        #if RNS_MODULI_SIZE > 17
                result[17] = cuda::mod_add(x[17], y[17], cuda::RNS_MODULI[17]);
        #endif
        #if RNS_MODULI_SIZE > 18
                result[18] = cuda::mod_add(x[18], y[18], cuda::RNS_MODULI[18]);
        #endif
        #if RNS_MODULI_SIZE > 19
                result[19] = cuda::mod_add(x[19], y[19], cuda::RNS_MODULI[19]);
        #endif
        #if RNS_MODULI_SIZE > 20
                result[20] = cuda::mod_add(x[20], y[20], cuda::RNS_MODULI[20]);
        #endif
        #if RNS_MODULI_SIZE > 21
                result[21] = cuda::mod_add(x[21], y[21], cuda::RNS_MODULI[21]);
        #endif
        #if RNS_MODULI_SIZE > 22
                result[22] = cuda::mod_add(x[22], y[22], cuda::RNS_MODULI[22]);
        #endif
        #if RNS_MODULI_SIZE > 23
                result[23] = cuda::mod_add(x[23], y[23], cuda::RNS_MODULI[23]);
        #endif
        #if RNS_MODULI_SIZE > 24
                result[24] = cuda::mod_add(x[24], y[24], cuda::RNS_MODULI[24]);
        #endif
        #if RNS_MODULI_SIZE > 25
                result[25] = cuda::mod_add(x[25], y[25], cuda::RNS_MODULI[25]);
        #endif
        #if RNS_MODULI_SIZE > 26
                result[26] = cuda::mod_add(x[26], y[26], cuda::RNS_MODULI[26]);
        #endif
        #if RNS_MODULI_SIZE > 27
                result[27] = cuda::mod_add(x[27], y[27], cuda::RNS_MODULI[27]);
        #endif
        #if RNS_MODULI_SIZE > 28
                result[28] = cuda::mod_add(x[28], y[28], cuda::RNS_MODULI[28]);
        #endif
        #if RNS_MODULI_SIZE > 29
                result[29] = cuda::mod_add(x[29], y[29], cuda::RNS_MODULI[29]);
        #endif
        #if RNS_MODULI_SIZE > 30
                result[30] = cuda::mod_add(x[30], y[30], cuda::RNS_MODULI[30]);
        #endif
        #if RNS_MODULI_SIZE > 31
                result[31] = cuda::mod_add(x[31], y[31], cuda::RNS_MODULI[31]);
        #endif
        #if RNS_MODULI_SIZE > 32
                result[32] = cuda::mod_add(x[32], y[32], cuda::RNS_MODULI[32]);
        #endif
        #if RNS_MODULI_SIZE > 33
                result[33] = cuda::mod_add(x[33], y[33], cuda::RNS_MODULI[33]);
        #endif
        #if RNS_MODULI_SIZE > 34
                result[34] = cuda::mod_add(x[34], y[34], cuda::RNS_MODULI[34]);
        #endif
        #if RNS_MODULI_SIZE > 35
                result[35] = cuda::mod_add(x[35], y[35], cuda::RNS_MODULI[35]);
        #endif
        #if RNS_MODULI_SIZE > 36
                result[36] = cuda::mod_add(x[36], y[36], cuda::RNS_MODULI[36]);
        #endif
        #if RNS_MODULI_SIZE > 37
                result[37] = cuda::mod_add(x[37], y[37], cuda::RNS_MODULI[37]);
        #endif
        #if RNS_MODULI_SIZE > 38
                result[38] = cuda::mod_add(x[38], y[38], cuda::RNS_MODULI[38]);
        #endif
        #if RNS_MODULI_SIZE > 39
                result[39] = cuda::mod_add(x[39], y[39], cuda::RNS_MODULI[39]);
        #endif
        #if RNS_MODULI_SIZE > 40
                result[40] = cuda::mod_add(x[40], y[40], cuda::RNS_MODULI[40]);
        #endif
        #if RNS_MODULI_SIZE > 41
                result[41] = cuda::mod_add(x[41], y[41], cuda::RNS_MODULI[41]);
        #endif
        #if RNS_MODULI_SIZE > 42
                result[42] = cuda::mod_add(x[42], y[42], cuda::RNS_MODULI[42]);
        #endif
        #if RNS_MODULI_SIZE > 43
                result[43] = cuda::mod_add(x[43], y[43], cuda::RNS_MODULI[43]);
        #endif
        #if RNS_MODULI_SIZE > 44
                result[44] = cuda::mod_add(x[44], y[44], cuda::RNS_MODULI[44]);
        #endif
        #if RNS_MODULI_SIZE > 45
                result[45] = cuda::mod_add(x[45], y[45], cuda::RNS_MODULI[45]);
        #endif
        #if RNS_MODULI_SIZE > 46
                result[46] = cuda::mod_add(x[46], y[46], cuda::RNS_MODULI[46]);
        #endif
        #if RNS_MODULI_SIZE > 47
                result[47] = cuda::mod_add(x[47], y[47], cuda::RNS_MODULI[47]);
        #endif
        #if RNS_MODULI_SIZE > 48
                result[48] = cuda::mod_add(x[48], y[48], cuda::RNS_MODULI[48]);
        #endif
        #if RNS_MODULI_SIZE > 49
                result[49] = cuda::mod_add(x[49], y[49], cuda::RNS_MODULI[49]);
        #endif
        #if RNS_MODULI_SIZE > 50
                result[50] = cuda::mod_add(x[50], y[50], cuda::RNS_MODULI[50]);
        #endif
        #if RNS_MODULI_SIZE > 51
                result[51] = cuda::mod_add(x[51], y[51], cuda::RNS_MODULI[51]);
        #endif
        #if RNS_MODULI_SIZE > 52
                result[52] = cuda::mod_add(x[52], y[52], cuda::RNS_MODULI[52]);
        #endif
        #if RNS_MODULI_SIZE > 53
                result[53] = cuda::mod_add(x[53], y[53], cuda::RNS_MODULI[53]);
        #endif
        #if RNS_MODULI_SIZE > 54
                result[54] = cuda::mod_add(x[54], y[54], cuda::RNS_MODULI[54]);
        #endif
        #if RNS_MODULI_SIZE > 55
                result[55] = cuda::mod_add(x[55], y[55], cuda::RNS_MODULI[55]);
        #endif
        #if RNS_MODULI_SIZE > 56
                result[56] = cuda::mod_add(x[56], y[56], cuda::RNS_MODULI[56]);
        #endif
        #if RNS_MODULI_SIZE > 57
                result[57] = cuda::mod_add(x[57], y[57], cuda::RNS_MODULI[57]);
        #endif
        #if RNS_MODULI_SIZE > 58
                result[58] = cuda::mod_add(x[58], y[58], cuda::RNS_MODULI[58]);
        #endif
        #if RNS_MODULI_SIZE > 59
                result[59] = cuda::mod_add(x[59], y[59], cuda::RNS_MODULI[59]);
        #endif
        #if RNS_MODULI_SIZE > 60
                result[60] = cuda::mod_add(x[60], y[60], cuda::RNS_MODULI[60]);
        #endif
        #if RNS_MODULI_SIZE > 61
                result[61] = cuda::mod_add(x[61], y[61], cuda::RNS_MODULI[61]);
        #endif
        #if RNS_MODULI_SIZE > 62
                result[62] = cuda::mod_add(x[62], y[62], cuda::RNS_MODULI[62]);
        #endif
        #if RNS_MODULI_SIZE > 63
                result[63] = cuda::mod_add(x[63], y[63], cuda::RNS_MODULI[63]);
        #endif
        #if RNS_MODULI_SIZE > 64
                result[64] = cuda::mod_add(x[64], y[64], cuda::RNS_MODULI[64]);
        #endif
        #if RNS_MODULI_SIZE > 65
                result[65] = cuda::mod_add(x[65], y[65], cuda::RNS_MODULI[65]);
        #endif
        #if RNS_MODULI_SIZE > 66
                result[66] = cuda::mod_add(x[66], y[66], cuda::RNS_MODULI[66]);
        #endif
        #if RNS_MODULI_SIZE > 67
                result[67] = cuda::mod_add(x[67], y[67], cuda::RNS_MODULI[67]);
        #endif
        #if RNS_MODULI_SIZE > 68
                result[68] = cuda::mod_add(x[68], y[68], cuda::RNS_MODULI[68]);
        #endif
        #if RNS_MODULI_SIZE > 69
                result[69] = cuda::mod_add(x[69], y[69], cuda::RNS_MODULI[69]);
        #endif
        #if RNS_MODULI_SIZE > 70
                result[70] = cuda::mod_add(x[70], y[70], cuda::RNS_MODULI[70]);
        #endif
        #if RNS_MODULI_SIZE > 71
                result[71] = cuda::mod_add(x[71], y[71], cuda::RNS_MODULI[71]);
        #endif
        #if RNS_MODULI_SIZE > 72
                result[72] = cuda::mod_add(x[72], y[72], cuda::RNS_MODULI[72]);
        #endif
        #if RNS_MODULI_SIZE > 73
                result[73] = cuda::mod_add(x[73], y[73], cuda::RNS_MODULI[73]);
        #endif
        #if RNS_MODULI_SIZE > 74
                result[74] = cuda::mod_add(x[74], y[74], cuda::RNS_MODULI[74]);
        #endif
        #if RNS_MODULI_SIZE > 75
                result[75] = cuda::mod_add(x[75], y[75], cuda::RNS_MODULI[75]);
        #endif
        #if RNS_MODULI_SIZE > 76
                result[76] = cuda::mod_add(x[76], y[76], cuda::RNS_MODULI[76]);
        #endif
        #if RNS_MODULI_SIZE > 77
                result[77] = cuda::mod_add(x[77], y[77], cuda::RNS_MODULI[77]);
        #endif
        #if RNS_MODULI_SIZE > 78
                result[78] = cuda::mod_add(x[78], y[78], cuda::RNS_MODULI[78]);
        #endif
        #if RNS_MODULI_SIZE > 79
                result[79] = cuda::mod_add(x[79], y[79], cuda::RNS_MODULI[79]);
        #endif
        #if RNS_MODULI_SIZE > 80
                result[80] = cuda::mod_add(x[80], y[80], cuda::RNS_MODULI[80]);
        #endif
        #if RNS_MODULI_SIZE > 81
                result[81] = cuda::mod_add(x[81], y[81], cuda::RNS_MODULI[81]);
        #endif
        #if RNS_MODULI_SIZE > 82
                result[82] = cuda::mod_add(x[82], y[82], cuda::RNS_MODULI[82]);
        #endif
        #if RNS_MODULI_SIZE > 83
                result[83] = cuda::mod_add(x[83], y[83], cuda::RNS_MODULI[83]);
        #endif
        #if RNS_MODULI_SIZE > 84
                result[84] = cuda::mod_add(x[84], y[84], cuda::RNS_MODULI[84]);
        #endif
        #if RNS_MODULI_SIZE > 85
                result[85] = cuda::mod_add(x[85], y[85], cuda::RNS_MODULI[85]);
        #endif
        #if RNS_MODULI_SIZE > 86
                result[86] = cuda::mod_add(x[86], y[86], cuda::RNS_MODULI[86]);
        #endif
        #if RNS_MODULI_SIZE > 87
                result[87] = cuda::mod_add(x[87], y[87], cuda::RNS_MODULI[87]);
        #endif
        #if RNS_MODULI_SIZE > 88
                result[88] = cuda::mod_add(x[88], y[88], cuda::RNS_MODULI[88]);
        #endif
        #if RNS_MODULI_SIZE > 89
                result[89] = cuda::mod_add(x[89], y[89], cuda::RNS_MODULI[89]);
        #endif
        #if RNS_MODULI_SIZE > 90
                result[90] = cuda::mod_add(x[90], y[90], cuda::RNS_MODULI[90]);
        #endif
        #if RNS_MODULI_SIZE > 91
                result[91] = cuda::mod_add(x[91], y[91], cuda::RNS_MODULI[91]);
        #endif
        #if RNS_MODULI_SIZE > 92
                result[92] = cuda::mod_add(x[92], y[92], cuda::RNS_MODULI[92]);
        #endif
        #if RNS_MODULI_SIZE > 93
                result[93] = cuda::mod_add(x[93], y[93], cuda::RNS_MODULI[93]);
        #endif
        #if RNS_MODULI_SIZE > 94
                result[94] = cuda::mod_add(x[94], y[94], cuda::RNS_MODULI[94]);
        #endif
        #if RNS_MODULI_SIZE > 95
                result[95] = cuda::mod_add(x[95], y[95], cuda::RNS_MODULI[95]);
        #endif
        #if RNS_MODULI_SIZE > 96
                result[96] = cuda::mod_add(x[96], y[96], cuda::RNS_MODULI[96]);
        #endif
        #if RNS_MODULI_SIZE > 97
                result[97] = cuda::mod_add(x[97], y[97], cuda::RNS_MODULI[97]);
        #endif
        #if RNS_MODULI_SIZE > 98
                result[98] = cuda::mod_add(x[98], y[98], cuda::RNS_MODULI[98]);
        #endif
        #if RNS_MODULI_SIZE > 99
                result[99] = cuda::mod_add(x[99], y[99], cuda::RNS_MODULI[99]);
        #endif
        #if RNS_MODULI_SIZE > 100
                result[100] = cuda::mod_add(x[100], y[100], cuda::RNS_MODULI[100]);
        #endif
        #if RNS_MODULI_SIZE > 101
                result[101] = cuda::mod_add(x[101], y[101], cuda::RNS_MODULI[101]);
        #endif
        #if RNS_MODULI_SIZE > 102
                result[102] = cuda::mod_add(x[102], y[102], cuda::RNS_MODULI[102]);
        #endif
        #if RNS_MODULI_SIZE > 103
                result[103] = cuda::mod_add(x[103], y[103], cuda::RNS_MODULI[103]);
        #endif
        #if RNS_MODULI_SIZE > 104
                result[104] = cuda::mod_add(x[104], y[104], cuda::RNS_MODULI[104]);
        #endif
        #if RNS_MODULI_SIZE > 105
                result[105] = cuda::mod_add(x[105], y[105], cuda::RNS_MODULI[105]);
        #endif
        #if RNS_MODULI_SIZE > 106
                result[106] = cuda::mod_add(x[106], y[106], cuda::RNS_MODULI[106]);
        #endif
        #if RNS_MODULI_SIZE > 107
                result[107] = cuda::mod_add(x[107], y[107], cuda::RNS_MODULI[107]);
        #endif
        #if RNS_MODULI_SIZE > 108
                result[108] = cuda::mod_add(x[108], y[108], cuda::RNS_MODULI[108]);
        #endif
        #if RNS_MODULI_SIZE > 109
                result[109] = cuda::mod_add(x[109], y[109], cuda::RNS_MODULI[109]);
        #endif
        #if RNS_MODULI_SIZE > 110
                result[110] = cuda::mod_add(x[110], y[110], cuda::RNS_MODULI[110]);
        #endif
        #if RNS_MODULI_SIZE > 111
                result[111] = cuda::mod_add(x[111], y[111], cuda::RNS_MODULI[111]);
        #endif
        #if RNS_MODULI_SIZE > 112
                result[112] = cuda::mod_add(x[112], y[112], cuda::RNS_MODULI[112]);
        #endif
        #if RNS_MODULI_SIZE > 113
                result[113] = cuda::mod_add(x[113], y[113], cuda::RNS_MODULI[113]);
        #endif
        #if RNS_MODULI_SIZE > 114
                result[114] = cuda::mod_add(x[114], y[114], cuda::RNS_MODULI[114]);
        #endif
        #if RNS_MODULI_SIZE > 115
                result[115] = cuda::mod_add(x[115], y[115], cuda::RNS_MODULI[115]);
        #endif
        #if RNS_MODULI_SIZE > 116
                result[116] = cuda::mod_add(x[116], y[116], cuda::RNS_MODULI[116]);
        #endif
        #if RNS_MODULI_SIZE > 117
                result[117] = cuda::mod_add(x[117], y[117], cuda::RNS_MODULI[117]);
        #endif
        #if RNS_MODULI_SIZE > 118
                result[118] = cuda::mod_add(x[118], y[118], cuda::RNS_MODULI[118]);
        #endif
        #if RNS_MODULI_SIZE > 119
                result[119] = cuda::mod_add(x[119], y[119], cuda::RNS_MODULI[119]);
        #endif
        #if RNS_MODULI_SIZE > 120
                result[120] = cuda::mod_add(x[120], y[120], cuda::RNS_MODULI[120]);
        #endif
        #if RNS_MODULI_SIZE > 121
                result[121] = cuda::mod_add(x[121], y[121], cuda::RNS_MODULI[121]);
        #endif
        #if RNS_MODULI_SIZE > 122
                result[122] = cuda::mod_add(x[122], y[122], cuda::RNS_MODULI[122]);
        #endif
        #if RNS_MODULI_SIZE > 123
                result[123] = cuda::mod_add(x[123], y[123], cuda::RNS_MODULI[123]);
        #endif
        #if RNS_MODULI_SIZE > 124
                result[124] = cuda::mod_add(x[124], y[124], cuda::RNS_MODULI[124]);
        #endif
        #if RNS_MODULI_SIZE > 125
                result[125] = cuda::mod_add(x[125], y[125], cuda::RNS_MODULI[125]);
        #endif
        #if RNS_MODULI_SIZE > 126
                result[126] = cuda::mod_add(x[126], y[126], cuda::RNS_MODULI[126]);
        #endif
        #if RNS_MODULI_SIZE > 127
                result[127] = cuda::mod_add(x[127], y[127], cuda::RNS_MODULI[127]);
        #endif
        #if RNS_MODULI_SIZE > 128
                result[128] = cuda::mod_add(x[128], y[128], cuda::RNS_MODULI[128]);
        #endif
        #if RNS_MODULI_SIZE > 129
                result[129] = cuda::mod_add(x[129], y[129], cuda::RNS_MODULI[129]);
        #endif
        #if RNS_MODULI_SIZE > 130
                result[130] = cuda::mod_add(x[130], y[130], cuda::RNS_MODULI[130]);
        #endif
        #if RNS_MODULI_SIZE > 131
                result[131] = cuda::mod_add(x[131], y[131], cuda::RNS_MODULI[131]);
        #endif
        #if RNS_MODULI_SIZE > 132
                result[132] = cuda::mod_add(x[132], y[132], cuda::RNS_MODULI[132]);
        #endif
        #if RNS_MODULI_SIZE > 133
                result[133] = cuda::mod_add(x[133], y[133], cuda::RNS_MODULI[133]);
        #endif
        #if RNS_MODULI_SIZE > 134
                result[134] = cuda::mod_add(x[134], y[134], cuda::RNS_MODULI[134]);
        #endif
        #if RNS_MODULI_SIZE > 135
                result[135] = cuda::mod_add(x[135], y[135], cuda::RNS_MODULI[135]);
        #endif
        #if RNS_MODULI_SIZE > 136
                result[136] = cuda::mod_add(x[136], y[136], cuda::RNS_MODULI[136]);
        #endif
        #if RNS_MODULI_SIZE > 137
                result[137] = cuda::mod_add(x[137], y[137], cuda::RNS_MODULI[137]);
        #endif
        #if RNS_MODULI_SIZE > 138
                result[138] = cuda::mod_add(x[138], y[138], cuda::RNS_MODULI[138]);
        #endif
        #if RNS_MODULI_SIZE > 139
                result[139] = cuda::mod_add(x[139], y[139], cuda::RNS_MODULI[139]);
        #endif
        #if RNS_MODULI_SIZE > 140
                result[140] = cuda::mod_add(x[140], y[140], cuda::RNS_MODULI[140]);
        #endif
        #if RNS_MODULI_SIZE > 141
                result[141] = cuda::mod_add(x[141], y[141], cuda::RNS_MODULI[141]);
        #endif
        #if RNS_MODULI_SIZE > 142
                result[142] = cuda::mod_add(x[142], y[142], cuda::RNS_MODULI[142]);
        #endif
        #if RNS_MODULI_SIZE > 143
                result[143] = cuda::mod_add(x[143], y[143], cuda::RNS_MODULI[143]);
        #endif
        #if RNS_MODULI_SIZE > 144
                result[144] = cuda::mod_add(x[144], y[144], cuda::RNS_MODULI[144]);
        #endif
        #if RNS_MODULI_SIZE > 145
                result[145] = cuda::mod_add(x[145], y[145], cuda::RNS_MODULI[145]);
        #endif
        #if RNS_MODULI_SIZE > 146
                result[146] = cuda::mod_add(x[146], y[146], cuda::RNS_MODULI[146]);
        #endif
        #if RNS_MODULI_SIZE > 147
                result[147] = cuda::mod_add(x[147], y[147], cuda::RNS_MODULI[147]);
        #endif
        #if RNS_MODULI_SIZE > 148
                result[148] = cuda::mod_add(x[148], y[148], cuda::RNS_MODULI[148]);
        #endif
        #if RNS_MODULI_SIZE > 149
                result[149] = cuda::mod_add(x[149], y[149], cuda::RNS_MODULI[149]);
        #endif
        #if RNS_MODULI_SIZE > 150
                result[150] = cuda::mod_add(x[150], y[150], cuda::RNS_MODULI[150]);
        #endif
        #if RNS_MODULI_SIZE > 151
                result[151] = cuda::mod_add(x[151], y[151], cuda::RNS_MODULI[151]);
        #endif
        #if RNS_MODULI_SIZE > 152
                result[152] = cuda::mod_add(x[152], y[152], cuda::RNS_MODULI[152]);
        #endif
        #if RNS_MODULI_SIZE > 153
                result[153] = cuda::mod_add(x[153], y[153], cuda::RNS_MODULI[153]);
        #endif
        #if RNS_MODULI_SIZE > 154
                result[154] = cuda::mod_add(x[154], y[154], cuda::RNS_MODULI[154]);
        #endif
        #if RNS_MODULI_SIZE > 155
                result[155] = cuda::mod_add(x[155], y[155], cuda::RNS_MODULI[155]);
        #endif
        #if RNS_MODULI_SIZE > 156
                result[156] = cuda::mod_add(x[156], y[156], cuda::RNS_MODULI[156]);
        #endif
        #if RNS_MODULI_SIZE > 157
                result[157] = cuda::mod_add(x[157], y[157], cuda::RNS_MODULI[157]);
        #endif
        #if RNS_MODULI_SIZE > 158
                result[158] = cuda::mod_add(x[158], y[158], cuda::RNS_MODULI[158]);
        #endif
        #if RNS_MODULI_SIZE > 159
                result[159] = cuda::mod_add(x[159], y[159], cuda::RNS_MODULI[159]);
        #endif
        #if RNS_MODULI_SIZE > 160
                result[160] = cuda::mod_add(x[160], y[160], cuda::RNS_MODULI[160]);
        #endif
        #if RNS_MODULI_SIZE > 161
                result[161] = cuda::mod_add(x[161], y[161], cuda::RNS_MODULI[161]);
        #endif
        #if RNS_MODULI_SIZE > 162
                result[162] = cuda::mod_add(x[162], y[162], cuda::RNS_MODULI[162]);
        #endif
        #if RNS_MODULI_SIZE > 163
                result[163] = cuda::mod_add(x[163], y[163], cuda::RNS_MODULI[163]);
        #endif
        #if RNS_MODULI_SIZE > 164
                result[164] = cuda::mod_add(x[164], y[164], cuda::RNS_MODULI[164]);
        #endif
        #if RNS_MODULI_SIZE > 165
                result[165] = cuda::mod_add(x[165], y[165], cuda::RNS_MODULI[165]);
        #endif
        #if RNS_MODULI_SIZE > 166
                result[166] = cuda::mod_add(x[166], y[166], cuda::RNS_MODULI[166]);
        #endif
        #if RNS_MODULI_SIZE > 167
                result[167] = cuda::mod_add(x[167], y[167], cuda::RNS_MODULI[167]);
        #endif
        #if RNS_MODULI_SIZE > 168
                result[168] = cuda::mod_add(x[168], y[168], cuda::RNS_MODULI[168]);
        #endif
        #if RNS_MODULI_SIZE > 169
                result[169] = cuda::mod_add(x[169], y[169], cuda::RNS_MODULI[169]);
        #endif
        #if RNS_MODULI_SIZE > 170
                result[170] = cuda::mod_add(x[170], y[170], cuda::RNS_MODULI[170]);
        #endif
        #if RNS_MODULI_SIZE > 171
                result[171] = cuda::mod_add(x[171], y[171], cuda::RNS_MODULI[171]);
        #endif
        #if RNS_MODULI_SIZE > 172
                result[172] = cuda::mod_add(x[172], y[172], cuda::RNS_MODULI[172]);
        #endif
        #if RNS_MODULI_SIZE > 173
                result[173] = cuda::mod_add(x[173], y[173], cuda::RNS_MODULI[173]);
        #endif
        #if RNS_MODULI_SIZE > 174
                result[174] = cuda::mod_add(x[174], y[174], cuda::RNS_MODULI[174]);
        #endif
        #if RNS_MODULI_SIZE > 175
                result[175] = cuda::mod_add(x[175], y[175], cuda::RNS_MODULI[175]);
        #endif
        #if RNS_MODULI_SIZE > 176
                result[176] = cuda::mod_add(x[176], y[176], cuda::RNS_MODULI[176]);
        #endif
        #if RNS_MODULI_SIZE > 177
                result[177] = cuda::mod_add(x[177], y[177], cuda::RNS_MODULI[177]);
        #endif
        #if RNS_MODULI_SIZE > 178
                result[178] = cuda::mod_add(x[178], y[178], cuda::RNS_MODULI[178]);
        #endif
        #if RNS_MODULI_SIZE > 179
                result[179] = cuda::mod_add(x[179], y[179], cuda::RNS_MODULI[179]);
        #endif
        #if RNS_MODULI_SIZE > 180
                result[180] = cuda::mod_add(x[180], y[180], cuda::RNS_MODULI[180]);
        #endif
        #if RNS_MODULI_SIZE > 181
                result[181] = cuda::mod_add(x[181], y[181], cuda::RNS_MODULI[181]);
        #endif
        #if RNS_MODULI_SIZE > 182
                result[182] = cuda::mod_add(x[182], y[182], cuda::RNS_MODULI[182]);
        #endif
        #if RNS_MODULI_SIZE > 183
                result[183] = cuda::mod_add(x[183], y[183], cuda::RNS_MODULI[183]);
        #endif
        #if RNS_MODULI_SIZE > 184
                result[184] = cuda::mod_add(x[184], y[184], cuda::RNS_MODULI[184]);
        #endif
        #if RNS_MODULI_SIZE > 185
                result[185] = cuda::mod_add(x[185], y[185], cuda::RNS_MODULI[185]);
        #endif
        #if RNS_MODULI_SIZE > 186
                result[186] = cuda::mod_add(x[186], y[186], cuda::RNS_MODULI[186]);
        #endif
        #if RNS_MODULI_SIZE > 187
                result[187] = cuda::mod_add(x[187], y[187], cuda::RNS_MODULI[187]);
        #endif
        #if RNS_MODULI_SIZE > 188
                result[188] = cuda::mod_add(x[188], y[188], cuda::RNS_MODULI[188]);
        #endif
        #if RNS_MODULI_SIZE > 189
                result[189] = cuda::mod_add(x[189], y[189], cuda::RNS_MODULI[189]);
        #endif
        #if RNS_MODULI_SIZE > 190
                result[190] = cuda::mod_add(x[190], y[190], cuda::RNS_MODULI[190]);
        #endif
        #if RNS_MODULI_SIZE > 191
                result[191] = cuda::mod_add(x[191], y[191], cuda::RNS_MODULI[191]);
        #endif
        #if RNS_MODULI_SIZE > 192
                result[192] = cuda::mod_add(x[192], y[192], cuda::RNS_MODULI[192]);
        #endif
        #if RNS_MODULI_SIZE > 193
                result[193] = cuda::mod_add(x[193], y[193], cuda::RNS_MODULI[193]);
        #endif
        #if RNS_MODULI_SIZE > 194
                result[194] = cuda::mod_add(x[194], y[194], cuda::RNS_MODULI[194]);
        #endif
        #if RNS_MODULI_SIZE > 195
                result[195] = cuda::mod_add(x[195], y[195], cuda::RNS_MODULI[195]);
        #endif
        #if RNS_MODULI_SIZE > 196
                result[196] = cuda::mod_add(x[196], y[196], cuda::RNS_MODULI[196]);
        #endif
        #if RNS_MODULI_SIZE > 197
                result[197] = cuda::mod_add(x[197], y[197], cuda::RNS_MODULI[197]);
        #endif
        #if RNS_MODULI_SIZE > 198
                result[198] = cuda::mod_add(x[198], y[198], cuda::RNS_MODULI[198]);
        #endif
        #if RNS_MODULI_SIZE > 199
                result[199] = cuda::mod_add(x[199], y[199], cuda::RNS_MODULI[199]);
        #endif
    }

    /*!
     * Unrolled subtraction of two RNS numbers.
     */
    DEVICE_CUDA_FORCEINLINE void rns_sub(int * result, int * x, int * y){
        #if RNS_MODULI_SIZE > 0
                result[0] = cuda::mod_psub(x[0], y[0], cuda::RNS_MODULI[0]);
        #endif
        #if RNS_MODULI_SIZE > 1
                result[1] = cuda::mod_psub(x[1], y[1], cuda::RNS_MODULI[1]);
        #endif
        #if RNS_MODULI_SIZE > 2
                result[2] = cuda::mod_psub(x[2], y[2], cuda::RNS_MODULI[2]);
        #endif
        #if RNS_MODULI_SIZE > 3
                result[3] = cuda::mod_psub(x[3], y[3], cuda::RNS_MODULI[3]);
        #endif
        #if RNS_MODULI_SIZE > 4
                result[4] = cuda::mod_psub(x[4], y[4], cuda::RNS_MODULI[4]);
        #endif
        #if RNS_MODULI_SIZE > 5
                result[5] = cuda::mod_psub(x[5], y[5], cuda::RNS_MODULI[5]);
        #endif
        #if RNS_MODULI_SIZE > 6
                result[6] = cuda::mod_psub(x[6], y[6], cuda::RNS_MODULI[6]);
        #endif
        #if RNS_MODULI_SIZE > 7
                result[7] = cuda::mod_psub(x[7], y[7], cuda::RNS_MODULI[7]);
        #endif
        #if RNS_MODULI_SIZE > 8
                result[8] = cuda::mod_psub(x[8], y[8], cuda::RNS_MODULI[8]);
        #endif
        #if RNS_MODULI_SIZE > 9
                result[9] = cuda::mod_psub(x[9], y[9], cuda::RNS_MODULI[9]);
        #endif
        #if RNS_MODULI_SIZE > 10
                result[10] = cuda::mod_psub(x[10], y[10], cuda::RNS_MODULI[10]);
        #endif
        #if RNS_MODULI_SIZE > 11
                result[11] = cuda::mod_psub(x[11], y[11], cuda::RNS_MODULI[11]);
        #endif
        #if RNS_MODULI_SIZE > 12
                result[12] = cuda::mod_psub(x[12], y[12], cuda::RNS_MODULI[12]);
        #endif
        #if RNS_MODULI_SIZE > 13
                result[13] = cuda::mod_psub(x[13], y[13], cuda::RNS_MODULI[13]);
        #endif
        #if RNS_MODULI_SIZE > 14
                result[14] = cuda::mod_psub(x[14], y[14], cuda::RNS_MODULI[14]);
        #endif
        #if RNS_MODULI_SIZE > 15
                result[15] = cuda::mod_psub(x[15], y[15], cuda::RNS_MODULI[15]);
        #endif
        #if RNS_MODULI_SIZE > 16
                result[16] = cuda::mod_psub(x[16], y[16], cuda::RNS_MODULI[16]);
        #endif
        #if RNS_MODULI_SIZE > 17
                result[17] = cuda::mod_psub(x[17], y[17], cuda::RNS_MODULI[17]);
        #endif
        #if RNS_MODULI_SIZE > 18
                result[18] = cuda::mod_psub(x[18], y[18], cuda::RNS_MODULI[18]);
        #endif
        #if RNS_MODULI_SIZE > 19
                result[19] = cuda::mod_psub(x[19], y[19], cuda::RNS_MODULI[19]);
        #endif
        #if RNS_MODULI_SIZE > 20
                result[20] = cuda::mod_psub(x[20], y[20], cuda::RNS_MODULI[20]);
        #endif
        #if RNS_MODULI_SIZE > 21
                result[21] = cuda::mod_psub(x[21], y[21], cuda::RNS_MODULI[21]);
        #endif
        #if RNS_MODULI_SIZE > 22
                result[22] = cuda::mod_psub(x[22], y[22], cuda::RNS_MODULI[22]);
        #endif
        #if RNS_MODULI_SIZE > 23
                result[23] = cuda::mod_psub(x[23], y[23], cuda::RNS_MODULI[23]);
        #endif
        #if RNS_MODULI_SIZE > 24
                result[24] = cuda::mod_psub(x[24], y[24], cuda::RNS_MODULI[24]);
        #endif
        #if RNS_MODULI_SIZE > 25
                result[25] = cuda::mod_psub(x[25], y[25], cuda::RNS_MODULI[25]);
        #endif
        #if RNS_MODULI_SIZE > 26
                result[26] = cuda::mod_psub(x[26], y[26], cuda::RNS_MODULI[26]);
        #endif
        #if RNS_MODULI_SIZE > 27
                result[27] = cuda::mod_psub(x[27], y[27], cuda::RNS_MODULI[27]);
        #endif
        #if RNS_MODULI_SIZE > 28
                result[28] = cuda::mod_psub(x[28], y[28], cuda::RNS_MODULI[28]);
        #endif
        #if RNS_MODULI_SIZE > 29
                result[29] = cuda::mod_psub(x[29], y[29], cuda::RNS_MODULI[29]);
        #endif
        #if RNS_MODULI_SIZE > 30
                result[30] = cuda::mod_psub(x[30], y[30], cuda::RNS_MODULI[30]);
        #endif
        #if RNS_MODULI_SIZE > 31
                result[31] = cuda::mod_psub(x[31], y[31], cuda::RNS_MODULI[31]);
        #endif
        #if RNS_MODULI_SIZE > 32
                result[32] = cuda::mod_psub(x[32], y[32], cuda::RNS_MODULI[32]);
        #endif
        #if RNS_MODULI_SIZE > 33
                result[33] = cuda::mod_psub(x[33], y[33], cuda::RNS_MODULI[33]);
        #endif
        #if RNS_MODULI_SIZE > 34
                result[34] = cuda::mod_psub(x[34], y[34], cuda::RNS_MODULI[34]);
        #endif
        #if RNS_MODULI_SIZE > 35
                result[35] = cuda::mod_psub(x[35], y[35], cuda::RNS_MODULI[35]);
        #endif
        #if RNS_MODULI_SIZE > 36
                result[36] = cuda::mod_psub(x[36], y[36], cuda::RNS_MODULI[36]);
        #endif
        #if RNS_MODULI_SIZE > 37
                result[37] = cuda::mod_psub(x[37], y[37], cuda::RNS_MODULI[37]);
        #endif
        #if RNS_MODULI_SIZE > 38
                result[38] = cuda::mod_psub(x[38], y[38], cuda::RNS_MODULI[38]);
        #endif
        #if RNS_MODULI_SIZE > 39
                result[39] = cuda::mod_psub(x[39], y[39], cuda::RNS_MODULI[39]);
        #endif
        #if RNS_MODULI_SIZE > 40
                result[40] = cuda::mod_psub(x[40], y[40], cuda::RNS_MODULI[40]);
        #endif
        #if RNS_MODULI_SIZE > 41
                result[41] = cuda::mod_psub(x[41], y[41], cuda::RNS_MODULI[41]);
        #endif
        #if RNS_MODULI_SIZE > 42
                result[42] = cuda::mod_psub(x[42], y[42], cuda::RNS_MODULI[42]);
        #endif
        #if RNS_MODULI_SIZE > 43
                result[43] = cuda::mod_psub(x[43], y[43], cuda::RNS_MODULI[43]);
        #endif
        #if RNS_MODULI_SIZE > 44
                result[44] = cuda::mod_psub(x[44], y[44], cuda::RNS_MODULI[44]);
        #endif
        #if RNS_MODULI_SIZE > 45
                result[45] = cuda::mod_psub(x[45], y[45], cuda::RNS_MODULI[45]);
        #endif
        #if RNS_MODULI_SIZE > 46
                result[46] = cuda::mod_psub(x[46], y[46], cuda::RNS_MODULI[46]);
        #endif
        #if RNS_MODULI_SIZE > 47
                result[47] = cuda::mod_psub(x[47], y[47], cuda::RNS_MODULI[47]);
        #endif
        #if RNS_MODULI_SIZE > 48
                result[48] = cuda::mod_psub(x[48], y[48], cuda::RNS_MODULI[48]);
        #endif
        #if RNS_MODULI_SIZE > 49
                result[49] = cuda::mod_psub(x[49], y[49], cuda::RNS_MODULI[49]);
        #endif
        #if RNS_MODULI_SIZE > 50
                result[50] = cuda::mod_psub(x[50], y[50], cuda::RNS_MODULI[50]);
        #endif
        #if RNS_MODULI_SIZE > 51
                result[51] = cuda::mod_psub(x[51], y[51], cuda::RNS_MODULI[51]);
        #endif
        #if RNS_MODULI_SIZE > 52
                result[52] = cuda::mod_psub(x[52], y[52], cuda::RNS_MODULI[52]);
        #endif
        #if RNS_MODULI_SIZE > 53
                result[53] = cuda::mod_psub(x[53], y[53], cuda::RNS_MODULI[53]);
        #endif
        #if RNS_MODULI_SIZE > 54
                result[54] = cuda::mod_psub(x[54], y[54], cuda::RNS_MODULI[54]);
        #endif
        #if RNS_MODULI_SIZE > 55
                result[55] = cuda::mod_psub(x[55], y[55], cuda::RNS_MODULI[55]);
        #endif
        #if RNS_MODULI_SIZE > 56
                result[56] = cuda::mod_psub(x[56], y[56], cuda::RNS_MODULI[56]);
        #endif
        #if RNS_MODULI_SIZE > 57
                result[57] = cuda::mod_psub(x[57], y[57], cuda::RNS_MODULI[57]);
        #endif
        #if RNS_MODULI_SIZE > 58
                result[58] = cuda::mod_psub(x[58], y[58], cuda::RNS_MODULI[58]);
        #endif
        #if RNS_MODULI_SIZE > 59
                result[59] = cuda::mod_psub(x[59], y[59], cuda::RNS_MODULI[59]);
        #endif
        #if RNS_MODULI_SIZE > 60
                result[60] = cuda::mod_psub(x[60], y[60], cuda::RNS_MODULI[60]);
        #endif
        #if RNS_MODULI_SIZE > 61
                result[61] = cuda::mod_psub(x[61], y[61], cuda::RNS_MODULI[61]);
        #endif
        #if RNS_MODULI_SIZE > 62
                result[62] = cuda::mod_psub(x[62], y[62], cuda::RNS_MODULI[62]);
        #endif
        #if RNS_MODULI_SIZE > 63
                result[63] = cuda::mod_psub(x[63], y[63], cuda::RNS_MODULI[63]);
        #endif
        #if RNS_MODULI_SIZE > 64
                result[64] = cuda::mod_psub(x[64], y[64], cuda::RNS_MODULI[64]);
        #endif
        #if RNS_MODULI_SIZE > 65
                result[65] = cuda::mod_psub(x[65], y[65], cuda::RNS_MODULI[65]);
        #endif
        #if RNS_MODULI_SIZE > 66
                result[66] = cuda::mod_psub(x[66], y[66], cuda::RNS_MODULI[66]);
        #endif
        #if RNS_MODULI_SIZE > 67
                result[67] = cuda::mod_psub(x[67], y[67], cuda::RNS_MODULI[67]);
        #endif
        #if RNS_MODULI_SIZE > 68
                result[68] = cuda::mod_psub(x[68], y[68], cuda::RNS_MODULI[68]);
        #endif
        #if RNS_MODULI_SIZE > 69
                result[69] = cuda::mod_psub(x[69], y[69], cuda::RNS_MODULI[69]);
        #endif
        #if RNS_MODULI_SIZE > 70
                result[70] = cuda::mod_psub(x[70], y[70], cuda::RNS_MODULI[70]);
        #endif
        #if RNS_MODULI_SIZE > 71
                result[71] = cuda::mod_psub(x[71], y[71], cuda::RNS_MODULI[71]);
        #endif
        #if RNS_MODULI_SIZE > 72
                result[72] = cuda::mod_psub(x[72], y[72], cuda::RNS_MODULI[72]);
        #endif
        #if RNS_MODULI_SIZE > 73
                result[73] = cuda::mod_psub(x[73], y[73], cuda::RNS_MODULI[73]);
        #endif
        #if RNS_MODULI_SIZE > 74
                result[74] = cuda::mod_psub(x[74], y[74], cuda::RNS_MODULI[74]);
        #endif
        #if RNS_MODULI_SIZE > 75
                result[75] = cuda::mod_psub(x[75], y[75], cuda::RNS_MODULI[75]);
        #endif
        #if RNS_MODULI_SIZE > 76
                result[76] = cuda::mod_psub(x[76], y[76], cuda::RNS_MODULI[76]);
        #endif
        #if RNS_MODULI_SIZE > 77
                result[77] = cuda::mod_psub(x[77], y[77], cuda::RNS_MODULI[77]);
        #endif
        #if RNS_MODULI_SIZE > 78
                result[78] = cuda::mod_psub(x[78], y[78], cuda::RNS_MODULI[78]);
        #endif
        #if RNS_MODULI_SIZE > 79
                result[79] = cuda::mod_psub(x[79], y[79], cuda::RNS_MODULI[79]);
        #endif
        #if RNS_MODULI_SIZE > 80
                result[80] = cuda::mod_psub(x[80], y[80], cuda::RNS_MODULI[80]);
        #endif
        #if RNS_MODULI_SIZE > 81
                result[81] = cuda::mod_psub(x[81], y[81], cuda::RNS_MODULI[81]);
        #endif
        #if RNS_MODULI_SIZE > 82
                result[82] = cuda::mod_psub(x[82], y[82], cuda::RNS_MODULI[82]);
        #endif
        #if RNS_MODULI_SIZE > 83
                result[83] = cuda::mod_psub(x[83], y[83], cuda::RNS_MODULI[83]);
        #endif
        #if RNS_MODULI_SIZE > 84
                result[84] = cuda::mod_psub(x[84], y[84], cuda::RNS_MODULI[84]);
        #endif
        #if RNS_MODULI_SIZE > 85
                result[85] = cuda::mod_psub(x[85], y[85], cuda::RNS_MODULI[85]);
        #endif
        #if RNS_MODULI_SIZE > 86
                result[86] = cuda::mod_psub(x[86], y[86], cuda::RNS_MODULI[86]);
        #endif
        #if RNS_MODULI_SIZE > 87
                result[87] = cuda::mod_psub(x[87], y[87], cuda::RNS_MODULI[87]);
        #endif
        #if RNS_MODULI_SIZE > 88
                result[88] = cuda::mod_psub(x[88], y[88], cuda::RNS_MODULI[88]);
        #endif
        #if RNS_MODULI_SIZE > 89
                result[89] = cuda::mod_psub(x[89], y[89], cuda::RNS_MODULI[89]);
        #endif
        #if RNS_MODULI_SIZE > 90
                result[90] = cuda::mod_psub(x[90], y[90], cuda::RNS_MODULI[90]);
        #endif
        #if RNS_MODULI_SIZE > 91
                result[91] = cuda::mod_psub(x[91], y[91], cuda::RNS_MODULI[91]);
        #endif
        #if RNS_MODULI_SIZE > 92
                result[92] = cuda::mod_psub(x[92], y[92], cuda::RNS_MODULI[92]);
        #endif
        #if RNS_MODULI_SIZE > 93
                result[93] = cuda::mod_psub(x[93], y[93], cuda::RNS_MODULI[93]);
        #endif
        #if RNS_MODULI_SIZE > 94
                result[94] = cuda::mod_psub(x[94], y[94], cuda::RNS_MODULI[94]);
        #endif
        #if RNS_MODULI_SIZE > 95
                result[95] = cuda::mod_psub(x[95], y[95], cuda::RNS_MODULI[95]);
        #endif
        #if RNS_MODULI_SIZE > 96
                result[96] = cuda::mod_psub(x[96], y[96], cuda::RNS_MODULI[96]);
        #endif
        #if RNS_MODULI_SIZE > 97
                result[97] = cuda::mod_psub(x[97], y[97], cuda::RNS_MODULI[97]);
        #endif
        #if RNS_MODULI_SIZE > 98
                result[98] = cuda::mod_psub(x[98], y[98], cuda::RNS_MODULI[98]);
        #endif
        #if RNS_MODULI_SIZE > 99
                result[99] = cuda::mod_psub(x[99], y[99], cuda::RNS_MODULI[99]);
        #endif
        #if RNS_MODULI_SIZE > 100
                result[100] = cuda::mod_psub(x[100], y[100], cuda::RNS_MODULI[100]);
        #endif
        #if RNS_MODULI_SIZE > 101
                result[101] = cuda::mod_psub(x[101], y[101], cuda::RNS_MODULI[101]);
        #endif
        #if RNS_MODULI_SIZE > 102
                result[102] = cuda::mod_psub(x[102], y[102], cuda::RNS_MODULI[102]);
        #endif
        #if RNS_MODULI_SIZE > 103
                result[103] = cuda::mod_psub(x[103], y[103], cuda::RNS_MODULI[103]);
        #endif
        #if RNS_MODULI_SIZE > 104
                result[104] = cuda::mod_psub(x[104], y[104], cuda::RNS_MODULI[104]);
        #endif
        #if RNS_MODULI_SIZE > 105
                result[105] = cuda::mod_psub(x[105], y[105], cuda::RNS_MODULI[105]);
        #endif
        #if RNS_MODULI_SIZE > 106
                result[106] = cuda::mod_psub(x[106], y[106], cuda::RNS_MODULI[106]);
        #endif
        #if RNS_MODULI_SIZE > 107
                result[107] = cuda::mod_psub(x[107], y[107], cuda::RNS_MODULI[107]);
        #endif
        #if RNS_MODULI_SIZE > 108
                result[108] = cuda::mod_psub(x[108], y[108], cuda::RNS_MODULI[108]);
        #endif
        #if RNS_MODULI_SIZE > 109
                result[109] = cuda::mod_psub(x[109], y[109], cuda::RNS_MODULI[109]);
        #endif
        #if RNS_MODULI_SIZE > 110
                result[110] = cuda::mod_psub(x[110], y[110], cuda::RNS_MODULI[110]);
        #endif
        #if RNS_MODULI_SIZE > 111
                result[111] = cuda::mod_psub(x[111], y[111], cuda::RNS_MODULI[111]);
        #endif
        #if RNS_MODULI_SIZE > 112
                result[112] = cuda::mod_psub(x[112], y[112], cuda::RNS_MODULI[112]);
        #endif
        #if RNS_MODULI_SIZE > 113
                result[113] = cuda::mod_psub(x[113], y[113], cuda::RNS_MODULI[113]);
        #endif
        #if RNS_MODULI_SIZE > 114
                result[114] = cuda::mod_psub(x[114], y[114], cuda::RNS_MODULI[114]);
        #endif
        #if RNS_MODULI_SIZE > 115
                result[115] = cuda::mod_psub(x[115], y[115], cuda::RNS_MODULI[115]);
        #endif
        #if RNS_MODULI_SIZE > 116
                result[116] = cuda::mod_psub(x[116], y[116], cuda::RNS_MODULI[116]);
        #endif
        #if RNS_MODULI_SIZE > 117
                result[117] = cuda::mod_psub(x[117], y[117], cuda::RNS_MODULI[117]);
        #endif
        #if RNS_MODULI_SIZE > 118
                result[118] = cuda::mod_psub(x[118], y[118], cuda::RNS_MODULI[118]);
        #endif
        #if RNS_MODULI_SIZE > 119
                result[119] = cuda::mod_psub(x[119], y[119], cuda::RNS_MODULI[119]);
        #endif
        #if RNS_MODULI_SIZE > 120
                result[120] = cuda::mod_psub(x[120], y[120], cuda::RNS_MODULI[120]);
        #endif
        #if RNS_MODULI_SIZE > 121
                result[121] = cuda::mod_psub(x[121], y[121], cuda::RNS_MODULI[121]);
        #endif
        #if RNS_MODULI_SIZE > 122
                result[122] = cuda::mod_psub(x[122], y[122], cuda::RNS_MODULI[122]);
        #endif
        #if RNS_MODULI_SIZE > 123
                result[123] = cuda::mod_psub(x[123], y[123], cuda::RNS_MODULI[123]);
        #endif
        #if RNS_MODULI_SIZE > 124
                result[124] = cuda::mod_psub(x[124], y[124], cuda::RNS_MODULI[124]);
        #endif
        #if RNS_MODULI_SIZE > 125
                result[125] = cuda::mod_psub(x[125], y[125], cuda::RNS_MODULI[125]);
        #endif
        #if RNS_MODULI_SIZE > 126
                result[126] = cuda::mod_psub(x[126], y[126], cuda::RNS_MODULI[126]);
        #endif
        #if RNS_MODULI_SIZE > 127
                result[127] = cuda::mod_psub(x[127], y[127], cuda::RNS_MODULI[127]);
        #endif
        #if RNS_MODULI_SIZE > 128
                result[128] = cuda::mod_psub(x[128], y[128], cuda::RNS_MODULI[128]);
        #endif
        #if RNS_MODULI_SIZE > 129
                result[129] = cuda::mod_psub(x[129], y[129], cuda::RNS_MODULI[129]);
        #endif
        #if RNS_MODULI_SIZE > 130
                result[130] = cuda::mod_psub(x[130], y[130], cuda::RNS_MODULI[130]);
        #endif
        #if RNS_MODULI_SIZE > 131
                result[131] = cuda::mod_psub(x[131], y[131], cuda::RNS_MODULI[131]);
        #endif
        #if RNS_MODULI_SIZE > 132
                result[132] = cuda::mod_psub(x[132], y[132], cuda::RNS_MODULI[132]);
        #endif
        #if RNS_MODULI_SIZE > 133
                result[133] = cuda::mod_psub(x[133], y[133], cuda::RNS_MODULI[133]);
        #endif
        #if RNS_MODULI_SIZE > 134
                result[134] = cuda::mod_psub(x[134], y[134], cuda::RNS_MODULI[134]);
        #endif
        #if RNS_MODULI_SIZE > 135
                result[135] = cuda::mod_psub(x[135], y[135], cuda::RNS_MODULI[135]);
        #endif
        #if RNS_MODULI_SIZE > 136
                result[136] = cuda::mod_psub(x[136], y[136], cuda::RNS_MODULI[136]);
        #endif
        #if RNS_MODULI_SIZE > 137
                result[137] = cuda::mod_psub(x[137], y[137], cuda::RNS_MODULI[137]);
        #endif
        #if RNS_MODULI_SIZE > 138
                result[138] = cuda::mod_psub(x[138], y[138], cuda::RNS_MODULI[138]);
        #endif
        #if RNS_MODULI_SIZE > 139
                result[139] = cuda::mod_psub(x[139], y[139], cuda::RNS_MODULI[139]);
        #endif
        #if RNS_MODULI_SIZE > 140
                result[140] = cuda::mod_psub(x[140], y[140], cuda::RNS_MODULI[140]);
        #endif
        #if RNS_MODULI_SIZE > 141
                result[141] = cuda::mod_psub(x[141], y[141], cuda::RNS_MODULI[141]);
        #endif
        #if RNS_MODULI_SIZE > 142
                result[142] = cuda::mod_psub(x[142], y[142], cuda::RNS_MODULI[142]);
        #endif
        #if RNS_MODULI_SIZE > 143
                result[143] = cuda::mod_psub(x[143], y[143], cuda::RNS_MODULI[143]);
        #endif
        #if RNS_MODULI_SIZE > 144
                result[144] = cuda::mod_psub(x[144], y[144], cuda::RNS_MODULI[144]);
        #endif
        #if RNS_MODULI_SIZE > 145
                result[145] = cuda::mod_psub(x[145], y[145], cuda::RNS_MODULI[145]);
        #endif
        #if RNS_MODULI_SIZE > 146
                result[146] = cuda::mod_psub(x[146], y[146], cuda::RNS_MODULI[146]);
        #endif
        #if RNS_MODULI_SIZE > 147
                result[147] = cuda::mod_psub(x[147], y[147], cuda::RNS_MODULI[147]);
        #endif
        #if RNS_MODULI_SIZE > 148
                result[148] = cuda::mod_psub(x[148], y[148], cuda::RNS_MODULI[148]);
        #endif
        #if RNS_MODULI_SIZE > 149
                result[149] = cuda::mod_psub(x[149], y[149], cuda::RNS_MODULI[149]);
        #endif
        #if RNS_MODULI_SIZE > 150
                result[150] = cuda::mod_psub(x[150], y[150], cuda::RNS_MODULI[150]);
        #endif
        #if RNS_MODULI_SIZE > 151
                result[151] = cuda::mod_psub(x[151], y[151], cuda::RNS_MODULI[151]);
        #endif
        #if RNS_MODULI_SIZE > 152
                result[152] = cuda::mod_psub(x[152], y[152], cuda::RNS_MODULI[152]);
        #endif
        #if RNS_MODULI_SIZE > 153
                result[153] = cuda::mod_psub(x[153], y[153], cuda::RNS_MODULI[153]);
        #endif
        #if RNS_MODULI_SIZE > 154
                result[154] = cuda::mod_psub(x[154], y[154], cuda::RNS_MODULI[154]);
        #endif
        #if RNS_MODULI_SIZE > 155
                result[155] = cuda::mod_psub(x[155], y[155], cuda::RNS_MODULI[155]);
        #endif
        #if RNS_MODULI_SIZE > 156
                result[156] = cuda::mod_psub(x[156], y[156], cuda::RNS_MODULI[156]);
        #endif
        #if RNS_MODULI_SIZE > 157
                result[157] = cuda::mod_psub(x[157], y[157], cuda::RNS_MODULI[157]);
        #endif
        #if RNS_MODULI_SIZE > 158
                result[158] = cuda::mod_psub(x[158], y[158], cuda::RNS_MODULI[158]);
        #endif
        #if RNS_MODULI_SIZE > 159
                result[159] = cuda::mod_psub(x[159], y[159], cuda::RNS_MODULI[159]);
        #endif
        #if RNS_MODULI_SIZE > 160
                result[160] = cuda::mod_psub(x[160], y[160], cuda::RNS_MODULI[160]);
        #endif
        #if RNS_MODULI_SIZE > 161
                result[161] = cuda::mod_psub(x[161], y[161], cuda::RNS_MODULI[161]);
        #endif
        #if RNS_MODULI_SIZE > 162
                result[162] = cuda::mod_psub(x[162], y[162], cuda::RNS_MODULI[162]);
        #endif
        #if RNS_MODULI_SIZE > 163
                result[163] = cuda::mod_psub(x[163], y[163], cuda::RNS_MODULI[163]);
        #endif
        #if RNS_MODULI_SIZE > 164
                result[164] = cuda::mod_psub(x[164], y[164], cuda::RNS_MODULI[164]);
        #endif
        #if RNS_MODULI_SIZE > 165
                result[165] = cuda::mod_psub(x[165], y[165], cuda::RNS_MODULI[165]);
        #endif
        #if RNS_MODULI_SIZE > 166
                result[166] = cuda::mod_psub(x[166], y[166], cuda::RNS_MODULI[166]);
        #endif
        #if RNS_MODULI_SIZE > 167
                result[167] = cuda::mod_psub(x[167], y[167], cuda::RNS_MODULI[167]);
        #endif
        #if RNS_MODULI_SIZE > 168
                result[168] = cuda::mod_psub(x[168], y[168], cuda::RNS_MODULI[168]);
        #endif
        #if RNS_MODULI_SIZE > 169
                result[169] = cuda::mod_psub(x[169], y[169], cuda::RNS_MODULI[169]);
        #endif
        #if RNS_MODULI_SIZE > 170
                result[170] = cuda::mod_psub(x[170], y[170], cuda::RNS_MODULI[170]);
        #endif
        #if RNS_MODULI_SIZE > 171
                result[171] = cuda::mod_psub(x[171], y[171], cuda::RNS_MODULI[171]);
        #endif
        #if RNS_MODULI_SIZE > 172
                result[172] = cuda::mod_psub(x[172], y[172], cuda::RNS_MODULI[172]);
        #endif
        #if RNS_MODULI_SIZE > 173
                result[173] = cuda::mod_psub(x[173], y[173], cuda::RNS_MODULI[173]);
        #endif
        #if RNS_MODULI_SIZE > 174
                result[174] = cuda::mod_psub(x[174], y[174], cuda::RNS_MODULI[174]);
        #endif
        #if RNS_MODULI_SIZE > 175
                result[175] = cuda::mod_psub(x[175], y[175], cuda::RNS_MODULI[175]);
        #endif
        #if RNS_MODULI_SIZE > 176
                result[176] = cuda::mod_psub(x[176], y[176], cuda::RNS_MODULI[176]);
        #endif
        #if RNS_MODULI_SIZE > 177
                result[177] = cuda::mod_psub(x[177], y[177], cuda::RNS_MODULI[177]);
        #endif
        #if RNS_MODULI_SIZE > 178
                result[178] = cuda::mod_psub(x[178], y[178], cuda::RNS_MODULI[178]);
        #endif
        #if RNS_MODULI_SIZE > 179
                result[179] = cuda::mod_psub(x[179], y[179], cuda::RNS_MODULI[179]);
        #endif
        #if RNS_MODULI_SIZE > 180
                result[180] = cuda::mod_psub(x[180], y[180], cuda::RNS_MODULI[180]);
        #endif
        #if RNS_MODULI_SIZE > 181
                result[181] = cuda::mod_psub(x[181], y[181], cuda::RNS_MODULI[181]);
        #endif
        #if RNS_MODULI_SIZE > 182
                result[182] = cuda::mod_psub(x[182], y[182], cuda::RNS_MODULI[182]);
        #endif
        #if RNS_MODULI_SIZE > 183
                result[183] = cuda::mod_psub(x[183], y[183], cuda::RNS_MODULI[183]);
        #endif
        #if RNS_MODULI_SIZE > 184
                result[184] = cuda::mod_psub(x[184], y[184], cuda::RNS_MODULI[184]);
        #endif
        #if RNS_MODULI_SIZE > 185
                result[185] = cuda::mod_psub(x[185], y[185], cuda::RNS_MODULI[185]);
        #endif
        #if RNS_MODULI_SIZE > 186
                result[186] = cuda::mod_psub(x[186], y[186], cuda::RNS_MODULI[186]);
        #endif
        #if RNS_MODULI_SIZE > 187
                result[187] = cuda::mod_psub(x[187], y[187], cuda::RNS_MODULI[187]);
        #endif
        #if RNS_MODULI_SIZE > 188
                result[188] = cuda::mod_psub(x[188], y[188], cuda::RNS_MODULI[188]);
        #endif
        #if RNS_MODULI_SIZE > 189
                result[189] = cuda::mod_psub(x[189], y[189], cuda::RNS_MODULI[189]);
        #endif
        #if RNS_MODULI_SIZE > 190
                result[190] = cuda::mod_psub(x[190], y[190], cuda::RNS_MODULI[190]);
        #endif
        #if RNS_MODULI_SIZE > 191
                result[191] = cuda::mod_psub(x[191], y[191], cuda::RNS_MODULI[191]);
        #endif
        #if RNS_MODULI_SIZE > 192
                result[192] = cuda::mod_psub(x[192], y[192], cuda::RNS_MODULI[192]);
        #endif
        #if RNS_MODULI_SIZE > 193
                result[193] = cuda::mod_psub(x[193], y[193], cuda::RNS_MODULI[193]);
        #endif
        #if RNS_MODULI_SIZE > 194
                result[194] = cuda::mod_psub(x[194], y[194], cuda::RNS_MODULI[194]);
        #endif
        #if RNS_MODULI_SIZE > 195
                result[195] = cuda::mod_psub(x[195], y[195], cuda::RNS_MODULI[195]);
        #endif
        #if RNS_MODULI_SIZE > 196
                result[196] = cuda::mod_psub(x[196], y[196], cuda::RNS_MODULI[196]);
        #endif
        #if RNS_MODULI_SIZE > 197
                result[197] = cuda::mod_psub(x[197], y[197], cuda::RNS_MODULI[197]);
        #endif
        #if RNS_MODULI_SIZE > 198
                result[198] = cuda::mod_psub(x[198], y[198], cuda::RNS_MODULI[198]);
        #endif
        #if RNS_MODULI_SIZE > 199
                result[199] = cuda::mod_psub(x[199], y[199], cuda::RNS_MODULI[199]);
        #endif
    }

} //end of namespace


#endif //MPRES_MODULAR_CUH
