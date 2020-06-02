/*
 *  Multiple-precision GE_LRSCALE function for GPU (BLAS Level-3)
 *  Two-sided diagonal scaling
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

#ifndef MPGELRSCALE_CUH
#define MPGELRSCALE_CUH


#include "../mpmatrix.cuh"
#include "../kernel_config.cuh"
#include "../mblas_enum.cuh"

namespace cuda
{

    /*!
     * Scales a general matrix A on the left side by a diagonal matrix DL and on the right side by a diagonal matrix DR
     * A = DL * A * DR
     * The matrix A should be stored in column-major order.
     * The matrices DL and DR should be stored as vectors.
     *
     * @tparam gridDim1 - number of blocks used to compute the signs, exponents, interval evaluations. A 2D grid of gridDim1 x gridDim1 blocks will be launched
     * @tparam blockDim1 - number of threads per block used to compute the signs, exponents, interval evaluations
     * @tparam gridDim2 - number of blocks used to compute the digits of multiple-precision significands. A 2D grid of gridDim2 x gridDim2 blocks will be launched
     *
     * @param m - specifies the number of rows of the matrix A. The value of m must be greater than zero.
     * @param n - specifies the number of columns of the matrix A. The value of n must be greater than zero.
     * @param DL - pointer to the diagonal matrix that is stored as a vector in the global GPU memory,
     * size at least ( 1+(m-1)*abs( incdl ) ).
     * @param incdl - storage spacing between elements of DL. The value of incdl must not be zero.
     * @param DR - pointer to the diagonal matrix that is stored as a vector in the global GPU memory,
     * size at least ( 1+(n-1)*abs( incdr ) ).
     * @param incdr - storage spacing between elements of DR. The value of incdr must not be zero.
     * @param A - pointer to the array, size lda * n, in the global GPU memory. Before entry, the leading m-by-n part of the array must contain the matrix A.
     * @param lda - specifies the leading dimension of A as declared in the calling (sub)program. The value of lda must be at least max(1, m).
     */
    template<int gridDim1, int blockDim1, int gridDim2>
    void mpgelrscale(const int m, const int n, mp_array_t &DL, const int incdl,  mp_array_t &DR, const int incdr, mp_array_t &A, const int lda){

        //Quick return if possible
        if( (m <= 0) || (n <= 0) ){
            return;
        }
        //Test the input parameters
        if( (incdl == 0) || (incdr == 0) || (lda < MAX(1, m)) ){
            return;
        }

        //Execution configuration
        //  To compute the signs, exponents, and interval evaluations
        dim3 grid1(gridDim1, gridDim1);
        //  To compute the digits in RNS
        dim3 grid2(gridDim2, gridDim2);
        int numThreadsL = (incdl == 1) ? BLOCK_SIZE_FOR_RESIDUES : RNS_MODULI_SIZE;
        int numThreadsR = (incdr == 1) ? BLOCK_SIZE_FOR_RESIDUES : RNS_MODULI_SIZE;
        //  To rounding the result (we do not currently parameterize rounding)
        dim3 block3(16, 16);
        dim3 grid3((m + block3.x - 1) / block3.x, (n + block3.y - 1) / block3.y);

        //A = DL * A

        mp_mat2vec_left_scal_esi_kernel<<<grid1, blockDim1>>> (A, lda, A, lda, DL, incdl, m, n);

        mp_mat2vec_left_scal_digits_kernel<<<grid2, numThreadsL>>> (A, lda, A, lda, DL, incdl, m, n);

        mp_matrix_round_kernel<<<grid3, block3>>> (A, lda, m, n);

        //A = A * DR

        mp_mat2vec_right_scal_esi_kernel<<<grid1, blockDim1>>> (A, lda, A, lda, DR, incdr, m, n);

        mp_mat2vec_right_scal_digits_kernel<<<grid2, numThreadsR>>> (A, lda, A, lda, DR, incdr, m, n);

        mp_matrix_round_kernel<<<grid3, block3>>> (A, lda, m, n);
    }

} // namespace cuda

#endif //MPGELRSCALE_CUH
