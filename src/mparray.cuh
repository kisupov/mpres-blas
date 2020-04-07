/*
 *  Utilities for working with multiple-precision vectors and matrices using the structure-of-arrays layout
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

#ifndef MPRES_MPARRAY_CUH
#define MPRES_MPARRAY_CUH

#include "mpfloat.cuh"

namespace cuda {

    /*!
     * Allocate a multiple-precision vector that holds size elements (in the GPU memory)
     * @param dev_dest - pointer to the vector to be allocated
     */
    void mp_array_init(mp_array_t &dev_dest, unsigned int size) {

        // Allocating digits
        size_t residue_vec_size = RNS_MODULI_SIZE * size * sizeof(int);
        checkDeviceHasErrors(cudaMalloc(&dev_dest.digits, residue_vec_size));

        // Allocating signs
        size_t vec_size = size * sizeof(int);
        checkDeviceHasErrors(cudaMalloc(&dev_dest.sign, vec_size));

        // Allocating exponents
        checkDeviceHasErrors(cudaMalloc(&dev_dest.exp, vec_size));

        // Allocating interval evaluations
        checkDeviceHasErrors(cudaMalloc(&dev_dest.eval, sizeof(er_float_t) * 2 * size));

        // Allocating temporary buffer
        checkDeviceHasErrors(cudaMalloc(&dev_dest.buf, size * sizeof(int4)));

        // Setting the actual size of the vector
        int length = size;
        checkDeviceHasErrors(cudaMalloc(&dev_dest.len, sizeof(int)));
        checkDeviceHasErrors(cudaMemcpy(dev_dest.len, &length, sizeof(int), cudaMemcpyHostToDevice));

        //Error checking
        checkDeviceHasErrors(cudaDeviceSynchronize());
        cudaCheckErrors();
    }

    /*!
     * Clear the GPU memory occupied by a vector
     */
    void mp_array_clear(mp_array_t &dev_dest) {
        checkDeviceHasErrors(cudaFree(dev_dest.digits));
        checkDeviceHasErrors(cudaFree(dev_dest.sign));
        checkDeviceHasErrors(cudaFree(dev_dest.exp));
        checkDeviceHasErrors(cudaFree(dev_dest.eval));
        checkDeviceHasErrors(cudaFree(dev_dest.buf));
        checkDeviceHasErrors(cudaFree(dev_dest.len));
        checkDeviceHasErrors(cudaDeviceSynchronize());
        cudaCheckErrors();
    }

    /*!
     * Convert a regular multiple-precision vector (mp_float_t) allocated on the host to a mp_array_t vector allocated on the GPU
     * @param dev_dest - pointer to the result vector in the GPU memory (must be initialized)
     * @param host_src - pointer to the source vector in the host memory
     * @param size - number of elements in a vector (must be the same as used in mp_array_init)
     */
    void mp_array_host2device(mp_array_t &dev_dest, mp_float_ptr host_src, unsigned int size) {
        int *buffer = new int[size];
        er_float_ptr eval_buffer = new er_float_t[size * 2];
        size_t buffer_size = sizeof(int) * size;

        // Copy residues
        unsigned int offset = 0;
        for (int i = 0; i < size; i++) {
            checkDeviceHasErrors(cudaMemcpy(&dev_dest.digits[offset], host_src[i].digits, RNS_MODULI_SIZE * sizeof(int), cudaMemcpyHostToDevice));
            offset += RNS_MODULI_SIZE;
        }
        cudaDeviceSynchronize();
        // Copy sign
        #pragma omp parallel
        for (int i = 0; i < size; i++) {
            buffer[i] = host_src[i].sign;
        }
        checkDeviceHasErrors(cudaMemcpy(dev_dest.sign, buffer, buffer_size, cudaMemcpyHostToDevice));

        // Copy exponent
        #pragma omp parallel
        for (int i = 0; i < size; i++) {
            buffer[i] = host_src[i].exp;
        }
        checkDeviceHasErrors(cudaMemcpy(dev_dest.exp, buffer, buffer_size, cudaMemcpyHostToDevice));

        //Copy interval evaluations
        #pragma omp parallel
        for (int i = 0; i < size; i++) {
            eval_buffer[i] = host_src[i].eval[0];
            eval_buffer[size + i] = host_src[i].eval[1];
        }
        checkDeviceHasErrors(cudaMemcpy(dev_dest.eval, eval_buffer, sizeof(er_float_t) * size * 2, cudaMemcpyHostToDevice));

        checkDeviceHasErrors(cudaDeviceSynchronize());
        cudaCheckErrors();

        //Cleanup
        delete[] buffer;
        delete[] eval_buffer;
    }

    /*!
     * Convert a multiple-precision mp_array_t vector allocated on the GPU to a regular vector (mp_float_t) allocated on the host
     * @param host_dest - pointer to the result vector in the host memory
     * @param dev_src - pointer to the source vector in the GPU memory
     * @param size - number of elements in a vector
     */
    void mp_array_device2host(mp_float_ptr host_dest, mp_array_t &dev_src, unsigned int size) {
        int *buffer = new int[size];
        er_float_ptr eval_buffer = new er_float_t[size * 2];
        size_t buffer_size = sizeof(int) * size;

        // Copy interval evaluations to buffer
        checkDeviceHasErrors(cudaMemcpy(eval_buffer, dev_src.eval, sizeof(er_float_t) * size * 2, cudaMemcpyDeviceToHost));
        #pragma omp parallel
        for (int i = 0; i < size; i++) {
            host_dest[i].eval[0] = eval_buffer[i];
            host_dest[i].eval[1] = eval_buffer[size + i];
        }

        // Copy sign
        checkDeviceHasErrors(cudaMemcpy(buffer, dev_src.sign, buffer_size, cudaMemcpyDeviceToHost));
        #pragma omp parallel
        for (int i = 0; i < size; i++) {
            host_dest[i].sign = buffer[i];
        }

        // Copy exponent
        checkDeviceHasErrors(cudaMemcpy(buffer, dev_src.exp, buffer_size, cudaMemcpyDeviceToHost));
        #pragma omp parallel
        for (int i = 0; i < size; i++) {
            host_dest[i].exp = buffer[i];
        }

        // Copy residues
        unsigned int offset = 0;

        for (int i = 0; i < size; i++) {
            checkDeviceHasErrors(cudaMemcpy(host_dest[i].digits, &dev_src.digits[offset], RNS_MODULI_SIZE * sizeof(int), cudaMemcpyDeviceToHost));
            offset += RNS_MODULI_SIZE;
        }

        checkDeviceHasErrors(cudaDeviceSynchronize());
        cudaCheckErrors();

        //Cleanup
        delete[] buffer;
        delete[] eval_buffer;
    }

} //end of namespace

#endif //MPRES_MPARRAY_CUH
