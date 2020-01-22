/*
 *  Parallel componentwise operations with vectors of multiple-precision numbers on the GPU
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
#include "kernel_config.cuh"

namespace cuda {

    /********************* Initialization and conversion functions *********************/

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


    /********************* Vector-vector (vv) multiplication kernels *********************/


    /*!
     * Parallel componentwise vector multiplication (result = x * y)
     * Kernel #1 --- Computing the exponents, signs, and interval evaluations (e-s-i)
     * @note For example of usage, see mpdot.cuh
     * @param result - pointer to the result vector in the GPU memory
     * @param incr - storage spacing between elements of result (must be non-zero)
     * @param x - pointer to the first vector in the GPU memory
     * @param incx -  storage spacing between elements of x (must be non-zero)
     * @param y - pointer to the second vector in the GPU memory
     * @param incy -  storage spacing between elements of y (must be non-zero)
     * @param n - operation size
     */
    __global__ void mp_array_mul_esi_vv(mp_array_t result, const int incr, mp_array_t x, const int incx, mp_array_t y, const int incy, const int n) {
        int numberIdx =  blockDim.x * blockIdx.x + threadIdx.x;
        // Actual vector lengths (may differ from the operation size, n)
        int lenx = x.len[0];
        int leny = y.len[0];
        int lenr = result.len[0];

        // code for all increments equal to 1
        if(incr == 1 && incx == 1 && incy == 1){
            while (numberIdx < n) {
                result.sign[numberIdx] = x.sign[numberIdx] ^ y.sign[numberIdx];
                result.exp[numberIdx] = x.exp[numberIdx] + y.exp[numberIdx];
                cuda::er_md_rd(&result.eval[numberIdx], &x.eval[numberIdx], &y.eval[numberIdx], &cuda::RNS_EVAL_UNIT.upp);
                cuda::er_md_ru(&result.eval[lenr + numberIdx], &x.eval[lenx + numberIdx], &y.eval[leny + numberIdx], &cuda::RNS_EVAL_UNIT.low);
                //Go to the next iteration
                numberIdx +=  gridDim.x * blockDim.x;
            }
        }
        // code for unequal increments or equal increments not equal to 1
        else{
            int ix = incx > 0 ? numberIdx * incx : (-n + numberIdx + 1)*incx;
            int iy = incy > 0 ? numberIdx * incy : (-n + numberIdx + 1)*incy;
            int ir = incr > 0 ? numberIdx * incr : (-n + numberIdx + 1)*incr;
            while (numberIdx < n) {
                result.sign[ir] = x.sign[ix] ^ y.sign[iy];
                result.exp[ir] = x.exp[ix] + y.exp[iy];
                cuda::er_md_rd(&result.eval[ir], &x.eval[ix], &y.eval[iy], &cuda::RNS_EVAL_UNIT.upp);
                cuda::er_md_ru(&result.eval[lenr + ir], &x.eval[lenx + ix], &y.eval[leny + iy], &cuda::RNS_EVAL_UNIT.low);
                //Go to the next iteration
                numberIdx +=  gridDim.x * blockDim.x;
                ix += gridDim.x * blockDim.x * incx;
                iy += gridDim.x * blockDim.x * incy;
                ir += gridDim.x * blockDim.x * incr;
            }
        }
    }

    /*!
     * Parallel componentwise vector multiplication (result = x * y)
     * Kernel #2 --- Computing the significands in the RNS (digits)
     * @note For example of usage, see mpdot.cuh
     * @note For this kernel, the block size is specified by either BLOCK_SIZE_FOR_RESIDUES (see kernel_config.cuh)
     * or RNS_MODULI_SIZE as declared in the calling subprogram
     * @param result - pointer to the result vector in the GPU memory
     * @param incr - storage spacing between elements of result (must be non-zero)
     * @param x - pointer to the first vector in the GPU memory
     * @param incx -  storage spacing between elements of x (must be non-zero)
     * @param y - pointer to the second vector in the GPU memory
     * @param incy -  storage spacing between elements of y (must be non-zero)
     * @param n - operation size
     */
    __global__ void mp_array_mul_digits_vv(mp_array_t result, const int incr, mp_array_t x, const int incx, mp_array_t y, const int incy, const int n){
        int lmodul = cuda::RNS_MODULI[threadIdx.x  % RNS_MODULI_SIZE];
        int index = blockIdx.x * blockDim.x + threadIdx.x;

        // code for all increments equal to 1
        if(incr == 1 && incx == 1 && incy == 1) {
            while (index < n * RNS_MODULI_SIZE) {
                result.digits[index] = cuda::mod_mul(x.digits[index], y.digits[index], lmodul);
                //Go to the next iteration
                index += gridDim.x * blockDim.x;
            }
        }
        // code for unequal increments or equal increments not equal to 1
        else{
            int ix = incx > 0 ? (blockIdx.x * blockDim.x * incx + threadIdx.x) : ((-n + blockIdx.x + 1) * blockDim.x * incx + threadIdx.x);
            int iy = incy > 0 ? (blockIdx.x * blockDim.x * incy + threadIdx.x) : ((-n + blockIdx.x + 1) * blockDim.x * incy + threadIdx.x);
            int ir = incr > 0 ? (blockIdx.x * blockDim.x * incr + threadIdx.x) : ((-n + blockIdx.x + 1) * blockDim.x * incr + threadIdx.x);
            while (index < n * RNS_MODULI_SIZE) {
                result.digits[ir] = cuda::mod_mul(x.digits[ix], y.digits[iy], lmodul);
                //Go to the next iteration
                index += gridDim.x * blockDim.x;
                ix += gridDim.x * blockDim.x * incx;
                iy += gridDim.x * blockDim.x * incy;
                ir += gridDim.x * blockDim.x * incr;
            }
        }
    }


    /********************* Vector-scalar (vs) multiplication kernels *********************/


    /*!
     * Parallel componentwise multiplication of a vector by a scalar (result = alpha * x)
     * Kernel #1 --- Computing the exponents, signs, and interval evaluations (e-s-i)
     * @note For example of usage, see mpscal.cuh
     * @param result - pointer to the result vector in the GPU memory
     * @param incr - storage spacing between elements of result (must be non-zero)
     * @param x - pointer to the vector in the GPU memory
     * @param incx -  storage spacing between elements of x (must be non-zero)
     * @param alpha - pointer to the scalar (vector of length one) in the GPU memory
     * @param n - operation size
     */
    __global__ void mp_array_mul_esi_vs(mp_array_t result, const int incr, mp_array_t x, const int incx, mp_array_t alpha, const int n) {
        int numberIdx =  blockDim.x * blockIdx.x + threadIdx.x;
        // Actual vector lengths (may differ from the operation size, n)
        int lenx = x.len[0];
        int lenr = result.len[0];

        int alpha_sign = alpha.sign[0];
        int alpha_exp = alpha.exp[0];

        er_float_t leval0 = alpha.eval[0];
        er_float_t leval1 = alpha.eval[1];

        er_float_ptr  leval0ptr = &leval0;
        er_float_ptr  leval1ptr = &leval1;

        // code for all increments equal to 1
        if(incr == 1 && incx == 1) {
            while (numberIdx < n) {
                result.sign[numberIdx] = x.sign[numberIdx] ^ alpha_sign;
                result.exp[numberIdx] = x.exp[numberIdx] + alpha_exp;
                cuda::er_md_rd(&result.eval[numberIdx], &x.eval[numberIdx], leval0ptr, &cuda::RNS_EVAL_UNIT.upp);
                cuda::er_md_ru(&result.eval[lenr + numberIdx], &x.eval[lenx + numberIdx], leval1ptr, &cuda::RNS_EVAL_UNIT.low);
                //Go to the next iteration
                numberIdx += gridDim.x * blockDim.x;
            }
        }
        // code for unequal increments or equal increments not equal to 1
        else{
            int ix = incx > 0 ? numberIdx * incx : (-n + numberIdx + 1)*incx;
            int ir = incr > 0 ? numberIdx * incr : (-n + numberIdx + 1)*incr;
            while (numberIdx < n) {
                result.sign[ir] = x.sign[ix] ^ alpha_sign;
                result.exp[ir] = x.exp[ix] + alpha_exp;
                cuda::er_md_rd(&result.eval[ir], &x.eval[ix], leval0ptr, &cuda::RNS_EVAL_UNIT.upp);
                cuda::er_md_ru(&result.eval[lenr + ir], &x.eval[lenx + ix], leval1ptr, &cuda::RNS_EVAL_UNIT.low);
                //Go to the next iteration
                numberIdx += gridDim.x * blockDim.x;
                ix += gridDim.x * blockDim.x * incx;
                ir += gridDim.x * blockDim.x * incr;
            }
        }
    }

    /*!
     * Parallel componentwise multiplication of a vector by a scalar (result = alpha * x)
     * Kernel #2 --- Computing the significands in the RNS (digits)
     * @note For example of usage, see mpscal.cuh
     * @note For this kernel, the block size is specified by either BLOCK_SIZE_FOR_RESIDUES (see kernel_config.cuh)
     * or RNS_MODULI_SIZE as declared in the calling subprogram
     * @param result - pointer to the result vector in the GPU memory
     * @param incr - storage spacing between elements of result (must be non-zero)
     * @param x - pointer to the first vector in the GPU memory
     * @param incx -  storage spacing between elements of x (must be non-zero)
     * @param alpha - pointer to the scalar (vector of length one) in the GPU memory
     * @param n - operation size
     */
    __global__ void mp_array_mul_digits_vs(mp_array_t result, const int incr, mp_array_t x, const int incx, mp_array_t alpha, const int n){
        int lmodul = cuda::RNS_MODULI[threadIdx.x % RNS_MODULI_SIZE];
        int lalpha = alpha.digits[threadIdx.x % RNS_MODULI_SIZE];
        int index = blockIdx.x * blockDim.x + threadIdx.x;

        // code for all increments equal to 1
        if(incr == 1 && incx == 1) {
            while (index < n * RNS_MODULI_SIZE) {
                result.digits[index] = cuda::mod_mul(x.digits[index], lalpha, lmodul);
                index += gridDim.x * blockDim.x;
            }
        }
        // code for unequal increments or equal increments not equal to 1
        else {
            int ix = incx > 0 ? (blockIdx.x * blockDim.x * incx + threadIdx.x) : ((-n + blockIdx.x + 1) * blockDim.x * incx + threadIdx.x);
            int ir = incr > 0 ? (blockIdx.x * blockDim.x * incr + threadIdx.x) : ((-n + blockIdx.x + 1) * blockDim.x * incr + threadIdx.x);
            while (index < n * RNS_MODULI_SIZE) {
                result.digits[ir] = cuda::mod_mul(x.digits[ix], lalpha, lmodul);
                //Go to the next iteration
                index += gridDim.x * blockDim.x;
                ix += gridDim.x * blockDim.x * incx;
                ir += gridDim.x * blockDim.x * incr;
            }
        }
    }


    /********************* Vector-vector (vv) addition and subtraction kernels *********************/


    /*!
     * Parallel componentwise vector addition (result = x + y)
     * Kernel #1 --- Computing the exponents, signs, and interval evaluations (e-s-i)
     * @note For example of usage, see mpaxpy.cuh
     * @param result - pointer to the result vector in the GPU memory
     * @param incr - storage spacing between elements of result (must be non-zero)
     * @param x - pointer to the first vector in the GPU memory
     * @param incx -  storage spacing between elements of x (must be non-zero)
     * @param y - pointer to the second vector in the GPU memory
     * @param incy -  storage spacing between elements of y (must be non-zero)
     * @param n - operation size
     */
    __global__ void mp_array_add_esi_vv(mp_array_t result, const int incr, mp_array_t x, const int incx, mp_array_t y, const int incy, const int n){
        int numberIdx =  blockDim.x * blockIdx.x + threadIdx.x;
        // Actual vector lengths (may differ from the operation size, n)
        int lenx = x.len[0];
        int leny = y.len[0];
        int lenr = result.len[0];

        er_float_t eval_x[2];
        er_float_t eval_y[2];
        int exp_x;
        int exp_y;
        int sign_x;
        int sign_y;

        // code for all increments equal to 1
        if(incr == 1 && incx == 1 && incy == 1){
            while (numberIdx < n) {
                eval_x[0] = x.eval[numberIdx];
                eval_x[1] = x.eval[numberIdx + lenx];
                eval_y[0] = y.eval[numberIdx];
                eval_y[1] = y.eval[numberIdx + leny];

                exp_x = x.exp[numberIdx];
                exp_y = y.exp[numberIdx];
                sign_x = x.sign[numberIdx];
                sign_y = y.sign[numberIdx];

                //Exponent alignment
                int dexp = exp_x - exp_y;
                int gamma =  dexp * (dexp > 0); //if dexp > 0, then gamma =  dexp; otherwise gamma = 0
                int theta = -dexp * (dexp < 0); //if dexp < 0, then theta = -dexp; otherwise theta = 0

                int nzx = ((eval_y[1].frac == 0) || (theta + eval_y[1].exp) < cuda::MP_J); //nzx (u) = 1 if x not need be zeroed; otherwise nzx = 0
                int nzy = ((eval_x[1].frac == 0) || (gamma + eval_x[1].exp) < cuda::MP_J); //nzy (v) = 1 if y not need be zeroed; otherwise nzy = 0

                gamma = gamma * nzy; //if nzy = 0 (y needs to be zeroed), then gamma = 0, i.e. we will multiply x by 2^0 without actually changing the value of x
                theta = theta * nzx; //if nzx = 0 (x needs to be zeroed), then theta = 0, i.e. we will multiply y by 2^0 without actually changing the value of y

                //Correction of the exponents
                exp_x = (exp_x - gamma) * nzx; //if x needs to be zeroed, exp_x will be equal to 0
                exp_y = (exp_y - theta) * nzy; //if y needs to be zeroed, exp_y will be equal to 0

                //Correction of the signs
                sign_x *= nzx;
                sign_y *= nzy;

                int factor_x = (1 - 2 * sign_x) * nzx; //-1 if  x is negative, 1 if x is positive, 0 if x needs to be zeroed (the exponent of x is too small)
                int factor_y = (1 - 2 * sign_y) * nzy; //-1 if  y is negative, 1 if y is positive, 0 if y needs to be zeroed (the exponent of y is too small)

                //Correction of the interval evaluations (multiplication by 2^gamma or 2^theta)
                eval_x[0].exp += gamma;
                eval_x[1].exp += gamma;
                eval_y[0].exp += theta;
                eval_y[1].exp += theta;

                //Change the signs of the interval evaluation bounds when the number is negative
                //The signs will not change when the number is positive
                //If the number needs to be reset, then the bounds will also be reset
                eval_x[0].frac *=  factor_x;
                eval_x[1].frac *=  factor_x;
                eval_y[0].frac *=  factor_y;
                eval_y[1].frac *=  factor_y;

                //Interval addition
                cuda::er_add_rd(&result.eval[numberIdx], &eval_x[sign_x], &eval_y[sign_y]);
                cuda::er_add_ru(&result.eval[numberIdx + lenr], &eval_x[1 - sign_x], &eval_y[1 - sign_y]);

                //Calculation of the exponent and preliminary calculation of the sign (the sign will be changed if restoring is required)
                result.sign[numberIdx] = 0;
                result.exp[numberIdx] = (exp_x == 0) ? exp_y : exp_x;

                //Restoring the negative result
                //int plus  = result.eval[numberIdx].frac >= 0 && result.eval[numberIdx + lenr].frac >= 0;
                int minus = result.eval[numberIdx].frac < 0 && result.eval[numberIdx + lenr].frac < 0;
                if(minus){
                    result.sign[numberIdx] = 1;
                    er_float_t tmp = result.eval[numberIdx];
                    result.eval[numberIdx].frac = -result.eval[numberIdx + lenr].frac;
                    result.eval[numberIdx].exp  = result.eval[numberIdx + lenr].exp;
                    result.eval[numberIdx + lenr].frac = -tmp.frac;
                    result.eval[numberIdx + lenr].exp  = tmp.exp;
                }

                //Storing data for Kernel #2 in the buffer
                int4 intBuf;
                intBuf.x = gamma;
                intBuf.y = theta;
                intBuf.z = factor_x;
                intBuf.w = factor_y;
                result.buf[numberIdx] = intBuf;

                //Go to the next iteration
                numberIdx +=  gridDim.x * blockDim.x;
            }
        }
        // code for unequal increments or equal increments not equal to 1
        else{
            int ix = incx > 0 ? numberIdx * incx : (-n + numberIdx + 1)*incx;
            int iy = incy > 0 ? numberIdx * incy : (-n + numberIdx + 1)*incy;
            int ir = incr > 0 ? numberIdx * incr : (-n + numberIdx + 1)*incr;
            while (numberIdx < n) {
                eval_x[0] = x.eval[ix];
                eval_x[1] = x.eval[ix + lenx];
                eval_y[0] = y.eval[iy];
                eval_y[1] = y.eval[iy + leny];

                exp_x = x.exp[ix];
                exp_y = y.exp[iy];
                sign_x = x.sign[ix];
                sign_y = y.sign[iy];

                int dexp = exp_x - exp_y;
                int gamma =  dexp * (dexp > 0);
                int theta = -dexp * (dexp < 0);

                int nzx = ((eval_y[1].frac == 0) || (theta + eval_y[1].exp) < cuda::MP_J);
                int nzy = ((eval_x[1].frac == 0) || (gamma + eval_x[1].exp) < cuda::MP_J);

                gamma = gamma * nzy;
                theta = theta * nzx;

                exp_x = (exp_x - gamma) * nzx;
                exp_y = (exp_y - theta) * nzy;

                sign_x *= nzx;
                sign_y *= nzy;

                int factor_x = (1 - 2 * sign_x) * nzx;
                int factor_y = (1 - 2 * sign_y) * nzy;

                eval_x[0].exp += gamma;
                eval_x[1].exp += gamma;
                eval_y[0].exp += theta;
                eval_y[1].exp += theta;

                eval_x[0].frac *=  factor_x;
                eval_x[1].frac *=  factor_x;
                eval_y[0].frac *=  factor_y;
                eval_y[1].frac *=  factor_y;

                cuda::er_add_rd(&result.eval[ir], &eval_x[sign_x], &eval_y[sign_y]);
                cuda::er_add_ru(&result.eval[ir + lenr], &eval_x[1 - sign_x], &eval_y[1 - sign_y]);

                result.sign[ir] = 0;
                result.exp[ir] = (exp_x == 0) ? exp_y : exp_x;

                //int plus  = result.eval[ir].frac >= 0 && result.eval[ir + lenr].frac >= 0;
                int minus = result.eval[ir].frac < 0 && result.eval[ir + lenr].frac < 0;
                if(minus){
                    result.sign[ir] = 1;
                    er_float_t tmp = result.eval[ir];
                    result.eval[ir].frac = -result.eval[ir+lenr].frac;
                    result.eval[ir].exp  = result.eval[ir+lenr].exp;
                    result.eval[ir+lenr].frac = -tmp.frac;
                    result.eval[ir+lenr].exp  = tmp.exp;
                }

                int4 intBuf;
                intBuf.x = gamma;
                intBuf.y = theta;
                intBuf.z = factor_x;
                intBuf.w = factor_y;
                result.buf[ir] = intBuf;

                //Go to the next iteration
                numberIdx +=  gridDim.x * blockDim.x;
                ix += gridDim.x * blockDim.x * incx;
                iy += gridDim.x * blockDim.x * incy;
                ir += gridDim.x * blockDim.x * incr;
            }
        }
    }

    /*!
     * Parallel componentwise vector addition (result = x + y)
     * Kernel #2 --- Computing the significands in the RNS (digits)
     * @note For example of usage, see mpaxpy.cuh
     * @note For this kernel, the block size is specified by either BLOCK_SIZE_FOR_RESIDUES (see kernel_config.cuh)
     * or RNS_MODULI_SIZE as declared in the calling subprogram
     * @param result - pointer to the result vector in the GPU memory
     * @param incr - storage spacing between elements of result (must be non-zero)
     * @param x - pointer to the first vector in the GPU memory
     * @param incx -  storage spacing between elements of x (must be non-zero)
     * @param y - pointer to the second vector in the GPU memory
     * @param incy -  storage spacing between elements of y (must be non-zero)
     * @param n - operation size
     */
    __global__ void mp_array_add_digits_vv(mp_array_t result, const int incr, mp_array_t x, const int incx, mp_array_t y, const int incy, const int n){
        int lmodul = cuda::RNS_MODULI[threadIdx.x % RNS_MODULI_SIZE];
        int index = blockIdx.x * blockDim.x + threadIdx.x; // Индекс остатка в векторе

        // code for all increments equal to 1
        if(incr == 1 && incx == 1 && incy == 1) {
            int numberIdx = (blockIdx.x * blockDim.x + threadIdx.x) / RNS_MODULI_SIZE; // Номер числа в векторе
            while (index < n * RNS_MODULI_SIZE) {
                int4 intBuf = result.buf[numberIdx];
                int residue = cuda::mod_axby(
                        intBuf.z * x.digits[index],
                        cuda::RNS_POW2[intBuf.x][threadIdx.x % RNS_MODULI_SIZE],
                        intBuf.w * y.digits[index],
                        cuda::RNS_POW2[intBuf.y][threadIdx.x % RNS_MODULI_SIZE],
                        lmodul,
                        cuda::RNS_MODULI_RECIPROCAL[threadIdx.x % RNS_MODULI_SIZE]);
                //Restoring the negative result
                if (result.sign[numberIdx] == 1) {
                    residue = cuda::mod_sub(lmodul, residue, lmodul);
                }
                result.digits[index] = residue < 0 ? residue + lmodul : residue;
                //Go to the next iteration
                index += gridDim.x * blockDim.x;
                numberIdx += gridDim.x * blockDim.x / RNS_MODULI_SIZE;
            }
        }
        // code for unequal increments or equal increments not equal to 1
        else {
            int ix = incx > 0 ? (blockIdx.x * blockDim.x * incx + threadIdx.x) : ((-n + blockIdx.x + 1) * blockDim.x * incx + threadIdx.x);
            int iy = incy > 0 ? (blockIdx.x * blockDim.x * incy + threadIdx.x) : ((-n + blockIdx.x + 1) * blockDim.x * incy + threadIdx.x);
            int ir = incr > 0 ? (blockIdx.x * blockDim.x * incr + threadIdx.x) : ((-n + blockIdx.x + 1) * blockDim.x * incr + threadIdx.x);
            int numberIdx = ir / RNS_MODULI_SIZE; // Номер числа в векторе
            while (index < n * RNS_MODULI_SIZE) {
                int4 intBuf = result.buf[numberIdx];
                int residue = cuda::mod_axby(
                        intBuf.z * x.digits[ix],
                        cuda::RNS_POW2[intBuf.x][threadIdx.x % RNS_MODULI_SIZE],
                        intBuf.w * y.digits[iy],
                        cuda::RNS_POW2[intBuf.y][threadIdx.x % RNS_MODULI_SIZE],
                        lmodul,
                        cuda::RNS_MODULI_RECIPROCAL[threadIdx.x % RNS_MODULI_SIZE]);
                //Restoring the negative result
                if (result.sign[numberIdx] == 1) {
                    residue = cuda::mod_sub(lmodul, residue, lmodul);
                }
                result.digits[ir] = residue < 0 ? residue + lmodul : residue;
                //Go to the next iteration
                index += gridDim.x * blockDim.x;
                ix += gridDim.x * blockDim.x * incx;
                iy += gridDim.x * blockDim.x * incy;
                ir += gridDim.x * blockDim.x * incr;
                numberIdx = ir / RNS_MODULI_SIZE; //may be optimized
            }
        }
    }

    /*!
     * Parallel componentwise vector subtraction (result = x - y)
     * Kernel #1 --- Computing the exponents, signs, and interval evaluations (e-s-i)
     * @note Kernel 2 is the same as for vector addition algorithm, i.e, mp_array_add_digits_vv
     * @note For example of usage, see mprot.cuh
     * @param result - pointer to the result vector in the GPU memory
     * @param incr - storage spacing between elements of result (must be non-zero)
     * @param x - pointer to the first vector in the GPU memory
     * @param incx -  storage spacing between elements of x (must be non-zero)
     * @param y - pointer to the second vector in the GPU memory
     * @param incy -  storage spacing between elements of y (must be non-zero)
     * @param n - operation size
     */
    __global__ void mp_array_sub_esi_vv(mp_array_t result, const int incr, mp_array_t x, const int incx, mp_array_t y, const int incy, const int n){
        int numberIdx =  blockDim.x * blockIdx.x + threadIdx.x;
        // Actual vector lengths (may differ from the operation size, n)
        int lenx = x.len[0];
        int leny = y.len[0];
        int lenr = result.len[0];

        er_float_t eval_x[2];
        er_float_t eval_y[2];
        int exp_x;
        int exp_y;
        int sign_x;
        int sign_y;

        // code for all increments equal to 1
        if(incr == 1 && incx == 1 && incy == 1){
            while (numberIdx < n) {
                eval_x[0] = x.eval[numberIdx];
                eval_x[1] = x.eval[numberIdx + lenx];
                eval_y[0] = y.eval[numberIdx];
                eval_y[1] = y.eval[numberIdx + leny];

                exp_x = x.exp[numberIdx];
                exp_y = y.exp[numberIdx];
                sign_x = x.sign[numberIdx];
                sign_y = y.sign[numberIdx] ^ 1; //invert the sign of y to perform subtraction instead of addition

                //Exponent alignment
                int dexp = exp_x - exp_y;
                int gamma =  dexp * (dexp > 0); //if dexp > 0, then gamma =  dexp; otherwise gamma = 0
                int theta = -dexp * (dexp < 0); //if dexp < 0, then theta = -dexp; otherwise theta = 0

                int nzx = ((eval_y[1].frac == 0) || (theta + eval_y[1].exp) < cuda::MP_J); //nzx (u) = 1 if x not need be zeroed; otherwise nzx = 0
                int nzy = ((eval_x[1].frac == 0) || (gamma + eval_x[1].exp) < cuda::MP_J); //nzy (v) = 1 if y not need be zeroed; otherwise nzy = 0

                gamma = gamma * nzy; //if nzy = 0 (y needs to be zeroed), then gamma = 0, i.e. we will multiply x by 2^0 without actually changing the value of x
                theta = theta * nzx; //if nzx = 0 (x needs to be zeroed), then theta = 0, i.e. we will multiply y by 2^0 without actually changing the value of y

                //Correction of the exponents
                exp_x = (exp_x - gamma) * nzx; //if x needs to be zeroed, exp_x will be equal to 0
                exp_y = (exp_y - theta) * nzy; //if y needs to be zeroed, exp_y will be equal to 0

                //Correction of the signs
                sign_x *= nzx;
                sign_y *= nzy;

                int factor_x = (1 - 2 * sign_x) * nzx; //-1 if  x is negative, 1 if x is positive, 0 if x needs to be zeroed (the exponent of x is too small)
                int factor_y = (1 - 2 * sign_y) * nzy; //-1 if  y is negative, 1 if y is positive, 0 if y needs to be zeroed (the exponent of y is too small)

                //Correction of the interval evaluations (multiplication by 2^gamma or 2^theta)
                eval_x[0].exp += gamma;
                eval_x[1].exp += gamma;
                eval_y[0].exp += theta;
                eval_y[1].exp += theta;

                //Change the signs of the interval evaluation bounds when the number is negative
                //The signs will not change when the number is positive
                //If the number needs to be reset, then the bounds will also be reset
                eval_x[0].frac *=  factor_x;
                eval_x[1].frac *=  factor_x;
                eval_y[0].frac *=  factor_y;
                eval_y[1].frac *=  factor_y;

                //Interval addition
                cuda::er_add_rd(&result.eval[numberIdx], &eval_x[sign_x], &eval_y[sign_y]);
                cuda::er_add_ru(&result.eval[numberIdx + lenr], &eval_x[1 - sign_x], &eval_y[1 - sign_y]);

                //Calculation of the exponent and preliminary calculation of the sign (the sign will be changed if restoring is required)
                result.sign[numberIdx] = 0;
                result.exp[numberIdx] = (exp_x == 0) ? exp_y : exp_x;

                //Restoring the negative result
                //int plus  = result.eval[numberIdx].frac >= 0 && result.eval[numberIdx + lenr].frac >= 0;
                int minus = result.eval[numberIdx].frac < 0 && result.eval[numberIdx + lenr].frac < 0;
                if(minus){
                    result.sign[numberIdx] = 1;
                    er_float_t tmp = result.eval[numberIdx];
                    result.eval[numberIdx].frac = -result.eval[numberIdx + lenr].frac;
                    result.eval[numberIdx].exp  = result.eval[numberIdx + lenr].exp;
                    result.eval[numberIdx + lenr].frac = -tmp.frac;
                    result.eval[numberIdx + lenr].exp  = tmp.exp;
                }

                //Storing data for Kernel #2 in the buffer
                int4 intBuf;
                intBuf.x = gamma;
                intBuf.y = theta;
                intBuf.z = factor_x;
                intBuf.w = factor_y;
                result.buf[numberIdx] = intBuf;

                //Go to the next iteration
                numberIdx +=  gridDim.x * blockDim.x;
            }
        }
        // code for unequal increments or equal increments not equal to 1
        else{
            int ix = incx > 0 ? numberIdx * incx : (-n + numberIdx + 1)*incx;
            int iy = incy > 0 ? numberIdx * incy : (-n + numberIdx + 1)*incy;
            int ir = incr > 0 ? numberIdx * incr : (-n + numberIdx + 1)*incr;
            while (numberIdx < n) {
                eval_x[0] = x.eval[ix];
                eval_x[1] = x.eval[ix + lenx];
                eval_y[0] = y.eval[iy];
                eval_y[1] = y.eval[iy + leny];

                exp_x = x.exp[ix];
                exp_y = y.exp[iy];
                sign_x = x.sign[ix];
                sign_y = y.sign[iy] ^ 1; //invert the sign of y to perform subtraction instead of addition

                int dexp = exp_x - exp_y;
                int gamma =  dexp * (dexp > 0);
                int theta = -dexp * (dexp < 0);

                int nzx = ((eval_y[1].frac == 0) || (theta + eval_y[1].exp) < cuda::MP_J);
                int nzy = ((eval_x[1].frac == 0) || (gamma + eval_x[1].exp) < cuda::MP_J);

                gamma = gamma * nzy;
                theta = theta * nzx;

                exp_x = (exp_x - gamma) * nzx;
                exp_y = (exp_y - theta) * nzy;

                sign_x *= nzx;
                sign_y *= nzy;

                int factor_x = (1 - 2 * sign_x) * nzx;
                int factor_y = (1 - 2 * sign_y) * nzy;

                eval_x[0].exp += gamma;
                eval_x[1].exp += gamma;
                eval_y[0].exp += theta;
                eval_y[1].exp += theta;

                eval_x[0].frac *=  factor_x;
                eval_x[1].frac *=  factor_x;
                eval_y[0].frac *=  factor_y;
                eval_y[1].frac *=  factor_y;

                cuda::er_add_rd(&result.eval[ir], &eval_x[sign_x], &eval_y[sign_y]);
                cuda::er_add_ru(&result.eval[ir + lenr], &eval_x[1 - sign_x], &eval_y[1 - sign_y]);

                result.sign[ir] = 0;
                result.exp[ir] = (exp_x == 0) ? exp_y : exp_x;

                //int plus  = result.eval[ir].frac >= 0 && result.eval[ir + lenr].frac >= 0;
                int minus = result.eval[ir].frac < 0 && result.eval[ir + lenr].frac < 0;
                if(minus){
                    result.sign[ir] = 1;
                    er_float_t tmp = result.eval[ir];
                    result.eval[ir].frac = -result.eval[ir+lenr].frac;
                    result.eval[ir].exp  = result.eval[ir+lenr].exp;
                    result.eval[ir+lenr].frac = -tmp.frac;
                    result.eval[ir+lenr].exp  = tmp.exp;
                }

                int4 intBuf;
                intBuf.x = gamma;
                intBuf.y = theta;
                intBuf.z = factor_x;
                intBuf.w = factor_y;
                result.buf[ir] = intBuf;

                //Go to the next iteration
                numberIdx +=  gridDim.x * blockDim.x;
                ix += gridDim.x * blockDim.x * incx;
                iy += gridDim.x * blockDim.x * incy;
                ir += gridDim.x * blockDim.x * incr;
            }
        }
    }


    /********************* Rounding kernels *********************/


    /*!
     * Rounding the result vector
     * For each multiple-precision entry, the rounding is performed as a single thread
     * @param result - pointer to the result vector in the GPU memory
     * @param incr - storage spacing between elements of result (must be non-zero)
     * @param n - operation size
     */
    __global__ void mp_array_round(mp_array_t result, const int incr, int n) {
        int numberIdx =  blockDim.x * blockIdx.x + threadIdx.x;
        // Actual vector length (may differ from the operation size, n)
        int lenr = result.len[0];

        // code for increment equal to 1
        if(incr == 1){
            while (numberIdx < n) {
                #if defined(DEBUG) || defined(_DEBUG)
                if( result.eval[lenr + numberIdx].exp != result.eval[numberIdx].exp ){
                    printf("\n [CUDA WARNING] Possible loss of accuracy");
                }
                #endif
                int bits = (result.eval[lenr + numberIdx].exp - cuda::MP_H + 1)*(result.eval[lenr + numberIdx].frac != 0);
                while (bits > 0) {
                    result.exp[numberIdx] += bits;
                    cuda::rns_scale2pow(&result.digits[numberIdx * RNS_MODULI_SIZE], &result.digits[numberIdx * RNS_MODULI_SIZE], bits);
                    cuda::ifc_compute_fast(&result.eval[numberIdx], &result.eval[lenr + numberIdx], &result.digits[numberIdx * RNS_MODULI_SIZE]);
                    bits = -1;
                }
                //Go to the next iteration
                numberIdx +=  gridDim.x * blockDim.x;
            }
        }
        // code for increment not equal to 1
        else{
            int ir = incr > 0 ? numberIdx * incr : (-n + numberIdx + 1)*incr;
            while (numberIdx < n) {
                #if defined(DEBUG) || defined(_DEBUG)
                if( result.eval[lenr + ir].exp != result.eval[ir].exp ){
                    printf("\n [CUDA WARNING] Possible loss of accuracy");
                }
                #endif
                int bits = (result.eval[lenr + ir].exp - cuda::MP_H + 1)*(result.eval[lenr + ir].frac != 0);
                while (bits > 0) {
                    result.exp[ir] += bits;
                    cuda::rns_scale2pow(&result.digits[ir * RNS_MODULI_SIZE], &result.digits[ir * RNS_MODULI_SIZE], bits);
                    cuda::ifc_compute_fast(&result.eval[ir], &result.eval[lenr + ir], &result.digits[ir * RNS_MODULI_SIZE]);
                    bits = -1;
                }
                //Go to the next iteration
                numberIdx +=  gridDim.x * blockDim.x;
                ir += gridDim.x * blockDim.x * incr;
            }
        }
    }

} //end of namespace

#endif //MPRES_MPARRAY_CUH
