/*
 *  Launch parameters (execution configuration) for multiple-precision CUDA kernels
 *  For details, see:
 *  https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities
 *  https://docs.nvidia.com/gameworks/content/developertools/desktop/analysis/report/cudaexperiments/kernellevel/achievedoccupancy.htm
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

#ifndef MPRES_KERNEL_CONFIG_CUH
#define MPRES_KERNEL_CONFIG_CUH

#include "params.h"

/*
 * Maximum number of resident threads per multiprocessor for current Compute capability, see https://en.wikipedia.org/wiki/CUDA
 */
#define MAX_THREADS_PER_SM (1024)

/*
 * Maximum number of resident blocks per multiprocessor for current Compute capability, see https://en.wikipedia.org/wiki/CUDA
 */
#define MAX_BLOCKS_PER_SM (16)

/*
 * Minimum block size that is enough to reach 100% occupancy of a multiprocessor
 */
#define MIN_BLOCK_SIZE (MAX_THREADS_PER_SM / MAX_BLOCKS_PER_SM)

/*
 * The block size for multiplying and adding digits of multiple-precision numbers in componentwise vector operations.
 * This parameter allows for achieving full occupancy of a streaming multiprocessor under the following restriction:
 * in each thread block, only an INTEGER number of multiple-precision values is computed,
 * i.e. processing the digits of the same multiple-precision value is not split between different thread blocks.
 */
#define BLOCK_SIZE_FOR_RESIDUES (RNS_MODULI_SIZE >= MIN_BLOCK_SIZE ? RNS_MODULI_SIZE : (MIN_BLOCK_SIZE / RNS_MODULI_SIZE) * RNS_MODULI_SIZE)

#endif //MPRES_KERNEL_CONFIG_CUH
