# MPRES-BLAS: Multiple-Precision GPU Accelerated BLAS Routines based on Residue Number System
###### Version 1.4.2, released 2021-04-15


The MPRES-BLAS library implements a number of linear algebra operations, like the BLAS (Basic Linear Algebra Subprograms) routines, with [multiple precision](https://en.wikipedia.org/wiki/Arbitrary-precision_arithmetic) on CUDA-enabled graphics processing units. The library uses a residue number system ([RNS](https://en.wikipedia.org/wiki/Residue_number_system)) to represent multiple-precision floating-point numbers. Non-positional nature and parallel arithmetic properties make RNS a good tool for high-performance computing on many-core architectures such as GPUs.

Underlying algorithms for multiple-precision floating-point arithmetic as well as algorithms for vectors of multiple-precision numbers used in MPRES-BLAS are discussed in [this paper](http://doi.org/10.1016/j.jpdc.2020.02.006). For further reading see References below.

1. The current version of MPRES-BLAS supports the following dense operations with multiple precision:

    * ASUM --- Sum of absolute values (`mpasum`)
    * DOT --- Dot product of two vectors (`mpdot`)
    * SCAL --- Vector-scalar product (`mpscal`)
    * AXPY --- Constant times a vector plus a vector (`mpaxpy`)
    * AXPY_DOT --- Combined AXPY and DOT (`mpaxpydot`)
    * WAXPBY --- Scaled vector addition (`mpwaxpby`)
    * NORM --- Vector norms (`mpnorm`)
    * ROT --- Apply a plane rotation to vectors (`mprot`)
    * GEMV --- Matrix-vector multiplication (`mpgemv`)
    * GEMM --- General matrix multiplication (`mpgemm`)
    * GER --- Rank-1 update of a general matrix (`mpger`)
    * GE_ADD --- Matrix add and scale (`mpgeadd`)
    * GE_ACC --- Matrix accumulation and scale (`mpgeacc`)
    * GE_DIAG_SCALE --- Diagonal scaling (`mpgediagscale`)
    * GE_LRSCALE --- Two-sided diagonal scaling (`mpgelrscale`)
    * GE_NORM --- Matrix norms (`mpgenorm`)

2. The following sparse matrix-vector multiplication (SpMV) routines are supported:
    
    * Multiple-precision CSR kernel for double-precision matrix using one thread per matrix row (`mpspmv_csr`)
    * Multiple-precision CSR kernel for double-precision matrix using a group (up to 32) of threads per matrix row (`mpspmv_csr_vector`)
    * Multiple-precision JAD kernel for double-precision matrix using one thread per matrix row (`mpspmv_jad`)
    * Multiple-precision JAD kernel for double-precision matrix using a group (up to 32) of threads per matrix row (`mpspmv_jad_vector`)
    * Multiple-precision ELLPACK kernel for double-precision matrix using one thread per matrix row (`mpspmv_ell`)
    * Multiple-precision DIA kernel for double-precision matrix using one thread per matrix row (`mpspmv_dia`)

The corresponding benchmarks are also provided.

3. In addition, the following SpMV routines using a multiple-precision matrix are implemented (can consume a lot of memory, not for iterative solvers):

   * Multiple-precision CSR kernel for multiple-precision matrix using one thread per matrix row (`mpspmv_mpmtx_csr_scalar`)
   * Multiple-precision CSR kernel for multiple-precision matrix using a group (up to 32) of threads per matrix row (`mpspmv_mpmtx_csr_vector`)
   * Multiple-precision two-step CSR implementation (`mpspmv_mpmtx_csr_2stage`)
   * Multiple-precision JAD kernel for multiple-precision matrix using one thread per matrix row (`mpspmv_mpmtx_jad`)
   * Multiple-precision ELLPACK kernel for multiple-precision matrix using one thread per matrix row (`mpspmv_mpmtx_ell`)
   * Multiple-precision two-step ELLPACK implementation (`mpspmv_mpmtx_ell_2stage`)
   * Multiple-precision DIA kernel for multiple-precision matrix using one thread per matrix row (`mpspmv_mpmtx_dia`)

4. Furthermore, MPRES-BLAS provides basic arithmetic operations with multiple precision for CPU and GPU through the `mp_float_t` data type (see `src/arith/`), so it can also be considered as a general purpose multiple-precision arithmetic library. 

5. In addition, the library implements a number of optimized RNS algorithms, such as magnitude comparison and power-of-two scaling (see `src/rns.cuh`), and also supports extended-range floating-point arithmetic with working precision (see `src/extrange.cuh`), which prevents underflow and overflow in a computation involving extremely large or small quantities.

For samples of usage, see `tests/` directory. Some benchmarks require third-party libraries.
Please check `tests/3rdparty/` and `tests/blas/performance/` subdirectories for details.

### Details and notes

1. MPRES-BLAS is intended for Linux and the GCC compiler. Some manipulations have to be done to run it in Windows.

2. The precision of computations (in bits) is specified by the RNS moduli set in `src/params.h`.
The subdirectory `src/params/` contains some predefined moduli sets that provide different
levels of precision. Using these moduli sets is preferred. Just replace the content of
`src/params.h` with the content of the file you want to use.

3. When using large moduli (like `1283742825`) to increase the precision, make sure your system uses LP64 programming model ('long, pointers are 64-bit').  Fortunately, all modern 64-bit Unix systems use LP64.

### References

1. K. Isupov, V. Knyazkov, and A. Kuvaev, "Design and implementation of multiple-precision BLAS Level 1 functions for graphics processing units," Journal of Parallel and Distributed Computing, vol. 140, pp. 25-36, 2020, https://doi.org/10.1016/j.jpdc.2020.02.006.

2. K. Isupov and V. Knyazkov, "Multiple-precision BLAS library for graphics processing units," in Communications in Computer and Information Science, vol. 1331, pp. 37-49, 2020, https://doi.org/10.1007/978-3-030-64616-5_4. 

3. K. Isupov, "Performance data of multiple-precision scalar and vector BLAS operations on CPU and GPU," Data in Brief, vol. 30, p. 105506, 2020, https://doi.org/10.1016/j.dib.2020.105506.

4. K. Isupov and V. Knyazkov, "Multiple-precision matrix-vector multiplication on graphics processing units," in Program Systems: Theory and Applications, vol. 11, no. 3(46), pp. 62-84, 2020, https://doi.org/10.25209/2079-3316-2020-11-3-61-84. 

5. K. Isupov, "Using floating-point intervals for non-modular computations in residue number system," IEEE Access, vol. 8, pp. 58603-58619, 2020, https://doi.org/10.1109/ACCESS.2020.2982365.


*Link: http://github.com/kisupov/mpres-blas*

*Copyright 2018, 2019, 2020, 2021 by Konstantin Isupov and the MPRES-BLAS team*
