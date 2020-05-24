# Multiple-Precision GPU Accelerated BLAS Routines based on Residue Number System (MPRES-BLAS)
###### Version 1.0


The MPRES-BLAS library implements a number of linear algebra operations, like the BLAS (Basic Linear Algebra Subprograms) routines, with [multiple precision](https://en.wikipedia.org/wiki/Arbitrary-precision_arithmetic) on CUDA-enabled graphics processing units. The library uses a residue number system ([RNS](https://en.wikipedia.org/wiki/Residue_number_system)) to represent multiple-precision floating-point numbers. Non-positional nature and parallel arithmetic properties make RNS a good tool for high-performance computing on many-core architectures such as GPUs.

The current version of MPRES-BLAS supports the following operations with multiple precision:

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

Furthermore, the MPRES-BLAS library provides basic arithmetic operations with multiple precision for CPU and GPU through the `mp_float_t` data type (see `src/mpfloat.cuh`), so it can also be considered as a general purpose multiple-precision arithmetic library. In addition, the library implements a number of optimized RNS algorithms, such as magnitude comparison and power-of-two scaling (see `src/rns.cuh`), and also supports extended-range floating-point arithmetic with working precision (see `src/extrange.cuh`), which prevents underflow and overflow in a computation involving extremely large or small quantities.

For samples of usage, see `tests/` directory. Some benchmarks require third-party libraries.
Please check `tests/3rdparty/` and `tests/blas/performance/` subdirectories for details.

### Details and notes

The precision of computations (in bits) is specified by the RNS moduli set in `src/params.h`.
The subdirectory `src/params/` contains some predefined moduli sets that provide different
levels of precision. Using these moduli sets is preferred. Just replace the content of
`src/params.h` with the content of the file you want to use.

For now, when using a custom or changing an existing moduli set or its size,
you need to change the following #defines in `src/params.h`:

* `RNS_MODULI_PRODUCT_LOG2`
* `RNS_PARALLEL_REDUCTION_IDX`

When using large moduli (like `1283742825`) to increase the precision of computations, make sure your environment has support for 8-byte `long` type.

### References

1. K. Isupov, V. Knyazkov, and A. Kuvaev, "Design and implementation of multiple-precision BLAS Level 1 functions for graphics processing units," Journal of Parallel and Distributed Computing, vol. 140, pp. 25-36, 2020, doi 
10.1016/j.jpdc.2020.02.006.
2. K. Isupov, "Using floating-point intervals for non-modular computations in residue number system," IEEE Access, vol. 8, pp. 58603-58619, 2020, doi 10.1109/ACCESS.2020.2982365.

*Link: http://github.com/kisupov/mpres-blas*

*Copyright 2018, 2019, 2020 by Konstantin Isupov.*
