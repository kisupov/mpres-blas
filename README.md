# Multiple-Precision GPU Accelerated BLAS Routines based on Residue Number System (MPRES-BLAS)
###### Version 0.1


The MPRES-BLAS library provides a number of parallel multiple-precision BLAS 
(Basic Linear Algebra Subprograms) routines for CUDA-enabled GPUs.
The library uses a residue number system (RNS) to represent multiple-precision
floating-point numbers. Non-positional nature and parallel arithmetic properties make RNS
a good tool for high-performance computing. The library also supports floating-point arithmetic
in extended range using CUDA and provides several useful RNS computation algorithms.

Currently the following level-1 multiple-precision functions are only supported:

* ASUM --- Sum of absolute values (`mpasum`)
* DOT --- Dot product of two vectors (`mpdot`)
* SCAL --- Vector-scalar product (`mpscal`)
* AXPY --- Constant times a vector plus a vector (`mpaxpy`)
* WAXPBY --- Scaled vector addition (`mpwaxpby`)
* ROT --- Apply a plane rotation to vectors (`mprot`)
* AXPY_DOT --- Combined AXPY and DOT (`mpaxpydot`)

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



*Link: http://github.com/kisupov/mpres-blas*

*Copyright 2018, 2019 by Konstantin Isupov and Alexander Kuvaev.*
