# Multiple-Precision GPU Accelerated BLAS Routines based on Residue Number System (MPRES-BLAS)
###### Version 0.1


The MPRES-BLAS library provides a number of parallel multiple-precision BLAS 
(Basic Linear Algebra Subprograms) routines for CUDA-enabled GPUs.
The library uses a residue number system (RNS) to represent multiple-precision
floating-point numbers. Non-positional nature and parallel arithmetic properties make RNS
a good tool for high-performance computing. The library also supports floating-point arithmetic
in extended range using CUDA and provides several useful RNS computation techniques.

Currently the following level-1 multiple-precision functions are only supported:

* ASUM --- Sum of absolute values (`mp_array_asum`)
* DOT --- Dot product of two vectors (`mp_array_dot`)
* SCAL --- Vector-scalar product (`mp_array_scal`)
* AXPY --- Constant times a vector plus a vector (`mp_array_axpy`)
* WAXPBY --- Scaled vector addition (`mp_array_waxpby`)

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
* `RNS_EVAL_MIN_LIMIT`
* `RNS_EVAL_OFFSET_VEC_SIZE`

To compute `RNS_EVAL_MIN_LIMIT` and `RNS_EVAL_OFFSET_VEC_SIZE` use `rns_eval_const_calc()` from `src/rns.cuh`.
These two constants will be removed in the near future, as soon as an improved algorithm for computing
the interval evaluation of an RNS number is implemented.

If the size of the moduli set increases, make sure that the loop-unrolled methods from `src/modular.cuh` remain correct.

When using large moduli (like `1283742825`) to increase the precision of computations, make sure your environment has support for 8-byte `long` type.



*Link: http://github.com/kisupov/mpres-blas*

*Copyright 2018, 2019 by Konstantin Isupov and Alexander Kuvaev.*
