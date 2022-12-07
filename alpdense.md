# Introduction
This file is intended to provide instructions for:
- Running smoke tests for the ALP/Dense reference backend (aka alp_reference);
- Running performance tests of the ALP/Dense reference backend with dispatch to BLAS (aka alp_dispatch);
- Running performance tests of the ALP/Dense shared memory backend with dispatch to BLAS (aka alp_omp).

An analogous [script-like](alpdense.sh) version of this page is available in the ALP root directory of this branch. You may decide to run it or follow the instructions in this page step by step. Either way, before running any instructions please make sure to define the following environment variables:

```
# The root folder where this branch is cloned.
export ALP_SOURCE="$(realpath ../)"
# The build folder from which running these steps.
export ALP_BUILD="$(pwd)"
# The KunpengBLAS installation folder.
# For example, the "kml" directory extracted from the "boostkit-kml-1.6.0-1.aarch64.rpm"
export BLAS_LIB="/path/to/kunpengblas/boostkit-kml-1.6.0.aarch64/usr/local/kml"
# The lib folder of the LAPACK library.
export LAPACK_LIB="/path/to/lapack/netlib/build/lib"
# The include folder of the LAPACK library.
# Must include the C/C++ LAPACKE interface.
export LAPACK_INCLUDE="/path/to/lapack/netlib/lapack-3.9.1/LAPACKE/include/"
```

Assuming this branch is cloned in the `ALP_SOURCE` folder, you may run it as follows from the `$ALP_SOURCE/build` directory:

```
bash ../alpdense.sh
```

Otherwise, you may follow step by step the instructions below.

# Source code location

Assuming this branch is cloned in the `ALP_SOURCE` folder, all ALP/Dense include files are located in the `$ALP_SOURCE/include/alp` folder:
- In particular, all the pre-implemented algorithms are located in `$ALP_SOURCE/include/alp/algorithms` 
- The reference, dispatch, and omp backends are located in `$ALP_SOURCE/include/alp/reference`, `$ALP_SOURCE/include/$ALP_SOURCE/alp/dispatch`, and `$ALP_SOURCE/include/alp/omp`, respectively.

All tests discussed below are collected in the `$ALP_SOURCE/tests/smoke` and `$ALP_SOURCE/tests/performance` folders. The folder `$ALP_SOURCE/tests/unit` contains additional unit tests not discuss in this page.

# Dependencies 

For all tests below, the standard ALP dependencies are required:
- LibNUMA: -lnuma
- Standard math library: -lm
- POSIX threads: -lpthread
- OpenMP: -fopenmp in the case of GCC

# Smoke tests

We collect the following smoke tests associated with the ALP/Dense reference backend:
- Basic targets:
  - General matrix-matrix multiplication ([source](tests/smoke/alp_gemm.cpp))
  - Householder tridiagonalization of a real symmetric/complex Hermitian matrix ([source](tests/smoke/alp_zhetrd.cpp))
  - Divide and conquer tridiagonal eigensolver for tridiagonal, real symmetric matrices ([source](tests/smoke/alp_dstedc.cpp))
  - Eigensolver for real symmetric matrices ([source](tests/smoke/alp_syevd.cpp))
  - Householder QR decomposition of a real/complex general matrix ([source](tests/smoke/alp_zgeqrf.cpp))
- Challenge targets:
  - Triangular linear system solve using backsubstitution of upper tridiagonal, real/complex matrix ([source](tests/smoke/alp_backsubstitution.cpp))
  - Triangular linear system solve using forwardsubstitution of lower tridiagonal, real/complex matrix ([source](tests/smoke/alp_forwardsubstitution.cpp))
  - Cholesky decomposition of a symmetric/Hermitian positive definite matrix ([source](tests/smoke/alp_cholesky.cpp))
  - Householder LU decomposition of a real/complex general matrices ([source](tests/smoke/alp_zgetrf.cpp))
  - Inverse of a symmetric/Hermitian positive definite matrix ([source](tests/smoke/alp_potri.cpp))
  - Singular value decomposition of a real/complex general matrix ([source](tests/smoke/alp_zgesvd.cpp))

This tests are collected and run as ALP smoketests.
From `$ALP_SOURCE/build` run:

```
export ALP_SOURCE="$(realpath ../)"
cmake -DCMAKE_INSTALL_PREFIX=./install $ALP_SOURCE || ( echo "test failed" &&  exit 1 )
make smoketests -j$(nproc)
```
The last command runs all registered smoke tests including the ALP/GraphBLAS smoketests.

# Performance tests

This tests have been executed:
- On a Kunpeng 920 node using 1 core for the sequential alp_dispatch tests and 64 cores for the alp_omp tests;
- Compiling with gcc 9.4.0;
- Linking against KunpengBLAS from the Kunpeng BoostKit 22.0.RC1 and the netlib LAPACK linking to the same BLAS library.
- All tests reported times are in milliseconds and printed after the "time (per repeat)" text.

In our evaluation we extracted the _Kunpeng BoostKit 22.0.RC1_ in a `BLAS_LIB` folder (the `kml` directory extracted from the `boostkit-kml-1.6.0-1.aarch64.rpm` package). `BLAS_LIB` should contain the `include/kblas.h` header file and the `lib/kblas` directory. 

## Compilation and execution of the sequential Cholesky decomposition tests

Here we compare our ALP Cholesky implementation, based on the alp_dispatch backend, against the `potrf` LAPACK functionality.

If no LAPACK library can be found by the compiler in system directories, `LAPACK_LIB` and `LAPACK_INCLUDE` have to be properly set (as mentioned at the beginning of this guide) and appropriately provided when calling cmake. In particular, in this test we assume the availability of the C/C++ LAPACKE interface.
If you are using locally installed KunpengBLAS, make sure to set the `BLAS_LIB` path to the `kml` directory extracted from the `boostkit-kml-1.6.0-1.aarch64.rpm` package.
In the example below, we assume no system libraries are available. From the `$ALP_SOURCE/build` folder run the following commands:

```
cmake -DKBLAS_ROOT="$BLAS_LIB" -DWITH_ALP_DISPATCH_BACKEND=ON -DCMAKE_INSTALL_PREFIX=./install $ALP_SOURCE || ( echo "test failed" &&  exit 1 )
make install  -j$(nproc) || ( echo "test failed" &&  exit 1 )
```

### LAPACK-based test

To compile and run the LAPACK-based Cholesky test (not ALP code) run the following commands:
```
install/bin/grbcxx  -b alp_dispatch -o cholesky_lapack_reference.exe $ALP_SOURCE/tests/performance/lapack_cholesky.cpp $LAPACK_LIB/liblapack.a -I$LAPACK_INCLUDE -lgfortran || ( echo "test failed" &&  exit 1 )
./cholesky_lapack_reference.exe -n 1024 -repeat 10 || ( echo "test failed" &&  exit 1 )
```
In our tests, we executed `./cholesky_lapack_reference.exe` with matrix sizes (`-n` flag) in the range [400:100:3000].

### ALP-based test

Some facts about this test:
- The algorithm is a blocked variant of Cholesky with block size BS = 64 (as done in LAPACK).
- It recursively requires an unblocked version of the same algorithm (of size BSxBS) which does not dispatch to LAPACK.
- All BLAS functions needed by the algorithm are dispatched to the external BLAS library.

```
make test_alp_cholesky_perf_alp_dispatch -j$(nproc) || ( echo "test failed" &&  exit 1 )
tests/performance/alp_cholesky_perf_alp_dispatch -n 1024 -repeat 10 || ( echo "test failed" &&  exit 1 )
```
As for the LAPACK-based test, we executed `tests/performance/alp_cholesky_perf_alp_dispatch` with matrix sizes (`-n` flag) in the range [400:100:3000].

**Note:** A consistent test should use the same BLAS in LAPACK-based as well as in the ALP-based tests.

## Compilation and execution of shared memory parallel `mxm` tests

Here we compare our ALP shared memory backend (alp_omp) `mxm` implementation against the BLAS's `gemm` functionality.
`mxm` is an inplace, ALP primitive that computes C = C + A*B, with matrices of conforming sizes.

Our current shared memory backend implementation is currently only supporting square thread grids (although the methodology is not limited to that in general). For this reason, in the tests below we run both LAPACK and ALP using 64 threads. To ensure a fair comparison, we link with the `omp` version of KunpengBLAS.

You can compile with the `omp` version of KunpengBLAS by additionally providing the `-DKBLAS_IMPL=omp` flag when calling cmake. However, this should be compiled in a different directory from the other BLAS-based builds, as follows:
```
CWD=$(pwd)
ompbuild="build_with_omp_blas"
rm -rf $ompbuild && mkdir $ompbuild && cd $ompbuild
cmake -DKBLAS_ROOT="$BLAS_LIB" -DKBLAS_IMPL=omp -DWITH_ALP_DISPATCH_BACKEND=ON -DCMAKE_INSTALL_PREFIX=./install $ALP_SOURCE || ( echo "test failed" &&  exit 1 )
make install -j$(nproc) || ( echo "test failed" &&  exit 1 )
```

### `gemm`-based BLAS test.

from `$ompbuild` run:
```
install/bin/grbcxx -b alp_dispatch -o blas_mxm.exe $ALP_SOURCE/tests/performance/blas_mxm.cpp -lgfortran || ( echo "test failed" &&  exit 1 )
OMP_NUM_THREADS=64 ./blas_mxm.exe -n 1024 -repeat 10 || ( echo "test failed" &&  exit 1 )
cd $CWD
```
In our tests, we executed `./blas_mxm.exe` with matrix sizes (`-n` flag) in the range [1024:1024:10240].

### ALP-based test.

Some facts about this test:
- The ALP `mxm` implementation is based on a [2.5D matrix multiplication algorithm](https://netlib.org/lapack/lawnspdf/lawn248.pdf);
- In this test we execute with a 3D thread grid of size 4x4x4;
- We set `OMP_NUM_THREADS=64` threads and fix `GOMP_CPU_AFFINITY="0-15 24-39 48-63 72-87"` to reflect the cores and NUMA topology of the node;
- The algorithm is allocating memory using a 2D block-cyclic layout with blocks of size 128x128.

From `$ALP_SOURCE/build` run:

```
make test_alp_mxm_perf_alp_omp -j$(nproc) || ( echo "test failed" &&  exit 1 )
GOMP_CPU_AFFINITY="0-15 24-39 48-63 72-87" OMP_NUM_THREADS=64 tests/performance/alp_mxm_perf_alp_omp -n 1024 -repeat 10 || ( echo "test failed" &&  exit 1 )
```
As for the gemm-based test, we executed `tests/performance/alp_mxm_perf_alp_omp` with matrix sizes (`-n` flag) in the range [1024:1024:10240].
