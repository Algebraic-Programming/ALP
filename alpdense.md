# Introduction
This file is intended to provide instructions for:
- Running smoke, unoptimized performance tests for the ALP/Dense sequential reference backend (aka alp_reference);
- Running optimized performance tests of the ALP/Dense sequential reference backend with dispatch to BLAS (aka alp_dispatch);
- Running optimized performance tests of the ALP/Dense shared memory backend with dispatch to BLAS (aka alp_omp).

# Performance Tests

This tests have been executed:
- On a Kunpeng 920 node using 1 core for the sequential reference and alp_dispatch tests and 64 cores for the alp_omp tests;
- Compiling with gcc 9.4.0;
- Linking against KunpengBLAS from the Kunpeng BoostKit 22.0.RC1 and the netlib LAPACK linking to the same BLAS library.
- All tests report runtime in milliseconds after the _time (ms, ...)_ text lines printed on screen.

In our evaluation we extracted the _Kunpeng BoostKit 22.0.RC1_ in a `BLAS_ROOT` folder (the `usr/local/kml` directory extracted from the `boostkit-kml-1.6.0-1.aarch64.rpm` package). `BLAS_ROOT` should contain the `include/kblas.h` header file and the `lib/kblas/{locking, nolocking, omp, pthread}/libkblas.so` library. 

If no system LAPACK library can be found by the compiler, `LAPACK_LIB` (containing the `liblapack.{a,so}` library) and `LAPACK_INCLUDE` (containing the `lapacke.h` header file) have to be appropriately set and provided to cmake, for example exporting them as follows:

```
# The root folder where this branch is cloned.
export ALP_SOURCE="$(realpath ../)"
# The build folder from which running these steps.
export ALP_BUILD="$(pwd)"
# The KML installation folder.
# For example, the "usr/local/kml" directory extracted from the "boostkit-kml-1.6.0-1.aarch64.rpm"
#export BLAS_ROOT="/path/to/kunpengblas/boostkit-kml-1.6.0.aarch64/usr/local/kml"
# The lib folder of the LAPACK library.
#export LAPACK_LIB="/path/to/lapack/netlib/build/lib"
# The include folder of the LAPACK library.
# Must include the C/C++ LAPACKE interface.
#export LAPACK_INCLUDE="/path/to/lapack/netlib/lapack-3.9.1/LAPACKE/include/"

if [ -z ${BLAS_ROOT+x} ] || [ -z ${LAPACK_LIB+x} ] || [ -z ${LAPACK_INCLUDE+x} ]; then
    echo "Please define BLAS_ROOT, LAPACK_LIB, and LAPACK_INCLUDE variables."
fi
```

In particular, we assume the availability of the C/C++ LAPACKE interface and, for all tests below, we assume no system libraries are available. 

***Assuming this branch is cloned in the `ALP_SOURCE` folder, all instructions provided below should be run from a `$ALP_SOURCE/build` folder.***

An analogous [script-like](alpdense.sh) version of this page is available in the ALP root directory of this branch. You may decide to run it directly (**note:** always making sure to customize the export commands above to your environment first) as follows:

```
bash ../alpdense.sh
```

The scripts also logs the output of each test group below into a separate file in `$ALP_BUILD/logs`, i.e.,
- Smoke tests:
  - `alp_smoketests.log` (ALP smoketests - reference backend, unoptimized)
  - `lapack_smoketests.log` (LAPACK smoketests - sequential KBLAS)
- Performance tests:
  - `lapack_doptrf_seq.log` (LAPACK `dpotrf` - sequential KBLAS)
  - `alp_dpotrf_seq.log` (ALP `dpotrf` - dispatch backend, sequential KBLAS)
  - `lapack_dpotrf_omp.log` (LAPACK `dpotrf` - shared-memory KBLAS)
  - `alp_dpotrf_omp.log` (ALP `dpotrf` - dispatch backend, shared-memory KBLAS)
  - `kblas_mxm_omp.log` (KunpengBLAS `mxm` - shared memory)
  - `alp_smoketests.log` (ALP `mxm` - omp+dispatch backends, shared-memory KBLAS)

The rest of this page describes each step of the script above.

# Source Code Location

Assuming this branch is cloned in the `ALP_SOURCE` folder, all ALP/Dense include files are located in the `$ALP_SOURCE/include/alp` folder:
- In particular, all the pre-implemented algorithms are located in `$ALP_SOURCE/include/alp/algorithms` 
- The reference, dispatch, and omp backends are located in `$ALP_SOURCE/include/alp/reference`, `$ALP_SOURCE/include/$ALP_SOURCE/alp/dispatch`, and `$ALP_SOURCE/include/alp/omp`, respectively.

All tests discussed below are collected in the `$ALP_SOURCE/tests/smoke` and `$ALP_SOURCE/tests/performance` folders. The folder `$ALP_SOURCE/tests/unit` contains additional unit tests not discuss in this page.

# Dependencies 

For all tests below, the standard ALP dependencies are required:
- gfortran: -lgfortran
- LibNUMA: -lnuma
- Standard math library: -lm
- POSIX threads: -lpthread
- OpenMP: -fopenmp in the case of GCC

# Sequential Smoke Tests (Functional, Unoptimized)

We collect the following smoke tests associated with the ALP/Dense reference backend:
- Basic targets:
  - General matrix-matrix multiplication ([source](tests/smoke/alp_gemm.cpp))
  - Householder tridiagonalization of a real symmetric/complex Hermitian matrix ([source](tests/smoke/alp_zhetrd.cpp))
  - Divide and conquer tridiagonal eigensolver for tridiagonal, real symmetric matrices ([source](tests/smoke/alp_stedc.cpp))
  - Eigensolver for real symmetric matrices ([source](tests/smoke/alp_zheevd.cpp))
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
cmake -DWITH_ALP_REFERENCE_BACKEND=ON -DCMAKE_INSTALL_PREFIX=./install $ALP_SOURCE || ( echo "test failed" &&  exit 1 )
make install -j$(nproc) || ( echo "test failed" &&  exit 1 )
SMOKE_PRINT_TIME=ON make smoketests_alp -j$(nproc)
```

**Note:** The variable `SMOKE_PRINT_TIME=ON` is used to print timing information of each test to screen. Set it to `OFF` or remove it from the command if this action is not desired.

If the tests run correctly, for each of them you should see an output similar to the following:

```
****************************************************************************************
      FUNCTIONAL    PERFORMANCE                       DESCRIPTION      
----------------------------------------------------------------------------------------

>>>      [x]           [ ]       Tests Cholesky decomposition for a random
                                 symmetric positive definite matrix (100x100).
Timing of blocked inplace version with bs = 64.
 time (ms, total) = 72.1747
 time (ms, per repeat) = 3.60873
Test OK

```

To compare with LAPACK+KunpengBLAS (not ALP code) you may run the following:

```
KBLAS_LIB=$BLAS_ROOT/lib/kblas/locking
USECASES=("dstedc" "dsyevd" "dsytrd" "zhetrd" "dgeqrf" "dgesvd" "dgetrf" "dpotri")

for USECASE in "${USECASES[@]}"
do
    install/bin/grbcxx -o ${USECASE}_lapack_reference.exe $ALP_SOURCE/tests/performance/lapack_${USECASE}.cpp $LAPACK_LIB/liblapack.a $KBLAS_LIB/libkblas.so -Wl,-rpath $KBLAS_LIB -I$LAPACK_INCLUDE -lgfortran || ( echo "Compiling ${USECASE} failed" &&  exit 1 )
done

for USECASE in "${USECASES[@]}"
do
    ./${USECASE}_lapack_reference.exe -n 100 -repeat 20 || ( echo "test ${USECASE} failed" &&  exit 1 )
done
```

# Sequential Cholesky Decomposition Tests (Dispatch, Optimized)

Here we compare our ALP Cholesky implementation, based on the alp_dispatch backend, against the `potrf` LAPACK functionality.

From the `$ALP_SOURCE/build` folder run the following commands:

```
cmake -DKBLAS_ROOT="$BLAS_ROOT" -DWITH_ALP_DISPATCH_BACKEND=ON -DCMAKE_INSTALL_PREFIX=./install $ALP_SOURCE || ( echo "test failed" &&  exit 1 )
make install  -j$(nproc) || ( echo "test failed" &&  exit 1 )
```

## LAPACK-Based Test (Sequential BLAS)

To compile and run the LAPACK-based Cholesky test (not ALP code) run the following commands:
```
install/bin/grbcxx  -b alp_dispatch -o cholesky_lapack_reference.exe $ALP_SOURCE/tests/performance/lapack_cholesky.cpp $LAPACK_LIB/liblapack.a -I$LAPACK_INCLUDE -lgfortran || ( echo "test failed" &&  exit 1 )
for MSIZE in {400..4000..100}
do 
    ./cholesky_lapack_reference.exe -n ${MSIZE} -repeat 10 || ( echo "test failed" &&  exit 1 )
done
```

If the commands run correctly the output on screen should look like the following:

```
Testing dpotrf_ for U^T * U = S, with S SPD of size ( 1024 x 1024 )
Test repeated 10 times.
 time (ms, total) = 433.652
 time (ms, per repeat) = 43.3652
Tests OK
```

In our tests, we executed `./cholesky_lapack_reference.exe` with matrix sizes (`-n` flag) in the range [400, 4000] in steps of 100.

## ALP-Based Test (Dispatch Sequential Building Blocks to Optimized BLAS)

Some facts about this test:
- The algorithm is a blocked variant of Cholesky with block size BS = 64 (as done in LAPACK).
- It recursively requires an unblocked version of the same algorithm (of size BSxBS) which does not dispatch to LAPACK.
- All BLAS functions needed by the algorithm are dispatched to the external BLAS library. In particular, as POC of what ALP could offer in terms of performance if its primitives could be efficiently generated/optimized (e.g., via our envisioned MLIR-based backend for delayed compilation), it dispatches the triangular solve and the fused `foldl`+`mxm` operations.

```
make test_alp_cholesky_perf_alp_dispatch -j$(nproc) || ( echo "test failed" &&  exit 1 )
for MSIZE in {400..4000..100}
do 
    tests/performance/alp_cholesky_perf_alp_dispatch -n ${MSIZE} -repeat 10 || ( echo "test failed" &&  exit 1 )
done
```

If the commands run correctly the output on screen should look like the following:

```
Testing Cholesky decomposition U^T * U = S, with S SPD of size ( 1024 x 1024 )
Test repeated 10 times.
 time (ms, total) = 463.652
 time (ms, per repeat) = 46.3652
Tests OK
```

As for the LAPACK-based test, we executed `tests/performance/alp_cholesky_perf_alp_dispatch` with matrix sizes (`-n` flag) in the range [400, 4000] in steps of 100.

**Note:** A consistent test should use the same BLAS in LAPACK-based as well as in the ALP-based tests.

## Cholesky Decomposition with Shmem BLAS

An analogous experiment can be conducted using the shared-memory BLAS library in place of the sequential one as follows (the following block runs both the LAPACK and the ALP tests):

```
subbuild="build_potrf_with_omp_blas"
rm -rf $subbuild && mkdir $subbuild && cd $subbuild
cmake -DKBLAS_ROOT="$BLAS_ROOT" -DKBLAS_IMPL=omp -DWITH_ALP_OMP_BACKEND=ON -DWITH_ALP_DISPATCH_BACKEND=ON -DCMAKE_INSTALL_PREFIX=./install $ALP_SOURCE || ( echo "test failed" &&  exit 1 )
make install -j$(nproc) || ( echo "test failed" &&  exit 1 )

install/bin/grbcxx  -b alp_dispatch -o cholesky_lapack_omp.exe $ALP_SOURCE/tests/performance/lapack_cholesky.cpp $LAPACK_LIB/liblapack.a -I$LAPACK_INCLUDE -lgfortran || ( echo "test failed" &&  exit 1 )
for NT in 1 64 96
do
    echo "#####################################################################"
    echo " Testing potrf: LAPACK + KunpengBLAS (omp) with OMP_NUM_THREADS=${NT}"
    echo "#####################################################################"
    for MSIZE in {400..4000..100}
    do 
        OMP_NUM_THREADS=${NT} ./cholesky_lapack_omp.exe -n ${MSIZE} -repeat 10 || ( echo "test failed" &&  exit 1 )
    done
    echo " Tests completed."
    echo "#####################################################################"
done

make test_alp_cholesky_perf_alp_dispatch -j$(nproc) || ( echo "test failed" &&  exit 1 )
for NT in 1 64 96
do
    echo "##########################################################################"
    echo "Testing potrf: Testing ALP + KunpengBLAS (omp) with OMP_NUM_THREADS=${NT}"
    echo "##########################################################################"
    for MSIZE in {400..4000..100}
    do 
        OMP_NUM_THREADS=${NT} tests/performance/alp_cholesky_perf_alp_dispatch -n ${MSIZE} -repeat 10 || ( echo "test failed" &&  exit 1 )
    done
    echo " Tests completed."
    echo "##########################################################################"
done
cd $ALP_BUILD

```

# Shared-Memory Parallel `mxm` Tests (Optimized)

Here we compare our ALP shared memory backend (alp_omp) `mxm` implementation against the BLAS's `gemm` functionality.
`mxm` is an inplace, ALP primitive that computes C = C + A*B, with matrices of conforming sizes.

Our current shared memory backend implementation is currently only supporting square thread grids (although the methodology is not limited to that in general). For this reason, in the tests below we run both LAPACK and ALP using 64 threads. To ensure a fair comparison, we link with the `omp` version of KunpengBLAS.

You can compile with the `omp` version of KunpengBLAS by additionally providing the `-DKBLAS_IMPL=omp` flag when calling cmake. However, this should be compiled in a different directory from the other BLAS-based builds, as follows:
```
subbuild="build_mxm_with_omp_blas"
rm -rf $subbuild && mkdir $subbuild && cd $subbuild
cmake -DKBLAS_ROOT="$BLAS_ROOT" -DKBLAS_IMPL=omp -DWITH_ALP_OMP_BACKEND=ON -DWITH_ALP_DISPATCH_BACKEND=ON -DCMAKE_INSTALL_PREFIX=./install $ALP_SOURCE || ( echo "test failed" &&  exit 1 )
make install -j$(nproc) || ( echo "test failed" &&  exit 1 )
```

## `gemm`-Based BLAS Test.

from `$subbuild` run:
```
install/bin/grbcxx -b alp_dispatch -o blas_mxm.exe $ALP_SOURCE/tests/performance/blas_mxm.cpp -lgfortran || ( echo "test failed" &&  exit 1 )
for MSIZE in {1024..10240..1024}
do 
    OMP_NUM_THREADS=64 ./blas_mxm.exe -n ${MSIZE} -repeat 10 || ( echo "test failed" &&  exit 1 )
done
cd $ALP_BUILD
```

If the commands run correctly the output on screen should look like the following:

```
Testing cblas_dgemm for C(1024 x 1024) +=   A(1024 x 1024) x B(1024 x 1024)  10 times.
 time (ms, total) = 116.494
 time (ms, per repeat) = 11.6494
Tests OK
```

In our tests, we executed `./blas_mxm.exe` with matrix sizes (`-n` flag) in the range [1024:1024:10240].

## ALP-Based Test (Dispatch Sequential Building Blocks to Optimized BLAS).

Some facts about this test:
- The ALP `mxm` shared memory implementation is based on a [2.5D matrix multiplication algorithm](https://netlib.org/lapack/lawnspdf/lawn248.pdf);
- In this test we execute with a 3D thread grid of size 4x4x4;
- We set `OMP_NUM_THREADS=64` threads and fix `GOMP_CPU_AFFINITY="0-15 24-39 48-63 72-87"` to reflect the cores and NUMA topology of the node;
- The algorithm is allocating memory using a 2D block-cyclic layout with blocks of size 128x128.
- Each sequential block-level `mxm` (128x128x128) is dispatched to the selected BLAS library.

From `$ALP_SOURCE/build` run:

```
subbuild="build_mxm_with_alp_omp"
rm -rf $subbuild && mkdir $subbuild && cd $subbuild
cmake -DKBLAS_ROOT="$BLAS_ROOT" -DWITH_ALP_DISPATCH_BACKEND=ON -DWITH_ALP_OMP_BACKEND=ON -DCMAKE_INSTALL_PREFIX=./install $ALP_SOURCE || ( echo "test failed" &&  exit 1 )
make test_alp_mxm_perf_alp_omp -j$(nproc) || ( echo "test failed" &&  exit 1 )
for MSIZE in {1024..10240..1024}
do 
    GOMP_CPU_AFFINITY="0-15 24-39 48-63 72-87" OMP_NUM_THREADS=64 tests/performance/alp_mxm_perf_alp_omp -n ${MSIZE} -repeat 10 || ( echo "test failed" &&  exit 1 )
done
cd $ALP_BUILD
```

If the commands run correctly the output on screen should look like the following:

```
Testing  C(1024 x 1024) += A(1024 x 1024) x B(1024 x 1024) 10 times.
 time (ms, total) = 69.7239
 time (ms, per repeat) = 6.97239
Tests OK
```

As for the gemm-based test, we executed `tests/performance/alp_mxm_perf_alp_omp` with matrix sizes (`-n` flag) in the range [1024:1024:10240].
