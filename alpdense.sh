# This file is intended to provide instructions for:
#    Running smoke tests for the ALP/Dense reference backend (aka alp_reference);
#    Running performance tests of the ALP/Dense reference backend with dispatch to BLAS (aka alp_dispatch);
#    Running performance tests of the ALP/Dense shared memory backend with dispatch to BLAS (aka alp_omp).

# For all tests below standard ALP dependencies are required:
#    LibNUMA: -lnuma
#    Standard math library: -lm
#    POSIX threads: -lpthread
#    OpenMP: -fopenmp in the case of GCC

# Before running please export: 

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
    exit 1
fi

####################
####################
# Smoke tests
####################
####################

# We collect the following smoke tests associated with the ALP/Dense reference backend:
#    (Basic targets)
#    General matrix-matrix multiplication (source: tests/smoke/alp_gemm.cpp)
#    Householder tridiagonalization of a real symmetric/complex Hermitian matrix (source: tests/smoke/alp_zhetrd.cpp)
#    Divide and conquer tridiagonal eigensolver for tridiagonal, real symmetric matrices (source: tests/smoke/alp_dstedc.cpp)
#    Eigensolver for real symmetric matrices (source: tests/smoke/alp_syevd.cpp)
#    Householder QR decomposition of a real/complex general matrix (source: tests/smoke/alp_zgeqrf.cpp)

#    (Challenge targets)
#    Triangular linear system solve using backsubstitution of upper tridiagonal, real/complex matrix (source: tests/smoke/alp_backsubstitution.cpp)
#    Triangular linear system solve using forwardsubstitution of lower tridiagonal, real/complex matrix (source: tests/smoke/alp_forwardsubstitution.cpp)
#    Cholesky decomposition of a symmetric/Hermitian positive definite matrix (source: tests/smoke/alp_cholesky.cpp)
#    Householder LU decomposition of a real/complex general matrices (source: tests/smoke/alp_zgetrf.cpp)
#    Inverse of a symmetric/Hermitian positive definite matrix (source code: tests/smoke/alp_potri.cpp)
#    Singular value decomposition of a real/complex general matrix (source code: tests/smoke/alp_zgesvd.cpp)

# This tests are collected and run as ALP smoketests as follows:

cmake -DCMAKE_INSTALL_PREFIX=./install $ALP_SOURCE || ( echo "test failed" &&  exit 1 )
SMOKE_PRINT_TIME=ON make smoketests_alp -j$(nproc)

####################
####################
# Performance tests
####################
####################

# This tests have been executed:
#    On a Kunpeng 920 node with 1 core (alp_dispatch) or 64 cores (alp_omp);
#    Compiling with gcc 9.4.0 compiler;
#    Linking against KunpengBLAS (Kunpeng BoostKit 22.0.RC1) and netlib LAPACK.
#    All tests report time in milliseconds after "time (ms, ...)" text line.
#
# These instructions assume that you are using "Kunpeng BoostKit 22.0.RC1" extracted in a directory BLAS_ROOT
# which should contain include/kblas.h file and the lib/kblas/ directory.
# However, any other blas library could also be used.

####################
# Compilation and execution of the sequential Cholesky decomposition tests
# which are testing our ALP Cholesky implementation, based on the alp_dispatch backend, against the potrf LAPACK functionality.
####################

# Assuming that you are currently in the ALP cloned directory, create a "build" directory and call the following commands from there.
# If no LAPACK library can be found by the compiler in system directories, LAPACK_LIB and LAPACK_INCLUDE have to be properly set and explicitly provided when calling cmake.
# If you are using locally installed kblas, make sure to set proper BLAS_ROOT path to "kml" directory, i.e. extracted boostkit-kml-1.6.0-1.aarch64.rpm.

cmake -DKBLAS_ROOT="$BLAS_ROOT" -DWITH_ALP_DISPATCH_BACKEND=ON -DCMAKE_INSTALL_PREFIX=./install $ALP_SOURCE || ( echo "test failed" &&  exit 1 )
make install -j$(nproc) || ( echo "test failed" &&  exit 1 )

# To compile and run the LAPACK Cholesky test (not ALP code).
# Here you can use gcc flags, i.e. "-L/path/toib/ -llapack" (or simply " -llapack" to use system installed lapack library).
# A consistent test should use the same BLAS in LAPACK as in the ALP-based tests.
install/bin/grbcxx  -b alp_dispatch -o cholesky_lapack_reference.exe $ALP_SOURCE/tests/performance/lapack_cholesky.cpp $LAPACK_LIB/liblapack.a -I$LAPACK_INCLUDE -lgfortran || ( echo "test failed" &&  exit 1 )
./cholesky_lapack_reference.exe -n 1024 -repeat 10 || ( echo "test failed" &&  exit 1 )

# Run the Cholesky ALP dispatch test. 
# Some facts about the test:
#    The algorithm is a blocked variant of Cholesky with block size BS = 64 (as done in LAPACK).
#    It recursively requires an unblocked version of the same algorithm (of size BSxBS) which does not dispatch to LAPACK.
#    All BLAS functions needed by the algorithm are dispatched to the external BLAS library.
make test_alp_cholesky_perf_alp_dispatch -j$(nproc) || ( echo "test failed" &&  exit 1 )
tests/performance/alp_cholesky_perf_alp_dispatch -n 1024 -repeat 10 || ( echo "test failed" &&  exit 1 )

####################
# Compilation and execution of shared memory parallel mxm tests
# which are testing our ALP shared memory backend (alp_omp) mxm implementation against the BLAS's gemm functionality.
# mxm is an inplace, ALP primitive that computes C = C + A*B, with matrices of conforming sizes.
####################

# Our current shared memory backend implementation is not very flexible and can only use squared thread grids.
# In the tests below we run both LAPACK and ALP using 64 threads.
# To ensure a fair comparison, we link with the omp version of KunpengBLAS.
#
# You can compile with omp version of kblas library by additionally providing " -DKBLAS_IMPL=omp"  flag when calling cmake.
# However, this should be compiled in a different directory from the other blas calls, as follows:
CWD=$(pwd)
ompbuild="build_with_omp_blas"
rm -rf $ompbuild && mkdir $ompbuild && cd $ompbuild
cmake -DKBLAS_ROOT="$BLAS_ROOT" -DKBLAS_IMPL=omp -DWITH_ALP_DISPATCH_BACKEND=ON -DCMAKE_INSTALL_PREFIX=./install $ALP_SOURCE || ( echo "test failed" &&  exit 1 )
make install  -j$(nproc) || ( echo "test failed" &&  exit 1 )

# Compile and run gemm-based BLAS test.
install/bin/grbcxx -b alp_dispatch -o blas_mxm.exe $ALP_SOURCE/tests/performance/blas_mxm.cpp -lgfortran || ( echo "test failed" &&  exit 1 )
OMP_NUM_THREADS=64 ./blas_mxm.exe -n 1024 -repeat 10 || ( echo "test failed" &&  exit 1 )
cd $CWD

# Run mxm omp test.
# Some facts about the ALP test:
#    The ALP mxm implementation is based on a 2.5D algorithm;
#    In this test we execute with a 3D thread grid of size 4x4x4;
#    We set OMP_NUM_THREADS=64 threads and fix GOMP_CPU_AFFINITY="0-15 24-39 48-63 72-87" to reflect the NUMA domains in the node;
#    The algorithm is allocating memory using a 2D block-cyclic layout with blocks of size 128x128.

make test_alp_mxm_perf_alp_omp -j$(nproc) || ( echo "test failed" &&  exit 1 )
GOMP_CPU_AFFINITY="0-15 24-39 48-63 72-87" OMP_NUM_THREADS=64 tests/performance/alp_mxm_perf_alp_omp -n 1024 -repeat 10 || ( echo "test failed" &&  exit 1 )

