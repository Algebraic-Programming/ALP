# This file is intended to provide instructins for  initial comparison of performances of alp_dispatch and alp_omp backeds.
# All tests report time in milliseconds after "times(per repeat)" text.
#
# Version tested with with kblas (Kunpeng BoostKit 22.0.RC1) and netliblapack using gcc compiler.
# Standard ALP/grapBLAS dependencies are required
#    LibNUMA: -lnuma
#    Standard math library: -lm
#    POSIX threads: -lpthread
#    OpenMP: -fopenmp in the case of GCC
#
# These instructions assume that you are using "Kunpeng BoostKit 22.0.RC1" installed in a directory BLAS_LIB
# which should containg include/kblas.h file lib/kblas/ directory form extracted kblas file.
# However, any other blas library could also be used.


####################
# Compilation and execution of cholesky decomposition tests
# which are tesing alp_cholesky_perf_alp_dispatch against cholesky_lapack_reference (reference).
####################

# Assuming that you are currently in the "ALP/grapBLAS" source directory (where repository is cloned), you can create "build" directory and call the following commands from there.
# If no lapack library can be found by compiler in system directories, LAPACK_LIB and LAPACK_INCLUDE have to be properly set and explictly provided in camke call.
# If you are using locally installed kblas, make sure to set proper BLAS_LIB path to "kml" directory, i.e. extracted boostkit-kml-1.6.0-1.aarch64.rpm

export GRAPHBLAS_BUILD="$(pwd)"
export GRAPHBLAS_SOURCE="$(realpath ../)"
export LAPACK_LIB="/path/to/lapack/netlib/build/lib"
export LAPACK_INCLUDE="/path/to/lapack/netlib/lapack-3.9.1/LAPACKE/include/"
export BLAS_LIB="/path/to/kunpengblas/boostkit-kml-1.6.0.aarch64/usr/local/kml"

# By default wthis will link against nolocking kblas library 
cmake -DKBLAS_ROOT="$BLAS_LIB" -DWITH_ALP_DISPATCH_BACKEND=ON -DCMAKE_INSTALL_PREFIX=./install $GRAPHBLAS_SOURCE || ( echo "test failed" &&  exit 1 )
make install -j20 || ( echo "test failed" &&  exit 1 )

# To compile and run lapack cholesky (reference library), not alp code.
# Here you can use gcc flags, i.e. "-L/path/toib/ -llapack", or simply " -llapack" to use system installed lapack library.
# A consisten test should use the same blas in lapack as in the other tests.
install/bin/grbcxx  -b alp_dispatch -o cholesky_lapack_reference.exe $GRAPHBLAS_SOURCE/tests/performance/lapack_cholesky.cpp $LAPACK_LIB/liblapack.a -I$LAPACK_INCLUDE -lgfortran || ( echo "test failed" &&  exit 1 )
./cholesky_lapack_reference.exe -n 1024 -repeat 10 || ( echo "test failed" &&  exit 1 )

# Run cholesky alp reference test, non performant test wihich is using internal (unotimized) blas functions.
make test_alp_cholesky_perf_alp_reference || ( echo "test failed" &&  exit 1 )
tests/performance/alp_cholesky_perf_alp_reference -n 1024 -repeat 10 || ( echo "test failed" &&  exit 1 )

# Run cholesky alp dispatch test, test alp_cholesky algorithm with blas function offloaded to external blas library.
make test_alp_cholesky_perf_alp_dispatch VERBOSE=3 || ( echo "test failed" &&  exit 1 )
tests/performance/alp_cholesky_perf_alp_dispatch -n 1024 -repeat 10 || ( echo "test failed" &&  exit 1 )

####################
# Compilation and execution of paralel mxm tests
# which are tesing alp_mxm_perf_alp_dispatch against blas_mxm.exe (reference).
####################

# The current implementation is not very flexible and is using fixed GOMP_CPU_AFFINITY="0-15 24-39 48-63 72-87" OMP_NUM_THREADS=64 variables.
# Depending on blas library you can test different blas implementation.
# To have a fair comparison with the test_alp_mxm_perf_alp_omp we should lik with omp version of blas and use 64 threads in the following example.
#
# You can cmpile with omp version of kblas library by additionally providing " -DKBLAS_IMPL=omp"  flag to cmake call.
# However, this should be compiled in a different directory from the other blas calls. For example
CWD=$(pwd)
ompbuild="build_with_omp_blas"
rm -rf $ompbuild && mkdir $ompbuild && cd $ompbuild
cmake -DKBLAS_ROOT="$BLAS_LIB" -DKBLAS_IMPL=omp -DWITH_ALP_DISPATCH_BACKEND=ON -DCMAKE_INSTALL_PREFIX=./install $GRAPHBLAS_SOURCE || ( echo "test failed" &&  exit 1 )
make install -j20 || ( echo "test failed" &&  exit 1 )
# Compile and run blas mxm (reference library).
install/bin/grbcxx -b alp_dispatch -o blas_mxm.exe $GRAPHBLAS_SOURCE/tests/performance/blas_mxm.cpp -lgfortran || ( echo "test failed" &&  exit 1 )
OMP_NUM_THREADS=64 ./blas_mxm.exe -n 1024 -repeat 10 || ( echo "test failed" &&  exit 1 )
cd $CWD

# Now we go back to the main build directoy where the sequential blas is used
# Run mxm dispatch test.
make test_alp_mxm_perf_alp_dispatch || ( echo "test failed" &&  exit 1 )
tests/performance/alp_mxm_perf_alp_dispatch -n 1024 -repeat 10 || ( echo "test failed" &&  exit 1 )

# Run mxm omp test.
make test_alp_mxm_perf_alp_omp || ( echo "test failed" &&  exit 1 )
GOMP_CPU_AFFINITY="0-15 24-39 48-63 72-87" OMP_NUM_THREADS=64 tests/performance/alp_mxm_perf_alp_omp -n 1024 -repeat 10 || ( echo "test failed" &&  exit 1 )

