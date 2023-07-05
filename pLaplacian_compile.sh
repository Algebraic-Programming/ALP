#!/bin/bash
# g++ -std=c++11 -fopenmp -g -fPIC  -D_GRB_WITH_REFERENCE -Wall -Wextra -Iinclude/ -Itests/  -D_GRB_BACKEND=reference tests/pLaplacian_multi.cpp -o pLaplacian_multi /home/pasadakis/graphblas/lib/sequential/libgraphblas.a -lnuma -I 3rd/ROPTLIB/ -I 3rd/ROPTLIB/cwrapper/blas/ -I 3rd/ROPTLIB/cwrapper/lapack/ -Wfatal-errors -lropt -L 3rd/ROPTLIB
# g++ -std=c++11 -fopenmp -fPIC  -D_GRB_WITH_REFERENCE -D_GRB_WITH_OMP -Wall -Wextra -Iinclude/ -Itests/  -D_GRB_BACKEND=reference_omp -march=native -mtune=native -O3 -funroll-loops -DNDEBUG tests/launcher/pLaplacian_multiway_launcher.cpp /home/pouya/install/graphblas/include/graphblas/algorithms/pLaplacian_spectral_partition.hpp -o pLaplacian_multiway_launcher_omp /home/pouya/install/lib/sequential/libgraphblas.a -lnuma -I 3rd/ROPTLIB/ -I 3rd/ROPTLIB/cwrapper/blas/ -I 3rd/ROPTLIB/cwrapper/lapack/ -L 3rd/ROPTLIB -lropt -Wfatal-errors -I /home/pouya/install/graphblas/3rd/arma/include/ -L /home/pouya/install/graphblas/3rd/arma/lib/ -larmadillo
# g++ -std=c++11 -fopenmp -fPIC  -D_GRB_WITH_REFERENCE -D_GRB_WITH_OMP -Wall -Wextra -Iinclude/ -Itests/  -D_GRB_BACKEND=reference_omp -march=native -mtune=native -O3 -funroll-loops -DNDEBUG tests/launcher/multilevel_partition_launcher.cpp /home/pouya/install/graphblas/include/graphblas/algorithms/multilevel_partition.hpp -o multilevel_partition_omp /home/pouya/install/lib/sequential/libgraphblas.a -lnuma 
# g++ -std=c++11 -fopenmp -fPIC  -D_GRB_WITH_REFERENCE -D_GRB_WITH_OMP -Wall -Wextra -Iinclude/ -Itests/  -D_GRB_BACKEND=reference_omp -march=native -mtune=native -O3 -funroll-loops -DNDEBUG tests/launcher/multilevel_partition_launcher.cpp /home/pouya/install/graphblas/include/graphblas/algorithms/multilevel_partition.hpp -o multilevel_launcher_omp /home/pouya/install/lib/sequential/libgraphblas.a -lnuma -I 3rd/ROPTLIB/ -I 3rd/ROPTLIB/cwrapper/blas/ -I 3rd/ROPTLIB/cwrapper/lapack/ -L 3rd/ROPTLIB -lropt -Wfatal-errors -I /home/pouya/install/graphblas/3rd/arma/include/ -L /home/pouya/install/graphblas/3rd/arma/lib/ -larmadillo
# g++ -std=c++11 -fopenmp -fPIC  -D_GRB_WITH_REFERENCE -D_GRB_WITH_OMP -Wall -Wextra -Iinclude/ -Itests/  -D_GRB_BACKEND=reference_omp -march=native -mtune=native -O3 -funroll-loops -DNDEBUG tests/launcher/multilevel_partition_launcher.cpp /home/pouya/install/graphblas/include/graphblas/algorithms/multilevel_partition.hpp -o multilevel_launcher_omp /home/pouya/install/lib/sequential/libgraphblas.a -lnuma -I 3rd/ROPTLIB/ -I 3rd/ROPTLIB/cwrapper/blas/ -I 3rd/ROPTLIB/cwrapper/lapack/ -L 3rd/ROPTLIB -lropt -Wfatal-errors -I /home/pouya/install/graphblas/3rd/arma/include/ -L /home/pouya/install/graphblas/3rd/arma/lib/ -larmadillo
#grbcxx -Iinclude/ -Itests/ -b reference_omp -O3 tests/launcher/multilevel_partition_launcher.cpp /home/pouya/install/graphblas/include/graphblas/algorithms/multilevel_partition.hpp -o multilevel_launcher_omp /home/pouya/install/lib/sequential/libgraphblas.a -lnuma -I 3rd/ROPTLIB/ -I 3rd/ROPTLIB/cwrapper/blas/ -I 3rd/ROPTLIB/cwrapper/lapack/ -L 3rd/ROPTLIB -lropt -Wfatal-errors -I /home/pouya/install/graphblas/3rd/arma/include/ -L /home/pouya/install/graphblas/3rd/arma/lib/ -larmadillo
#Updated version
EXTRA_FLAGS=
if [[ -z "${ARMADILLO_PATH}" ]]; then
	echo "Warning: no ARMADILLO_PATH found; GCC will revert to using system defaults"
	echo "         (If this is not intended, then please set the ARMADILLO_PATH envvar)"
else
	if [ ! -d "$ARMADILLO_PATH" ]; then
		echo "Warning: Armadillo not found at '${ARMADILLO_PATH}'; GCC will revert to using system defaults"
		echo "         (If this is not intended, then please set the ARMADILLO_PATH envvar)"
		EXTRA_FLAGS=
	else
		ARMADILLO_INCLUDE_PATH=${ARMADILLO_PATH}/include
		EXTRA_FLAGS="-I ${ARMADILLO_INCLUDE_PATH} "
		echo "The Armadillo include path is ${ARMADILLO_INCLUDE_PATH}"
	fi
fi
if [[ ! -z "${ROPTLIB_PATH}" ]]; then
	echo "ROPTLIB_PATH found and reads ${ROPTLIB_PATH}"
	if [[ -z "${ROPTLIB_INCLUDE_PATH}" ]]; then
		ROPTLIB_INCLUDE_PATH=${ROPTLIB_PATH}/
		echo "ROPTLIB_INCLUDE_PATH set to ${ROPTLIB_INCLUDE_PATH}"
	else
		echo "ROPTLIB_INCLUDE_PATH found and reads ${ROPTLIB_INCLUDE_PATH}"
	fi
	if [[ -z "${ROPTLIB_LIBRARY_PATH}" ]]; then
		ROPTLIB_LIBRARY_PATH=${ROPTLIB_PATH}/
		echo "ROPTLIB_LIBRARY_PATH set to ${ROPTLIB_LIBRARY_PATH}"
	else
		echo "ROPTLIB_LIBRARY_PATH found and reads ${ROPTLIB_LIBRARY_PATH}"
	fi
fi
if [[ ! -z "${ROPTLIB_PATH}" ]]; then
	echo "The ROPTLIB path is ${ROPTLIB_PATH}"
	if [ ! -d "$ROPTLIB_PATH" ]; then
		echo "Warning: ROPTLIB_PATH is not found; GCC will revert to using system defaults for ROPTLIB's dependences"
		echo "         (If this is not intended, then please set the ROPTLIB_PATH envvar)"
	else
		EXTRA_FLAGS+="-I ${ROPTLIB_PATH}/cwrapper/blas -I ${ROPTLIB_PATH}/cwrapper/lapack "
	fi
else
	echo "Warning: no ROPTLIB_INCLUDE_PATH set; GCC will revert to using system defaults"
	echo "         (If this is not intended, then please set the ROPTLIB_PATH envvar)"
fi
if [[ ! -z "${ROPTLIB_INCLUDE_PATH}" ]]; then
	echo "The ROPTLIB include path is ${ROPTLIB_INCLUDE_PATH}"
	if [ ! -d "$ROPTLIB_INCLUDE_PATH" ]; then
		echo "Warning: ROPTLIB_INCLUDE_PATH is not found; GCC will revert to using system defaults"
		echo "         (If this is not intended, then please set the ROPTLIB_INCLUDE_PATH envvar)"
	else
		EXTRA_FLAGS+="-I ${ROPTLIB_INCLUDE_PATH} "
	fi
else
	echo "Warning: no ROPTLIB_INCLUDE_PATH set; GCC will revert to using system defaults"
	echo "         (If this is not intended, then please set the ROPTLIB_INCLUDE_PATH envvar)"
fi
if [[ ! -z "${ARMADILLO_PATH}" ]]; then
	if [ -d "${ARMADILLO_PATH}" ]; then
		ARMADILLO_LIBRARY_PATH=${ARMADILLO_PATH}/lib64
		EXTRA_FLAGS+="-L ${ARMADILLO_LIBRARY_PATH} "
		echo "The Armadillo library path is ${ARMADILLO_LIBRARY_PATH}"
	fi
fi
if [[ ! -z "${ROPTLIB_LIBRARY_PATH}" ]]; then
	echo "The ROPTLIB library path is ${ROPTLIB_LIBRARY_PATH}"
	if [ ! -d "$ROPTLIB_LIBRARY_PATH" ]; then
		echo "Warning: ROPTLIB library path is not found; GCC will revert to using system defaults"
		echo "         (If this is not intended, then please set the ROPTLIB_PATH envvar)"
	else
		EXTRA_FLAGS+="-L ${ROPTLIB_LIBRARY_PATH} "
	fi
else
	echo "Warning: no ROPTLIB_LIBRARY_PATH set; GCC will revert to using system defaults"
	echo "         (If this is not intended, then please set the ROPTLIB_PATH envvar)"
fi
echo "The compilation commands looks something like:"
grbcxx --show -Iinclude/ -Itests/ -b reference_omp -O3 -DNDEBUG -funroll-loops -mtune=native -march=native tests/smoke/pLaplacian_multiway_launcher.cpp include/graphblas/algorithms/pLaplacian_spectral_partition.hpp -o pLaplacian_launcher_omp -lnuma ${EXTRA_FLAGS} -lropt -Wfatal-errors -larmadillo #-DDETERMINISTIC
echo "Compiling valgrind version..."
grbcxx  -Iinclude/ -Itests/ -b reference_omp -O0 -march=x86-64 -mtune=x86-64 -g tests/smoke/pLaplacian_multiway_launcher.cpp include/graphblas/algorithms/pLaplacian_spectral_partition.hpp -o pLaplacian_launcher_valgrind ${EXTRA_FLAGS} -lropt -Wfatal-errors -larmadillo #-DDETERMINISTIC
echo "Compiling parallel version..."
grbcxx  -Iinclude/ -Itests/ -b reference_omp -O3 -DNDEBUG -funroll-loops -mtune=native -march=native tests/smoke/pLaplacian_multiway_launcher.cpp include/graphblas/algorithms/pLaplacian_spectral_partition.hpp -o pLaplacian_launcher_omp ${EXTRA_FLAGS} -lropt -Wfatal-errors -larmadillo #-DDETERMINISTIC
echo "Compiling sequential version..."
grbcxx  -Iinclude/ -Itests/ -b reference     -O3 -DNDEBUG -funroll-loops -mtune=native -march=native tests/smoke/pLaplacian_multiway_launcher.cpp include/graphblas/algorithms/pLaplacian_spectral_partition.hpp -o pLaplacian_launcher ${EXTRA_FLAGS} -lropt -Wfatal-errors -larmadillo #-DDETERMINISTIC
#echo "Compiling old sequential version..."
#Original version
#grbcxx  -Iinclude/ -Itests/ -b reference     -O3 -DNDEBUG -funroll-loops -mtune=native -march=native tests/smoke/pLaplacian_multiway_launcher.cpp include/graphblas/algorithms/pLaplacian_spectral_partition.hpp -o pLaplacian_launcher_old ${EXTRA_FLAGS} -lropt -Wfatal-errors -larmadillo  -DPLOLD #-DDETERMINISTIC

