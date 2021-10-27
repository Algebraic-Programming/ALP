to compile pLaplacian method:
1. fix the paths, make sure ROPTLIB and ARMA are installed and included in environment variable
2. fix the below paths
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/pouya/graphblas/code/3rd/ROPTLIB
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/pouya/graphblas/code/3rd/arma/lib


3. run:
g++ -std=c++11 -fopenmp -fPIC  -D_GRB_WITH_REFERENCE -D_GRB_WITH_OMP -Wall -Wextra -Iinclude/ -Itests/  -D_GRB_BACKEND=reference_omp -march=native -mtune=native -O3 -funroll-loops -DNDEBUG tests/launcher/pLaplacian_multiway_launcher.cpp /home/pouya/install/graphblas/include/graphblas/algorithms/pLaplacian_spectral_partition.hpp -o pLaplacian_multiway_launcher_omp /home/pouya/install/lib/sequential/libgraphblas.a -lnuma -I 3rd/ROPTLIB/ -I 3rd/ROPTLIB/cwrapper/blas/ -I 3rd/ROPTLIB/cwrapper/lapack/ -L 3rd/ROPTLIB -lropt -Wfatal-errors -I /home/pouya/install/graphblas/3rd/arma/include/ -L /home/pouya/install/graphblas/3rd/arma/lib/ -larmadillo
OMP_PLACES=cores OMP_PROC_BIND=close ./pLaplacian_multiway_launcher_omp datasets/Rect_5pt.mtx direct unweighted Solution_vector.txt 4 2>&1 | tee SomeFile.txt


