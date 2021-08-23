# g++ -std=c++11 -fopenmp -g -fPIC  -D_GRB_WITH_REFERENCE -Wall -Wextra -Iinclude/ -Itests/  -D_GRB_BACKEND=reference tests/pLaplacian_multi.cpp -o pLaplacian_multi /home/pasadakis/graphblas/lib/sequential/libgraphblas.a -lnuma -I 3rd/ROPTLIB/ -I 3rd/ROPTLIB/cwrapper/blas/ -I 3rd/ROPTLIB/cwrapper/lapack/ -Wfatal-errors -lropt -L 3rd/ROPTLIB
g++ -std=c++11 -fopenmp -fPIC  -D_GRB_WITH_REFERENCE -D_GRB_WITH_OMP -Wall -Wextra -Iinclude/ -Itests/  -D_GRB_BACKEND=reference_omp -march=native -mtune=native -O3 -funroll-loops -DNDEBUG tests/launcher/pLaplacian_multiway_launcher.cpp /home/pouya/install/graphblas/include/graphblas/algorithms/pLaplacian_spectral_partition.hpp -o pLaplacian_multiway_launcher_omp /home/pouya/install/lib/sequential/libgraphblas.a -lnuma -I 3rd/ROPTLIB/ -I 3rd/ROPTLIB/cwrapper/blas/ -I 3rd/ROPTLIB/cwrapper/lapack/ -L 3rd/ROPTLIB -lropt -Wfatal-errors -I /home/pouya/install/graphblas/3rd/arma/include/ -L /home/pouya/install/graphblas/3rd/arma/lib/ -larmadillo
