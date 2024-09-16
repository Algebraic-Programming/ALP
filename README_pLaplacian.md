
# P-Laplacian spectral clustering on Grassman manifolds
Clustering algorithm based on [this](https://link.springer.com/article/10.1007/s10994-021-06108-1) paper
# Setup    
In order to compile and run you need to install the following libraries correctly: GraphBLAS, Armadillo and ROPTLIB
## GraphBLAS
See the GraphBLAS [README.md](README.md) for instructions.
To use the external libraries without any extra changes, create a new directory called 3rd using
```
mkdir 3rd
```
in the root directory of the repository
## Armadillo
Download the newest Armadillo library [here](https://arma.sourceforge.net/download.html). Unpack the tar-ball into {graphblas-root}/3rd/armadillo. 

Prerequisite libraries that should be installed:
```
sudo apt-get install libblas-dev
sudo apt-get install liblapack-dev
sudo apt-get install libarpack2-dev
sudo apt-get install superlu-dev
```

Navigate to the directory from the root:
```
cd 3rd/armadillo
```
Build and install the library:
```
make -j 
make -j install
```

See Armadillos [README](/3rd/armadillo/README.md) for more details
## ROPTLIB
Download the newest version of ROPTLIB [here](https://github.com/whuang08/ROPTLIB/releases). Unpack the compressed directory into {graphblas-root}/3rd/ROPTLIB. 

Prerequisite libraries that should be installed:
```
sudo apt-get install build-essential
sudo apt-get install liblapack*
sudo apt-get install libblas*
sudo apt-get install libfftw3*
```

To build the library:
```
make -j ROPTLIB
```

In order to run the executables later, you also need to add ROPTLIB to your library paths:
```
export LD_LIBRARY_PATH={graphblas-root}/3rd/ROPTLIB/
```

See the ROPTLIB [user Manual](https://github.com/whuang08/ROPTLIB/releases/download/0.8/USER_MANUAL_2020-08-11.pdf) for more detailed instructions or for compilation on other OS/programming languages.

# Compilation
Simply run the shell script [pLaplacian_compile.sh](./pLaplacian_compile.sh).
If the external libraries are installed somewhere other than {graphblas-root}/3rd/ then this file needs to be modified to link correctly.

# Running
Summary of command
```
./pLaplacian_launcher[_omp/_old] <dataset> <direct/indirect> <weighted/unweighted> <output_file> <num_clusters> [-p]
```
With <> denoting required parameters, [] denotes optional and / specifies exact options.

Example: 
```
./pLaplacian_launcher_omp ../datasets/delaunay_n11_Nodiag.mtx direct unweighted out.txt 4
```
The dataset should be of matrix market format.

## P=2
The optional parameter in the end [-p] denotes if you should run the entire algorithm to refine the estimation of the eigenvectors. If this is active, the algorithm is skipped and performs the clustering on the armadillo eigenvectors, meaning for p=2.

# Making changes
If changes are made to the algorithm are made in the .hpp files, i.e. [/include/graphblas/algorithms/ROPTLIB/Grassmann_pLap.hpp](/include/graphblas/algorithms/ROPTLIB/Grassmann_pLap.hpp). Then you need to re-install ALP/GraphBLAS before changes take effect. 

If a change is made run:
```
cd build 
make -j install
cd ..
```
Re-compile:
```
./pLaplacian_compile.sh
```
Run:
```
.pLaplacian_launcher ...(args)... 
```

# Updated from previous version
These are changes made from the old version, authored by Anders Hansson.

## Armadillo
The initial guess for the eigen vectors, i.e. for p=2, are now done by the external library armadillo (see [Setup](#setup) for installation).
This change is in [/include/graphblas/algorithms/pLaplacian_spectral_partition.hpp](/include/graphblas/algorithms/pLaplacian_spectral_partition.hpp).

## Parameters
An optional parameter has been added in the end when running, '-p'. 
See [Running](#running) for more info.

## DETERMINISTIC FLAG
A compilation flag has been added to make any random event deterministic. It does this by setting the random seed to the same number every compilation. To enable this you should add -DDETERMINISTIC as a flag while compiling.

The purpose of this is to verify correct results across different runs, making sure all calculations are correct. The parallel will though not be fully deterministic due to scheduling differences.

## Hessian
The hessian matrix is now calculated and stored in the EcuGrad() function. It is then reused multiple times in the EcuHess() function, leading to roughly 10 times speedup.

## OLD/NEW Version
If you want to run the old version, you can compile with the flag -DPLOLD. Works for both serial and omp, but is an order of magnitude slower than the new version. The old algorithm can be found in [this file](/include/graphblas/algorithms/ROPTLIB/Grassmann_pLap_old.hpp)

