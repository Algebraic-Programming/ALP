# MLIR Backend

## What you are executing

We will execute a single mxm (aka GEMM) operation in a loop of 5
iterations. See the C++ program is located in tests/unit/gemm_mlir.cpp

The call to mxm will build linalg IR in MLIR and it will jit-compile
and execute it.

## Build and run the mxm exmaple.
```

#!/bin/bash

# Deps.
# Ninja build: apt-get install ninja-build
# Git: apt-get install git
# make: apt-get install cmake
# python3: apt-get install python3
# We also provide a docker:
#
# docker pull lchelini/alp
# docker run -it lchelini/alp
# ./runme.sh
#

# ALP
echo "Cloning ALP..."
git clone -b mlir https://gitee.com/CSL-ALP/graphblas.git
cd graphblas

# LLVM
echo "Cloning LLVM..."
git clone -b alp https://github.com/chelini/llvm-project.git

mkdir -p llvm-project/build
cd llvm-project/build
echo "Building LLVM..."
cmake -G Ninja ../llvm \
   -DLLVM_ENABLE_PROJECTS=mlir \
   -DLLVM_BUILD_EXAMPLES=ON \
   -DLLVM_TARGETS_TO_BUILD="X86;NVPTX;AMDGPU" \
   -DCMAKE_BUILD_TYPE=Release \
   -DLLVM_ENABLE_ASSERTIONS=ON 
ninja

cd ../../

# Configure and build ALP
export MLIR_DIR=$(pwd)/llvm-project/build/lib/cmake/mlir/
export LLVM_DIR=$(pwd)/llvm-project/build/lib/cmake/llvm/
echo "Building ALP..."
mkdir -p build
cd build
cmake ../
make -j32
make -j build_tests_backend_mlir
echo "Running GEMM Test..."
./tests/unit/gemm_mlir_debug_mlir

```

## What to expect as output

The printed llvm module and the operation's result:

```
50 50 50 50 50 
50 50 50 50 50 
50 50 50 50 50 
50 50 50 50 50 
50 50 50 50 50 
Test OK

```
