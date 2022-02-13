# MLIR Backend

## Build and run a simple GEMM operation.
```

#!/bin/bash

# Deps.
# Ninja build: apt-get install ninja-build
# Git: apt-get install git
# make: apt-get install cmake
# python3: apt-get install python3

# ALP
echo "Cloning ALP..."
git clone -b mlir https://gitee.com/CSL-ALP/graphblas.git
cd graphblas

# LLVM
echo "Cloning LLVM..."
git clone -b alp https://github.com/chelini/llvm-project.git

mkdir llvm-project/build
cd llvm-project/build
echo "Building LLVM..."
cmake -G Ninja ../llvm \
   -DLLVM_ENABLE_PROJECTS=mlir \
   -DLLVM_BUILD_EXAMPLES=ON \
   -DLLVM_TARGETS_TO_BUILD="X86;NVPTX;AMDGPU" \
   -DCMAKE_BUILD_TYPE=Release \
   -DLLVM_ENABLE_ASSERTIONS=ON \
ninja

cd ../../

# Configure and build ALP
export MLIR_DIR=$(pwd)/llvm-project/build/lib/cmake/mlir/
export LLVM_DIR=$(pwd)/llvm-project/build/lib/cmake/llvm/
echo "Building ALP..."
mkdir build
cd build
cmake ../
make -j32
make build_tests_backend_mlir
echo "Running GEMM Test..."
./tests/unit/gemm_mlir_debug_mlir

```
