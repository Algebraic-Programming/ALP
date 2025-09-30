# ALP Python Bindings Installation Guide

This guide provides step-by-step instructions to build and install the ALP Python bindings.

## Installation Steps

### 1. Clone ALP Repository
```bash
git clone <ALP_REPOSITORY_URL>
cd ALP
```

### 2. Initialize Submodules
```bash
# Initialize and update the pybind11 submodule
git submodule update --init --recursive
```

### 3. Configure Build System
```bash
mkdir build && cd build
cmake .. -DENABLE_PYALP=ON
```

### 4. Build Python Modules
```bash
make -j
# OR build specific modules only:
# make pyalp_ref        # Reference backend
# make pyalp_omp        # OpenMP backend  
# make pyalp_nonblocking # Non-blocking backend
```

### 5. Set Python Path
```bash
export PYTHONPATH=$(pwd)/python
```

### 6. Test Installation
```bash
python3 ../tests/python/test.py
# Expected output should show:
# - Matrix construction
# - Solver iterations and residual
# - Final solution vector
# - Success assertion
```

## Available Python Modules

After successful installation, you'll have access to:

- **`pyalp_ref`** - Reference backend (sequential)
- **`pyalp_omp`** - OpenMP backend (shared-memory parallel)
- **`pyalp_nonblocking`** - Non-blocking backend (asynchronous)