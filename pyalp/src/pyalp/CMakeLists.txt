cmake_minimum_required(VERSION 3.14)
project(pyalp_bindings LANGUAGES CXX)

# Use the pinned pybind11 submodule
add_subdirectory(${CMAKE_SOURCE_DIR}/pyalp/extern/pybind11 ${CMAKE_BINARY_DIR}/pyalp/extern/pybind11)

pybind11_add_module(_pyalp bindings.cpp ../matrix_wrappers.hpp ../utils.hpp ../vector_wrappers.hpp ../conjugate_gradient.hpp)
target_compile_definitions(_pyalp PRIVATE PYALP_MODULE_NAME=_pyalp)
set_target_properties(_pyalp PROPERTIES CXX_STANDARD 14 CXX_STANDARD_REQUIRED YES)cmake_minimum_required(VERSION 3.14)
project(pyalp_bindings LANGUAGES CXX)

# Expect pybind11 to be available under ../../extern/pybind11 when used in this repo.
add_subdirectory(${CMAKE_SOURCE_DIR}/pyalp/extern/pybind11 ${CMAKE_BINARY_DIR}/pyalp/extern/pybind11)

pybind11_add_module(_pyalp bindings.cpp ../matrix_wrappers.hpp ../utils.hpp ../vector_wrappers.hpp ../conjugate_gradient.hpp)
target_compile_definitions(_pyalp PRIVATE PYALP_MODULE_NAME=_pyalp)
set_target_properties(_pyalp PROPERTIES CXX_STANDARD 14 CXX_STANDARD_REQUIRED YES)
