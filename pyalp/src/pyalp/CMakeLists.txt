cmake_minimum_required(VERSION 3.14)
project(pyalp_bindings LANGUAGES CXX)

# Locate pybind11 submodule whether building from repo root (CMAKE_SOURCE_DIR is repo root)
# or from the package root (CMAKE_SOURCE_DIR is pyalp/).
if(EXISTS "${CMAKE_SOURCE_DIR}/extern/pybind11/CMakeLists.txt")
	add_subdirectory(${CMAKE_SOURCE_DIR}/extern/pybind11 ${CMAKE_BINARY_DIR}/extern/pybind11)
elseif(EXISTS "${CMAKE_SOURCE_DIR}/pyalp/extern/pybind11/CMakeLists.txt")
	add_subdirectory(${CMAKE_SOURCE_DIR}/pyalp/extern/pybind11 ${CMAKE_BINARY_DIR}/pyalp/extern/pybind11)
else()
	message(FATAL_ERROR "pybind11 submodule not found. Ensure pyalp/extern/pybind11 is available.")
endif()

# Add GraphBLAS headers include directory. Support both repo-root and package-root layouts.
set(_gb_include_candidates
	"${CMAKE_SOURCE_DIR}/include"           # when building from repo root
	"${CMAKE_SOURCE_DIR}/../include"        # when building from pyalp package root
)
set(_gb_include_dir "")
foreach(_cand IN LISTS _gb_include_candidates)
	if(EXISTS "${_cand}/graphblas.hpp")
		set(_gb_include_dir "${_cand}")
		break()
	endif()
endforeach()
if(NOT _gb_include_dir)
	message(FATAL_ERROR "Could not locate GraphBLAS headers (graphblas.hpp). Please ensure 'include/' is present in the source tree.")
endif()

pybind11_add_module(_pyalp bindings.cpp ../matrix_wrappers.hpp ../utils.hpp ../vector_wrappers.hpp ../conjugate_gradient.hpp)
target_compile_definitions(_pyalp PRIVATE PYALP_MODULE_NAME=_pyalp _GRB_NO_LIBNUMA)
set_target_properties(_pyalp PROPERTIES CXX_STANDARD 14 CXX_STANDARD_REQUIRED YES)
target_include_directories(_pyalp PRIVATE "${_gb_include_dir}")
