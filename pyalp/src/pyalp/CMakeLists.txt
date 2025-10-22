
cmake_minimum_required(VERSION 3.15)
project(pyalp_bindings LANGUAGES CXX)

include(GNUInstallDirs)

# Prefer using an installed pybind11 via find_package. scikit-build-core/CMake will help
# provide a proper Python interpreter and pybind11 config when used by the build frontend.
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG)

if(NOT pybind11_FOUND)
	# Fallback when building inside the ALP repo where pybind11 is vendored
	message(WARNING "pybind11 not found by find_package — falling back to in-tree extern/pybind11")
	set(_vendored_pybind11_dir "${CMAKE_CURRENT_SOURCE_DIR}/../../extern/pybind11")
	if(EXISTS ${_vendored_pybind11_dir})
		add_subdirectory(${_vendored_pybind11_dir} ${CMAKE_BINARY_DIR}/pyalp/extern/pybind11)
	else()
		message(FATAL_ERROR "pybind11 not found and vendored copy not found at ${_vendored_pybind11_dir}")
	endif()
endif()

# Define the extension; name _pyalp to match package import (pyalp._pyalp)
pybind11_add_module(_pyalp bindings.cpp)
target_compile_definitions(_pyalp PRIVATE PYALP_MODULE_NAME=_pyalp)
set_target_properties(_pyalp PROPERTIES CXX_STANDARD 14 CXX_STANDARD_REQUIRED YES)

# Add the pyalp local headers (one directory up) so includes like "utils.hpp" resolve.
target_include_directories(_pyalp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)

# If your bindings depend on other ALP libraries, link them here. Example:
# target_link_libraries(_pyalp PRIVATE backend_shmem_shared backend_reference common_flags)

# Try to locate ALP/GraphBLAS headers (graphblas.hpp). There are two common cases:
# 1) Building inside the full ALP repo: headers are available at ../.. (relative to pyalp/)
# 2) Building from a packaged environment: the user can set ALP_ROOT environment variable
#    to point to the ALP repo root (which contains the top-level include/ directory).

if(NOT TARGETS_ALP_HEADERS)
	find_path(GRAPHBLAS_INCLUDE_DIR
			  NAMES graphblas.hpp
			  HINTS $ENV{ALP_ROOT}
			  PATHS ${CMAKE_CURRENT_SOURCE_DIR}/../../../../include ${CMAKE_CURRENT_SOURCE_DIR}/../../include ${CMAKE_CURRENT_SOURCE_DIR}/../../../include
			  NO_DEFAULT_PATH)

	if(GRAPHBLAS_INCLUDE_DIR)
		message(STATUS "Found graphblas headers in ${GRAPHBLAS_INCLUDE_DIR}")
		target_include_directories(_pyalp PRIVATE ${GRAPHBLAS_INCLUDE_DIR})
		# If there are ALP libraries to link, you can locate them via find_library here
	else()
		message(FATAL_ERROR "Could not find ALP GraphBLAS headers (graphblas.hpp).
Please either:
  - Build the full ALP project so headers and libraries are available, then run wheel build from the repo root, or
  - Set the environment variable ALP_ROOT to point to the ALP repository root (the directory that contains 'include/graphblas.hpp').
")
	endif()
endif()

# Install the module to the standard library location so scikit-build-core can collect it.
install(TARGETS _pyalp
		LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
		RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
		ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}")
