#
#   Copyright 2021 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

#[===================================================================[
#
# ROOT FILE FOR THE GraphBLAS project
# content:
#  - project options and related logic
#  - dependencies checks
#  - definition of compilation options
#
#]===================================================================]

cmake_minimum_required( VERSION 3.13 )

set( MAJORVERSION 0 )
set( MINORVERSION 4 )
set( BUGVERSION 0 )
set( VERSION "${MAJORVERSION}.${MINORVERSION}.${BUGVERSION}" )

# set the project name
project( GraphBLAS
	VERSION ${VERSION}
	DESCRIPTION "The ultimate engine for sparse computation"
	LANGUAGES CXX C
)
set( CMAKE_CXX_STANDARD 14 )
set( CMAKE_CXX_STANDARD_REQUIRED ON )

# install within the build directory by default (NOT to /usr/local or the likes)
if( CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT )
	set( CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "default install path" FORCE )
	message( STATUS "setting install directory to ${CMAKE_INSTALL_PREFIX}" )
endif()


### CONFIGURATION OPTIONS
# to choose backends and dependencies
option( WITH_REFERENCE_BACKEND "With Reference backend" ON )
option( WITH_DENSEREF_BACKEND "With Reference Dense backend" ON )
option( WITH_OMP_BACKEND "With OMP backend" ON )
option( WITH_NUMA "With NUMA support" ON )
option( LPF_INSTALL_PATH "Path to the LPF tools for the BSP1D and Hybrid backends" OFF )
# the following options depend on LPF_INSTALL_PATH being set
include(CMakeDependentOption)
cmake_dependent_option( WITH_BSP1D_BACKEND "Build the BSP1D backend (needs \
LPF_INSTALL_PATH set)" ON LPF_INSTALL_PATH OFF
)
cmake_dependent_option( WITH_HYBRID_BACKEND "Also build the Hybrid backend \
(needs LPF_INSTALL_PATH set)" ON LPF_INSTALL_PATH OFF
)
# to customize build flags for either backends or tests
option( COMMON_COMPILE_DEFINITIONS
	"Compilation definitions for BOTH backends and tests; they override the defaults"
	OFF
)
option( COMMON_COMPILE_OPTIONS
	"Compilation options for BOTH backends and tests; they override the defaults"
	OFF
)
option( ADDITIONAL_BACKEND_DEFINITIONS
	"Compilation definitions added to default definitions (or common definitions, \
if set) of backends only; they are ALWAYS APPENDED and never override the defaults"
	OFF
)
option( ADDITIONAL_BACKEND_OPTIONS
	"Compilation options added to default options (or common options, if set) of backends \
only; they are ALWAYS APPENDED and never override the defaults" OFF
)
option( ADDITIONAL_TEST_DEFINITIONS
	"Compilation definitions added to default definitions (or common definitions, if set) \
of tests only; they are ALWAYS APPENDED and never override the defaults" OFF
)
option( ADDITIONAL_TEST_OPTIONS
	"Compilation options added to default options (or common options, if set) of tests only; \
they are ALWAYS APPENDED and never override the defaults" OFF
)
option( TEST_PERFORMANCE_DEFINITIONS
	"Compilation definitions for tests, for performance-related tunings; they override the \
defaults" OFF
)
option( TEST_PERFORMANCE_OPTIONS
	"Compilation options for tests, for performance-related tunings; they override the defaults"
	OFF
)

# to select the directories with the datasets: a value MUST be provided for DATASETS_DIR
option( DATASETS_DIR "Directory with datasets for tests" )
option( GNN_DATASET_PATH "Directory with the GraphChallengeDataset dataset" )

### CHECK THE OPTIONS ARE COHERENT
list( APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" )
include( Utils )

assert_valid_directory( DATASETS_DIR "${PROJECT_SOURCE_DIR}/datasets" )
assert_valid_directory( GNN_DATASET_PATH "" ) # do NOT run GNN tests by default (slow!)

# check whether the user forced distributed backends without providing LPF
if( NOT LPF_INSTALL_PATH AND
	( WITH_BSP1D_BACKEND OR WITH_HYBRID_BACKEND ) )
	message( SEND_ERROR "The BSP1D and Hybrid backends require LPF" )
	message( SEND_ERROR "Hence, you should set LPF_INSTALL_PATH" )
	message( FATAL_ERROR "or not enable WITH_BSP1D_BACKEND or WITH_HYBRID_BACKEND")
endif()

if( NOT WITH_REFERENCE_BACKEND AND
	NOT WITH_OMP_BACKEND AND
	NOT WITH_BSP1D_BACKEND AND
	NOT WITH_HYBRID_BACKEND )
	message( FATAL_ERROR "At least one backend should be enabled")
endif()

# set LPF_INSTALL_PATH as FILEPATH inside the cache
if( LPF_INSTALL_PATH )
	set( LPF_INSTALL_PATH "${LPF_INSTALL_PATH}" CACHE FILEPATH "Path to LPF installation" FORCE )
endif()

if( WITH_BSP1D_BACKEND OR WITH_HYBRID_BACKEND AND ( NOT WITH_NUMA ) )
	message( FATAL_ERROR "BSP1D and Hybrid backends require NUMA support" )
endif()

### CHECK DEPENDENCIES

# always look for math and rt libraries
find_package( LibM REQUIRED )
find_library( LIBRT rt REQUIRED )

# pthreads is needed for hpparser
find_package( Threads REQUIRED )
if( NOT Threads_FOUND AND NOT CMAKE_USE_PTHREADS_INIT )
	message( FATAL_ERROR "A pthread-compatible threading library is needed" )
endif()

# OpenMP is needed to compile the shared memory backend
# TODO remove when dis-entangling (internal issue #290)
find_package( OpenMP REQUIRED )

if( WITH_NUMA )
	find_package( Numa REQUIRED )
endif( )


### CHECK DEPENDENCIES W.R.T. BACKENDS
if( WITH_BSP1D_BACKEND OR WITH_HYBRID_BACKEND )
	find_package( MPI REQUIRED )
	# this MUST find a dedicated .cmake file with a function to compile against LPF
	find_package( LPF REQUIRED )
endif( )


### SETTINGS FOR COMPILATION

set( TEST_CATEGORIES "unit" "smoke" "performance" )

### ADD GRB VARIABLES FOR BUILDING, TESTING AND INSTALLATION

# like compile-time definitions, LPF options and so on
# variables inside AddGRBVars are not cached, hence AddGRBVars MUST be included
# (NOT imported via 'add_subdirectory()') here for variables to be in the same
# scope and propagate down to the other files
include( AddGRBVars )

# here, add information for wrappers generated during installation
include( AddGRBInstall )

# compilation flags for all targets
include( CompileFlags )

if( WITH_BSP1D_BACKEND OR WITH_HYBRID_BACKEND )
	# required to create targets against LPF (provides dedicated functions)
	include( AddLPFTargets )
endif()

### BACKEND HEADERS

# by default no headers are built
set( WITH_REFERENCE_BACKEND_HEADERS OFF )
set( WITH_OMP_BACKEND_HEADERS OFF )

# activate headers based on requested backends
if( WITH_REFERENCE_BACKEND OR WITH_BSP1D_BACKEND )
	# both reference and bsp1d backends need reference headers
	set( WITH_REFERENCE_BACKEND_HEADERS ON )
endif()

if( WITH_OMP_BACKEND OR WITH_HYBRID_BACKEND )
	# both reference_omp and hynrid backends need reference headers
	set( WITH_OMP_BACKEND_HEADERS ON )
endif()

add_subdirectory( include )

### BACKEND IMPLEMENTATIONS
add_subdirectory( src )

### TESTS and EXAMPLES

# specify test categories and the directory where ALL tests are stored
set( TESTS_EXE_OUTPUT_DIR "${PROJECT_BINARY_DIR}/tests" )

include( AddGRBTests )

add_subdirectory( tests )

add_subdirectory( examples )

### DOXYGEN DOCUMENTATION GENERATION

set( DOCS_DIR "${PROJECT_SOURCE_DIR}/docs/code" )
add_custom_command( OUTPUT "${DOCS_DIR}"
	COMMAND bash -c "if [[ ! -d docs/code ]]; then doxygen docs/doxy.conf &> doxygen.log; fi"
	WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
	DEPENDS "${PROJECT_SOURCE_DIR}/docs/doxy.conf"
	COMMENT "producing code documentation in ${DOCS_DIR}"
	VERBATIM
	#USES_TERMINAL
)
add_custom_target( docs DEPENDS "${DOCS_DIR}" )
