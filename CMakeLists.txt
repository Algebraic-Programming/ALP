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
set( MINORVERSION 7 )
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
option( WITH_ALP_REFERENCE_BACKEND "With Reference Dense backend" OFF )
option( WITH_ALP_DISPATCH_BACKEND "With Dispatch Dense backend" OFF )
option( WITH_OMP_BACKEND "With OMP backend" ON )
option( WITH_HYPERDAGS_BACKEND "With Hyperdags backend" ON )
if( WITH_HYPERDAGS_BACKEND )
	if( NOT DEFINED WITH_HYPERDAGS_USING )
		set( WITH_HYPERDAGS_USING "reference" )
	endif()
endif()
option( WITH_NONBLOCKING_BACKEND "With Nonblocking backend" ON )
option( WITH_ALP_OMP_BACKEND "With OMP Dense backend" OFF )
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

option( SPBLAS_PREFIX "Prefix for SparseBLAS library" OFF )
option( ENABLE_SOLVER_LIB "Compile solver library against the reference and nonblocking backends" ON )
option( ENABLE_EXTRA_SOLVER_LIBS "Compile solver library against the reference_omp backend, in addition to reference and nonblocking" OFF )


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
	NOT WITH_NONBLOCKING_BACKEND AND
	NOT WITH_BSP1D_BACKEND AND
	NOT WITH_HYBRID_BACKEND AND
	NOT WITH_HYPERDAGS_BACKEND )
	message( FATAL_ERROR "At least one backend should be enabled")
endif()

# Adding backends for tests only
set( AVAILABLE_TEST_BACKENDS "" )
if( WITH_REFERENCE_BACKEND )
	list( APPEND AVAILABLE_TEST_BACKENDS "reference" )
endif()
if( WITH_OMP_BACKEND )
	list( APPEND AVAILABLE_TEST_BACKENDS "reference_omp" )
endif()
if( WITH_HYPERDAGS_BACKEND )
	list( APPEND AVAILABLE_TEST_BACKENDS "hyperdags" )
endif()
if( WITH_NONBLOCKING_BACKEND )
	list( APPEND AVAILABLE_TEST_BACKENDS "nonblocking" )
endif()
if( WITH_BSP1D_BACKEND )
	list( APPEND AVAILABLE_TEST_BACKENDS "bsp1d" )
endif()
if( WITH_HYBRID_BACKEND )
	list( APPEND AVAILABLE_TEST_BACKENDS "hybrid" )
endif()
if( WITH_ALP_REFERENCE_BACKEND )
	list( APPEND AVAILABLE_TEST_BACKENDS "alp_reference" )
endif()
if( WITH_ALP_DISPATCH_BACKEND )
	list( APPEND AVAILABLE_TEST_BACKENDS "alp_dispatch" )
endif()
if( WITH_ALP_OMP_BACKEND )
	list( APPEND AVAILABLE_TEST_BACKENDS "alp_omp" )
endif()

# Enable backends based on features
if( ENABLE_SOLVER_LIB )
	if( NOT WITH_NONBLOCKING_BACKEND )
		set( WITH_NONBLOCKING_BACKEND ON )
		message( STATUS "Enabling compilation of nonblocking backend: required by solver libraries" )
	endif()
	if( NOT WITH_REFERENCE_BACKEND )
		set( WITH_REFERENCE_BACKEND ON )
		message( STATUS "Enabling compilation of reference backend: required by solver libraries" )
	endif()
endif()

if( ENABLE_EXTRA_SOLVER_LIBS )
	if( NOT WITH_OMP_BACKEND )
		set( WITH_OMP_BACKEND ON )
		message( STATUS "Enabling compilation of reference_omp backend: required by extra solver libraries" )
	endif()
endif()

# Enabling reference_omp backend if bsp1d is active
if( WITH_BSP1D_BACKEND )
	if( NOT WITH_OMP_BACKEND )
		set( WITH_OMP_BACKEND ON )
		message( STATUS "Enabling compilation of reference_omp backend: required by bsp1d" )
	endif()
endif()

# Enabling reference_omp backend if hybrid is active
if( WITH_HYBRID_BACKEND )
	if( NOT WITH_OMP_BACKEND )
		set( WITH_OMP_BACKEND ON )
		message( STATUS "Enabling compilation of reference_omp backend: required by hybrid" )
	endif()
endif()

# Enabling reference_omp backend if non-blocking is active
if( WITH_NONBLOCKING_BACKEND )
	if( NOT WITH_OMP_BACKEND )
		set( WITH_OMP_BACKEND ON )
		message( STATUS "Enabling compilation of reference_omp backend: required by nonblocking" )
	endif()
endif()

# Enabling backend required by hyperdags (if not)
if( WITH_HYPERDAGS_BACKEND )
	if( WITH_HYPERDAGS_USING STREQUAL "reference" )
		if( NOT WITH_REFERENCE_BACKEND )
			set( WITH_REFERENCE_BACKEND ON )
			message( STATUS "Enabling reference backend for hyperdags" )
		endif()
	elseif(WITH_HYPERDAGS_USING STREQUAL "reference_omp" )
		if( NOT WITH_OMP_BACKEND )
			set( WITH_OMP_BACKEND ON )
			message( STATUS "Enabling reference_omp backend for hyperdags" )
		endif()
	elseif(WITH_HYPERDAGS_USING STREQUAL "nonblocking" )
		if( NOT WITH_NONBLOCKING_BACKEND )
			set( WITH_NONBLOCKING_BACKEND ON )
			message( STATUS "Enabling non-blocking backend for hyperdags" )
		endif()
	else()
		message( FATAL_ERROR "Unrecognised backend for hyperdags: ${WITH_HYPERDAGS_USING}.\nAvailable options: reference, reference_omp, nonblocking" )
	endif()
endif()

# Enabling reference backend if reference_omp is active
if( WITH_OMP_BACKEND )
	if( NOT WITH_REFERENCE_BACKEND )
		set( WITH_REFERENCE_BACKEND ON )
		message( STATUS "Enabling compilation of reference backend: required by reference_omp" )
	endif()
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

if( WITH_ALP_DISPATCH_BACKEND )
        include( Blas )
endif( )

# here, add information for wrappers generated during installation
include( AddGRBInstall )

if( WITH_BSP1D_BACKEND OR WITH_HYBRID_BACKEND )
	# required to create targets against LPF (provides dedicated functions)
	include( AddLPFTargets )
endif()

# Coverage flags
if( CMAKE_BUILD_TYPE STREQUAL "Coverage")
	include( Coverage )
endif()

# compilation flags for all targets
include( CompileFlags )

# information for the transition path
include( Transition )

### BACKEND HEADERS

# by default no headers are built
set( WITH_REFERENCE_BACKEND_HEADERS OFF )
set( WITH_ALP_REFERENCE_BACKEND_HEADERS OFF )
set( WITH_ALP_DISPATCH_BACKEND_HEADERS OFF )
set( WITH_OMP_BACKEND_HEADERS OFF )
set( WITH_HYPERDAGS_BACKEND_HEADERS OFF )
set( WITH_ALP_OMP_BACKEND_HEADERS OFF )

# activate headers based on requested backends
if( WITH_REFERENCE_BACKEND OR WITH_BSP1D_BACKEND OR WITH_NONBLOCKING_BACKEND )
	# reference, bsp1d and nonblocking backends need reference headers
	set( WITH_REFERENCE_BACKEND_HEADERS ON )
endif()

if( WITH_HYPERDAGS_BACKEND )
	set( WITH_HYPERDAGS_BACKEND_HEADERS ON )
endif()

if( WITH_OMP_BACKEND OR WITH_HYBRID_BACKEND )
	# both reference_omp and hynrid backends need reference headers
	set( WITH_OMP_BACKEND_HEADERS ON )
endif()

if( WITH_ALP_REFERENCE_BACKEND )
	set( WITH_ALP_REFERENCE_BACKEND_HEADERS ON )
endif()

if( WITH_ALP_DISPATCH_BACKEND )
	set( WITH_ALP_DISPATCH_BACKEND_HEADERS ON )
endif()

if( WITH_ALP_OMP_BACKEND )
	set( WITH_ALP_OMP_BACKEND_HEADERS ON )
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


### COVERAGE REPORT GENERATION

if( CMAKE_BUILD_TYPE STREQUAL "Coverage")
	create_coverage_command( "coverage_json" "coverage.json"  "--json-pretty" )
	create_coverage_command( "coverage_cobertura" "coverage.xml"  "--xml-pretty" )
	create_coverage_command( "coverage_csv" "coverage.csv"  "--csv" )
	create_coverage_command( "coverage_coveralls" "coveralls.json"  "--coveralls" )
	create_coverage_command( "coverage_html" "index.html"  "--html-details" )
endif()


### DOXYGEN DOCUMENTATION GENERATION

set( DOCS_DIR "${PROJECT_SOURCE_DIR}/docs/developer" )
add_custom_command( OUTPUT "${DOCS_DIR}"
	COMMAND bash -c "doxygen docs/doxy.conf &> doxygen-developer.log;"
	WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
	DEPENDS "${PROJECT_SOURCE_DIR}/docs/doxy.conf"
	COMMENT "producing code documentation in ${DOCS_DIR}"
	VERBATIM
	#USES_TERMINAL
)
add_custom_target( devdocs DEPENDS "${DOCS_DIR}" )

set( PUBLIC_DOCS_DIR "${PROJECT_SOURCE_DIR}/docs/user" )
add_custom_command( OUTPUT "${PUBLIC_DOCS_DIR}"
	COMMAND bash -c "doxygen docs/user.conf &> doxygen-user.log;"
	WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
	DEPENDS "${PROJECT_SOURCE_DIR}/docs/user.conf"
	COMMENT "producing public code documentation in ${PUBLIC_DOCS_DIR}"
	VERBATIM
)
add_custom_target( userdocs DEPENDS "${PUBLIC_DOCS_DIR}" )
add_custom_target( docs )
add_dependencies( docs userdocs devdocs )

message( "Tests enabled for backends: ${AVAILABLE_TEST_BACKENDS}" )
message( "Enabled backend targets: ${AVAILABLE_BACKENDS}\n" )

