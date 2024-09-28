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

#
# defines variables for the creation of wrapper scripts and the installation
#

assert_defined_variables( WITH_REFERENCE_BACKEND WITH_OMP_BACKEND WITH_NONBLOCKING_BACKEND
	WITH_BSP1D_BACKEND WITH_HYBRID_BACKEND WITH_NUMA
)
assert_valid_variables( CMAKE_INSTALL_PREFIX AVAILABLE_BACKENDS CMAKE_CXX_COMPILER )

### PATHS FOR INSTALLATION OF TARGETS

# root paths for the various components
set( INCLUDE_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/include" )
set( BIN_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/bin" )
set( BACKEND_LIBRARY_OUTPUT_NAME "graphblas" )
set( ALP_UTILS_LIBRARY_OUTPUT_NAME "alp_utils" )
set( BINARY_LIBRARIES_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/lib" )
set( CMAKE_CONFIGS_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/cmake" )
set( NAMESPACE_NAME "ALPGraphBLAS")

# installation export unit for ALL targets
install( EXPORT GraphBLASTargets
	FILE ALPGraphBLASTargets.cmake
	NAMESPACE "${NAMESPACE_NAME}::"
	DESTINATION "${CMAKE_CONFIGS_INSTALL_DIR}"
)

# paths where to install the binaries of the various backends
set( ALP_UTILS_INSTALL_DIR "${BINARY_LIBRARIES_INSTALL_DIR}" )
set( SHMEM_BACKEND_INSTALL_DIR "${BINARY_LIBRARIES_INSTALL_DIR}/sequential" )
set( HYPERDAGS_BACKEND_INSTALL_DIR "${BINARY_LIBRARIES_INSTALL_DIR}/hyperdags" )
set( BSP1D_BACKEND_INSTALL_DIR "${BINARY_LIBRARIES_INSTALL_DIR}/spmd" )
set( HYBRID_BACKEND_INSTALL_DIR "${BINARY_LIBRARIES_INSTALL_DIR}/hybrid" )

# definitions and options common to all backends:  all backends include
# REFERENCE_INCLUDE_DEFS, REFERENCE_OMP_INCLUDE_DEFS and -fopenmp due to the
# dependency on OpenMP -- to be resolved
set( COMMON_COMPILE_DEFINITIONS  "${REFERENCE_INCLUDE_DEFS};${REFERENCE_OMP_INCLUDE_DEFS}" )
set( COMMON_COMPILE_OPTIONS "-fopenmp" )

# link flags common to all backends, to be inserted after the backend-specific flags
if( WITH_NUMA )
	list( APPEND COMMON_LFLAGS_POST "-lnuma"  )
endif()

# addBackendWrapperGenOptions
# creates the variables to store the settings for a backend, in order to create
# the wrapper scripts for the installation; unless otherwise specified, arguments
# that are not passed are left empty
#
# arguments:
# backend: (mandatory) argument name
# COMPILER_COMMAND: (optional) Bash command (also including options, as a CMake list)
#	to invoke the compiler; if left empty, it is set to CMAKE_CXX_COMPILER
# RUNNER: (optional) runner command (also including options, as a CMake list) to
#	run the executable
# COMPILE_DEFINITIONS: (optional) definitions for compilation, as "SYMBOL" or
#	"KEY=VALUE" (without "-D"!)
# COMPILE_OPTIONS: (optional) options for compilations
# LINK_FLAGS: (optional) flags for linking
#
# WARNING! do NOT turn this into a macro, otherwise escaped paths don't work anymore
# since they are expanded as macro arguments
#
function( addBackendWrapperGenOptions backend lib_dir )
	set( multiValueArgs "COMPILER_COMMAND;RUNNER"
		"COMPILE_DEFINITIONS;COMPILE_OPTIONS;LINK_FLAGS"
	)
	cmake_parse_arguments( parsed "${options}" ""
		"${multiValueArgs}" "${ARGN}"
	)

	if( NOT "${backend}" IN_LIST AVAILABLE_BACKENDS )
		message( FATAL_ERROR "cannot find ${backend} among available backends")
	endif()

	assert_valid_variables( lib_dir )

	set( ${backend}_WRAPPER_COMPILER_COMMAND "${parsed_COMPILER_COMMAND}" PARENT_SCOPE )
	if( NOT parsed_COMPILER_COMMAND )
		set( ${backend}_WRAPPER_COMPILER_COMMAND "${CMAKE_CXX_COMPILER}" PARENT_SCOPE )
	endif()
	set( ${backend}_WRAPPER_RUNNER "${parsed_RUNNER}" PARENT_SCOPE )
	set( ${backend}_LIB_DIR "${lib_dir}" PARENT_SCOPE )

	list( APPEND __cd "${COMMON_COMPILE_DEFINITIONS}" "${parsed_COMPILE_DEFINITIONS}" )
	set( ${backend}_WRAPPER_COMPILE_DEFINITIONS "${__cd}" PARENT_SCOPE )

	list( APPEND __co "${COMMON_COMPILE_OPTIONS}" "${parsed_COMPILE_OPTIONS}" )
	set( ${backend}_WRAPPER_COMPILE_OPTIONS "${__co}" PARENT_SCOPE )
	set( ${backend}_WRAPPER_LINK_FLAGS "${parsed_LINK_FLAGS}" PARENT_SCOPE )
endfunction( addBackendWrapperGenOptions )

### POPULATING WRAPPER INFORMATION FOR INSTALLATION TARGETS
# for each enabled backend, add its information for the wrapper generation
# paths may have spaces, hence wrap them inside single quotes ''

# shared memory backends
if( WITH_REFERENCE_BACKEND )
	addBackendWrapperGenOptions( "reference" "${SHMEM_BACKEND_INSTALL_DIR}"
		COMPILE_DEFINITIONS "${REFERENCE_SELECTION_DEFS}"
	)
endif()

if( WITH_OMP_BACKEND )
	addBackendWrapperGenOptions( "reference_omp" "${SHMEM_BACKEND_INSTALL_DIR}"
		COMPILE_DEFINITIONS "${REFERENCE_OMP_SELECTION_DEFS}"
	)
endif()

# dependent backends
if( WITH_HYPERDAGS_BACKEND )
	addBackendWrapperGenOptions( "hyperdags" "${HYPERDAGS_BACKEND_INSTALL_DIR}"
		COMPILE_DEFINITIONS "${HYPERDAGS_INCLUDE_DEFS};${HYPERDAGS_SELECTION_DEFS}"
	)
endif()

if( WITH_NONBLOCKING_BACKEND )
	addBackendWrapperGenOptions( "nonblocking" "${SHMEM_BACKEND_INSTALL_DIR}"
		COMPILE_DEFINITIONS "${NONBLOCKING_INCLUDE_DEFS};${NONBLOCKING_SELECTION_DEFS}"
	)
endif()

# distributed memory backends
if( WITH_BSP1D_BACKEND OR WITH_HYBRID_BACKEND )
	assert_valid_variables( LPFRUN LPFCPP )

	set( LPF_ENGINE "mpimsg" )

	# since the following commands "inject" paths into Bash scripts, just sanitize
	# them according to the system's rules, e.g. with ' ' -> '\ '
	# CAVEAT: do NOT expand escaped strings for macro arguments, since escape
	# symbols disappear (https://gitlab.kitware.com/cmake/cmake/-/issues/19281)
	file( TO_NATIVE_PATH "${LPFCPP}" LPFCPP_SANITIZED )
	file( TO_NATIVE_PATH "${LPFRUN}" LPFRUN_SANITIZED )

	# this is the command for compilation and execution using LPF
	# if there are spaces in the LPFCPP and LPFRUN paths, then
	# the below assumes they are properly escaped already, as from above
	set( LPF_CXX_COMPILER "${LPFCPP_SANITIZED}" "-engine" "${LPF_ENGINE}" )
	set( LPFRUN_CMD "${LPFRUN_SANITIZED}" "-engine" "${LPF_ENGINE}" )

	set( MANUALRUN_ARGS "-np" "1" )
	set( MANUALRUN "${LPFRUN_CMD}" "${MANUALRUN_ARGS}" )

	if( WITH_BSP1D_BACKEND )
		addBackendWrapperGenOptions( "bsp1d" "${BSP1D_BACKEND_INSTALL_DIR}"
			COMPILER_COMMAND "${LPF_CXX_COMPILER}"
			RUNNER "${LPFRUN_CMD}"
			COMPILE_DEFINITIONS "${LPF_INCLUDE_DEFS};${BSP1D_SELECTION_DEFS}"
		)
	endif()

	if( WITH_HYBRID_BACKEND )
		addBackendWrapperGenOptions( "hybrid" "${HYBRID_BACKEND_INSTALL_DIR}"
			COMPILER_COMMAND "${LPF_CXX_COMPILER}"
			RUNNER "${LPFRUN_CMD}"
			COMPILE_DEFINITIONS "${LPF_INCLUDE_DEFS};${HYBRID_SELECTION_DEFS}"
		)
	endif()
endif()

