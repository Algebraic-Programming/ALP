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
# This file creates the basic target(s) needed by all backends
#

assert_defined_targets( backend_headers_nodefs backend_flags )

assert_valid_variables( ALP_UTILS_LIBRARY_OUTPUT_NAME BACKEND_LIBRARY_OUTPUT_NAME VERSION
	SHMEM_BACKEND_INSTALL_DIR
)

# convenience variable: a few tests need the path of utility sources
set( ALP_UTILS_SRC_PATH "${CMAKE_CURRENT_SOURCE_DIR}/utils"
	CACHE INTERNAL "ALP utility sources path"
)

# little utility library needed by backends
add_library( alp_utils_static STATIC
	${ALP_UTILS_SRC_PATH}/hpparser.c
)
set_target_properties( alp_utils_static PROPERTIES
	OUTPUT_NAME "${ALP_UTILS_LIBRARY_OUTPUT_NAME}"
)
target_compile_definitions( alp_utils_static PRIVATE _GNU_SOURCE )
target_link_libraries( alp_utils_static
	PRIVATE Threads::Threads backend_flags
	PUBLIC alp_utils_headers
)
install( TARGETS alp_utils_static
	EXPORT GraphBLASTargets
	ARCHIVE DESTINATION "${ALP_UTILS_INSTALL_DIR}"
)

add_library( alp_utils_dynamic SHARED
	${ALP_UTILS_SRC_PATH}/hpparser.c
)
set_target_properties( alp_utils_dynamic PROPERTIES
	OUTPUT_NAME "${ALP_UTILS_LIBRARY_OUTPUT_NAME}"
)
target_compile_definitions( alp_utils_dynamic PRIVATE _GNU_SOURCE )
target_link_libraries( alp_utils_dynamic
	PRIVATE Threads::Threads backend_flags
	PUBLIC alp_utils_headers
)
install( TARGETS alp_utils_dynamic
	EXPORT GraphBLASTargets
	LIBRARY DESTINATION "${ALP_UTILS_INSTALL_DIR}"
)

### ACTUAL BACKEND IMPLEMENTATIONS

# target to compile all backend libraries
add_custom_target( libs ALL )

# the sources common to all single-process (aka shmem) backends
set( backend_reference_srcs
	${CMAKE_CURRENT_SOURCE_DIR}/descriptors.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/rc.cpp
)

# source files common to all BSP-based backends
set( backend_bsp_srcs
	${CMAKE_CURRENT_SOURCE_DIR}/bsp/utils.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/bsp/collectives.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/bsp/exec_broadcast_routines.cpp
)

# include only selected backends
if( WITH_REFERENCE_BACKEND OR WITH_OMP_BACKEND )
	add_subdirectory( reference )
endif()

if( WITH_HYPERDAGS_BACKEND )
	add_subdirectory( hyperdags )
endif()

if( WITH_NONBLOCKING_BACKEND )
	add_subdirectory( nonblocking )
endif()

if( WITH_DENSE_BACKEND )
	add_subdirectory( denseref )
endif()

if( WITH_BSP1D_BACKEND OR WITH_HYBRID_BACKEND )
	add_subdirectory( bsp1d )
endif()


# macro to create a target for the shared memory library, with all minimal properties
#
# Parameters:
#   targetName name of the target
#   targetType type of library (STATIC or SHARED)
#   outDir directory to compile the binary to (not install)
#
macro( make_reference_target targetName targetType outDir )

	add_library( "${targetName}" "${targetType}"
		"${backend_reference_srcs}"
	)
	set_target_properties( "${targetName}" PROPERTIES
		OUTPUT_NAME "${BACKEND_LIBRARY_OUTPUT_NAME}"
	)
	if( "${targetType}" STREQUAL "SHARED" )
		set_target_properties( "${targetName}" PROPERTIES
			SOVERSION "${VERSION}"
			LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${outDir}"
		)
	elseif( "${targetType}" STREQUAL "STATIC" )
		set_target_properties( "${targetName}" PROPERTIES
			ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${outDir}"
		)
	else()
		message( FATAL_ERROR "Unknown library type: ${targetType}")
	endif()
	target_link_libraries( "${targetName}" PRIVATE backend_flags )

endmacro( make_reference_target )


### BINARY-ONLY TARGETS: create the actual library with all code inside
### but storing only the minimum interface definitions (e.g. no OMP);
### no default backend is set!

if( WITH_REFERENCE_BACKEND OR WITH_OMP_BACKEND OR WITH_NONBLOCKING_BACKEND )

	# alias target for basic propagation of headers and definitions
	# from existing header target, which depends on the enabled backends;
	# use the most advanced ones, i.e. those of reference_omp if available
	if( WITH_OMP_BACKEND_HEADERS )
		set( backend_shmem_base_headers backend_reference_omp_headers)
	elseif( WITH_REFERENCE_BACKEND_HEADERS )
		set( backend_shmem_base_headers backend_reference_headers )
	endif()

	## STATIC
	make_reference_target( backend_shmem_static STATIC "shmem" )
	target_link_libraries( backend_shmem_static PRIVATE ${backend_shmem_base_headers} )
	if( WITH_NONBLOCKING_BACKEND )
		target_link_libraries( backend_shmem_static PRIVATE ${backend_nonblocking_headers} )
	endif()
	# this is the actual binary file, i.e. the one to be installed
	install( TARGETS backend_shmem_static
		EXPORT GraphBLASTargets
		ARCHIVE DESTINATION "${SHMEM_BACKEND_INSTALL_DIR}"
	)

	## DYNAMIC
	make_reference_target( backend_shmem_shared SHARED "shmem" )
	target_link_libraries( backend_shmem_shared PRIVATE ${backend_shmem_base_headers} )
	if( WITH_NONBLOCKING_BACKEND )
		target_link_libraries( backend_shmem_shared PRIVATE ${backend_nonblocking_headers} )
	endif()
	install( TARGETS backend_shmem_shared
		EXPORT GraphBLASTargets
		LIBRARY DESTINATION "${SHMEM_BACKEND_INSTALL_DIR}"
	)

	add_dependencies( libs backend_shmem_static )
	add_dependencies( libs backend_shmem_shared )

endif()

