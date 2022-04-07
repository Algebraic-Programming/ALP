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
# Creation of all shared-memory backends (reference and reference_omp, a.k.a. OMP),
# both as static and dynamic library. Any target importing a backend also imports
# the compiler definition(s) required to set it as default. If a target want to do it
# manually, the target 'backend_shmem_static' exists with no default backend selection
# in its compilation interface.
#

assert_valid_variables( BACKEND_LIBRARY_OUTPUT_NAME VERSION
	SHMEM_BACKEND_INSTALL_DIR INCLUDE_INSTALL_DIR
	REFERENCE_BACKEND_DEFAULT_NAME REFERENCE_OMP_BACKEND_DEFAULT_NAME
	REFERENCE_SELECTION_DEFS REFERENCE_OMP_SELECTION_DEFS backend_reference_srcs
)

assert_defined_targets( backend_flags )

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

if( ${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.19.0" )

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
	# this is the actual binary file, i.e. the one to be installed
	install( TARGETS backend_shmem_static
		EXPORT GraphBLASTargets
		ARCHIVE DESTINATION "${SHMEM_BACKEND_INSTALL_DIR}"
	)

	## DYNAMIC
	make_reference_target( backend_shmem_shared SHARED "shmem" )
	target_link_libraries( backend_shmem_shared PRIVATE ${backend_shmem_base_headers} )

	install( TARGETS backend_shmem_shared
		EXPORT GraphBLASTargets
		LIBRARY DESTINATION "${SHMEM_BACKEND_INSTALL_DIR}"
	)

	add_dependencies( libs backend_shmem_static )
	add_dependencies( libs backend_shmem_shared )
endif()


### CONSUMABLE TARGETS
### i.e. targets with the default backend already set in the compilation interface:
### if you import one of them, than it is already selected as a backend

if( WITH_REFERENCE_BACKEND )

	## reference static
	if( ${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.19.0" )
		# with CMake 3.19 or higher interface targets can inherit output properties like filename;
		# hence, just do that and link the relevant libraries as interface dependencies
		add_library( backend_reference_static INTERFACE )
		target_link_libraries( backend_reference_static INTERFACE backend_shmem_static )
		target_link_libraries( backend_reference_static INTERFACE backend_reference_headers )
	else()
		# otherwise, we must create a dedicated binary
		make_reference_target( backend_reference_static STATIC "reference" )
		target_link_libraries( backend_reference_static PUBLIC backend_reference_headers )
		add_dependencies( libs backend_reference_static )
	endif()
	# add the suitable definitions for backend selection for users' targets to get them automatically
	target_compile_definitions( backend_reference_static INTERFACE "${REFERENCE_SELECTION_DEFS}" )

	install( TARGETS backend_reference_static
		EXPORT GraphBLASTargets
		ARCHIVE DESTINATION "${SHMEM_BACKEND_INSTALL_DIR}"
	)

	## reference shared
	if( ${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.19.0" )
		add_library( backend_reference_shared INTERFACE )
		target_link_libraries( backend_reference_shared INTERFACE backend_shmem_shared )
		target_link_libraries( backend_reference_shared INTERFACE backend_reference_headers )
	else()
		make_reference_target( backend_reference_shared SHARED "reference" )
		target_link_libraries( backend_reference_shared PUBLIC backend_reference_headers )
		add_dependencies( libs backend_reference_shared )
	endif()
	target_compile_definitions( backend_reference_shared INTERFACE "${REFERENCE_SELECTION_DEFS}" )

	install( TARGETS backend_reference_shared
		EXPORT GraphBLASTargets
		LIBRARY DESTINATION "${SHMEM_BACKEND_INSTALL_DIR}"
	)

	# this is an alias for add_grb_executables() to select the backend to link against
	# DO NOT CHANGE THE ALIAS NAME!
	add_library( "${REFERENCE_BACKEND_DEFAULT_NAME}" ALIAS backend_reference_static )

endif( WITH_REFERENCE_BACKEND )


if( WITH_OMP_BACKEND )

	## reference_omp static
	if( ${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.19.0" )
		add_library( backend_reference_omp_static INTERFACE )
		target_link_libraries( backend_reference_omp_static INTERFACE backend_shmem_static )
		target_link_libraries( backend_reference_omp_static INTERFACE backend_reference_omp_headers )
	else()
		make_reference_target( backend_reference_omp_static STATIC "reference_omp" )
		target_link_libraries( backend_reference_omp_static PUBLIC backend_reference_omp_headers )
		add_dependencies( libs backend_reference_omp_static )
	endif()
	target_compile_definitions( backend_reference_omp_static INTERFACE "${REFERENCE_OMP_SELECTION_DEFS}" )

	install( TARGETS backend_reference_omp_static
		EXPORT GraphBLASTargets
		ARCHIVE DESTINATION "${SHMEM_BACKEND_INSTALL_DIR}"
	)

	## reference_omp shared
	if( ${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.19.0" )
		add_library( backend_reference_omp_shared INTERFACE )
		target_link_libraries( backend_reference_omp_shared INTERFACE backend_shmem_shared )
		target_link_libraries( backend_reference_omp_shared INTERFACE backend_reference_omp_headers )
	else()
		make_reference_target( backend_reference_omp_shared SHARED "reference_omp" )
		target_link_libraries( backend_reference_omp_shared PUBLIC backend_reference_omp_headers )
		add_dependencies( libs backend_reference_omp_shared )
	endif()
	target_compile_definitions( backend_reference_omp_shared INTERFACE "${REFERENCE_OMP_SELECTION_DEFS}" )

	install( TARGETS backend_reference_omp_shared
		EXPORT GraphBLASTargets
		LIBRARY DESTINATION "${SHMEM_BACKEND_INSTALL_DIR}"
	)

	# this is an alias for add_grb_executables() to select the backend to link against
	# DO NOT CHANGE THE ALIAS NAME!
	add_library( "${REFERENCE_OMP_BACKEND_DEFAULT_NAME}" ALIAS backend_reference_omp_static )

endif( WITH_OMP_BACKEND )

