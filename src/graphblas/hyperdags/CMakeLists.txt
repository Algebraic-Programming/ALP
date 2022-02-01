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
	HYPERDAGS_BACKEND_INSTALL_DIR INCLUDE_INSTALL_DIR
	HYPERDAGS_BACKEND_DEFAULT_NAME HYPERDAGS_SELECTION_DEFS
	WITH_HYPERDAGS_USING
)

assert_defined_targets( backend_flags )

set( backend_hyperdags_srcs
	${CMAKE_CURRENT_SOURCE_DIR}/init.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/hyperdags.cpp
)

if( WITH_HYPERDAGS_BACKEND )

	# static
	add_library( backend_hyperdags_static STATIC ${backend_hyperdags_srcs} )
	set_target_properties( backend_hyperdags_static PROPERTIES
		OUTPUT_NAME "${BACKEND_LIBRARY_OUTPUT_NAME}"
		ARCHIEVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/hyperdags"
	)
	target_link_libraries( backend_hyperdags_static PRIVATE backend_flags )
	target_link_libraries( backend_hyperdags_static PUBLIC backend_${WITH_HYPERDAGS_USING}_headers )
	target_link_libraries( backend_hyperdags_static PUBLIC backend_hyperdags_headers )
	target_link_libraries( backend_hyperdags_static PRIVATE backend_${WITH_HYPERDAGS_USING}_static )
	target_compile_definitions( backend_hyperdags_static PUBLIC "${HYPERDAGS_SELECTION_DEFS}" )
	install( TARGETS backend_hyperdags_static
		EXPORT GraphBLASTargets
		ARCHIVE DESTINATION "${HYPERDAGS_BACKEND_INSTALL_DIR}"
	)

	# shared
	add_library( backend_hyperdags_shared SHARED ${backend_hyperdags_srcs} )
	set_target_properties( backend_hyperdags_shared PROPERTIES
		OUTPUT_NAME "${BACKEND_LIBRARY_OUTPUT_NAME}"
		SOVERSION "${VERSION}"
		LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/hyperdags"
	)
	target_link_libraries( backend_hyperdags_shared PRIVATE backend_flags )
	target_link_libraries( backend_hyperdags_shared PRIVATE backend_${WITH_HYPERDAGS_USING}_headers )
	target_link_libraries( backend_hyperdags_shared INTERFACE backend_hyperdags_headers )
	target_link_libraries( backend_hyperdags_shared PRIVATE backend_${WITH_HYPERDAGS_USING}_shared )
	target_compile_definitions( backend_hyperdags_shared PUBLIC "${HYPERDAGS_SELECTION_DEFS}" )
	install( TARGETS backend_hyperdags_shared
		EXPORT GraphBLASTargets
		LIBRARY DESTINATION "${HYPERDAGS_BACKEND_INSTALL_DIR}"
	)

	# propagate targets and use static as default linkage
	add_dependencies( libs backend_hyperdags_static backend_hyperdags_shared )
	add_library( "${HYPERDAGS_BACKEND_DEFAULT_NAME}" ALIAS backend_hyperdags_static )

endif()

