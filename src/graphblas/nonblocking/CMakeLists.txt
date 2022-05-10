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
# Creation of the nonblocking backend, both as static and dynamic library. Any target
# importing a backend also imports the compiler definition(s) required to set it as
# default. If a target want to do it manually, the target 'backend_shmem_static' exists
# with no default backend selection in its compilation interface.
#

assert_valid_variables( SHMEM_BACKEND_INSTALL_DIR
	NONBLOCKING_BACKEND_DEFAULT_NAME NONBLOCKING_SELECTION_DEFS
)

assert_defined_targets( backend_flags )


### CONSUMABLE TARGETS
### i.e. targets with the default backend already set in the compilation interface:
### if you import one of them, than it is already selected as a backend

if( WITH_NONBLOCKING_BACKEND )

	## nonblocking static
	if( ${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.19.0" )
		add_library( backend_nonblocking_static INTERFACE )
		target_link_libraries( backend_nonblocking_static INTERFACE backend_shmem_static )
		target_link_libraries( backend_nonblocking_static INTERFACE backend_nonblocking_headers )
	else()
		make_reference_target( backend_nonblocking_static STATIC "nonblocking" )
		target_link_libraries( backend_nonblocking_static PUBLIC backend_nonblocking_headers )
		add_dependencies( libs backend_nonblocking_static )
	endif()
	target_compile_definitions( backend_nonblocking_static INTERFACE "${NONBLOCKING_SELECTION_DEFS}" )

	install( TARGETS backend_nonblocking_static
		EXPORT GraphBLASTargets
		ARCHIVE DESTINATION "${SHMEM_BACKEND_INSTALL_DIR}"
	)

	## nonblocking shared
	if( ${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.19.0" )
		add_library( backend_nonblocking_shared INTERFACE )
		target_link_libraries( backend_nonblocking_shared INTERFACE backend_shmem_shared )
		target_link_libraries( backend_nonblocking_shared INTERFACE backend_nonblocking_headers )
	else()
		make_reference_target( backend_nonblocking_shared SHARED "nonblocking" )
		target_link_libraries( backend_nonblocking_shared PUBLIC backend_nonblocking_headers )
		add_dependencies( libs backend_nonblocking_shared )
	endif()
	target_compile_definitions( backend_nonblocking_shared INTERFACE "${NONBLOCKING_SELECTION_DEFS}" )

	install( TARGETS backend_nonblocking_shared
		EXPORT GraphBLASTargets
		LIBRARY DESTINATION "${SHMEM_BACKEND_INSTALL_DIR}"
	)

	# this is an alias for add_grb_executables() to select the backend to link against
	# DO NOT CHANGE THE ALIAS NAME!
	add_library( "${NONBLOCKING_BACKEND_DEFAULT_NAME}" ALIAS backend_nonblocking_static )

endif( WITH_NONBLOCKING_BACKEND )

