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

assert_valid_variables( BACKEND_LIBRARY_OUTPUT_NAME VERSION
	SHMEM_BACKEND_INSTALL_DIR INCLUDE_INSTALL_DIR
	DENSE_BACKEND_DEFAULT_NAME
	DENSE_SELECTION_DEFS backend_reference_srcs
)

# sources for bsp1d backend
set( backend_reference_dense_srcs
	${CMAKE_CURRENT_SOURCE_DIR}/init.cpp
)

add_library( backend_reference_dense_static STATIC "${backend_reference_dense_srcs}" )

target_link_libraries( backend_reference_dense_static PUBLIC backend_reference_dense_headers )
target_link_libraries( backend_reference_dense_static PRIVATE backend_common_utils )

set_target_properties( backend_reference_dense_static PROPERTIES
	POSITION_INDEPENDENT_CODE true
	INTERFACE_POSITION_INDEPENDENT_CODE true
	OUTPUT_NAME "${BACKEND_LIBRARY_OUTPUT_NAME}"
)

set_target_properties( backend_reference_dense_static PROPERTIES
	ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/reference_dense"
)

target_compile_options( backend_reference_dense_static PRIVATE "${BACKEND_COMPILE_OPTIONS}" )
target_compile_definitions( backend_reference_dense_static PRIVATE "${BACKEND_COMPILE_DEFINITIONS}" )
target_compile_definitions( backend_reference_dense_static INTERFACE "${REFERENCE_DENSE_SELECTION_DEFS}" )

add_dependencies( libs backend_reference_dense_static )

install( TARGETS backend_reference_dense_static
	EXPORT GraphBLASTargets
	ARCHIVE DESTINATION "${REFERENCE_DENSE_BACKEND_INSTALL_DIR}"
)

# this is an alias for add_grb_executables() to select the backend to link against
# DO NOT CHANGE THE ALIAS NAME!
add_library( "${DENSE_BACKEND_DEFAULT_NAME}" ALIAS backend_reference_dense_static )
