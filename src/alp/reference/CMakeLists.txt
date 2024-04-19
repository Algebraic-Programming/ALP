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
	ALP_REFERENCE_BACKEND_INSTALL_DIR INCLUDE_INSTALL_DIR
	ALP_REFERENCE_BACKEND_DEFAULT_NAME
	ALP_REFERENCE_SELECTION_DEFS 
)

# sources for ALP reference backend
set( backend_alp_reference_srcs
	${CMAKE_CURRENT_SOURCE_DIR}/../descriptors.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/../rc.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/init.cpp
)

add_library( backend_alp_reference_static STATIC "${backend_alp_reference_srcs}" )

target_link_libraries( backend_alp_reference_static PUBLIC backend_alp_reference_headers )

set_target_properties( backend_alp_reference_static PROPERTIES
	OUTPUT_NAME "${BACKEND_LIBRARY_OUTPUT_NAME}"
)

set_target_properties( backend_alp_reference_static PROPERTIES
	ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/reference"
)

target_compile_definitions( backend_alp_reference_static PUBLIC "${ALP_REFERENCE_SELECTION_DEFS}" )
target_link_libraries( backend_alp_reference_static PRIVATE backend_flags )

add_dependencies( libs backend_alp_reference_static )

install( TARGETS backend_alp_reference_static
	EXPORT GraphBLASTargets
	ARCHIVE DESTINATION "${ALP_REFERENCE_BACKEND_INSTALL_DIR}"
	)

# this is an alias for add_grb_executables() to select the backend to link against
# DO NOT CHANGE THE ALIAS NAME!
add_library( "${ALP_REFERENCE_BACKEND_DEFAULT_NAME}" ALIAS backend_alp_reference_static )
