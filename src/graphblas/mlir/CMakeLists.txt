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
	MLIR_BACKEND_INSTALL_DIR INCLUDE_INSTALL_DIR
	MLIR_BACKEND_DEFAULT_NAME 
	MLIR_SELECTION_DEFS
)

set( backend_mlir_srcs
    ${common_sources} 
    ${CMAKE_CURRENT_SOURCE_DIR}/init.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/jitCtx.cpp
    CACHE INTERNAL "MLIR source files"
)

add_library( backend_mlir_static STATIC "${backend_mlir_srcs}" )

target_link_libraries( backend_mlir_static PUBLIC backend_mlir_headers )
#target_link_libraries( backend_mlir_static PRIVATE backend_common_utils )
target_link_libraries( backend_mlir_static PRIVATE ${LLVM_LIBS} )

set_target_properties( backend_mlir_static PROPERTIES
	POSITION_INDEPENDENT_CODE true
	INTERFACE_POSITION_INDEPENDENT_CODE true
	OUTPUT_NAME "${BACKEND_LIBRARY_OUTPUT_NAME}"
)

set_target_properties( backend_mlir_static PROPERTIES
	ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/mlir"
)

target_compile_definitions( backend_mlir_static INTERFACE "${MLIR_SELECTION_DEFS}" )

add_dependencies( libs backend_mlir_static )

install( TARGETS backend_mlir_static
	EXPORT GraphBLASTargets
	ARCHIVE DESTINATION "${MLIR_BACKEND_INSTALL_DIR}"
)

# this is an alias for add_grb_executables() to select the backend to link against
# DO NOT CHANGE THE ALIAS NAME!
add_library( "${MLIR_BACKEND_DEFAULT_NAME}" ALIAS backend_mlir_static )
target_link_libraries(backend_mlir_static PRIVATE backend_flags)
