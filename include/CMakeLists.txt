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
# Definition of GraphBLAS include targets: all targets here defined
# are interface targets for headers and basic definitions required to build
# GraphBLAS backends and tests. Importing targets have all basic dependencies
# and definitions to compile against each backend, but MUST explicitly
# set a default backend (if they want to do so).
#
assert_valid_variables( INCLUDE_INSTALL_DIR NO_NUMA_DEF )

# basic graphblas includes all backends depend on
add_library( backend_headers_nodefs INTERFACE )
target_include_directories( backend_headers_nodefs INTERFACE
	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
	$<INSTALL_INTERFACE:.>
)

# regex to filter all headers files based on the extension and
# exclude other files possibly present in the hierarchy
# (e.g., .gitignore, *.md or possible compilation garbage)
set( HEADERS_REGEX ".+\.(hpp|h|hxx|hh|h\\+\\+)$" )

# ALP utils headers, which are independent from ALP/GraphBLAS itself;
# note that the include root is the same as of ALP/GraphBLAS for compatibility
# with existing code (they contain "#include <graphblas/utils/<header>.hpp>"),
# but this may change in future
add_library( alp_utils_headers INTERFACE )
target_include_directories( alp_utils_headers INTERFACE
	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
	$<INSTALL_INTERFACE:.>
)

# everybody needs the math library
target_link_libraries( backend_headers_nodefs INTERFACE LibM::LibM )

if( WITH_NUMA )
	# some headers need NUMA
	target_link_libraries( backend_headers_nodefs INTERFACE Numa::Numa )
	target_link_libraries( alp_utils_headers INTERFACE Numa::Numa )
else()
	# if NUMA is not needed, everybody needs to compile with NO_NUMA_DEF
	# to explicitly exclude it
	target_compile_definitions( backend_headers_nodefs INTERFACE "${NO_NUMA_DEF}" )
	target_compile_definitions( alp_utils_headers INTERFACE "${NO_NUMA_DEF}" )
endif()

install( TARGETS backend_headers_nodefs EXPORT GraphBLASTargets
	INCLUDES DESTINATION "${INCLUDE_INSTALL_DIR}"
)

install( TARGETS alp_utils_headers EXPORT GraphBLASTargets
	INCLUDES DESTINATION "${INCLUDE_INSTALL_DIR}"
)

if( WITH_ALP_REFERENCE_BACKEND_HEADERS )
	add_subdirectory( alp )
endif()

if( WITH_REFERENCE_BACKEND_HEADERS )
	add_subdirectory( graphblas )
endif()
