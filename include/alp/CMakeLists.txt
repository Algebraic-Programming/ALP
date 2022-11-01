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
assert_defined_variables( ALP_REFERENCE_INCLUDE_DEFS WITH_ALP_REFERENCE_BACKEND_HEADERS )
assert_valid_variables( INCLUDE_INSTALL_DIR )

# to avoid flaky acrobatics with regex or glob expressions, copy main files directly
install( FILES "../alp.hpp" DESTINATION "${INCLUDE_INSTALL_DIR}" )
set( root_files
	"../alp.hpp" "backends.hpp" "benchmark.hpp"
	"blas0.hpp" "blas1.hpp" "blas2.hpp"
	"blas3.hpp" "collectives.hpp" "config.hpp"
	"descriptors.hpp" "distribution.hpp"
	"exec.hpp" "identities.hpp" "init.hpp"
	"internalops.hpp" "io.hpp" "iomode.hpp"
	"matrix.hpp" "monoid.hpp" "ops.hpp"
	"phase.hpp" "pinnedvector.hpp" "properties.hpp"
	"rc.hpp" "semiring.hpp" "spmd.hpp"
	"tags.hpp" "type_traits.hpp" "utils.hpp"
	"vector.hpp"
)
set( ALP_INCLUDE_INSTALL_DIR "${INCLUDE_INSTALL_DIR}/alp")
install( FILES ${root_files} DESTINATION "${ALP_INCLUDE_INSTALL_DIR}" )

# copy base headers and all its subdirectories (if any)
# note: leave the slash "/" at the end of the DIRECTORY path,
# othwerise CMake will create a "graphblas/base" directory inside DESTINATION !!!
# https://cmake.org/cmake/help/latest/command/install.html#installing-directories
install( DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/base/"
	DESTINATION "${ALP_INCLUDE_INSTALL_DIR}/base"
	FILES_MATCHING REGEX "${HEADERS_REGEX}"
)

# copy utils headers
install( DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/utils/"
	DESTINATION "${ALP_INCLUDE_INSTALL_DIR}/utils"
	FILES_MATCHING REGEX "${HEADERS_REGEX}"
)

if( WITH_ALP_REFERENCE_BACKEND_HEADERS )
	add_library( backend_alp_reference_headers INTERFACE )
	target_link_libraries( backend_alp_reference_headers INTERFACE backend_headers_nodefs )
	target_compile_definitions( backend_alp_reference_headers INTERFACE "${ALP_REFERENCE_INCLUDE_DEFS}" )

	install( TARGETS backend_alp_reference_headers EXPORT GraphBLASTargets )
endif()

if( WITH_ALP_REFERENCE_BACKEND )
	install( DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/alp/reference"
		DESTINATION "${ALP_INCLUDE_INSTALL_DIR}/alp_reference"
		FILES_MATCHING REGEX "${HEADERS_REGEX}"
	)
endif()

# this target lists the algorithms implemented on top of the generic functionalities,
# hence it depends only on backend_headers_nodefs
add_library( alp_algorithms INTERFACE )
target_link_libraries( alp_algorithms INTERFACE backend_headers_nodefs )

target_include_directories(
	alp_algorithms INTERFACE

	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/alp/algorithms>
	$<INSTALL_INTERFACE:alp/algorithms>
)

install( DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/algorithms/"
	DESTINATION "${ALP_INCLUDE_INSTALL_DIR}/algorithms"
	FILES_MATCHING REGEX "${HEADERS_REGEX}"
)

install( TARGETS alp_algorithms EXPORT GraphBLASTargets )
