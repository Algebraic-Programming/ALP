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
assert_defined_variables( REFERENCE_INCLUDE_DEFS REFERENCE_OMP_INCLUDE_DEFS NONBLOCKING_INCLUDE_DEFS LPF_INCLUDE_DEFS
	WITH_REFERENCE_BACKEND_HEADERS WITH_OMP_BACKEND_HEADERS WITH_NONBLOCKING_BACKEND WITH_BSP1D_BACKEND WITH_HYBRID_BACKEND
	HYPERDAGS_INCLUDE_DEFS WITH_HYPERDAGS_BACKEND_HEADERS WITH_HYPERDAGS_BACKEND
)
assert_valid_variables( INCLUDE_INSTALL_DIR NO_NUMA_DEF )

# basic graphblas includes all backends depend on
add_library( backend_headers_nodefs INTERFACE )
target_include_directories( backend_headers_nodefs INTERFACE
	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)

# regex to filter all headers files based on the extension and
# exclude other files possibly present in the hierarchy
# (e.g., .gitignore, *.md or possible compilation garbage)
set( HEADERS_REGEX ".+\.(hpp|h|hxx|hh|h\\+\\+)$" )

# to avoid flaky acrobatics with regex or glob expressions, copy main files directly
install( FILES "graphblas.hpp" DESTINATION "${INCLUDE_INSTALL_DIR}" )
set( root_files
	"graphblas/backends.hpp" "graphblas/benchmark.hpp"
	"graphblas/blas0.hpp" "graphblas/blas1.hpp" "graphblas/blas2.hpp"
	"graphblas/blas3.hpp" "graphblas/collectives.hpp" "graphblas/config.hpp"
	"graphblas/coordinates.hpp" "graphblas/descriptors.hpp"
	"graphblas/distribution.hpp" "graphblas/exec.hpp" "graphblas/final.hpp"
	"graphblas/identities.hpp" "graphblas/init.hpp" "graphblas/internalops.hpp"
	"graphblas/io.hpp" "graphblas/iomode.hpp" "graphblas/matrix.hpp"
	"graphblas/monoid.hpp" "graphblas/ops.hpp" "graphblas/phase.hpp"
	"graphblas/pinnedvector.hpp" "graphblas/properties.hpp" "graphblas/rc.hpp"
	"graphblas/semiring.hpp" "graphblas/spmd.hpp" "graphblas/tags.hpp"
	"graphblas/type_traits.hpp" "graphblas/utils.hpp" "graphblas/vector.hpp"
	"graphblas/synchronizedNonzeroIterator.hpp" "graphblas/nonzeroStorage.hpp"
	"graphblas/selection_ops.hpp"
)

set( GRB_INCLUDE_INSTALL_DIR "${INCLUDE_INSTALL_DIR}/graphblas")
install( FILES ${root_files} DESTINATION "${GRB_INCLUDE_INSTALL_DIR}" )

# copy base headers and all its subdirectories (if any)
# note: leave the slash "/" at the end of the DIRECTORY path,
# othwerise CMake will create a "graphblas/base" directory inside DESTINATION !!!
# https://cmake.org/cmake/help/latest/command/install.html#installing-directories
install( DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/graphblas/base/"
	DESTINATION "${GRB_INCLUDE_INSTALL_DIR}/base"
	FILES_MATCHING REGEX "${HEADERS_REGEX}"
)

# ALP utils headers, which are independent from ALP/GraphBLAS itself;
# note that the include root is the same as of ALP/GraphBLAS for compatibility
# with existing code (they contain "#include <graphblas/utils/<header>.hpp>"),
# but this may change in future
add_library( alp_utils_headers INTERFACE )
target_include_directories( alp_utils_headers INTERFACE
	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)

# copy utils headers
install( DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/graphblas/utils/"
	DESTINATION "${GRB_INCLUDE_INSTALL_DIR}/utils"
	FILES_MATCHING REGEX "${HEADERS_REGEX}"
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

# the utils library currently depends heavily on other ALP functionalities,
# but could be easily split out in future; in view of this, export an
# independent include path (incidentally the same as ALP)
install( TARGETS alp_utils_headers EXPORT GraphBLASTargets
	INCLUDES DESTINATION "${INCLUDE_INSTALL_DIR}"
)

if( WITH_REFERENCE_BACKEND_HEADERS )
	add_library( backend_reference_headers INTERFACE )
	target_link_libraries( backend_reference_headers INTERFACE backend_headers_nodefs )
	target_compile_definitions( backend_reference_headers INTERFACE "${REFERENCE_INCLUDE_DEFS}" )
	# currently, the OMP header files and definitions are required also for the reference backend
	# TODO: "dis-entangle" code from OMP backend dependence and remove this
	target_compile_definitions( backend_reference_headers INTERFACE "${REFERENCE_OMP_INCLUDE_DEFS}" )
	target_link_libraries( backend_reference_headers INTERFACE OpenMP::OpenMP_CXX )

	install( TARGETS backend_reference_headers EXPORT GraphBLASTargets )
endif()

if( WITH_DENSE_BACKEND )
	add_library( backend_reference_dense_headers INTERFACE )
	target_link_libraries( backend_reference_dense_headers INTERFACE backend_headers_nodefs )
	install( DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/graphblas/denseref"
		DESTINATION "${GRB_INCLUDE_INSTALL_DIR}"
		FILES_MATCHING REGEX "${HEADERS_REGEX}"
	)
	install( TARGETS backend_reference_dense_headers EXPORT GraphBLASTargets )
	target_link_libraries( all_headers INTERFACE backend_reference_dense_headers )
endif()

if( WITH_REFERENCE_BACKEND OR WITH_OMP_BACKEND )
	install( DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/graphblas/reference/"
		DESTINATION "${GRB_INCLUDE_INSTALL_DIR}/reference"
		FILES_MATCHING REGEX "${HEADERS_REGEX}"
	)
endif()

if( WITH_OMP_BACKEND_HEADERS )
	add_library( backend_reference_omp_headers INTERFACE )
	target_link_libraries( backend_reference_omp_headers INTERFACE backend_headers_nodefs )
	target_link_libraries( backend_reference_omp_headers INTERFACE OpenMP::OpenMP_CXX )
	target_compile_definitions( backend_reference_omp_headers INTERFACE
		"${REFERENCE_INCLUDE_DEFS}" "${REFERENCE_OMP_INCLUDE_DEFS}"
	)

	install( DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/graphblas/omp/"
		DESTINATION "${GRB_INCLUDE_INSTALL_DIR}/omp"
		FILES_MATCHING REGEX "${HEADERS_REGEX}"
	)
	install( TARGETS backend_reference_omp_headers EXPORT GraphBLASTargets )
endif()

if( WITH_HYPERDAGS_BACKEND )
	add_library( backend_hyperdags_headers INTERFACE )
	target_link_libraries( backend_hyperdags_headers INTERFACE "backend_${WITH_HYPERDAGS_USING}_headers" )
	target_compile_definitions( backend_hyperdags_headers INTERFACE "${HYPERDAGS_INCLUDE_DEFS}" )
	install( TARGETS backend_hyperdags_headers EXPORT GraphBLASTargets )

	install( DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/graphblas/hyperdags/"
		DESTINATION "${GRB_INCLUDE_INSTALL_DIR}/hyperdags"
		FILES_MATCHING REGEX "${HEADERS_REGEX}"
	)
endif()

if( WITH_NONBLOCKING_BACKEND )
	add_library( backend_nonblocking_headers INTERFACE )
	# the nonblocking backend depends on the reference backend
	target_link_libraries( backend_nonblocking_headers INTERFACE backend_reference_headers )
	target_link_libraries( backend_nonblocking_headers INTERFACE OpenMP::OpenMP_CXX )
	target_compile_definitions( backend_nonblocking_headers INTERFACE
		"${NONBLOCKING_INCLUDE_DEFS}"
	)

	install( DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/graphblas/nonblocking/"
		DESTINATION "${GRB_INCLUDE_INSTALL_DIR}/nonblocking"
		FILES_MATCHING REGEX "${HEADERS_REGEX}"
	)
	install( TARGETS backend_nonblocking_headers EXPORT GraphBLASTargets )
endif()

if( WITH_BSP1D_BACKEND OR WITH_HYBRID_BACKEND )
	# copy headers, which are common to both distributed backends
	install( DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/graphblas/bsp/"
		DESTINATION "${GRB_INCLUDE_INSTALL_DIR}/bsp"
		FILES_MATCHING REGEX "${HEADERS_REGEX}"
	)
	install( DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/graphblas/bsp1d/"
		DESTINATION "${GRB_INCLUDE_INSTALL_DIR}/bsp1d"
		FILES_MATCHING REGEX "${HEADERS_REGEX}"
	)
endif()

if( WITH_BSP1D_BACKEND )
	add_library( backend_bsp1d_headers INTERFACE )
	target_link_libraries( backend_bsp1d_headers INTERFACE Numa::Numa backend_reference_headers )
	target_compile_definitions( backend_bsp1d_headers INTERFACE "${LPF_INCLUDE_DEFS}" )

	install( TARGETS backend_bsp1d_headers EXPORT GraphBLASTargets )
endif()

if( WITH_HYBRID_BACKEND )
	add_library( backend_hybrid_headers INTERFACE )
	# the hybrid backend needs OMP by design
	target_link_libraries( backend_hybrid_headers INTERFACE Numa::Numa backend_reference_omp_headers )
	target_compile_definitions( backend_hybrid_headers INTERFACE "${LPF_INCLUDE_DEFS}" )

	install( TARGETS backend_hybrid_headers EXPORT GraphBLASTargets )
endif()

# this target lists the algorithms implemented on top of the generic functionalities,
# hence it depends only on backend_headers_nodefs
add_library( algorithms INTERFACE )
target_link_libraries( algorithms INTERFACE backend_headers_nodefs )

target_include_directories(
	algorithms INTERFACE
	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/graphblas/algorithms>
)

install( DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/graphblas/algorithms/"
	DESTINATION "${GRB_INCLUDE_INSTALL_DIR}/algorithms"
	FILES_MATCHING REGEX "${HEADERS_REGEX}"
)

install( DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/graphblas/interfaces/"
	DESTINATION "${GRB_INCLUDE_INSTALL_DIR}/interfaces"
	FILES_MATCHING REGEX "${HEADERS_REGEX}"
)

install( TARGETS algorithms EXPORT GraphBLASTargets )


# generate the spblas header with the library prefix
configure_file( ${CMAKE_CURRENT_SOURCE_DIR}/transition/spblas.h.in
	${CMAKE_CURRENT_BINARY_DIR}/transition/spblas.h @ONLY
)

# this target lists the transition path headers
# these are plain C headers and do not have any dependences
add_library( transition_headers INTERFACE )

target_include_directories( transition_headers INTERFACE
	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/transition/>
	$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/transition/>
)

set( TRANSITION_INCLUDE_INSTALL_DIR "${INCLUDE_INSTALL_DIR}/transition" )

install( FILES ${CMAKE_CURRENT_BINARY_DIR}/transition/spblas.h
	DESTINATION "${TRANSITION_INCLUDE_INSTALL_DIR}"
)

install( DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/transition/"
	DESTINATION "${TRANSITION_INCLUDE_INSTALL_DIR}"
	FILES_MATCHING REGEX "${HEADERS_REGEX}"
)

install( TARGETS transition_headers EXPORT GraphBLASTargets
	INCLUDES DESTINATION "${TRANSITION_INCLUDE_INSTALL_DIR}"
)

