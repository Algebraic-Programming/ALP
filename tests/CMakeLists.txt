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
# Main tests, which do not use the Launcher mechanism
#
assert_valid_variables( ALP_UTILS_SRC_PATH AVAILABLE_TEST_BACKENDS DATASETS_DIR TESTS_EXE_OUTPUT_DIR )
assert_defined_variables( GNN_DATASET_PATH WITH_BSP1D_BACKEND WITH_HYBRID_BACKEND )


### CONSUMABLE TARGETS
### i.e. targets with the default backend already set in the compilation interface:
### if you import one of them, than it is already selected as a backend

if( WITH_REFERENCE_BACKEND )

	assert_defined_targets( backend_shmem_static )

	## reference static
	add_library( backend_reference_static INTERFACE )
	target_link_libraries( backend_reference_static INTERFACE backend_shmem_static )
	target_link_libraries( backend_reference_static INTERFACE backend_reference_headers )
	# add the suitable definitions for backend selection for users' targets to get them automatically
	target_compile_definitions( backend_reference_static INTERFACE "${REFERENCE_SELECTION_DEFS}" )

	install( TARGETS backend_reference_static
		EXPORT GraphBLASTargets
		ARCHIVE DESTINATION "${SHMEM_BACKEND_INSTALL_DIR}"
	)

	## reference shared
	add_library( backend_reference_shared INTERFACE )
	target_link_libraries( backend_reference_shared INTERFACE backend_shmem_shared )
	target_link_libraries( backend_reference_shared INTERFACE backend_reference_headers )
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

	assert_defined_targets( backend_shmem_static )

	## reference_omp static
	add_library( backend_reference_omp_static INTERFACE )
	target_link_libraries( backend_reference_omp_static INTERFACE backend_shmem_static )
	target_link_libraries( backend_reference_omp_static INTERFACE backend_reference_omp_headers )
	target_compile_definitions( backend_reference_omp_static INTERFACE "${REFERENCE_OMP_SELECTION_DEFS}" )

	install( TARGETS backend_reference_omp_static
		EXPORT GraphBLASTargets
		ARCHIVE DESTINATION "${SHMEM_BACKEND_INSTALL_DIR}"
	)

	## reference_omp shared
	add_library( backend_reference_omp_shared INTERFACE )
	target_link_libraries( backend_reference_omp_shared INTERFACE backend_shmem_shared )
	target_link_libraries( backend_reference_omp_shared INTERFACE backend_reference_omp_headers )
	target_compile_definitions( backend_reference_omp_shared INTERFACE "${REFERENCE_OMP_SELECTION_DEFS}" )

	install( TARGETS backend_reference_omp_shared
		EXPORT GraphBLASTargets
		LIBRARY DESTINATION "${SHMEM_BACKEND_INSTALL_DIR}"
	)

	# this is an alias for add_grb_executables() to select the backend to link against
	# DO NOT CHANGE THE ALIAS NAME!
	add_library( "${REFERENCE_OMP_BACKEND_DEFAULT_NAME}" ALIAS backend_reference_omp_static )

endif( WITH_OMP_BACKEND )


if( WITH_NONBLOCKING_BACKEND )

	assert_defined_targets( backend_shmem_static )

	## nonblocking static
	add_library( backend_nonblocking_static INTERFACE )
	target_link_libraries( backend_nonblocking_static INTERFACE backend_shmem_static )
	target_link_libraries( backend_nonblocking_static INTERFACE backend_nonblocking_headers )
	target_compile_definitions( backend_nonblocking_static INTERFACE "${NONBLOCKING_SELECTION_DEFS}" )

	install( TARGETS backend_nonblocking_static
		EXPORT GraphBLASTargets
		ARCHIVE DESTINATION "${SHMEM_BACKEND_INSTALL_DIR}"
	)

	## nonblocking shared
	add_library( backend_nonblocking_shared INTERFACE )
	target_link_libraries( backend_nonblocking_shared INTERFACE backend_shmem_shared )
	target_link_libraries( backend_nonblocking_shared INTERFACE backend_nonblocking_headers )
	target_compile_definitions( backend_nonblocking_shared INTERFACE "${NONBLOCKING_SELECTION_DEFS}" )

	install( TARGETS backend_nonblocking_shared
		EXPORT GraphBLASTargets
		LIBRARY DESTINATION "${SHMEM_BACKEND_INSTALL_DIR}"
	)

	# this is an alias for add_grb_executables() to select the backend to link against
	# DO NOT CHANGE THE ALIAS NAME!
	add_library( "${NONBLOCKING_BACKEND_DEFAULT_NAME}" ALIAS backend_nonblocking_static )

endif( WITH_NONBLOCKING_BACKEND )


# library with utilities for tests, to be used optionally
# i.e. NOT linked by default
add_library( test_utils_headers INTERFACE )

# source files currently use "utils/*.hpp" as a path for test_utils headers
# so we must use the current directory for inclusion
target_include_directories( test_utils_headers INTERFACE ${CMAKE_CURRENT_SOURCE_DIR} )
target_link_libraries( test_utils_headers INTERFACE backend_headers_nodefs )


add_library( test_utils STATIC
	utils/argument_parser.cpp
	utils/assertion_engine.cpp
)
target_link_libraries( test_utils PUBLIC test_utils_headers )


# SCRIPTS_COMMON_ARGS is used by underlying test categories
set( SCRIPTS_COMMON_ARGS
	"--backends" "\"${AVAILABLE_TEST_BACKENDS}\""
	"--input-dir" "\"${DATASETS_DIR}\""
)

if( WITH_BSP1D_BACKEND OR WITH_HYBRID_BACKEND )
	assert_valid_variables( LPFRUN LPF_ENGINE MANUALRUN_ARGS )
	list( APPEND SCRIPTS_COMMON_ARGS
		"--lpfexe" "\"${LPFRUN}\""
		"--lpf-engine" "\"${LPF_ENGINE}\""
		"--manual-run-args" "\"${MANUALRUN_ARGS}\""
	)
endif()

if( GNN_DATASET_PATH )
	list( APPEND SCRIPTS_COMMON_ARGS
		"--gnn-dataset-path" "\"${GNN_DATASET_PATH}\""
	)
endif()

add_subdirectory( unit )

add_subdirectory( smoke )

add_subdirectory( performance )


### TARGETS TO LIST TESTS

# list categories
add_custom_target( "list_tests_categories"
	COMMAND echo "${TEST_CATEGORIES}"
	COMMAND_EXPAND_LISTS
)

# list available tests per backend
foreach( b ${AVAILABLE_TEST_BACKENDS} )
	get_property( bt GLOBAL PROPERTY tests_backend_${b} )
	add_custom_target( "list_tests_backend_${b}"
		COMMAND echo "${bt}"
		COMMAND_EXPAND_LISTS
	)
	add_custom_target( "build_tests_backend_${b}"
		DEPENDS "${bt}"
	)
endforeach()

# a test may not belong to a backend (because of add_grb_executable_custom)
# but it must belong to a category: hence, get all tests from there
foreach( cat ${TEST_CATEGORIES} )
	get_property( ct GLOBAL PROPERTY tests_category_${cat} )
	list( APPEND all_tests "${ct}" )
endforeach()

### TARGETS INVOLVING ALL TESTS
add_custom_target( "list_tests_all"
	COMMAND echo "${all_tests}"
	COMMAND_EXPAND_LISTS
	USES_TERMINAL
)

add_custom_target( build_tests_all
	DEPENDS "${all_tests}"
)

# run all test categories this way, by calling the various targets separately
# this guarantees that they are executed serially
# do NOT use DEPENDS for their execution, since it parallelizes
add_custom_target( tests_all
	COMMAND ${CMAKE_MAKE_PROGRAM} tests_unit
	COMMAND ${CMAKE_MAKE_PROGRAM} tests_smoke
	COMMAND ${CMAKE_MAKE_PROGRAM} tests_performance

	# do not change, or make cannot find the targets!
	WORKING_DIRECTORY "${PROJECT_BINARY_DIR}"

	# give scripts direct terminal access, to correctly render colors
	USES_TERMINAL

	# since previous make invocations are serial (and no -j flag can be given),
	# build tests first with user-given parallelism
	DEPENDS build_tests_all
)
# custom target "tests" just to keep old naming
add_custom_target( tests DEPENDS tests_all )

