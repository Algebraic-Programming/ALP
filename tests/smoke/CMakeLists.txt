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


# write here the name of the category
# add_grb_executables and add_grb_executable_custom need this information
set( TEST_CATEGORY "smoke" )


add_grb_executables( manual_hook_hw manual_launcher.cpp
	hook/hello_world.cpp
	BACKENDS bsp1d NO_BACKEND_NAME
)

add_grb_executables( manual_hook_grb_set manual_launcher.cpp
	hook/grb_launcher.cpp hook/setvector.cpp
	BACKENDS bsp1d NO_BACKEND_NAME
)

add_grb_executables( manual_hook_grb_reduce manual_launcher.cpp
	hook/grb_launcher.cpp ../unit/launcher/reduce.cpp
	BACKENDS bsp1d NO_BACKEND_NAME
)

add_grb_executables( manual_hook_grb_dot manual_launcher.cpp
	hook/grb_launcher.cpp hook/dot.cpp
	BACKENDS bsp1d NO_BACKEND_NAME
)

add_grb_executables( manual_hook_grb_collectives_blas0 manual_launcher.cpp
	hook/grb_launcher.cpp ../unit/launcher/collectives_blas0.cpp
	BACKENDS bsp1d NO_BACKEND_NAME
)

add_grb_executables( manual_hook_grb_collectives_blas1 manual_launcher.cpp
	hook/grb_launcher.cpp hook/collectives_blas1.cpp
	BACKENDS bsp1d NO_BACKEND_NAME
)

add_grb_executables( manual_hook_grb_collectives_blas1_raw manual_launcher.cpp
	hook/grb_launcher.cpp hook/collectives_blas1_raw.cpp
	BACKENDS bsp1d NO_BACKEND_NAME
)

add_grb_executables( small_knn ../unit/auto_launcher.cpp
	hook/knn.cpp
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
)

add_grb_executables( manual_hook_small_knn manual_launcher.cpp
	hook/grb_launcher.cpp hook/knn.cpp
	BACKENDS bsp1d NO_BACKEND_NAME
)

add_grb_executables( from_mpi_launch_simple_pagerank simple_pagerank_from_mpi.cpp
	BACKENDS bsp1d NO_BACKEND_NAME
	ADDITIONAL_LINK_LIBRARIES MPI::MPI_CXX
)

add_grb_executables( from_mpi_launch_simple_pagerank_multiple_entry simple_pagerank_from_mpi.cpp
	BACKENDS bsp1d NO_BACKEND_NAME
	ADDITIONAL_LINK_LIBRARIES MPI::MPI_CXX
	COMPILE_DEFINITIONS MULTIPLE_ENTRY
)

add_grb_executables( from_mpi_launch_simple_pagerank_broadcast_pinning_multiple_entry simple_pagerank_broadcast.cpp
	BACKENDS bsp1d NO_BACKEND_NAME
	ADDITIONAL_LINK_LIBRARIES MPI::MPI_CXX
	COMPILE_DEFINITIONS MULTIPLE_ENTRY PINNED_OUTPUT
)

add_grb_executables( from_mpi_launch_simple_pagerank_broadcast_multiple_entry simple_pagerank_broadcast.cpp
	BACKENDS bsp1d NO_BACKEND_NAME
	ADDITIONAL_LINK_LIBRARIES MPI::MPI_CXX
	COMPILE_DEFINITIONS MULTIPLE_ENTRY
)

add_grb_executables( from_mpi_launch_simple_pagerank_broadcast simple_pagerank_broadcast.cpp
	BACKENDS bsp1d NO_BACKEND_NAME
	ADDITIONAL_LINK_LIBRARIES MPI::MPI_CXX
)

add_grb_executables( knn knn.cpp ../unit/parser.cpp
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
)

add_grb_executables( hpcg hpcg.cpp
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
	ADDITIONAL_LINK_LIBRARIES test_utils
)

add_grb_executables( graphchallenge_nn_single_inference graphchallenge_nn_single_inference.cpp
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
	ADDITIONAL_LINK_LIBRARIES test_utils_headers
)

add_grb_executables( simple_pagerank simple_pagerank.cpp
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
	ADDITIONAL_LINK_LIBRARIES test_utils_headers
)

add_grb_executables( pregel_pagerank_local pregel_pagerank.cpp
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
	ADDITIONAL_LINK_LIBRARIES test_utils_headers
	COMPILE_DEFINITIONS PR_CONVERGENCE_MODE=true
)

add_grb_executables( pregel_pagerank_global pregel_pagerank.cpp
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
	ADDITIONAL_LINK_LIBRARIES test_utils_headers
	COMPILE_DEFINITIONS PR_CONVERGENCE_MODE=false
)

add_grb_executables( pregel_connected_components pregel_connected_components.cpp
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
)

add_grb_executables( conjugate_gradient conjugate_gradient.cpp
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
	ADDITIONAL_LINK_LIBRARIES test_utils_headers
)

add_grb_executables( conjugate_gradient_complex conjugate_gradient.cpp
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
	ADDITIONAL_LINK_LIBRARIES test_utils_headers
	COMPILE_DEFINITIONS _CG_COMPLEX
)

add_grb_executables( gmres gmres.cpp
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
	ADDITIONAL_LINK_LIBRARIES test_utils_headers
)

add_grb_executables( gmres_complex gmres.cpp
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
	ADDITIONAL_LINK_LIBRARIES test_utils_headers
	COMPILE_DEFINITIONS _GMRES_COMPLEX
)

add_grb_executables( bicgstab bicgstab.cpp
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
	ADDITIONAL_LINK_LIBRARIES test_utils_headers
)

add_grb_executables( kmeans kmeans.cpp
	BACKENDS reference reference_omp hyperdags nonblocking
)

add_grb_executables( labeltest label_test.cpp
	../unit/parser.cpp
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
)

add_grb_executables( small_pagerank ../unit/auto_launcher.cpp
	hook/small_simple_pagerank.cpp
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
)

add_grb_executables( kcore_decomposition_critical kcore_decomposition.cpp
	ADDITIONAL_LINK_LIBRARIES test_utils_headers
	BACKENDS reference reference_omp
	COMPILE_DEFINITIONS KCORE_VARIANT=true
)

add_grb_executables( kcore_decomposition kcore_decomposition.cpp
	ADDITIONAL_LINK_LIBRARIES test_utils_headers
	BACKENDS reference reference_omp hyperdags nonblocking bsp1d hybrid
)

add_grb_executables( alp_gemm alp_gemm.cpp
	BACKENDS alp_reference
)

add_grb_executables( alp_cholesky alp_cholesky.cpp
	BACKENDS alp_reference alp_dispatch
)

# add_grb_executables( alp_cholesky_complex alp_cholesky.cpp
# 	BACKENDS alp_reference
# 	COMPILE_DEFINITIONS _COMPLEX
# )

add_grb_executables( alp_zhetrd alp_zhetrd.cpp
        BACKENDS alp_reference
)

add_grb_executables( alp_zhetrd_complex alp_zhetrd.cpp
        BACKENDS alp_reference
        COMPILE_DEFINITIONS _COMPLEX
)

add_grb_executables( alp_zgeqrf alp_zgeqrf.cpp
        BACKENDS alp_reference
)

add_grb_executables( alp_zgeqrf_complex alp_zgeqrf.cpp
        BACKENDS alp_reference
	COMPILE_DEFINITIONS _COMPLEX
)

add_grb_executables( alp_zgesvd alp_zgesvd.cpp
        BACKENDS alp_reference
)

add_grb_executables( alp_zgesvd_complex alp_zgesvd.cpp
        BACKENDS alp_reference
	COMPILE_DEFINITIONS _COMPLEX
)

add_grb_executables( alp_zgetrf alp_zgetrf.cpp
        BACKENDS alp_reference
)

add_grb_executables( alp_zgetrf_complex alp_zgetrf.cpp
        BACKENDS alp_reference
	COMPILE_DEFINITIONS _COMPLEX
)

add_grb_executables( alp_dstedc alp_stedc.cpp
        BACKENDS alp_reference
)

add_grb_executables( alp_backsubstitution alp_backsubstitution.cpp
        BACKENDS alp_reference
)

add_grb_executables( alp_backsubstitution_complex alp_backsubstitution.cpp
        BACKENDS alp_reference
	COMPILE_DEFINITIONS _COMPLEX
)

add_grb_executables( alp_forwardsubstitution alp_forwardsubstitution.cpp
        BACKENDS alp_reference
)

add_grb_executables( alp_forwardsubstitution_complex alp_forwardsubstitution.cpp
        BACKENDS alp_reference
	COMPILE_DEFINITIONS _COMPLEX
)

add_grb_executables( alp_syevd alp_zheevd.cpp
        BACKENDS alp_reference
)

add_grb_executables( alp_potri alp_potri.cpp
        BACKENDS alp_reference
)

add_grb_executables( alp_potri_complex alp_potri.cpp
        BACKENDS alp_reference
	COMPILE_DEFINITIONS _COMPLEX
)      

# targets to list and build the test for this category
get_property( smoke_tests_list GLOBAL PROPERTY tests_category_smoke )
add_custom_target( "list_tests_category_smoke"
	COMMAND echo "${smoke_tests_list}"
	COMMAND_EXPAND_LISTS
)

add_custom_target( "build_tests_category_smoke"
	DEPENDS "${smoke_tests_list}"
)

# target to run the tests in this category by calling the appropriate runner
add_custom_target( tests_smoke
	COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/smoketests.sh
		${SCRIPTS_COMMON_ARGS}
		"--test-bin-dir" "\"${CMAKE_CURRENT_BINARY_DIR}\""
		"--test-out-dir" "\"${CMAKE_CURRENT_BINARY_DIR}/output\""
		"--output-verification-dir" "\"${CMAKE_CURRENT_SOURCE_DIR}/output_verification\""
		"--test-data-dir" "\"${CMAKE_CURRENT_SOURCE_DIR}/test_data\""

	WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
	DEPENDS build_tests_category_smoke # add dependency on this target
	                                   # to automatically build before running
	COMMAND_EXPAND_LISTS
	USES_TERMINAL
)

# custom target "smoketests" just to keep old naming
add_custom_target( smoketests DEPENDS tests_smoke )

set( ALP_SCRIPTS_COMMON_ARGS "--backends" )

foreach( b ${AVAILABLE_BACKENDS} )
	string( FIND ${b} "alp" alp_pos_in_b )
	# message("alp_pos_in_b = ${alp_pos_in_b}")
	if( "${alp_pos_in_b}" STREQUAL "-1" )
		continue()
	endif()
	# message("Adding ALP backend ${b} to the test list")
	get_property( bt GLOBAL PROPERTY tests_backend_${b} )
	list( APPEND ALP_TEST_BACKENDS "${bt}" )
	list( APPEND ALP_SCRIPTS_COMMON_ARGS ${b} )
endforeach()

# message("ALP backends enabled for testing: ${ALP_SCRIPTS_COMMON_ARGS}")
# message("Collected ALP test targets: ${ALP_TEST_BACKENDS}")

set( alp_smoke_tests_list ${ALP_TEST_BACKENDS})
list(APPEND S_minus_T ${ALP_TEST_BACKENDS})
list(REMOVE_ITEM S_minus_T ${smoke_tests_list})
list(REMOVE_ITEM alp_smoke_tests_list ${S_minus_T} )

add_custom_target( smoketests_alp
	COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/smoketests.sh
		${ALP_SCRIPTS_COMMON_ARGS}
		"--test-bin-dir" "\"${CMAKE_CURRENT_BINARY_DIR}\""
		"--test-out-dir" "\"${CMAKE_CURRENT_BINARY_DIR}/output\""
		"--output-verification-dir" "\"${CMAKE_CURRENT_SOURCE_DIR}/output_verification\""
		"--test-data-dir" "\"${CMAKE_CURRENT_SOURCE_DIR}/test_data\""

	WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
	DEPENDS ${alp_smoke_tests_list} # add dependency on this target
	                                   # to automatically build before running
	COMMAND_EXPAND_LISTS
	USES_TERMINAL
)
