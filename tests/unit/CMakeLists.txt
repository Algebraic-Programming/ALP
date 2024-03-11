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
set( TEST_CATEGORY "unit" )

# list tests, without the CATEROR[IES,Y] keyword (it's now passed via TEST_CATEGORY)
add_grb_executables( equals equals.cpp
	BACKENDS reference NO_BACKEND_NAME
)

add_grb_executables( add15d add15d.cpp
	BACKENDS reference NO_BACKEND_NAME
)

add_grb_executables( add15m add15m.cpp
	BACKENDS reference NO_BACKEND_NAME
)

add_grb_executables( argmax argmax.cpp
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
)

add_grb_executables( argmin argmin.cpp
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
)

add_grb_executables( buildVector buildVector.cpp
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
)

add_grb_executables( clearMatrix clearMatrix.cpp
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
)

add_grb_executables( compareParserTest parser.cpp
	BACKENDS reference NO_BACKEND_NAME
	COMPILE_DEFINITIONS COMPARE
)

add_grb_executables( copyAndAssignVectorIterator copyAndAssignVectorIterator.cpp
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
)

add_grb_executables( copyVector copyVector.cpp
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
)

add_grb_executables( distribution_bsp1d distribution_bsp1d.cpp
	BACKENDS reference NO_BACKEND_NAME
)

add_grb_executables( distribution_matrix_bsp1d distribution_matrix_bsp1d.cpp
	BACKENDS bsp1d NO_BACKEND_NAME
	ADDITIONAL_LINK_LIBRARIES test_utils
)

add_grb_executables( distribution distribution.cpp
	BACKENDS bsp1d NO_BACKEND_NAME
)

add_grb_executables( id id.cpp
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
)

add_grb_executables( id_distributed id_distributed.cpp
	BACKENDS bsp1d hybrid
)

add_grb_executables( dot dot.cpp
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
)

add_grb_executables( emptyVector emptyVector.cpp
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
)

add_grb_executables( ewiseapply ewiseapply.cpp
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
)

add_grb_executables( eWiseMatrix eWiseMatrix.cpp
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
)

add_grb_executables( collectives_blas0 auto_launcher.cpp
	launcher/collectives_blas0.cpp
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
)

add_grb_executables( fold_to_scalar auto_launcher.cpp
	launcher/reduce.cpp
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
)

add_grb_executables( fork_launcher fork_launcher.cpp
	BACKENDS bsp1d NO_BACKEND_NAME
)

add_grb_executable_custom( hpparser ${ALP_UTILS_SRC_PATH}/hpparser.c
	LINK_LIBRARIES backend_headers_nodefs OpenMP::OpenMP_C
	COMPILE_DEFINITIONS TEST_HPPARSER _GNU_SOURCE _DEBUG
)

add_grb_executables( masked_mxv masked_mxv.cpp
	BACKENDS reference reference_omp hyperdags nonblocking
)

add_grb_executables( masked_vxm masked_vxm.cpp
	BACKENDS reference reference_omp hyperdags nonblocking
)

add_grb_executables( matrixIterator matrixIterator.cpp
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
	ADDITIONAL_LINK_LIBRARIES test_utils_headers
)

add_grb_executables( fold_matrix_to_scalar fold_matrix_to_scalar.cpp
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
)

add_grb_executables( doubleAssign doubleAssign.cpp
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
)

add_grb_executables( matrixSet matrixSet.cpp
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
)

add_grb_executables( moveVector moveVector.cpp
	BACKENDS reference reference_omp hyperdags bsp1d hybrid nonblocking
)

add_grb_executables( mul15i mul15i.cpp
	BACKENDS reference NO_BACKEND_NAME
)

add_grb_executables( mul15m mul15m.cpp
	BACKENDS reference NO_BACKEND_NAME
)

add_grb_executables( eWiseMul eWiseMul.cpp
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
)

add_grb_executables( factories factories.cpp
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
)

add_grb_executables( muladd muladd.cpp
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
)

add_grb_executables( mxm mxm.cpp
	BACKENDS reference reference_omp hyperdags nonblocking #bsp1d hybrid
	ADDITIONAL_LINK_LIBRARIES test_utils
)

add_grb_executables( parserTest utilParserTest.cpp
	BACKENDS reference NO_BACKEND_NAME
	COMPILE_DEFINITIONS COMPARE
)

add_grb_executables( iteratorFilter iteratorFilter.cpp
	BACKENDS reference NO_BACKEND_NAME
)

add_grb_executables( RBGaussSeidel RBGaussSeidel.cpp
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
)

add_grb_executables( selectMatrix selectMatrix.cpp
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
	ADDITIONAL_LINK_LIBRARIES test_utils_headers
)

add_grb_executables( set set.cpp
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
)

add_grb_executables( sparse_mxv sparse_mxv.cpp
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
)

add_grb_executables( sparse_vxm sparse_vxm.cpp
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
)

add_grb_executables( stdVector stdVector.cpp
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
)

add_grb_executables( swapVector swapVector.cpp
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
)

add_grb_executables( thread_local_storage thread_local_storage.cpp
	BACKENDS reference NO_BACKEND_NAME
	ADDITIONAL_LINK_LIBRARIES Threads::Threads
)

add_grb_executables( vectorToMatrix vectorToMatrix.cpp
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
)

add_grb_executables( vmxa vmxa.cpp
	BACKENDS reference reference_omp bsp1d hyperdags nonblocking
)

add_grb_executables( vmx vmx.cpp
	BACKENDS reference reference_omp bsp1d hyperdags nonblocking
)

add_grb_executables( zip zip.cpp
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
)

add_grb_executables( copyVoidMatrices copyVoidMatrices.cpp
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
)

add_grb_executables( masked_muladd masked_muladd.cpp
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
)

add_grb_executables( spy spy.cpp
	BACKENDS reference reference_omp hyperdags nonblocking
)

add_grb_executables( dense_spmv dense_spmv.cpp
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
)

add_grb_executables( moveMatrix moveMatrix.cpp
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
)

add_grb_executables( stdMatrix stdMatrix.cpp
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
)

add_grb_executables( eWiseApply_matrix eWiseApply_matrix.cpp
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
)

# in the below, test_utils_headers is retained in case CMake is configured to
# include _DEBUG flags
add_grb_executables( eWiseApplyMatrixReference eWiseApplyMatrixReference.cpp
	BACKENDS reference reference_omp hyperdags nonblocking
	ADDITIONAL_LINK_LIBRARIES test_utils_headers
)

add_grb_executables( eWiseLambda eWiseLambda.cpp
	BACKENDS reference reference_omp hyperdags nonblocking
)

add_grb_executables( outer outer.cpp
	BACKENDS reference reference_omp hyperdags nonblocking
)

# must generate the golden output for other tests
force_add_grb_executable( mxv mxv.cpp
	BACKEND reference
)
add_grb_executables( mxv mxv.cpp
	BACKENDS reference_omp bsp1d hybrid hyperdags nonblocking
)

# must generate the golden output for other tests
force_add_grb_executable( vxm vxm.cpp
	BACKEND reference
)
add_grb_executables( vxm vxm.cpp
	BACKENDS reference_omp bsp1d hybrid hyperdags nonblocking
)

add_grb_executables( capacity capacity.cpp
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
)

add_grb_executables( wait wait.cpp
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
)

add_grb_executables( buildMatrixUnique buildMatrixUnique.cpp
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
	ADDITIONAL_LINK_LIBRARIES test_utils
)

add_grb_executables( pinnedVector pinnedVector.cpp
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
)

add_grb_executables( parallelRegularIterators parallelRegularIterators.cpp
	BACKENDS reference reference_omp hyperdags nonblocking bsp1d hybrid
)

add_grb_executables( adapterIterator adapterIterator.cpp
	BACKENDS reference reference_omp hyperdags nonblocking bsp1d hybrid
)

add_grb_executables( vectorFromListConstructor vectorFromListConstructor.cpp
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
)

# the below targets test successfully when they compile -- they do not need to
# be executed successfully as part of the unit test suite.

add_grb_executables( properties static_asserts/properties.cpp
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
)

add_grb_executables( matrix_type static_asserts/matrix.cpp
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
)

add_grb_executables( launch_benchmark_auto launcherAndBenchmarker.cpp
	BACKENDS bsp1d hybrid
	COMPILE_DEFINITIONS DISTRIBUTED_EXECUTION
)

add_grb_executables( launch_benchmark_auto launcherAndBenchmarker.cpp
	BACKENDS reference reference_omp hyperdags nonblocking
)

add_grb_executables( launch_benchmark_frommpi_manual launcherAndBenchmarker.cpp
	BACKENDS bsp1d hybrid
	COMPILE_DEFINITIONS DISTRIBUTED_EXECUTION NO_LPF_AUTO_INIT
)

add_grb_executables( launch_benchmark_frommpi_manual launcherAndBenchmarker.cpp
	BACKENDS reference reference_omp hyperdags nonblocking
	COMPILE_DEFINITIONS NO_LPF_AUTO_INIT
)

# targets to list and build the test for this category
get_property( unit_tests_list GLOBAL PROPERTY tests_category_unit )
add_custom_target( "list_tests_category_unit"
	COMMAND echo "${unit_tests_list}"
	COMMAND_EXPAND_LISTS
)

add_custom_target( "build_tests_category_unit"
	DEPENDS "${unit_tests_list}"
)


foreach( mode ${MODES_unit} )
	# Removing the prefix
	string( SUBSTRING "${MODES_${mode}_suffix}" 1 -1 __mode )
	list( APPEND __MODES "${__mode}" )
endforeach()

# target to run the tests in this category by calling the appropriate runner
add_custom_target( tests_unit
	COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/unittests.sh
		${SCRIPTS_COMMON_ARGS}
		"--test-bin-dir" "\"${CMAKE_CURRENT_BINARY_DIR}\""
		"--test-out-dir" "\"${CMAKE_CURRENT_BINARY_DIR}/output\""
		"--output-verification-dir" "\"${CMAKE_CURRENT_SOURCE_DIR}/output_verification\""
		"--test-data-dir" "\"${CMAKE_CURRENT_SOURCE_DIR}/test_data\""
		"--modes" "\"${__MODES}\""

	WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
	DEPENDS build_tests_category_unit # add dependency on this target
	                                  # to automatically build before running
	COMMAND_EXPAND_LISTS
	USES_TERMINAL
)

# custom target "unittests" just to keep old naming
add_custom_target( unittests DEPENDS tests_unit )

