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

assert_valid_variables( DATASETS_DIR MAX_THREADS )

# write here the name of the category
# add_grb_executables and friends need this information

set( shmem_test_threads "1;2" )
if( "${MAX_THREADS}" GREATER 2 )
	list( APPEND test_threads "${MAX_THREADS}" )
endif()

set( hybrid_threads 2 )
math( EXPR _hybrid_max_threads "${MAX_THREADS}/7" OUTPUT_FORMAT DECIMAL )
if( "${_hybrid_max_threads}" GREATER 2 )
	list( APPEND hybrid_threads "${_hybrid_max_threads}" )
endif()

# Setup the environment for the tests in this directory
setup_grb_tests_environment(
	CATEGORY "unit"

	BSP1D_PROCESSES "1;2;16"
	HYBRID_PROCESSES "2;7"

	REFERENCE_OMP_THREADS "${shmem_test_threads}"
	NONBLOCKING_THREADS "${shmem_test_threads}"
	HYBRID_THREADS "${hybrid_threads}"
)

set( OUTPUT_VERIFICATION_DIR "${CMAKE_CURRENT_SOURCE_DIR}/output_verification" )

# list tests, without the CATEROR[IES,Y] keyword (it's now passed via TEST_CATEGORY)
add_grb_executables( equals equals.cpp
	BACKENDS reference
)
add_grb_tests( equals equals BACKENDS reference Test_OK_SUCCESS )

add_grb_executables( add15d add15d.cpp
	BACKENDS reference
)
add_grb_tests( add15d add15d BACKENDS reference Test_OK_SUCCESS )

add_grb_executables( add15m add15m.cpp
	BACKENDS reference
)
add_grb_tests( add15m add15m BACKENDS reference Test_OK_SUCCESS )

add_grb_executables( argmax argmax.cpp
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
)
add_grb_tests( argmax argmax Test_OK_SUCCESS
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
)

add_grb_executables( argmin argmin.cpp
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
)
add_grb_tests( argmin argmin Test_OK_SUCCESS
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
)

add_grb_executables( buildVector buildVector.cpp
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
)
add_grb_tests( buildVector buildVector Test_OK_SUCCESS
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
)

add_grb_executables( clearMatrix clearMatrix.cpp
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
)
add_grb_tests( clearMatrix clearMatrix ARGUMENTS 10000000 Test_OK_SUCCESS
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
)

add_grb_executables( compareParserTest parser.cpp
	BACKENDS reference
	COMPILE_DEFINITIONS COMPARE
)
add_grb_tests( compareParserTest compareParserTest BACKENDS reference Test_OK_SUCCESS
	ARGUMENTS ${DATASETS_DIR}/cit-HepTh.txt
	REQUIRED_FILES ${DATASETS_DIR}/cit-HepTh.txt
)

add_grb_executables( copyAndAssignVectorIterator copyAndAssignVectorIterator.cpp
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
)
add_grb_tests( copyAndAssignVectorIterator copyAndAssignVectorIterator
	ARGUMENTS 10000000 Test_OK_SUCCESS
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
)

add_grb_executables( copyVector copyVector.cpp
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
)
add_grb_tests( copyVector copyVector ARGUMENTS 100 Test_OK_SUCCESS
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
)
add_grb_tests( copyVector_large copyVector ARGUMENTS 10000000 Test_OK_SUCCESS
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
)

add_grb_executables( distribution_indices distribution_bsp1d.cpp
	BACKENDS bsp1d
)
add_grb_tests( distribution_indices distribution_indices
	BACKENDS bsp1d Test_OK_SUCCESS
)

add_grb_executables( distribution_matrix_bsp1d distribution_matrix_bsp1d.cpp
	BACKENDS bsp1d
	ADDITIONAL_LINK_LIBRARIES test_utils
)

add_grb_executables( distribution distribution.cpp
	BACKENDS bsp1d
)

add_grb_executables( distribution_length distribution.cpp
	BACKENDS bsp1d
)
add_grb_tests( distribution_length distribution_length
	BACKENDS bsp1d Test_OK_SUCCESS PROCESSES 1
)

add_grb_executables( id id.cpp
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
)
add_grb_tests( id id Test_OK_SUCCESS
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
)

add_grb_executables( id_distributed id_distributed.cpp
	BACKENDS bsp1d hybrid
)
add_grb_tests( id_distributed id_distributed Test_OK_SUCCESS
	BACKENDS bsp1d hybrid
)

add_grb_executables( dot dot.cpp
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
)
add_grb_tests( dot dot ARGUMENTS 1874 Test_OK_SUCCESS
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
)
add_grb_tests( dot_large dot ARGUMENTS 10000000 Test_OK_SUCCESS
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
)

add_grb_executables( emptyVector emptyVector.cpp
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
)
add_grb_tests( emptyVector emptyVector Test_OK_SUCCESS
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
)

add_grb_executables( ewiseapply ewiseapply.cpp
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
)
add_grb_tests( ewiseapply_small ewiseapply ARGUMENTS 14 Test_OK_SUCCESS
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
)
add_grb_tests( ewiseapply ewiseapply ARGUMENTS 100 Test_OK_SUCCESS
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
)
add_grb_tests( ewiseapply_large ewiseapply ARGUMENTS 10000000 Test_OK_SUCCESS
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
)

add_grb_executables( eWiseMatrix eWiseMatrix.cpp
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
)
add_grb_tests( eWiseMatrix eWiseMatrix Test_OK_SUCCESS
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
)

add_grb_executables( collectives_blas0 auto_launcher.cpp
	launcher/collectives_blas0.cpp
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
)
add_grb_tests( collectives_blas0 collectives_blas0 Test_OK_SUCCESS
	BACKENDS reference reference_omp hyperdags nonblocking ARGUMENTS 1
)
foreach( P 1 2 16 )
	add_grb_tests( collectives_blas0 collectives_blas0 Test_OK_SUCCESS
		BACKENDS bsp1d hybrid
		PROCESSES ${P} ARGUMENTS ${P}
	)
endforeach()


add_grb_executables( fold_to_scalar auto_launcher.cpp
	launcher/reduce.cpp
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
)
add_grb_tests( fold_to_scalar fold_to_scalar Test_OK_SUCCESS
	BACKENDS reference reference_omp hyperdags nonblocking ARGUMENTS 1
)
foreach( P 1 2 16 )
	add_grb_tests( fold_to_scalar fold_to_scalar Test_OK_SUCCESS
		BACKENDS bsp1d hybrid
		PROCESSES ${P} ARGUMENTS ${P}
	)
endforeach()

add_grb_executables( fork_launcher fork_launcher.cpp
	BACKENDS bsp1d
)

add_grb_executables( hpparser ${ALP_UTILS_SRC_PATH}/hpparser.c
	BACKENDS none
	ADDITIONAL_LINK_LIBRARIES backend_headers_nodefs OpenMP::OpenMP_C
	COMPILE_DEFINITIONS TEST_HPPARSER _GNU_SOURCE _DEBUG
)
set( hpparser_golden_out "[ 0, *] nrow =           59, ncol =           59, nnnz =          163
[ 0, *] offb =          564, fsiz =         1494, offe =         1493
[ *, *] ntot =          163" )
add_grb_tests( hpparser hpparser
	BACKENDS none
	REQUIRED_FILES ${DATASETS_DIR}/dwt_59.mtx
	ARGUMENTS 1 ${MAX_THREADS} 131072 8388608 ${DATASETS_DIR}/dwt_59.mtx 1
	OUTPUT_VALIDATE diff "<(echo \"${hpparser_golden_out}\")" @@TEST_OUTPUT_FILE@@
)

add_grb_executables( masked_mxv masked_mxv.cpp
	BACKENDS reference reference_omp hyperdags nonblocking
)
add_grb_tests( masked_mxv masked_mxv Test_OK_SUCCESS
	BACKENDS reference reference_omp hyperdags nonblocking
)

add_grb_executables( masked_vxm masked_vxm.cpp
	BACKENDS reference reference_omp hyperdags nonblocking
)
add_grb_tests( masked_vxm masked_vxm Test_OK_SUCCESS
	BACKENDS reference reference_omp hyperdags nonblocking
)

add_grb_executables( matrixIterator matrixIterator.cpp
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
	ADDITIONAL_LINK_LIBRARIES test_utils_headers
)
add_grb_tests( matrixIterator matrixIterator ARGUMENTS 10000000 Test_OK_SUCCESS
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
)

add_grb_executables( fold_matrix_to_scalar fold_matrix_to_scalar.cpp
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
)

add_grb_executables( doubleAssign doubleAssign.cpp
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
)
add_grb_tests( doubleAssign doubleAssign ARGUMENTS 1337 Test_OK_SUCCESS
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
)

add_grb_executables( matrixSet matrixSet.cpp
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
)
add_grb_tests( matrixSet matrixSet Test_OK_SUCCESS
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
)

add_grb_executables( moveVector moveVector.cpp
	BACKENDS reference reference_omp hyperdags bsp1d hybrid nonblocking
)
add_grb_tests( moveVector moveVector ARGUMENTS 100 Test_OK_SUCCESS
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
)

add_grb_executables( mul15i mul15i.cpp
	BACKENDS reference
)
add_grb_tests( mul15i mul15i BACKENDS reference Test_OK_SUCCESS )

add_grb_executables( mul15m mul15m.cpp
	BACKENDS reference
)
add_grb_tests( mul15m mul15m BACKENDS reference Test_OK_SUCCESS )

add_grb_executables( eWiseMul eWiseMul.cpp
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
)
add_grb_tests( eWiseMul eWiseMul ARGUMENTS 100 Test_OK_SUCCESS
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
)
add_grb_tests( eWiseMul_large eWiseMul ARGUMENTS 100002 Test_OK_SUCCESS
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
)

add_grb_executables( factories factories.cpp
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
)
add_grb_tests( factories factories Test_OK_SUCCESS
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
)

add_grb_executables( muladd muladd.cpp
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
)
add_grb_tests( muladd_large muladd ARGUMENTS 10000000 Test_OK_SUCCESS
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
)

add_grb_executables( mxm mxm.cpp
	BACKENDS reference reference_omp hyperdags nonblocking #bsp1d hybrid
	ADDITIONAL_LINK_LIBRARIES test_utils
)
add_grb_tests( mxm mxm Test_OK_SUCCESS
	BACKENDS reference reference_omp hyperdags nonblocking
)

add_grb_executables( parserTest utilParserTest.cpp
	BACKENDS reference
	COMPILE_DEFINITIONS COMPARE
)
add_grb_tests( parserTest parserTest
	BACKENDS reference Test_OK_SUCCESS
	REQUIRED_FILES ${DATASETS_DIR}/west0497.mtx
	ARGUMENTS ${DATASETS_DIR}/west0497.mtx
)

add_grb_executables( iteratorFilter iteratorFilter.cpp
	BACKENDS reference
)
add_grb_tests( iteratorFilter_tiny iteratorFilter
	BACKENDS reference ARGUMENTS 3 Test_OK_SUCCESS
)
add_grb_tests( iteratorFilter_default iteratorFilter
	BACKENDS reference Test_OK_SUCCESS
)
add_grb_tests( iteratorFilter_large iteratorFilter
	BACKENDS reference ARGUMENTS 7013 Test_OK_SUCCESS
)

add_grb_executables( RBGaussSeidel RBGaussSeidel.cpp
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
)
add_grb_tests( RBGaussSeidel RBGaussSeidel
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
	REQUIRED_FILES ${DATASETS_DIR}/west0497.mtx
	ARGUMENTS ${DATASETS_DIR}/west0497.mtx Test_OK_SUCCESS
)

add_grb_executables( selectMatrix selectMatrix.cpp
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
	ADDITIONAL_LINK_LIBRARIES test_utils_headers
)
add_grb_tests( selectMatrix_small selectMatrix
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
	ARGUMENTS 3 Test_OK_SUCCESS
)
add_grb_tests( selectMatrix_big selectMatrix
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
	ARGUMENTS 5000 Test_OK_SUCCESS
)

add_grb_executables( set set.cpp
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
)
add_grb_tests( set set ARGUMENTS 1000000 Test_OK_SUCCESS
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
)

add_grb_executables( sparse_mxv sparse_mxv.cpp
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
)
add_grb_tests( sparse_mxv sparse_mxv Test_OK_SUCCESS
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
)

add_grb_executables( sparse_vxm sparse_vxm.cpp
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
)
add_grb_tests( sparse_vxm_10_1 sparse_vxm Test_OK_SUCCESS
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
	ARGUMENTS 10 1 1 1
)
add_grb_tests( sparse_vxm_10_2 sparse_vxm Test_OK_SUCCESS
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
	ARGUMENTS 10 2 1 1
)
add_grb_tests( sparse_vxm_10_3 sparse_vxm Test_OK_SUCCESS
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
	ARGUMENTS 10 3 1 1
)
add_grb_tests( sparse_vxm_10_4 sparse_vxm Test_OK_SUCCESS
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
	ARGUMENTS 10 4 1 1
)
add_grb_tests( sparse_vxm_1000_1 sparse_vxm Test_OK_SUCCESS
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
	ARGUMENTS 1000 1 1 1
)
add_grb_tests( sparse_vxm_1000_2 sparse_vxm Test_OK_SUCCESS
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
	ARGUMENTS 1000 2 1 1
)
add_grb_tests( sparse_vxm_1000_3 sparse_vxm Test_OK_SUCCESS
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
	ARGUMENTS 1000 3 1 1
)
add_grb_tests( sparse_vxm_1000_4 sparse_vxm Test_OK_SUCCESS
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
	ARGUMENTS 1000 4 1 1
)

add_grb_executables( stdVector stdVector.cpp
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
)
add_grb_tests( stdVector stdVector ARGUMENTS 100 Test_OK_SUCCESS
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
)

add_grb_executables( swapVector swapVector.cpp
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
)
add_grb_tests( swapVector swapVector ARGUMENTS 100 Test_OK_SUCCESS
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
)

add_grb_executables( thread_local_storage thread_local_storage.cpp
	BACKENDS reference
	ADDITIONAL_LINK_LIBRARIES Threads::Threads
)
add_grb_tests( thread_local_storage thread_local_storage BACKENDS reference )

add_grb_executables( vectorToMatrix vectorToMatrix.cpp
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
)
add_grb_tests( vectorToMatrix vectorToMatrix Test_OK_SUCCESS
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
)

add_grb_executables( vmxa vmxa.cpp
	BACKENDS reference reference_omp hyperdags nonblocking
)
add_grb_tests( vmxa vmxa Test_OK_SUCCESS
	BACKENDS reference reference_omp hyperdags nonblocking
)

add_grb_executables( vmx vmx.cpp
	BACKENDS reference reference_omp hyperdags nonblocking
)
add_grb_tests( vmx vmx Test_OK_SUCCESS
	BACKENDS reference reference_omp hyperdags nonblocking
)

add_grb_executables( zip zip.cpp
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
)
add_grb_tests( zip_large zip ARGUMENTS 10000000 Test_OK_SUCCESS
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
)

add_grb_executables( copyVoidMatrices copyVoidMatrices.cpp
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
)
add_grb_tests( copyVoidMatrices copyVoidMatrices ARGUMENTS 1003 Test_OK_SUCCESS
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
)

add_grb_executables( masked_muladd masked_muladd.cpp
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
)
add_grb_tests( masked_muladd_large masked_muladd ARGUMENTS 7000000 Test_OK_SUCCESS
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
)

add_grb_executables( spy spy.cpp
	BACKENDS reference reference_omp hyperdags nonblocking
)
add_grb_tests( spy spy
	BACKENDS reference reference_omp hyperdags nonblocking
	REQUIRED_FILES ${DATASETS_DIR}/west0497.mtx
	ARGUMENTS ${DATASETS_DIR}/west0497.mtx
	OUTPUT_VALIDATE grep 'Spy matrix' @@TEST_OUTPUT_FILE@@ | cut -d' ' -f9 | grep -q 315
)

add_grb_executables( dense_spmv dense_spmv.cpp
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
)
add_grb_tests( dense_spmv_Ax dense_spmv Test_OK_SUCCESS
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
	REQUIRED_FILES ${DATASETS_DIR}/west0497.mtx
	ARGUMENTS ${DATASETS_DIR}/west0497.mtx direct 1 1 1
)
add_grb_tests( dense_spmv_ATx dense_spmv Test_OK_SUCCESS
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
	REQUIRED_FILES ${DATASETS_DIR}/west0497.mtx
	ARGUMENTS ${DATASETS_DIR}/west0497.mtx direct 2 1 1
)
add_grb_tests( dense_spmv_xA dense_spmv Test_OK_SUCCESS
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
	REQUIRED_FILES ${DATASETS_DIR}/west0497.mtx
	ARGUMENTS ${DATASETS_DIR}/west0497.mtx direct 3 1 1
)
add_grb_tests( dense_spmv_xAT dense_spmv Test_OK_SUCCESS
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
	REQUIRED_FILES ${DATASETS_DIR}/west0497.mtx
	ARGUMENTS ${DATASETS_DIR}/west0497.mtx direct 4 1 1
)

add_grb_executables( moveMatrix moveMatrix.cpp
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
)
add_grb_tests( moveMatrix moveMatrix ARGUMENTS 100 Test_OK_SUCCESS
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
)

add_grb_executables( stdMatrix stdMatrix.cpp
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
)
add_grb_tests( stdMatrix stdMatrix ARGUMENTS 100 Test_OK_SUCCESS
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
)

add_grb_executables( eWiseApply_matrix eWiseApply_matrix.cpp
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
)
add_grb_tests( eWiseApply_matrix eWiseApply_matrix Test_OK_SUCCESS
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
)

# in the below, test_utils_headers is retained in case CMake is configured to
# include _DEBUG flags
add_grb_executables( eWiseApplyMatrixReference eWiseApplyMatrixReference.cpp
	BACKENDS reference reference_omp hyperdags nonblocking
	ADDITIONAL_LINK_LIBRARIES test_utils_headers
)
add_grb_tests( eWiseApplyMatrixReference_tiny eWiseApplyMatrixReference
	BACKENDS reference reference_omp hyperdags nonblocking Test_OK_SUCCESS
)

add_grb_executables( eWiseLambda eWiseLambda.cpp
	BACKENDS reference reference_omp hyperdags nonblocking
)
add_grb_tests( eWiseLambda eWiseLambda
	BACKENDS reference reference_omp hyperdags nonblocking Test_OK_SUCCESS
)

add_grb_executables( outer outer.cpp
	BACKENDS reference reference_omp hyperdags nonblocking
)
add_grb_tests( outer outer Test_OK_SUCCESS
	BACKENDS reference reference_omp hyperdags nonblocking
)

add_grb_executables( mxv mxv.cpp
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
	ADDITIONAL_LINK_LIBRARIES test_utils_headers
)
add_grb_tests( mxv_west0497 mxv Test_OK_SUCCESS
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
	REQUIRED_FILES ${DATASETS_DIR}/west0497.mtx
	ARGUMENTS ${DATASETS_DIR}/west0497.mtx ${OUTPUT_VERIFICATION_DIR}/mxv_reference_golden.log
)
# extra tests for bsp1d backend
add_grb_tests( mxv_west0497_extra mxv Test_OK_SUCCESS
	BACKENDS bsp1d PROCESSES "1;2;3;4"
	REQUIRED_FILES ${DATASETS_DIR}/west0497.mtx
	ARGUMENTS ${DATASETS_DIR}/west0497.mtx ${OUTPUT_VERIFICATION_DIR}/mxv_reference_golden.log
)

add_grb_executables( vxm vxm.cpp
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
	ADDITIONAL_LINK_LIBRARIES test_utils_headers
)
add_grb_tests( vxm_west0497 vxm Test_OK_SUCCESS
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
	REQUIRED_FILES ${DATASETS_DIR}/west0497.mtx
	ARGUMENTS ${DATASETS_DIR}/west0497.mtx ${OUTPUT_VERIFICATION_DIR}/vxm_reference_golden.log
)
# extra tests for bsp1d backend
add_grb_tests( vxm_west0497_extra vxm Test_OK_SUCCESS
	BACKENDS bsp1d PROCESSES "1;2;3;4"
	REQUIRED_FILES ${DATASETS_DIR}/west0497.mtx
	ARGUMENTS ${DATASETS_DIR}/west0497.mtx ${OUTPUT_VERIFICATION_DIR}/vxm_reference_golden.log
)

add_grb_executables( capacity capacity.cpp
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
)
add_grb_tests( capacity capacity ARGUMENTS 5230 Test_OK_SUCCESS
	BACKENDS reference reference_omp hyperdags nonblocking
)

add_grb_executables( wait wait.cpp
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
)
add_grb_tests( wait wait Test_OK_SUCCESS
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
)
add_grb_tests( wait_large wait ARGUMENTS 11733 Test_OK_SUCCESS
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
)

add_grb_executables( buildMatrixUnique buildMatrixUnique.cpp
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
	ADDITIONAL_LINK_LIBRARIES test_utils
)
add_grb_tests( buildMatrixUnique buildMatrixUnique Test_OK_SUCCESS
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
)

add_grb_executables( pinnedVector pinnedVector.cpp
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
)
add_grb_tests( pinnedVector pinnedVector Test_OK_SUCCESS
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
)

add_grb_executables( parallelRegularIterators parallelRegularIterators.cpp
	BACKENDS reference reference_omp hyperdags nonblocking bsp1d hybrid
)
add_grb_tests( parallelRegularIterators parallelRegularIterators Test_OK_SUCCESS
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
)

add_grb_executables( adapterIterator adapterIterator.cpp
	BACKENDS reference reference_omp hyperdags nonblocking bsp1d hybrid
)
add_grb_tests( adapterIterator adapterIterator Test_OK_SUCCESS
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
)

add_grb_executables( vectorFromListConstructor vectorFromListConstructor.cpp
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
)
add_grb_tests( vectorFromListConstructor vectorFromListConstructor Test_OK_SUCCESS
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
# AUTOMATIC mode
add_grb_tests( launch_benchmark_auto launch_benchmark_auto Test_OK_SUCCESS
	BACKENDS reference reference_omp hyperdags nonblocking bsp1d hybrid
)

add_grb_executables( launch_benchmark_frommpi_manual launcherAndBenchmarker.cpp
	BACKENDS bsp1d hybrid
	COMPILE_DEFINITIONS DISTRIBUTED_EXECUTION NO_LPF_AUTO_INIT
)
# FROM_MPI mode for distributed backends
add_grb_tests( launch_benchmark_frommpi launch_benchmark_frommpi_manual Test_OK_SUCCESS
	BACKENDS bsp1d hybrid
)
# MANUAL mode for distributed backends
add_grb_tests( launch_benchmark_manual launch_benchmark_frommpi_manual Test_OK_SUCCESS
	BACKENDS bsp1d hybrid
	PROCESSES 1 PARALLEL_PROCESSES 4
	ARGUMENTS localhost 77770 4
)

add_grb_executables( launch_benchmark_frommpi_manual launcherAndBenchmarker.cpp
	BACKENDS reference reference_omp hyperdags nonblocking
	COMPILE_DEFINITIONS NO_LPF_AUTO_INIT
)
# MANUAL mode for shared-memory backends
add_grb_tests( launch_benchmark_manual launch_benchmark_frommpi_manual Test_OK_SUCCESS
	BACKENDS reference reference_omp hyperdags nonblocking
	ARGUMENTS localhost 77770 1 0
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

# target to run the tests in this category by calling the appropriate runner
add_custom_target( unittests
	COMMAND ${CMAKE_CTEST_COMMAND} -L "mode:unit" --output-on-failure
	WORKING_DIRECTORY "${PROJECT_BINARY_DIR}"
	DEPENDS build_tests_category_unit
)
