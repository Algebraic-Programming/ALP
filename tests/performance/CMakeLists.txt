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


# ======= CONFIGURATION =======
# get number of sockets from lscpu command
execute_process( COMMAND lscpu
	COMMAND grep -i "socket(s)"
	COMMAND awk "{print \$2}"
	RESULT_VARIABLE result
	OUTPUT_VARIABLE num_sockets
	OUTPUT_STRIP_TRAILING_WHITESPACE
)
# check the result makes sense
if ( NOT result STREQUAL "0" OR NOT num_sockets MATCHES "^[0-9]+$")
    message( FATAL_ERROR "cannot parse number of sockets" )
endif ()

math( EXPR _hybrid_max_threads "${MAX_THREADS}/${num_sockets}" OUTPUT_FORMAT DECIMAL )
set( hybrid_threads "1" )
if( "${_hybrid_max_threads}" GREATER "1" )
	set( hybrid_threads "${_hybrid_max_threads}" )
endif()

# Setup the environment for the tests in this directory
setup_grb_tests_environment(
	CATEGORY "performance"

	BSP1D_PROCESSES "1"
	HYBRID_PROCESSES "${num_sockets}"

	HYBRID_THREADS "${hybrid_threads}"
	REFERENCE_OMP_THREADS "${MAX_THREADS}"
	NONBLOCKING_THREADS "${MAX_THREADS}"
)

# Datasets to use & common variables
set( DATASETS
	west0497.mtx facebook_combined.txt cit-HepTh.txt com-amazon.ungraph.txt
	com-youtube.ungraph.txt cit-Patents.txt com-orkut.ungraph.txt
)
set( DATASET_MODES direct direct indirect indirect indirect indirect indirect )
# set( DATASET_SIZES 97 4039 27770 334863 1134890 3774768 3072441 )
list( LENGTH DATASETS __num_datasets )
math( EXPR num_datasets "${__num_datasets}-1" )

set( MULTIPLICATION_DATASETS
	west0497.mtx fidap037.mtx cavity17.mtx s3rmt3m3.mtx bloweybq.mtx
	bcsstk17.mtx Pres_Poisson.mtx gyro_m.mtx memplus.mtx lhr34.mtx
	bcsstk32.mtx vanbody.mtx s3dkt3m2.mtx G2_circuit.mtx Stanford.mtx
	coPapersCiteseer.mtx bundle_adj.mtx Stanford_Berkeley.mtx apache2.mtx
	Emilia_923.mtx ldoor.mtx ecology2.mtx Serena.mtx cage14.mtx G3_circuit.mtx
	wikipedia-20051105.mtx wikipedia-20061104.mtx Freescale1.mtx
	wikipedia-20070206.mtx Queen_4147.mtx cage15.mtx adaptive.mtx
	rgg_n_2_24_s0.mtx uk-2002.mtx road_usa.mtx MOLIERE_2016.mtx europe_osm.mtx
	twitter.mtx com-Friendster.mtx
)
set( KNN4SOLS 59 421 1138 1 32 1 609122 )
set( KNN6SOLS 238 526 4189 1 181 1 1268035 )


### Kernels tests
add_library( bench_kernels OBJECT bench_kernels.c bench_kernels.h )
add_library( bench_kernels_omp OBJECT bench_kernels.c bench_kernels.h )
target_compile_definitions( bench_kernels_omp PRIVATE BENCH_KERNELS_OPENMP )

add_grb_executables( fma fma.cpp $<TARGET_OBJECTS:bench_kernels>
	BACKENDS reference
	ADDITIONAL_LINK_LIBRARIES "rt"
)
add_grb_tests( fma fma
	ARGUMENTS 10000000 0
	Test_OK_SUCCESS
	BACKENDS reference
)

add_grb_executables( fma-openmp fma.cpp $<TARGET_OBJECTS:bench_kernels_omp>
	BACKENDS reference_omp
	ADDITIONAL_LINK_LIBRARIES OpenMP::OpenMP_CXX "rt"
)
add_grb_tests( fma fma-openmp
	ARGUMENTS 10000000 0
	Test_OK_SUCCESS
	BACKENDS reference_omp
)

add_grb_executables( reduce reduce.cpp $<TARGET_OBJECTS:bench_kernels>
	BACKENDS reference
)
add_grb_tests( reduce reduce
	ARGUMENTS 10000000 0
	Test_OK_SUCCESS
	BACKENDS reference
)

add_grb_executables( reduce-openmp reduce.cpp $<TARGET_OBJECTS:bench_kernels_omp>
	BACKENDS reference_omp
	ADDITIONAL_LINK_LIBRARIES OpenMP::OpenMP_CXX
)
add_grb_tests( reduce reduce-openmp
	ARGUMENTS 10000000 0
	Test_OK_SUCCESS
	BACKENDS reference_omp
)

add_grb_executables( dot dot.cpp $<TARGET_OBJECTS:bench_kernels>
	BACKENDS reference
	ADDITIONAL_LINK_LIBRARIES backend_headers_nodefs
)
add_grb_tests( dot dot
	ARGUMENTS 10000000 0
	Test_OK_SUCCESS
	BACKENDS reference
)

add_grb_executables( dot-openmp dot.cpp $<TARGET_OBJECTS:bench_kernels_omp>
	BACKENDS reference_omp
	ADDITIONAL_LINK_LIBRARIES backend_headers_nodefs OpenMP::OpenMP_CXX
)
add_grb_tests( dot dot-openmp
	ARGUMENTS 10000000 0
	Test_OK_SUCCESS
	BACKENDS reference_omp
)

# Scaling tests
add_grb_executables( scaling scaling.cpp ../unit/parser.cpp
	BACKENDS reference reference_omp bsp1d hybrid nonblocking hyperdags
)
foreach( t 1 2 3 4 )
	foreach( d 1000 1000000 10000000 )
		add_grb_tests( "scaling_${d}_${t}" scaling
			ARGUMENTS ${d} ${t} 0
			Test_OK_SUCCESS
			BACKENDS reference reference_omp bsp1d hybrid nonblocking #hyperdags
		)
	endforeach()

	# Hyperdags only
	set( d 1000 )
	add_grb_tests( "scaling_${d}_${t}" scaling
		ARGUMENTS ${d} ${t} 0
		Test_OK_SUCCESS
		BACKENDS hyperdags
	)
endforeach()

# KNN tests
add_grb_executables( driver_knn ../smoke/knn.cpp
	../unit/parser.cpp
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
)
foreach( i RANGE 0 ${num_datasets} )
	list( GET DATASETS ${i} dataSet )
	list( GET DATASET_MODES ${i} parseMode )
	list( GET KNN4SOLS ${i} size )
	add_grb_tests( "driver_knn.4_${dataSet}_${parseMode}" driver_knn
		ARGUMENTS 4 "${DATASETS_DIR}/${dataSet}" ${parseMode}
		REQUIRED_FILES "${DATASETS_DIR}/${dataSet}"
		Test_OK_SUCCESS
		BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
		OUTPUT_VALIDATE grep -q "\"Neighbourhood size is ${size}\"" @@TEST_OUTPUT_FILE@@
	)
endforeach()

# Hyperdags is skipped for kValue == 6
foreach( i RANGE 0 ${num_datasets} )
	list( GET DATASETS ${i} dataSet )
	list( GET DATASET_MODES ${i} parseMode )
	list( GET KNN6SOLS ${i} size )
	add_grb_tests( "driver_knn.6.${dataSet}.${parseMode}" driver_knn
		ARGUMENTS 6 "${DATASETS_DIR}/${dataSet}" ${parseMode}
		REQUIRED_FILES "${DATASETS_DIR}/${dataSet}"
		Test_OK_SUCCESS
		BACKENDS reference reference_omp bsp1d hybrid nonblocking
		OUTPUT_VALIDATE grep -q "Neighbourhood size is ${size}" @@TEST_OUTPUT_FILE@@
	)
endforeach()

# Other tests (simple_pagerank, label)
add_grb_executables( driver_simple_pagerank
	../smoke/simple_pagerank.cpp
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
	ADDITIONAL_LINK_LIBRARIES test_utils_headers
)
add_grb_executables( driver_label label.cpp
	../unit/parser.cpp
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
	ADDITIONAL_LINK_LIBRARIES test_utils_headers
)

# for hyperdags backend, only smallest input for simple_pagerank and label tests
add_grb_tests( "driver_simple_pagerank_${dataSet}_${parseMode}" driver_simple_pagerank
	ARGUMENTS ${DATASETS_DIR}/west0497.mtx direct
	REQUIRED_FILES ${DATASETS_DIR}/west0497.mtx
	Test_OK_SUCCESS
	BACKENDS hyperdags
)
add_grb_tests( "driver_label_${dataSet}_${parseMode}" driver_label
	ARGUMENTS ${DATASETS_DIR}/west0497.mtx direct
	REQUIRED_FILES ${DATASETS_DIR}/west0497.mtx
	Test_OK_SUCCESS
	BACKENDS hyperdags
)
# for all other backends, test for all inputs
foreach( i RANGE 0 ${num_datasets} )
	list( GET DATASETS ${i} dataSet )
	list( GET DATASET_MODES ${i} parseMode )
	add_grb_tests( "driver_simple_pagerank_${dataSet}_${parseMode}" driver_simple_pagerank
		ARGUMENTS "${DATASETS_DIR}/${dataSet}" ${parseMode}
		REQUIRED_FILES "${DATASETS_DIR}/${dataSet}"
		Test_OK_SUCCESS
		BACKENDS reference reference_omp bsp1d hybrid nonblocking
	)
	add_grb_tests( "driver_label_${dataSet}_${parseMode}" driver_label
		ARGUMENTS "${DATASETS_DIR}/${dataSet}" ${parseMode}
		REQUIRED_FILES "${DATASETS_DIR}/${dataSet}"
		Test_OK_SUCCESS
		BACKENDS reference reference_omp bsp1d hybrid nonblocking
	)
endforeach()

# Multiplication kernels tests
add_grb_executables( driver_spmv spmv.cpp
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
	ADDITIONAL_LINK_LIBRARIES test_utils_headers
)
add_grb_executables( driver_spmspv spmspv.cpp
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
	ADDITIONAL_LINK_LIBRARIES test_utils_headers
)
add_grb_executables( driver_spmspm spmspm.cpp
	BACKENDS reference reference_omp hyperdags nonblocking
	ADDITIONAL_LINK_LIBRARIES test_utils_headers
)

add_grb_tests( driver_spmv_west0497_direct driver_spmv
	ARGUMENTS ${kValue} ${DATASETS_DIR}/west0497.mtx direct
	REQUIRED_FILES ${DATASETS_DIR}/west0497.mtx
	Test_OK_SUCCESS
	BACKENDS hyperdags
)
add_grb_tests( driver_spmspv_west0497_direct driver_spmspv
	ARGUMENTS ${kValue} ${DATASETS_DIR}/west0497.mtx direct
	REQUIRED_FILES ${DATASETS_DIR}/west0497.mtx
	Test_OK_SUCCESS
	BACKENDS hyperdags
)
add_grb_tests( driver_spmspm_west0497_direct driver_spmspm
	ARGUMENTS ${kValue} ${DATASETS_DIR}/west0497.mtx ${DATASETS_DIR}/west0497.mtx direct
	REQUIRED_FILES ${DATASETS_DIR}/west0497.mtx
	Test_OK_SUCCESS
	BACKENDS hyperdags
)

set( _index 0 )
foreach( dataSet ${MULTIPLICATION_DATASETS} )
	add_grb_tests( "driver_spmv_${dataSet}_direct" driver_spmv
		ARGUMENTS ${kValue} "${DATASETS_DIR}/${dataSet}" direct
		REQUIRED_FILES "${DATASETS_DIR}/${dataSet}"
		Test_OK_SUCCESS
		BACKENDS reference reference_omp bsp1d hybrid nonblocking
	)

	add_grb_tests( "driver_spmspv_${dataSet}_direct" driver_spmspv
		ARGUMENTS ${kValue} "${DATASETS_DIR}/${dataSet}" direct
		REQUIRED_FILES "${DATASETS_DIR}/${dataSet}"
		Test_OK_SUCCESS
		BACKENDS reference reference_omp bsp1d hybrid nonblocking
	)

	# long-running sparse matrix--sparse matrix multiplications are disabled
	if( _index LESS_EQUAL "14" )
		add_grb_tests( "driver_spmspm_${dataSet}_direct" driver_spmspm
			ARGUMENTS ${kValue} ${DATASETS_DIR}/${dataSet} ${DATASETS_DIR}/${dataSet} direct
			REQUIRED_FILES "${DATASETS_DIR}/${dataSet}"
			Test_OK_SUCCESS
			BACKENDS reference reference_omp nonblocking
		)
	endif()

	# Increment the index
	math( EXPR _index "${_index} + 1" )
endforeach()

# targets to list and build the test for this category
get_property( performance_tests_list GLOBAL PROPERTY tests_category_performance )
add_custom_target( "list_tests_category_performance"
	COMMAND echo "${performance_tests_list}"
	COMMAND_EXPAND_LISTS
)

add_custom_target( "build_tests_category_performance"
	DEPENDS "${performance_tests_list}"
)

# target to run the tests in this category by calling the appropriate runner
add_custom_target( tests_performance
	COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/performancetests.sh
		${SCRIPTS_COMMON_ARGS}
		"--test-bin-dir" "\"${CMAKE_CURRENT_BINARY_DIR}\""
		"--test-out-dir" "\"${CMAKE_CURRENT_BINARY_DIR}/output\""

	WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
	DEPENDS build_tests_category_performance # add dependency on this target
											 # to automatically build before running
	COMMAND_EXPAND_LISTS
	USES_TERMINAL
)

add_custom_target( performancetests DEPENDS tests_performance )
add_custom_target( perftests DEPENDS tests_performance )

