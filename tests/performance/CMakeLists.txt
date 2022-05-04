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
set( TEST_CATEGORY "performance" )


# list tests, without the CATEROR[IES,Y] keyword (it's now passed via TEST_CATEGORY)

### Kernels tests
add_library( bench_kernels OBJECT bench_kernels.c bench_kernels.h )
add_library( bench_kernels_omp OBJECT bench_kernels.c bench_kernels.h )
target_compile_definitions( bench_kernels_omp PRIVATE BENCH_KERNELS_OPENMP )

add_grb_executables( fma fma.cpp $<TARGET_OBJECTS:bench_kernels>
	BACKENDS reference NO_BACKEND_NAME
	ADDITIONAL_LINK_LIBRARIES "rt"
)

add_grb_executables( fma-openmp fma.cpp $<TARGET_OBJECTS:bench_kernels_omp>
	BACKENDS reference_omp NO_BACKEND_NAME
	ADDITIONAL_LINK_LIBRARIES OpenMP::OpenMP_CXX "rt"
)

add_grb_executables( reduce reduce.cpp $<TARGET_OBJECTS:bench_kernels>
	BACKENDS reference NO_BACKEND_NAME
)

add_grb_executables( reduce-openmp reduce.cpp $<TARGET_OBJECTS:bench_kernels_omp>
	BACKENDS reference_omp NO_BACKEND_NAME
	ADDITIONAL_LINK_LIBRARIES OpenMP::OpenMP_CXX
)

add_grb_executables( dot dot.cpp $<TARGET_OBJECTS:bench_kernels>
	BACKENDS reference NO_BACKEND_NAME
	ADDITIONAL_LINK_LIBRARIES backend_headers_nodefs
)

add_grb_executables( dot-openmp dot.cpp $<TARGET_OBJECTS:bench_kernels_omp>
	BACKENDS reference_omp NO_BACKEND_NAME
	ADDITIONAL_LINK_LIBRARIES backend_headers_nodefs OpenMP::OpenMP_CXX
)

add_grb_executables( scaling scaling.cpp
	../unit/parser.cpp
	BACKENDS reference reference_omp bsp1d hybrid hyperdags
)

add_grb_executables( driver_knn ../smoke/knn.cpp
	../unit/parser.cpp
	BACKENDS reference reference_omp bsp1d hybrid hyperdags
)

add_grb_executables( driver_simple_pagerank
	../smoke/simple_pagerank.cpp
	BACKENDS reference reference_omp bsp1d hybrid hyperdags
	ADDITIONAL_LINK_LIBRARIES test_utils_headers
)

add_grb_executables( driver_label ../smoke/label.cpp
	../unit/parser.cpp
	BACKENDS reference reference_omp bsp1d hybrid hyperdags
	ADDITIONAL_LINK_LIBRARIES test_utils_headers
)

add_grb_executables( driver_spmv spmv.cpp
	BACKENDS reference reference_omp bsp1d hybrid
	ADDITIONAL_LINK_LIBRARIES test_utils_headers
)

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

