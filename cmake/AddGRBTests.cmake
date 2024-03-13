#
#   Copyright 2023 Huawei Technologies Co., Ltd.
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
# LOGIC TO ADD TEST CASES based on CTest; this logic adds one test per backend/mode/
# <resource configuration> (= processes/threads)
#

include_guard( GLOBAL )

assert_valid_variables( DATASETS_DIR ALL_BACKENDS AVAILABLE_BACKENDS TEST_CATEGORIES
	TEST_RUNNER Python3_EXECUTABLE MAX_THREADS
)

### GLOBAL CONFIGURATION

# set( CTEST_PARALLEL_LEVEL ${MAX_THREADS} )

set( TEST_PASSED_REGEX "Test OK" )

### BACKEND-SPECIFIC CONFIGURATION

macro( setup_grb_tests_environment )
	# === MANDATORY FIELDS
	set( one_value_args "CATEGORY" )
	set( multi_value_args
		"BSP1D_PROCESSES" "HYBRID_PROCESSES"
		"BSP1D_EXEC_SLOTS" "HYBRID_EXEC_SLOTS"
		"HYBRID_THREADS" "REFERENCE_OMP_THREADS" "NONBLOCKING_THREADS"
	)
	cmake_parse_arguments( parsed "" "${one_value_args}" "${multi_value_args}" "${ARGN}" )
	# === MANDATORY FIELDS

	set_if_var_valid( TEST_CATEGORY "${parsed_CATEGORY}" )
	set( TEST_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/output" )
	file( MAKE_DIRECTORY "${TEST_OUTPUT_DIR}" )
	if( ${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.15.0" )
		# recent versions of CMake allow cleaning custom files/directories
		# as part of the 'make clean' command; the ${TEST_OUTPUT_DIR} directory may grow
		# pretty big due to all the test logs, so it makes sense to clean it as well
		get_directory_property( to_clean ADDITIONAL_CLEAN_FILES )
		list( APPEND to_clean "${TEST_OUTPUT_DIR}" )
		set_directory_properties( PROPERTIES ADDITIONAL_CLEAN_FILES "${to_clean}" )
	endif()

	if( NOT CMAKE_BUILD_TYPE STREQUAL Coverage )
		set_valid_string( bsp1d_processes "${parsed_BSP1D_PROCESSES}" "1" )
		set_valid_string( hybrid_processes "${parsed_HYBRID_PROCESSES}" "1" )

		set_valid_string( bsp1d_exec_resource "${parsed_BSP1D_EXEC_SLOTS}" "1" )
		set_valid_string( hybrid_exec_resource "${parsed_HYBRID_EXEC_SLOTS}" "1" )

		set_valid_string( hybrid_threads "${parsed_HYBRID_THREADS}" "1" )
		set_valid_string( reference_omp_threads "${parsed_REFERENCE_OMP_THREADS}" "1" )
		set_valid_string( nonblocking_threads "${parsed_NONBLOCKING_THREADS}" "1" )
	else()
		# for coverage-built binaries, successive runs of the same binary overwrite
		# previous coverage information; hence it is useless to run the same binary multiple times,
		# we run it just once with maximum resources; this assumes the coverage does NOT depend
		# on the number of resources (e.g., branches depending on the number of threads);
		# code paths against this assumption are expected to be very rare,
		# hence negligible w.r.t. coverage
		set( reference_omp_threads "${MAX_THREADS}" )
		set( nonblocking_threads "${MAX_THREADS}" )
		set( bsp1d_processes "${MAX_THREADS}" )

		set( hybrid_processes "7" )
		math( EXPR hybrid_threads "${MAX_THREADS}/${hybrid_processes}" OUTPUT_FORMAT DECIMAL )
	endif()
endmacro()


set( RUNNER_COMMAND ${Python3_EXECUTABLE} ${TEST_RUNNER} )

# sets into var the first valid value between 2 (can also be a list)
macro( __set_valid_resource2 var res1 res2 )
	set( ${var} ${res1} )
	if( NOT ${var} )
		set( ${var} ${res2} )
	endif()
endmacro()

# sets into var the first valid value among 3 (can also be a list)
macro( __set_valid_resource3 var res1 res2 res3 )
	set( ${var} ${res1} )
	if( NOT ${var} )
		set( ${var} ${res2} )
	endif()
	if( NOT ${var} )
		set( ${var} ${res3} )
	endif()
endmacro()


#
# creates a fixture test for the validation of a test output.
# Arguments:
#
# full_test_name: name of the test
# __command: the command to run (can be any Bash command, also with piping)
# outfile: absolute path to test output file
# mainTestFixtures: variable name with list of test fixtures
#
# macro( __set_validation_test mainTest mainTestFixtures outfile validationTest validationExe )
# 	add_test( NAME ${validationTest} COMMAND ${BASH_RUNNER} ${validationExe} ${ARGN}
macro( __set_validation_test full_test_name __command outfile mainTestFixtures )

	set( __validation_test_name "${full_test_name}-validate" )
	set( __validate_out "${TEST_OUTPUT_DIR}/${__validation_test_name}-output.log" )

	string( REPLACE "@@TEST_OUTPUT_FILE@@" "${outfile}" __validate_command "${__command}" )
	add_test( NAME ${__validation_test_name} COMMAND ${BASH_RUNNER} "${outfile}" "${__validate_out}"
		${__validate_command}
		WORKING_DIRECTORY "${PROJECT_BINARY_DIR}"
	)
	set_tests_properties( ${__validation_test_name} PROPERTIES REQUIRED_FILES ${outfile} )

	#  __validation_test_name must run after the test, i.e., as a cleanup fixture
	set( validate_fixture_name "${__validation_test_name}_fixture" )
	set_tests_properties( ${__validation_test_name} PROPERTIES FIXTURES_CLEANUP "${validate_fixture_name}" )
	list( APPEND ${mainTestFixtures} "${validate_fixture_name}" )
endmacro( __set_validation_test )

#
# adds a single test, configuring all of its features (resources, file dependencies, fixtures)
# Arguments:
#
# full_test_name: final full name of the test (for CTest)
# full_target_name: name of the executable target to run
# runner_backend: backend the test should run against (cannot be empty)
# mode: mode of the test
# num_procs: number of processes (cannot be empty)
# num_threads: number of threads (cannot be empty)
# test_OK_SUCCESS: whether to look for `Test OK` in the test output (boolean)
# required_files: files required to run the test (typically datasets) (can be empty)
# output_validate_command: command to validate the output (can be empty)
#
function(  __do_add_single_test full_test_name full_target_name runner_backend mode num_procs num_threads
	parallel_processes arguments test_OK_SUCCESS required_files output_validate_command
)
	set( runner_config_cmdline "--backend" "${runner_backend}" "--processes" "${num_procs}" "--threads" "${num_threads}" )
	set( exe_file "$<TARGET_FILE:${full_target_name}>" )
	set( stdout_file "${TEST_OUTPUT_DIR}/${full_test_name}-output.log" )
	set( ok_success_test )
	if ( test_OK_SUCCESS )
		set( ok_success_test "--success-string" "${TEST_PASSED_REGEX}" )
	endif()

	if( parallel_processes )
		set( parallel_opts "--parallel-instance" "${parallel_processes}" )
	endif()
	# if( arguments )
	# 	set( __args "--args" ${arguments} )
	# endif()

	add_test( NAME ${full_test_name} COMMAND ${RUNNER_COMMAND} ${runner_config_cmdline} ${ok_success_test}
		"--output" ${stdout_file} ${parallel_opts} ${exe_file} ${arguments}
		WORKING_DIRECTORY "${PROJECT_BINARY_DIR}"
	)
	set_tests_properties( ${full_test_name} PROPERTIES RESOURCE_LOCK "${${runner_backend}_exec_resource}" )

	if( required_files )
		set_tests_properties( ${full_test_name} PROPERTIES REQUIRED_FILES "${required_files}" )
	endif()

	if( output_validate_command )
		__set_validation_test( "${full_test_name}" "${output_validate_command}" ${stdout_file} fixtures )
	endif()

	set_tests_properties( ${full_test_name} PROPERTIES FIXTURES_REQUIRED "${fixtures}" )
	# compute total number of used threads, to run in parallel via "ctest -j<num>"
	math( EXPR total_threads "${num_procs}*${num_threads}" OUTPUT_FORMAT DECIMAL )
	if( total_threads GREATER "${MAX_THREADS}" )
		set( total_threads "${MAX_THREADS}" )
	endif()
	set_tests_properties( ${full_test_name} PROPERTIES PROCESSORS "${total_threads}" )
	set_tests_properties( ${full_test_name} PROPERTIES LABELS "mode:${mode};backend:${runner_backend}" )
endfunction()

#
# adds multiple tests against one or more backends: for each backend/mode/resources it generates a test
#
# Syntax:
# add_grb_tests( test_name target_name source1 [source2 ...]
#    BACKENDS backend1 [backend2 ...]
#    [Test_OK_SUCCESS]
#    [ARGUMENTS] arg1 [arg2 ...]
#    [REQUIRED_FILES] file1 [file2 ...]
#    [PROCESSES] p1 [p2 ...]
#    [THREADS] t1 [t2 ...]
#    [OUTPUT_VALIDATE] <Bash command>
# )
#
# Arguments:
#
# BACKENDS backend1 [backend2 ...]: backends to run the test against, can also be "none"
# [Test_OK_SUCCESS]: (optional) whether to look for `Test OK` in the test output (boolean)
# [ARGUMENTS] arg1 [arg2 ...]: (optional) arguments to pass to the executable
# [REQUIRED_FILES] file1 [file2 ...]: (optional) files required to run the test (e.g., datasets)
# [PROCESSES] p1 [p2 ...]: (optional) number of processes, can also be a list; if none, the default for each backend applies
# [THREADS] t1 [t2 ...]: (optional) number of threads, can also be a list; if none, the default for each backend applies
# [OUTPUT_VALIDATE] <Bash command>: (optional) Bash command to validate the output; 0 return code means success
#
function( add_grb_tests test_name target_name )
	assert_valid_variables( test_name )
	assert_valid_variables( target_name )
	assert_in_list( TEST_CATEGORY TEST_CATEGORIES )
	assert_valid_variables( TEST_OUTPUT_DIR )

	set( options "Test_OK_SUCCESS" )
	set( oneValueArgs "PARALLEL_PROCESSES" )
	set( multiValueArgs "ARGUMENTS" "BACKENDS" "REQUIRED_FILES"
		"PROCESSES" "THREADS" "OUTPUT_VALIDATE"
	)
	cmake_parse_arguments( parsed "${options}"
		"${oneValueArgs}" "${multiValueArgs}" "${ARGN}"
	)

	if( NOT __modes )
		set( __modes "${MODES_${TEST_CATEGORY}}" )
	endif()
	assert_valid_variables( parsed_BACKENDS )

	if( DEFINED parsed_OUTPUT_VALIDATE AND NOT parsed_OUTPUT_VALIDATE )
		message( FATAL_ERROR "OUTPUT_VALIDATE defined but empty" )
	endif()

	# special case for "none" backend: generate variables accordingly
	if( parsed_BACKENDS STREQUAL "none" )
		foreach( mode ${__modes} )
			__make_test_name( full_target_name __file_name "${target_name}" "${mode}" "" )
			assert_defined_targets( ${__target_name} )
			__set_valid_resource2( num_procs "${parsed_PROCESSES}" "1" )
			__set_valid_resource2( num_threads "${parsed_THREADS}" "1" )

			__do_add_single_test(
				"${test_name}${MODES_${mode}_suffix}-processes:${num_procs},threads:${num_threads}"
				"${full_target_name}"
				"none"
				"${mode}"
				"${num_procs}"
				"${num_threads}"
				"${parsed_PARALLEL_PROCESSES}"
				"${parsed_ARGUMENTS}"
				"${parsed_Test_OK_SUCCESS}"
				"${parsed_REQUIRED_FILES}"
				"${parsed_OUTPUT_VALIDATE}"
			)
		endforeach()
		return()
	endif()

	# generic case: all possible backends, modes, resources
	foreach( backend ${parsed_BACKENDS} )
		if( NOT ${backend} IN_LIST ALL_BACKENDS  )
			message( FATAL_ERROR "no backend named ${backend}; existing backends are ${ALL_BACKENDS}")
		endif()
		if( NOT ${backend} IN_LIST AVAILABLE_TEST_BACKENDS )
			continue()
		endif()

		__set_valid_resource3( __procs "${parsed_PROCESSES}" "${${backend}_processes}" "1" )
		__set_valid_resource3( __threads "${parsed_THREADS}" "${${backend}_threads}" "1" )
		foreach( mode ${__modes} )
			__make_test_name( full_target_name __exe_name "${target_name}" "${mode}" "${backend}" )
			assert_defined_targets( ${full_target_name} )

			foreach( num_procs ${__procs} )
				foreach( num_threads ${__threads} )
					__do_add_single_test(
						"${test_name}${MODES_${mode}_suffix}-${backend}-processes:${num_procs},threads:${num_threads}"
						"${full_target_name}" "${backend}" "${mode}" "${num_procs}" "${num_threads}"
						"${parsed_PARALLEL_PROCESSES}" "${parsed_ARGUMENTS}" "${parsed_Test_OK_SUCCESS}"
						"${parsed_REQUIRED_FILES}" "${parsed_OUTPUT_VALIDATE}"
					)
				endforeach()
			endforeach()
		endforeach()
	endforeach()
endfunction( add_grb_tests )
