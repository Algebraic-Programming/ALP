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
# functions to add GraphBLAs tests linked against the given backend
#

# protection against double inclusion
include_guard( GLOBAL )

assert_valid_variables( ALL_BACKENDS AVAILABLE_TEST_BACKENDS TEST_CATEGORIES
	#TESTS_EXE_OUTPUT_DIR
	ALP_UTILS_LIBRARY_OUTPUT_NAME )


# create variables to store tests against each backend
foreach( b ${AVAILABLE_TEST_BACKENDS} )
	define_property( GLOBAL PROPERTY tests_backend_${b} BRIEF_DOCS "${b} tests" FULL_DOCS "tests for backend ${b}" )
endforeach()

foreach( c ${TEST_CATEGORIES} )
	define_property( GLOBAL PROPERTY tests_category_${c} BRIEF_DOCS "${c} tests" FULL_DOCS "tests for category ${c}" )
	assert_valid_variables( MODES_${c}  )
	foreach( p ${MODES_${c}} )
		assert_defined_targets( test_${p}_flags )
		assert_defined_variables( MODES_${p}_suffix )
	endforeach()
endforeach()

# append var to the list named listName with description description
macro( append_test_to_category category test )
	set_property( GLOBAL APPEND PROPERTY tests_category_${category} "${test}" )
endmacro()

macro( append_test_to_backend backend test )
	set_property( GLOBAL APPEND PROPERTY tests_backend_${backend} "${test}" )
endmacro()

macro( make_test_names target_out_var filename_out_var test_name backend mode  )
	set( ${filename_out_var} "${test_name}${MODES_${mode}_suffix}" )
	set( ${target_out_var} "test_${${filename_out_var}}" )
	set( __back "${backend}" )
	if( __back )
		string( APPEND ${filename_out_var} "_" "${__back}" )
		string( APPEND ${target_out_var} "_" "${__back}" )
	endif()
endmacro( make_test_names )


#
# [internal!] creates a test target from passed information, also querying the mode(s)
# defined for the given category
# Arguments:
#
# test_prefix name of test from the user
# backend_name name of backend (mandatory, even if not used for the file and target name)
# suffix file name and category name suffix, either empty or with _<backend name>
# sources source files
# libs libraries to link (including backend)
# defs definitions
# no_perf_opt whether to exclude performance optimizations
#
# For each mode in the given category, it generates a target as
#   test_<test_prefix>_<mode>_<suffix>
# where _<mode> and _<suffix> may be skipped if the respective strings are empty.
# Similarly, the compiled file name is called as the target, without test_ at the beginning
#
macro( add_grb_executables_with_category test_prefix backend_name suffix sources libs defs )

	if( NOT TEST_CATEGORY )
		message( FATAL_ERROR "variable TEST_CATEGORY not specified" )
	endif()
	if( NOT "${TEST_CATEGORY}" IN_LIST TEST_CATEGORIES )
		message( FATAL_ERROR "the category ${TEST_CATEGORY} is not among TEST_CATEGORIES: ${TEST_CATEGORIES}" )
	endif()
	set( category "${TEST_CATEGORY}")

	foreach( mode ${MODES_${category}} )

		# set( __file_name "${test_prefix}${MODES_${mode}_suffix}" )
		# set( __target_name "test_${__file_name}" )
		# #set( __target_name "test_${test_prefix}_${category}${MODES_${mode}_suffix}" )
		# if( suffix )
		# 	string( APPEND __file_name "_" "${suffix}" )
		# 	string( APPEND __target_name "_" "${suffix}" )
		# endif()
		# message( AUTHOR_WARNING "invoking
		make_test_names( __target_name __file_name "${test_prefix}" "${suffix}" "${mode}" )

		if( TARGET "${__target_name}" )
			message( FATAL_ERROR "Target \"${__target_name}\" already exists!")
		endif()
		# message( AUTHOR_WARNING "gor target name ${__target_name}" )
		add_executable( "${__target_name}" EXCLUDE_FROM_ALL "${sources}" )

		set_target_properties( "${__target_name}" PROPERTIES
			#RUNTIME_OUTPUT_DIRECTORY "${TESTS_EXE_OUTPUT_DIR}"
			OUTPUT_NAME "${__file_name}" # use the bare test name, WITHOUT "test_" at the beginning
		)
		target_link_libraries( "${__target_name}" PRIVATE "${libs}" )
		target_compile_definitions( "${__target_name}" PRIVATE "${defs}" )
		target_link_libraries( "${__target_name}" PRIVATE test_${mode}_flags )
		append_test_to_category( "${category}" "${__target_name}" )
		set( __b "${backend_name}" )
		if( __b )
			append_test_to_backend( "${__b}" "${__target_name}" )
		endif()
	endforeach()
endmacro( add_grb_executables_with_category )

#
# add a GraphBLAS test to be compiled against one or more backends: for each backend,
# it generates an executable target name test_<testName>_<backend name>
#
# Syntax:
# add_grb_tests( testName source1 [source2 ...]
#   BACKENDS backend1 [backend2...] [NO_BACKEND_NAME]
#   COMPILE_DEFINITIONS def1 [def2...]
#   ADDITIONAL_LINK_LIBRARIES lib1 [lib2...]
#   CATEGORIES cat1 [cat2...]
# )
#
# NO_BACKEND_NAME: if one only backend is selected, do not put its name
# at the end of the test name
# COMPILE_DEFINITIONS: additional compile definitions
# ADDITIONAL_LINK_LIBRARIES: additional libraries to link to each target
#
# The generated test name is also added to the list of per-backend tests,
# namely tests_backend_<backend name> and is also added to the per-category
# tests lists, namely tests_category_<category>.
#
# The backend name must correspond to one of the backends available in ${ALL_BACKENDS},
# otherwise an error occurs; since not all backends may be enabled, only targets
# to be built against backends stored in ${AVAILABLE_TEST_BACKENDS} are actually built.
#
function( add_grb_executables testName )
	if( NOT testName )
		message( FATAL_ERROR "no test name specified")
	endif()

	set(options "NO_BACKEND_NAME" )
	set(oneValueArgs "" )
	set(multiValueArgs
		"SOURCES"
		"BACKENDS"
		"COMPILE_DEFINITIONS"
		"ADDITIONAL_LINK_LIBRARIES"
	)

	set( args "SOURCES" "${ARGN}" )
	cmake_parse_arguments( parsed "${options}"
		"${oneValueArgs}" "${multiValueArgs}" ${args}
	)

	assert_valid_variables( parsed_SOURCES parsed_BACKENDS )

	list( LENGTH parsed_BACKENDS num_backends )
	if( parsed_NO_BACKEND_NAME AND ( NOT num_backends EQUAL "1" ) )
		message( FATAL_ERROR "NO_BACKEND_NAME can be used only with one backend listed")
	endif()

	set_valid( defs "${parsed_COMPILE_DEFINITIONS}" "" )

	foreach( back ${parsed_BACKENDS} )
		if( NOT ${back} IN_LIST AVAILABLE_TEST_BACKENDS )
			continue()
		endif()
		if( NOT ${back} IN_LIST ALL_BACKENDS  )
			message( FATAL_ERROR "no backend named ${back}; existing backends are ${ALL_BACKENDS}")
		endif()
		assert_defined_targets( backend_${back} )

		set( libs "backend_${back};alp_utils_static" )
		append_if_valid( libs "${parsed_ADDITIONAL_LINK_LIBRARIES}" )

		if( NOT parsed_NO_BACKEND_NAME )
			set( suffix "${back}" )
		endif()

		add_grb_executables_with_category( "${testName}" "${back}" "${suffix}"
			"${parsed_SOURCES}" "${libs}" "${defs}"
		)
	endforeach()
endfunction( add_grb_executables )

# force add executable even if the test backend is not enabled
# useful for tests that produce a "golden output" for other tests;
# for one backend only
function( force_add_grb_executable testName )
	if( NOT testName )
		message( FATAL_ERROR "no test name specified")
	endif()

	set(options "" )
	set(oneValueArgs "BACKEND" )
	set(multiValueArgs
		"SOURCES"
		"COMPILE_DEFINITIONS"
		"ADDITIONAL_LINK_LIBRARIES"
	)

	set( args "SOURCES" "${ARGN}" )
	cmake_parse_arguments( parsed "${options}"
		"${oneValueArgs}" "${multiValueArgs}" ${args}
	)

	assert_valid_variables( parsed_SOURCES parsed_BACKEND )

	set_valid( defs "${parsed_COMPILE_DEFINITIONS}" "" )

	if( NOT "${parsed_BACKEND}" IN_LIST ALL_BACKENDS  )
		return()
	endif()
	assert_defined_targets( backend_${parsed_BACKEND} )

	set( libs "backend_${parsed_BACKEND};alp_utils_static" )
	append_if_valid( libs "${parsed_ADDITIONAL_LINK_LIBRARIES}" )

	__add_test_with_category( "${testName}" "${parsed_BACKEND}" "${parsed_BACKEND}"
		"${parsed_SOURCES}" "${libs}" "${defs}"
	)
endfunction()

