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


#
# [internal!] returns the CMake target name for a test executable and the associated file name,
# implementing the naming conventions of the test suite: all functionalities using these names
# should use this function.
# Arguments:
#
# target_name[out]: name of the variable to store the target name
# exe_name[out]: name of the variable to store the executable name
# test_name: user's name for the test executable
# mode: mode to generate the name for
# backend_name: name of backend (can be empty)
#
# The executable name exe_name is
#   test_<test_name>_<mode>_<backend_name>
# where _<mode> and _<backend_name> may be skipped if the respective strings are empty;
# the corresponding target name is "test_${exe_name}"
#
function( __make_test_name target_name exe_name test_name mode backend )
	set( file "${test_name}${MODES_${mode}_suffix}" )
	if( backend )
		string( APPEND file "-" "${backend}" )
	endif()
	set( ${exe_name} "${file}" PARENT_SCOPE )
	set( ${target_name} "test_${file}" PARENT_SCOPE )
endfunction( __make_test_name )


#
# [internal!] creates a test executable target from passed information, also querying
# the mode(s) defined for the given category
# Arguments:
#
# test_prefix: name of test from the user
# backend_name: name of backend (can be empty)
# sources: source files
# libs: libraries to link (including backend)
# defs: definitions
#
# For each mode in the given category, it generates a target as
#   test_<test_prefix>_<mode>_<backend_name>
# where _<mode> and _<backend_name> may be skipped if the respective strings are empty.
# Similarly, the compiled file name is called as the target, without test_ at the beginning
#
function( __add_grb_executables_with_category test_prefix backend_name sources libs defs )

	assert_valid_variables( TEST_CATEGORY )
	if( NOT "${TEST_CATEGORY}" IN_LIST TEST_CATEGORIES )
		message( FATAL_ERROR "the category ${TEST_CATEGORY} is not among TEST_CATEGORIES: ${TEST_CATEGORIES}" )
	endif()
	set( category "${TEST_CATEGORY}")

	foreach( mode ${MODES_${category}} )

		__make_test_name( full_target_name exe_name "${test_prefix}" "${mode}" "${backend_name}" )
		if( TARGET "${full_target_name}" )
			message( FATAL_ERROR "Target \"${full_target_name}\" already exists!")
		endif()
		add_executable( "${full_target_name}" EXCLUDE_FROM_ALL "${sources}" )

		set_target_properties( "${full_target_name}" PROPERTIES
			OUTPUT_NAME "${exe_name}" # use the bare test name, WITHOUT "test_" at the beginning
		)
		target_link_libraries( "${full_target_name}" PRIVATE "${libs}" )
		target_compile_definitions( "${full_target_name}" PRIVATE "${defs}" )
		target_link_libraries( "${full_target_name}" PRIVATE test_${mode}_flags )
		append_test_to_category( "${category}" "${full_target_name}" )
		if( backend_name )
			append_test_to_backend( "${backend_name}" "${full_target_name}" )
		endif()
	endforeach()
endfunction( __add_grb_executables_with_category )

#
# add a GraphBLAS executable to be compiled against one or more backends: for each backend,
# it generates an executable target name test_<testName>_<mode>_<backend name>
#
# Syntax:
# add_grb_executables( testName source1 [source2 ...]
#   BACKENDS backend1 [backend2...]
#   COMPILE_DEFINITIONS def1 [def2...]
#   ADDITIONAL_LINK_LIBRARIES lib1 [lib2...]
# )
#
# Arguments:
#
# testName: unique name, which is used to generate the test executable target
# source1 [source2 ...]: sources to compile (at least one)
# BACKENDS backend1 [backend2...]: backends to compile the executable against (at least one)
# COMPILE_DEFINITIONS: additional compile definitions
# ADDITIONAL_LINK_LIBRARIES: additional libraries to link to each target
#
# The generated test name is also added to the list of per-backend tests,
# namely tests_backend_<backend name> and to the per-category tests lists,
# namely tests_category_<category>.
#
# The backend name must correspond to one of the backends available in ${ALL_BACKENDS},
# otherwise an error occurs; since not all backends may be enabled, only targets
# to be built against backends stored in ${AVAILABLE_TEST_BACKENDS} are actually built.
#
function( add_grb_executables testName )
	if( NOT testName )
		message( FATAL_ERROR "no test name specified")
	endif()

	set(options "" )
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

	set_valid_string( defs "${parsed_COMPILE_DEFINITIONS}" "" )

	if( "${parsed_BACKENDS}" STREQUAL "none" )
		list( APPEND libs "alp_utils_static" "${parsed_ADDITIONAL_LINK_LIBRARIES}" )
		__add_grb_executables_with_category( "${testName}" ""
			"${parsed_SOURCES}" "${libs}" "${defs}"
		)
		return()
	endif()

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

		__add_grb_executables_with_category( "${testName}" "${back}"
			"${parsed_SOURCES}" "${libs}" "${defs}"
		)
	endforeach()
endfunction( add_grb_executables )
