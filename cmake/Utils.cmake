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
# UTILITY MACROS
# defines various utilities used across the entire build infrastructure
#

# asserts that the variable named dirPathVer is set:
# if it is set to a valid directory path, set it to its absolute path, if not set it to defValue
# if the given path does not exist or is not a directory, raise an error
macro( assert_valid_directory dirPathVer defValue )
	if( NOT ${dirPathVer} )
		# if invalid, tests will just skip
		set( ${dirPathVer} "${defValue}" )
		message( STATUS "Setting the datasets directory ${dirPathVer} to \"${${dirPathVer}}\"")
	else()
		# get absolute path referred to project binary root
		get_filename_component( datasets_path "${${dirPathVer}}" ABSOLUTE BASE_DIR "${PROJECT_BINARY_DIR}" )
		set( ${dirPathVer} "${datasets_path}" )
		if ( NOT EXISTS "${${dirPathVer}}" OR NOT IS_DIRECTORY "${${dirPathVer}}" )
			message( FATAL_ERROR "${${dirPathVer}} does not exist or is not a directory")
		endif()
		message( STATUS "Datasets directory ${dirPathVer} set to \"${${dirPathVer}}\"")
	endif()
endmacro()

# asserts that each passed variable is valid according to CMake rules
# in https://cmake.org/cmake/help/latest/command/if.html#basic-expressions
macro( assert_valid_variables )
	foreach( var ${ARGN} )
		if( NOT ${var} )
			message( FATAL_ERROR "The variable \"${var}\" is not valid" )
		endif()
	endforeach()
endmacro()

# asserts that each passed variable has been defined (can also be empty or false)
macro( assert_defined_variables )
	foreach( var ${ARGN} )
		if( NOT DEFINED ${var} )
			message( FATAL_ERROR "The variable \"${var}\" should be defined" )
		endif()
	endforeach()
endmacro()

macro( assert_defined_targets )
	foreach( var ${ARGN} )
		if( NOT TARGET ${var} )
			message( FATAL_ERROR "The target \"${var}\" should be defined" )
		endif()
	endforeach()
endmacro()

# creates a variable ${out_name} with
# * the content of str ONLY var is not valid
# * the content of var appended to that of str if var is valid
macro( append_if_valid out_name )
	foreach( __v ${ARGN} )
		if( __v )
			list( APPEND ${out_name} "${__v}" )
		endif()
	endforeach()
endmacro( append_if_valid )

# set first string if valid, otherwise second
function( set_valid out_name first second )
	if( first )
		set( ${out_name} "${first}" PARENT_SCOPE )
	else()
		set( ${out_name} "${second}" PARENT_SCOPE )
	endif()
endfunction( set_valid )

# set first string if valid, otherwise second
function( set_valid_string out_name first second )
	if( first )
		set( ${out_name} "${first}" PARENT_SCOPE )
	else()
		set( ${out_name} "${second}" PARENT_SCOPE )
	endif()
endfunction( set_valid_string )
