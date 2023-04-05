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
# Generic compilation flags
#
# Compilation flags are ALWAYS passed to consuming targets (backends, tests)
# VIA THE TARGETS HERE DEFINED and never via strings, to minimize the amount of
# information and dependencies between CMake files; strings defined here follow
# certain patterns and are to be used ONLY HERE
#
# Flags for specific dependencies (e.g., NUMA or OpenMP) are not handled here,
# but in the corresponding include and library targets where they are actually
# used (usually in include/CMakeLists.txt, since ALP/GraphBLAS is template-based)
#

assert_valid_variables( TEST_CATEGORIES )
assert_defined_variables( 
	COMMON_COMPILE_DEFINITIONS COMMON_COMPILE_OPTIONS COMMON_COMPILE_LIBRARIES 
	WITH_NUMA ADDITIONAL_BACKEND_DEFINITIONS ADDITIONAL_BACKEND_OPTIONS 
	ADDITIONAL_TEST_DEFINITIONS ADDITIONAL_TEST_OPTIONS
	TEST_PERFORMANCE_DEFINITIONS TEST_PERFORMANCE_OPTIONS
)

# allow only Relase, Debug and Coverage
set( CMAKE_CONFIGURATION_TYPES "Release;Debug;Coverage" CACHE STRING
	"Add the configurations that we need" FORCE
)

### COMMMON COMPILATION FLAGS

if( NOT CMAKE_BUILD_TYPE )
	# if no build type, set Release
	set( CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE )
else()
	if( CMAKE_BUILD_TYPE AND( NOT CMAKE_BUILD_TYPE IN_LIST CMAKE_CONFIGURATION_TYPES ) )
		message( SEND_ERROR "Built type \"${CMAKE_BUILD_TYPE}\" unrecognized" )
		message( FATAL_ERROR "CMAKE_BUILD_TYPE must be one of: \"${CMAKE_CONFIGURATION_TYPES}\"" )
	endif()
	# clear CMake defaults
	foreach( bt ${CMAKE_CONFIGURATION_TYPES} )
		string( TOUPPER ${bt} btu )
		set( CMAKE_CXX_FLAGS_${btu} "" CACHE STRING "${CMAKE_BUILD_TYPE} CXX compilation flags" FORCE )
		set( CMAKE_C_FLAGS_${btu} "" CACHE STRING "${CMAKE_BUILD_TYPE} C compilation flags" FORCE )
	endforeach()
endif()

set( COMMON_OPTS "-g" "-Wall" "-Wextra" )

# cache variable to allow manual tweaks from CMake cache
set_valid_cache_string( COMMON_DEFS_Release "${COMMON_COMPILE_DEFINITIONS}" 
	"" "common Release definitions"
)
set_valid_cache_string( COMMON_DEFS_Debug "${COMMON_COMPILE_DEFINITIONS}" 
	"" "common Debug definitions"
)
set_valid_cache_string( COMMON_DEFS_Coverage "${COMMON_COMPILE_DEFINITIONS}" 
	"" "common Coverage definitions"
)
set_valid_cache_string( COMMON_OPTS_Release "${COMMON_COMPILE_OPTIONS}" 
	"${COMMON_OPTS}" "common Release options"
)
set_valid_cache_string( COMMON_OPTS_Debug "${COMMON_COMPILE_OPTIONS}"
	"${COMMON_OPTS};-fno-omit-frame-pointer" "common Debug options"
)
set_valid_cache_string( COMMON_OPTS_Coverage "${COMMON_COMPILE_OPTIONS}"
	"${COMMON_OPTS};-fprofile-arcs;-ftest-coverage" "common Coverage options" 
)
set_valid_cache_string( COMMON_LIBS_Release "${COMMON_COMPILE_LIBRARIES}" 
	"${COMMON_LIBS}" "common Release libraries"
)
set_valid_cache_string( COMMON_LIBS_Debug "${COMMON_COMPILE_LIBRARIES}"
	"${COMMON_LIBS}" "common Debug libraries"
)
set_valid_cache_string( COMMON_LIBS_Coverage "${COMMON_COMPILE_LIBRARIES}"
	"${COMMON_LIBS};gcov" "common Coverage libraries" 
)

add_library( common_flags INTERFACE )
target_compile_definitions( common_flags INTERFACE
	"$<$<CONFIG:Release>:${COMMON_DEFS_Release}>"
	"$<$<CONFIG:Debug>:${COMMON_DEFS_Debug}>"
	"$<$<CONFIG:Coverage>:${COMMON_DEFS_Coverage}>"
)
target_compile_options( common_flags INTERFACE
	"$<$<CONFIG:Release>:${COMMON_OPTS_Release}>"
	"$<$<CONFIG:Debug>:${COMMON_OPTS_Debug}>"
	"$<$<CONFIG:Coverage>:${COMMON_OPTS_Coverage}>"
)
target_link_libraries( common_flags INTERFACE
	"$<$<CONFIG:Release>:${COMMON_LIBS_Release}>"
	"$<$<CONFIG:Debug>:${COMMON_LIBS_Debug}>"
	"$<$<CONFIG:Coverage>:${COMMON_LIBS_Coverage}>"
)

## defaults performance options for all targets (backends and tests)

set( COMMON_PERF_DEFS_Release "NDEBUG" )
set( COMMON_PERF_OPTS_Release "-O3" "-march=native" "-mtune=native" "-funroll-loops" )
set( COMMON_PERF_DEFS_Debug "" )
set( COMMON_PERF_OPTS_Debug "-O0" )
set( COMMON_PERF_DEFS_Coverage "" )
set( COMMON_PERF_OPTS_Coverage "-O0")

### COMPILATION FLAGS FOR BACKENDS

append_if_valid( _BACKEND_DEFS_Release "${COMMON_PERF_DEFS_Release}" "${ADDITIONAL_BACKEND_DEFINITIONS}" )
append_if_valid( _BACKEND_DEFS_Debug "${COMMON_PERF_DEFS_Debug}" "${ADDITIONAL_BACKEND_DEFINITIONS}" )
append_if_valid( _BACKEND_DEFS_Coverage "${COMMON_PERF_DEFS_Coverage}" "${ADDITIONAL_BACKEND_DEFINITIONS}" )
append_if_valid( _BACKEND_OPTS_Release "${COMMON_PERF_OPTS_Release}" "${ADDITIONAL_BACKEND_OPTIONS}" )
append_if_valid( _BACKEND_OPTS_Debug "${COMMON_PERF_OPTS_Debug}" "${ADDITIONAL_BACKEND_OPTIONS}" )
append_if_valid( _BACKEND_OPTS_Coverage "${COMMON_PERF_OPTS_Coverage}" "${ADDITIONAL_BACKEND_OPTIONS}" )

# cache variable to allow manual tweaks form CMake cache
set( BACKEND_DEFS_Release "${_BACKEND_DEFS_Release}" CACHE STRING "backend Release definitions" )
set( BACKEND_DEFS_Debug "${_BACKEND_DEFS_Debug}" CACHE STRING "backend Debug definitions" )
set( BACKEND_DEFS_Coverage "${_BACKEND_DEFS_Coverage}" CACHE STRING "backend Coverage definitions" )
set( BACKEND_OPTS_Release "${_BACKEND_OPTS_Release}" CACHE STRING "backend Release options" )
set( BACKEND_OPTS_Debug "${_BACKEND_OPTS_Debug}" CACHE STRING "backend Debug options" )
set( BACKEND_OPTS_Coverage "${_BACKEND_OPTS_Coverage}" CACHE STRING "backend Coverage options" )

add_library( backend_flags INTERFACE )
target_compile_definitions( backend_flags INTERFACE
	"$<$<CONFIG:Release>:${BACKEND_DEFS_Release}>"
	"$<$<CONFIG:Debug>:${BACKEND_DEFS_Debug}>"
	"$<$<CONFIG:Coverage>:${BACKEND_DEFS_Coverage}>"
)
target_compile_options( backend_flags INTERFACE
	"$<$<CONFIG:Release>:${BACKEND_OPTS_Release}>"
	"$<$<CONFIG:Debug>:${BACKEND_OPTS_Debug}>"
	"$<$<CONFIG:Coverage>:${BACKEND_OPTS_Coverage}>"
)
target_link_libraries( backend_flags INTERFACE common_flags )

install( TARGETS common_flags backend_flags
	EXPORT GraphBLASTargets
)

### COMPILATION FLAGS FOR TESTS

# pattern for variables ({} for mandatory choice, [] for optional parts):
#   TEST_{DEFAULT,category[_mode]}[_PERF]_{DEFS,OPTS}_{<build type>}
# corresponding pattern for targets:
#   test_{default,category[_mode]}[_perf]_flags

# cache variable to allow manual tweaks form CMake cache
set_valid_cache_string( TEST_DEFAULT_DEFS_Release "${ADDITIONAL_TEST_DEFINITIONS}" ""
	"default Release definitions for tests"
)
set_valid_cache_string( TEST_DEFAULT_DEFS_Debug "${ADDITIONAL_TEST_DEFINITIONS}" ""
	"default Debug definitions for tests"
)
set_valid_cache_string( TEST_DEFAULT_DEFS_Coverage "${ADDITIONAL_TEST_DEFINITIONS}" ""
	"default Coverage definitions for tests"
)
set_valid_cache_string( TEST_DEFAULT_OPTS_Release "${ADDITIONAL_TEST_OPTIONS}" ""
	"default Release options for tests"
)
set_valid_cache_string( TEST_DEFAULT_OPTS_Debug "${ADDITIONAL_TEST_OPTIONS}" ""
	"default Debug options for tests"
)
set_valid_cache_string( TEST_DEFAULT_OPTS_Coverage "${ADDITIONAL_TEST_OPTIONS}" ""
	"default Coverage options for tests"
)

set_valid_cache_string( TEST_DEFAULT_PERF_DEFS_Release "${TEST_PERFORMANCE_DEFINITIONS}"
	"${COMMON_PERF_DEFS_Release}" "default performance Release definitions for tests"
)
set_valid_cache_string( TEST_DEFAULT_PERF_DEFS_Debug "${TEST_PERFORMANCE_DEFINITIONS}"
	"${COMMON_PERF_DEFS_Debug}" "default performance Debug definitions for tests"
)
set_valid_cache_string( TEST_DEFAULT_PERF_DEFS_Coverage "${TEST_PERFORMANCE_DEFINITIONS}"
	"${COMMON_PERF_DEFS_Coverage}" "default performance Coverage definitions for tests"
)
set_valid_cache_string( TEST_DEFAULT_PERF_OPTS_Release "${TEST_PERFORMANCE_OPTIONS}"
	"${COMMON_PERF_OPTS_Release}" "default performance Release options for tests"
)
set_valid_cache_string( TEST_DEFAULT_PERF_OPTS_Debug "${TEST_PERFORMANCE_OPTIONS}"
	"${COMMON_PERF_OPTS_Debug}" "default performance Debug options for tests"
)
set_valid_cache_string( TEST_DEFAULT_PERF_OPTS_Coverage "${TEST_PERFORMANCE_OPTIONS}"
	"${COMMON_PERF_OPTS_Coverage}" "default performance Coverage options for tests"
)

add_library( test_default_flags INTERFACE )
target_link_libraries( test_default_flags INTERFACE common_flags )
target_compile_definitions( test_default_flags INTERFACE
	"$<$<CONFIG:Release>:${TEST_DEFAULT_DEFS_Release}>"
	"$<$<CONFIG:Debug>:${TEST_DEFAULT_DEFS_Debug}>"
	"$<$<CONFIG:Coverage>:${TEST_DEFAULT_DEFS_Coverage}>"

	"$<$<CONFIG:Release>:${TEST_DEFAULT_PERF_DEFS_Release}>"
	"$<$<CONFIG:Debug>:${TEST_DEFAULT_PERF_DEFS_Debug}>"
	"$<$<CONFIG:Coverage>:${TEST_DEFAULT_PERF_DEFS_Coverage}>"
)
target_compile_options( test_default_flags INTERFACE
	"$<$<CONFIG:Release>:${TEST_DEFAULT_OPTS_Release}>"
	"$<$<CONFIG:Debug>:${TEST_DEFAULT_OPTS_Debug}>"
	"$<$<CONFIG:Coverage>:${TEST_DEFAULT_OPTS_Coverage}>"

	"$<$<CONFIG:Release>:${TEST_DEFAULT_PERF_OPTS_Release}>"
	"$<$<CONFIG:Debug>:${TEST_DEFAULT_PERF_OPTS_Debug}>"
	"$<$<CONFIG:Coverage>:${TEST_DEFAULT_PERF_OPTS_Coverage}>"
)

# list of categories with default test settings: at the beginning, all of them
set( default_categories "${TEST_CATEGORIES}" )

# settings about modes are passed to tests via the MODES_* variables
#
# MODES_{category}
# MODES_{category}[_<mode name>]_suffix
#
# that MUST be defined for each category. The former stores the list of modes for
# the given test category, while the second the file suffix (with also _, or empty)
# to be appendedn to each test's target and file name


# macro to assemble custom projects for a given category and optional mode
# it is based on the above conditions to populate the flags, hence the corresponding
# strings (tests and performance) MUST BE DEFINED according to the conditions above
macro( add_category_flags category )
	if( NOT "${category}" IN_LIST TEST_CATEGORIES )
		message( FATAL_ERROR "category \"${category}\" not among: ${TEST_CATEGORIES}" )
	endif()

	set(oneValueArgs "MODE" )
	cmake_parse_arguments( parsed "${options}"
		"${oneValueArgs}" "${multiValueArgs}" "${ARGN}"
	)
	set( __prefix "${category}" )
	if( parsed_MODE )
		string( APPEND __prefix "_" "${parsed_MODE}" )
		if( "${__prefix}" IN_LIST MODES_${category} )
			message( FATAL_ERROR
				"mode \"${__prefix}\" already specified in MODES_${category}: ${MODES_${category}}"
			)
		endif()
		list( APPEND MODES_${category} "${__prefix}" )
		set( MODES_${__prefix}_suffix "_${parsed_MODE}" )
	else()
		set( MODES_${__prefix}_suffix "" )
	endif()

	list( REMOVE_ITEM default_categories "${category}" )

	# we're in a macro: reset the variables first!
	# previous invocations may already have set them
	set( __defs "" )
	set( __opts "" )
	foreach( bt ${CMAKE_CONFIGURATION_TYPES} )
		set( __defs_name "TEST_${__prefix}_DEFS_${bt}")
		assert_defined_variables( ${__defs_name} )
		list( APPEND __defs "$<$<CONFIG:${bt}>:${${__defs_name}}>" )

		set( __opts_name "TEST_${__prefix}_OPTS_${bt}")
		assert_defined_variables( ${__opts_name} )
		list( APPEND __opts "$<$<CONFIG:${bt}>:${${__opts_name}}>" )

		set( __perf_defs_name "TEST_${__prefix}_PERF_DEFS_${bt}")
		assert_defined_variables( ${__perf_defs_name} )
		list( APPEND __defs "$<$<CONFIG:${bt}>:${${__perf_defs_name}}>" )

		set( __perf_opts_name "TEST_${__prefix}_PERF_OPTS_${bt}")
		assert_defined_variables( ${__perf_opts_name} )
		list( APPEND __opts "$<$<CONFIG:${bt}>:${${__perf_opts_name}}>" )
	endforeach()

	set( __tgt_name "test_${__prefix}_flags" )
	add_library( ${__tgt_name} INTERFACE )
	target_compile_definitions( ${__tgt_name} INTERFACE "${__defs}" )
	target_compile_options( ${__tgt_name} INTERFACE "${__opts}" )
	target_link_libraries( ${__tgt_name} INTERFACE common_flags )
endmacro( add_category_flags )



set_valid_cache_string( TEST_unit_ndebug_DEFS_Release "${ADDITIONAL_TEST_DEFINITIONS}" ""
	"Release definitions for category unit, mode ndebug"
)
set_valid_cache_string( TEST_unit_ndebug_DEFS_Debug "${ADDITIONAL_TEST_DEFINITIONS}" ""
	"Debug definitions for category unit, mode ndebug"
)
set_valid_cache_string( TEST_unit_ndebug_DEFS_Coverage "${ADDITIONAL_TEST_DEFINITIONS}" ""
	"Coverage definitions for category unit, mode ndebug"
)
set_valid_cache_string( TEST_unit_ndebug_OPTS_Release "${ADDITIONAL_TEST_OPTIONS}" ""
	"Release options for category unit, mode ndebug"
)
set_valid_cache_string( TEST_unit_ndebug_OPTS_Debug "${ADDITIONAL_TEST_OPTIONS}" ""
	"Debug options for category unit, mode ndebug"
)
set_valid_cache_string( TEST_unit_ndebug_OPTS_Coverage "${ADDITIONAL_TEST_OPTIONS}" ""
	"Coverage options for category unit, mode ndebug"
)
set( TEST_unit_ndebug_PERF_DEFS_Release "${COMMON_PERF_DEFS_Release}" CACHE STRING
	"Release definitions for category unit, mode debug "
)
set( TEST_unit_ndebug_PERF_DEFS_Debug "${COMMON_PERF_DEFS_Release}" CACHE STRING
	"Debug definitions for category unit, mode debug "
)
set( TEST_unit_ndebug_PERF_DEFS_Coverage "${COMMON_PERF_DEFS_Release}" CACHE STRING
	"Coverage definitions for category unit, mode debug "
)
set( TEST_unit_ndebug_PERF_OPTS_Release "${COMMON_PERF_OPTS_Release}" CACHE STRING
	"Release options for category unit, mode debug "
)
set( TEST_unit_ndebug_PERF_OPTS_Debug "${COMMON_PERF_OPTS_Release}" CACHE STRING
	"Debug options for category unit, mode debug "
)
set( TEST_unit_ndebug_PERF_OPTS_Coverage "${COMMON_PERF_OPTS_Release}" CACHE STRING
	"Coverage options for category unit, mode debug "
)
add_category_flags( "unit" MODE ndebug )

set_valid_cache_string( TEST_unit_debug_DEFS_Release "${ADDITIONAL_TEST_DEFINITIONS}" ""
	"Release definitions for category unit, mode debug"
)
set_valid_cache_string( TEST_unit_debug_DEFS_Debug "${ADDITIONAL_TEST_DEFINITIONS}" ""
	"Debug definitions for category unit, mode debug"
)
set_valid_cache_string( TEST_unit_debug_DEFS_Coverage "${ADDITIONAL_TEST_DEFINITIONS}" ""
	"Coverage definitions for category unit, mode debug"
)
set_valid_cache_string( TEST_unit_debug_OPTS_Release "${ADDITIONAL_TEST_OPTIONS}" ""
	"Release options for category unit, mode debug"
)
set_valid_cache_string( TEST_unit_debug_OPTS_Debug "${ADDITIONAL_TEST_OPTIONS}" ""
	"Debug options for category unit, mode debug"
)
set_valid_cache_string( TEST_unit_debug_OPTS_Coverage "${ADDITIONAL_TEST_OPTIONS}" ""
	"Coverage options for category unit, mode debug"
)
set( TEST_unit_debug_PERF_DEFS_Release "${COMMON_PERF_DEFS_Debug}" CACHE STRING
	"Release performance definitions for category unit, mode debug"
)
set( TEST_unit_debug_PERF_DEFS_Debug "${COMMON_PERF_DEFS_Debug}" CACHE STRING
	"Debug performance definitions for category unit, mode debug"
)
set( TEST_unit_debug_PERF_DEFS_Coverage "${COMMON_PERF_DEFS_Debug}" CACHE STRING
	"Coverage performance definitions for category unit, mode debug"
)
set( TEST_unit_debug_PERF_OPTS_Release "${COMMON_PERF_OPTS_Debug}" CACHE STRING
	"Release options definitions for category unit, mode debug"
)
set( TEST_unit_debug_PERF_OPTS_Debug "${COMMON_PERF_OPTS_Debug}" CACHE STRING
	"Debug options definitions for category unit, mode debug"
)
set( TEST_unit_debug_PERF_OPTS_Coverage "${COMMON_PERF_OPTS_Debug}" CACHE STRING
	"Coverage options definitions for category unit, mode debug"
)
add_category_flags( "unit" MODE debug )

# for categories with no specific options, set default:
# - modes with same name as category
# - flags are as default, via an alias
foreach( cat ${default_categories} )
	set( MODES_${cat} "${cat}" )
	set( MODES_${cat}_suffix "" )
	add_library( "test_${cat}_flags" ALIAS test_default_flags )
endforeach()

# generate a report with all subsets of compilation flags, for each target type
message( "" )
message( "######### COMPILATION OPTIONS AND DEFINITIONS #########" )

message( "Build type: ${CMAKE_BUILD_TYPE}")
message( "global flags (from CMake): ${CMAKE_CXX_FLAGS} \
	${CMAKE_CXX_FLAGS_${CMAKE_BUILD_TYPE}}"
)
message( "common definitions: ${COMMON_DEFS_${CMAKE_BUILD_TYPE}}" )
message( "common options: ${COMMON_OPTS_${CMAKE_BUILD_TYPE}}" )

message( "flags for BACKENDS:")
message( "  definitions: ${BACKEND_DEFS_${CMAKE_BUILD_TYPE}}")
message( "  options: ${BACKEND_OPTS_${CMAKE_BUILD_TYPE}}")

message( "flags for TESTS:")

foreach( cat ${TEST_CATEGORIES} )
	if( cat IN_LIST default_categories )
		continue()
	endif()
	message( "  category: ${cat}" )
	foreach( mode ${MODES_${cat}} )
		assert_defined_variables( TEST_${mode}_DEFS_${CMAKE_BUILD_TYPE}
			TEST_${mode}_OPTS_${CMAKE_BUILD_TYPE}
			TEST_${mode}_PERF_DEFS_${CMAKE_BUILD_TYPE}
			TEST_${mode}_PERF_OPTS_${CMAKE_BUILD_TYPE}
		)
		assert_defined_targets( test_${mode}_flags )
		message( "    mode: ${mode}" )
		message( "      definitions: ${TEST_${mode}_DEFS_${CMAKE_BUILD_TYPE}}" )
		message( "      options: ${TEST_${mode}_OPTS_${CMAKE_BUILD_TYPE}}" )
		message( "      performance definitions: ${TEST_${mode}_PERF_DEFS_${CMAKE_BUILD_TYPE}}" )
		message( "      performance options: ${TEST_${mode}_PERF_OPTS_${CMAKE_BUILD_TYPE}}" )
	endforeach()
endforeach()

list( JOIN default_categories ", " cats )
message( "default test flags (categories: ${cats})")
message( "  common definitions: ${TEST_DEFAULT_DEFS_${CMAKE_BUILD_TYPE}}")
message( "  common options: ${TEST_DEFAULT_OPTS_${CMAKE_BUILD_TYPE}}")
message( "  performance definitions: ${TEST_DEFAULT_PERF_DEFS_${CMAKE_BUILD_TYPE}}")
message( "  performance options: ${TEST_DEFAULT_PERF_OPTS_${CMAKE_BUILD_TYPE}}")

message( "######### END OF COMPILATION OPTIONS AND DEFINITIONS #########" )
message( "" )
