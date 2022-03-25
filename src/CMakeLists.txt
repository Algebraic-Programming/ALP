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
# This file contains mainly the logic to create the wrapper scripts for the
# installation directory, to build GraphBLAS binaries outside of CMake's workflow.
#

### ADD BACKENDS
add_subdirectory( graphblas )
add_subdirectory( alp )


### BUILD WRAPPER SCRIPTS FOR INSTALLATION

assert_valid_variables( AVAILABLE_BACKENDS CMAKE_INSTALL_PREFIX INCLUDE_INSTALL_DIR
	VERSION BIN_INSTALL_DIR
)
assert_defined_variables( COMMON_WRAPPER_DEFINITIONS COMMON_WRAPPER_OPTIONS )

if( WITH_BSP1D_BACKEND OR WITH_HYBRID_BACKEND )
	assert_valid_variables( LPFRUN_CMD MANUALRUN )
endif()

# join elements in the list named inListName by a space ' ', wrap them inside
# quotes "" and append them to the list named outListName
# EXAMPLE: input list: 'compiler;--switch;value' -> appended string: '"compiler --switch value"'
#   (quotes "" are also part of the text)
#
# do NOT expand strings with paths as macro arguments, pass only NAMES of variables
# (not their content)
macro( joinAndAppend outListName inListName )
	list( JOIN ${inListName} " " tmp_spaced )
	list( APPEND ${outListName} "\"${tmp_spaced}\"" )
endmacro( joinAndAppend )

# for each backend, check for the wrapper information variables and create the lists
# of values to be stored inside the wrappers
foreach( backend ${AVAILABLE_BACKENDS} )
	assert_valid_variables( ${backend}_WRAPPER_COMPILER_COMMAND )
	assert_defined_variables( ${backend}_WRAPPER_RUNENV ${backend}_WRAPPER_RUNNER
		${backend}_WRAPPER_COMPILE_DEFINITIONS ${backend}_WRAPPER_COMPILE_OPTIONS
		${backend}_WRAPPER_LINK_FLAGS
	)

	list( APPEND backend_list "\"${backend}\"" )
	joinAndAppend( compiler_list ${backend}_WRAPPER_COMPILER_COMMAND )

	joinAndAppend( runenv_list ${backend}_WRAPPER_RUNENV )
	joinAndAppend( runner_list ${backend}_WRAPPER_RUNNER )
	set( compile_options "${${backend}_WRAPPER_COMPILE_DEFINITIONS}" )
	list( TRANSFORM compile_options PREPEND "-D" )
	list( APPEND compile_options ${${backend}_WRAPPER_COMPILE_OPTIONS} )
	joinAndAppend( compile_options_list compile_options )
	joinAndAppend( link_flags_list ${backend}_WRAPPER_LINK_FLAGS )
endforeach( )

# space the options and pass them to the scripts via configure_file
#
# internal issue #317: cannot use quoted LPFRUN_CMD_Q here, because
# sourcing the resulting setenv makes the quotes seem as part of the
# executable name. Until this is fixed, we do not support spaces in
# the LPF install path (FIXME)
list( JOIN backend_list " " AVAILABLE_BACKENDS_SPACED )
list( JOIN LPFRUN_CMD " " LPFRUN_CMD_SPACED )
list( JOIN MANUALRUN " " MANUALRUN_SPACED )
configure_file( ${CMAKE_CURRENT_SOURCE_DIR}/setenv.in ${CMAKE_CURRENT_BINARY_DIR}/setenv @ONLY )

set( cflags "${COMMON_WRAPPER_DEFINITIONS}" )
list( TRANSFORM cflags PREPEND "-D" )
list( APPEND cflags "${COMMON_WRAPPER_OPTIONS}" )
joinAndAppend( COMMON_FLAGS_SPACED cflags )

list( JOIN compiler_list " " BACKEND_COMPILERS_SPACED )
list( JOIN compile_options_list " " BACKEND_CFLAGS_SPACED )
list( JOIN link_flags_list " " BACKEND_LFLAGS_SPACED )
configure_file( ${CMAKE_CURRENT_SOURCE_DIR}/grbcxx.in ${CMAKE_CURRENT_BINARY_DIR}/grbcxx @ONLY )

list( JOIN runenv_list " " BACKEND_RUNENV_SPACED )
list( JOIN runner_list " " BACKEND_RUNNER_SPACED )
configure_file( ${CMAKE_CURRENT_SOURCE_DIR}/grbrun.in ${CMAKE_CURRENT_BINARY_DIR}/grbrun @ONLY )

# install them to the install folder with execute permission
install( FILES ${CMAKE_CURRENT_BINARY_DIR}/setenv
	${CMAKE_CURRENT_BINARY_DIR}/grbcxx
	${CMAKE_CURRENT_BINARY_DIR}/grbrun

	DESTINATION "${BIN_INSTALL_DIR}"
	PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
		GROUP_READ GROUP_EXECUTE
		WORLD_READ WORLD_EXECUTE
)

### GENERATE CMAKE INFRASTRUCTURE INSIDE INSTALLATION DIRECTORY

include(CMakePackageConfigHelpers)

# write file with version information
write_basic_package_version_file(
	ALPGraphBLASVersion.cmake
	VERSION ${VERSION}
	COMPATIBILITY AnyNewerVersion
)

# write custom file to check for dependencies and set custom variables
configure_file( ALPGraphBLASConfig.cmake.in ALPGraphBLASConfig.cmake @ONLY )

#install modules for dependencies lookup
install(FILES "${PROJECT_SOURCE_DIR}/cmake/FindNuma.cmake"
		"${PROJECT_SOURCE_DIR}/cmake/FindLibM.cmake"
		DESTINATION "${CMAKE_CONFIGS_INSTALL_DIR}/modules"
)

# install custom file and version file
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/ALPGraphBLASVersion.cmake"
			"${CMAKE_CURRENT_BINARY_DIR}/ALPGraphBLASConfig.cmake"
		DESTINATION "${CMAKE_CONFIGS_INSTALL_DIR}"
)
