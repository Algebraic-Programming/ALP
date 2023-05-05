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

#[===================================================================[
Find libnuma inside the standard system directories

Read-only output variables:
  NUMA_FOUND
	Indicates that the library has been found.

  NUMA_INCLUDE_DIR
	Points to the libnuma include directory.
make
  NUMA_LIBRARY
	Points to the libnuma that can be passed to target_link_libararies.

creates a target Numa::Numa to link against libnuma
#]===================================================================]

# documentation of find_path() https://cmake.org/cmake/help/latest/command/find_path.html
# documentation of find_library() https://cmake.org/cmake/help/latest/command/find_library.html

if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "GNU" )
	message( FATAL_ERROR "GNU compiler is required to enable coverage" )
endif()

find_program( GCOV_PATH gcov )
find_program( GENINFO_PATH geninfo )
find_program( GENHTML_PATH genhtml )

# if the listed variables are set to existing paths, set the GCov_FOUND variable
# if not and the REQUIRED option was given when calling this find_module(),
# raise an error (some components were not found and we need all of them)
include( FindPackageHandleStandardArgs )
find_package_handle_standard_args( GCov
	REQUIRED_VARS GCOV_PATH GENINFO_PATH GENHTML_PATH
)

# if we found the library, create a dedicated target with all needed information
if( GCov_FOUND )
	# do not show these variables as cached ones
	mark_as_advanced( GCOV_PATH GENINFO_PATH GENHTML_PATH )

	# create an imported target, i.e. a target NOT built internally, as from
	# https://cmake.org/cmake/help/latest/command/add_library.html#imported-libraries
	# this way, depending targets may link against libnuma with target_link_libraries(),
	# as if it was an internal target
	# UNKNOWN tells CMake to inspect the library type (static or shared)
	# e.g., if you compiled your own static libnuma and injected it via NUMA_ROOT
	# it will work out without changes
	add_library ( GCov::GCov INTERFACE IMPORTED )
	# set its properties to the appropiate locations, for both headers and binaries
	set_target_properties( GCov::GCov
		PROPERTIES
		IMPORTED_LIBNAME gcov
	)
endif()
