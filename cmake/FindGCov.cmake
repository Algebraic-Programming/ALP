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
Find gcov program inside the standard system directories

Read-only output variables:
  GCOV_PATH
	Points to the gcov binary.

creates a target Gcov::Gcov to link against libgcov
#]===================================================================]

# documentation of find_path() https://cmake.org/cmake/help/latest/command/find_path.html
# documentation of find_library() https://cmake.org/cmake/help/latest/command/find_library.html

if(NOT CMAKE_COMPILER_IS_GNUCC )
	message( FATAL_ERROR "GNU compiler is required to enable coverage" )
endif()

find_program( GCOV_PATH gcov )

# if the listed variables are set to existing paths, set the GCov_FOUND variable
# if not and the REQUIRED option was given when calling this find_module(),
# raise an error (some components were not found and we need all of them)
include( FindPackageHandleStandardArgs )
find_package_handle_standard_args( GCov REQUIRED_VARS GCOV_PATH )

# if we found the library, create a dedicated target with all needed information
if( GCov_FOUND )
	# do not show these variables as cached ones
	mark_as_advanced( GCOV_PATH )

	add_library ( GCov::GCov INTERFACE IMPORTED )
	# set its properties to the appropiate locations, for both headers and binaries
	set_target_properties( GCov::GCov
		PROPERTIES
		IMPORTED_LIBNAME gcov
	)
endif()
