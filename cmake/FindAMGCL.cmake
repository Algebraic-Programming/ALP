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
Find libamgcl inside the standard system directories

Read-only output variables:
  AMGCL_FOUND
	Indicates that the library has been found.

  AMGCL_INCLUDE_DIR
	Points to the libamgcl include directory.

creates a target amgcl::amgcl to link against libamgcl
#]===================================================================]

# documentation of find_path() https://cmake.org/cmake/help/latest/command/find_path.html
# documentation of find_library() https://cmake.org/cmake/help/latest/command/find_library.html

# find the root directory for libamgcl
find_path( AMGCL_ROOT_DIR
	NAMES lib/amgcl.h # by checking where "lib/amgcl.h" exists
	HINTS ${AMGCL_SOURCE} # start looking from AMGCL_SOURCE, the most likely place
)

# if the listed variables are set to existing paths, set the amgcl_FOUND variable
# if not and the REQUIRED option was given when calling this find_module(),
# raise an error (some components were not found and we need all of them)
include( FindPackageHandleStandardArgs )
find_package_handle_standard_args( AMGCL
	REQUIRED_VARS AMGCL_ROOT_DIR
)

# if we found the library, create a dedicated target with all needed information
if( AMGCL_FOUND  )
	# do not show these variables as cached ones
	mark_as_advanced( AMGCL_ROOT_DIR )

	# create an imported target, i.e. a target NOT built internally, as from
	# https://cmake.org/cmake/help/latest/command/add_library.html#imported-libraries
	# this way, depending targets may link against libamgcl with target_link_libraries(),
	# as if it was an internal target
	# UNKNOWN tells CMake to inspect the library type (static or shared)
	# e.g., if you compiled your own static libamgcl and injected it via AMGCL_ROOT
	# it will work out without changes
	add_library ( amgcl INTERFACE )
	# set its properties to the appropiate locations, for both headers and binaries
	# set_target_properties( amgcl::amgcl
	# 	PROPERTIES
	# 	INTERFACE_INCLUDE_DIRECTORIES "${AMGCL_ROOT_DIR}"
	# )
	target_include_directories ( amgcl INTERFACE ${AMGCL_ROOT_DIR} 
	)
endif()
