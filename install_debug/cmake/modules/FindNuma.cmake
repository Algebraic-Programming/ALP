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

  NUMA_LIBRARY
	Points to the libnuma that can be passed to target_link_libararies.

creates a target Numa::Numa to link against libnuma
#]===================================================================]

# documentation of find_path() https://cmake.org/cmake/help/latest/command/find_path.html
# documentation of find_library() https://cmake.org/cmake/help/latest/command/find_library.html

# find the root directory for libnuma
find_path( NUMA_ROOT_DIR
	NAMES include/numa.h # by checking where "include/numa.h" exists
	PATHS ENV NUMA_ROOT  # take as a hint the environment variable NUMA_ROOT and
						 # add it to the default search paths
	DOC "NUMA root directory"
)

# look for the include directory
# we should not assume the header is present, because some distributions have
# different packages for binary-only versions (e.g., libnuma) and for
# development-oriented versions (e.g., libnuma-dev); hence, look for the header
# explicitly and raise an error if you cannot find it (otherwise targets will
# surely not compile!)
find_path( NUMA_INCLUDE_DIR
	NAMES numa.h # by looking for this header file
	HINTS ${NUMA_ROOT_DIR} # start looking from NUMA_ROOT_DIR, the most likely place
	PATH_SUFFIXES include  # when inspecting a path, look inside the include directory
	DOC "NUMA include directory"
)

# look for the binary library libnuma
# do not give thorough hints here, because various Linux distributions may have different
# conventions on shared binarie directories (/lib, /usr/lib, /usr/lib64, ...)
# and we don't want to "blind" CMake's search
find_library( NUMA_LIBRARY
	NAMES numa # hence, CMake looks for libnuma.so, libnuma.so.<some version>,
			   # libnuma.a and so on (read find_library() guide for more details)
	HINTS ${NUMA_ROOT_DIR} # start looking from NUMA_ROOT_DIR, the most likely place
	DOC "NUMA library"
)

# if the listed variables are set to existing paths, set the Numa_FOUND variable
# if not and the REQUIRED option was given when calling this find_module(),
# raise an error (some components were not found and we need all of them)
include( FindPackageHandleStandardArgs )
find_package_handle_standard_args( Numa
	REQUIRED_VARS NUMA_ROOT_DIR NUMA_INCLUDE_DIR NUMA_LIBRARY
)

# if we found the library, create a dedicated target with all needed information
if( Numa_FOUND )
	# do not show these variables as cached ones
	mark_as_advanced( NUMA_ROOT_DIR NUMA_INCLUDE_DIR NUMA_LIBRARY )

	# create an imported target, i.e. a target NOT built internally, as from
	# https://cmake.org/cmake/help/latest/command/add_library.html#imported-libraries
	# this way, depending targets may link against libnuma with target_link_libraries(),
	# as if it was an internal target
	# UNKNOWN tells CMake to inspect the library type (static or shared)
	# e.g., if you compiled your own static libnuma and injected it via NUMA_ROOT
	# it will work out without changes
	add_library ( Numa::Numa UNKNOWN IMPORTED )
	# set its properties to the appropiate locations, for both headers and binaries
	set_target_properties( Numa::Numa
		PROPERTIES
		IMPORTED_LOCATION ${NUMA_LIBRARY}
		INTERFACE_INCLUDE_DIRECTORIES ${NUMA_INCLUDE_DIR}
	)
endif()
