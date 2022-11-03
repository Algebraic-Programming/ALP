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
Find libkblas inside the standard system directories

Read-only output variables:
  KBLAS_FOUND
	Indicates that the library has been found.

  KBLAS_INCLUDE_DIR
	Points to the libkblas include directory.

  KBLAS_LIBRARY
	Points to the libkblas that can be passed to target_link_libararies.

creates a target Kblas::Kblas to link against libkblas
#]===================================================================]

# documentation of find_path() https://cmake.org/cmake/help/latest/command/find_path.html
# documentation of find_library() https://cmake.org/cmake/help/latest/command/find_library.html

if(NOT KBLAS_IMPL)
	set(KBLAS_IMPL "nolocking")
else()
	if(NOT ${KBLAS_IMPL} IN_LIST "locking;nolocking;omp;pthread")
		message(ERROR "wrong kblas implementation requested")
	endif()
endif()

# find the root directory for libkblas
find_path( KBLAS_ROOT_DIR
	NAMES include/kblas.h # by checking where "include/kblas.h" exists
	PATHS ${KBLAS_ROOT}  # take as a hint the environment variable KBLAS_ROOT and
						 # add it to the default search paths
	DOC "KBLAS root directory"
)

# look for the include directory
# we should not assume the header is present, because some distributions have
# different packages for binary-only versions (e.g., libkblas) and for
# development-oriented versions (e.g., libkblas-dev); hence, look for the header
# explicitly and raise an error if you cannot find it (otherwise targets will
# surely not compile!)
find_path( KBLAS_INCLUDE_DIR
	NAMES kblas.h # by looking for this header file
	HINTS ${KBLAS_ROOT_DIR} # start looking from KBLAS_ROOT_DIR, the most likely place
	PATH_SUFFIXES include  # when inspecting a path, look inside the include directory
	DOC "KBLAS include directory"
)

# look for the binary library libkblas
# do not give thorough hints here, because various Linux distributions may have different
# conventions on shared binarie directories (/lib, /usr/lib, /usr/lib64, ...)
# and we don't want to "blind" CMake's search
find_library( KBLAS_LIBRARY
	NAMES kblas # hence, CMake looks for libkblas.so, libkblas.so.<some version>,
			   # libkblas.a and so on (read find_library() guide for more details)
	HINTS "${KBLAS_ROOT_DIR}/lib/kblas/${KBLAS_IMPL}" # start looking from KBLAS_ROOT_DIR, the most likely place
	DOC "KBLAS library"
)

find_library( GFORTRAN_LIBRARY
	NAMES gfortran # hence, CMake looks for libgfortran.so, libgfortran.so.<some version>,
			   # libgfortran.a and so on (read find_library() guide for more details)
	HINTS ${CMAKE_CXX_IMPLICIT_LINK_DIRECTORIES}
)

# if the listed variables are set to existing paths, set the Kblas_FOUND variable
# if not and the REQUIRED option was given when calling this find_module(),
# raise an error (some components were not found and we need all of them)
include( FindPackageHandleStandardArgs )
find_package_handle_standard_args( Kblas
	REQUIRED_VARS KBLAS_ROOT_DIR KBLAS_INCLUDE_DIR KBLAS_LIBRARY GFORTRAN_LIBRARY
)

# if we found the library, create a dedicated target with all needed information
if( Kblas_FOUND )
	# do not show these variables as cached ones
	mark_as_advanced( KBLAS_ROOT_DIR KBLAS_INCLUDE_DIR KBLAS_LIBRARY )

	# create an imported target, i.e. a target NOT built internally, as from
	# https://cmake.org/cmake/help/latest/command/add_library.html#imported-libraries
	# this way, depending targets may link against libkblas with target_link_libraries(),
	# as if it was an internal target
	# UNKNOWN tells CMake to inspect the library type (static or shared)
	# e.g., if you compiled your own static libkblas and injected it via KBLAS_ROOT
	# it will work out without changes
        add_library ( gfortran::gfortran UNKNOWN IMPORTED )
	add_library ( Kblas::Kblas UNKNOWN IMPORTED )
	# set its properties to the appropiate locations, for both headers and binaries
        set_target_properties( gfortran::gfortran
                PROPERTIES
                IMPORTED_LOCATION "${GFORTRAN_LIBRARY}"
        )
	set_target_properties( Kblas::Kblas
		PROPERTIES
		IMPORTED_LOCATION "${KBLAS_LIBRARY}"
		INTERFACE_INCLUDE_DIRECTORIES ${KBLAS_INCLUDE_DIR}
	)
	if(NOT LibM_FOUND)
		find_package(LibM REQUIRED)
	endif()
	target_link_libraries(Kblas::Kblas INTERFACE LibM::LibM gfortran::gfortran)
	if(${KBLAS_IMPL} STREQUAL "omp")
		if(NOT OpenMP_FOUND)
			find_package(OpenMP REQUIRED)
		endif()
	        target_link_libraries(Kblas::Kblas INTERFACE OpenMP::OpenMP_C)
	elif(${KBLAS_IMPL} STREQUAL "pthread")
		if(NOT Threads_FOUND)
			find_package(Threads REQUIRED)
		endif()
		if(NOT CMAKE_USE_PTHREADS_INIT)
			message(ERROR "pthread not found")
		endif()
		target_link_libraries(Kblas::Kblas INTERFACE Threads::Threads)
	endif()
endif()


