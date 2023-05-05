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

#[===================================================================[
Find gcov compiler support by compiling a test program with coverage flags

Read-only output variables:
  GCov_COMPILES
	Whether a simple C++ binary compiles with coverage options.

creates a target Gcov::Gcov to link against libgcov and with the necessary
compile flags
#]===================================================================]

if( NOT CMAKE_COMPILER_IS_GNUCC )
	message( FATAL_ERROR "GNU compiler is required to enable coverage" )
endif()

set( GCOV_COMPILE_FLAGS -fprofile-arcs -ftest-coverage )
set( GCOV_LIBNAME gcov )

# try compiling a basic application with GCov flags to see if the compiler supports it
try_compile( GCov_COMPILES "${CMAKE_BINARY_DIR}"
	SOURCES "${CMAKE_SOURCE_DIR}/cmake/simple.cpp"
	LINK_LIBRARIES "${GCOV_LIBNAME}"
	CMAKE_FLAGS -DCMAKE_CXX_FLAGS:STRING="${GCOV_COMPILE_FLAGS};-O0"
)

# get major number of g++ compiler
string( REPLACE "." ";" VERSION_LIST ${CMAKE_CXX_COMPILER_VERSION} )
list( GET VERSION_LIST 0 CXX_COMPILER_VERSION_MAJOR )

# look for the gcov utility of the specific gcc version we are using
# if none, look for the generic one named "gcov"
find_program( GCOV_EXECUTABLE NAMES gcov-${CXX_COMPILER_VERSION_MAJOR} gcov REQUIRED )

include( FindPackageHandleStandardArgs )
find_package_handle_standard_args( GCov REQUIRED_VARS GCov_COMPILES GCOV_EXECUTABLE )

# if we found the library, create a dedicated target with all needed information
if( GCov_FOUND )
	add_library ( GCov::GCov INTERFACE IMPORTED )

	# do not look for the exact path of libgcov.a, since this depends on the compiler
	# (each GCC version ships its own); just pass the "-lgcov" flag during linking,
	# since CMake invokes CMAKE_C[XX]_COMPILER also for linking
	set_target_properties( GCov::GCov
		PROPERTIES
		IMPORTED_LIBNAME "${GCOV_LIBNAME}"
		INTERFACE_COMPILE_OPTIONS "${GCOV_COMPILE_FLAGS}"
	)
endif()
