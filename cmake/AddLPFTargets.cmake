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
# Adds the minimal functionalities to generate an LPF library
# needs:
#  LPF_ENGINE LPF engine to link against
#  target_link_lpf command to link against LPF
#

if( NOT COMMAND target_link_lpf )
	message( FATAL_ERROR "Need \"target_link_lpf\" function from LPF CMake to add an LPF library")
endif()

assert_valid_variables( LPF_ENGINE )

#
# create a library and link it against LPF
#  libName: name of the target library
#  libType: any of the types allowed for add_library()
#  sources: source to build the library from
#
function( add_lpf_library libName libType )
	if( NOT libName )
		message( FATAL_ERROR "Specify library name")
	endif()
	if( TARGET ${libName} )
		message( FATAL_ERROR "Target ${libName} already exists")
	endif()

	if( NOT ARGN )
		message( FATAL_ERROR "Specify library sources")
	endif()

	add_library( ${libName} ${libType} "${ARGN}" )
	target_link_lpf( ${libName} PUBLIC HL ENGINE "${LPF_ENGINE}" )
	# LPF may not be aware we are building a C++ program, and some MPI
	# implementations may require C++-specific libraries to link against
	set_property( TARGET ${libName}
		APPEND PROPERTY
			INTERFACE_LINK_LIBRARIES "${MPI_CXX_LIBRARIES}"
	)
endfunction()

# function to link a target against LPF with default values
function( target_link_lpf_default targetName )
	target_link_lpf( ${targetName} PRIVATE ENGINE "${LPF_ENGINE}" )
endfunction( target_link_lpf_default targetName)

