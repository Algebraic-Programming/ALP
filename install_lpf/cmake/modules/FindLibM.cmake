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
Find the math library

creates a target LibM::LibM to link against libm
#]===================================================================]

find_path( LIBM_ROOT_DIR
	NAMES include/math.h
	PATHS ENV LIBM_ROOT
	DOC "LibM root directory"
	REQUIRED
)

find_path( LIBM_INCLUDE_DIR
	NAMES math.h
	HINTS ${LIBM_ROOT_DIR}
	PATH_SUFFIXES include
	DOC "LibM include directory"
	REQUIRED
)


find_library( LIBM_LIBRARY m REQUIRED )

include( FindPackageHandleStandardArgs )
find_package_handle_standard_args( LibM
	REQUIRED_VARS LIBM_LIBRARY LIBM_INCLUDE_DIR
)

if( LibM_FOUND )
	mark_as_advanced( LIBM_INCLUDE_DIR LIBM_LIBRARY )
	add_library( LibM::LibM UNKNOWN IMPORTED )
	set_target_properties( LibM::LibM PROPERTIES
		IMPORTED_LOCATION "${LIBM_LIBRARY}"
		INTERFACE_INCLUDE_DIRECTORIES "${LIBM_INCLUDE_DIR}"
	)
endif()
