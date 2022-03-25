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
# This file creates the basic target(s) needed by all backends
#

assert_defined_targets( backend_headers_nodefs backend_flags )

assert_valid_variables( ALP_UTILS_LIBRARY_OUTPUT_NAME )

# convenience variable: a few tests need the path of utility sources
set( ALP_UTILS_SRC_PATH "${CMAKE_CURRENT_SOURCE_DIR}/utils"
	CACHE INTERNAL "ALP utility sources path"
)

# little utility library needed by backends
add_library( alp_utils_static STATIC
	${ALP_UTILS_SRC_PATH}/hpparser.c
)
set_target_properties( alp_utils_static PROPERTIES
	OUTPUT_NAME "${ALP_UTILS_LIBRARY_OUTPUT_NAME}"
)
target_compile_definitions( alp_utils_static PRIVATE _GNU_SOURCE )
target_link_libraries( alp_utils_static
	PRIVATE Threads::Threads backend_flags
	PUBLIC alp_utils_headers
)
install( TARGETS alp_utils_static
	EXPORT GraphBLASTargets
	ARCHIVE DESTINATION "${ALP_UTILS_INSTALL_DIR}"
)

add_library( alp_utils_dynamic SHARED
	${ALP_UTILS_SRC_PATH}/hpparser.c
)
set_target_properties( alp_utils_dynamic PROPERTIES
	OUTPUT_NAME "${ALP_UTILS_LIBRARY_OUTPUT_NAME}"
)
target_compile_definitions( alp_utils_dynamic PRIVATE _GNU_SOURCE )
target_link_libraries( alp_utils_dynamic
	PRIVATE Threads::Threads backend_flags
	PUBLIC alp_utils_headers
)
install( TARGETS alp_utils_dynamic
	EXPORT GraphBLASTargets
	LIBRARY DESTINATION "${ALP_UTILS_INSTALL_DIR}"
)

### ACTUAL BACKEND IMPLEMENTATIONS

# target to compile all backend libraries
add_custom_target( libs ALL )

# these sources are common to all backends, although some are in reference
set( backend_reference_srcs
	${CMAKE_CURRENT_SOURCE_DIR}/descriptors.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/rc.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/reference/init.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/reference/config.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/reference/io.cpp
)

# the only source file common to all BSP-based backends
set( backend_bsp_srcs
	${CMAKE_CURRENT_SOURCE_DIR}/bsp/collectives.cpp
)

if( WITH_REFERENCE_BACKEND OR WITH_OMP_BACKEND )
	add_subdirectory( reference )
endif()

if( WITH_BSP1D_BACKEND OR WITH_HYBRID_BACKEND )
	add_subdirectory( bsp1d )
endif()

