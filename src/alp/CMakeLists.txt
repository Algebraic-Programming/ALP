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

### ACTUAL BACKEND IMPLEMENTATIONS

# target to compile all backend libraries
# add_custom_target( libs ALL )

# these sources are common to all backends, although some are in reference
set( backend_alp_reference_srcs
	${CMAKE_CURRENT_SOURCE_DIR}/descriptors.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/rc.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/reference/init.cpp
	# ${CMAKE_CURRENT_SOURCE_DIR}/reference/config.cpp
)

if( WITH_ALP_REFERENCE_BACKEND )
	add_subdirectory( reference )
endif()

if( WITH_ALP_OMP_BACKEND )
	add_subdirectory( omp )
endif()

set( backend_alp_dispatch_srcs
	${CMAKE_CURRENT_SOURCE_DIR}/descriptors.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/rc.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/dispatch/init.cpp
)

if( WITH_ALP_DISPATCH_BACKEND )
	add_subdirectory( dispatch )
endif()
