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

cmake_minimum_required( VERSION 3.13 )

include( CMakeFindDependencyMacro )

# Capturing values from configure (optional)
set( ALPGraphBLAS_AVAILABLE_BACKENDS reference;reference_omp;hyperdags;nonblocking;bsp1d;hybrid )

# standard CMake dependencies
find_dependency( Threads REQUIRED )
find_dependency( OpenMP REQUIRED )

set( ALPGraphBLAS_WITH_BSP1D_BACKEND ON )
set( ALPGraphBLAS_WITH_HYBRID_BACKEND ON )
if( ALPGraphBLAS_WITH_BSP1D_BACKEND
	OR ALPGraphBLAS_WITH_HYBRID_BACKEND )
	find_dependency( MPI REQUIRED )
	# discover LPF from its own path
	find_package( LPF REQUIRED CONFIG PATHS "/alp_deployment/lpf/build_mpich/install/lib"
)
endif()

# custom dependencies
list( APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/modules" )
find_package( LibM REQUIRED )

set( ALPGraphBLAS_WITH_NUMA ON )
if( ALPGraphBLAS_WITH_NUMA )
	find_dependency( Numa REQUIRED )
endif()


# Add the targets file
include( "${CMAKE_CURRENT_LIST_DIR}/ALPGraphBLASTargets.cmake" )
