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
# defines variables for compilation and testing
# used throughout the entire build infrastructure
#
# to add a new backend, add your own to each ### SECTION
#

assert_defined_variables( WITH_REFERENCE_BACKEND WITH_OMP_BACKEND WITH_BSP1D_BACKEND
	WITH_HYBRID_BACKEND WITH_NUMA
)

### STANDARD TARGET NAMES
# tests set to compile against backends link against these target names
set( REFERENCE_BACKEND_DEFAULT_NAME "backend_reference" )
set( REFERENCE_OMP_BACKEND_DEFAULT_NAME "backend_reference_omp" )
set( BSP1D_BACKEND_DEFAULT_NAME "backend_bsp1d" )
set( HYBRID_BACKEND_DEFAULT_NAME "backend_hybrid" )
set( DENSE_BACKEND_DEFAULT_NAME "backend_reference_dense" )


### COMPILER DEFINITIONS FOR HEADERS INCLUSION AND FOR BACKEND SELECTION

# compiler definitions to include backend headers
set( REFERENCE_INCLUDE_DEFS "_GRB_WITH_REFERENCE" )
set( REFERENCE_OMP_INCLUDE_DEFS "_GRB_WITH_OMP" )
set( LPF_INCLUDE_DEFS "_GRB_WITH_LPF" )
set( DENSE_INCLUDE_DEFS "_GRB_WITH_DENSE" )

# compiler definitions to select a backend
set( REFERENCE_SELECTION_DEFS "_GRB_BACKEND=reference" )
set( REFERENCE_OMP_SELECTION_DEFS "_GRB_BACKEND=reference_omp" )
set( DENSE_SELECTION_DEFS "_GRB_BACKEND=reference_dense" )
set( BSP1D_SELECTION_DEFS
		"_GRB_BACKEND=BSP1D"
		"_GRB_BSP1D_BACKEND=reference"
)
set( HYBRID_SELECTION_DEFS
		"_GRB_BACKEND=BSP1D"
		"_GRB_BSP1D_BACKEND=reference_omp"
)

# definition to set if not depending on libnuma
set( NO_NUMA_DEF "_GRB_NO_LIBNUMA" )

### **ALL** BACKENDS, EVEN IF NOT ENABLED BY USER
set( ALL_BACKENDS "reference" "reference_omp" "bsp1d" "hybrid" "reference_dense" )


# list of user-enabled backends, for tests and wrapper scripts (do not change!)
set( AVAILABLE_BACKENDS "" )

### POPULATING LISTS FOR INSTALLATION TARGETS (see src/CMakeLists.txt)
# backends that are enabled by the user: append as in the following

# shared memory backends
if( WITH_REFERENCE_BACKEND )
	list( APPEND AVAILABLE_BACKENDS "reference" )
endif()

if( WITH_OMP_BACKEND )
	list( APPEND AVAILABLE_BACKENDS "reference_omp" )
endif()

if( WITH_DENSE_BACKEND )
	list( APPEND AVAILABLE_BACKENDS "reference_dense" )
endif()

# distributed memory backends
if( WITH_BSP1D_BACKEND )
	list( APPEND AVAILABLE_BACKENDS "bsp1d" )
endif()

if( WITH_HYBRID_BACKEND )
	list( APPEND AVAILABLE_BACKENDS "hybrid" )
endif()

message( STATUS "\n######### Configured with the following backends: #########\n${AVAILABLE_BACKENDS}\n" )

# add your own here!

