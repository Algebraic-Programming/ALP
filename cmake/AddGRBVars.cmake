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

assert_defined_variables( WITH_REFERENCE_BACKEND WITH_OMP_BACKEND WITH_NONBLOCKING_BACKEND
	WITH_BSP1D_BACKEND WITH_HYBRID_BACKEND
)

### STANDARD TARGET NAMES
# tests set to compile against backends link against these target names
set( REFERENCE_BACKEND_DEFAULT_NAME "backend_reference" )
set( REFERENCE_OMP_BACKEND_DEFAULT_NAME "backend_reference_omp" )
set( BSP1D_BACKEND_DEFAULT_NAME "backend_bsp1d" )
set( HYBRID_BACKEND_DEFAULT_NAME "backend_hybrid" )
set( HYPERDAGS_BACKEND_DEFAULT_NAME "backend_hyperdags" )
set( NONBLOCKING_BACKEND_DEFAULT_NAME "backend_nonblocking" )
set( ALP_REFERENCE_BACKEND_DEFAULT_NAME "backend_alp_reference" )
set( ALP_DISPATCH_BACKEND_DEFAULT_NAME "backend_alp_dispatch" )
set( ALP_OMP_BACKEND_DEFAULT_NAME "backend_alp_omp" )


### COMPILER DEFINITIONS FOR HEADERS INCLUSION AND FOR BACKEND SELECTION

# compiler definitions to include backend headers
set( REFERENCE_INCLUDE_DEFS "_GRB_WITH_REFERENCE" )
set( REFERENCE_OMP_INCLUDE_DEFS "_GRB_WITH_OMP" )
set( HYPERDAGS_INCLUDE_DEFS "_GRB_WITH_HYPERDAGS" )
set( NONBLOCKING_INCLUDE_DEFS "_GRB_WITH_NONBLOCKING" )
set( LPF_INCLUDE_DEFS "_GRB_WITH_LPF" )
set( ALP_REFERENCE_INCLUDE_DEFS "_ALP_WITH_REFERENCE" )
set( ALP_DISPATCH_INCLUDE_DEFS "_ALP_WITH_DISPATCH" )
set( ALP_OMP_INCLUDE_DEFS "_ALP_WITH_OMP;_ALP_OMP_WITH_DISPATCH" )

# compiler definitions to select a backend
set( REFERENCE_SELECTION_DEFS "_GRB_BACKEND=reference" )
set( REFERENCE_OMP_SELECTION_DEFS "_GRB_BACKEND=reference_omp" )
set( HYPERDAGS_SELECTION_DEFS
	"_GRB_BACKEND=hyperdags"
	"_GRB_WITH_HYPERDAGS_USING=${WITH_HYPERDAGS_USING}"
)
set( NONBLOCKING_SELECTION_DEFS "_GRB_BACKEND=nonblocking" )
set( ALP_REFERENCE_SELECTION_DEFS "_ALP_BACKEND=reference" )
set( ALP_DISPATCH_SELECTION_DEFS "_ALP_BACKEND=dispatch" )
set( ALP_OMP_SELECTION_DEFS
		"_ALP_BACKEND=omp"
		"_ALP_SECONDARY_BACKEND=dispatch"
)
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
set( ALL_BACKENDS "reference" "reference_omp" "hyperdags" "nonblocking" "bsp1d" "hybrid" "alp_reference" "alp_dispatch" "alp_omp" )

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

# dependent backends
if( WITH_HYPERDAGS_BACKEND )
	list( APPEND AVAILABLE_BACKENDS "hyperdags" )
endif()

if( WITH_NONBLOCKING_BACKEND )
	list( APPEND AVAILABLE_BACKENDS "nonblocking" )
endif()

if( WITH_ALP_REFERENCE_BACKEND )
	list( APPEND AVAILABLE_BACKENDS "alp_reference" )
endif()

if( WITH_ALP_DISPATCH_BACKEND )
	list( APPEND AVAILABLE_BACKENDS "alp_dispatch" )
endif()

if( WITH_ALP_OMP_BACKEND )
	list( APPEND AVAILABLE_BACKENDS "alp_omp" )
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

