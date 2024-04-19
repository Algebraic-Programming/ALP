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
# Creation of all shared-memory backends (reference and reference_omp, a.k.a. OMP),
# both as static and dynamic library. Any target importing a backend also imports
# the compiler definition(s) required to set it as default. If a target want to do it
# manually, the target 'backend_shmem_static' exists with no default backend selection
# in its compilation interface.
#

assert_valid_variables( SHMEM_BACKEND_INSTALL_DIR
	REFERENCE_BACKEND_DEFAULT_NAME REFERENCE_OMP_BACKEND_DEFAULT_NAME
	REFERENCE_SELECTION_DEFS REFERENCE_OMP_SELECTION_DEFS
	DENSE_BACKEND_DEFAULT_NAME DENSE_SELECTION_DEFS
)

assert_defined_variables( backend_reference_srcs )


set( backend_reference_srcs ${backend_reference_srcs}
	${CMAKE_CURRENT_SOURCE_DIR}/init.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/config.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/io.cpp
	PARENT_SCOPE
)

