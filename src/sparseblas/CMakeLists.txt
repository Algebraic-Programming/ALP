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

assert_defined_variables( WITH_REFERENCE_BACKEND WITH_OMP_BACKEND )

add_library( sparseblas_static STATIC
	${CMAKE_CURRENT_SOURCE_DIR}/sparseblas.cpp
)

set_target_properties( sparseblas_static PROPERTIES
	OUTPUT_NAME "sparseblas"
	ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/shmem"
)

target_link_libraries( sparseblas_static PUBLIC backend_reference sparseblas )

target_link_libraries( sparseblas_static PRIVATE backend_flags )

add_dependencies( libs sparseblas_static )

install( TARGETS sparseblas_static
	EXPORT GraphBLASTargets
	ARCHIVE DESTINATION "${SHMEM_BACKEND_INSTALL_DIR}"
)

add_library( sparseblas_omp_static STATIC
	${CMAKE_CURRENT_SOURCE_DIR}/sparseblas.cpp
)

set_target_properties( sparseblas_omp_static PROPERTIES
	OUTPUT_NAME "sparseblas_omp"
	ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/shmem"
)

target_link_libraries( sparseblas_omp_static PUBLIC backend_reference_omp sparseblas )

target_link_libraries( sparseblas_omp_static PRIVATE backend_flags )

add_dependencies( libs sparseblas_omp_static )

install( TARGETS sparseblas_omp_static
	EXPORT GraphBLASTargets
	ARCHIVE DESTINATION "${SHMEM_BACKEND_INSTALL_DIR}"
)

