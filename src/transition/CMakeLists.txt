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

assert_defined_variables( WITH_REFERENCE_BACKEND WITH_OMP_BACKEND WITH_NONBLOCKING_BACKEND )

function( add_transition_library target_name lib_type lib_name src1 )

	set( multiValueArgs
		"SOURCES"
		"PUBLIC_LINK_LIBRARIES"
	)
	cmake_parse_arguments( parsed "" "" "${multiValueArgs}" "SOURCES;${src1};${ARGN}" )

	add_library( ${target_name} ${lib_type} ${parsed_SOURCES} )
	set_target_properties( ${target_name} PROPERTIES
		OUTPUT_NAME ${lib_name}
		ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/shmem"
	)

	target_link_libraries( ${target_name} PUBLIC transition_headers ${parsed_PUBLIC_LINK_LIBRARIES} )
	target_link_libraries( ${target_name} PRIVATE backend_flags )
	add_dependencies( libs ${target_name} )
	install( TARGETS ${target_name} EXPORT GraphBLASTargets
		ARCHIVE DESTINATION "${SHMEM_BACKEND_INSTALL_DIR}"
	)

endfunction( add_transition_library )

if( WITH_REFERENCE_BACKEND )
	add_transition_library( sparseblas_static STATIC "sparseblas" ${CMAKE_CURRENT_SOURCE_DIR}/sparseblas.cpp
		PUBLIC_LINK_LIBRARIES backend_reference
	)

	set( _SPBLAS_PREFIX "kml_" )
	if( SPBLAS_PREFIX )
		set( _SPBLAS_PREFIX "${SPBLAS_PREFIX}_" )
	endif()
	add_transition_library( ${_SPBLAS_PREFIX}sparseblas_static STATIC "${_SPBLAS_PREFIX}sparseblas"
		${CMAKE_CURRENT_SOURCE_DIR}/sparseblas.cpp PUBLIC_LINK_LIBRARIES backend_reference
	)
	target_compile_definitions( ${_SPBLAS_PREFIX}sparseblas_static PUBLIC SPBLAS_PREFIX=${_SPBLAS_PREFIX} )

	add_transition_library( spsolver_static STATIC "spsolver" ${CMAKE_CURRENT_SOURCE_DIR}/solver.cpp
		PUBLIC_LINK_LIBRARIES backend_reference
	)
endif()

if( WITH_OMP_BACKEND )
	add_transition_library( sparseblas_omp_static STATIC "sparseblas_omp" ${CMAKE_CURRENT_SOURCE_DIR}/sparseblas.cpp
		PUBLIC_LINK_LIBRARIES backend_reference_omp
	)

	add_transition_library( kml_sparseblas_omp_static STATIC "kml_sparseblas_omp" ${CMAKE_CURRENT_SOURCE_DIR}/sparseblas.cpp
		PUBLIC_LINK_LIBRARIES backend_reference_omp
	)

	add_transition_library( spsolver_omp_static STATIC "spsolver_omp" ${CMAKE_CURRENT_SOURCE_DIR}/solver.cpp
		PUBLIC_LINK_LIBRARIES backend_reference_omp
	)
endif()

if( WITH_NONBLOCKING_BACKEND )
	add_transition_library( spsolver_nonblocking_static STATIC "spsolver_nonblocking" ${CMAKE_CURRENT_SOURCE_DIR}/solver.cpp
		PUBLIC_LINK_LIBRARIES backend_nonblocking
	)

	add_transition_library( ksolver STATIC "ksolver" ${CMAKE_CURRENT_SOURCE_DIR}/kml_iss.cpp
		PUBLIC_LINK_LIBRARIES spsolver_nonblocking_static
	)
endif()
