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
# Definitions of targets for LPF backend, namely BSP1D and HYBRID.
# As for the reference* backends, importing targets already get
# the default backend set in the compilation interface.
#

assert_defined_variables( WITH_BSP1D_BACKEND WITH_HYBRID_BACKEND )

assert_valid_variables( backend_reference_srcs backend_bsp_srcs
	BACKEND_LIBRARY_OUTPUT_NAME
	BSP1D_BACKEND_DEFAULT_NAME HYBRID_BACKEND_DEFAULT_NAME
	BSP1D_BACKEND_INSTALL_DIR HYBRID_BACKEND_INSTALL_DIR
	INCLUDE_INSTALL_DIR BSP1D_SELECTION_DEFS HYBRID_SELECTION_DEFS
)

assert_defined_targets( backend_flags )

# macro to create an LPF-based library target, with all minimal dependencies required by both
# the BSP1D and the Hybrid backends.
#
# Parameters:
#   targetName name of the target
#   targetType type of library (STATIC or SHARED)
#   outDir directory to compile the binary to (not install)
#   src1 first (mandatory) source file; other sources are appended as ARGN
#
macro( make_bsp1d_target targetName targetType outDir src1 )

	add_lpf_library( "${targetName}" "${targetType}" "${src1}" "${ARGN}" )
	set_target_properties( "${targetName}" PROPERTIES
		OUTPUT_NAME "${BACKEND_LIBRARY_OUTPUT_NAME}"
	)
	if( "${targetType}" STREQUAL "SHARED" )
		set_target_properties( "${targetName}" PROPERTIES
			SOVERSION "${VERSION}"
			LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${outDir}"
		)
	elseif( "${targetType}" STREQUAL "STATIC" )
		set_target_properties( "${targetName}" PROPERTIES
			ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${outDir}"
		)
	else()
		message( FATAL_ERROR "Unknown library type: ${targetType}")
	endif()

	#
	# "-rdynamic" is for LPF to be able to find all symbols within an executable
	# see https://gcc.gnu.org/onlinedocs/gcc/Link-Options.html
	# reasons to keep "-rdynamic" for LPF:
	# 1. passing function pointers around
	# 2. choosing between MPI auto-initialization or LPF-sides initialization
	# hence, DO NOT REMOVE IT!!!
	target_link_options( "${targetName}" INTERFACE "-rdynamic" )
	target_link_libraries( "${targetName}" PRIVATE backend_flags )

endmacro( make_bsp1d_target )

# sources for bsp1d backend
set( backend_bsp1d_srcs
	"${backend_reference_srcs}"
	"${backend_bsp_srcs}"
	${CMAKE_CURRENT_SOURCE_DIR}/exec.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/init.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/config.cpp
)

if( WITH_BSP1D_BACKEND )

	make_bsp1d_target( backend_bsp1d_static STATIC "spmd" "${backend_bsp1d_srcs}" )
	target_link_libraries( backend_bsp1d_static PUBLIC backend_bsp1d_headers )
	target_compile_definitions( backend_bsp1d_static PUBLIC "${BSP1D_SELECTION_DEFS}" )
	add_dependencies( libs backend_bsp1d_static )

	install( TARGETS backend_bsp1d_static
		EXPORT GraphBLASTargets
		ARCHIVE DESTINATION "${BSP1D_BACKEND_INSTALL_DIR}"
	)

	make_bsp1d_target( backend_bsp1d_shared SHARED "spmd" "${backend_bsp1d_srcs}" )
	target_link_libraries( backend_bsp1d_shared PUBLIC backend_bsp1d_headers )
	target_compile_definitions( backend_bsp1d_shared PUBLIC "${BSP1D_SELECTION_DEFS}" )
	add_dependencies( libs backend_bsp1d_shared )

	install( TARGETS backend_bsp1d_shared
		EXPORT GraphBLASTargets
		LIBRARY DESTINATION "${BSP1D_BACKEND_INSTALL_DIR}"
	)

	# alias for add_grb_executables()
	# DO NOT CHANGE THE ALIAS NAME!
	add_library( "${BSP1D_BACKEND_DEFAULT_NAME}" ALIAS backend_bsp1d_static )

endif()

if( WITH_HYBRID_BACKEND )
	# sources for hybrid backend
	set( backend_hybrid_srcs "${backend_bsp1d_srcs}" )

	make_bsp1d_target( backend_hybrid_static STATIC "hybrid" "${backend_hybrid_srcs}" )
	target_link_libraries( backend_hybrid_static PUBLIC backend_hybrid_headers )
	target_compile_definitions( backend_hybrid_static PUBLIC "${HYBRID_SELECTION_DEFS}" )
	add_dependencies( libs backend_hybrid_static )

	install( TARGETS backend_hybrid_static
		EXPORT GraphBLASTargets
		ARCHIVE DESTINATION "${HYBRID_BACKEND_INSTALL_DIR}"
	)

	make_bsp1d_target( backend_hybrid_shared SHARED "hybrid" "${backend_hybrid_srcs}" )
	target_link_libraries( backend_hybrid_shared PUBLIC backend_hybrid_headers )
	target_compile_definitions( backend_hybrid_shared PUBLIC "${HYBRID_SELECTION_DEFS}" )
	add_dependencies( libs backend_hybrid_shared )

	install( TARGETS backend_hybrid_shared
		EXPORT GraphBLASTargets
		LIBRARY DESTINATION "${HYBRID_BACKEND_INSTALL_DIR}"
	)

	# alias for add_grb_executables()
	# DO NOT CHANGE THE ALIAS NAME!
	add_library( "${HYBRID_BACKEND_DEFAULT_NAME}" ALIAS backend_hybrid_static )

endif()

