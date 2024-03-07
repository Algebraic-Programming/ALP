#
#   Copyright 2024 Huawei Technologies Co., Ltd.
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

assert_valid_variables( ARCH_DETECT_APPS_DIR )

set( _supported_arches "x86_64;arm" )
if( NOT CMAKE_SYSTEM_PROCESSOR IN_LIST _supported_arches )
	message( FATAL_ERROR "Architecture \"${CMAKE_SYSTEM_PROCESSOR}\" not supported" )
endif()

set( DEFAULT_SIMD_SIZE 64 )
set( DEFAULT_L1CACHE_SIZE 32768 )
set( DEFAULT_CACHE_LINE_SIZE 64 )

if( CMAKE_VERSION VERSION_LESS "3.25.0" )
	# old CMake versions have a different signature for try_compile()
	# https://cmake.org/cmake/help/latest/command/try_run.html#try-compiling-and-running-source-files
	set( _dest ${CMAKE_CURRENT_BINARY_DIR} )
endif()

set( ARCH_DETECT_APPS_DIR ${CMAKE_CURRENT_BINARY_DIR}/src/arch_info )
set( _simd_detect_destination detect_simd_isa )

set( SIMD_ISA_DETECT_APP OFF )
try_compile( COMPILED ${_dest} SOURCES ${CMAKE_SOURCE_DIR}/cmake/${CMAKE_SYSTEM_PROCESSOR}_simd_detect.c
	COPY_FILE ${ARCH_DETECT_APPS_DIR}/${_simd_detect_destination}
	COPY_FILE_ERROR COPY_MSG
)
if( COMPILED )
	execute_process(
		COMMAND ${ARCH_DETECT_APPS_DIR}/${_simd_detect_destination}
		RESULT_VARIABLE RES
		OUTPUT_VARIABLE SIMD_ISA
		OUTPUT_STRIP_TRAILING_WHITESPACE
	)
endif()

if( NOT COMPILED OR ( NOT RES STREQUAL "0" ) OR COPY_MSG )
	set( SIMD_SIZE ${DEFAULT_SIMD_SIZE} )
	message( WARNING "Cannot detect SIMD ISA, thus applying default vector size: ${SIMD_SIZE}B" )
else()
	set( SIMD_ISA_DETECT_APP ${_simd_detect_destination} )
	if( SIMD_ISA STREQUAL "SVE" OR SIMD_ISA STREQUAL "SVE2" )
		set( SIMD_SIZE 64 )
		message( WARNING "Detected SIMD ISA ${SIMD_ISA}, whose size is implementation-dependant and currently not detected. Please, consider filing an issue to the authors. Applying default vector size: ${SIMD_SIZE}B" )
	else()
		if( SIMD_ISA STREQUAL "AVX512" )
			set( SIMD_SIZE 64 )
		elseif( SIMD_ISA STREQUAL "AVX2" )
			set( SIMD_SIZE 32 )
		elseif( SIMD_ISA STREQUAL "AVX" )
			set( SIMD_SIZE 16 )
		elseif( SIMD_ISA STREQUAL "NEON" )
			set( SIMD_SIZE 16 )
		endif()
		message( "Detected SIMD ISA: ${SIMD_ISA}; vector size : ${SIMD_SIZE}B" )
	endif()
endif()

set( L1CACHE_DETECT_APP OFF )
execute_process(
	COMMAND ${CMAKE_SOURCE_DIR}/cmake/l1_cache_info.sh
	RESULT_VARIABLE RES
	OUTPUT_VARIABLE CACHE_DETECT_OUTPUT
	OUTPUT_STRIP_TRAILING_WHITESPACE
)
file( COPY ${CMAKE_SOURCE_DIR}/cmake/l1_cache_info.sh DESTINATION ${ARCH_DETECT_APPS_DIR} )
if( NOT RES STREQUAL "0" )
	set( L1CACHE_SIZE ${DEFAULT_L1CACHE_SIZE} )
	set( CACHE_LINE_SIZE ${DEFAULT_CACHE_LINE_SIZE} )
	message( WARNING "Cannot detect L1 cache features, thus applying default settigs" )
else()
	set( L1CACHE_DETECT_APP l1_cache_info.sh )
	string( REGEX MATCHALL
		"TYPE:[ \t]*(Data|Unified)[ \t\r\n]+SIZE:[ \t]*([0-9]+)[ \t\r\n]+LINE:[ \t]*([0-9]+)[ \t\r\n]*"
		MATCH_OUTPUT "${CACHE_DETECT_OUTPUT}"
	)
	set( L1DCACHE_TYPE ${CMAKE_MATCH_1} )
	set( L1DCACHE_SIZE ${CMAKE_MATCH_2} )
	set( CACHE_LINE_SIZE ${CMAKE_MATCH_3} )
	if( L1DCACHE_TYPE STREQUAL "Unified" )
		message( WARNING "The L1 cache is Unified, so it may not be possible to effectively utilize its entire size (${L1DCACHE_SIZE}B) for the data." )
	endif()
endif()
message( "L1 cache size: ${L1DCACHE_SIZE}B; cacheline size: ${CACHE_LINE_SIZE}B" )
