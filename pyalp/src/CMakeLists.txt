cmake_minimum_required(VERSION 3.14)
project(python_module LANGUAGES CXX)
# Include pybind11 relative to this CMakeLists.txt directory
get_filename_component(PYBIND11_SRC_DIR "${CMAKE_CURRENT_LIST_DIR}/../extern/pybind11" ABSOLUTE)
add_subdirectory(${PYBIND11_SRC_DIR} ${CMAKE_BINARY_DIR}/pyalp/extern/pybind11)

# Try to find OpenMP - on macOS this may require Homebrew libomp to be installed.
# We do this quietly here and link the imported target to the module targets that
# require OpenMP so CMake's FindOpenMP is used consistently instead of relying on
# environment CPPFLAGS/LDFLAGS.
find_package(OpenMP QUIET)

# When configuring from the pyalp package directory (e.g. in cibuildwheel
# containers) the top-level CMake that usually defines these options is not
# executed. Provide sensible cached defaults so this CMakeLists can be used
# standalone. The top-level configuration will override these cached values
# when present.
if(NOT DEFINED WITH_REFERENCE_BACKEND)
	set(WITH_REFERENCE_BACKEND ON CACHE BOOL "Build Reference backend (default for pyalp package)")
endif()
if(NOT DEFINED WITH_OMP_BACKEND)
	set(WITH_OMP_BACKEND ON CACHE BOOL "Build OMP backend (default for pyalp package)")
endif()

assert_defined_variables( WITH_REFERENCE_BACKEND WITH_OMP_BACKEND )

# target listing all examples, to build them at once with 'make examples'


if( WITH_REFERENCE_BACKEND )
	set(PYALP_MODULE_NAME pyalp_ref)

	pybind11_add_module( ${PYALP_MODULE_NAME} pyalp.cpp conjugate_gradient.hpp matrix_wrappers.hpp utils.hpp vector_wrappers.hpp )
	target_compile_definitions(${PYALP_MODULE_NAME} PRIVATE PYALP_MODULE_NAME=${PYALP_MODULE_NAME})
	# Link your required libraries
	target_link_libraries(${PYALP_MODULE_NAME} PRIVATE backend_shmem_shared backend_reference common_flags)
	# If OpenMP support is available, ensure the module links the OpenMP imported target
	if(OpenMP_CXX_FOUND)
		target_link_libraries(${PYALP_MODULE_NAME} PRIVATE OpenMP::OpenMP_CXX)
	endif()
	# (Optional) If you need extra compile flags or C++ standard:
	 set_target_properties(${PYALP_MODULE_NAME} PROPERTIES CXX_STANDARD 14 CXX_STANDARD_REQUIRED YES)
	# target_compile_options(conjugate_gradient_python PRIVATE -Wall -Wextra)
	
	# On Apple platforms with newer Clang, downgrade the template keyword warning from error to warning
	if(APPLE)
		target_compile_options(${PYALP_MODULE_NAME} PRIVATE -Wno-error=missing-template-arg-list-after-template-kw)
	endif()

	add_custom_command(TARGET ${PYALP_MODULE_NAME} POST_BUILD
	    COMMAND ${CMAKE_COMMAND} -E echo ""
	    COMMAND ${CMAKE_COMMAND} -E echo "============================================================"
	    COMMAND ${CMAKE_COMMAND} -E echo "Build complete!"
	    COMMAND ${CMAKE_COMMAND} -E echo "To use the Python bindings, append ${CMAKE_BINARY_DIR}/python to your PYTHONPATH:"
	    COMMAND ${CMAKE_COMMAND} -E echo "============================================================"
	)

endif()


if( WITH_OMP_BACKEND )
	set(PYALP_MODULE_NAME pyalp_omp)

	pybind11_add_module( ${PYALP_MODULE_NAME} pyalp.cpp conjugate_gradient.hpp matrix_wrappers.hpp utils.hpp vector_wrappers.hpp )
	target_compile_definitions( ${PYALP_MODULE_NAME} PRIVATE PYALP_MODULE_NAME=${PYALP_MODULE_NAME})
	# Link your required libraries
	target_link_libraries(${PYALP_MODULE_NAME} PRIVATE backend_shmem_shared backend_reference_omp common_flags)
	# If OpenMP support is available, ensure the module links the OpenMP imported target
	if(OpenMP_CXX_FOUND)
		target_link_libraries(${PYALP_MODULE_NAME} PRIVATE OpenMP::OpenMP_CXX)
	endif()
	# (Optional) If you need extra compile flags or C++ standard:
	set_target_properties(${PYALP_MODULE_NAME} PROPERTIES CXX_STANDARD 14 CXX_STANDARD_REQUIRED YES)
	# target_compile_options(conjugate_gradient_python PRIVATE -Wall -Wextra)

	add_custom_command(TARGET ${PYALP_MODULE_NAME} POST_BUILD
	    COMMAND ${CMAKE_COMMAND} -E echo ""
	    COMMAND ${CMAKE_COMMAND} -E echo "============================================================"
	    COMMAND ${CMAKE_COMMAND} -E echo "Build complete!"
	    COMMAND ${CMAKE_COMMAND} -E echo "To use the Python bindings, append ${CMAKE_BINARY_DIR}/python to your PYTHONPATH:"
	    COMMAND ${CMAKE_COMMAND} -E echo "============================================================"
	)

endif()


if( WITH_NONBLOCKING_BACKEND )
	set(PYALP_MODULE_NAME pyalp_nonblocking)

	pybind11_add_module( ${PYALP_MODULE_NAME} pyalp.cpp conjugate_gradient.hpp matrix_wrappers.hpp utils.hpp vector_wrappers.hpp )
	target_compile_definitions( ${PYALP_MODULE_NAME} PRIVATE PYALP_MODULE_NAME=${PYALP_MODULE_NAME})
	# Link your required libraries
	target_link_libraries(${PYALP_MODULE_NAME} PRIVATE backend_shmem_shared backend_reference_omp common_flags)
	# If OpenMP support is available, ensure the module links the OpenMP imported target
	if(OpenMP_CXX_FOUND)
		target_link_libraries(${PYALP_MODULE_NAME} PRIVATE OpenMP::OpenMP_CXX)
	endif()
	# (Optional) If you need extra compile flags or C++ standard:
	set_target_properties(${PYALP_MODULE_NAME} PROPERTIES CXX_STANDARD 14 CXX_STANDARD_REQUIRED YES)
	# target_compile_options(conjugate_gradient_python PRIVATE -Wall -Wextra)

	add_custom_command(TARGET ${PYALP_MODULE_NAME} POST_BUILD
	    COMMAND ${CMAKE_COMMAND} -E echo ""
	    COMMAND ${CMAKE_COMMAND} -E echo "============================================================"
	    COMMAND ${CMAKE_COMMAND} -E echo "Build complete!"
	    COMMAND ${CMAKE_COMMAND} -E echo "To use the Python bindings, append ${CMAKE_BINARY_DIR}/python to your PYTHONPATH:"
	    COMMAND ${CMAKE_COMMAND} -E echo "============================================================"
	)

endif()

# Also include the self-contained pyalp module that builds a private
# extension named _pyalp using only headers. This allows building
# from the pyalp package root inside cibuildwheel containers.
# Do not attempt to build pyalp as a standalone subproject from here.
# This project must be configured from the repository top-level CMake only.

# ---------------------------------------------------------------------------
# Generate runtime metadata (algorithm vs build) exposed to Python
# This writes a small Python module into the source tree so it becomes
# part of the packaged wheel and can be inspected at runtime.
# ---------------------------------------------------------------------------

# Prepare README and LICENSE contents (if available)
if(EXISTS "${CMAKE_SOURCE_DIR}/README.md")
	file(READ "${CMAKE_SOURCE_DIR}/README.md" README_RAW)
else()
	set(README_RAW "")
endif()
string(REPLACE "\"\"\"" "\\\"\\\"\\\"" README_ESC "${README_RAW}")

if(EXISTS "${CMAKE_SOURCE_DIR}/LICENSE")
	file(READ "${CMAKE_SOURCE_DIR}/LICENSE" LICENSE_RAW)
else()
	set(LICENSE_RAW "")
endif()
string(REPLACE "\"\"\"" "\\\"\\\"\\\"" LICENSE_ESC "${LICENSE_RAW}")

# Short git commit (or 'unknown')
execute_process(COMMAND git rev-parse --short HEAD
				WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
				RESULT_VARIABLE _git_res
				OUTPUT_VARIABLE GIT_COMMIT
				OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT _git_res EQUAL 0)
	set(GIT_COMMIT "unknown")
endif()

string(TIMESTAMP BUILD_TIME UTC)

# Collect declared Python requirements if egg-info is available
set(_reqs_file "${CMAKE_SOURCE_DIR}/pyalp/src/pyalp.egg-info/requires.txt")
set(REQS_PY "")
if(EXISTS "${_reqs_file}")
	file(READ "${_reqs_file}" REQ_RAW)
	# trim trailing newline
	string(REGEX REPLACE "\r?\n$" "" REQ_RAW "${REQ_RAW}")
	string(REPLACE "\n" ";" REQ_ITEMS "${REQ_RAW}")
	foreach(_r IN LISTS REQ_ITEMS)
		if(NOT _r STREQUAL "")
			string(REPLACE '"' '\\"' _r_esc "${_r}")
			if(REQS_PY STREQUAL "")
				set(REQS_PY '"${_r_esc}"')
			else()
				set(REQS_PY "${REQS_PY}, \"${_r_esc}\"")
			endif()
		endif()
	endforeach()
endif()

# Provide defaults for variables that may be set at top-level
if(NOT DEFINED ALP_PORTABLE_BUILD)
	set(ALP_PORTABLE_BUILD OFF)
endif()

set(METADATA_IN "${CMAKE_CURRENT_LIST_DIR}/pyalp/_metadata.py.in")
set(METADATA_OUT "${CMAKE_CURRENT_LIST_DIR}/pyalp/_metadata.py")

configure_file(${METADATA_IN} ${METADATA_OUT} @ONLY)


