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

# --- Metadata generation ---
# These variables are expected to be set by the top-level CMakeLists.txt,
# but we provide defaults for standalone builds.
if(NOT DEFINED pyalp_VERSION)
    set(pyalp_VERSION "0.0.0")
endif()
if(NOT DEFINED ALP_VERSION)
    set(ALP_VERSION "unknown")
	# If the top-level project provided ALP_VERSION (cached) this will be used.
	# Otherwise keep 'unknown'. When building from the top-level CMakeLists
	# ALP_VERSION should already be set via cache.
	if(DEFINED ALP_VERSION)
		# already defined (likely cached) — keep it
	else()
		set(ALP_VERSION "unknown")
	endif()
endif()
if(NOT DEFINED ALP_BUILD_TYPE)
    set(ALP_BUILD_TYPE "unknown")
	# Use CMAKE_BUILD_TYPE as the ALP build type unless explicitly supplied
	if(DEFINED CMAKE_BUILD_TYPE)
		set(ALP_BUILD_TYPE "${CMAKE_BUILD_TYPE}")
	else()
		set(ALP_BUILD_TYPE "unknown")
	endif()
endif()
if(NOT DEFINED ALP_GIT_COMMIT_SHA)
    set(ALP_GIT_COMMIT_SHA "unknown")
endif()
if(NOT DEFINED ALP_GIT_BRANCH)
    set(ALP_GIT_BRANCH "unknown")
endif()

# Determine repository license (simple heuristic from LICENSE file) if not provided
if(NOT DEFINED ALP_LICENSE)
	if(EXISTS "${CMAKE_SOURCE_DIR}/LICENSE")
		file(READ "${CMAKE_SOURCE_DIR}/LICENSE" ALP_LICENSE_RAW)
		string(FIND "${ALP_LICENSE_RAW}" "Apache" _has_apache)
		if(_has_apache GREATER -1)
			set(ALP_LICENSE "Apache-2.0")
		else()
			string(FIND "${ALP_LICENSE_RAW}" "BSD" _has_bsd)
			if(_has_bsd GREATER -1)
				set(ALP_LICENSE "BSD-3-Clause")
			else()
				set(ALP_LICENSE "unknown")
			endif()
		endif()
	else()
		set(ALP_LICENSE "unknown")
	endif()
endif()

# If a top-level README.md exists, read it and make it available to the
# metadata template. We escape triple quotes so the Python triple-quoted
# string in the template remains valid.
if(EXISTS "${CMAKE_SOURCE_DIR}/README.md")
	file(READ "${CMAKE_SOURCE_DIR}/README.md" PYALP_README_RAW)
	# Escape triple-quotes to avoid breaking the Python triple-quoted string
	string(REPLACE "\"\"\"" "\\\"\\\"\\\"" PYALP_README_ESCAPED "${PYALP_README_RAW}")
else()
	set(PYALP_README_ESCAPED "")
endif()
if(EXISTS "${CMAKE_SOURCE_DIR}/pyalp/README.md")
	file(READ "${CMAKE_SOURCE_DIR}/pyalp/README.md" PYALP_README_RAW)
elseif(EXISTS "${CMAKE_SOURCE_DIR}/README.md")
	file(READ "${CMAKE_SOURCE_DIR}/README.md" PYALP_README_RAW)
else()
	set(PYALP_README_ESCAPED "")
endif()
if(DEFINED PYALP_README_RAW)
	# Escape triple-quotes to avoid breaking the Python triple-quoted string
	string(REPLACE "\"\"\"" "\\\"\\\"\\\"" PYALP_README_ESCAPED "${PYALP_README_RAW}")
else()
	set(PYALP_README_ESCAPED "")
endif()

# Get Python and pybind11 versions
find_package(PythonInterp REQUIRED)
set(PYTHON_VERSION ${PYTHON_VERSION_STRING})
set(pybind11_VERSION ${pybind11_VERSION})

# This is a simplified list. A more robust solution would inspect the build targets.
set(pyalp_ALGORITHMS "conjugate_gradient")
set(pyalp_BACKENDS "reference, reference_omp")

# Configure the metadata file from the template
set(METADATA_TEMPLATE "${CMAKE_CURRENT_SOURCE_DIR}/pyalp/_metadata.py.in")
set(METADATA_OUTPUT "${CMAKE_BINARY_DIR}/pyalp_metadata.py")
# Make the README content available as @PYALP_README_ESCAPED@ to the
# template (CMake configure_file will perform the substitution).
configure_file(${METADATA_TEMPLATE} ${METADATA_OUTPUT} @ONLY)

# This command is useful for debugging inside a cibuildwheel container
# to verify that the file is being generated correctly.
# add_custom_command(TARGET ${PYALP_MODULE_NAME} POST_BUILD
#     COMMAND ${CMAKE_COMMAND} -E echo "Generated metadata file content:"
#     COMMAND ${CMAKE_COMMAND} -E cat ${METADATA_OUTPUT}
# )

