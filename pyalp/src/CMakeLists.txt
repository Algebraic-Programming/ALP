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
endif()
if(NOT DEFINED ALP_BUILD_TYPE)
    set(ALP_BUILD_TYPE "unknown")
endif()
if(NOT DEFINED ALP_GIT_COMMIT_SHA)
    set(ALP_GIT_COMMIT_SHA "unknown")
endif()
if(NOT DEFINED ALP_GIT_BRANCH)
    set(ALP_GIT_BRANCH "unknown")
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
configure_file(${METADATA_TEMPLATE} ${METADATA_OUTPUT} @ONLY)

# This command is useful for debugging inside a cibuildwheel container
# to verify that the file is being generated correctly.
# add_custom_command(TARGET ${PYALP_MODULE_NAME} POST_BUILD
#     COMMAND ${CMAKE_COMMAND} -E echo "Generated metadata file content:"
#     COMMAND ${CMAKE_COMMAND} -E cat ${METADATA_OUTPUT}
# )

