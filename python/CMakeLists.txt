cmake_minimum_required(VERSION 3.14)
project(python_module LANGUAGES CXX)

add_subdirectory(extern/pybind11)

assert_defined_variables( WITH_REFERENCE_BACKEND WITH_OMP_BACKEND )

# target listing all examples, to build them at once with 'make examples'


if( WITH_REFERENCE_BACKEND )
	set(PYALP_MODULE_NAME pyalp_ref)

	pybind11_add_module( ${PYALP_MODULE_NAME} pyalp.cpp conjugate_gradient.hpp matrix_wrappers.hpp utils.hpp vector_wrappers.hpp )
	target_compile_definitions(${PYALP_MODULE_NAME} PRIVATE PYALP_MODULE_NAME=${PYALP_MODULE_NAME})
	# Link your required libraries
	target_link_libraries(${PYALP_MODULE_NAME} PRIVATE backend_shmem_shared backend_reference common_flags)
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


if( WITH_OMP_BACKEND )
	set(PYALP_MODULE_NAME pyalp_omp)

	pybind11_add_module( ${PYALP_MODULE_NAME} pyalp.cpp conjugate_gradient.hpp matrix_wrappers.hpp utils.hpp vector_wrappers.hpp )
	target_compile_definitions( ${PYALP_MODULE_NAME} PRIVATE PYALP_MODULE_NAME=${PYALP_MODULE_NAME})
	# Link your required libraries
	target_link_libraries(${PYALP_MODULE_NAME} PRIVATE backend_shmem_shared backend_reference_omp common_flags)
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

