cmake_minimum_required(VERSION 3.14)
project(python_module LANGUAGES CXX)

add_subdirectory(extern/pybind11)

pybind11_add_module(python2alp python2alp.cpp)
pybind11_add_module(numpy2alp numpy2alp.cpp)

assert_defined_variables( WITH_REFERENCE_BACKEND WITH_OMP_BACKEND )

# target listing all examples, to build them at once with 'make examples'
add_custom_target( examples_python)

if( WITH_REFERENCE_BACKEND )
	add_executable( sp_reference_python sp.cpp )
	target_link_libraries( sp_reference_python backend_reference common_flags )
endif()


if( WITH_REFERENCE_BACKEND )


	pybind11_add_module( conjugate_gradient_python conjugate_gradient.cpp )
	#target_compile_definitions(conjugate_gradient_python common_flags)

	# Link your required libraries
	target_link_libraries(conjugate_gradient_python PRIVATE backend_shmem_shared backend_reference common_flags)
	# (Optional) If you need extra compile flags or C++ standard:
	# set_target_properties(conjugate_gradient_python PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)
	# target_compile_options(conjugate_gradient_python PRIVATE -Wall -Wextra)


	#add_executable( conjugate_gradient_python conjugate_gradient.cpp )
	#target_link_libraries( conjugate_gradient_python backend_reference common_flags )

endif()

