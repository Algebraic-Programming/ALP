cmake_minimum_required(VERSION 3.14)
project(pyalp_bindings LANGUAGES CXX)

pybind11_add_module(_pyalp bindings.cpp ../matrix_wrappers.hpp ../utils.hpp ../vector_wrappers.hpp ../conjugate_gradient.hpp)
target_compile_definitions(_pyalp PRIVATE PYALP_MODULE_NAME=_pyalp)
set_target_properties(_pyalp PROPERTIES CXX_STANDARD 14 CXX_STANDARD_REQUIRED YES)
target_include_directories(_pyalp PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/..)

# Honor top-level NUMA switch: add NO_NUMA_DEF when WITH_NUMA=OFF
if(DEFINED WITH_NUMA AND NOT WITH_NUMA)
  if(DEFINED NO_NUMA_DEF)
	target_compile_definitions(_pyalp PRIVATE ${NO_NUMA_DEF})
  else()
	target_compile_definitions(_pyalp PRIVATE _GRB_NO_LIBNUMA)
  endif()
endif()
