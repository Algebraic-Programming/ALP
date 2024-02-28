#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "ALPGraphBLAS::alp_utils_static" for configuration "Debug"
set_property(TARGET ALPGraphBLAS::alp_utils_static APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(ALPGraphBLAS::alp_utils_static PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_DEBUG "C"
  IMPORTED_LOCATION_DEBUG "/home/belozes/graphblas/install_debug/lib/libalp_utils.a"
  )

list(APPEND _cmake_import_check_targets ALPGraphBLAS::alp_utils_static )
list(APPEND _cmake_import_check_files_for_ALPGraphBLAS::alp_utils_static "/home/belozes/graphblas/install_debug/lib/libalp_utils.a" )

# Import target "ALPGraphBLAS::alp_utils_dynamic" for configuration "Debug"
set_property(TARGET ALPGraphBLAS::alp_utils_dynamic APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(ALPGraphBLAS::alp_utils_dynamic PROPERTIES
  IMPORTED_LOCATION_DEBUG "/home/belozes/graphblas/install_debug/lib/libalp_utils.so"
  IMPORTED_SONAME_DEBUG "libalp_utils.so"
  )

list(APPEND _cmake_import_check_targets ALPGraphBLAS::alp_utils_dynamic )
list(APPEND _cmake_import_check_files_for_ALPGraphBLAS::alp_utils_dynamic "/home/belozes/graphblas/install_debug/lib/libalp_utils.so" )

# Import target "ALPGraphBLAS::backend_hyperdags_static" for configuration "Debug"
set_property(TARGET ALPGraphBLAS::backend_hyperdags_static APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(ALPGraphBLAS::backend_hyperdags_static PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_DEBUG "CXX"
  IMPORTED_LOCATION_DEBUG "/home/belozes/graphblas/install_debug/lib/hyperdags/libgraphblas.a"
  )

list(APPEND _cmake_import_check_targets ALPGraphBLAS::backend_hyperdags_static )
list(APPEND _cmake_import_check_files_for_ALPGraphBLAS::backend_hyperdags_static "/home/belozes/graphblas/install_debug/lib/hyperdags/libgraphblas.a" )

# Import target "ALPGraphBLAS::backend_hyperdags_shared" for configuration "Debug"
set_property(TARGET ALPGraphBLAS::backend_hyperdags_shared APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(ALPGraphBLAS::backend_hyperdags_shared PROPERTIES
  IMPORTED_LOCATION_DEBUG "/home/belozes/graphblas/install_debug/lib/hyperdags/libgraphblas.so.0.7.0"
  IMPORTED_SONAME_DEBUG "libgraphblas.so.0.7.0"
  )

list(APPEND _cmake_import_check_targets ALPGraphBLAS::backend_hyperdags_shared )
list(APPEND _cmake_import_check_files_for_ALPGraphBLAS::backend_hyperdags_shared "/home/belozes/graphblas/install_debug/lib/hyperdags/libgraphblas.so.0.7.0" )

# Import target "ALPGraphBLAS::backend_shmem_static" for configuration "Debug"
set_property(TARGET ALPGraphBLAS::backend_shmem_static APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(ALPGraphBLAS::backend_shmem_static PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_DEBUG "CXX"
  IMPORTED_LOCATION_DEBUG "/home/belozes/graphblas/install_debug/lib/sequential/libgraphblas.a"
  )

list(APPEND _cmake_import_check_targets ALPGraphBLAS::backend_shmem_static )
list(APPEND _cmake_import_check_files_for_ALPGraphBLAS::backend_shmem_static "/home/belozes/graphblas/install_debug/lib/sequential/libgraphblas.a" )

# Import target "ALPGraphBLAS::backend_shmem_shared" for configuration "Debug"
set_property(TARGET ALPGraphBLAS::backend_shmem_shared APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(ALPGraphBLAS::backend_shmem_shared PROPERTIES
  IMPORTED_LOCATION_DEBUG "/home/belozes/graphblas/install_debug/lib/sequential/libgraphblas.so.0.7.0"
  IMPORTED_SONAME_DEBUG "libgraphblas.so.0.7.0"
  )

list(APPEND _cmake_import_check_targets ALPGraphBLAS::backend_shmem_shared )
list(APPEND _cmake_import_check_files_for_ALPGraphBLAS::backend_shmem_shared "/home/belozes/graphblas/install_debug/lib/sequential/libgraphblas.so.0.7.0" )

# Import target "ALPGraphBLAS::sparseblas_static" for configuration "Debug"
set_property(TARGET ALPGraphBLAS::sparseblas_static APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(ALPGraphBLAS::sparseblas_static PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_DEBUG "CXX"
  IMPORTED_LOCATION_DEBUG "/home/belozes/graphblas/install_debug/lib/sequential/libsparseblas.a"
  )

list(APPEND _cmake_import_check_targets ALPGraphBLAS::sparseblas_static )
list(APPEND _cmake_import_check_files_for_ALPGraphBLAS::sparseblas_static "/home/belozes/graphblas/install_debug/lib/sequential/libsparseblas.a" )

# Import target "ALPGraphBLAS::sparseblas_omp_static" for configuration "Debug"
set_property(TARGET ALPGraphBLAS::sparseblas_omp_static APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(ALPGraphBLAS::sparseblas_omp_static PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_DEBUG "CXX"
  IMPORTED_LOCATION_DEBUG "/home/belozes/graphblas/install_debug/lib/sequential/libsparseblas_omp.a"
  )

list(APPEND _cmake_import_check_targets ALPGraphBLAS::sparseblas_omp_static )
list(APPEND _cmake_import_check_files_for_ALPGraphBLAS::sparseblas_omp_static "/home/belozes/graphblas/install_debug/lib/sequential/libsparseblas_omp.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
