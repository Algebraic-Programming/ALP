#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "ALPGraphBLAS::alp_utils_static" for configuration "Release"
set_property(TARGET ALPGraphBLAS::alp_utils_static APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(ALPGraphBLAS::alp_utils_static PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "C"
  IMPORTED_LOCATION_RELEASE "/host/install_lpf/lib/libalp_utils.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS ALPGraphBLAS::alp_utils_static )
list(APPEND _IMPORT_CHECK_FILES_FOR_ALPGraphBLAS::alp_utils_static "/host/install_lpf/lib/libalp_utils.a" )

# Import target "ALPGraphBLAS::alp_utils_dynamic" for configuration "Release"
set_property(TARGET ALPGraphBLAS::alp_utils_dynamic APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(ALPGraphBLAS::alp_utils_dynamic PROPERTIES
  IMPORTED_LOCATION_RELEASE "/host/install_lpf/lib/libalp_utils.so"
  IMPORTED_SONAME_RELEASE "libalp_utils.so"
  )

list(APPEND _IMPORT_CHECK_TARGETS ALPGraphBLAS::alp_utils_dynamic )
list(APPEND _IMPORT_CHECK_FILES_FOR_ALPGraphBLAS::alp_utils_dynamic "/host/install_lpf/lib/libalp_utils.so" )

# Import target "ALPGraphBLAS::backend_hyperdags_static" for configuration "Release"
set_property(TARGET ALPGraphBLAS::backend_hyperdags_static APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(ALPGraphBLAS::backend_hyperdags_static PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "/host/install_lpf/lib/hyperdags/libgraphblas.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS ALPGraphBLAS::backend_hyperdags_static )
list(APPEND _IMPORT_CHECK_FILES_FOR_ALPGraphBLAS::backend_hyperdags_static "/host/install_lpf/lib/hyperdags/libgraphblas.a" )

# Import target "ALPGraphBLAS::backend_hyperdags_shared" for configuration "Release"
set_property(TARGET ALPGraphBLAS::backend_hyperdags_shared APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(ALPGraphBLAS::backend_hyperdags_shared PROPERTIES
  IMPORTED_LOCATION_RELEASE "/host/install_lpf/lib/hyperdags/libgraphblas.so.0.7.0"
  IMPORTED_SONAME_RELEASE "libgraphblas.so.0.7.0"
  )

list(APPEND _IMPORT_CHECK_TARGETS ALPGraphBLAS::backend_hyperdags_shared )
list(APPEND _IMPORT_CHECK_FILES_FOR_ALPGraphBLAS::backend_hyperdags_shared "/host/install_lpf/lib/hyperdags/libgraphblas.so.0.7.0" )

# Import target "ALPGraphBLAS::backend_bsp1d_static" for configuration "Release"
set_property(TARGET ALPGraphBLAS::backend_bsp1d_static APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(ALPGraphBLAS::backend_bsp1d_static PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "/host/install_lpf/lib/spmd/libgraphblas.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS ALPGraphBLAS::backend_bsp1d_static )
list(APPEND _IMPORT_CHECK_FILES_FOR_ALPGraphBLAS::backend_bsp1d_static "/host/install_lpf/lib/spmd/libgraphblas.a" )

# Import target "ALPGraphBLAS::backend_bsp1d_shared" for configuration "Release"
set_property(TARGET ALPGraphBLAS::backend_bsp1d_shared APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(ALPGraphBLAS::backend_bsp1d_shared PROPERTIES
  IMPORTED_LOCATION_RELEASE "/host/install_lpf/lib/spmd/libgraphblas.so.0.7.0"
  IMPORTED_SONAME_RELEASE "libgraphblas.so.0.7.0"
  )

list(APPEND _IMPORT_CHECK_TARGETS ALPGraphBLAS::backend_bsp1d_shared )
list(APPEND _IMPORT_CHECK_FILES_FOR_ALPGraphBLAS::backend_bsp1d_shared "/host/install_lpf/lib/spmd/libgraphblas.so.0.7.0" )

# Import target "ALPGraphBLAS::backend_hybrid_static" for configuration "Release"
set_property(TARGET ALPGraphBLAS::backend_hybrid_static APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(ALPGraphBLAS::backend_hybrid_static PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "/host/install_lpf/lib/hybrid/libgraphblas.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS ALPGraphBLAS::backend_hybrid_static )
list(APPEND _IMPORT_CHECK_FILES_FOR_ALPGraphBLAS::backend_hybrid_static "/host/install_lpf/lib/hybrid/libgraphblas.a" )

# Import target "ALPGraphBLAS::backend_hybrid_shared" for configuration "Release"
set_property(TARGET ALPGraphBLAS::backend_hybrid_shared APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(ALPGraphBLAS::backend_hybrid_shared PROPERTIES
  IMPORTED_LOCATION_RELEASE "/host/install_lpf/lib/hybrid/libgraphblas.so.0.7.0"
  IMPORTED_SONAME_RELEASE "libgraphblas.so.0.7.0"
  )

list(APPEND _IMPORT_CHECK_TARGETS ALPGraphBLAS::backend_hybrid_shared )
list(APPEND _IMPORT_CHECK_FILES_FOR_ALPGraphBLAS::backend_hybrid_shared "/host/install_lpf/lib/hybrid/libgraphblas.so.0.7.0" )

# Import target "ALPGraphBLAS::backend_shmem_static" for configuration "Release"
set_property(TARGET ALPGraphBLAS::backend_shmem_static APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(ALPGraphBLAS::backend_shmem_static PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "/host/install_lpf/lib/sequential/libgraphblas.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS ALPGraphBLAS::backend_shmem_static )
list(APPEND _IMPORT_CHECK_FILES_FOR_ALPGraphBLAS::backend_shmem_static "/host/install_lpf/lib/sequential/libgraphblas.a" )

# Import target "ALPGraphBLAS::backend_shmem_shared" for configuration "Release"
set_property(TARGET ALPGraphBLAS::backend_shmem_shared APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(ALPGraphBLAS::backend_shmem_shared PROPERTIES
  IMPORTED_LOCATION_RELEASE "/host/install_lpf/lib/sequential/libgraphblas.so.0.7.0"
  IMPORTED_SONAME_RELEASE "libgraphblas.so.0.7.0"
  )

list(APPEND _IMPORT_CHECK_TARGETS ALPGraphBLAS::backend_shmem_shared )
list(APPEND _IMPORT_CHECK_FILES_FOR_ALPGraphBLAS::backend_shmem_shared "/host/install_lpf/lib/sequential/libgraphblas.so.0.7.0" )

# Import target "ALPGraphBLAS::sparseblas_static" for configuration "Release"
set_property(TARGET ALPGraphBLAS::sparseblas_static APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(ALPGraphBLAS::sparseblas_static PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "/host/install_lpf/lib/sequential/libsparseblas.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS ALPGraphBLAS::sparseblas_static )
list(APPEND _IMPORT_CHECK_FILES_FOR_ALPGraphBLAS::sparseblas_static "/host/install_lpf/lib/sequential/libsparseblas.a" )

# Import target "ALPGraphBLAS::sparseblas_omp_static" for configuration "Release"
set_property(TARGET ALPGraphBLAS::sparseblas_omp_static APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(ALPGraphBLAS::sparseblas_omp_static PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "/host/install_lpf/lib/sequential/libsparseblas_omp.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS ALPGraphBLAS::sparseblas_omp_static )
list(APPEND _IMPORT_CHECK_FILES_FOR_ALPGraphBLAS::sparseblas_omp_static "/host/install_lpf/lib/sequential/libsparseblas_omp.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
