
add_library(cblas INTERFACE)
if(KBLAS_ROOT)
	find_package(Kblas REQUIRED)
	target_link_libraries(cblas INTERFACE Kblas::Kblas)
	set(HEADER_NAME "kblas")
else()
	find_package(BLAS REQUIRED)
	target_link_libraries(cblas INTERFACE BLAS::BLAS)
        set(HEADER_NAME "cblas")
endif()

file(WRITE "${CMAKE_BINARY_DIR}/blas_wrapper/blas.h" "#include \"${HEADER_NAME}.h\"\n" )
target_include_directories(cblas INTERFACE "${CMAKE_BINARY_DIR}/blas_wrapper" )
