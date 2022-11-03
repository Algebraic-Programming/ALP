assert_valid_variables(INCLUDE_INSTALL_DIR)
add_library(cblas INTERFACE)
if(KBLAS_ROOT)
	find_package(Kblas REQUIRED)
	target_link_libraries(cblas INTERFACE Kblas::Kblas)
	set(HEADER_NAME "kblas")
else()
	find_package(BLAS REQUIRED)
        add_library( extBlas::extBlas UNKNOWN IMPORTED )
        set_target_properties( extBlas::extBlas
                PROPERTIES
                IMPORTED_LOCATION "${BLAS_LIBRARIES}"
		INTERFACE_LINK_OPTIONS "${BLAS_LINKER_FLAGS}"
                #INTERFACE_INCLUDE_DIRECTORIES ${}
        )

	target_link_libraries(cblas INTERFACE extBlas::extBlas)
        set(HEADER_NAME "cblas")
endif()

file(WRITE "${CMAKE_BINARY_DIR}/blas_wrapper/blas.h" "#include \"${HEADER_NAME}.h\"\n" )
#target_include_directories(cblas INTERFACE "${CMAKE_BINARY_DIR}/blas_wrapper" )

target_include_directories( cblas INTERFACE
        $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/blas_wrapper>
        $<INSTALL_INTERFACE:.>
)
install(FILES "${CMAKE_BINARY_DIR}/blas_wrapper/blas.h" DESTINATION "${INCLUDE_INSTALL_DIR}/blas_wrapper")

install(
	TARGETS cblas EXPORT GraphBLASTargets
	INCLUDES DESTINATION "${INCLUDE_INSTALL_DIR}/blas_wrapper"
)
