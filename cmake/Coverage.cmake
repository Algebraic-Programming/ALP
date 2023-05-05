find_package( GCov REQUIRED )
find_package( Python3 REQUIRED )
execute_process(
    COMMAND ${Python3_EXECUTABLE} -m gcovr --help
    RESULT_VARIABLE GCOVR_FOUND
    OUTPUT_QUIET
    ERROR_QUIET
)
if( NOT GCOVR_FOUND STREQUAL "0" )
    message( FATAL_ERROR "gcovr not installed, install it with ${Python3_EXECUTABLE} -m pip install gcovr" )
endif()

set( COVERAGE_REPORT_DIR "${PROJECT_BINARY_DIR}/coverage" )
string(JOIN + _COVERAGE_TITLE "GraphBLAS_${VERSION}" ${AVAILABLE_TEST_BACKENDS})
file( MAKE_DIRECTORY "${COVERAGE_REPORT_DIR}" )
message( STATUS "COVERAGE_REPORT_DIR: ${COVERAGE_REPORT_DIR}" )

function( create_coverage_command command_name output_file output_switch )
    message( STATUS "COVERAGE_REPORT_DIR: ${COVERAGE_REPORT_DIR}" )
    add_custom_target( ${command_name}
        COMMAND echo ${COVERAGE_REPORT_DIR}/${output_file}
		COMMAND ${Python3_EXECUTABLE} -m gcovr ${output_switch}
			--sort-percentage
			--print-summary
            --html-title ${_COVERAGE_TITLE}
			--exclude-directories "/usr/*"
			--root ${PROJECT_SOURCE_DIR}
			--output ${COVERAGE_REPORT_DIR}/${output_file}
            ${ARGN}
		COMMAND echo --> Generated report: ${COVERAGE_REPORT_DIR}/${output_file} <--
		WORKING_DIRECTORY "${PROJECT_BINARY_DIR}"
		COMMENT "producing JSON coverage report in ${COVERAGE_REPORT_DIR}"
		VERBATIM
        COMMAND_EXPAND_LISTS
	)
endfunction()

add_custom_target( clean_coverage
    COMMAND find "coverage" -mindepth 1 -delete
    COMMAND find . -name "*.gcno" -delete
    COMMAND find . -name "*.gcda" -delete
    WORKING_DIRECTORY "${PROJECT_BINARY_DIR}"
    COMMENT "cleaning coverage-related in ${COVERAGE_REPORT_DIR}"
)