#
#   Copyright 2023 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

find_package( GCov REQUIRED )
find_package( Gcovr REQUIRED )

set( COVERAGE_REPORT_DIR "${PROJECT_BINARY_DIR}/coverage" )
string( JOIN + _COVERAGE_TITLE "GraphBLAS_${VERSION}" ${AVAILABLE_TEST_BACKENDS} )
file( MAKE_DIRECTORY "${COVERAGE_REPORT_DIR}" )
message( STATUS "Directory of coverage reports: ${COVERAGE_REPORT_DIR}" )

function( create_coverage_command command_name output_file output_switch )
    add_custom_target( ${command_name}
		COMMAND ${GCOV_COMMAND} ${output_switch}
            --gcov-executable ${GCOV_EXECUTABLE}
			--sort-percentage
			--print-summary
            --html-title ${_COVERAGE_TITLE}
			--exclude-directories "/usr/*"
			--root ${PROJECT_SOURCE_DIR}
			--output ${COVERAGE_REPORT_DIR}/${output_file}
            ${ARGN}
		COMMAND echo "--> Generated report: ${COVERAGE_REPORT_DIR}/${output_file} <--"
		WORKING_DIRECTORY "${PROJECT_BINARY_DIR}"
		COMMENT "producing coverage report into ${COVERAGE_REPORT_DIR}/${output_file}"
		VERBATIM
        COMMAND_EXPAND_LISTS
	)
endfunction()

add_custom_target( coverage_clean
    COMMAND find "coverage" -mindepth 1 -delete
    COMMAND find . -name "*.gcno" -delete
    COMMAND find . -name "*.gcda" -delete
    WORKING_DIRECTORY "${PROJECT_BINARY_DIR}"
    COMMENT "cleaning coverage-related in ${COVERAGE_REPORT_DIR}"
)
