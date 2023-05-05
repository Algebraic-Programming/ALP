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

# prefer gcovr installed via pip, as it may be more recent and also allows
# user-local installations: we need python3 explicitly accessible
if( NOT Python3_FOUND )
    find_package( Python3 )
endif()

if( Python3_FOUND )
    # test wether gcovr is already installed: do not install it if not
    # let the user choose which installation method (distro package manager,
    # pip3 global, pip3 local, custom)
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -m gcovr --help
        RESULT_VARIABLE PIP_GCOVR_FOUND
        OUTPUT_QUIET
        ERROR_QUIET
    )
    if( NOT PIP_GCOVR_FOUND STREQUAL "0" )
        message( STATUS "gcovr not installed via pip (preferred),"
            "you may install it with: ${Python3_EXECUTABLE} -m pip install gcovr"
        )
    else()
        set( GCOV_COMMAND ${Python3_EXECUTABLE} -m gcovr )
    endif()
endif()

# if not found via pip, then look for a "standalone" executable in standard paths
if( NOT GCOV_COMMAND )
    find_program( GCOV_COMMAND NAMES gcovr )
endif()

if( NOT GCOV_COMMAND )
    message( WARNING "gcovr was found neither as a Python3 module"
        "nor as standalone executable within system directories"
    )
endif()

include( FindPackageHandleStandardArgs )
find_package_handle_standard_args( Gcovr REQUIRED_VARS GCOV_COMMAND )
