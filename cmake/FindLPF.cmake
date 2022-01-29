#
#   Copyright 2021 Huawei Technologies Co., Ltd.
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

#[===================================================================[
Find LPF

LPF_INSTALL_PATH path to look for LPF (required)
creates:
  LPFRUN path to the LPf runner
  LPFCPP path to the LPF compiler wrapper

loads the LFP CMake module
#]===================================================================]

find_package( LPF REQUIRED CONFIG
	PATHS "${LPF_INSTALL_PATH}/lib"
)

find_file( LPFRUN "lpfrun" REQUIRED
	PATHS "${LPF_INSTALL_PATH}/bin"
	DOC "LPF runner script"
)

find_file( LPFCPP "lpfcxx" REQUIRED
	PATHS "${LPF_INSTALL_PATH}/bin"
	DOC "LPF CXX compiler"
)

mark_as_advanced( LPFRUN LPFCPP )
include( FindPackageHandleStandardArgs )
find_package_handle_standard_args( LPF LPFRUN LPFCPP )
