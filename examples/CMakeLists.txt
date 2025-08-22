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

assert_defined_variables( WITH_REFERENCE_BACKEND WITH_OMP_BACKEND )

# target listing all examples, to build them at once with 'make examples'
add_custom_target( examples)

if( WITH_REFERENCE_BACKEND )
	add_executable( sp_reference sp.cpp )
	target_link_libraries( sp_reference backend_reference common_flags )
	add_dependencies( examples sp_reference )
endif()

if( WITH_OMP_BACKEND )
	add_executable( sp_reference_omp sp.cpp )
	target_link_libraries( sp_reference_omp backend_reference_omp common_flags )
	add_dependencies( examples sp_reference_omp )
endif()

