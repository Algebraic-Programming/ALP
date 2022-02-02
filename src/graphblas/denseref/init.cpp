
/*
 *   Copyright 2021 Huawei Technologies Co., Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * @author A. N. Yzelman
 * @date 2nd of February, 2017
 */

#include <graphblas/denseref/init.hpp>

template<>
grb::RC grb::init< grb::reference_dense >( const size_t s, const size_t P, void * const data ) {
	// we don't use any implementation-specific init data
	(void)data;
	// print output
	std::cerr << "Info: grb::init (reference_dense) called.\n";
	// sanity checks
	if( P > 1 ) {
		return grb::UNSUPPORTED;
	}
	if( s > 0 ) {
		return grb::PANIC;
	}
	// done
	return grb::SUCCESS;
}


template<>
grb::RC grb::finalize< grb::reference_dense >() {
	return grb::SUCCESS;
}
