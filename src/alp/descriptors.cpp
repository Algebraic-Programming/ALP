
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

#include <alp/descriptors.hpp>

std::string alp::descriptors::toString( const alp::Descriptor descr ) {
	std::ostringstream os;
	if( descr == 0 ) {
		os << "no-op descriptor\n";
	} else {
		os << "specialised descriptor:\n";
		if( descr & invert_mask ) {
			os << " inverted mask\n";
		}
		if( descr & transpose_matrix ) {
			os << " transpose input matrix\n";
		}
		if( descr & no_duplicates ) {
			os << " user guarantees no duplicate coordinate on input\n";
		}
		if( descr & structural ) {
			os << " mask must be interpreted structurally, and not by value\n";
		}
		if( descr & dense ) {
			os << " user guarantees all vectors in this call are dense\n";
		}
		if( descr & add_identity ) {
			os << " an identity matrix is added to the input matrix\n";
		}
		if( descr & use_index ) {
			os << " instead of using input vector elements, use their index "
				  "instead\n";
		}
		if( descr & explicit_zero ) {
			os << " the operation should take zeroes into account explicitly "
				  "when computing output\n";
		}
		if( descr & no_casting ) {
			os << " disallow casting between types during the requested "
				  "computation\n";
		}
	}
	return os.str();
}
