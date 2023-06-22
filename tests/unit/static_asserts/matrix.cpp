
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

#include <graphblas.hpp>


int main() {
	typedef grb::Matrix< double > MatrixType;
	static_assert( std::is_same<
			std::pair<
				std::pair< const size_t, const size_t >,
				const MatrixType::value_type
			>,
			std::iterator_traits< typename MatrixType::const_iterator >::value_type
		>::value, "Matrix iterator has an unexpected value type" );
	return 0;
}

