
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


#include <string>

#include <graphblas/ascend/grid.hpp>

namespace alp {

	namespace internal {

		iGrid *igrid;
	}
}

alp::internal::iGrid::iGrid( size_t proc, size_t prob ) {

	process_order = proc;
	problem_order = prob;
}

size_t alp::internal::iGrid::getProcessOrder() const noexcept {

	return process_order;
}

size_t alp::internal::iGrid::getProblemOrder() const noexcept {

	return problem_order;
}

std::string alp::internal::iGrid::processSize( const size_t k ) const noexcept {

	return "p" + std::to_string( k );
}

std::string alp::internal::iGrid::processMode( const size_t k ) const noexcept {

	return "a" + std::to_string( k );
}

std::string alp::internal::iGrid::problemSize( const size_t k ) const noexcept {

	return "n" + std::to_string( k );
}

std::string alp::internal::iGrid::problemMode( const size_t k ) const noexcept {

	return "i" + std::to_string( k );
}

std::string alp::internal::iGrid::problemMainMode( const size_t k ) const noexcept {

	return "z" + std::to_string( k );
}

std::string alp::internal::iGrid::problemTileMode( const size_t k ) const noexcept {

	return "t" + std::to_string( k );
}

std::string alp::internal::iGrid::tileSize( const size_t k ) const noexcept {

	return "tile_size" + std::to_string( k );
}

