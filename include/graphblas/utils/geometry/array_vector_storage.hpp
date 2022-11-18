
/*
 *   Copyright 2022 Huawei Technologies Co., Ltd.
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

/**
 * @file array_vector_storage.cpp
 * @author Alberto Scolari (alberto.scolari@huawei.com)
 * Extension of std::array<> exposing a larger interface and the underlying
 * 	storage structure.
 *
 * @date 2022-10-24
 */

#ifndef _H_GRB_ALGORITHMS_GEOMETRY_ARRAY_VECTOR_STORAGE
#define _H_GRB_ALGORITHMS_GEOMETRY_ARRAY_VECTOR_STORAGE

#include <array>
#include <stdexcept>
#include <algorithm>
#include <cstddef>

namespace grb {
	namespace utils {
		namespace geometry {

			/**
			 * Array with fixed size based on std::array with an interface compliant to what other classes
			 * in the geometry namespace expect, like storage() and dimensions() methods.
			 *
			 * It describes a vector of dimensions #dimensions().
			 *
			 * @tparam DataType the data type of the vector elements
			 * @tparam DIMS the dimensions of the vector
			 */
			template<
				typename DataType,
				size_t DIMS
			> class ArrayVectorStorage: public std::array< DataType, DIMS > {

			public:

				using VectorStorageType = std::array< DataType, DIMS >&;
				using ConstVectorStorageType = const std::array< DataType, DIMS >&;

				ArrayVectorStorage( size_t _dimensions ) {
					static_assert( DIMS > 0, "cannot allocate 0-sized array" );
					if( _dimensions != DIMS ) {
						throw std::invalid_argument("given dimensions must match the type dimensions");
					}
				}

				ArrayVectorStorage() = delete;

				// only copy constructor/assignment, since there's no external storage
				ArrayVectorStorage( const ArrayVectorStorage< DataType, DIMS > &o ) noexcept {
					std::copy_n( o.cbegin(), DIMS, this->begin() );
				}

				ArrayVectorStorage( ArrayVectorStorage< DataType, DIMS > &&o ) = delete;

				ArrayVectorStorage< DataType, DIMS >& operator=(
					const ArrayVectorStorage< DataType, DIMS > &original
				) noexcept {
					std::copy_n( original.begin(), DIMS, this->begin() );
					return *this;
				}

				ArrayVectorStorage< DataType, DIMS >& operator=( ArrayVectorStorage< DataType, DIMS > &&original ) = delete;

				constexpr size_t dimensions() const {
					return DIMS;
				}

				inline VectorStorageType storage() {
					return *this;
				}

				inline ConstVectorStorageType storage() const {
					return *this;
				}
			};

		} // namespace geometry
	} // namespace utils
} // namespace grb

#endif // _H_GRB_ALGORITHMS_GEOMETRY_ARRAY_VECTOR_STORAGE
