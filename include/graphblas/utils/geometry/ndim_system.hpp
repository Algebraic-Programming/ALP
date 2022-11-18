
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

#ifndef _H_GRB_ALGORITHMS_GEOMETRY_NDIM_SYSTEM
#define _H_GRB_ALGORITHMS_GEOMETRY_NDIM_SYSTEM

#include <cstddef>
#include <algorithm>
#include <vector>
#include <utility>
#include <type_traits>
#include <cstddef>

#include "array_vector_storage.hpp"

/**
 * @file ndim_system.cpp
 * @author Alberto Scolari (alberto.scolari@huawei.com)
 * Definition of \p NDimSystem.
 *
 * @date 2022-10-24
 */

namespace grb {
	namespace utils {
		namespace geometry {

			/**
			 * Describes a #dimensions()-dimensional system by storing its size along each dimension.
			 *
			 * It is meant to represent a grid of #dimensions() dimensions and size #get_sizes()[d]
			 * for each dimension \a d in the interval <em>[0, #dimensions())<\em>.
			 *
			 * @tparam SizeType integral type to store the size of each dimension
			 * @tparam InternalStorageType internal vector type to store the sizes
			 */
			template<
				typename SizeType,
				typename InternalVectorType
			> class NDimSystem {

			public:
				static_assert( std::is_integral< SizeType >::value, "SizeType must be an integral type");

				using VectorType = InternalVectorType;
				using VectorReference = VectorType&;
				using ConstVectorReference = const VectorType&;
				using SelfType = NDimSystem< SizeType, InternalVectorType >;

				/**
				 * Construct a new NDimSystem object from an iterable range.
				 *
				 * The dimension is computed as \a std::distance(begin,end), i.e.
				 * \p IterT should be a random-access iterator for performance.
				 *
				 * @tparam IterT iterator type
				 * @param begin range begin
				 * @param end end of range
				 */
				template< typename IterT > NDimSystem( IterT begin, IterT end) noexcept :
					_sizes( std::distance( begin, end ) )
				{
					std::copy( begin, end, this->_sizes.begin() );
				}

				/**
				 * Construct a new NDimSystem object from an std::vector<>, taking its values
				 * as system sizes and its length as number of dimensions.
				 */
				NDimSystem( const std::vector< size_t > &_sizes ) noexcept :
					SelfType( _sizes.cbegin(), _sizes.cend() ) {}

				/**
				 * Construct a new NDimSystem object of dimensions \p dimensions
				 *  and with all sizes initialized to \p max_size
				 */
				NDimSystem( size_t _dimensions, size_t max_size ) noexcept :
					_sizes( _dimensions )
				{
					std::fill_n( this->_sizes.begin(), _dimensions, max_size );
				}

				NDimSystem() = delete;

				NDimSystem( const SelfType & ) = default;

				// NDimSystem( SelfType && ) = default;

				// NDimSystem( SelfType &&original ) noexcept: _sizes( std::move( original._sizes ) ) {}
				NDimSystem( SelfType && ) = delete;

				~NDimSystem() {}

				SelfType & operator=( const SelfType &original ) = default;

				SelfType & operator=( SelfType &&original ) = delete;

				inline size_t dimensions() const noexcept {
					return _sizes.dimensions();
				}

				/**
				 * Get the sizes of the represented system as an iterable \p InternalStorageType
				 * 	object.
				 */
				inline ConstVectorReference get_sizes() const noexcept {
					return this->_sizes;
				}

			protected:

				InternalVectorType _sizes;
			};

		} // namespace geometry
	} // namespace utils
} // namespace grb

#endif // _H_GRB_ALGORITHMS_GEOMETRY_NDIM_SYSTEM
