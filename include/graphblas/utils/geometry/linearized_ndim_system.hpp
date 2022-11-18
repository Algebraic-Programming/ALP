
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

#ifndef _H_GRB_ALGORITHMS_GEOMETRY_NDIM_SYSTEM_LINEARIZER
#define _H_GRB_ALGORITHMS_GEOMETRY_NDIM_SYSTEM_LINEARIZER

#include <cstddef>
#include <algorithm>
#include <vector>
#include <utility>
#include <stdexcept>
#include <cassert>
#include <string>
#include <cstddef>

#include "ndim_system.hpp"
#include "linearized_ndim_iterator.hpp"
// #include "array_vector_storage.hpp"

/**
 * @file linearized_ndim_system.cpp
 * @author Alberto Scolari (alberto.scolari@huawei.com)
 * Definition of \p LinearizedNDimSystem.
 *
 * @date 2022-10-24
 */

namespace grb {
	namespace utils {
		namespace geometry {

			/**
			 * Extends a \p NDimSystem by linearizing it, i.e. it provides facilities to map a vector in
			 * NDimSystem#dimensions() dimensions to a linear value ranging from \a 0 to #system_size()
			 * and vice versa. Such a linearized representation allows user logic to iterate over the system:
			 * iterators are indeed available via #begin()/#end().
			 *
			 * Further facilities are methods to map users' vectors from linear to NDimSystem#dimensions()-dimensional
			 * or vice versa and also to "retaget" the system, i.e. to represent a system of same dimensionality
			 * but different sizes.
			 *
			 * @tparam SizeType integral type to store the size of each dimension
			 * @tparam InternalStorageType internal vector type to store the sizes
			 */
			template<
				typename SizeType,
				typename InternalVectorType
			> class LinearizedNDimSystem: public NDimSystem< SizeType, InternalVectorType > {

			public:
				static_assert( std::is_integral< SizeType >::value, "SizeType must be an integral type");

				using BaseType = NDimSystem< SizeType, InternalVectorType >;
				using SelfType = LinearizedNDimSystem< SizeType, InternalVectorType >;
				using VectorType = typename BaseType::VectorType;

				using VectorReference = typename BaseType::VectorReference;
				using ConstVectorReference = typename BaseType::ConstVectorReference;
				using VectorStorageType = typename VectorType::VectorStorageType;
				using ConstVectorStorageType = typename VectorType::ConstVectorStorageType;
				using Iterator = LinearizedNDimIterator< SizeType, InternalVectorType >;

				template< typename IterT > LinearizedNDimSystem( IterT begin, IterT end) noexcept :
					BaseType( begin, end ),
					offsets( std::distance( begin, end ) )
				{
					this->_system_size = compute_offsets( begin, end, this->offsets.begin() ) ;
				}

				LinearizedNDimSystem( const std::vector< size_t > &_sizes ) noexcept :
					LinearizedNDimSystem( _sizes.cbegin(), _sizes.cend() ) {}

				LinearizedNDimSystem( size_t _dimensions, size_t max_value ) noexcept :
					BaseType( _dimensions, max_value ),
					offsets( _dimensions ),
					_system_size( _dimensions )
				{
					SizeType v{1};
					for( size_t i{0}; i < _dimensions; i++ ) {
						this->offsets[i] = v;
						v *= max_value;
					}
					this->_system_size = v;
				}

				LinearizedNDimSystem() = delete;

				LinearizedNDimSystem( const SelfType &original ) = default;

				LinearizedNDimSystem( SelfType &&original ) noexcept:
					BaseType( std::move(original) ), offsets( std::move( original.offsets ) ),
					_system_size( original._system_size ) {
						original._system_size = 0;
				}

				~LinearizedNDimSystem() {}

				SelfType& operator=( const SelfType & ) = default;

				SelfType& operator=( SelfType &&original ) = delete;

				inline size_t system_size() const {
					return this->_system_size;
				}

				inline ConstVectorReference get_offsets() const {
					return this->offsets;
				}

				void linear_to_ndim( size_t linear, VectorReference output ) const {
					if( linear > this->_system_size ) {
						throw std::range_error( "linear value beyond system" );
					}
					for( size_t _i{ this->offsets.dimensions() }; _i > 0; _i-- ) {
						const size_t dim{ _i - 1 };
						const size_t coord{ linear / this->offsets[dim] };
						output[dim] = coord;
						linear -= ( coord * this->offsets[dim] );
					}
					assert( linear == 0 );
				}

				size_t ndim_to_linear_check( ConstVectorReference ndim_vector) const {
					return this->ndim_to_linear_check( ndim_vector.storage() );
				}

				size_t ndim_to_linear_check( ConstVectorStorageType ndim_vector ) const {
					size_t linear { 0 };
					for( size_t i { 0 }; i < this->dimensions(); i++ ) {
						if( ndim_vector[i] >= this->get_sizes()[i] ) {
							throw std::invalid_argument( "input vector beyond system sizes" );
						}
					}
					return ndim_to_linear( ndim_vector );
				}

				size_t ndim_to_linear( ConstVectorReference ndim_vector) const {
					return this->ndim_to_linear( ndim_vector.storage() );
				}

				size_t ndim_to_linear( ConstVectorStorageType ndim_vector ) const {
					size_t linear { 0 };
					for( size_t i { 0 }; i < this->dimensions(); i++ ) {
						linear += this->offsets[i] * ndim_vector[i];
					}
					return linear;
				}

				// probably same as ndim_to_linear !!!
				size_t ndim_to_linear_offset( ConstVectorStorageType ndim_vector ) const {
					size_t linear{ 0 };
					size_t steps{ 1 };
					for( size_t i{ 0 }; i < this->dimensions(); i++ ) {
						linear += steps * ndim_vector[i];
						steps *= this->_sizes[i];
					}
					return linear;
				}

				// must be same dimensionality
				void retarget( ConstVectorReference _new_sizes ) {
					if( _new_sizes.dimensions() != this->_sizes.dimensions() ) {
						throw std::invalid_argument("new system must have same dimensions as previous: new "
							+ std::to_string( _new_sizes.dimensions() ) + ", old "
							+ std::to_string( this->_sizes.dimensions() ) );
					}
					this->_sizes = _new_sizes; // copy
					this->_system_size = compute_offsets( _new_sizes.begin(), _new_sizes.end(), this->offsets.begin() ) ;
				}

				Iterator begin() const {
					return Iterator( *this );
				}

				Iterator end() const {
					return Iterator::make_system_end_iterator( *this );
				}

			private:

				VectorType offsets;
				size_t _system_size;

				 template<
					typename IterIn,
					typename IterOut
				> static size_t compute_offsets( IterIn in_begin, IterIn in_end, IterOut out_begin ) {
					size_t prod{1};
					for( ; in_begin != in_end; ++in_begin, ++out_begin ) {
						*out_begin = prod;
						prod *= *in_begin;
					}
					return prod;
				}
			};


		} // namespace geometry
	} // namespace utils
} // namespace grb

#endif // _H_GRB_ALGORITHMS_GEOMETRY_NDIM_SYSTEM_LINEARIZER
