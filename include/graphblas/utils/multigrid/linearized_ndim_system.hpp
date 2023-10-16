
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
 * @file linearized_ndim_system.cpp
 * @author Alberto Scolari (alberto.scolari@huawei.com)
 * Definition of \p LinearizedNDimSystem.
 */

#ifndef _H_GRB_ALGORITHMS_MULTIGRID_NDIM_SYSTEM_LINEARIZER
#define _H_GRB_ALGORITHMS_MULTIGRID_NDIM_SYSTEM_LINEARIZER

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "linearized_ndim_iterator.hpp"
#include "ndim_system.hpp"

namespace grb {
	namespace utils {
		namespace multigrid {

			/**
			 * Extends an NDimSystem by linearizing it, i.e. it provides facilities to map a vector in
			 * NDimSystem#dimensions() dimensions to a linear value ranging from \a 0 to #system_size() (excluded)
			 * and vice versa. Such a linearized representation allows user logic to iterate over the system:
			 * iterators are indeed available via #begin()/#end(). Consecutive system elements along dimension 0
			 * are mapped to consecutive linear values, while elements consecutive along dimension 1
			 * are mapped at offset #get_offsets()[1] = #get_sizes()[0], elements along dimension 2
			 * are mapped at offset #get_offsets()[2] = #get_sizes()[0] * #get_sizes()[0], and so on.
			 *
			 * Further facilities are methods to map users' vectors from linear to NDimSystem#dimensions()-dimensional
			 * or vice versa and also to "retaget" the system, i.e. to represent a system of same dimensionality
			 * but different sizes; this last feature is a mere performance optimization aimed at
			 * reusing existing objects instead of deleting them and allocating new memory.
			 *
			 * @tparam SizeType integral type to store the size of each dimension
			 * @tparam InternalStorageType internal vector type to store the sizes
			 */
			template<
				typename SizeType,
				typename InternalVectorType
			> class LinearizedNDimSystem : public NDimSystem< SizeType, InternalVectorType > {
			public:
				static_assert( std::is_integral< SizeType >::value, "SizeType must be an integral type" );

				using BaseType = NDimSystem< SizeType, InternalVectorType >;
				using SelfType = LinearizedNDimSystem< SizeType, InternalVectorType >;
				using VectorType = typename BaseType::VectorType;
				using VectorReference = typename BaseType::VectorReference;
				using ConstVectorReference = typename BaseType::ConstVectorReference;
				using VectorStorageType = typename VectorType::VectorStorageType;
				using ConstVectorStorageType = typename VectorType::ConstVectorStorageType;
				using Iterator = LinearizedNDimIterator< SizeType, InternalVectorType >;

				/**
				 * Construct a new LinearizedNDimSystem object from an iterable range,
				 * where each iterator's position stores the size along each dimension; example:
				 * *begin is the size along dimension 0, *(++begin) is the size along dimension 1 ...
				 */
				template< typename IterT >
				LinearizedNDimSystem(
					IterT begin,
					IterT end
				) noexcept :
					BaseType( begin, end ),
					_offsets( std::distance( begin, end ) )
				{
					this->_system_size = compute_range_product( begin, end, this->_offsets.begin() );
				}

				/**
				 * Construct a new LinearizedNDimSystem object with dimensions \p _sizes.size()
				 * and sizes stored in \p _sizes.
				 */
				LinearizedNDimSystem( const std::vector< size_t > & _sizes ) noexcept :
					LinearizedNDimSystem( _sizes.cbegin(), _sizes.cend() ) {}

				/**
				 * Construct a new LinearizedNDimSystem object with \p _dimensions dimensions
				 * and sizes all equal to \p max_value.
				 */
				LinearizedNDimSystem(
					size_t _dimensions,
					size_t _size
				) noexcept :
					BaseType( _dimensions, _size ),
					_offsets( _dimensions ),
					_system_size( _dimensions )
				{
					SizeType v = 1;
					for( size_t i = 0; i < _dimensions; i++ ) {
						this->_offsets[ i ] = v;
						v *= _size;
					}
					this->_system_size = v;
				}

				LinearizedNDimSystem() = delete;

				LinearizedNDimSystem( const SelfType & original ) = default;

				LinearizedNDimSystem( SelfType && original ) noexcept :
					BaseType( std::move( original ) ),
					_offsets( std::move( original._offsets ) ),
					_system_size( original._system_size )
				{
					original._system_size = 0;
				}

				~LinearizedNDimSystem() {}

				SelfType & operator=( const SelfType & ) = default;

				SelfType & operator=( SelfType && original ) = delete;

				/**
				 * Computes the size of the system, i.e. its number of elements;
				 * this corresponds to the product of the sizes along all dimensions.
				 */
				inline size_t system_size() const {
					return this->_system_size;
				}

				/**
				 * Get the offsets of the system, i.e. by how many linear elements moving along
				 * a dimension corresponds to.
				 */
				inline ConstVectorReference get_offsets() const {
					return this->_offsets;
				}

				/**
				 * Computes the #dimensions()-dimensions vector the linear value in input corresponds to.
				 *
				 * @param[in] linear linear index
				 * @param[out] output output vector \p linear corresponds to
				 */
				void linear_to_ndim(
					size_t linear,
					VectorReference output
				) const {
					if( linear > this->_system_size ) {
						throw std::range_error( "linear value beyond system" );
					}
					for( size_t _i = this->_offsets.dimensions(); _i > 0; _i-- ) {
						const size_t dim = _i - 1;
						const size_t coord = linear / this->_offsets[ dim ];
						output[ dim ] = coord;
						linear -= ( coord * this->_offsets[ dim ] );
					}
					assert( linear == 0 );
				}

				/**
				 * Computes the linear value the input vector corresponds to; this method takes in input
				 * a const reference to \p InternalVectorType and checks whether each value in the input
				 * vector \p ndim_vector is within the system sizes (otherwise it throws).
				 */
				size_t ndim_to_linear_check( ConstVectorReference ndim_vector ) const {
					return this->ndim_to_linear_check( ndim_vector.storage() );
				}

				/**
				 * Computes the linear value the input vector corresponds to; this method takes in input
				 * a const reference to the underlying storage of \p InternalVectorType and checks
				 * whether each value in the input vector \p ndim_vector is within the system sizes
				 * (otherwise it throws).
				 */
				size_t ndim_to_linear_check( ConstVectorStorageType ndim_vector ) const {
					size_t linear = 0;
					for( size_t i = 0; i < this->dimensions(); i++ ) {
						if( ndim_vector[ i ] >= this->get_sizes()[ i ] ) {
							throw std::invalid_argument( "input vector beyond system sizes" );
						}
					}
					return ndim_to_linear( ndim_vector );
				}

				/**
				 * Computes the linear value the input vector corresponds to; this method takes in input
				 * a const reference to \p InternalVectorType but does not check whether each value in the input
				 * vector \p ndim_vector is within the system sizes.
				 */
				size_t ndim_to_linear( ConstVectorReference ndim_vector ) const {
					return this->ndim_to_linear( ndim_vector.storage() );
				}

				/**
				 * Computes the linear value the input vector corresponds to; this method takes in input
				 * a const reference to the underlying storage of \p InternalVectorType but does not check
				 * whether each value in the input vector \p ndim_vector is within the system sizes.
				 */
				size_t ndim_to_linear( ConstVectorStorageType ndim_vector ) const {
					size_t linear = 0;
					for( size_t i = 0; i < this->dimensions(); i++ ) {
						linear += this->_offsets[ i ] * ndim_vector[ i ];
					}
					return linear;
				}

				// must be same dimensionality
				/**
				 * Retargets the current object to describe a system with the same number of dimensions
				 * and sizes \p _new_sizes. If the number of dimensions of \p _new_sizes does not match
				 * #dimensions(), an exception is thrown.
				 */
				void retarget( ConstVectorReference _new_sizes ) {
					if( _new_sizes.dimensions() != this->_sizes.dimensions() ) {
						throw std::invalid_argument( "new system must have same dimensions as previous: new "
						+ std::to_string( _new_sizes.dimensions() ) + ", old "
						+ std::to_string( this->_sizes.dimensions() ) );
					}
					this->_sizes = _new_sizes; // copy
					this->_system_size = compute_range_product( _new_sizes.begin(), _new_sizes.end(),
						this->_offsets.begin() );
				}

				/**
				 * Returns a beginning iterator to the #dimensions()-dimensional system \c this describes.
				 * The provided iterator references a system point, described both via its #dimensions()-dimensional
				 * coordinates and via a linear value from \a 0 to #system_size() (excluded).
				 */
				Iterator begin() const {
					return Iterator( *this );
				}

				/**
				 * Return an iterator to the end of the system; this iterator should not be
				 * referenced nor incremented.
				 */
				Iterator end() const {
					return Iterator::make_system_end_iterator( *this );
				}

			private:
				VectorType _offsets;
				size_t _system_size;

				/**
				 * Incrementally computes the product of the input iterator's range, storing each value
				 * into the position pointed to the output iterator; the accumulation starts from 1
				 * (also the first output values), and the last accumulated value is returned directly
				 * (and not stored). This assumes that the output container can store at least as many values
				 * as in the input range.
				 */
				template<
					typename IterIn,
					typename IterOut
				> static size_t compute_range_product(
					IterIn in_begin,
					IterIn in_end,
					IterOut out_begin
				) {
					size_t prod = 1;
					for( ; in_begin != in_end; ++in_begin, ++out_begin ) {
						*out_begin = prod;
						prod *= *in_begin;
					}
					return prod;
				}
			};

		} // namespace multigrid
	}     // namespace utils
} // namespace grb

#endif // _H_GRB_ALGORITHMS_MULTIGRID_NDIM_SYSTEM_LINEARIZER
