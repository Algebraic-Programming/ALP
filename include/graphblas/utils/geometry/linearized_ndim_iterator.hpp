
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

#ifndef _H_GRB_ALGORITHMS_GEOMETRY_NDIM_ITERATOR
#define _H_GRB_ALGORITHMS_GEOMETRY_NDIM_ITERATOR

#include <cstddef>
#include <algorithm>
#include <stdexcept>
#include <type_traits>
#include <limits>
#include <cstddef>

#include "array_vector_storage.hpp"

namespace grb {
	namespace utils {
		namespace geometry {

			// forward declaration for default
			template<
				typename SizeType,
				typename InternalVectorType
			> class LinearizedNDimSystem;

			template<
				typename SizeType,
				typename InternalVectorType
			> class LinearizedNDimIterator {
			public:

				using VectorType = InternalVectorType;
				using LinNDimSysType = LinearizedNDimSystem< SizeType, VectorType >;
				using ConstVectorReference = const VectorType&;
				using SelfType = LinearizedNDimIterator< SizeType, InternalVectorType >;

				struct NDimPoint {
				private:

					const LinNDimSysType* system; // pointer because of copy assignment
					VectorType coords;

				public:

					friend SelfType;

					NDimPoint() = delete;

					NDimPoint( const NDimPoint& ) = default;

					NDimPoint( NDimPoint&& ) = delete;

					NDimPoint( const LinNDimSysType& _system ) noexcept :
						system( &_system ),
						coords( _system.dimensions() )
					{
						std::fill_n( this->coords.begin(), _system.dimensions(), 0 );
					}

					NDimPoint& operator=( const NDimPoint& ) = default;

					inline ConstVectorReference get_position() const {
						return coords;
					}

					size_t get_linear_position() const {
						return system->ndim_to_linear( coords );
					}
				};


				// interface for std::random_access_iterator
				using iterator_category = std::random_access_iterator_tag;
				using value_type = NDimPoint;
				using pointer = const value_type*;
				using reference = const value_type&;
				using difference_type = signed long;

				LinearizedNDimIterator( const LinNDimSysType &_system ) noexcept :
					_p( _system )
				{}

				template< typename IterT > LinearizedNDimIterator( const LinNDimSysType &_system, IterT begin ) noexcept :
					_p( _system )
				{
					std::copy_n( begin, _system.dimensions(), this->_p.coords.begin() );
				}

				LinearizedNDimIterator() = delete;

				LinearizedNDimIterator( const SelfType &original ):
					_p( original._p ) {}

				SelfType& operator=( const SelfType &original ) = default;

				// LinearizedNDimIterator( SelfType && ) = delete;

				// SelfType operator=( SelfType && ) = delete;

				~LinearizedNDimIterator() {}

				SelfType & operator++() noexcept {
					bool rewind{ true };
					// rewind only the first N-1 coordinates
					for( size_t i { 0 }; i < this->_p.system->dimensions() - 1 && rewind; i++ ) {
						SizeType& coord = this->_p.coords[ i ];
						// must rewind dimension if we wrap-around
						/*
						SizeType new_coord = ( coord + 1 ) % this->_p.system->get_sizes()[ i ];
						rewind = new_coord < coord;
						coord = new_coord;
						*/
						SizeType plus = coord + 1;
						rewind = plus >= this->_p.system->get_sizes()[ i ];
						coord = rewind ? 0 : plus;
					}
					// if we still have to rewind, increment the last coordinate, which is unbounded
					if( rewind ) {
						this->_p.coords[ this->_p.system->dimensions() - 1 ]++;
					}
					return *this;
				}

				SelfType & operator+=( size_t offset ) {
					size_t linear{ _p.get_linear_position() + offset };
					if( linear > _p.system->system_size() ) {
						throw std::invalid_argument("increment is too large");
					}
					_p.system->linear_to_ndim( linear, _p.coords );
					return *this;
				}

				difference_type operator-( const SelfType &other ) const {
					size_t a_pos{ _p.get_linear_position() },
						b_pos{ other._p.get_linear_position() };
					size_t lowest{ std::min( a_pos, b_pos ) }, highest{ std::max( a_pos, b_pos )};
					if( highest - lowest > static_cast< size_t >(
						std::numeric_limits< difference_type >::max() ) ) {
						throw std::invalid_argument( "iterators are too distant" );
					}
					return ( static_cast< difference_type >( a_pos - b_pos ) );
				}

				reference operator*() const {
					return this->_p;
				}

				pointer operator->() const {
					return &( this->_p );
				}

				bool operator!=( const SelfType &o ) const {
					const size_t dims{ this->_p.system->dimensions() };
					if( dims != o._p.system->dimensions() ) {
						throw std::invalid_argument("system sizes do not match");
					}
					bool equal{ true };
					for( size_t i{0}; i < dims && equal; i++) {
						equal &= ( this->_p.coords[i] == o._p.coords[i] );
					}
					return !equal;
				}

				// implementation depending on logic in operator++
				static SelfType make_system_end_iterator( const LinNDimSysType &_system ) {
					// fill with 0s
					SelfType iter( _system );
					size_t last{ iter->system->dimensions() - 1 };
					// store last size in last position
					iter._p.coords[ last ] = iter->system->get_sizes()[ last ];
					return iter;
				}

			private:
				NDimPoint _p;

			};

		} // namespace geometry
	} // namespace utils
} // namespace grb

#endif // _H_GRB_ALGORITHMS_GEOMETRY_NDIM_ITERATOR
