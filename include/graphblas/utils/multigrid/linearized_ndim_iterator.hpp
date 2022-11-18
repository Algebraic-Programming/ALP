
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
 * @file linearized_ndim_iterator.hpp
 * @author Alberto Scolari (alberto.scolari@huawei.com)
 * Definition of LinearizedNDimIterator.
 */

#ifndef _H_GRB_ALGORITHMS_MULTIGRID_NDIM_ITERATOR
#define _H_GRB_ALGORITHMS_MULTIGRID_NDIM_ITERATOR

#include <cstddef>
#include <algorithm>
#include <stdexcept>
#include <type_traits>
#include <limits>
#include <cstddef>

#include <graphblas/utils/iterators/utils.hpp>

#include "array_vector_storage.hpp"

namespace grb {
	namespace utils {
		namespace multigrid {

			// forward declaration for default
			template<
				typename SizeType,
				typename InternalVectorType
			> class LinearizedNDimSystem;

			/**
			 * Iterator object couled to a LinearizedNDimSystem: each object points to a vector
			 * in the creating LinearizedNDimSystem#dimensions()-dimensions space, to which also a
			 * linear position is associated; both the vector and the linear position can be retrieved
			 * via the \a -> method.
			 *
			 * It meets the requirements of a random access iterator.
			 *
			 * @tparam SizeType integral type to store the size of each dimension
			 * @tparam InternalStorageType internal vector type to store the sizes
			 */
			template<
				typename SizeType,
				typename InternalVectorType
			> class LinearizedNDimIterator {
			public:
				using VectorType = InternalVectorType;
				using LinNDimSysType = LinearizedNDimSystem< SizeType, VectorType >;
				using ConstVectorReference = const VectorType&;
				using SelfType = LinearizedNDimIterator< SizeType, InternalVectorType >;

				/**
				 * Structure describing a couple vector/linear coordinate: the vector
				 * can be obtained via #get_position() while the linear coordinate via
				 * #get_linear_position().
				 */
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

				/**
				 * Construct a new LinearizedNDimIterator object from the original LinNDimSysType
				 * object, storing the information about system dimensionality and sizes. The referenced
				 * vector is the first one in the system, i.e. with all coordinates being \a 0.
				 *
				 * If \p _system is not a valid object anymore, all iterators created from it are also
				 * not valid.
				 */
				LinearizedNDimIterator( const LinNDimSysType &_system ) noexcept :
					_p( _system )
				{}

				/**
				 * Construct a new LinearizedNDimIterator object from the original LinNDimSysType
				 * object, storing the information about system dimensionality and sizes. The referenced
				 * vector is initialized with the coordinates referenced via the iterator \p begin,
				 * which should have at least \p _system.dimensions() valid successors.
				 *
				 * If \p _system is not a valid object anymore, all iterators created from it are also
				 * not valid.
				 */
				template< typename IterT > LinearizedNDimIterator(
					const LinNDimSysType &_system, IterT begin
				) noexcept :
					_p( _system )
				{
					std::copy_n( begin, _system.dimensions(), this->_p.coords.begin() );
				}

				LinearizedNDimIterator() = delete;

				LinearizedNDimIterator( const SelfType &original ):
					_p( original._p ) {}

				SelfType& operator=( const SelfType &original ) = default;

				~LinearizedNDimIterator() {}

				/**
				 * Moves to the next vector in the multi-dimensional space, corresponding to
				 * advancing the linear coordinate by 1.
				 */
				SelfType & operator++() noexcept {
					bool rewind = true;
					// rewind only the first N-1 coordinates
					for( size_t i = 0; i < this->_p.system->dimensions() - 1 && rewind; i++ ) {
						SizeType& coord = this->_p.coords[ i ];
						// must rewind dimension if we wrap-around
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

				/**
				 * Moves \p _offset vectors ahead in the multi-dimensional space, corresponding to
				 * advancing the linear coordinate by \p _offset.
				 *
				 * If the destination vector is outside of the system (i.e. the corresponding
				 * linear coordinate is beyond the underlying LinearizedNDimSystem#system_size()),
				 * an exception is thrown.
				 */
				SelfType & operator+=( size_t offset ) {
					size_t linear = _p.get_linear_position() + offset;
					if( linear > _p.system->system_size() ) {
						throw std::invalid_argument("increment is too large");
					}
					if( offset == 1 ) {
						return operator++();
					}
					_p.system->linear_to_ndim( linear, _p.coords );
					return *this;
				}

				/**
				 * Returns the difference between \p _other and \c this in the linear space.
				 *
				 * It throws if the result cannot be stored as a difference_type variable.
				 */
				difference_type operator-( const SelfType &other ) const {
					return grb::utils::compute_signed_distance< difference_type, SizeType >(
						_p.get_linear_position(), other._p.get_linear_position() );

				}

				reference operator*() const {
					return this->_p;
				}

				pointer operator->() const {
					return &( this->_p );
				}

				bool operator!=( const SelfType &o ) const {
					const size_t dims = this->_p.system->dimensions();
					if( dims != o._p.system->dimensions() ) {
						throw std::invalid_argument("system sizes do not match");
					}
					bool equal = true;
					for( size_t i =0; i < dims && equal; i++) {
						equal &= ( this->_p.coords[i] == o._p.coords[i] );
					}
					return !equal;
				}

				/**
				 * Facility to build an end iterator.
				 *
				 * Its implementation depending on the logic in operator++.
				 */
				static SelfType make_system_end_iterator( const LinNDimSysType &_system ) {
					// fill with 0s
					SelfType iter( _system );
					size_t last = iter->system->dimensions() - 1;
					// store last size in last position
					iter._p.coords[ last ] = iter->system->get_sizes()[ last ];
					return iter;
				}

			private:
				NDimPoint _p;
			};

		} // namespace multigrid
	} // namespace utils
} // namespace grb

#endif // _H_GRB_ALGORITHMS_MULTIGRID_NDIM_ITERATOR
