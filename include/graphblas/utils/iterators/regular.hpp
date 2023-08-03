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

/**
 * A set of iterators that mimic containers with regular structure.
 *
 * @author A. N. Yzelman 
 * @date 3/8/2023
 */

#ifndef _H_GRB_ITERATOR_REGULAR
#define _H_GRB_ITERATOR_REGULAR

#include <iterator>

#include <assert.h>


namespace grb {

	namespace utils {

		namespace internal {

			/**
			 * A random access const-iterator whose returned value only depends on a
			 * position within some range \f$ (0,c] \f$.
			 *
			 * Here, \f$ c \f$ is the so-called \a count.
			 *
			 * @tparam T Some position-independent state of the iterator.
			 *
			 * @tparam SelfType The type of the final iterator class that inherits from
			 *                  this base iterator type.
			 *
			 * \warning This iterator performs the bare minimum in error checking.
			 *          Invalid use of the resulting iterators will result in undefined
			 *          behaviour.
			 *
			 * \note Enable assertions to catch errors.
			 */
			template< typename T, typename SelfType >
			class PosBasedIterator {

				protected:

					size_t _count;

					size_t _pos;

					T _state;

					/**
					 * Internal constructor that directly sets all state.
					 */
					PosBasedIterator( const T state, const size_t pos, const size_t count ) :
						_count( count ), _pos( pos ), _state( state )
					{}


				public:

					// standard STL typedefs

					typedef std::random_access_iterator_tag iterator_category;

					typedef size_t difference_type;

					// deriving iterators must define
					//  - value_type
					//  - pointer_type
					//  - reference_type
					
					// STL-like typedefs

					typedef SelfType self_type;

					typedef self_type & self_reference_type;

					// constructor

					/**
					 * @param[in] state The position-independent state of the iterator.
					 * @param[in] count How many times the constructed iterator can be
					 *                  incremented without moving past its end position.
					 *
					 * \warning The maximum value for \a count is <tt>SIZE_MAX</tt>. After
					 *          incrementing the iterator returned by this constructor that
					 *          many times, the iterator shall be in end position.
					 */
					PosBasedIterator( const T state, const size_t count ) :
						_count( count ), _pos( 0 ), _state( state )
					{}

					// destructor does nothing special
					~PosBasedIterator() {}

					// standard iterator interface

					self_reference_type operator=( const self_reference_type other ) {
						_count = other._count;
						_pos = other._pos;
						_state = other._state;
						return *this;
					}

					self_reference_type operator++() noexcept {
						assert( _pos < _count );
						(void) ++_pos;
						return *this;
					}

					friend void swap( self_reference_type left, self_reference_type right ) {
						std::swap( left._count, right._count );
						std::swap( left._pos, right._pos );
						std::swap( left._state, right._state );
					}

					// input iterator interface

					self_reference_type operator++(int) noexcept {
						assert( _pos < _count );
						self_type ret( _state, _pos, _count );
						(void) _pos++;
						return ret;
					}

					friend bool operator==(
						self_reference_type left, self_reference_type right
					) noexcept {
						assert( left._count == right._count );
						assert( left._state == right._state );
						return left._pos == right._pos;
					}

					friend bool operator!=(
						self_reference_type left, self_reference_type right
					) noexcept {
						assert( left._count == right._count );
						assert( left._state == right._state );
						return left._pos != right._pos;
					}

					// bi-directional iterator interface

					self_reference_type operator--() noexcept {
						assert( _pos > 0 );
						(void) --_pos;
						return *this;
					}

					self_reference_type operator--(int) noexcept {
						assert( _pos > 0 );
						self_type ret( _state, _pos, _count );
						(void) _pos--;
						return ret;
					}

					// random access iterator interface

					friend bool operator<(
						const self_reference_type left,
						const self_reference_type right
					) {
						assert( left._count == right._count );
						assert( left._state == right._state );
						return left._pos < right._pos;
					}

					friend bool operator>(
						const self_reference_type left,
						const self_reference_type right
					) {
						assert( left._count == right._count );
						assert( left._state == right._state );
						return left._pos > right._pos;
					}

					friend bool operator<=(
						const self_reference_type left,
						const self_reference_type right
					) {
						assert( left._count == right._count );
						assert( left._state == right._state );
						return left._pos <= right._pos;
					}

					friend bool operator>=(
						const self_reference_type left,
						const self_reference_type right
					) {
						assert( left._count == right._count );
						assert( left._state == right._state );
						return left._pos >= right._pos;
					}

					self_reference_type operator+=( const size_t count ) noexcept {
						assert( _pos + count < _count );
						_pos += count;
						return *this;
					}

					friend self_type operator+(
						 const self_reference_type iterator,
						 const size_t count
					) noexcept {
						assert( iterator._pos + count < iterator._count );
						return self_type(
								iterator._state, iterator._pos + count, iterator._count
							);
					}

					friend self_type operator+(
						const size_t count,
						const self_reference_type iterator
					) noexcept {
						assert( iterator._pos + count < iterator._count );
						return self_type(
								iterator._state, iterator._pos + count, iterator._count
							);
					}

					self_reference_type operator-=( const size_t count ) noexcept {
						assert( _pos >= count );
						_pos -= count;
						return *this;
					}

					friend self_type operator-(
						const self_reference_type iterator,
						const size_t count
					) noexcept {
						assert( iterator._pos >= count );
						return self_type(
								iterator._state, iterator._pos - count, iterator._count
							);
					}

					friend self_type operator-(
						const size_t count,
						const self_reference_type iterator
					) noexcept {
						assert( iterator._pos >= count );
						return self_type(
								iterator._state, iterator._pos - count, iterator._count
							);
					}

					// derived iterators should provide the following
					//  - standard iterator interface
					//    - reference_type operator*() const
					//  - input iterator interface
					//    - value_type operator*() const
					//    - pointer_type operator->() const
					//  - random access iterator interface
					//    - reference_type operator[]() const

			};

		}

		namespace iterators {

			/**
			 * An iterator that repeats the same value for a set number of times.
			 */
			template< typename T >
			class Repeater : public PosBasedIterator< T, Repeater< T > > {

				public:

					// standard STL typedefs
					typedef const T value_type;

					typedef const T * pointer_type;

					typedef const T & reference_type;

					// constructor
					Repeater( const T _val, const size_t count ) :
						PosBasedIterator( _val, count )
					{}

					// destructor
					~Repeater() {}

					// standard iterator interface

					reference_type operator*() const noexcept {
						return PosBasedIterator._state;
					}

					// input iterator interface

					value_type operator*() const noexcept { return PosBasedIterator::_state; }

					pointer_type operator->() const noexcept {
						return &(PosBasedIterator::_state);
					}

					// random access iterator interface
					reference_type operator[]( const size_t i ) const noexcept {
						assert( i < PosBasedIterator::_count );
						return PosBasedIterator::_state;
					}
	
			};

		}

	}

}

#endif // end _H_GRB_ITERATOR_REGULAR

