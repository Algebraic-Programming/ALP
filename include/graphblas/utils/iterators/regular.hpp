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
 * Also contains dummy containers that wrap these iterators.
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
			 * @tparam R The return type of the iterator.
			 *
			 * @tparam T    Some position-independent state of the iterator.
			 * @tparam func Transforms the current position and state into a return
			 *              value.
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
			template<
				typename R,
				typename T, std::function< void(R&,const T&,size_t) > func,
				typename SelfType
			>
			class PosBasedIterator {

				protected:

					size_t _count;

					size_t _pos;

					R _val;

					T _state;

					/**
					 * Internal constructor that directly sets all state.
					 */
					PosBasedIterator(
						const size_t count, const size_t pos,
						const R val, const T state
					) :
						_count( count ), _pos( pos ), _val( val ), _state( state )
					{}


				public:

					// standard STL typedefs

					typedef std::random_access_iterator_tag iterator_category;

					typedef size_t difference_type;

					typedef const R value_type;

					typedef const R * pointer_type;

					typedef const R & reference_type;
					
					// STL-like typedefs

					typedef SelfType self_type;

					typedef self_type & self_reference_type;

					// constructor

					/**
					 * @param[in] count How many times the constructed iterator can be
					 *                  incremented without moving past its end position.
					 * @param[in] start Whether the iterator is in start position (or end
					 *                  position).
					 * @param[in] state The position-independent state of the iterator.
					 * @param[in] dummy An optional dummy value of type \a R in case instances
					 *                  of \a R are not default-constructible.
					 *
					 * \warning The maximum value for \a count is <tt>SIZE_MAX</tt>. After
					 *          incrementing the iterator returned by this constructor that
					 *          many times, the iterator shall be in end position.
					 */
					PosBasedIterator(
						const size_t count,
						const bool start,
						const T state,
						const R dummy = R()
					) :
						_count( count ), _val( dummy ), _state( state )
					{
						if( start ) {
							_pos = 0;
							if( count > 0 ) {
								func( _val, state, 0 );
							}
						} else {
							_pos = count;
						}
					}

					// destructor does nothing special

					~PosBasedIterator() {}

					// standard iterator interface

					reference_type operator*() const noexcept {
						return _val;
					}

					self_reference_type operator=( const self_reference_type other ) {
						_count = other._count;
						_pos = other._pos;
						_val = other._val;
						_state = other._state;
						return *this;
					}

					self_reference_type operator++() noexcept {
						assert( _pos < _count );
						(void) ++_pos;
						func( _val, _state, _pos );
						return *this;
					}

					friend void swap( self_reference_type left, self_reference_type right ) {
						std::swap( left._count, right._count );
						std::swap( left._pos, right._pos );
						std::swap( left._val, right._val );
						std::swap( left._state, right._state );
					}

					// input iterator interface

					value_type operator*() const noexcept {
						return _val;
					}

					pointer_type operator->() const noexcept {
						return &_val;
					}

					self_reference_type operator++(int) noexcept {
						assert( _pos < _count );
						self_type ret( _pos, _count, _val, _state );
						(void) _pos++;
						func( _val, _state, _pos );
						return ret;
					}

					friend bool operator==(
						self_reference_type left, self_reference_type right
					) noexcept {
						assert( left._count == right._count );
						assert( left._val == right._val );
						assert( left._state == right._state );
						return left._pos == right._pos;
					}

					friend bool operator!=(
						self_reference_type left, self_reference_type right
					) noexcept {
						assert( left._count == right._count );
						assert( left._val == right._val );
						assert( left._state == right._state );
						return left._pos != right._pos;
					}

					// bi-directional iterator interface

					self_reference_type operator--() noexcept {
						assert( _pos > 0 );
						(void) --_pos;
						func( _val, _state, _pos );
						return *this;
					}

					self_reference_type operator--(int) noexcept {
						assert( _pos > 0 );
						self_type ret( _pos, _count, _val, _pos );
						(void) _pos--;
						func( _val, _state, _pos );
						return ret;
					}

					// random access iterator interface

					/**
					 * \internal This bracket-operator cannot return a reference, as there is
					 *           no storage associated to all items iterated over. (This is
					 *           also the reason why this is a const-iterator only.)
					 */
					value_type operator[]( const size_t i ) const noexcept {
						assert( i < PosBasedIterator::_count );
						R ret = _val;
						func( ret, _state, i );
						return ret;
					}
	
					friend bool operator<(
						const self_reference_type left,
						const self_reference_type right
					) {
						assert( left._count == right._count );
						assert( left._val == right._val );
						assert( left._state == right._state );
						return left._pos < right._pos;
					}

					friend bool operator>(
						const self_reference_type left,
						const self_reference_type right
					) {
						assert( left._count == right._count );
						assert( left._val == right._val );
						assert( left._state == right._state );
						return left._pos > right._pos;
					}

					friend bool operator<=(
						const self_reference_type left,
						const self_reference_type right
					) {
						assert( left._count == right._count );
						assert( left._val == right._val );
						assert( left._state == right._state );
						return left._pos <= right._pos;
					}

					friend bool operator>=(
						const self_reference_type left,
						const self_reference_type right
					) {
						assert( left._count == right._count );
						assert( left._val == right._val );
						assert( left._state == right._state );
						return left._pos >= right._pos;
					}

					self_reference_type operator+=( const size_t count ) noexcept {
						assert( _pos + count < _count );
						_pos += count;
						func( _val, _state, _pos );
						return *this;
					}

					friend self_type operator+(
						 const self_reference_type iterator,
						 const size_t count
					) noexcept {
						assert( iterator._pos + count < iterator._count );
						const size_t pos = iterator._pos + count;
						R val = iterator._val;
						func( val, iterator._state, pos );
						return self_type(
							pos, iterator._count,
							val, iterator._state
						);
					}

					friend self_type operator+(
						const size_t count,
						const self_reference_type iterator
					) noexcept {
						assert( iterator._pos + count < iterator._count );
						const size_t pos = iterator._pos + count;
						R val = iterator._val;
						func( val, iterator._state, pos );
						return self_type(
							pos, iterator._count,
							val, iterator._state
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
						const size_t pos = iterator._pos - count;
						R val = iterator._val;
						func( val, iterator._state, pos );
						return self_type(
							pos, iterator._count,
							val, iterator._state
						);
					}

					friend self_type operator-(
						const size_t count,
						const self_reference_type iterator
					) noexcept {
						assert( iterator._pos >= count );
						const size_t pos = iterator._pos - count;
						R val = iterator._val;
						func( val, iterator._state, pos );
						return self_type(
							pos, iterator._count,
							val, iterator._state
						);
					}

			};

		} // end namespace grb::utils::internal

		namespace iterators {

			/**
			 * An iterator that repeats the same value for a set number of times.
			 */
			template< typename T >
			class Repeater : public PosBasedIterator<
				T, T,
				[](T &, const T &, const size_t) { return; },
				Repeater< T >
			> {

				protected:

					// direct constructor

					Repeater(
						const size_t count,
						const size_t pos,
						const T val
					) :
						PosBasedIterator( count, pos, val, val )
					{}


				public:

					// constructor

					/**
					 * Constructs an iterator over a collection that contains the same constant
					 * value \a val \a count times.
					 *
					 * @param[in] count How many elements are in the collection.
					 * @param[in] start Whether the constructed iterator is in start or end
					 *                  position.
					 * @param[in] val   The constant value that should be returned \a count
					 *                  times.
					 */
					Repeater(
						const size_t count,
						const bool start,
						const T val
					) :
						PosBasedIterator( count, start, val, val )
					{}

					// destructor

					~Repeater() {}

			};

			/**
			 * An iterator over a collection of \f$ c \f$ items that for each item
			 * \f$ i \in \{0,1,\dots,c-1\} \f$ returns \f$ f(i) \f$, for given \f$ f \f$.
			 *
			 * @tparam T The value type of the iterator, i.e., the type returned by
			 *           \f$ f \f$.
			 *
			 * Rather than using Sequence iterators directly, users may consider
			 * referring to #grb::utils::containers::Range instead.
			 */
			template< typename T >
			class Sequence : public PosBasedIterator<
				T, std::pair< size_t, size_t >,
				[]( T &val, const std::pair< size_t, size_t > &state, const size_t pos ) {
					val = state.first + pos * state.second;
				},
				Sequence< T >
			> {

				protected:

					// direct constructor

					Sequence(
						const size_t count,
						const size_t pos,
						const T val,
						const std::pair< size_t, size_t > state
					) :
						PosBasedIterator( count, pos, val, state )
					{}


				public:

					// constructor

					/**
					 * Constructs an iterator over a given sequence.
					 *
					 * @param[in] count  The number of elements in the sequence.
					 * @param[in] start  Whether the iterator is in start position (or in end
					 *                   position instead).
					 * @param[in] offset The first element in the sequence.
					 * @param[in] stride The distance between two elements in the sequence.
					 * @param[in] dummy  A dummy initialiser for return elements; optional, in
					 *                   case \a T is not default-constructible.
					 */
					Sequence(
						const size_t count,
						const bool start,
						const size_t offset = 0,
						const size_t stride = 1,
						T dummy = T()
					) :
						PosBasedIterator(
							count,
							start,
							std::make_pair< size_t, size_t >( offset, stride ),
							dummy
						)
					{}

					// destructor

					~Sequence() {}

			};

		} // end namespace grb::utils::iterators

		namespace containers {

			/**
			 * A (dense) vector of a given size that holds the same constant value at
			 * each entry.
			 *
			 * Instances of this container are immutable in terms of both value and size.
			 *
			 * @tparam T The type of the value.
			 *
			 * The storage requirement of this container is \f$ \Theta(1) \f$.
			 */
			template< typename T >
			class ConstantVector {

				private:

					const T val;

					const size_t n;


				public:

					typedef grb::utils::iterators::Repeater< T > iterator;

					typedef iterator const_iterator;

					/**
					 * Constructs a container with \f$ \Theta(1) \f$ memory usage
					 * that represents some vector of length \a n with contents
					 * \f$ ( c, c, \ldots, c ) \f$.
					 *
					 * @param[in] val The value of the constants \f$ c \f$ in this vector.
					 * @param[in] n   The size of the vector.
					 */
					ConstantVector( const T val, const size_t n ) : _val( val ), _n( n ) {}

					iterator begin() const noexcept {
						return iterator( val, n, true );
					}

					iterator end() const noexcept {
						return iterator( val, n, false );
					}

					const_iterator begin() const noexcept {
						return const_iterator( val, n, true );
					}

					const_iterator end() const noexcept {
						return const_iterator( val, n, false );
					}

			};

			/**
			 * A container that contains a sequence of numbers with a given stride.
			 *
			 * @tparam T The type of numbers; optional, default is <tt>size_t</tt>.
			 *
			 * The storage of this container is \f$ \Theta(1) \f$.
			 *
			 * This is an unmodifiable (const) container.
			 */
			template< typename T = size_t >
			class Range {

				private:

					const size_t _start;

					const size_t _stride;

					const size_t _count;


				public:

					typedef grb::utils::iterators::Sequence< T > iterator;

					typedef iterator const_iterator;

					/**
					 * Constructs a range.
					 *
					 * @param[in] start  The start of the range (inclusive)
					 * @param[in] stride The stride of the range
					 * @param[in] end    The end of the range (exclusive)
					 *
					 * The value \a end must be larger than or equal to \a start. Equal values
					 * for \a start and \a end result in an empty range. A larger value for
					 * \a end than \a start will result in a range consisting at least one
					 * element (\a start).
					 *
					 * For example, the range \f$ (1, 2, 3, 4, 5, 6, 7, 8, 9, 10) \f$ may be
					 * constructed by \a start 1, \a stride 1, and \a end 11.
					 * 
					 * The range \f$ (0, 2, 4, 6, 8, 10) \f$ may be constructed by \a start 0,
					 * \a stride 2, and \a end 12. Alternatively, \a end may be 11 since both
					 * 11 and 12 are not part of the intended range.
					 */
					Range(
						const size_t start,
						const size_t stride,
						const size_t end
					) noexcept :
						_start( start ), _stride( stride ), _count( 0 )
					{
						assert( start >= end );

						// catch trivial case for count
						if( start == end ) { return; }

						// compute count
						_count = (end-start) % stride > 0
							? ((end-start) / stride + 1)
							: (end - start) / stride;
					}

					iterator begin() const noexcept {
						return iterator( count, true, offset, stride );
					}

					iterator end() const noexcept {
						return iterator( count, false, offset, stride );
					}

					const_iterator begin() const noexcept {
						return const_iterator( count, true, offset, stride );
					}

					const_iterator end() const noexcept {
						return const_iterator( count, false, offset, stride );
					}

			};

		} // end namespace grb::utils::containers

	} // end namespace grb::utils

}

#endif // end _H_GRB_ITERATOR_REGULAR

