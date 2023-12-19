
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

#include <utility>
#include <iterator>
#include <functional>

#include <assert.h>


namespace grb {

	namespace utils {

		namespace iterators {

			template< typename T >
			class Repeater;

			template< typename T >
			class Sequence;

		}

		namespace internal {

			/**
			 * A random access const-iterator whose returned value only depends on a
			 * position within some range \f$ (0,c] \f$.
			 *
			 * Here, \f$ c \f$ is the so-called \a count.
			 *
			 * @tparam R        The return type of the iterator.
			 * @tparam T        Some position-independent state of the iterator.
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
				typename T,
				typename SelfType
			>
			class PosBasedIterator {

				friend class grb::utils::iterators::Repeater< R >;

				friend class grb::utils::iterators::Sequence< R >;

				protected:

					/**
					 * Configurable block size for parallel I/O.
					 *
					 * The default here is a small multiple of standard cache line sizes.
					 */
					static constexpr const size_t block_size = 256;

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

					typedef R value_type;

					typedef const R * pointer;

					typedef const R & reference;

					// STL-like typedefs

					typedef PosBasedIterator< R, T, SelfType > self_type;

					typedef self_type & self_reference_type;

					typedef const PosBasedIterator< R, T, SelfType > self_const_reference_type;

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
					 * To create iterators that mimic parallel I/O, the following optional
					 * arguments exist:
					 *
					 * @param[in] s     The process ID. Optional; default is zero.
					 * @param[in] P     The number of processes. Optional; default is one.
					 *
					 * The parameter \a s must be strictly smaller than \a P. The parameter
					 * \a P may not be zero.
					 *
					 * If \a s is 0 and \a P is 1, default sequential semantics are selected;
					 * meaning, the iterator constructed will iterate over the entire range.
					 * If \a s is nonzero and \a P is larger than one, the created iterator
					 * iterates but over part of the range.
					 *
					 * \warning The maximum value for \a count is <tt>SIZE_MAX</tt>. After
					 *          incrementing the iterator returned by this constructor that
					 *          many times, the iterator shall be in end position.
					 */
					PosBasedIterator(
						const size_t count,
						const bool start,
						const T state,
						const R dummy = R(),
						const size_t s = 0,
						const size_t P = 1
					) :
						_count( count ), _val( dummy ), _state( state )
					{
						// run-time checks
						if( P == 0 || s >= P ) {
							throw std::runtime_error( "Illegal values for s and/or P" );
						}
						// adjust count according to P
						if( P > 1 && count > block_size ) {
							const size_t bcount = (count % block_size) > 0
								? count / block_size + 1
								: count / block_size;
							const size_t bcount_per_process = (bcount % P) > 0
								? bcount / P + 1
								: bcount / P;
							_count = bcount_per_process * block_size;
						}
						// adjust count according to s
						_count = (s+1) * _count;
						if( _count > count ) { _count = count; }
						// select start position according to s
						_pos = start
							? s * _count
							: count;
						// correct potential overflow of starting position
						if( _pos > count ) { _pos = count; }
						// initialise selected starting position
						if( count > 0 ) {
							SelfType::func( _val, state, _pos );
						}
					}

					PosBasedIterator( const self_const_reference_type &other ) :
						_count( other._count ), _pos( other._pos ),
						_val( other._val ), _state( other._state )
					{}

					PosBasedIterator( PosBasedIterator< R, T, SelfType > &&other ) :
						_count( other._count ), _pos( other._pos )
					{
						_val = std::move( other._val );
						_state = std::move( other._state );
						other._count = other._pos = 0;
					}

					// destructor does nothing special

					~PosBasedIterator() {}

					// standard iterator interface

					reference operator*() const noexcept {
						return _val;
					}

					self_reference_type operator=( self_const_reference_type other ) noexcept {
						_count = other._count;
						_pos = other._pos;
						_val = other._val;
						_state = other._state;
						return *this;
					}

					self_reference_type operator=(
						PosBasedIterator< R, T, SelfType > &&other
					) noexcept {
						_count = other._count;
						_pos = other._pos;
						_val = std::move( other._val );
						_state = std::move( other._state );
						other._count = other._pos = 0;
						return *this;
					}

					self_reference_type operator++() noexcept {
						assert( _pos < _count );
						(void) ++_pos;
						SelfType::func( _val, _state, _pos );
						return *this;
					}

					friend void swap( self_reference_type left, self_reference_type right ) {
						std::swap( left._count, right._count );
						std::swap( left._pos, right._pos );
						std::swap( left._val, right._val );
						std::swap( left._state, right._state );
					}

					// input iterator interface

					pointer operator->() const noexcept {
						return &_val;
					}

					self_type operator++(int) noexcept {
						assert( _pos < _count );
						self_type ret =
							SelfType::create_iterator( _count, _pos, _val, _state );
						(void) _pos++;
						SelfType::func( _val, _state, _pos );
						return ret;
					}

					friend bool operator==(
						self_const_reference_type left, self_const_reference_type right
					) noexcept {
						return left._pos == right._pos &&
							left._count == right._count &&
							left._state == right._state &&
							left._val == right._val;
					}

					friend bool operator!=(
						self_const_reference_type left, self_const_reference_type right
					) noexcept {
						return !(left == right);
					}

					// bi-directional iterator interface

					self_reference_type operator--() noexcept {
						assert( _pos > 0 );
						(void) --_pos;
						SelfType::func( _val, _state, _pos );
						return *this;
					}

					self_type operator--(int) noexcept {
						assert( _pos > 0 );
						self_type ret =
							SelfType::make_iterator( _pos, _count, _val, _pos );
						(void) _pos--;
						SelfType::func( _val, _state, _pos );
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
						SelfType::func( ret, _state, i );
						return ret;
					}

					friend bool operator<(
						self_const_reference_type left,
						self_const_reference_type right
					) {
						return left._count == right._count &&
							left._state == right._state &&
							left._pos < right._pos;
					}

					friend bool operator>(
						self_const_reference_type left,
						self_const_reference_type right
					) {
						return left._count == right._count &&
							left._state == right._state &&
							left._pos > right._pos;
					}

					friend bool operator<=(
						self_const_reference_type left,
						self_const_reference_type right
					) {
						return left._count == right._count &&
							left._state == right._state &&
							left._pos <= right._pos;
					}

					friend bool operator>=(
						self_const_reference_type left,
						self_const_reference_type right
					) {
						return left._count == right._count &&
							left._state == right._state &&
							left._pos >= right._pos;
					}

					self_reference_type operator+=( const size_t count ) noexcept {
						assert( _pos + count <= _count );
						_pos += count;
						SelfType::func( _val, _state, _pos );
						return *this;
					}

					friend self_type operator+(
						 self_const_reference_type iterator,
						 const size_t count
					) noexcept {
						assert( iterator._pos + count <= iterator._count );
						const size_t pos = iterator._pos + count;
						R val = iterator._val;
						SelfType::func( val, iterator._state, pos );
						return self_type(
							iterator._count, pos,
							val, iterator._state
						);
					}

					friend self_type operator+(
						const size_t count,
						self_const_reference_type iterator
					) noexcept {
						assert( iterator._pos + count <= iterator._count );
						const size_t pos = iterator._pos + count;
						R val = iterator._val;
						SelfType::func( val, iterator._state, pos );
						return self_type(
							iterator._count, pos,
							val, iterator._state
						);
					}

					self_reference_type operator-=( const size_t count ) noexcept {
						assert( _pos >= count );
						_pos -= count;
						return *this;
					}

					friend self_type operator-(
						self_const_reference_type iterator,
						const size_t count
					) noexcept {
						assert( iterator._pos >= count );
						const size_t pos = iterator._pos - count;
						R val = iterator._val;
						SelfType::func( val, iterator._state, pos );
						return self_type(
							iterator._count, pos,
							val, iterator._state
						);
					}

					friend self_type operator-(
						const size_t count,
						self_const_reference_type iterator
					) noexcept {
						assert( iterator._pos >= count );
						const size_t pos = iterator._pos - count;
						R val = iterator._val;
						SelfType::func( val, iterator._state, pos );
						return self_type(
							iterator._count, pos,
							val, iterator._state
						);
					}

					difference_type operator-(
						self_const_reference_type iterator
					) const noexcept {
						assert( iterator._count == _count );
						assert( iterator._state == _state );
						return static_cast< difference_type >( _pos - iterator._pos );
					}

			};

		} // end namespace grb::utils::internal

		namespace iterators {

			/**
			 * An iterator that repeats the same value for a set number of times.
			 *
			 * Rather than using this iterator directly, users may want to refer to the
			 * #grb::utils::containers::ConstantVector container instead.
			 */
			template< typename T >
			class Repeater {

				friend class internal::PosBasedIterator< T, T, Repeater< T > >;

				public:

					typedef grb::utils::internal::PosBasedIterator< T, T, Repeater< T > >
						RealType;


				protected:


					// direct constructor

					static RealType create_iterator(
						const size_t count,
						const size_t pos,
						const T val,
						const T state
					) {
						return RealType( count, pos, val, state );
					}


				public:

					inline static void func( T&, const T&, const size_t ) {}

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
					 *
					 * The following are optional arguments for optionally creating parallel
					 * I/O iterators:
					 *
					 * @param[in] s The process ID; default is zero.
					 * @param[in] P The number of processes; default is one.
					 *
					 * The parameter \a s must be strictly smaller than \a P, and \a P must be
					 * larger than zero.
					 */
					static RealType make_iterator(
						const size_t count,
						const bool start,
						const T val,
						const size_t s = 0,
						const size_t P = 1
					) {
						return RealType( count, start, val, val, s, P );
					}

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
			class Sequence {

				friend class internal::PosBasedIterator<
					T, std::tuple< size_t, size_t, size_t >,
					Sequence< T >
				>;

				public:

					typedef grb::utils::internal::PosBasedIterator<
						T, std::tuple< size_t, size_t, size_t >, Sequence< T >
					> RealType;


				protected:

					// direct constructor

					static RealType create_iterator(
						const size_t count,
						const size_t pos,
						const T val,
						const std::tuple< size_t, size_t, size_t > state
					) {
						return RealType( count, pos, val, state );
					}


				public:

					inline static void func(
						T &val,
						const std::tuple< size_t, size_t, size_t > &state,
						const size_t pos
					) {
						const size_t offset = std::get<0>(state);
						const size_t stride = std::get<1>(state);
						const size_t repetitions = std::get<2>(state);
						val = offset + ( pos / repetitions ) * stride;
					}

					// constructor

					/**
					 * Constructs an iterator over a given sequence.
					 *
					 * @param[in] count        The number of elements in the sequence.
					 * @param[in] start        Whether the iterator is in start position (or in end
					 *                         position instead).
					 * @param[in] offset       The first element in the sequence.
					 * @param[in] stride       The distance between two elements in the sequence.
					 * @param[in] repetitions The number of times each element is repeated.
					 * @param[in] dummy        A dummy initialiser for return elements; optional, in
					 *                         case \a T is not default-constructible.
					 *
					 * The following are optional arguments for optionally creating parallel
					 * I/O iterators:
					 *
					 * @param[in] s The process ID; default is zero.
					 * @param[in] P The number of processes; default is one.
					 *
					 * The parameter \a s must be strictly smaller than \a P, and \a P must be
					 * larger than zero.
					 */
					static RealType make_iterator(
						const size_t count,
						const bool start,
						const size_t offset = static_cast< size_t >( 0 ),
						const size_t stride = static_cast< size_t >( 1 ),
						const size_t repetitions = static_cast< size_t >( 1 ),
						T dummy = T(),
						const size_t s = 0,
						const size_t P = 1
					) {
						return RealType(
							count,
							start,
							std::tuple< size_t, size_t, size_t >( offset, stride, repetitions ),
							dummy,
							s, P
						);
					}

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

					typedef typename grb::utils::iterators::Repeater< T > FactoryType;

					const T _val;

					const size_t _n;


				public:

					typedef typename grb::utils::iterators::Repeater< T >::RealType iterator;

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

					iterator begin( const size_t s = 0, const size_t P = 1 ) const {
						return FactoryType::make_iterator( _n, true, _val, s, P );
					}

					iterator end( const size_t s = 0, const size_t P = 1 ) const {
						return FactoryType::make_iterator( _n, false, _val, s, P );
					}

					const_iterator cbegin( const size_t s = 0, const size_t P = 1 ) const {
						return FactoryType::make_iterator( _n, true, _val, s, P );
					}

					const_iterator cend( const size_t s = 0, const size_t P = 1 ) const {
						return FactoryType::make_iterator( _n, false, _val, s, P );
					}

			};

			/**
			 * A container that contains a sequence of numbers with a given stride,
			 * and optionally a given number of repetitions.
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

					typedef grb::utils::iterators::Sequence< T > FactoryType;

					const size_t _start, _end;

					const size_t _stride;

					const size_t _repetitions;

					const size_t _count;


				public:

					typedef typename grb::utils::iterators::Sequence< T >::RealType iterator;

					typedef iterator const_iterator;

					/**
					 * Constructs a range.
					 *
					 * @param[in] start       The start of the range (inclusive)
					 * @param[in] end         The end of the range (exclusive)
					 * @param[in] stride      The stride of the range
					 *                        (optional, default is 1)
					 * @param[in] repetitions The number of repetitions of
					 *                        each value (optional, default is 1)
					 *
					 * The value \a end must be larger than or equal to \a start. Equal values
					 * for \a start and \a end result in an empty range. A larger value for
					 * \a end than \a start will result in a range consisting at least one
					 * element (\a start).
					 *
					 * \parblock
					 * \par Examples
					 * The range \f$ (1, 2, 3, 4, 5, 6, 7, 8, 9, 10) \f$ may be
					 * constructed by \a start 1, \a end 11, \a stride 1 and \a repetitions 1.
					 *
					 * The range \f$ (1, 3, 5, 7, 9) \f$ may be constructed by \a start 1,
					 * \a end 11, \a stride 2 and \a repetitions 1.
					 *
					 * The range \f$ (1, 1, 2, 2, 3, 3) \f$ may be constructed by
					 * \a start 1, \a end 4, \a stride 1 and \a repetitions 2.
					 *
					 * \endparblock
					 */
					Range(
						const size_t start,
						const size_t end,
						const size_t stride = static_cast< size_t >( 1 ),
						const size_t repetitions = static_cast< size_t >( 1 )
					) noexcept :
						_start( start ),
						_end( end ),
						_stride( stride ),
						_repetitions( repetitions ),
						_count(
							start == end
								? 0
								: ( (end-start) % stride > 0
									? ((end-start) / stride + 1)
									: (end-start) / stride
								) * repetitions
						)
					{
						assert( start <= end );
					}

					iterator begin( const size_t s = 0, const size_t P = 1 ) const {
						return FactoryType::make_iterator( _count, true, _start, _stride,
							_repetitions, T(), s, P );
					}

					iterator end( const size_t s = 0, const size_t P = 1 ) const {
						return FactoryType::make_iterator( _count, false, _start, _stride,
							_repetitions, T(), s, P );
					}

					const_iterator cbegin( const size_t s = 0, const size_t P = 1 ) const {
						return FactoryType::make_iterator( _count, true, _start, _stride,
							_repetitions, T(), s, P );
					}

					const_iterator cend( const size_t s = 0, const size_t P = 1 ) const {
						return FactoryType::make_iterator( _count, false, _start, _stride,
							_repetitions, T(), s, P );
					}

			};

		} // end namespace grb::utils::containers

	} // end namespace grb::utils

} // end namespace grb

#endif // end _H_GRB_ITERATOR_REGULAR

