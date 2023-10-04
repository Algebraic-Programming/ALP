
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
 * @file
 *
 * Provides an iterator that filters the contents of another iterator.
 *
 * @author A. N. Yzelman
 * @date 2nd of October, 2024
 */

#ifndef _H_ALP_UTILS_ITERATORS_FILTER
#define _H_ALP_UTILS_ITERATORS_FILTER

#include <iterator>


namespace grb {

	namespace utils {

		namespace iterators {

			/**
			 * This iterator filters elements from another iterator based on a user-
			 * specified filter function.
			 *
			 * @tparam FwdSubIter The underlying iterator type.
			 *
			 * Instances of this class may only be created via #IteratorFilter::create.
			 * This is because incrementing an instance of this iterator needs to ensure
			 * that while filtering input elements, it does not move past its end
			 * position. Therefore the safe use of this class always requires instances
			 * to be aware of what the underlying end position is, which is ensured by
			 * the #IteratorFilter::create API.
			 */
			template< typename FwdSubIter >
			class IteratorFilter;

			/**
			 * The factory function that takes any iterator pair over the same
			 * container and returns a matching filtered iterator pair.
			 *
			 * @param[in] begin A sub iterator in begin position.
			 * @param[in] end   A sub iterator that matches the end position of
			 *                  \a begin.
			 * @param[in] func  Values between \a begin and \a end for which this
			 *                  function evaluates <tt>true</tt> shall \em not be
			 *                  iterated over by the returned IteratorFilter instances.
			 *
			 * returns A pair of IteratorFilter instances, the first matching \a begin
			 *         and the second matching \a end.
			 *
			 * \note If all elements from \a begin to \a end in the underlying
			 *       collection are to be filtered according to \a func, then two
			 *       equal IteratorFilter instances may be returned, indicative of the
			 *       resulting empty set.
			 */
			template< typename FwdSubIter >
			std::pair<
				IteratorFilter< FwdSubIter >, IteratorFilter< FwdSubIter >
			> createIteratorFilter(
				FwdSubIter begin, const FwdSubIter end,
				const std::function< bool(const typename FwdSubIter::value_type &) > func
			);

			template< typename FwdSubIter >
			class IteratorFilter {

				static_assert( std::is_base_of<
						std::forward_iterator_tag,
						typename std::iterator_traits< FwdSubIter >::iterator_category
					>::value,
					"The sub-iterator to IteratorFilter must be a forward iterator."
				);


				protected:

					/** The value type of the underlying iterator. */
					typedef typename std::iterator_traits< FwdSubIter >::value_type ValT;

					/** The type of an user-specified filter function. */
					typedef typename std::function< bool( const ValT & ) > FilterT;


				private:

					/**
					 * Short-cut to our own type.
					 */
					typedef IteratorFilter< FwdSubIter > self_type;

					/**
					 * The underlying iterator.
					 */
					FwdSubIter it;

					/**
					 * Matching iterator in end position.
					 */
					const FwdSubIter end;

					/**
					 * The filter on the underlying iterator.
					 */
					FilterT filter;


				protected:

					/**
					 * \internal
					 * \warning This constructor directly sets its member fields. There is no
					 *          error checking of any kind.
					 * \endinternal
					 */
					IteratorFilter(
						FwdSubIter it_in, const FwdSubIter end_in,
						const FilterT func_in
					) :
						it( it_in ), end( end_in ), filter( func_in )
					{}


				public:

					// friend functions

					/**
					 * Swap two instances of the #grb::utils::iterators::IteratorFilter type.
					 *
					 * @tparam FwdSubIter The underlying iterator type.
					 *
					 * @param[in] left  The left-hand side input.
					 * @param[in] right The right-hand side input.
					 *
					 * This function will swap the contents of \a left with that of \a right.
					 */
					friend void swap(
						IteratorFilter< FwdSubIter > &left, IteratorFilter< FwdSubIter > &right
					) noexcept {
						std::swap( left.it, right.it );
						std::swap( left.end, right.end );
						std::swap( left.filter, right.filter );
					}

					template< typename T >
					friend std::pair<
						IteratorFilter< T >, IteratorFilter< T >
					> createIteratorFilter(
						T begin, const T end,
						const std::function< bool(const typename T::value_type &) > func
					);

					// STL typedefs

					typedef typename std::iterator_traits< self_type >::difference_type
						difference_type;
					typedef typename std::iterator_traits< self_type >::value_type value_type;
					typedef typename std::iterator_traits< self_type >::pointer pointer;
					typedef typename std::iterator_traits< self_type >::reference reference;
					typedef typename std::iterator_traits< self_type >::iterator_category
						iterator_category;

					/**
					 * An IteratorFilter may never be default-constructed.
					 *
					 * IteratorFilter instances may only be created via a call to #create.
					 */
					IteratorFilter() = delete;

					/**
					 * Copy constructor.
					 *
					 * @param[in] toCopy The iterator to create a copy of.
					 */
					IteratorFilter( const IteratorFilter< FwdSubIter > &toCopy ) :
						it( toCopy.it ), end( toCopy.end ), filter( toCopy.filter )
					{}

					/**
					 * Default destructor.
					 */
					~IteratorFilter() {}

					/**
					 * Copy-assignment.
					 *
					 * @param[in] toCopy The iterator to create a copy of.
					 *
					 * @returns A reference to this iterator, reflecting the state of this
					 *          instance after the copy operation has completed.
					 */
					IteratorFilter< FwdSubIter >& operator=(
						const IteratorFilter< FwdSubIter > &toCopy
					) {
						it = toCopy.it;
						filter = toCopy.filter;
						return *this;
					}

					/**
					 * Iterator increment operation.
					 *
					 * Always first performs the increment and \em then proceeds with
					 * filtering.
					 *
					 * \warning Hence, just as in the STL spec, calling this iterator on an
					 *          iterator instance in end position invites undefined behaviour.
					 *
					 * @returns A reference to this iterator, reflecting the state of this
					 *          instance after the increment operation has completed.
					 */
					IteratorFilter< FwdSubIter >& operator++() {
						(void) ++it;
						while( it != end && filter( *it ) ) {
							(void) ++it;
						}
						return *this;
					}

					/**
					 * Iterator increment operation.
					 *
					 * Always first performs the increment and \em then proceeds with
					 * filtering.
					 *
					 * \warning Hence, just as in the STL spec, calling this iterator on an
					 *          iterator instance in end position invites undefined behaviour.
					 *
					 * @returns Another instance of this iterator, reflecting the state of this
					 *          instance before the increment operation had completed.
					 */
					IteratorFilter< FwdSubIter > operator++(int) {
						IteratorFilter< FwdSubIter > ret( *this );
						(void) it++;
						while( it != end && filter( *it ) ) {
							(void) it++;
						}
						return ret;
					}

					/**
					 * Dereference operator.
					 *
					 * Does not check whether the iterator was in an end position.
					 *
					 * \warning Dereferencing an iterator in end position invitess undefined
					 *          behaviour.
					 */
					reference operator*() const {
						return *it;
					}

					/**
					 * Pointer operator.
					 *
					 * Does not check whether the iterator was in an end position.
					 *
					 * \warning Dereferencing an iterator in end position invitess undefined
					 *          behaviour.
					 */
					pointer operator->() const {
						return it.operator->();
					}

					/**
					 * Equality check.
					 *
					 * @param[in] other The iterator to check equality with.
					 *
					 * @return <tt>false</tt> if the iterators are not in the same position;
					 * @return <tt>true</tt> if the iterators are in the same position.
					 *
					 * \warning Only iterators that were constructed over the same container
					 *          may be compared. Otherwise, undefined behaviour is invited.
					 */
					bool operator==( const IteratorFilter< FwdSubIter >& other ) const {
						return it == other.it;
					}

					/**
					 * Inequality check.
					 *
					 * @param[in] other The iterator to check inequality with.
					 *
					 * @return <tt>false</tt> if the iterators are in the same position;
					 * @return <tt>true</tt> if the iterators are not in the same position.
					 *
					 * \warning Only iterators that were constructed over the same container
					 *          may be compared. Otherwise, undefined behaviour is invited.
					 */
					bool operator!=( const IteratorFilter< FwdSubIter >& other ) const {
						return it != other.it;
					}

			};

			template< typename FwdSubIter >
			std::pair<
				IteratorFilter< FwdSubIter >, IteratorFilter< FwdSubIter >
			> createIteratorFilter(
				FwdSubIter begin, const FwdSubIter end,
				const std::function< bool(const typename FwdSubIter::value_type &) > func
			) {
				std::pair <
					IteratorFilter< FwdSubIter >, IteratorFilter< FwdSubIter >
				> ret = {
					IteratorFilter< FwdSubIter >( begin, end, func ),
					IteratorFilter< FwdSubIter >( end, end, func )
				};
				if( begin != end ) {
					while( begin != end && func( *begin ) ) {
						(void) ++begin;
					}
					ret.first = IteratorFilter< FwdSubIter >( begin, end, func );
				}
				return ret;
			}

		} // end grb::utils::iterators namespace

	} // end grb::utils namespace

} // end grb namespace

namespace std {

	/**
	 * Iterator traits for the IteratorFilter.
	 *
	 * @tparam FwdSubIter The underlying iterator type.
	 *
	 * Inherits all values from the \a FwdSubIter, except for the iterator
	 * category tag, which shall be reduced to a forward iterator.
	 */
	template< typename FwdSubIter >
	struct iterator_traits<
		grb::utils::iterators::IteratorFilter< FwdSubIter >
	> {

		/** This trait is inherited from \a FwdSubIter. */
		typedef typename std::iterator_traits< FwdSubIter >::difference_type
			difference_type;

		/** This trait is inherited from \a FwdSubIter. */
		typedef typename std::iterator_traits< FwdSubIter >::value_type value_type;

		/** This trait is inherited from \a FwdSubIter. */
		typedef typename std::iterator_traits< FwdSubIter >::pointer pointer;

		/** This trait is inherited from \a FwdSubIter. */
		typedef typename std::iterator_traits< FwdSubIter >::reference reference;

		/** This trait is always forward iterator. */
		typedef typename std::forward_iterator_tag iterator_category;

	};

}

#endif // end _H_ALP_UTILS_ITERATORS_FILTER

