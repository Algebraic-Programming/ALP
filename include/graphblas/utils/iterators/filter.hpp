
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

/*
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
			 */
			template< typename FwdSubIter >
			class IteratorFilter {

				static_assert( std::is_base_of<
						std::forward_iterator_tag,
						std::iterator_traits< FwdSubIter >::iterator_category
					>::value,
					"The sub-iterator to IteratorFilter must be a forward iterator."
				);

				protected:

					/** The value type of the underlying iterator. */
					typedef typename std::iterator_traits< FwdSubIter >::value_type ValT;

					/** The type of the user-specified filter function. */
					typedef typename std::function< bool( const ValT & ) > FilterT;

					/**
					 * The return type of the factory function that creates IteratorFilter
					 * instances.
					 */
					typedef std::pair<
						IteratorFilter< FwdSubIter >,
						IteratorFilter< FwdSubIter >
					> FactoryOutputT;


				private:

					/**
					 * The underlying iterator.
					 */
					FwdSubIter it;

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
					IteratorFilter( FwdSubIter it_in, const FilterT func_in ) :
						it( it_in ), filter( func_in )
					{}


				public:

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
					static FactoryOutputT create(
						FwdSubIter begin, const FwdSubIter end, const FilterT func
					) {
						FactoryOutputT ret = {
							IteratorFilter( begin, func ),
							IteratorFilter( end, func )
						};
						if( begin != end ) {
							while( begin != end && !func( *begin ) ) {
								(void) ++begin;
							}
							ret.first = IteratorFilter( begin, func );
						}
						return ret;
					}

					/**
					 * An IteratorFilter may never be default-constructed.
					 *
					 * IteratorFilter instances may only be created via a call to #create.
					 */
					IteratorFilter() = delete;

					// TODO

			};

			/**
			 * Iterator traits for the IteratorFilter.
			 */
			template< typename FwdSubIter >
			struct std::iterator_traits< IteratorFilter< FwdSubIter > > {

				/** This trait is inherited from \a FwdSubIter. */
				typedef std::iterator_traits< FwdSubIter >::difference_type difference_type;

				/** This trait is inherited from \a FwdSubIter. */
				typedef std::iterator_traits< FwdSubIter >::value_type value_type;

				/** This trait is inherited from \a FwdSubIter. */
				typedef std::iterator_traits< FwdSubIter >::pointer pointer;

				/** This trait is inherited from \a FwdSubIter. */
				typedef std::iterator_traits< FwdSubIter >::reference reference;

				/** This trait is always forward iterator. */
				typedef std::forward_iterator_tag iterator_category;

			};

		}

	}

}

#endif // end _H_ALP_UTILS_ITERATORS_FILTER

