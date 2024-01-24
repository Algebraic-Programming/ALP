
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
 * Defines an iterator that adapts the values returned by a sub-iterator
 * according to some user-defined lambda.
 *
 * @author A. N. Yzelman
 * @date 19/12/2023
 */

#ifndef _H_GRB_ITERATOR_ADAPTER
#define _H_GRB_ITERATOR_ADAPTER

#include <utility>
#include <iterator>
#include <functional>

#include <assert.h>


namespace grb::utils {

	namespace iterators {

		/**
		 * An iterator that simply adapts the values returned by another iterator.
		 *
		 * The resulting iterator takes all the properties of the sub-iterator. It
		 * will only support <tt>const</tt> views of the collection underlying the
		 * sub-iterator.
		 *
		 * \note Currently the adapter supports random access iterators only. If
		 *       support for other iterator categories is required, please submit
		 *       a feature request and/or contact the maintainers.
		 *
		 * @tparam SubIterT The underlying iterator type.
		 *
		 * For ease-of-use, see #make_adapter_iterator(). If not using that factory
		 * method and rather this class directly, the type \a SubIterT must always be
		 * explicitly given.
		 *
		 * \internal This adapter implementation is for random access iterators.
		 */
		template<
			typename SubIterT,
			typename std::enable_if< std::is_same<
				typename std::iterator_traits< SubIterT >::iterator_category,
				std::random_access_iterator_tag
			>::value, void >::type * = nullptr
		>
		class Adapter {

			public:

				// standard STL typedefs

				typedef typename SubIterT::iterator_category iterator_category;

				typedef typename SubIterT::difference_type difference_type;

				typedef typename SubIterT::value_type value_type;

				typedef typename SubIterT::pointer pointer;

				typedef typename SubIterT::reference reference;


			protected:

				// STL-like typedefs

				typedef Adapter< SubIterT > self_type;

				typedef self_type & self_reference_type;

				typedef const Adapter< SubIterT > self_const_reference_type;

				typedef const value_type & const_reference;

				typedef const value_type * const_pointer;


			private:

				value_type v;

				SubIterT iter;

				SubIterT end_it;

				std::function< value_type(const value_type) > adapter_func;


			public:

				// constructors

				Adapter(
					const SubIterT iter_in, const SubIterT iter_end,
					const std::function< value_type( const value_type ) > func_in
				) : iter( iter_in ), end_it( iter_end ), adapter_func( func_in ) {
					if( iter != end_it ) {
						v = adapter_func( *iter );
					}
				}

				Adapter( const self_const_reference_type &other ) :
					v( other.v ),
					iter( other.iter ), end_it( other.end_it ),
					adapter_func( other.adapter_func )
				{}

				Adapter( Adapter< SubIterT > &&other ) :
					v( std::move( other.v ) ),
					iter( std::move( other.iter ) ), end_it( std::move( other.end_it ) ),
					adapter_func( std::move( other.adapter_func ) )
				{}

				// destructor

				~Adapter() {}

				// standard iterator interface

				const_reference operator*() const noexcept {
					return v;
				}

				self_reference_type operator=( self_const_reference_type other ) noexcept {
					v = other.v;
					iter = other.iter;
					end_it = other.end_it;
					adapter_func = other.adapter_func;
					return *this;
				}

				self_reference_type operator=(
					Adapter< SubIterT > &&other
				) noexcept {
					v = std::move( other.v );
					iter = std::move( other.iter );
					end_it = std::move( other.end_it );
					adapter_func = std::move( other.adapter_func );
					return *this;
				}

				self_reference_type operator++() noexcept {
					(void) iter++;
					if( iter != end_it ) {
						v = adapter_func( *iter );
					}
					return *this;
				}

				friend void swap( self_reference_type left, self_reference_type right ) {
					std::swap( left.v, right.v );
					std::swap( left.iter, right.iter );
					std::swap( left.end_it, right.end_it );
					std::swap( left.adapter_func, right.adapter_func );
				}

				// input iterator interface

				const_pointer operator->() const noexcept {
					return &v;
				}

				self_type operator++(int) noexcept {
					self_type ret = self_type( *this );
					(void) ++iter;
					if( iter != end_it ) {
						v = adapter_func( *iter );
					}
					return ret;
				}

				friend bool operator==(
					self_const_reference_type left, self_const_reference_type right
				) noexcept {
					return left.iter == right.iter;
				}

				friend bool operator!=(
					self_const_reference_type left, self_const_reference_type right
				) noexcept {
					return !(left == right);
				}

				// bi-directional iterator interface

				self_reference_type operator--() noexcept {
					(void) --iter;
					if( iter != end_it ) {
						v = adapter_func( *iter );
					}
					return *this;
				}

				self_type operator--(int) noexcept {
					self_type ret = self_type( *this );
					(void) iter--;
					if( iter != end_it ) {
						v = adapter_func( *iter );
					}
					return ret;
				}

				// random access iterator interface

				/**
				 * \internal This bracket-operator cannot return a reference, as there is
				 *           no storage associated to all items iterated over. (This is
				 *           also a reason why this is a const-iterator only.)
				 */
				value_type operator[]( const size_t i ) const noexcept {
					return adapter_func( iter[ i ] );
				}

				friend bool operator<(
					self_const_reference_type left,
					self_const_reference_type right
				) {
					return left.iter < right.iter;
				}

				friend bool operator>(
					self_const_reference_type left,
					self_const_reference_type right
				) {
					return left.iter > right.iter;
				}

				friend bool operator<=(
					self_const_reference_type left,
					self_const_reference_type right
				) {
					return left.iter <= right.iter;
				}

				friend bool operator>=(
					self_const_reference_type left,
					self_const_reference_type right
				) {
					return left.iter >= right.iter;
				}

				self_reference_type operator+=( const size_t count ) noexcept {
					iter += count;
					if( iter != end_it ) {
						v = adapter_func( *iter );
					}
					return *this;
				}

				friend self_type operator+(
					 self_const_reference_type iterator,
					 const size_t count
				) noexcept {
					const SubIterT subRet = iterator.iter + count;
					return self_type( subRet, iterator.end_it, iterator.adapter_func );
				}

				friend self_type operator+(
					const size_t count,
					self_const_reference_type iterator
				) noexcept {
					const SubIterT subRet = iterator.iter + count;
					return self_type( subRet, iterator.end_it, iterator.adapter_func );
				}

				self_reference_type operator-=( const size_t count ) noexcept {
					iter -= count;
					if( iter != end_it ) {
						v = adapter_func( *iter );
					}
					return *this;
				}

				friend self_type operator-(
					self_const_reference_type iterator,
					const size_t count
				) noexcept {
					const SubIterT subRet = iterator.iter - count;
					return self_type( subRet, iterator.end_it, iterator.adapter_func );
				}

				friend self_type operator-(
					const size_t count,
					self_const_reference_type iterator
				) noexcept {
					const SubIterT subRet = iterator.iter - count;
					return self_type( subRet, iterator.end_it, iterator.adapter_func );
				}

				difference_type operator-(
					self_const_reference_type iterator
				) const noexcept {
					return iter - iterator.iter;
				}

		};

		// factory

		/**
		 * Creates an adapter of a given iterator.
		 *
		 * @tparam SubIterT The type of the given iterator.
		 *
		 * \warning Not all iterator categories are presently supported.
		 *
		 * Only random random access iterators are currently supported.
		 *
		 * @param[in] start The given iterator whose values shall be adapted.
		 * @param[in] end   The end-iterator that matches \a start.
		 * @param[in] func  The function by which the values of the given iterator
		 *                  shall be adapted. The function must take a single const
		 *                  value of the iterator value type, and shall return the
		 *                  modified value type.
		 *
		 * @returns An iterator in the same position as \a start but whose values
		 *          will be modified according to \a func.
		 *
		 * The returned iterator is a const-iterator, meaning the values iterated over
		 * cannot be modified-- even if the original iterator supported this.
		 *
		 * The iterator adapter does not support changing the original value type of
		 * the underlying \a start and \a end iterators. If such functionality would
		 * be useful, please submit a feature request.
		 */
		template< typename SubIterT >
		static Adapter< SubIterT > make_adapter_iterator(
			const SubIterT start, const SubIterT end,
			const std::function<
				typename SubIterT::value_type(const typename SubIterT::value_type)
			> func
		) {
			return Adapter< SubIterT >( start, end, func );
		}

	} // end namespace grb::utils::iterators

} // end namespace grb

#endif // end _H_GRB_ITERATOR_ADAPTER

