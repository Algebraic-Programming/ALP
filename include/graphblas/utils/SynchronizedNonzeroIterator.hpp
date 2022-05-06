
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
 * @date 2nd of August, 2017
 */

#ifndef _H_SYNCHRONIZEDNONZEROITERATOR
#define _H_SYNCHRONIZEDNONZEROITERATOR

#include <cstdlib>
#include <assert.h>
#include <utility>
#include <iterator>
#include <type_traits>

#include "NonZeroStorage.hpp"

#ifdef _DEBUG
#ifndef _GRB_NO_STDIO
#include <iostream>
#endif
#endif

namespace grb {
	namespace utils {

		// base class for storage, where V can be void
		template<
			typename RowIndexT,
			typename ColIndexT,
			typename V,
			typename fwd_it1,
			typename fwd_it2,
			typename _iterator_category
		> class __SynchronizedIteratorBaseStorage {

		protected:
			// iterators to synchronise:
			fwd_it1 row_it, row_end;
			fwd_it2 col_it, col_end;

			using self_t = __SynchronizedIteratorBaseStorage< RowIndexT, ColIndexT, V,
				fwd_it1, fwd_it2, _iterator_category >;
			using storage_t = NonZeroStorage< RowIndexT, ColIndexT, V >;

			mutable bool row_col_updated;
			mutable storage_t nonzero;

			/** Base constructor. Takes three sub-iterators as arguments. */
			__SynchronizedIteratorBaseStorage( fwd_it1 it1, fwd_it2 it2, fwd_it1 it1_end, fwd_it2 it2_end ) :
				row_it( it1 ), row_end( it1_end ), col_it( it2 ), col_end( it2_end ),
				row_col_updated( false ), nonzero() {
			}

			/** Copy constructor. */
			__SynchronizedIteratorBaseStorage( const self_t & other ) :
				row_it( other.row_it ), row_end( other.row_end ),
				col_it( other.col_it ), col_end( other.col_end ),
				row_col_updated( other.row_col_updated ), nonzero() {
			}

			/** Assignment operator. */
			self_t & operator=( const self_t & other ) {
				row_it = other.row_it;
				row_end = other.row_end;
				col_it = other.col_it;
				col_end = other.col_end;
				row_col_updated = other.row_col_updated;
				return *this;
			}

			bool row_col_iterators_are_valid() const {
				return row_it != row_end && col_it != col_end;
			}

			inline void row_col_update_if_needed() const {
				if( !row_col_updated ) {
					assert( row_col_iterators_are_valid() );
					row_col_update();
				}
			}

			/** Updates the #nonzero fields using the current iterator values. */
			void row_col_update() const {
				assert( row_col_iterators_are_valid() );
				nonzero.i() = *row_it;
				nonzero.j() = *col_it;
				row_col_updated = true;
			}

			/** Inequality check. */
			bool operator!=( const self_t &other ) const {
				return row_it != other.row_it || col_it != other.col_it;
			};

			/** Equality check. */
			bool operator==( const self_t &other ) const {
				return ! operator!=( other );
			}

			/** Increment operator. */
			self_t & operator++() {
				(void) ++row_it;
				(void) ++col_it;
				row_col_updated = false;
				return *this;
			}

		public:

			// STL iterator's typedefs:
			using iterator_category = _iterator_category;
    		using value_type = typename storage_t::storage_t;
    		using difference_type = long;
			using reference = const value_type &;
			using pointer = const value_type *;

			// GraphBLAS typedefs:
			using row_coordinate_type = RowIndexT;
			using column_coordinate_type = ColIndexT;

			/** Direct derefence operator. */
			reference operator*() const {
				row_col_update_if_needed();
				return nonzero.storage();
			}

			/** Pointer update. */
			pointer operator->() const {
				row_col_update_if_needed();
				return &(nonzero.storage());
			}

			/** Returns the row coordinate. */
			const row_coordinate_type & i() const {
				row_col_update_if_needed();
				return nonzero.i();
			}

			/** Returns the column coordinate. */
			const column_coordinate_type & j() const {
				row_col_update_if_needed();
				return nonzero.j();
			}
		};


		template<
			typename RowIndexT,
			typename ColIndexT,
			typename V,
			typename fwd_it1,
			typename fwd_it2,
			typename fwd_it3,
			typename _iterator_category
		> class __SynchronizedIteratorBase:
			public __SynchronizedIteratorBaseStorage< RowIndexT, ColIndexT, V,
				fwd_it1, fwd_it2, _iterator_category > {

			using base_t = __SynchronizedIteratorBaseStorage< RowIndexT, ColIndexT, V,
				fwd_it1, fwd_it2, _iterator_category>;

		protected:
			// iterators to synchronise:
			fwd_it3 val_it, val_end;
			mutable bool val_updated;

			using self_t = __SynchronizedIteratorBase< RowIndexT, ColIndexT, V,
				fwd_it1, fwd_it2, fwd_it3, _iterator_category >;

			/** Base constructor. Takes three sub-iterators as arguments. */
			__SynchronizedIteratorBase( fwd_it1 it1, fwd_it2 it2, fwd_it3 it3,
				fwd_it1 it1_end, fwd_it2 it2_end, fwd_it3 it3_end ) :
				base_t( it1, it2, it1_end, it2_end), val_it( it3 ), val_end( it3_end ), val_updated( false ) {
			}

			/** Copy constructor. */
			__SynchronizedIteratorBase( const self_t &other ) : base_t( other ),
				val_it( other.val_it ), val_end( other.val_end ), val_updated( other.val_updated ) {}

			/** Assignment operator. */
			self_t & operator=( const self_t &other ) {
				(void) base_t::operator=( other );
				val_it = other.val_it;
				val_end = other.val_end;
				val_updated = other.val_updated;
				return *this;
			}

			bool val_iterator_is_valid() const {
				return val_it != val_end;
			}

			/** Updates the #nonzero.v() field using the current iterator value. */
			void val_update() const {
				assert( val_iterator_is_valid() );
				this->nonzero.v() = *val_it;
				val_updated = true;
			}

			void val_update_if_needed() const {
				if( !val_updated ) {
					val_update();
				}
			}

			inline void update_if_needed() const {
				this->row_col_update_if_needed();
				val_update_if_needed();
			}

			/** Equality check. */
			bool operator==( const self_t & other ) const {
				return base_t::operator==( other ) && val_it == other.val_it;
			}

			/** Inequality check. */
			bool operator!=( const self_t & other ) const {
				return base_t::operator!=( other ) || val_it != other.val_it;
			};

			/** Increment operator. */
			self_t & operator++() {
				(void) base_t::operator++();
				(void) ++val_it;
				val_updated = false;
				return *this;
			}

		public:

			// GraphBLAS typedefs:
			using nonzero_value_type = V;

			/** Direct derefence operator. */
			typename base_t::reference operator*() const {
				update_if_needed();
				return this->nonzero;
			}

			/** Pointer update. */
			typename base_t::pointer operator->() const {
				update_if_needed();
				return &(this->nonzero);
			}

			/** Returns the nonzero coordinate. */
			const nonzero_value_type & v() const {
				val_update_if_needed();
				return this->nonzero.v();
			}
		};

		// for value matrices
		template<
			typename RowIndexT,
			typename ColIndexT,
			typename V,
			typename fwd_it1,
			typename fwd_it2,
			typename fwd_it3,
			class _iterator_category
		> class SynchronizedNonzeroIterator:
			public __SynchronizedIteratorBase< RowIndexT, ColIndexT, V,
				fwd_it1, fwd_it2, fwd_it3, _iterator_category > {

			using base_t = __SynchronizedIteratorBase< RowIndexT, ColIndexT, V,
				fwd_it1, fwd_it2, fwd_it3, _iterator_category >;

		public:

			using self_t = SynchronizedNonzeroIterator< RowIndexT, ColIndexT, V,
				fwd_it1, fwd_it2, fwd_it3, _iterator_category >;

			/** Base constructor. Takes three sub-iterators as arguments. */
			SynchronizedNonzeroIterator( fwd_it1 it1, fwd_it2 it2, fwd_it3 it3,
				fwd_it1 it1_end, fwd_it2 it2_end, fwd_it3 it3_end ):
				base_t( it1, it2, it3, it1_end, it2_end, it3_end ){}

			/** Copy constructor. */
			SynchronizedNonzeroIterator( const self_t &other ): base_t( other ) {}

			/** Assignment operator. */
			self_t & operator=( const self_t &other ) {
				(void) base_t::operator=( other );
				return *this;
			}

			/** Equality check. */
			bool operator==( const self_t &other ) const {
				return base_t::operator==( other );
			}

			/** Inequality check. */
			bool operator!=( const self_t &other ) const {
				return base_t::operator!=( other );
			};

			/** Increment operator. */
			self_t & operator++() {
				(void) base_t::operator++();
				return *this;
			}

			/** Offset operator, enabled only for random access iterators */
			template< typename __cat = _iterator_category > self_t & operator+=(
				typename std::enable_if<
					std::is_same< __cat, std::random_access_iterator_tag >::value // enable only for random access
						&& std::is_same< typename self_t::iterator_category,
							std::random_access_iterator_tag
						>::value,
					std::size_t
				>::type offset ) {
				this->row_it += offset;
				this->col_it += offset;
				this->val_it += offset;
				this->row_col_updated = false;
				this->val_updated = false;
				return *this;
			}

			/* Difference operator, enabled only for random access iterators */
			template< typename _cat = _iterator_category >
				typename self_t::difference_type operator-(
				typename std::enable_if<
					std::is_same< _cat, std::random_access_iterator_tag >::value // to enable/disable operator
						&& std::is_same< _iterator_category, // to prevent misues
						std::random_access_iterator_tag
					>::value,
					const self_t&
				>::type other ) const {
				return static_cast< typename self_t::difference_type >(
					this->row_it - other.row_it );
			}
		};


		// for pattern matrices
		template<
			typename RowIndexT,
			typename ColIndexT,
			typename fwd_it1,
			typename fwd_it2,
			class _iterator_category
		> class SynchronizedNonzeroIterator< RowIndexT, ColIndexT, void,
			fwd_it1, fwd_it2, void, _iterator_category >:
			public __SynchronizedIteratorBaseStorage< RowIndexT, ColIndexT, void,
				fwd_it1, fwd_it2, _iterator_category > {

			using base_t = __SynchronizedIteratorBaseStorage< RowIndexT, ColIndexT, void,
				fwd_it1, fwd_it2, _iterator_category >;

		public:

			using self_t = SynchronizedNonzeroIterator< RowIndexT, ColIndexT, void,
				fwd_it1, fwd_it2, void, _iterator_category >;

			/** Base constructor. Takes three sub-iterators as arguments. */
			SynchronizedNonzeroIterator( fwd_it1 it1, fwd_it2 it2,
				fwd_it1 it1_end, fwd_it2 it2_end ):
				base_t( it1, it2, it1_end, it2_end ){}

			/** Copy constructor. */
			SynchronizedNonzeroIterator( const self_t &other ): base_t( other ) {}

			/** Assignment operator. */
			self_t & operator=( const self_t &other ) {
				(void) base_t::operator=( other );
				return *this;
			}

			/** Equality check. */
			bool operator==( const self_t &other ) const {
				return base_t::operator==( other );
			}

			/** Inequality check. */
			bool operator!=( const self_t &other ) const {
				return base_t::operator!=( other );
			};

			/** Increment operator. */
			self_t & operator++() {
				(void) base_t::operator++();
				return *this;
			}


			/** Offset operator, enabled only for random access iterators */
			template< typename __cat = _iterator_category > self_t & operator+=(
				typename std::enable_if<
					std::is_same< __cat, std::random_access_iterator_tag >::value // enable only for random access
						&& std::is_same< typename self_t::iterator_category,
							std::random_access_iterator_tag
						>::value,
					std::size_t
				>::type offset ) {
				this->row_it += offset;
				this->col_it += offset;
				this->row_col_updated = false;
				return *this;
			}

			/* Difference operator, enabled only for random access iterators */
			template< typename _cat = _iterator_category >
				typename self_t::difference_type operator-(
				typename std::enable_if<
					std::is_same< _cat, std::random_access_iterator_tag >::value // to enable/disable operator
						&& std::is_same< _iterator_category, // to prevent misues
						std::random_access_iterator_tag
					>::value,
					const self_t&
				>::type other ) const {
				return static_cast< typename self_t::difference_type >(
					this->row_it - other.row_it );
			}
		};

#ifdef _DEBUG
		template< typename RowIndexT, typename ColIndexT, typename V,
				  typename fwd_it1, typename fwd_it2, typename fwd_it3, typename iterator_category >
		std::ostream & operator<<( std::ostream & os,
			const SynchronizedNonzeroIterator< RowIndexT, ColIndexT, V, fwd_it1, fwd_it2, fwd_it3, iterator_category > & it ) {
			os << it.i() << ", " << it.j() << ", " << it.v();
			return os;
		}

		template< typename RowIndexT, typename ColIndexT, typename fwd_it1, typename fwd_it2, typename iterator_category >
			std::ostream & operator<<( std::ostream & os,
			const SynchronizedNonzeroIterator< RowIndexT, ColIndexT, void, fwd_it1, fwd_it2, void, iterator_category > & it ) {
			os << it.i() << ", " << it.j();
			return os;
		}
#endif

		// overload for 3 pointers and custom tag
		template<
			typename RowIndexT,
			typename ColIndexT,
			typename V,
			typename iterator_category
		> SynchronizedNonzeroIterator<RowIndexT, ColIndexT, V,
			const RowIndexT *, const ColIndexT *, const V *, iterator_category >
		makeSynchronized( const RowIndexT * const it1, const ColIndexT * const it2, const V  * const it3,
			const RowIndexT * const it1_end, const ColIndexT * const it2_end, const V  * const it3_end, iterator_category ) {
			return SynchronizedNonzeroIterator < RowIndexT, ColIndexT, V, const RowIndexT *, const ColIndexT *, const V *, iterator_category >
		    	( it1, it2, it3, it1_end, it2_end, it3_end );
		}


		// overload for 3 pointers without tag: get the best (random access iterator)
		template<
			typename RowIndexT,
			typename ColIndexT,
			typename V
		> SynchronizedNonzeroIterator< RowIndexT, ColIndexT, V,
			const RowIndexT *, const ColIndexT *, const V *, std::random_access_iterator_tag >
		makeSynchronized( const RowIndexT * const it1, const ColIndexT * const it2, const V  * const it3,
			const RowIndexT * const it1_end, const ColIndexT * const it2_end, const V  * const it3_end ) {
			return makeSynchronized( it1, it2, it3,
				it1_end, it2_end, it3_end, std::random_access_iterator_tag() );
		}


		// overload for 2 pointers and custom tag
		template< typename RowIndexT, typename ColIndexT, typename iterator_category >
		SynchronizedNonzeroIterator< RowIndexT, ColIndexT, void,
			const RowIndexT *, const ColIndexT *, void, iterator_category >
		makeSynchronized( const RowIndexT * const it1, const ColIndexT * const it2,
			const RowIndexT * const it1_end, const ColIndexT * const it2_end, iterator_category ) {
			return SynchronizedNonzeroIterator < RowIndexT, ColIndexT, void,
				const RowIndexT *, const ColIndexT *, void, iterator_category >
		    	( it1, it2, it1_end, it2_end );
		}


		// overload for 2 pointers with no tag: get the best (random access iterator)
		template< typename RowIndexT, typename ColIndexT >
		SynchronizedNonzeroIterator< RowIndexT, ColIndexT, void, const RowIndexT *, const ColIndexT *, void,
			std::random_access_iterator_tag >
		makeSynchronized( const RowIndexT * const it1, const ColIndexT * const it2,
			const RowIndexT * const it1_end, const ColIndexT * const it2_end ) {
#ifdef _DEBUG
#ifndef _GRB_NO_STDIO
			std::cout << "SynchronizedNonzeroIterator::makeSynchronized "
						 "received iterators "
					  << it1 << " (start) and " << it2 << " (end)\n";
#else
			printf( "SynchronizedNonzeroIterator::makeSynchronized received "
					"iterators %p (start) and %p (end)\n",
				it1, it2 );
#endif
#endif
			return makeSynchronized( it1, it2, it1_end, it2_end, std::random_access_iterator_tag() );
		}

		// overload for 2 iterators with custom tag
		template<
			typename fwd_it1,
			typename fwd_it2,
			typename iterator_category
		> SynchronizedNonzeroIterator<
			typename iterator_value< fwd_it1 >::type,
			typename iterator_value< fwd_it2 >::type,
			void, fwd_it1, fwd_it2, void, iterator_category
		> makeSynchronized( const fwd_it1 it1, const fwd_it2 it2,
			const fwd_it1 it1_end, const fwd_it2 it2_end, iterator_category ) {
			return SynchronizedNonzeroIterator<
				typename iterator_value< fwd_it1 >::type,
				typename iterator_value< fwd_it2 >::type,
				void, fwd_it1, fwd_it2, void, iterator_category
			>( it1, it2, it1_end, it2_end );
		}

		// overload for 2 iterators without tag: get common tag
		template<
			typename fwd_it1,
			typename fwd_it2
		> SynchronizedNonzeroIterator<
			typename iterator_value< fwd_it1 >::type,
			typename iterator_value< fwd_it2 >::type,
			void, fwd_it1, fwd_it2, void,
			typename common_iterator_tag< fwd_it1, fwd_it2 >::iterator_category
		> makeSynchronized( const fwd_it1 it1, const fwd_it2 it2,
			const fwd_it1 it1_end, const fwd_it2 it2_end ) {
			using cat = typename common_iterator_tag< fwd_it1, fwd_it2 >::iterator_category;
		  	return makeSynchronized(it1, it2, it1_end, it2_end, cat() );
		}

		// overload for 3 iterators with custom tag
		template<
			typename fwd_it1,
			typename fwd_it2,
			typename fwd_it3,
			typename iterator_category
		> SynchronizedNonzeroIterator<
			typename iterator_value< fwd_it1 >::type,
			typename iterator_value< fwd_it2 >::type,
			typename iterator_value< fwd_it3 >::type,
			fwd_it1, fwd_it2, fwd_it3, iterator_category
		> makeSynchronized( const fwd_it1 it1, const fwd_it2 it2, const fwd_it3 it3,
			const fwd_it1 it1_end, const fwd_it2 it2_end, const fwd_it3 it3_end, iterator_category ) {
		  	return SynchronizedNonzeroIterator<
				typename iterator_value< fwd_it1 >::type,
				typename iterator_value< fwd_it2 >::type,
				typename iterator_value< fwd_it3 >::type,
				fwd_it1, fwd_it2, fwd_it3, iterator_category
			> (it1, it2, it3, it1_end, it2_end, it3_end );
		}

		// overload for 3 iterators without tag: get common tag
		template<
			typename fwd_it1,
			typename fwd_it2,
			typename fwd_it3
		> SynchronizedNonzeroIterator<
			typename iterator_value< fwd_it1 >::type,
			typename iterator_value< fwd_it2 >::type,
			typename iterator_value< fwd_it3 >::type,
			fwd_it1, fwd_it2, fwd_it3,
			typename common_iterator_tag< fwd_it1, fwd_it2, fwd_it3 >::iterator_category
		> makeSynchronized( const fwd_it1 it1, const fwd_it2 it2, const fwd_it3 it3,
			const fwd_it1 it1_end, const fwd_it2 it2_end, const fwd_it3 it3_end ) {
			using cat = typename common_iterator_tag< fwd_it1, fwd_it2, fwd_it3 >::iterator_category;
		  	return makeSynchronized(it1, it2, it3, it1_end, it2_end, it3_end, cat() );
		}

	} // namespace utils
} // namespace grb

#endif // end ``_H_SYNCHRONIZEDNONZEROITERATOR''
