
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

#include "nonzeroStorage.hpp"
#include "utils/iterators/type_traits.hpp"

#ifdef _DEBUG
 #ifndef _GRB_NO_STDIO
  #include <iostream>
 #endif
#endif


namespace grb {

	namespace internal {

		/**
		 * Base class for the #SynchronizedIterator which allows for <tt>void</tt>
		 * value types.
		 */
		template<
			typename RowIndexT, typename ColIndexT, typename V,
			typename fwd_it1, typename fwd_it2,
			typename IteratorCategory
		>
		class SynchronizedIteratorBaseStorage {

			protected:

				// iterators to synchronise:

				/** The row coordinate iterator in start position. */
				fwd_it1 row_it;

				/** The row coordinate iterator in end position. */
				fwd_it1 row_end;

				/** The column coordinate iterator in start position. */
				fwd_it2 col_it;

				/** The column coordinate iterator in end position. */
				fwd_it2 col_end;

				/** The type of this class. */
				using SelfType = SynchronizedIteratorBaseStorage<
					RowIndexT, ColIndexT, V,
					fwd_it1, fwd_it2, IteratorCategory
				>;

				/** The type of the storage of a single nonzero. */
				using StorageType = NonzeroStorage< RowIndexT, ColIndexT, V >;

				/** Whether the row or column coordinate fields have been updated. */
				mutable bool row_col_updated;

				/**
				 * Stores a nonzero.
				 *
				 * In the <tt>void</tt> case, this corresponds to a single coordinate pair.
				 *
				 * In the non-void case, this corresponds to a pair of coordinates with a
				 * value, where the coordinate is a pair of integers.
				 */
				mutable StorageType nonzero;

				/**
				 * Base constructor.
				 *
				 * Takes the coordinate iterators as arguments.
				 */
				SynchronizedIteratorBaseStorage(
					fwd_it1 it1, fwd_it2 it2,
					fwd_it1 it1_end, fwd_it2 it2_end
				) :
					row_it( it1 ), row_end( it1_end ),
					col_it( it2 ), col_end( it2_end ),
					row_col_updated( false ), nonzero()
				{}

				/** Copy constructor. */
				SynchronizedIteratorBaseStorage( const SelfType &other ) :
					row_it( other.row_it ), row_end( other.row_end ),
					col_it( other.col_it ), col_end( other.col_end ),
					row_col_updated( other.row_col_updated ), nonzero()
				{}

				/** Assignment operator. */
				SelfType& operator=( const SelfType &other ) {
					row_it = other.row_it;
					row_end = other.row_end;
					col_it = other.col_it;
					col_end = other.col_end;
					row_col_updated = other.row_col_updated;
					return *this;
				}

				/** Whether the current coordinate iterators are in a valid position. */
				inline bool row_col_iterators_are_valid() const {
					return row_it != row_end && col_it != col_end;
				}

				/**
				 * Checks whether the coordinates require updating and, if so, updates them.
				 */
				inline void row_col_update_if_needed() const {
					if( !row_col_updated ) {
						assert( row_col_iterators_are_valid() );
						row_col_update();
					}
				}

				/** Updates the #nonzero fields using the current iterator values. */
				inline void row_col_update() const {
					assert( row_col_iterators_are_valid() );
					nonzero.i() = *row_it;
					nonzero.j() = *col_it;
					row_col_updated = true;
				}

				/** Inequality check. */
				bool operator!=( const SelfType &other ) const {
					return row_it != other.row_it || col_it != other.col_it;
				};

				/** Equality check. */
				bool operator==( const SelfType &other ) const {
					return !(operator!=( other ));
				}

				/** Increment operator. */
				SelfType & operator++() {
					(void) ++row_it;
					(void) ++col_it;
					row_col_updated = false;
					return *this;
				}


			public:

				// STL iterator's typedefs:
				typedef IteratorCategory iterator_category;
				typedef typename StorageType::StorageType value_type;
				typedef size_t difference_type;
				typedef const value_type & reference;
				typedef const value_type * pointer;

				// ALP typedefs:
				typedef RowIndexT RowIndexType;
				typedef ColIndexT ColumnIndexType;

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
				const RowIndexType & i() const {
					row_col_update_if_needed();
					return nonzero.i();
				}

				/** Returns the column coordinate. */
				const ColumnIndexType & j() const {
					row_col_update_if_needed();
					return nonzero.j();
				}

		};

		/**
		 * Synchronises three input iterators to act as a single iterator over matrix
		 * nonzeroes.
		 *
		 * This is the non-void (for value types) variant-- entries consist of a
		 * coordinate with a nonzero value. A coordinate consists of a pair of integer
		 * values.
		 */
		template<
			typename RowIndexT, typename ColIndexT, typename V,
			typename fwd_it1, typename fwd_it2, typename fwd_it3,
			class iterator_category
		>
		class SynchronizedNonzeroIterator :
			public internal::SynchronizedIteratorBaseStorage<
				RowIndexT, ColIndexT, V,
				fwd_it1, fwd_it2,
				iterator_category
			>
		{

			private:

				/** Short name of the parent class */
				using BaseType = internal::SynchronizedIteratorBaseStorage<
					RowIndexT, ColIndexT, V,
					fwd_it1, fwd_it2,
					iterator_category
				>;


			protected:

				/**
				 * The value iterator in start position.
				 */
				fwd_it3 val_it;

				/**
				 * The value iterator in end position.
				 */
				fwd_it3 val_end;

				/** Whether the value was updated. */
				mutable bool val_updated;

				/** Short name for the type of this class. */
				using SelfType = SynchronizedNonzeroIterator<
					RowIndexT, ColIndexT, V,
					fwd_it1, fwd_it2, fwd_it3,
					iterator_category
				>;

				/** Updates the #nonzero.v() field using the current iterator value. */
				inline void val_update() const {
					assert( val_it != val_end );
					this->nonzero.v() = *val_it;
					val_updated = true;
				}

				/**
				 * If the value storage was not updated, updates it.
				 */
				inline void val_update_if_needed() const {
					if( !val_updated ) {
						val_update();
					}
				}

				/**
				 * Check whether all nonzero entries are up to date.
				 */
				inline void update_if_needed() const {
					this->row_col_update_if_needed();
					val_update_if_needed();
				}


			public:

				/** ALP value typedef */
				typedef V ValueType;

				/** Base constructor. Takes three sub-iterators as arguments. */
				SynchronizedNonzeroIterator(
					fwd_it1 it1, fwd_it2 it2, fwd_it3 it3,
					fwd_it1 it1_end, fwd_it2 it2_end, fwd_it3 it3_end
				) : BaseType(
						it1, it2,
						it1_end, it2_end
					), val_it( it3 ), val_end( it3_end ), val_updated( false )
				{}

				/** Copy constructor. */
				SynchronizedNonzeroIterator( const SelfType &other ):
					BaseType( other ),
					val_it( other.val_it ), val_end( other.val_end ),
					val_updated( other.val_updated )
				{}

				/** Assignment operator. */
				SelfType & operator=( const SelfType &other ) {
					(void) BaseType::operator=( other );
					val_it = other.val_it;
					val_end = other.val_end;
					val_updated = other.val_updated;
					return *this;
				}

				/** Equality check. */
				bool operator==( const SelfType &other ) const {
					return BaseType::operator==( other ) && val_it == other.val_it;
				}

				/** Inequality check. */
				bool operator!=( const SelfType &other ) const {
					return BaseType::operator!=( other ) || val_it != other.val_it;
				};

				/** Increment operator. */
				SelfType & operator++() {
					(void) BaseType::operator++();
					(void) ++val_it;
					val_updated = false;
					return *this;
				}

				/** Offset operator, enabled only for random access iterators */
				template< typename cat = iterator_category > SelfType & operator+=(
					typename std::enable_if<
						std::is_same< cat, std::random_access_iterator_tag >::value &&
						std::is_same< typename SelfType::iterator_category,
								std::random_access_iterator_tag
							>::value,
						size_t
					>::type offset
				) {
					this->row_it += offset;
					this->col_it += offset;
					this->val_it += offset;
					this->row_col_updated = false;
					this->val_updated = false;
					return *this;
				}

				/** Difference operator, enabled only for random access iterators */
				template< typename _cat = iterator_category >
					typename SelfType::difference_type operator-(
					typename std::enable_if<
						std::is_same< _cat, std::random_access_iterator_tag >::value
							&& std::is_same< iterator_category,
							std::random_access_iterator_tag
						>::value,
						const SelfType&
					>::type other
				) const {
					return static_cast< typename SelfType::difference_type >(
						this->row_it - other.row_it );
				}

				/** Direct derefence operator. */
				typename BaseType::reference operator*() const {
					update_if_needed();
					return this->nonzero;
				}

				/** Pointer update. */
				typename BaseType::pointer operator->() const {
					update_if_needed();
					return &(this->nonzero);
				}

				/** Returns the nonzero coordinate. */
				const ValueType& v() const {
					val_update_if_needed();
					return this->nonzero.v();
				}

		};

		/**
		 * Synchronises two input iterators to act as a single iterator over matrix
		 * nonzeroes.
		 *
		 * This is the <tt>void</tt> (for value types) variant-- nonzeroes consist
		 * solely of a single nonzero coordinate, without any value.
		 */
		template<
			typename RowIndexT, typename ColIndexT,
			typename fwd_it1, typename fwd_it2,
			typename iterator_category
		>
		class SynchronizedNonzeroIterator<
			RowIndexT, ColIndexT, void,
			fwd_it1, fwd_it2, void,
			iterator_category
		> :
			public internal::SynchronizedIteratorBaseStorage<
				RowIndexT, ColIndexT, void,
				fwd_it1, fwd_it2, iterator_category
			>
		{

			private:

				/** The base type this class inherits from. */
				using BaseType = internal::SynchronizedIteratorBaseStorage<
					RowIndexT, ColIndexT, void,
					fwd_it1, fwd_it2, iterator_category
				>;


			public:

				/** ALP value typedef */
				typedef void ValueType;

				/** The type of this class for a short-hand. */
				using SelfType = SynchronizedNonzeroIterator<
					RowIndexT, ColIndexT, void,
					fwd_it1, fwd_it2, void,
					iterator_category
				>;

				/** Base constructor. Takes three sub-iterators as arguments. */
				SynchronizedNonzeroIterator(
					fwd_it1 it1, fwd_it2 it2,
					fwd_it1 it1_end, fwd_it2 it2_end
				) : BaseType( it1, it2, it1_end, it2_end ) {}

				/** Copy constructor. */
				SynchronizedNonzeroIterator( const SelfType &other ): BaseType( other ) {}

				/** Assignment operator. */
				SelfType & operator=( const SelfType &other ) {
					(void) BaseType::operator=( other );
					return *this;
				}

				/** Equality check. */
				bool operator==( const SelfType &other ) const {
					return BaseType::operator==( other );
				}

				/** Inequality check. */
				bool operator!=( const SelfType &other ) const {
					return BaseType::operator!=( other );
				};

				/** Increment operator. */
				SelfType & operator++() {
					(void) BaseType::operator++();
					return *this;
				}

				/**
				 * Offset operator.
				 *
				 * Enabled only for random access source iterators.
				 */
				template< typename cat = iterator_category > SelfType& operator+=(
					typename std::enable_if<
						std::is_same< cat, std::random_access_iterator_tag >::value &&
						std::is_same< typename SelfType::iterator_category,
								std::random_access_iterator_tag
							>::value,
						size_t
					>::type offset
				) {
					this->row_it += offset;
					this->col_it += offset;
					this->row_col_updated = false;
					return *this;
				}

				/**
				 * Difference operator.
				 *
				 * Enabled only for random access source iterators.
				*/
				template< typename _cat = iterator_category >
				typename SelfType::difference_type operator-(
					typename std::enable_if<
						std::is_same< _cat, std::random_access_iterator_tag >::value
							&& std::is_same< iterator_category,
							std::random_access_iterator_tag
						>::value,
						const SelfType&
					>::type other
				) const {
					return static_cast< typename SelfType::difference_type >(
						this->row_it - other.row_it );
				}

		};

#ifdef _DEBUG
		/** \internal Prints a SynchronizedNonzeroIterator to an output stream. */
		template<
			typename RowIndexT, typename ColIndexT, typename V,
			typename fwd_it1, typename fwd_it2, typename fwd_it3,
			typename iterator_category
		>
		std::ostream & operator<<(
			std::ostream &os,
			const SynchronizedNonzeroIterator<
				RowIndexT, ColIndexT, V,
				fwd_it1, fwd_it2, fwd_it3,
				iterator_category
			> &it
		) {
			os << it.i() << ", " << it.j() << ", " << it.v();
			return os;
		}

		/** \internal Prints a SynchronizedNonzeroIterator to an output stream. */
		template<
			typename RowIndexT, typename ColIndexT,
			typename fwd_it1, typename fwd_it2,
			typename iterator_category
		>
		std::ostream & operator<<(
			std::ostream &os,
			const SynchronizedNonzeroIterator<
				RowIndexT, ColIndexT, void,
				fwd_it1, fwd_it2, void,
				iterator_category
			> &it
		) {
			os << it.i() << ", " << it.j();
			return os;
		}
#endif

		/**
		 * Make a synchronized iterator out of three source pointers.
		 *
		 * The three pointers refer to row indices, column indices, and nonzero
		 * values, respectively.
		 *
		 * Both start and end pointers are required by this variant. The region thus
		 * indicated is assumed to be accessible in a random access fashion.
		 */
		template< typename RowIndexT, typename ColIndexT, typename V >
		SynchronizedNonzeroIterator<
			RowIndexT, ColIndexT, V,
			const RowIndexT *, const ColIndexT *, const V *,
			std::random_access_iterator_tag
		>
		makeSynchronized(
			const RowIndexT * const it1,
			const ColIndexT * const it2,
			const V  * const it3,
			const RowIndexT * const it1_end,
			const ColIndexT * const it2_end,
			const V  * const it3_end
		) {
			return SynchronizedNonzeroIterator<
				RowIndexT, ColIndexT, V,
				const RowIndexT *, const ColIndexT *, const V *,
				std::random_access_iterator_tag
			>(
				it1, it2, it3,
				it1_end, it2_end, it3_end
			);
		}

		/**
		 * Make a synchronized iterator out of three source pointers.
		 *
		 * The three pointers refer to row indices, column indices, and nonzero
		 * values, respectively.
		 *
		 * The start pointers are augmented with a \a length argument that indicates
		 * the size of each memory region pointed to. The region thus indicated is
		 * assumed to be accessible in a random access fasion.
		 */
		template< typename RowT, typename ColT, typename V >
		SynchronizedNonzeroIterator<
			RowT, ColT, V,
			const RowT *, const ColT *, const V *,
			std::random_access_iterator_tag
		>
		makeSynchronized(
			const RowT * const it1,
			const ColT * const it2,
			const V  * const it3,
			const size_t length
		) {
			return makeSynchronized(
				it1, it2, it3,
				it1 + length, it2 + length, it3 + length
			);
		}

		/**
		 * Make a synchronized iterator out of two source pointers.
		 *
		 * The two pointers refer to row indices, column indices, respectively.
		 *
		 * Both start and end pointers are required by this variant. The region thus
		 * indicated is assumed to be accessible in a random access fashion.
		 */
		template<
			typename RowIndexT,
			typename ColIndexT
		>
		SynchronizedNonzeroIterator<
			RowIndexT, ColIndexT, void,
			const RowIndexT *, const ColIndexT *, void,
			std::random_access_iterator_tag
		>
		makeSynchronized(
			const RowIndexT * const it1, const ColIndexT * const it2,
			const RowIndexT * const it1_end, const ColIndexT * const it2_end
		) {
#ifdef _DEBUG
			std::cout << "SynchronizedNonzeroIterator::makeSynchronized received "
				<< "iterators " << it1 << " (start) and " << it2 << " (end)\n";
#endif
			return SynchronizedNonzeroIterator<
				RowIndexT, ColIndexT, void,
				const RowIndexT *, const ColIndexT *, void,
				std::random_access_iterator_tag
			>( it1, it2, it1_end, it2_end );
		}

		/**
		 * Make a synchronized iterator out of two source pointers.
		 *
		 * The two pointers refer to row indices, column indices, respectively.
		 *
		 * The start pointers are augmented with a \a length argument that indicates
		 * the size of each memory region pointed to. The region thus indicated is
		 * assumed to be accessible in a random access fasion.
		 */
		template<
			typename RowT,
			typename ColT
		>
		SynchronizedNonzeroIterator<
			RowT, ColT, void,
			const RowT *, const ColT *, void,
			std::random_access_iterator_tag
		>
		makeSynchronized(
			const RowT * const it1, const ColT * const it2,
			const size_t length
		) {
			return SynchronizedNonzeroIterator<
				RowT, ColT, void,
				const RowT *, const ColT *, void,
				std::random_access_iterator_tag
			>( it1, it2, it1 + length, it2 + length );
		}

		/**
		 * Creates a synchronized iterator out of two source iterators.
		 *
		 * The resulting iterator has the ``weakest'' tag of the two source iterators.
		 */
		template<
			typename fwd_it1,
			typename fwd_it2
		>
		SynchronizedNonzeroIterator<
			typename std::iterator_traits< fwd_it1 >::value_type,
			typename std::iterator_traits< fwd_it2 >::value_type,
			void, fwd_it1, fwd_it2, void,
			typename utils::common_iterator_tag< fwd_it1, fwd_it2 >::iterator_category
		> makeSynchronized(
			const fwd_it1 it1, const fwd_it2 it2,
			const fwd_it1 it1_end, const fwd_it2 it2_end
		) {
			return SynchronizedNonzeroIterator<
				typename std::iterator_traits< fwd_it1 >::value_type,
				typename std::iterator_traits< fwd_it2 >::value_type,
				void,
				fwd_it1, fwd_it2, void,
				typename utils::common_iterator_tag<
					fwd_it1, fwd_it2
				>::iterator_category
			>( it1, it2, it1_end, it2_end );
		}

		/**
		 * Creates a synchronized iterator out of three source iterators.
		 *
		 * The resulting iterator has the ``weakest'' tag of the three source
		 * iterators.
		 */
		template<
			typename fwd_it1, typename fwd_it2, typename fwd_it3
		>
		SynchronizedNonzeroIterator<
			typename std::iterator_traits< fwd_it1 >::value_type,
			typename std::iterator_traits< fwd_it2 >::value_type,
			typename std::iterator_traits< fwd_it3 >::value_type,
			fwd_it1, fwd_it2, fwd_it3,
			typename utils::common_iterator_tag<
				fwd_it1, fwd_it2, fwd_it3
			>::iterator_category
		>
		makeSynchronized(
			const fwd_it1 it1, const fwd_it2 it2, const fwd_it3 it3,
			const fwd_it1 it1_end, const fwd_it2 it2_end, const fwd_it3 it3_end
		) {
			return SynchronizedNonzeroIterator<
				typename std::iterator_traits< fwd_it1 >::value_type,
				typename std::iterator_traits< fwd_it2 >::value_type,
				typename std::iterator_traits< fwd_it3 >::value_type,
				fwd_it1, fwd_it2, fwd_it3,
				typename utils::common_iterator_tag<
					fwd_it1, fwd_it2, fwd_it3
				>::iterator_category
			>( it1, it2, it3, it1_end, it2_end, it3_end );
		}

	} // end namespace grb::internal

} // end namespace grb

#endif // end ``_H_SYNCHRONIZEDNONZEROITERATOR''

