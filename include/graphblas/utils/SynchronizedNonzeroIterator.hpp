
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

#include "NonzeroStorage.hpp"
#include "iterators/type_traits.hpp"

#ifdef _DEBUG
 #ifndef _GRB_NO_STDIO
  #include <iostream>
 #endif
#endif


namespace grb {

	namespace utils {

		namespace internal {

			// base class for storage, where V can be void
			template<
				typename RowIndexT, typename ColIndexT, typename V,
				typename fwd_it1, typename fwd_it2,
				typename _iterator_category
			>
			class SynchronizedIteratorBaseStorage {

				protected:

					// iterators to synchronise:
					fwd_it1 row_it, row_end;
					fwd_it2 col_it, col_end;

					using SelfType = SynchronizedIteratorBaseStorage<
						RowIndexT, ColIndexT, V,
						fwd_it1, fwd_it2, _iterator_category
					>;

					using StorageType = NonzeroStorage< RowIndexT, ColIndexT, V >;

					mutable bool row_col_updated;
					mutable StorageType nonzero;

					/** Base constructor. Takes three sub-iterators as arguments. */
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
					bool operator!=( const SelfType &other ) const {
						return row_it != other.row_it || col_it != other.col_it;
					};

					/** Equality check. */
					bool operator==( const SelfType &other ) const {
						return ! operator!=( other );
					}

					/** Increment operator. */
					SelfType& operator++() {
						(void) ++row_it;
						(void) ++col_it;
						row_col_updated = false;
						return *this;
					}


				public:

					// STL iterator's typedefs:
					typedef _iterator_category iterator_category;
					typedef typename StorageType::StorageType value_type;
					typedef size_t difference_type;
					typedef const value_type& reference;
					typedef const value_type * pointer;

					// GraphBLAS typedefs:
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
					const RowIndexType& i() const {
						row_col_update_if_needed();
						return nonzero.i();
					}

					/** Returns the column coordinate. */
					const ColumnIndexType& j() const {
						row_col_update_if_needed();
						return nonzero.j();
					}

			};

			template<
				typename RowIndexT, typename ColIndexT, typename V,
				typename fwd_it1, typename fwd_it2, typename fwd_it3,
				typename _iterator_category
			>
			class SynchronizedIteratorBase :
				public SynchronizedIteratorBaseStorage<
					RowIndexT, ColIndexT, V,
					fwd_it1, fwd_it2,
					_iterator_category
				>
			{

				private:

					using BaseType = SynchronizedIteratorBaseStorage<
						RowIndexT, ColIndexT, V,
						fwd_it1, fwd_it2,
						_iterator_category
					>;


				protected:

					// iterators to synchronise:
					fwd_it3 val_it, val_end;
					mutable bool val_updated;

					using SelfType = SynchronizedIteratorBase<
						RowIndexT, ColIndexT, V,
						fwd_it1, fwd_it2, fwd_it3,
						_iterator_category
					>;

					/** Base constructor. Takes three sub-iterators as arguments. */
					SynchronizedIteratorBase(
						fwd_it1 it1, fwd_it2 it2, fwd_it3 it3,
						fwd_it1 it1_end, fwd_it2 it2_end, fwd_it3 it3_end
					) :
						BaseType(
							it1, it2,
							it1_end, it2_end
						), val_it( it3 ), val_end( it3_end ), val_updated( false )
					{}

					/** Copy constructor. */
					SynchronizedIteratorBase( const SelfType &other ) :
						BaseType( other ),
						val_it( other.val_it ), val_end( other.val_end ),
						val_updated( other.val_updated )
					{}

					/** Assignment operator. */
					SelfType& operator=( const SelfType &other ) {
						(void) BaseType::operator=( other );
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
					bool operator==( const SelfType &other ) const {
						return BaseType::operator==( other ) && val_it == other.val_it;
					}

					/** Inequality check. */
					bool operator!=( const SelfType & other ) const {
						return BaseType::operator!=( other ) || val_it != other.val_it;
					};

					/** Increment operator. */
					SelfType& operator++() {
						(void) BaseType::operator++();
						(void) ++val_it;
						val_updated = false;
						return *this;
					}


				public:

					// GraphBLAS typedefs:
					typedef V ValueType;

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

		} // end namespace grb::utils::internal

		// for value matrices
		template<
			typename RowIndexT, typename ColIndexT, typename V,
			typename fwd_it1, typename fwd_it2, typename fwd_it3,
			class _iterator_category
		>
		class SynchronizedNonzeroIterator :
			public internal::SynchronizedIteratorBase<
				RowIndexT, ColIndexT, V,
				fwd_it1, fwd_it2, fwd_it3,
				_iterator_category
			>
		{

			private:

				using BaseType = internal::SynchronizedIteratorBase<
					RowIndexT, ColIndexT, V,
					fwd_it1, fwd_it2, fwd_it3,
					_iterator_category
				>;


			public:

				using SelfType = SynchronizedNonzeroIterator<
					RowIndexT, ColIndexT, V,
					fwd_it1, fwd_it2, fwd_it3,
					_iterator_category
				>;

				/** Base constructor. Takes three sub-iterators as arguments. */
				SynchronizedNonzeroIterator(
					fwd_it1 it1, fwd_it2 it2, fwd_it3 it3,
					fwd_it1 it1_end, fwd_it2 it2_end, fwd_it3 it3_end
				) : BaseType(
					it1, it2, it3,
					it1_end, it2_end, it3_end
				) {}

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

				/** Offset operator, enabled only for random access iterators */
				template< typename cat = _iterator_category > SelfType & operator+=(
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

				/* Difference operator, enabled only for random access iterators */
				template< typename _cat = _iterator_category >
					typename SelfType::difference_type operator-(
					typename std::enable_if<
						std::is_same< _cat, std::random_access_iterator_tag >::value
							&& std::is_same< _iterator_category,
							std::random_access_iterator_tag
						>::value,
						const SelfType&
					>::type other
				) const {
					return static_cast< typename SelfType::difference_type >(
						this->row_it - other.row_it );
				}
		};

		// for pattern matrices
		template<
			typename RowIndexT, typename ColIndexT,
			typename fwd_it1, typename fwd_it2,
			class _iterator_category
		>
		class SynchronizedNonzeroIterator<
			RowIndexT, ColIndexT, void,
			fwd_it1, fwd_it2, void,
			_iterator_category
		> :
			public internal::SynchronizedIteratorBaseStorage<
				RowIndexT, ColIndexT, void,
				fwd_it1, fwd_it2, _iterator_category
			>
		{

			private:

				using BaseType = internal::SynchronizedIteratorBaseStorage<
					RowIndexT, ColIndexT, void,
					fwd_it1, fwd_it2, _iterator_category
				>;


			public:

				using SelfType = SynchronizedNonzeroIterator<
					RowIndexT, ColIndexT, void,
					fwd_it1, fwd_it2, void,
					_iterator_category
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

				/** Offset operator, enabled only for random access iterators */
				template< typename cat = _iterator_category > SelfType& operator+=(
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

				/* Difference operator, enabled only for random access iterators */
				template< typename _cat = _iterator_category >
				typename SelfType::difference_type operator-(
					typename std::enable_if<
						std::is_same< _cat, std::random_access_iterator_tag >::value
							&& std::is_same< _iterator_category,
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

		// overload for 3 pointers and custom tag
		template<
			typename RowIndexT, typename ColIndexT, typename V,
			typename iterator_category
		>
		SynchronizedNonzeroIterator<
			RowIndexT, ColIndexT, V,
			const RowIndexT *, const ColIndexT *, const V *,
			iterator_category
		>
		makeSynchronized(
			const RowIndexT * const it1,
			const ColIndexT * const it2,
			const V  * const it3,
			const RowIndexT * const it1_end,
			const ColIndexT * const it2_end,
			const V  * const it3_end,
			iterator_category
		) {
			return SynchronizedNonzeroIterator<
				RowIndexT, ColIndexT, V,
				const RowIndexT *, const ColIndexT *, const V *,
				iterator_category
			>(
				it1, it2, it3,
				it1_end, it2_end, it3_end
			);
		}

		// overload for 3 pointers without tag: get the best (random access iterator)
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
			return makeSynchronized(
				it1, it2, it3,
				it1_end, it2_end, it3_end,
				std::random_access_iterator_tag()
			);
		}

		// overload for 3 pointers, number and with no tag: get the best (random access iterator)
		template< typename RowT, typename ColT, typename V >
		SynchronizedNonzeroIterator<
			RowT, ColT, V,
			const RowT *, const ColT *, const V *,
			std::random_access_iterator_tag
		>
		makeSynchronized(
			const RowT * const it1, const ColT * const it2, const V  * const it3,
			size_t length
		) {
			return makeSynchronized(
				it1, it2, it3,
				it1 + length, it2 + length, it3 + length,
				std::random_access_iterator_tag()
			);
		}


		// overload for 2 pointers and custom tag
		template<
			typename RowIndexT,
			typename ColIndexT,
			typename iterator_category
		>
		SynchronizedNonzeroIterator<
			RowIndexT, ColIndexT, void,
			const RowIndexT *, const ColIndexT *, void, iterator_category
		>
		makeSynchronized(
			const RowIndexT * const it1, const ColIndexT * const it2,
			const RowIndexT * const it1_end, const ColIndexT * const it2_end,
			iterator_category
		) {
			return SynchronizedNonzeroIterator<
				RowIndexT, ColIndexT, void,
				const RowIndexT *, const ColIndexT *, void,
				iterator_category
			>( it1, it2, it1_end, it2_end );
		}


		// overload for 2 pointers with no tag: get the best (random access iterator)
		template<
			typename RowIndexT,
			typename ColIndexT
		>
		SynchronizedNonzeroIterator<
			RowIndexT, ColIndexT, void,
			const RowIndexT *, const ColIndexT *, void, std::random_access_iterator_tag
		>
		makeSynchronized(
			const RowIndexT * const it1, const ColIndexT * const it2,
			const RowIndexT * const it1_end, const ColIndexT * const it2_end
		) {
#ifdef _DEBUG
			std::cout << "SynchronizedNonzeroIterator::makeSynchronized received "
				<< "iterators " << it1 << " (start) and " << it2 << " (end)\n";
#endif
			return makeSynchronized(
				it1, it2, it1_end, it2_end,
				std::random_access_iterator_tag()
			);
		}

		// overload for 2 iterators with custom tag
		template<
			typename fwd_it1,
			typename fwd_it2,
			typename iterator_category
		> SynchronizedNonzeroIterator<
			typename std::iterator_traits< fwd_it1 >::value_type,
			typename std::iterator_traits< fwd_it2 >::value_type,
			void, fwd_it1, fwd_it2, void, iterator_category
		>
		makeSynchronized(
			const fwd_it1 it1, const fwd_it2 it2,
			const fwd_it1 it1_end, const fwd_it2 it2_end,
			iterator_category
		) {
			return SynchronizedNonzeroIterator<
				typename std::iterator_traits< fwd_it1 >::value_type,
				typename std::iterator_traits< fwd_it2 >::value_type,
				void, fwd_it1, fwd_it2, void, iterator_category
			>( it1, it2, it1_end, it2_end );
		}

		// overload for 2 pointers, number and with no tag: get the best (random access iterator)
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
			const RowT * const it1, const ColT * const it2, size_t length
		) {
			return makeSynchronized( it1, it2, it1 + length, it2 + length,
				std::random_access_iterator_tag() );
		}

		// overload for 2 iterators without tag: get common tag
		template<
			typename fwd_it1,
			typename fwd_it2
		>
		SynchronizedNonzeroIterator<
			typename std::iterator_traits< fwd_it1 >::value_type,
			typename std::iterator_traits< fwd_it2 >::value_type,
			void, fwd_it1, fwd_it2, void,
			typename common_iterator_tag< fwd_it1, fwd_it2 >::iterator_category
		> makeSynchronized(
			const fwd_it1 it1, const fwd_it2 it2,
			const fwd_it1 it1_end, const fwd_it2 it2_end
		) {
			using cat = typename common_iterator_tag<
				fwd_it1, fwd_it2
			>::iterator_category;
			return makeSynchronized( it1, it2, it1_end, it2_end, cat() );
		}

		// overload for 3 iterators with custom tag
		template<
			typename fwd_it1,
			typename fwd_it2,
			typename fwd_it3,
			typename iterator_category
		>
		SynchronizedNonzeroIterator<
			typename std::iterator_traits< fwd_it1 >::value_type,
			typename std::iterator_traits< fwd_it2 >::value_type,
			typename std::iterator_traits< fwd_it3 >::value_type,
			fwd_it1, fwd_it2, fwd_it3, iterator_category
		>
		makeSynchronized(
			const fwd_it1 it1, const fwd_it2 it2, const fwd_it3 it3,
			const fwd_it1 it1_end, const fwd_it2 it2_end, const fwd_it3 it3_end,
			iterator_category
		) {
			return SynchronizedNonzeroIterator<
				typename std::iterator_traits< fwd_it1 >::value_type,
				typename std::iterator_traits< fwd_it2 >::value_type,
				typename std::iterator_traits< fwd_it3 >::value_type,
				fwd_it1, fwd_it2, fwd_it3, iterator_category
			>( it1, it2, it3, it1_end, it2_end, it3_end );
		}

		// overload for 3 iterators without tag: get common tag
		template<
			typename fwd_it1,
			typename fwd_it2,
			typename fwd_it3
		>
		SynchronizedNonzeroIterator<
			typename std::iterator_traits< fwd_it1 >::value_type,
			typename std::iterator_traits< fwd_it2 >::value_type,
			typename std::iterator_traits< fwd_it3 >::value_type,
			fwd_it1, fwd_it2, fwd_it3,
			typename common_iterator_tag< fwd_it1, fwd_it2, fwd_it3 >::iterator_category
		> makeSynchronized(
			const fwd_it1 it1, const fwd_it2 it2, const fwd_it3 it3,
			const fwd_it1 it1_end, const fwd_it2 it2_end, const fwd_it3 it3_end
		) {
			using cat = typename common_iterator_tag<
				fwd_it1, fwd_it2, fwd_it3
			>::iterator_category;
			return makeSynchronized( it1, it2, it3, it1_end, it2_end, it3_end, cat() );
		}

	} // namespace utils

} // namespace grb

#endif // end ``_H_SYNCHRONIZEDNONZEROITERATOR''

