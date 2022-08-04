
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
 * @brief Utilities to iterate over nonzeroes to sort a CRS/CCS data structure in-place.
 *  For internal use only.
 *
 * @author Alberto Scolari
 * @date 16/06/2022
 */

#ifndef _H_GRB_NONZERO_WRAPPER
#define _H_GRB_NONZERO_WRAPPER

#include <iostream>
#include <type_traits>
#include <cstddef>
#include <iterator>

#include "compressed_storage.hpp"


namespace grb {

	namespace internal {

		// Forward declarations:

		/**
		 * Wrapper for CRS/CCS with a row/column buffer (not in the CRS/CCS)
		 * which allows iterating over the nonzeroes in order to sort them.
		 *
		 * The iteration over the nonzeroes is achieved by internally coordinating
		 * the access to the CRS/CCS and to the coordinated rows/columns buffer,
		 * so that accessing the iterator as \a *it returns an {i,j,v}() triple
		 * with the pointed nonzero values.
		 *
		 * This class is designed to work with std::sort() and has a custom storage
		 * type and a custom reference/pointer type.
		 */
		template<
			typename ValType,
			typename RowIndexType,
			typename NonzeroIndexType,
			typename ColIndexType
		>
		struct NZIterator;

		/**
		 * Wrapper class provide a {row,col,val}() interface over a nonzero.
		 *
		 * It internally points directly to the data inside an underlying CRS/CCS
		 * and row/column buffer, and defines an order based on nonzero coordinates.
		 *
		 * Use only in conjunction with NZIterator.
		 */
		template<
			typename ValType,
			typename RowIndexType,
			typename NonzeroIndexType,
			typename ColIndexType
		>
		class NZWrapper;

		/**
		 * Stores a nonzero and provides a row,col,val}() interface.
		 *
		 * The val function is only defined when ValType is non-void.
		 */
		template<
			typename ValType,
			typename RowIndexType,
			typename ColIndexType
		>
		class NZStorage;

		/**
		 * Specialized swap function for NZWrapper.
		 *
		 * Called within std::sort().
		 */
		template<
			typename ValType,
			typename RowIndexType,
			typename NonzeroIndexType,
			typename ColIndexType
		> void swap(
			NZWrapper< ValType, RowIndexType, NonzeroIndexType, ColIndexType > &a,
			NZWrapper< ValType, RowIndexType, NonzeroIndexType, ColIndexType > &b
		);


		// Implementations:

		template<
			typename ValType,
			typename RowIndexType,
			typename NonzeroIndexType,
			typename ColIndexType
		>
		class NZWrapper {

			friend class NZIterator<
				ValType, RowIndexType, NonzeroIndexType, ColIndexType
			>;
			friend class NZStorage< ValType, RowIndexType, ColIndexType >;
			friend void swap< ValType, RowIndexType, NonzeroIndexType, ColIndexType >(
				NZWrapper< ValType, RowIndexType, NonzeroIndexType, ColIndexType > &,
				NZWrapper< ValType, RowIndexType, NonzeroIndexType, ColIndexType > &
			);


			private:

				/** Short-hand for our own type. */
				using SelfType =
					NZWrapper< ValType, RowIndexType, NonzeroIndexType, ColIndexType >;

				/** The underlying compressed storage. */
				Compressed_Storage< ValType, RowIndexType, NonzeroIndexType > * _CXX;

				/** The underlying buffer. */
				ColIndexType *_col_values_buffer;

				/** The index in the #_CXX storage that this nonzero refers to. */
				size_t _off;


			public:

				NZWrapper() = delete;

				/** Base constructor. */
				NZWrapper(
					Compressed_Storage< ValType, RowIndexType, NonzeroIndexType > &CXX,
					ColIndexType *col_values_buffer,
					size_t off
				) noexcept :
					_CXX( &CXX ),
					_col_values_buffer( col_values_buffer ),
					_off( off)
				{}

				NZWrapper( const SelfType & ) = delete;

				/** Move constructor */
				NZWrapper( SelfType && ) = default;

				SelfType & operator=( const SelfType & ) = delete;

				/**
				 * @returns The column index.
				 */
				inline ColIndexType & col() {
					return this->_col_values_buffer[ this->_off ];
				}

				/**
				 * @returns The column index.
				 *
				 * Const-variant.
				 */
				inline const ColIndexType & col() const {
					return this->_col_values_buffer[ this->_off ];
				}

				/** @returns The row index. */
				inline RowIndexType & row() { return this->_CXX->row_index[ this->_off ]; }

				/**
				 * @returns The row index.
				 *
				 * Const-variant.
				 */
				inline const RowIndexType & row() const {
					return this->_CXX->row_index[ this->_off ];
				}

				/** @returns The nonzero index. */
				inline size_t & off() { return this->_off; }

				/**
				 * @returns The nonzero index.
				 *
				 * Const-variant.
				 */
				inline const size_t & off() const { return this->_off; }

				/** Move assignment from another wrapper. */
				SelfType & operator=( SelfType &&other ) noexcept {
#ifdef _DEBUG
					std::cout << "transfer ";
					print( std::cout, *this);
					std::cout << " <- ";
					print( std::cout, other );
					std::cout << std::endl;
#endif
					this->col() = other.col();
					this->row() = other.row();
					this->write_value( other );
					return *this;
				}

				/**
				 * Move assignment from a nonzero storage.
				 *
				 * Does not invalidate the source nonzero storage.
				 */
				SelfType & operator=(
					NZStorage< ValType, RowIndexType, ColIndexType > &&storage
				) noexcept {
#ifdef _DEBUG
					std::cout << "copying into wrapper ";
					print( std::cout, *this );
					std::cout << " <- ";
					NZStorage< ValType, RowIndexType, ColIndexType >::print(
						std::cout, storage );
					std::cout << std::endl;
#endif
					storage.copyTo( *this );
					return *this;
				}

				/**
				 * Comparator based on nonzero coordinates.
				 */
				bool operator<( const SelfType &other ) const noexcept {
					const bool result = ( this->col() < other.col() ) || (
							this->col() == other.col() && this->row() >= other.row() // reverse order
						);
#ifdef _DEBUG
					std::cout << "compare:: ";
					print( std::cout, *this );
					std::cout << " < ";
					print( std::cout, other );
					std::cout << ( result ? " true" : " false" ) << std::endl;
#endif
					return result;
				}

#ifdef _DEBUG
				/** Overload print function. */
				template< typename T = ValType >
				static void print( std::ostream &s, const SelfType &nz,
					typename std::enable_if<
						!(std::is_same< T, void >::value)
					>::type * = nullptr
				) {
					s << nz.off() << ": [ " << nz.col() << ", "
						<< nz.row() << "]: "
						<< nz.val();
				}

				/** Overload print function for pattern nonzeroes. */
				template< typename T = ValType >
				static void print( std::ostream &s, const SelfType &nz,
					typename std::enable_if<
						std::is_same< T, void >::value
					>::type * = nullptr
				) {
					s << nz.off() << ": [ " << nz.col() << ", "
						<< nz.row() << "]";
				}
#endif


			private:

				/** Swaps two entries. */
				void swap( SelfType &other ) noexcept {
					std::swap( this->col(), other.col() );
					std::swap( this->row(), other.row() );
					this->swap_value( other );
				}

				/** Swaps non-void values. */
				template< typename T >
				inline void swap_value(
					NZWrapper< T, RowIndexType, NonzeroIndexType, ColIndexType > &other,
					typename std::enable_if<
						!std::is_same< T, void >::value
					>::type * = nullptr
				) noexcept {
					std::swap( this->_CXX->values[ this->_off ],
						other._CXX->values[ other._off ] );
				}

				/** No-op in the case of void values. */
				template< typename T >
				inline void swap_value(
					NZWrapper< T, RowIndexType, NonzeroIndexType, ColIndexType > &other,
					typename std::enable_if< std::is_same< T, void >::value >::type * = nullptr
				) noexcept {
					(void) other;
				}

				/** Writes a value of a given nonzero into this nonzero. */
				template< typename T >
				inline void write_value(
					NZWrapper< T, RowIndexType, NonzeroIndexType, ColIndexType > &other,
					typename std::enable_if<
						!std::is_same< T, void >::value
					>::type * = nullptr
				) noexcept {
					this->_CXX->values[ this->_off ] = other._CXX->values[ other._off ];
				}

				/** No-op in the case of void values. */
				template< typename T >
				inline void write_value(
					NZWrapper< T, RowIndexType, NonzeroIndexType, ColIndexType > &other,
					typename std::enable_if< std::is_same< T, void >::value >::type * = nullptr
				) noexcept {
					(void) other;
				}

				/** Returns the value of this nonzero. */
				template< typename T = ValType >
				typename std::enable_if<
					!(std::is_same< T, void >::value) &&
						std::is_same< T, ValType >::value,
					ValType
				>::type & val(
					typename std::enable_if<
						!(std::is_same< T, void >::value) &&
							std::is_same< T,
						ValType
					>::value >::type * = nullptr
				) {
					return this->_CXX->values[ this->_off ];
				}

				/**
				 * Returns the value of this nonzero.
				 *
				 * Const-variant.
				 */
				template< typename T = ValType >
				const typename std::enable_if<
					!(std::is_same< T, void >::value) &&
						std::is_same< T, ValType >::value,
					ValType
				>::type & val(
					typename std::enable_if<
						!(std::is_same< T, void >::value) &&
							std::is_same< T, ValType >::value
					>::type * = nullptr
				) const {
					return this->_CXX->values[ this->_off ];
				}

		};

		template<
			typename ValType,
			typename RowIndexType,
			typename NonzeroIndexType,
			typename ColIndexType
		> void swap(
			NZWrapper< ValType, RowIndexType, NonzeroIndexType, ColIndexType > &a,
			NZWrapper< ValType, RowIndexType, NonzeroIndexType, ColIndexType > &b
		) {
#ifdef _DEBUG
			std::cout << "calling swap" << std::endl;
#endif
			a.swap( b );
		}

		/** Stores a nonzero coordinate. */
		template<
			typename RowIndexType,
			typename ColIndexType
		>
		class NZStorageBase {

			private:

				/** Short-hand to our own type. */
				using SelfType = NZStorageBase< RowIndexType, ColIndexType >;

				/** Column index. */
				ColIndexType _col;

				/** Row index. */
				RowIndexType _row;


			public:

				NZStorageBase() = delete;

				SelfType operator=( const SelfType & ) = delete;

				/**
				 * Constructor from a nonzero wrapper.
				 *
				 * Copies into the storage from the CRS/CCS underlying the wrapper.
				 */
				template< typename V, typename NonzeroIndexType >
				NZStorageBase(
					const NZWrapper< V, RowIndexType, NonzeroIndexType, ColIndexType > &orig
				) noexcept : _col( orig.col() ), _row( orig.row() ) {}

				/** @returns The column index. */
				inline ColIndexType & col() { return this->_col; }

				/**
				 * @returns The column index.
				 *
				 * This is the const-variant.
				 */
				inline const ColIndexType & col() const { return this->_col; }

				/** @returns The row index. */
				inline RowIndexType & row() { return this->_row; }

				/**
				 * @returns The row index.
				 *
				 * This is the const-variant.
				 */
				inline const RowIndexType & row() const { return this->_row; }

				/**
				 * Move-assignment from a wrapper.
				 *
				 * \internal Does not invalidate the source temporary \a orig.
				 */
				template< typename V, typename NonzeroIndexType >
				SelfType operator=(
					NZWrapper< V, RowIndexType, NonzeroIndexType, ColIndexType > &&orig
				) noexcept {
					this->_col = orig.col();
					this->_row = orig.row();
					return *this;
				}

				/**
				 * Copies the current nonzero coordinates into a destination nonzero.
				 */
				template< typename V, typename NonzeroIndexType >
				void copyTo(
					NZWrapper< V, RowIndexType, NonzeroIndexType, ColIndexType > &dest
				) noexcept {
					dest.col() = this->_col;
					dest.row() = this->_row;
				}

		};

		/**
		 * Stores a nonzero with {row,col,val}() interface.
		 *
		 * Includes comparison operators as well as copy/move logic to and from
		 * #NZWrapper.
		 *
		 * Used within NZIterator to store nonzeroes as a local caches, within sorting
		 * algorithms like insertion sort that extracts nonzeroes as
		 * <em> CacheT c = *iterator <\em>.
		 *
		 * This is the non-void (non-pattern) variant.
		 */
		template<
			typename ValType,
			typename RowIndexType,
			typename ColIndexType
		>
		class NZStorage : public NZStorageBase< RowIndexType, ColIndexType > {

			private:

				/** Short-hand for our own type. */
				using SelfType = NZStorage< ValType, RowIndexType, ColIndexType >;

				/** Short-hand for our base type. */
				using BaseType = NZStorageBase< RowIndexType, ColIndexType >;

				/** Nonzero value. */
				ValType _val;


			public:

				NZStorage() = delete;

				/**
				 * Base constructor from a nonzero wrapper.
				 */
				template< typename NonzeroIndexType >
				NZStorage(
					const NZWrapper<
						ValType, RowIndexType, NonzeroIndexType, ColIndexType
					> &orig
				) noexcept :
					BaseType( orig ),
					_val( orig.val() )
				{
#ifdef _DEBUG
					std::cout << "create storage ";
					print( std::cout, *this );
					std::cout << std::endl;
#endif
				}

				/** @returns The nonzero value. */
				ValType& val() { return this->_val; }

				/**
				 * @returns The nonzero value.
				 *
				 * This is the const variant.
				 */
				const ValType & val() const { return this->_val; }

				SelfType operator=( const SelfType & ) = delete;

				/**
				 * Move-assignment from a NZWrapper instance.
				 *
				 * This does not invalidate the source wrapper.
				 */
				template< typename NonzeroIndexType > SelfType operator=(
					NZWrapper< ValType, RowIndexType, NonzeroIndexType, ColIndexType > &&orig
				) noexcept {
#ifdef _DEBUG
					std::cout << "moving into storage ";
					NZWrapper< ValType, RowIndexType, NonzeroIndexType, ColIndexType >::print(
						std::cout, orig );
					std::cout << std::endl;
#endif
					(void) this->BaseType::operator=( orig );
					this->_val = orig.val();
					return *this;
				}

				/**
				 * Copies nonzero into a CRS/CCS underlying a given wrapper, at the location
				 * the wrapper points to.
				 */
				template< typename NonzeroIndexType >
				void copyTo(
					NZWrapper< ValType, RowIndexType, NonzeroIndexType, ColIndexType > &dest
				) noexcept {
					this->BaseType::copyTo( dest );
					dest.val() = this->_val;
				}
#ifdef _DEBUG
				/** Prints an instance to an output stream. */
				static void print( std::ostream &s, const SelfType &nz ) {
					s << "( " << nz.col() << ", " << nz.row() << " )" << ": " << nz.val();
				}
#endif

		};

		/**
		 * This is the void / pattern matrix variant of the above class.
		 *
		 * \internal This is a simplified re-hash of the above and therefore not
		 *           documented in detail.
		 */
		template<
			typename RowIndexType,
			typename ColIndexType
		>
		class NZStorage< void, RowIndexType, ColIndexType > :
			public NZStorageBase< RowIndexType, ColIndexType >
		{

			private:

				using SelfType = NZStorage< void, RowIndexType, ColIndexType >;
				using BaseType = NZStorageBase< RowIndexType, ColIndexType >;


			public:

				NZStorage() = delete;

				SelfType operator=( const SelfType & ) = delete;

				template< typename NonzeroIndexType >
				NZStorage(
					const NZWrapper< void, RowIndexType, NonzeroIndexType, ColIndexType > &orig
				) noexcept : BaseType( orig ) {
#ifdef _DEBUG
					std::cout << "create storage ";
					print( std::cout, *this );
					std::cout << std::endl;
#endif
				}

				template< typename NonzeroIndexType >
				SelfType operator=(
					NZWrapper< void, RowIndexType, NonzeroIndexType, ColIndexType > &&orig
				) noexcept {
#ifdef _DEBUG
					std::cout << "moving into storage ";
					NZWrapper< void, RowIndexType, NonzeroIndexType, ColIndexType >::print(
						std::cout, orig
					);
					std::cout << std::endl;
#endif
					(void) this->BaseType::operator=( orig );
					return *this;
				}

#ifdef _DEBUG
				static void print( std::ostream &s, const SelfType &nz ) {
					s << "( " << nz.col() << ", " << nz.row() << " )";
				}
#endif

		};

		/**
		 * Comparison operator between an NZStorage-object \a a and an
		 * NZWrapper-object \a b. Invoked during std::sort().
		 */
		template<
			typename ValType,
			typename RowIndexType,
			typename NonzeroIndexType,
			typename ColIndexType
		>
		bool operator<(
			const NZStorage< ValType, RowIndexType, ColIndexType > &a,
			const NZWrapper< ValType, RowIndexType, NonzeroIndexType, ColIndexType > &b
		) {
			const bool result = (a.col() < b.col()) ||
				(a.col() == b.col() && a.row() >= b.row());

#ifdef _DEBUG
			std::cout << "compare:: ";
			NZStorage< ValType, RowIndexType, ColIndexType >::print( std::cout, a );
			std::cout << " < ";
			NZWrapper< ValType, RowIndexType, NonzeroIndexType, ColIndexType >::print(
				std::cout, b );
			std::cout << ( result ? " true" : " false" ) << std::endl;
#endif
			return result;
		}

		/**
		 * Comparison operator between an NZWrapper-object \a a and an
		 * NZStorage-object \a b. Invoked during std::sort().
		 */
		template<
			typename ValType,
			typename RowIndexType,
			typename NonzeroIndexType,
			typename ColIndexType
		> bool operator<(
			const NZWrapper< ValType, RowIndexType, NonzeroIndexType, ColIndexType > &a,
			const NZStorage< ValType, RowIndexType, ColIndexType > &b
		) {
			const bool result = (a.col() < b.col()) ||
				(a.col() == b.col() && a.row() >= b.row());
#ifdef _DEBUG
			std::cout << "compare:: ";
			NZWrapper< ValType, RowIndexType, NonzeroIndexType, ColIndexType >::print(
				std::cout, a );
			std::cout << " < ";
			NZStorage< ValType, RowIndexType, ColIndexType >::print( std::cout, b );
			std::cout << ( result ? " true" : " false" ) << std::endl;
#endif
			return result;
		}

		// The main NZIterator class implementation follows.
		template<
			typename ValType,
			typename RowIndexType,
			typename NonzeroIndexType,
			typename ColIndexType
		>
		class NZIterator {

			public:

				// STL iterator fields

				using iterator_category = std::random_access_iterator_tag;
				using value_type = NZStorage< ValType, RowIndexType, ColIndexType >;
				using difference_type = long;
				using pointer = NZWrapper< ValType, RowIndexType, NonzeroIndexType, ColIndexType >*;
				using reference = NZWrapper< ValType, RowIndexType, NonzeroIndexType, ColIndexType >&;


			private:

				using SelfType = NZIterator< ValType, RowIndexType, NonzeroIndexType, ColIndexType >;

				NZWrapper< ValType, RowIndexType, NonzeroIndexType, ColIndexType > _val;


			public:

				/** Base constructor. */
				NZIterator(
					Compressed_Storage< ValType, RowIndexType, NonzeroIndexType > &CXX,
					ColIndexType *row_values_buffer,
					size_t off
				) noexcept : _val( CXX, row_values_buffer, off ) {}


				/** Copy constructor. */
				NZIterator( const SelfType &other ) noexcept :
					_val( *other._val._CXX, other._val._col_values_buffer, other._val.off() )
				{}

				/** Copy-assignment. */
				SelfType & operator=( const SelfType &other ) noexcept {
					this->_val._CXX = other._val._CXX;
					this->_val._col_values_buffer = other._val._col_values_buffer;
					this->_val.off() = other._val.off();
					return *this;
				}

				/** Increment by one. */
				SelfType & operator++() noexcept {
					(void) this->_val.off()++;
					return *this;
				}

				/** Decrement by one. */
				SelfType & operator--() noexcept {
					(void)this->_val.off()--;
					return *this;
				}

				/** Increment by arbitrary value. */
				SelfType & operator+=( const size_t off ) noexcept {
					(void) (this->_val.off() += off);
					return *this;
				}

				/** Create copy at offset. */
				SelfType operator+( const size_t offset ) const noexcept {
					SelfType copy( *this );
					(void) (copy += offset);
					return copy;
				}

				/** Create copy at negative offset. */
				SelfType operator-( const size_t offset ) const noexcept {
					SelfType copy( *this );
					(void) (copy._val.off() -= offset);
					return copy;
				}

				/** Whether not equal to another iterator into the same container. */
				inline bool operator!=( const SelfType &other ) const {
					return this->_val.off() != other._val.off();
				}

				/** Whether equal to another iterator into the same container. */
				inline bool operator==( const SelfType &other ) const {
					return !(this->operator!=( other ));
				}

				/** Whether compares less-than another iterator. */
				bool operator<( const SelfType &other ) const {
					return this->_val.off() < other._val.off();
				}

				/** Dereferences this iterator. */
				typename SelfType::reference operator*() {
					return _val;
				}

				/** Returns a pointer to the underlying value. */
				typename SelfType::pointer operator->() {
					return &_val;
				}

				/** Returns the difference between this and another iterator. */
				typename SelfType::difference_type operator-(
					const SelfType &other
				) const {
					if( this->_val.off() > other._val.off() ) {
						return static_cast< typename SelfType::difference_type >(
							this->_val.off() - other._val.off() );
					} else {
						return - static_cast< typename SelfType::difference_type >(
							other._val.off() - this->_val.off() );
					}
				}

		};

	} // end namespace grb::internal

} // end namespace grb

#endif // _H_GRB_NONZERO_WRAPPER

