
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

#ifndef _GRB_NONZERO_WRAPPER
#define _GRB_NONZERO_WRAPPER

#include <iostream>
#include <type_traits>
#include <cstddef>
#include <iterator>

#include "compressed_storage.hpp"

namespace grb {

	namespace internal {

		// FORWARD DECLARATIONS
		template<
			typename ValType,
			typename RowIndexType,
			typename ColIndexType
		>
		struct NZStorage;

		template<
			typename ValType,
			typename RowIndexType,
			typename NonzeroIndexType,
			typename ColIndexType
		>
		struct NZWrapper;

		/**
		 * @brief Wrapper class provide a {row,col,val}() interface over a nonzero.
		 * 	It internally points directly to the data inside an underlying
		 * 	CRS/CCS storage + row/column buffer.
		 *
		 * Use only in conjunction with NZIterator.
		 */
		template<
			typename ValType,
			typename RowIndexType,
			typename NonzeroIndexType,
			typename ColIndexType
		>
		struct NZWrapper {

			Compressed_Storage< ValType, RowIndexType, NonzeroIndexType >* _CXX;
			ColIndexType *_col_values_buffer;
			size_t _off;

			using self_t = NZWrapper< ValType, RowIndexType, NonzeroIndexType, ColIndexType >;

			NZWrapper(
				Compressed_Storage< ValType, RowIndexType, NonzeroIndexType > &CXX,
				ColIndexType *col_values_buffer,
				size_t off
			) noexcept :
				_CXX( &CXX ),
				_col_values_buffer( col_values_buffer ),
				_off( off)
			{}

			NZWrapper() = delete;

			NZWrapper( const self_t & ) = delete;

			NZWrapper( self_t && ) = default;

			self_t & operator=( const self_t & ) = delete;

			inline ColIndexType& col() { return this->_col_values_buffer[ this->_off ]; }
			inline ColIndexType col() const { return this->_col_values_buffer[ this->_off ]; }

			inline RowIndexType& row() { return this->_CXX->row_index[ this->_off ]; }
			inline RowIndexType row() const { return this->_CXX->row_index[ this->_off ]; }

			inline size_t& off() { return this->_off; }
			inline size_t off() const { return this->_off; }

			self_t & operator=( self_t && other ) noexcept {
#ifdef _DEBUG
				std::cout << "transfer ";
				print( std::cout, *this);
				std::cout << " <- ";
				print( std::cout, other );
				std::cout << std::endl;
#endif
				this->col() = other.col();
				this->row() = other.row();
				this->__write_value( other );
				return *this;
			}

			self_t & operator=( NZStorage< ValType, RowIndexType, ColIndexType >&& storage ) noexcept {
#ifdef _DEBUG
				std::cout << "copying into wrapper ";
				print( std::cout, *this );
				std::cout << " <- ";
				NZStorage< ValType, RowIndexType, ColIndexType >::print( std::cout, storage );
				std::cout << std::endl;
#endif
				storage.copyTo( *this );
				return *this;
			}

			bool operator<( const self_t &other ) const noexcept {
				const bool result = ( this->col() < other.col() )
					|| ( this->col() == other.col() && this->row() >= other.row() // reverse order
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

			void __swap( self_t &other ) noexcept {
				std::swap( this->col(), other.col() );
				std::swap( this->row(), other.row() );
				this->__swap_value( other );
			}

			template< typename T > void inline __swap_value(
				NZWrapper< T, RowIndexType, NonzeroIndexType, ColIndexType > &other,
				typename std::enable_if< ! std::is_same< T, void >::value >::type * = nullptr
			) noexcept {
				std::swap( this->_CXX->values[ this->_off ], other._CXX->values[ other._off ] );
			}

			template< typename T > void inline __swap_value(
				NZWrapper< T, RowIndexType, NonzeroIndexType, ColIndexType > &other,
				typename std::enable_if< std::is_same< T, void >::value >::type * = nullptr
			) noexcept {
				(void)other;
			}

			template< typename T > void inline __write_value(
				NZWrapper< T, RowIndexType, NonzeroIndexType, ColIndexType > &other,
				typename std::enable_if< ! std::is_same< T, void >::value >::type * = nullptr
			) noexcept {
				this->_CXX->values[ this->_off ] = other._CXX->values[ other._off ];
			}

			template< typename T > void inline __write_value(
				NZWrapper< T, RowIndexType, NonzeroIndexType, ColIndexType > &other,
				typename std::enable_if< std::is_same< T, void >::value >::type * = nullptr
			) noexcept {
				(void)other;
			}

			template< typename T = ValType >
				typename std::enable_if< ( ! std::is_same< T, void >::value )
					&& std::is_same< T, ValType >::value, ValType >::type
			&val(
				typename std::enable_if< ( ! std::is_same< T, void >::value )
					&& std::is_same< T, ValType >::value >::type * = nullptr
			) {
				return this->_CXX->values[ this->_off ];
			}

			template< typename T = ValType > ValType val(
				typename std::enable_if< ( ! std::is_same< T, void >::value )
					&& std::is_same< T, ValType >::value >::type * = nullptr
			) const {
				return this->_CXX->values[ this->_off ];
			}

			template< typename T = ValType > char val(
				typename std::enable_if< std::is_same< T, void >::value
					&& std::is_same< T, ValType >::value >::type * = nullptr
			) const { // const to fail if assigning
				return '\0';
			}

#ifdef _DEBUG
			static void print( std::ostream &s, const self_t &nz ) {
				s << nz.off() << ": [ " << nz.col() << ", "
					<< nz.row() << "]: "
					<< nz.val();
			}
#endif
		};

		/**
		 * @brief specialized swap function for NZWrapper, swapping the triple
		 * 	{row,col,val}.
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
		) {
#ifdef _DEBUG
			std::cout << "calling swap" << std::endl;
#endif
			a.__swap( b );
		}

		/**
		 * @brief Base class for wor and column storage with {row,col}() interface.
		 */
		template<
			typename RowIndexType,
			typename ColIndexType
		>
		struct _NZStorageBase {

			using self_t = _NZStorageBase< RowIndexType, ColIndexType >;

			ColIndexType _col;
			RowIndexType _row;

			_NZStorageBase() = delete;

			self_t operator=( const self_t & ) = delete;

			template< typename V, typename NonzeroIndexType > _NZStorageBase(
				const NZWrapper< V, RowIndexType, NonzeroIndexType, ColIndexType > &orig
			) noexcept :
				_col( orig.col() ), _row( orig.row() ) {}

			inline ColIndexType& col() { return this->_col; }
			inline ColIndexType col() const { return this->_col; }

			inline RowIndexType& row() { return this->_row; }
			inline RowIndexType row() const { return this->_row; }

			template< typename V, typename NonzeroIndexType > self_t operator=(
				NZWrapper< V, RowIndexType, NonzeroIndexType, ColIndexType > &&orig
			) noexcept {
				this->_col = orig.col();
				this->_row = orig.row();
				return *this;
			}

			template< typename V, typename NonzeroIndexType > void copyTo(
				NZWrapper< V, RowIndexType, NonzeroIndexType, ColIndexType > &dest
			) noexcept {
				dest.col() = this->_col;
				dest.row() = this->_row;
			}
		};

		/**
		 * @brief Storage for a nonzero with {row,col,val}() interface and comparisn
		 * 	and copy/move logic from NZWrapper.
		 *
		 * Used within NZIterator to store nonzeroes as a local caches, within sorting
		 * algorithms like insertion sort that extracts nonzeroes as <em> CacheT c = *iterator <\em>.
		 */
		template<
			typename ValType,
			typename RowIndexType,
			typename ColIndexType
		>
		struct NZStorage: public _NZStorageBase< RowIndexType, ColIndexType > {

			using self_t = NZStorage< ValType, RowIndexType, ColIndexType >;
			using base_t = _NZStorageBase< RowIndexType, ColIndexType >;
			ValType _val;

			NZStorage() = delete;

			template< typename NonzeroIndexType > NZStorage(
				const NZWrapper< ValType, RowIndexType, NonzeroIndexType, ColIndexType > &orig
			) noexcept :
				base_t( orig ),
				_val( orig.val() )
			{
#ifdef _DEBUG
					std::cout << "create storage ";
					print( std::cout, *this );
					std::cout << std::endl;
#endif
			}

			ValType& val() { return this->_val; }
			ValType val() const { return this->_val; }

			self_t operator=( const self_t & ) = delete;

			template< typename NonzeroIndexType > self_t operator=(
				NZWrapper< ValType, RowIndexType, NonzeroIndexType, ColIndexType > &&orig
			) noexcept {
#ifdef _DEBUG
				std::cout << "moving into storage ";
				NZWrapper< ValType, RowIndexType, NonzeroIndexType, ColIndexType >::print(
					std::cout, orig );
				std::cout << std::endl;
#endif
				(void)this->base_t::operator=( orig );
				this->_val = orig.val();
				return *this;
			}

			template< typename NonzeroIndexType > void copyTo(
				NZWrapper< ValType, RowIndexType, NonzeroIndexType, ColIndexType > &dest
			) noexcept {
				this->base_t::copyTo( dest );
				dest.val() = this->_val;
			}
#ifdef _DEBUG
			static void print( std::ostream &s, const self_t &nz ) {
				s << "( " << nz.col() << ", " << nz.row() << " )" << ": " << nz.val();
			}
#endif
		};

#ifdef _DEBUG

#endif

		/**
		 * @brief Same as NZStorage< ValType, RowIndexType, NonzeroIndexType, ColIndexType >,
		 * 	but for pattern matrices, i.e. without value.
		 */
		template<
			typename RowIndexType,
			typename ColIndexType
		>
		struct NZStorage< void, RowIndexType, ColIndexType >:
			public _NZStorageBase< RowIndexType, ColIndexType > {

			using self_t = NZStorage< void, RowIndexType, ColIndexType >;
			using base_t = _NZStorageBase< RowIndexType, ColIndexType >;

			NZStorage() = delete;

			self_t operator=( const self_t & ) = delete;

			template< typename NonzeroIndexType > NZStorage(
				const NZWrapper< void, RowIndexType, NonzeroIndexType, ColIndexType > &orig
			) noexcept :
				base_t( orig )
			{
#ifdef _DEBUG
					std::cout << "create storage ";
					print( std::cout, *this );
					std::cout << std::endl;
#endif
			}

			template< typename NonzeroIndexType > self_t operator=(
				NZWrapper< void, RowIndexType, NonzeroIndexType, ColIndexType > &&orig
			) noexcept {
#ifdef _DEBUG
				std::cout << "moving into storage ";
				NZWrapper< void, RowIndexType, NonzeroIndexType, ColIndexType >::print(
					std::cout, orig );
				std::cout << std::endl;
#endif
				(void)this->base_t::operator=( orig );
				return *this;
			}

#ifdef _DEBUG
			static void print( std::ostream &s, const self_t &nz ) {
				s << "( " << nz.col() << ", " << nz.row() << " )";
			}
#endif
		};

		/**
		 * @brief comparison operator between an NZStorage-object \p a and an
		 * NZWrapper-object \b. Invoked during std::sort().
		 */
		template<
			typename ValType,
			typename RowIndexType,
			typename NonzeroIndexType,
			typename ColIndexType
		> bool operator<(
			const NZStorage< ValType, RowIndexType, ColIndexType > &a,
			const NZWrapper< ValType, RowIndexType, NonzeroIndexType, ColIndexType > &b
		) {

			const bool result = ( a.col() < b.col() )
				|| ( a.col() == b.col() && a.row() >= b.row() );

#ifdef _DEBUG
			std::cout << "compare:: ";
			NZStorage< ValType, RowIndexType, ColIndexType >::print( std::cout, a );
			std::cout << " < ";
			NZWrapper< ValType, RowIndexType, NonzeroIndexType, ColIndexType >::print( std::cout, b );
			std::cout << ( result ? " true" : " false" ) << std::endl;
#endif
			return result;
		}

		/**
		 * @brief comparison operator between an NZWrapper-object \p a and an
		 * NZStorage-object \b. Invoked during std::sort().
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
			const bool result = ( a.col() < b.col() )
				|| ( a.col() == b.col() && a.row() >= b.row() );
#ifdef _DEBUG
			std::cout << "compare:: ";
			NZWrapper< ValType, RowIndexType, NonzeroIndexType, ColIndexType >::print( std::cout, a );
			std::cout << " < ";
			NZStorage< ValType, RowIndexType, ColIndexType >::print( std::cout, b );
			std::cout << ( result ? " true" : " false" ) << std::endl;
#endif
			return result;
		}

		/**
		 * @brief Wrapper utility for a CRS/CCS with coordinated rows/columns buffer
		 * 	(not stored in the CRS/CCS), storing the pointers to the actual data.
		 * 	It  allows iterating over the nonzeroes in order to sort them.
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
		struct NZIterator {
			// STL iterator fields
			using iterator_category = std::random_access_iterator_tag;
			using value_type = NZStorage< ValType, RowIndexType, ColIndexType >;
			using difference_type = long;
			using pointer = NZWrapper< ValType, RowIndexType, NonzeroIndexType, ColIndexType >*;
			using reference = NZWrapper< ValType, RowIndexType, NonzeroIndexType, ColIndexType >&;

			using self_t = NZIterator< ValType, RowIndexType, NonzeroIndexType, ColIndexType >;
			using __ref_value_type = NZWrapper< ValType, RowIndexType, NonzeroIndexType, ColIndexType >;

			NZIterator(
				Compressed_Storage< ValType, RowIndexType, NonzeroIndexType > &CXX,
				ColIndexType *row_values_buffer,
				size_t off
			) noexcept :
				_val( CXX, row_values_buffer, off )
			{}


			NZIterator( const self_t &other ) noexcept :
				_val( *other._val._CXX, other._val._col_values_buffer, other._val.off() ) {}

			self_t & operator=( const self_t &other ) noexcept {
				this->_val._CXX = other._val._CXX;
				this->_val._col_values_buffer = other._val._col_values_buffer;
				this->_val.off() = other._val.off();
				return *this;
			}

			self_t & operator++() noexcept {
				(void)this->_val.off()++;
				return *this;
			}

			self_t & operator--() noexcept {
				(void)this->_val.off()--;
				return *this;
			}

			self_t & operator+=( size_t off ) noexcept {
				(void)(this->_val.off() += off);
				return *this;
			}

			self_t operator+( size_t offset ) const noexcept {
				self_t copy( *this );
				(void)(copy += offset );
				return copy;
			}

			self_t operator-( size_t offset ) const noexcept {
				self_t copy( *this );
				(void)(copy._val.off() -= offset );
				return copy;
			}

			bool operator!=( const self_t &other ) const {
				return this->_val.off() != other._val.off();
			}

			bool inline operator==( const self_t &other ) const {
				return ! this->operator!=( other );
			}

			bool operator<( const self_t &other ) const {
				return this->_val.off() < other._val.off();
			}

			typename self_t::reference operator*() {
				return _val;
			}

			typename self_t::pointer operator->() {
				return &_val;
			}

			typename self_t::difference_type operator-( const self_t &other ) const {
				if( this->_val.off() > other._val.off() ) {
					return static_cast< typename self_t::difference_type >(
						this->_val.off() - other._val.off() );
				} else {
					return - static_cast< typename self_t::difference_type >(
						other._val.off() - this->_val.off() );
				}
			}

		private:
			__ref_value_type _val;
		};

#ifdef _DEBUG
		template<
			typename ValType,
			typename RowIndexType,
			typename NonzeroIndexType
		> inline ValType& get_value(
			const Compressed_Storage< ValType, RowIndexType, NonzeroIndexType > &ccs,
			size_t s
		) {
			return ccs.values[s];
		}

		template<
			typename RowIndexType,
			typename NonzeroIndexType
		> inline char get_value(
			const Compressed_Storage< void, RowIndexType, NonzeroIndexType >&,
			size_t
		) {
			return '\0';
		}
#endif
	}
}

#endif // _GRB_NONZERO_WRAPPER

