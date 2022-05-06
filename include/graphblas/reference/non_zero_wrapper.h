
#ifndef _GRB_NONZERO_WRAPPER
#define _GRB_NONZERO_WRAPPER

#include "compressed_storage.hpp"
#include <iostream>


namespace grb {

	namespace internal {

        template<
			typename ValType,
			typename RowIndexType,
			typename NonzeroIndexType
		> struct NZStorage;

        template<
			typename ValType,
			typename RowIndexType,
			typename NonzeroIndexType
		> struct NZWrapper;

		template<
			typename V,
			typename R,
			typename N
		> std::ostream& operator<<( std::ostream& s, const NZWrapper< V, R, N >& nz );


        template<
			typename V,
			typename R,
			typename N
		> std::ostream& operator<<( std::ostream& s, const NZStorage< V, R, N >& nz );



		template<
			typename ValType,
			typename RowIndexType,
			typename NonzeroIndexType
		> struct NZWrapper {

			Compressed_Storage< ValType, RowIndexType, NonzeroIndexType >* _CXX;
			RowIndexType *_row_values_buffer;
			size_t _off;

			using self_t = NZWrapper< ValType, RowIndexType, NonzeroIndexType >;

			NZWrapper( Compressed_Storage< ValType, RowIndexType, NonzeroIndexType > &CXX,
				RowIndexType *row_values_buffer,
				size_t off ): _CXX( &CXX ), _row_values_buffer( row_values_buffer ), _off( off) {}

			NZWrapper() = delete;

			NZWrapper( const self_t& ) = delete;

			NZWrapper( self_t&& ) = default;

			self_t& operator=( const self_t& ) = delete;

			self_t& operator=( self_t&& other ) {
#ifdef __WR_DBG
                std::cout << "transfer " << *this << " <- " << other << std::endl;
#endif
                this->_row_values_buffer[ this->_off ] = other._row_values_buffer[ other._off ];
                this->_CXX->row_index[ this->_off ] = other._CXX->row_index[ other._off ];
                this->_CXX->row_index[ this->_off ] = other._CXX->row_index[ other._off ];
                this->__write_value( other );
                return *this;
            }

            self_t& operator=( NZStorage< ValType, RowIndexType, NonzeroIndexType >&& storage ) {
#ifdef __WR_DBG
                std::cout << "copying into wrapper " << *this << " <- " << storage << std::endl;
#endif
                storage.copyTo( *this );
                return *this;
            }

			bool operator<( const self_t &other ) const {
				const bool result = ( this->_row_values_buffer[ this->_off ] < other._row_values_buffer[ other._off ] )
                    || ( this->_row_values_buffer[ this->_off ] == other._row_values_buffer[ other._off ]
					    && this->_CXX->row_index[ this->_off ] >= other._CXX->row_index[ other._off ] // reverse order
                    );
#ifdef __WR_DBG
                std::cout << "compare:: " << *this << " < " << other
                    << ( result ? " true" : " false" ) << std::endl;
#endif
				return result;
			}

			void __swap( self_t& other ) {
				std::swap( this->_row_values_buffer[ this->_off ], other._row_values_buffer[ other._off ] );
				std::swap( this->_CXX->row_index[ this->_off ], other._CXX->row_index[ other._off ] );
				this->__swap_value( other );
			}

			template< typename T > void inline __swap_value( NZWrapper< T, RowIndexType, NonzeroIndexType >& other,
				typename std::enable_if< ! std::is_same< T, void >::value >::type * = nullptr ) {
				std::swap( this->_CXX->values[ this->_off ], other._CXX->values[ other._off ] );
			}

			template< typename T > void inline __swap_value( NZWrapper< T, RowIndexType, NonzeroIndexType >& other,
				typename std::enable_if< std::is_same< T, void >::value >::type * = nullptr ) {
			}




            template< typename T > void inline __write_value( NZWrapper< T, RowIndexType, NonzeroIndexType >& other,
				typename std::enable_if< ! std::is_same< T, void >::value >::type * = nullptr ) {
				this->_CXX->values[ this->_off ] = other._CXX->values[ other._off ];
			}

			template< typename T > void inline __write_value( NZWrapper< T, RowIndexType, NonzeroIndexType >& other,
				typename std::enable_if< std::is_same< T, void >::value >::type * = nullptr ) {
			}
		};

		template<
			typename ValType,
			typename RowIndexType,
			typename NonzeroIndexType
		> void swap(
			NZWrapper< ValType, RowIndexType, NonzeroIndexType >& a,
			NZWrapper< ValType, RowIndexType, NonzeroIndexType >& b ) {
			std::cout << "calling swap" << std::endl;
			a.__swap( b );
		}

		template<
			typename V,
			typename R,
			typename N
		> std::ostream& operator<<( std::ostream& s, const NZWrapper< V, R, N >& nz ) {
            s << nz._off << ": [ " << nz._row_values_buffer[ nz._off ] << ", "
                << nz._CXX->row_index[ nz._off ] << ": "
                << nz._CXX->values[ nz._off ] << " ]";
            return s;
        }


        template<
			typename RowIndexType,
			typename NonzeroIndexType
		> struct _NZStorageBase {

			using self_t = _NZStorageBase< RowIndexType, NonzeroIndexType >;

            RowIndexType _row;
            NonzeroIndexType _col;

            _NZStorageBase() = delete;

            self_t operator=( const self_t& ) = delete;

            template< typename V > _NZStorageBase( const NZWrapper< V, RowIndexType, NonzeroIndexType >& orig ):
                _row( orig._row_values_buffer[ orig._off ] ), _col( orig._CXX->row_index[ orig._off ]) {}

            template< typename V > self_t operator=( NZWrapper< V, RowIndexType, NonzeroIndexType >&& orig ) {
                this->_row = orig._row_values_buffer[ orig._off ];
                this->_col = orig._CXX->row_index[ orig._off ];
                return *this;
            }

            template< typename V > void copyTo( NZWrapper< V, RowIndexType, NonzeroIndexType >& dest ) {
                dest._row_values_buffer[ dest._off ] = this->_row;
                dest._CXX->row_index[ dest._off ] = this->_col;
            }
        };

        template<
			typename ValType,
			typename RowIndexType,
			typename NonzeroIndexType
		> struct NZStorage: public _NZStorageBase< RowIndexType, NonzeroIndexType > {

			using self_t = NZStorage< ValType, RowIndexType, NonzeroIndexType >;
            using base_t = _NZStorageBase< RowIndexType, NonzeroIndexType >;
            ValType _val;

            NZStorage() = delete;

            self_t operator=( const self_t& ) = delete;

            template< typename V > NZStorage( const NZWrapper< V, RowIndexType, NonzeroIndexType >& orig ):
                base_t( orig ), _val( orig._CXX->values[ orig._off ] ) {
#ifdef __WR_DBG
                    std::cout << "create storage " << *this << std::endl;
#endif
            }

            self_t operator=( NZWrapper< ValType, RowIndexType, NonzeroIndexType >&& orig ) {
#ifdef __WR_DBG
                std::cout << "copying into storage " << orig << std::endl;
#endif
                (void)this->base_t::operator=( orig );
                this->_val = orig._CXX->values[ orig._off ];
                return *this;
            }

            void copyTo( NZWrapper< ValType, RowIndexType, NonzeroIndexType >& dest ) {
                this->base_t::copyTo( dest );
                dest._CXX->values[ dest._off ] = this->_val;
            }
        };

#ifdef __WR_DBG
        template<
			typename V,
			typename R,
			typename N
		> std::ostream& operator<<( std::ostream& s, const NZStorage< V, R, N >& nz ) {
            s << "( " << nz._row << ", " << nz._col << ": " << nz._val << " )";
            return s;
        }
#endif

        template<
			typename RowIndexType,
			typename NonzeroIndexType
		> struct NZStorage< void, RowIndexType, NonzeroIndexType >:
            public _NZStorageBase< RowIndexType, NonzeroIndexType > {

			using self_t = NZStorage< void, RowIndexType, NonzeroIndexType >;
            using base_t = _NZStorageBase< RowIndexType, NonzeroIndexType >;

            NZStorage() = delete;

            self_t operator=( const self_t& ) = delete;

            template< typename V > NZStorage( const NZWrapper< void, RowIndexType, NonzeroIndexType >& orig ):
                base_t( orig ) {}

            self_t operator=( NZWrapper< void, RowIndexType, NonzeroIndexType >&& orig ) {
                (void)this->base_t::operator=( orig );
                return *this;
            }
        };


        template<
			typename ValType,
			typename RowIndexType,
			typename NonzeroIndexType
		> bool operator<( const NZStorage< ValType, RowIndexType, NonzeroIndexType >& a,
            const NZWrapper< ValType, RowIndexType, NonzeroIndexType >& b ) {

            const bool result = ( a._row < b._row_values_buffer[ b._off ] )
                || ( a._row == b._row_values_buffer[ b._off ] && a._col >= b._CXX->row_index[ b._off ] );

#ifdef __WR_DBG
            std::cout << "compare:: " << a << " < " << b
                << ( result ? " true" : " false" ) << std::endl;
#endif
            return result;
        }

        template<
			typename ValType,
			typename RowIndexType,
			typename NonzeroIndexType
		> bool operator<( const NZWrapper< ValType, RowIndexType, NonzeroIndexType >& a,
            const NZStorage< ValType, RowIndexType, NonzeroIndexType >& b ) {       
            const bool result = ( a._row_values_buffer[ a._off ] < b._row )
                || ( a._row_values_buffer[ a._off ] == b._row && a._CXX->row_index[ a._off ] >= b._col );
#ifdef __WR_DBG
            std::cout << "compare:: " << a << " < " << b
                << ( result ? " true" : " false" ) << std::endl;
#endif
            return result;
        }



		template<
			typename ValType,
			typename RowIndexType,
			typename NonzeroIndexType
		> struct NZIterator {

			using self_t = NZIterator< ValType, RowIndexType, NonzeroIndexType >;
			using iterator_category = std::random_access_iterator_tag;
			using value_type = NZStorage< ValType, RowIndexType, NonzeroIndexType >;
			using __ref_value_type = NZWrapper< ValType, RowIndexType, NonzeroIndexType >;
			using pointer = __ref_value_type*;
			using reference = __ref_value_type&;
			using difference_type = signed long;


			NZIterator( Compressed_Storage< ValType, RowIndexType, NonzeroIndexType > &CXX,
				config::RowIndexType *row_values_buffer,
				size_t off ): _val( CXX, row_values_buffer, off ) {}
			
			
			NZIterator( const self_t& other ):
				_val( *other._val._CXX, other._val._row_values_buffer, other._val._off ) {}

			self_t& operator=( const self_t& other ) {
				this->_val._CXX = other._val._CXX;
				this->_val._row_values_buffer = other._val._row_values_buffer;
				this->_val._off = other._val._off;
				return *this;
			}

			self_t& operator++() {
				(void)this->_val._off++;
				return *this;
			}

			self_t& operator--() {
				(void)this->_val._off--;
				return *this;
			}

			self_t& operator+=( size_t off ) {
				(void)(this->_val._off += off);
				return *this;
			}


			self_t operator+( size_t offset ) const {
				self_t copy( *this );
				(void)(copy += offset );
				return copy;
			}

			self_t operator-( size_t offset ) const {
				self_t copy( *this );
				(void)(copy._val._off -= offset );
				return copy;
			}

			bool operator!=( const self_t& other ) const {
				return this->_val._off != other._val._off;
			}

			bool inline operator==( const self_t& other ) const {
				return ! this->operator!=( other );
			}

			bool operator<( const self_t& other ) const {
				return this->_val._off < other._val._off;
			}

			reference& operator*() {
				return _val;
			}

			difference_type operator-( const self_t& other ) const {
				if( this->_val._off > other._val._off ) {
					return static_cast< difference_type >( this->_val._off - other._val._off );
				} else {
					return - static_cast< difference_type >( other._val._off - this->_val._off );
				}
			}

		private:
			__ref_value_type _val;
		};

	}
}

#endif // _GRB_NONZERO_WRAPPER
