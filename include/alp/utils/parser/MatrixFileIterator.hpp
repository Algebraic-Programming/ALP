
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
 * @date 25th of May, 2017
 */

#ifndef _H_MATRIXFILEITERATOR
#define _H_MATRIXFILEITERATOR

#include <cstddef> //std::ptrdiff_t
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept> //std::runtime_error

#include <stdlib.h> //posix_memalign

#include <utility> //std::pair

#include <assert.h>

#include <alp/config.hpp>
#include <alp/iomode.hpp>
//#include <alp/spmd.hpp>
//#include <alp/utils/config.hpp>
//#include <alp/utils/hpparser.h>

#include "MatrixFileProperties.hpp"
#include "config.hpp"

#ifdef _GRB_WITH_OMP
#include <alp/omp/config.hpp>
#endif

namespace alp {
	namespace utils {
		namespace internal {

			// template <typename T,
			// 	  bool data_reflect = false,
			// 	  size_t buffer_size = 0
			// >
			// struct BufferType {
			// 	typename std::conditional<
			// 		data_reflect,
			// 		std::pair <
			// 			std::array< T, buffer_size >, // direct buffer
			// 			std::array< T, buffer_size >  // reflected buffer
			// 			>,
			// 		std::array<T, buffer_size> // direct buffer
			// 	>::type data;
			// 	typename std::conditional<
			// 		data_reflect,
			// 		std::pair <
			// 			size_t, // direct buffer counter
			// 			size_t  // reflecetd buffer counter
			// 		>,
			// 		size_t
			// 	>::type pos;
			// };


			template< typename T, bool data_reflect = false,  typename S = size_t >
			class MatrixFileIterator {

				// template< typename X1, typename X2 >
				// friend std::ostream & operator<<( std::ostream &, const MatrixFileIterator< X1, X2 > & );

			private:
				/** The output type of the base iterator. */
				typedef T OutputType;

				/** Iterators will retrieve this many lines at a time from the input file. */
				static constexpr size_t buffer_size = alp::config::PARSER::bsize();

				/** The nonzero buffer. */
				OutputType *buffer;

				/** The underlying MatrixReader. */
				MatrixFileProperties & properties;

				/** The input stream. */
				std::ifstream infile;

				/** The input stream position. */
				std::streampos spos;

				/** The current position in the buffer. */
				size_t pos;

				/** The buffer couter (buffers used so far). */
				size_t buffercount;

				/** The i index counter. */
				size_t icount;

				/** The j index counter. */
				size_t jcount;

				/** Whether the \a infile stream \em and \a buffer have been depleted. */
				bool ended;

				/** Whether the first fill of the buffer is held until the first dereference of this iterator is taking place. */
				bool started;

				/** A function to apply to convert input values on the fly. */
				std::function< void( T & ) > converter;

				/** Strips comments and possible MatrixMarket header from input stream start. */
				void preprocess() {
					// check if first header indicates MatrixMarket
					const std::streampos start = infile.tellg();
					// assume input is matrix market until we detect otherwise
					bool mmfile = true;
					// try and parse header
					std::string mmheader;
					if( ! std::getline( infile, mmheader ) ) {
						// some type of error occurred-- rewind and let a non-mmfile parse try
						mmfile = false;
						(void)infile.seekg( start );
					}
					if( mmfile && ( mmheader.size() < 14 || mmheader.substr( 0, 14 ) != "%%MatrixMarket" ) ) {
						// some type of error occurred-- rewind and let a non-mmfile parse try
						mmfile = false;
						(void)infile.seekg( start );
					}
					// ignore all comment lines
					char peek = infile.peek();
					while( infile.good() && ( peek == '%' || peek == '#' ) ) {
						(void)infile.ignore( std::numeric_limits< std::streamsize >::max(), '\n' );
						peek = infile.peek();
					}
					// ignore non-comment matrix market header if we expect one
					if( mmfile ) {
						std::string ignore;
						std::getline( infile, ignore );
					} else {
						mmheader.clear();
					}
					// done
				}


			public:

				// standard STL iterator typedefs
				typedef std::ptrdiff_t difference_type;
				typedef OutputType value_type;
				typedef OutputType & reference;
				typedef OutputType * pointer;
				typedef std::forward_iterator_tag iterator_category;

				typedef T nonzero_value_type;

				/** Base constructor, starts in begin position. */
				MatrixFileIterator(
					MatrixFileProperties & prop,
					IOMode mode, const std::function< void( T & ) > valueConverter,
					const bool end = false
				) :
					buffer( NULL ), properties( prop ), infile( properties._fn ), spos(), pos( 0 ), buffercount( 0 ),
					icount( 0 ), jcount( 0 ), ended( end ),
					started( ! end ),
					converter( valueConverter ) {
					if( mode != SEQUENTIAL ) {
						throw std::runtime_error(
							"Only sequential IO is supported by this iterator at "
							"present, sorry."
						);
					}
				}

				/** Copy constructor. */
				MatrixFileIterator( const MatrixFileIterator< T > & other ) :
					buffer( NULL ), properties( other.properties ), infile( properties._fn ), spos( other.spos ),
					pos( other.pos ), buffercount( other.buffercount ),
					icount( other.icount ), jcount( other.jcount ), ended( other.ended ), started( other.started ),
					converter( other.converter ) {
					// set latest stream position
					(void)infile.seekg( spos );
					// if buffer is nonempty
					if( pos > 0 ) {
						// allocate buffer
						if( posix_memalign( (void **)&buffer, config::CACHE_LINE_SIZE::value(), buffer_size * sizeof( OutputType ) ) != 0 ) {
							buffer = NULL;
							throw std::runtime_error( "Error during allocation of internal iterator memory." );
						}
						// copy any remaining buffer contents
						for( size_t i = 0; i < pos; ++i ) {
							buffer[ i ] = other.buffer[ i ];
						}
					}
				}

				/** Base destructor. */
				~MatrixFileIterator() {
					if( buffer != NULL ) {
						free( buffer );
					}
				}

				/** Copies an iterator state. */
				MatrixFileIterator & operator=( const MatrixFileIterator & x ) {
					// copy ended state
					ended = x.ended;
					// copy started state
					started = x.started;
					// copy converter
					converter = x.converter;
					// check if we are done already
					if( ended ) {
						return *this;
					}
					// check if both iterators have a similar ifstream
					if( properties._fn == x.properties._fn ) {
						// NO, so re-create infile
						if( infile.is_open() ) {
							infile.close();
						}
						infile.open( x.properties._fn );
						properties = x.properties;
					}
					// not yet done, copy input file stream position
					(void)infile.seekg( x.spos );
					spos = x.spos;
					pos = x.pos;
					// copy any remaining buffer contents
					if( pos > 0 ) {
						// check if buffer is allocated
						if( buffer == NULL ) {
							// no, so allocate buffer
							if( posix_memalign(
								    (void **)&buffer,
								    config::CACHE_LINE_SIZE::value(),
								    buffer_size * sizeof( OutputType )
							    ) != 0 ) {
								buffer = NULL;
								throw std::runtime_error(
									"Error during allocation of internal iterator memory."
								);
							}
						}
						// copy remote buffer contents
						assert( pos < buffer_size );
						for( size_t i = 0; i <= pos; ++i ) {
							buffer[ i ] = x.buffer[ i ];
						}
					}
					// done
					return *this;
				}

				/** Standard check for equality. */
				bool operator==( const MatrixFileIterator &x ) const {
					// check if both are in end position
					if( ended && x.ended ) {
						return true;
					}
#ifndef NDEBUG
					if( properties._fn != x.properties._fn ) {
						std::cerr << "Warning: comparing two instances of "
									 "MatrixFileIterator that are 1) nonempty "
									 "*and* 2) not reading from the same "
									 "file.\n";
					}
#endif
					// check if both are in start position
					if( started && x.started ) {
						return true;
					}
					// otherwise, only can compare equal if in the same position
					if( pos && x.pos ) {
						// AND in the same input stream position
						return spos == x.spos;
					}
					// otherwise, not equal
					return false;
				}

				/** Standard check for inequality, relies on equality check. */
				bool operator!=( const MatrixFileIterator &x ) const {
					return ! ( operator==( x ) );
				}

				// this assumes full triplet data
				MatrixFileIterator & operator++() {
					// if ended, stop
					if( ended ) {
						return *this;
					}
					// if this is the first function call on this iterator, call preprocess first
					if( started ) {
						preprocess();
						started = false;
						(void)operator++();
					}


					// check if we need to parse from infile
					if( pos == 0 ) {
						// try and parse buffer_size new values
						size_t i = 0;
						if( ! infile.good() ) {
							ended = true;
						}

						// check if buffer is allocated
						if( buffer == NULL ) {
							// no, so allocate buffer
							if( posix_memalign(
								    (void **)&buffer,
								    config::CACHE_LINE_SIZE::value(),
								    buffer_size * sizeof( OutputType )
							    ) != 0 ) {
								buffer = NULL;
								throw std::runtime_error(
									"Error during allocation of internal iterator memory."
								);
							}
						}

						for( ; ! ended && i < buffer_size; ++i ) {
							S row, col;
							T val;
							if( ! ( infile >> val ) ) {
								if( i == 0 ) {
									ended = true;
								}
								break;
							} else {
#ifdef _DEBUG
								T temp = val;
								converter( temp );
								std::cout << "MatrixFileIterator::operator++  parsed line ``"
									  << val  << "'', with value after conversion "
									  << temp << "\n";
#endif
								// convert value
								converter( val );

								// store read values
								buffer[ buffer_size - 1 - i ] = val;
							}

#ifdef _DEBUG
							std::cout << "MatrixFileIterator::operator++ "
								": buffer at index "
								  << i << " now contains " << val << "\n";
#endif
						}

						if( buffer_size == i ){
							++buffercount;
						}

						// store new buffer position
						if( i > 0 ) {
							pos = i - 1;
						} else {
							assert( ended );
						}
						// store new stream position
						spos = infile.tellg();
					} else {
						// simply increment and done
						--pos;
					}

					// done
					return *this;
				}

				/** Standard dereferencing of iterator. */
				const OutputType & operator*() {
					if( started ) {
						preprocess();
						started = false;
						(void)operator++();
					}
					if( ended ) {
						throw std::runtime_error(
							"Attempt to dereference (via operator*) "
							"MatrixFileIterator in end "
							"position." );
					}
					return buffer[ pos ];
				}

				/** Standard pointer request of iterator. */
				const OutputType * operator->() {
					if( started ) {
						preprocess();
						started = false;
						(void)operator++();
					}
					if( ended ) {
						throw std::runtime_error( "Attempt to dereference (via "
												  "operator->) "
												  "MatrixFileIterator in end "
												  "position." );
					}
					return &( buffer[ pos ] );
				}

				/** Returns the current row index. */
				const S & j() const {
					if( started ) {
						const_cast< MatrixFileIterator< T > * >( this )->preprocess();
						const_cast< MatrixFileIterator< T > * >( this )->started = false;
						(void)const_cast< MatrixFileIterator< T > * >( this )->operator++();
					}
					if( ended ) {
						throw std::runtime_error( "Attempt to dereference (via "
									  "operator*) "
									  "MatrixFileIterator in end "
									  "position." );
					}
					size_t I = buffercount * buffer_size - 1 - pos;

					if(
						properties._symmetry == MatrixFileProperties::MMsymmetries::SYMMETRIC ||
						properties._symmetry == MatrixFileProperties::MMsymmetries::HERMITIAN
					) {
						throw std::runtime_error(
							"Not implemented i,j: SYMMETRIC & HERMITIAN."
						);
					} else if (
						properties._symmetry == MatrixFileProperties::MMsymmetries::SKEWSYMMETRIC
					) {
						throw std::runtime_error(
							"Not implemented i,j: SKEWSYMMETRIC."
						);
					} else if (
						properties._symmetry == MatrixFileProperties::MMsymmetries::GENERAL
					) {
						I = I / properties._m;
					} else {
						throw std::runtime_error(
							"Unknown Matrix Market format."
						);
					}

					const_cast< MatrixFileIterator< T > * >( this )->icount = I;
					return icount;
				}

				/** Returns the current column index. */
				const S & i() const {
					if( started ) {
						const_cast< MatrixFileIterator< T > * >( this )->preprocess();
						const_cast< MatrixFileIterator< T > * >( this )->started = false;
						(void)const_cast< MatrixFileIterator< T > * >( this )->operator++();
					}
					if( ended ) {
						throw std::runtime_error( "Attempt to dereference (via "
									  "operator*) "
									  "MatrixFileIterator in end "
									  "position." );
					}
					size_t I = buffercount * buffer_size - 1 - pos;

					if(
						properties._symmetry == MatrixFileProperties::MMsymmetries::SYMMETRIC ||
						properties._symmetry == MatrixFileProperties::MMsymmetries::HERMITIAN
					) {
						throw std::runtime_error(
							"Not implemented i,j: SYMMETRIC & HERMITIAN."
						);
					} else if (
						properties._symmetry == MatrixFileProperties::MMsymmetries::SKEWSYMMETRIC
					) {
						throw std::runtime_error(
							"Not implemented i,j: SKEWSYMMETRIC."
						);
					} else if (
						properties._symmetry == MatrixFileProperties::MMsymmetries::GENERAL
					) {
						I = I % properties._m;
					} else {
						throw std::runtime_error(
							"Unknown Matrix Market format."
						);
					}


					const_cast< MatrixFileIterator< T > * >( this )->jcount = I;
					return jcount;
				}

				/** Returns the current nonzero value. */
				const T & v() const {
					if( started ) {
						const_cast< MatrixFileIterator< T > * >( this )->preprocess();
						const_cast< MatrixFileIterator< T > * >( this )->started = false;
						(void)const_cast< MatrixFileIterator< T > * >( this )->operator++();
					}
					if( ended ) {
						throw std::runtime_error( "Attempt to dereference (via "
												  "operator*) "
												  "MatrixFileIterator in end "
												  "position." );
					}

					return buffer[ pos ];
				}
			};


		} // namespace internal
	}     // namespace utils
} // namespace alp

#endif // end ``_H_MATRIXFILEITERATOR''
