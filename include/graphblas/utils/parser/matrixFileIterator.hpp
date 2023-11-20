
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

#ifndef _H_GRB_UTILS_MATRIXFILEITERATOR
#define _H_GRB_UTILS_MATRIXFILEITERATOR

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

#include <graphblas/config.hpp>
#include <graphblas/iomode.hpp>
#include <graphblas/spmd.hpp>

#include <graphblas/utils/config.hpp>
#include <graphblas/utils/hpparser.h>
#include <graphblas/utils/iscomplex.hpp>

#include "matrixFileProperties.hpp"

#ifdef _GRB_WITH_OMP
 #include <graphblas/omp/config.hpp>
#endif


namespace grb {

	namespace utils {

		namespace internal {

			/**
			 * An iterator over a matrix file.
			 *
			 * Handles both Matrix Market and SNAP formats.
			 *
			 * @tparam S The index type.
			 * @tparam T The value type.
			 */
			template< typename S, typename T >
			class MatrixFileIterator {

				template< typename X1, typename X2 >
				friend std::ostream & operator<<(
					std::ostream &, const MatrixFileIterator< X1, X2 > & );

				private:

					/** The output type of the base iterator. */
					typedef std::pair< std::pair< S, S >, T > OutputType;

					/**
					 * Iterators will retrieve this many lines at a time from the input
					 * file.
					 */
					static constexpr size_t buffer_size = grb::config::PARSER::bsize();

					/** The nonzero buffer. */
					OutputType * buffer;

					/** The underlying MatrixReader. */
					MatrixFileProperties &properties;

					/** The input stream. */
					std::ifstream infile;

					/** The input stream position. */
					std::streampos spos;

					/** The current position in the buffer. */
					size_t pos;

					/** Whether the \a infile stream \em and \a buffer have been depleted. */
					bool ended;

					/**
					 * Whether the first fill of the buffer is held until the first
					 * dereference of this iterator is taking place.
					 */
					bool started;

					/** Whether the smmetric counterpart of the current nonzero was output. */
					bool symmetricOut;

					/** A function to apply to convert input values on the fly. */
					std::function< void( T & ) > converter;

					/**
					 * In case we are reading from a pattern matrix, which value should
					 * substitute a nonzero.
					 */
					const T patternValue;

					/**
					 * Strips comments and possible MatrixMarket header from input stream
					 * start.
					 */
					void preprocess() {
#ifdef _DEBUG
						std::cout << "\t In MatrixFileIterator::preprocess()\n";
#endif
						// check if first header indicates MatrixMarket
						const std::streampos start = infile.tellg();
						// assume input is matrix market until we detect otherwise
						bool mmfile = true;
						// try and parse header
						std::string mmheader;
						if( !std::getline( infile, mmheader ) ) {
							// some type of error occurred-- rewind and let a non-mmfile parse try
							mmfile = false;
							(void) infile.seekg( start );
						}
						if( mmfile && ( mmheader.size() < 14 ||
							mmheader.substr( 0, 14 ) != "%%MatrixMarket"
						) ) {
							// some type of error occurred-- rewind and let a non-mmfile parse try
							mmfile = false;
							(void) infile.seekg( start );
						}
						// ignore all comment lines
						char peek = infile.peek();
						while( infile.good() && ( peek == '%' || peek == '#' ) ) {
							(void) infile.ignore( std::numeric_limits< std::streamsize >::max(),
								'\n' );
							peek = infile.peek();
						}
						// ignore non-comment matrix market header if we expect one
						if( mmfile ) {
							std::string ignore;
							std::getline( infile, ignore );
#ifdef _DEBUG
							std::cout << "\t\t Assuming MatrixMarket file with header line "
								<< ignore << "\n";
#endif
						} else {
							mmheader.clear();
#ifdef _DEBUG
							std::cout << "\t\t Assuming non-MatrixMarket format\n";
#endif
						}
						// done
					}

#ifdef _DEBUG
					void printIteratorState() const {
						std::cout << "\t buffer = " << (static_cast<const void*>(buffer)) << "\n"
							<< "\t properties = " << (static_cast<const void*>(&properties)) << "\n"
							<< "\t infile is " << (infile.is_open() ? "open, stream is " : "closed" )
							<< (infile.is_open() ? (infile.good() ? "OK" : "not OK") : "") << "\n"
							<< "\t spos = " << spos << "\n"
							<< "\t pos = " << pos << "\n"
							<< "\t ended = " << ended << "\n"
							<< "\t started = " << started << "\n"
							<< "\t symmetricOut = " << symmetricOut << "\n"
							<< "\t converter value and pointer supressed\n"
							<< "\t patternValue = " << patternValue << "\n";
					}
#endif


				public:

					// standard STL iterator typedefs
					typedef std::ptrdiff_t difference_type;
					typedef OutputType value_type;
					typedef OutputType & reference;
					typedef OutputType * pointer;
					typedef std::forward_iterator_tag iterator_category;

					// standard GraphBLAS iterator typedefs
					typedef S RowIndexType;
					typedef S ColumnIndexType;
					typedef T ValueType;

					/** Base constructor, starts in begin position. */
					MatrixFileIterator(
						MatrixFileProperties &prop, IOMode mode,
						const std::function< void( T & ) > valueConverter,
						const T &patternVal,
						const bool end = false
					) : buffer( nullptr ), properties( prop ), infile( properties._fn ),
						spos(), pos( 0 ), ended( end ), started( !end ),
						symmetricOut( prop._symmetric ? true : false ),
						converter( valueConverter ), patternValue( patternVal )
					{
						if( mode != SEQUENTIAL ) {
							throw std::runtime_error( "Only sequential IO is supported by this "
								"iterator at present, sorry."
							); // internal issue #48
						}
					}

					/** Copy constructor. */
					MatrixFileIterator( const MatrixFileIterator< S, T > &other ) :
						buffer( nullptr ), properties( other.properties ),
						infile( properties._fn ), spos( other.spos ), pos( other.pos ),
						ended( other.ended ), started( other.started ),
						symmetricOut( other.symmetricOut ), converter( other.converter ),
						patternValue( other.patternValue )
					{
#ifdef _DEBUG
						std::cout << "In MatrixFileIterator copy-constructor, "
							<< "non pattern variant\n";
#endif
						// set latest stream position
						(void) infile.seekg( spos );
						// if buffer is nonempty
						if( pos > 0 ) {
							// allocate buffer
							if( posix_memalign(
									(void **)&buffer,
									config::CACHE_LINE_SIZE::value(),
									buffer_size * sizeof( OutputType )
								) != 0
							) {
								buffer = nullptr;
								throw std::runtime_error( "Error during allocation of internal "
									"iterator memory." );
							}
							// copy any remaining buffer contents
							for( size_t i = 0; i <= pos; ++i ) {
								buffer[ i ] = other.buffer[ i ];
							}
						}
#ifdef _DEBUG
						printIteratorState();
#endif
					}

					/** Base destructor. */
					~MatrixFileIterator() {
#ifdef _DEBUG
						std::cout << "In MatrixFileIterator destructor, "
							<< "non-pattern variant\n";
						printIteratorState();
#endif
						if( buffer != nullptr ) {
							free( buffer );
						}
					}

					/** Copies an iterator state. */
					MatrixFileIterator & operator=( const MatrixFileIterator &x ) {
#ifdef _DEBUG
						std::cout << "In MatrixFileIterator::operator=(other), "
							<< "non pattern variant\n"
							<< "State before copy-assignment:\n";
						printIteratorState();
#endif
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
						(void) infile.seekg( x.spos );
						spos = x.spos;
						pos = x.pos;
						// copy any remaining buffer contents
						if( pos > 0 ) {
							// check if buffer is allocated
							if( buffer == nullptr ) {
								// no, so allocate buffer
								if( posix_memalign(
										(void **) &buffer,
										config::CACHE_LINE_SIZE::value(),
										buffer_size * sizeof( OutputType )
									) != 0
								) {
									buffer = nullptr;
									throw std::runtime_error( "Error during allocation of internal "
										"iterator memory." );
								}
							}
							// copy remote buffer contents
							assert( pos < buffer_size );
							for( size_t i = 0; i <= pos; ++i ) {
								buffer[ i ] = x.buffer[ i ];
							}
						}
						// copy symmetry state
						symmetricOut = x.symmetricOut;
#ifdef _DEBUG
						std::cout << "State after copy-assignment:\n";
						printIteratorState();
#endif
						// done
						return *this;
					}

					/** Standard check for equality. */
					bool operator==( const MatrixFileIterator< S, T > &x ) const {
#ifdef _DEBUG
						std::cout << "In MatrixFileIterator::operator==(), "
							<< "non-pattern variant\n";
#endif
						// it could be that iterators in start position are actually pointing to
						// an end position. Therefore, we first ensure that iterators are properly
						// started.
						if( started ) {
#ifdef _DEBUG
							std::cout << "\t starting this iterator. Current state:\n";
							printIteratorState();
#endif
							const_cast< MatrixFileIterator< S, T > * >( this )->preprocess();
							const_cast< MatrixFileIterator< S, T > * >( this )->started = false;
							(void)const_cast< MatrixFileIterator< S, T > * >( this )->operator++();
#ifdef _DEBUG
							std::cout << "\t State after starting:\n";
							printIteratorState();
#endif
						}
						if( x.started ) {
#ifdef _DEBUG
							std::cout << "\t starting other iterator\n";
#endif
							const_cast< MatrixFileIterator< S, T > * >( &x )->preprocess();
							const_cast< MatrixFileIterator< S, T > * >( &x )->started = false;
							(void)const_cast< MatrixFileIterator< S, T > * >( &x )->operator++();
						}
						// check if both are in end position
						if( ended && x.ended ) {
							return true;
						}
#ifndef NDEBUG
						if( properties._fn != x.properties._fn ) {
							std::cerr << "Warning: comparing two instances of MatrixFileIterator "
								"that are 1) nonempty *and* 2) not reading from the same file.\n";
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
						return !(operator==( x ));
					}

					// this assumes full triplet data
					MatrixFileIterator & operator++() {
#ifdef _DEBUG
						std::cout << "MatrixFileIterator::operator++() called, "
							<< "non-pattern variant\n";
#endif
						// if ended, stop
						if( ended ) {
							return *this;
						}
						// if this is the first function call on this iterator, call preprocess first
						if( started ) {
#ifdef _DEBUG
							std::cout << "\t iterator was not yet started-- doing so now\n";
#endif
							preprocess();
							started = false;
							(void) operator++();
						}
						// if symmtric and not given output yet and not diagonal
						if( properties._symmetric ) {
#ifdef _DEBUG
							std::cout << "\t matrix is symmetric --";
#endif
							// toggle symmetricOut
							symmetricOut = !symmetricOut;
							// if we are giving symmetric output now
							if( symmetricOut ) {
								// make symmetric pair & exit if current nonzero is not diagonal
								if( buffer[ pos ].first.first != buffer[ pos ].first.second ) {
#ifdef _DEBUG
									std::cout << " generating symmetric entry now\n";
#endif
									std::swap( buffer[ pos ].first.first, buffer[ pos ].first.second );
									if( properties._symmetric == Hermitian ) {
										buffer[ pos ].second = grb::utils::is_complex< T >::conjugate(
											buffer[ pos ].second );
									}
									return *this;
								} else {
									// if diagonal, reset symmetricOut and continue normal path
									symmetricOut = false;
								}
							}
#ifdef _DEBUG
							if( !symmetricOut ) {
								std::cout << " not generating symmetric entry yet\n";
							}
#endif
						}
						// check if we need to parse from infile
						if( pos == 0 ) {
#ifdef _DEBUG
							std::cout << "\t starting parse from input file\n";
#endif
							// try and parse buffer_size new values
							size_t i = 0;
							if( !infile.good() ) {
#ifdef _DEBUG
								std::cout << "\t\t infile stream was not OK-- "
									<< "setting iterator to end position\n";
#endif
								ended = true;
							}
							// check if buffer is allocated
							if( buffer == nullptr ) {
								// no, so allocate buffer
								if( posix_memalign(
										(void **) &buffer,
										config::CACHE_LINE_SIZE::value(),
										buffer_size * sizeof( OutputType )
									) != 0
								) {
									buffer = nullptr;
									throw std::runtime_error( "Error during allocation of internal "
										"iterator memory." );
								}
							}
							// bring if on pattern to the front in an attempt to speed up compiled parse code
							if( properties._pattern ) {
								for( ; !ended && i < buffer_size; ++i ) {
									S row, col;
									if( !(infile >> row >> col) ) {
										if( i == 0 ) {
											ended = true;
										}
										break;
									}
									// correct 1-base input if necessary
									if( properties._oneBased ) {
										assert( row > 0 );
										assert( col > 0 );
										(void) --row;
										(void) --col;
									}
									// if indirect, translate
									if( !properties._direct ) {
										// find row index
										const auto rit = properties._row_map.find( row );
										if( rit == properties._row_map.end() ) {
											const size_t new_index = properties._row_map.size();
											properties._row_map[ row ] = new_index;
											row = new_index;
										} else {
											row = rit->second;
										}
										if( properties._symmetricmap ) {
											const auto cit = properties._row_map.find( col );
											if( cit == properties._row_map.end() ) {
												const size_t new_index = properties._row_map.size();
												properties._row_map[ col ] = new_index;
												col = new_index;
											} else {
												col = cit->second;
											}
										} else {
											const auto cit = properties._col_map.find( col );
											if( cit == properties._col_map.end() ) {
												const size_t new_index = properties._col_map.size();
												properties._col_map[ col ] = new_index;
												col = new_index;
											} else {
												col = cit->second;
											}
										}
									}
									// store (corrected) values
									buffer[ i ].first.first = row;
									buffer[ i ].first.second = col;
									buffer[ i ].second = patternValue;
#ifdef _DEBUG
									std::cout << "\t storing pattern value\n";
#endif
								}
							} else {
								// non-pattern matrices
								for( ; !ended && i < buffer_size; ++i ) {
									S row, col;
									T val;
									bool error = false;
									if( properties._type == MatrixFileProperties::Type::MATRIX_MARKET &&
										properties._complex
									) {
#ifdef _DEBUG
										std::cout << "\t attempting complex value parse\n";
#endif
										typename is_complex< T >::type re, im;
										error = !(infile >> row >> col >> re >> im);
										if( !error ) {
											std::stringstream oss;
											oss << "(" << re << "," << im << ")";
											error = !(oss >> val);
										}
									} else {
#ifdef _DEBUG
										std::cout << "\t attempting regular value parse\n";
#endif
										error = !(infile >> row >> col >> val);
									}
									if( error ) {
										if( i == 0 ) {
#ifdef _DEBUG
											std::cout << "\t parse error detected at first position-- "
												<< "setting iterator to end position\n";
#endif
											ended = true;
										}
										break;
									} else {
#ifdef _DEBUG
										T temp = val;
										converter( temp );
										std::cout << "\t parsed line ``" << row << " " << col << " " << val << "'', "
											<< "with value after conversion " << temp << "\n";
#endif
										// convert value
										converter( val );
										// store read values
										buffer[ i ].second = val;
									}
									// correct 1-base input if necessary
									if( properties._oneBased ) {
										assert( row > 0 );
										assert( col > 0 );
										(void) --row;
										(void) --col;
									}
									// if indirect, translate
									if( !properties._direct ) {
										// find row index
										const auto rit = properties._row_map.find( row );
										if( rit == properties._row_map.end() ) {
											const size_t new_index = properties._row_map.size();
											properties._row_map[ row ] = new_index;
											row = new_index;
										} else {
											row = rit->second;
										}
										if( properties._symmetricmap ) {
											const auto cit = properties._row_map.find( col );
											if( cit == properties._row_map.end() ) {
												const size_t new_index = properties._row_map.size();
												properties._row_map[ col ] = new_index;
												col = new_index;
											} else {
												col = cit->second;
											}
										} else {
											const auto cit = properties._col_map.find( col );
											if( cit == properties._col_map.end() ) {
												const size_t new_index = properties._col_map.size();
												properties._col_map[ col ] = new_index;
												col = new_index;
											} else {
												col = cit->second;
											}
										}
#ifdef _DEBUG
										std::cout << "MatrixFileIterator::operator++ (non-pattern, indirect "
											<< "variant): mapped row and col to " << row << " and " << col
											<< ", resp.\n";
#endif
									}
									// store (corrected) values
									buffer[ i ].first.first = row;
									buffer[ i ].first.second = col;
#ifdef _DEBUG
									std::cout << "\t buffer at index " << i << " now contains: "
										<< row << ", " << col << ", " << val << "\n";
#endif
								}
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
							(void) --pos;
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
							throw std::runtime_error( "Attempt to dereference (via operator*) "
								"MatrixFileIterator in end position." );
						}
						return buffer[ pos ];
					}

					/** Standard pointer request of iterator. */
					const OutputType * operator->() {
						if( started ) {
							preprocess();
							started = false;
							(void) operator++();
						}
						if( ended ) {
							throw std::runtime_error( "Attempt to dereference (via operator->) "
								"MatrixFileIterator in end position." );
						}
						return &( buffer[ pos ] );
					}

					/** Returns the current row index. */
					const S & i() const {
#ifdef _DEBUG
						std::cout << "MatrixFileIterator::i() called. State at entry:\n";
						printIteratorState();
#endif
						if( started ) {
#ifdef _DEBUG
							std::cout << "\t starting iterator\n";
#endif
							const_cast< MatrixFileIterator< S, T > * >( this )->preprocess();
							const_cast< MatrixFileIterator< S, T > * >( this )->started = false;
							(void)const_cast< MatrixFileIterator< S, T > * >( this )->operator++();
#ifdef _DEBUG
							std::cout << "\t state after preprocess and operator++:\n";
							printIteratorState();
#endif
						}
						if( ended ) {
							throw std::runtime_error( "Attempt to dereference (via i()) "
								"MatrixFileIterator in end position." );
						}
#ifdef _DEBUG
						std::cout << "\t returning row index " << buffer[ pos ].first.first << "\n";
#endif
						return buffer[ pos ].first.first;
					}

					/** Returns the current column index. */
					const S & j() const {
						if( started ) {
							const_cast< MatrixFileIterator< S, T > * >( this )->preprocess();
							const_cast< MatrixFileIterator< S, T > * >( this )->started = false;
							(void)const_cast< MatrixFileIterator< S, T > * >( this )->operator++();
						}
						if( ended ) {
							throw std::runtime_error( "Attempt to dereference (via j()) "
								"MatrixFileIterator in end position." );
						}
						return buffer[ pos ].first.second;
					}

					/** Returns the current nonzero value. */
					const T & v() const {
						if( started ) {
							const_cast< MatrixFileIterator< S, T > * >( this )->preprocess();
							const_cast< MatrixFileIterator< S, T > * >( this )->started = false;
							(void)const_cast< MatrixFileIterator< S, T > * >( this )->operator++();
						}
						if( ended ) {
							throw std::runtime_error( "Attempt to dereference (via j()) "
								"MatrixFileIterator in end position." );
						}
						return buffer[ pos ].second;
					}
				};

			/** An iterator over nonzeroes in the matrix file. Pattern matrix specialisation. */
			template< typename S >
			class MatrixFileIterator< S, void > {

				template< typename X1 >
				friend std::ostream & operator<<( std::ostream &, const MatrixFileIterator< X1, void > & );

			public:
				typedef typename std::pair< S, S > Coordinates;

			private:
				mutable size_t * row;
				mutable size_t * col;
				mutable size_t pos;
				mutable Coordinates coordinates;
				bool symmetricOut;
				mutable void * hpparser;
				mutable size_t incs;
				mutable bool started;
				bool ended;
				MatrixFileProperties properties;
				IOMode mode;

				static constexpr size_t buffer_length = config::PARSER::bsize() / 2 / sizeof( size_t );
				static_assert( buffer_length > 0, "Please increase grb::config::PARSER::bsize()" );

				/** Allocates buffer. */
				void allocate() const {
					// convenience fields
					constexpr size_t alignment = config::CACHE_LINE_SIZE::value();
					constexpr size_t bufferSize = config::PARSER::bsize();
					constexpr size_t chunk = alignment > sizeof( size_t ) ? alignment : sizeof( size_t );
					// allocate buffer space
					if( posix_memalign( (void **)&row, alignment, bufferSize + chunk ) != 0 ) {
						throw std::runtime_error( "Error during allocation of "
												  "internal iterator memory." );
					}
					// set col buffer pointer
					col = reinterpret_cast< size_t * >( reinterpret_cast< char * >( row ) + static_cast< size_t >( ( bufferSize / 2 ) / chunk ) * chunk + chunk );
					// set new position
					pos = 0;
				}

				/**
				 * Copies the state of another buffer.
				 *
				 * This function only copies the state of #hpparser, #row, #col, and #pos;
				 * all other fields must be set by the caller.
				 */
				void copyState( const MatrixFileIterator< S, void > &other ) {
					// copy underlying parser
					if( other.hpparser == nullptr ||
						TprdCopy( other.hpparser, &hpparser ) != APL_SUCCESS
					) {
						throw std::runtime_error( "Could not copy underlying hpparser." );
					}
					// allocate buffer
					if( row == nullptr ) {
						allocate();
					}
					// copy buffer contents
					assert( other.pos < buffer_length );
					if( other.pos > 0 ) {
						(void) memcpy( row, other.row, other.pos );
						(void) memcpy( col, other.col, other.pos );
					}
					// set buffer position
					pos = other.pos;
					incs = other.incs;
					// done
				}

				/**
				 * Updates coordinates according to current position.
				 * Also updates row- and/or column- maps if requested.
				 * Caller must ensure this does not result in bad memory access.
				 */
				void updateCoordinates() {
					// sanity check
					assert( pos < buffer_length );
					// update coordinates
					size_t row = this->row[ pos ];
					size_t col = this->col[ pos ];
					// correct base
					if( properties._oneBased ) {
						assert( row > 0 );
						assert( col > 0 );
						(void)--row;
						(void)--col;
					}
					// update row map
					if( ! properties._direct ) {
						// do row translation
						const auto rit = properties._row_map.find( row );
						if( rit == properties._row_map.end() ) {
							const size_t new_index = properties._row_map.size();
							properties._row_map[ row ] = new_index;
							// std::cout << "MatrixFileIterator: Added new row index " << row << "...\n";
							row = new_index;
						} else {
							row = rit->second;
						}
						// do col translation
						if( properties._symmetricmap ) {
							// symmetric map, so use row map
							const auto cit = properties._row_map.find( col );
							if( cit == properties._row_map.end() ) {
								const size_t new_index = properties._row_map.size();
								properties._row_map[ col ] = new_index;
								col = new_index;
							} else {
								col = cit->second;
							}
						} else {
							// map is not symmetric, so use dedicated col map
							const auto cit = properties._col_map.find( col );
							if( cit == properties._col_map.end() ) {
								const size_t new_index = properties._col_map.size();
								properties._col_map[ col ] = new_index;
								col = new_index;
							} else {
								col = cit->second;
							}
						}
					}
					// update coordinates
					coordinates.first = row;
					coordinates.second = col;
				}

				/**
				 * Sets the iterator in started position.
				 *
				 * This function is declared const because it only affects non-public
				 * buffers; that is, the public behaviour of this instance is const.
				 */
				void start() const {
#ifdef _DEBUG
					std::cout << "MatrixFileIterator: " << this << " is starting up. Target file: " << properties._fn << "\n"; // DBG
#endif
					// cache SPMD info
					const size_t P = mode == SEQUENTIAL ? 1 : spmd<>::nprocs();
					const size_t s = mode == SEQUENTIAL ? 0 : spmd<>::pid();
					// sanity checks
					assert( hpparser == NULL );
					assert( P > 0 );
					assert( s < P );
#ifdef _GRB_WITH_OMP
					const auto num_threads = config::OMP::threads();
#else
					const unsigned int num_threads = 1;
#endif
					// start the hpparser
					if( properties._type == MatrixFileProperties::Type::MATRIX_MARKET ) {
						// if matrix market, signal to hpparser to skip first header line by passing non-NULL value for row, col, and nnz
						size_t row, col, nnz;
						if( ReadEdgeBegin( properties._fn.c_str(), config::PARSER::read_bsize(), P, num_threads, s, &row, &col, &nnz, &hpparser ) != APL_SUCCESS ) {
							throw std::runtime_error( "Could not create "
													  "hpparser." );
						}
					} else {
						// if snap, no need to pass non-NULL row, col, and nnz
						assert( properties._type == MatrixFileProperties::Type::SNAP );
						if( ReadEdgeBegin( properties._fn.c_str(), config::PARSER::read_bsize(), P, num_threads, s, NULL, NULL, NULL, &hpparser ) != APL_SUCCESS ) {
							throw std::runtime_error( "Could not create "
													  "hpparser." );
						}
					}
					// check if buffer is allocated
					if( row == NULL ) {
						allocate();
					}
					// done
					started = true;
				}

			public:
				// standard STL iterator typedefs
				typedef ptrdiff_t difference_type;
				typedef Coordinates value_type;
				typedef Coordinates & reference;
				typedef Coordinates * pointer;
				typedef std::forward_iterator_tag iterator_category;

				// standard GraphBLAS iterator typedefs
				typedef S RowIndexType;
				typedef S ColumnIndexType;
				typedef void ValueType;

				/** Base constructor, starts in begin position. */
				MatrixFileIterator( MatrixFileProperties & props, IOMode mode_in, const bool end = false ) :
					row( NULL ), col( NULL ), pos( -1 ), symmetricOut( props._symmetric ? true : false ), hpparser( NULL ), incs( 0 ), started( false ), ended( end ), properties( props ),
					mode( mode_in ) {}

				/** Copy constructor. */
				MatrixFileIterator( const MatrixFileIterator< S, void > & other ) :
					row( NULL ), col( NULL ), pos( -1 ), // these correspond to state
					symmetricOut( other.symmetricOut ), hpparser( NULL ), incs( 0 ), started( other.started ), ended( other.ended ), properties( other.properties ), mode( other.mode ) {
#ifdef _DEBUG
					// std::cout << "MatrixFileIterator: copy constructor called on " << this << "\n"; //DBG
#endif
					// if we have state
					if( started ) {
						// copy the state
						copyState( other );
					}
				}

				/** Base destructor. */
				~MatrixFileIterator() {
#ifdef _DEBUG
					// std::cout << "MatrixFileIterator: destructor called on " << this << "\n"; //DBG
#endif
					if( hpparser != NULL ) {
						ReadEdgeEnd( hpparser );
					}
					if( row != NULL ) {
						free( row );
					}
					row = col = NULL;
					started = false;
				}

				/** Copies an iterator state. */
				MatrixFileIterator & operator=( const MatrixFileIterator< S, void > &other ) {
#ifdef _DEBUG
					std::cout << "MatrixFileIterator: assignment operator called on "
						<< this << "\n";
#endif
					// if I already had an hpparser open, I should close it
					if( hpparser != nullptr ) {
						if( ReadEdgeEnd( hpparser ) != APL_SUCCESS ) {
							throw std::runtime_error(
								"Could not properly destroy hpparser instance."
							);
						}
						hpparser = nullptr;
					}
					// copy static fields
					coordinates = other.coordinates;
					symmetricOut = other.symmetricOut;
					started = other.started;
					ended = other.ended;
					properties = other.properties;
					mode = other.mode;
					// if started, copy hpparser and buffer state
					if( started ) {
						// copy the state of the underlying parser and the iterator buffer
						copyState( other );
						// this copies the following fields:
						//  - hpparser
						//  - row
						//  - col
						//  - pos
						//  - incs
					}
					// done
					return *this;
				}

				/** Standard check for equality. */
				bool operator==( const MatrixFileIterator & x ) const {
#ifdef _DEBUG
					// std::cout << "MatrixFileIterator: equals operator called on " << this << "\n"; //DBG
#endif
					// sanity check against UB
					assert( properties._fn == x.properties._fn );
					assert( mode == x.mode );

					// check for mismatching end positions
					if( ended == x.ended ) {
						// check if both are in end position
						if( ended && x.ended ) {
							return true;
						}
					} else {
						// not matching means never equal
						return false;
					}

					// check if both are in new position
					if( ! started && ! ( x.started ) ) {
						return true;
					}

					// otherwise, only can compare equal if readEdge was called equally many times
					if( incs == x.incs ) {
						// AND in the same buffer position
						return pos == x.pos;
					}

					// otherwise, not equal
					return false;
				}

				/** Standard check for inequality, relies on equality check. */
				bool operator!=( const MatrixFileIterator & x ) const {
#ifdef _DEBUG
					// std::cout << "MatrixFileIterator: not-equals operator called on " << this << "\n"; //DBG
#endif
					return ! ( operator==( x ) );
				}

				/** Increments the iterator. Checks for new position first-- if yes, calls #start. */
				MatrixFileIterator & operator++() {
#ifdef _DEBUG
					std::cout << "MatrixFileIterator: increment operator "
								 "called on "
							  << this << "\n"; // DBG
#endif
					// sanity checks
					assert( ! ended );

					// if this is the first function call on this iterator, open hpparser first
					if( ! started ) {
						assert( hpparser == NULL );
						start();
					}

					// if symmtric and not given output yet and not diagonal
					if( properties._symmetric ) {
#ifdef _DEBUG
						std::cout << "MatrixFileIterator: operator++ is "
									 "symmetric\n"; // DBG
#endif
						// toggle symmetricOut
						symmetricOut = ! symmetricOut;
						// if we are giving symmetric output now
						if( symmetricOut ) {
							// make symmetric pair & exit if current nonzero is not diagonal
							if( row[ pos ] != col[ pos ] ) {
								std::swap( row[ pos ], col[ pos ] );
								updateCoordinates();
								return *this;
							} else {
								// if diagonal, reset symmetricOut and continue normal path
								symmetricOut = false;
							}
						}
					}

					// check if we need to parse from infile
					if( pos == 0 ) {
						// expected number of nonzeroes
						size_t nnzsToRead = buffer_length;
#ifdef _DEBUG
						std::cout << "MatrixFileIterator: preparing to read up "
									 "to "
								  << nnzsToRead << " nonzeroes.\n"; // DBG
#endif
						// call hpparser
						if( ReadEdge( hpparser, &nnzsToRead, row, col ) != APL_SUCCESS ) {
							throw std::runtime_error( "Error while parsing "
													  "file." );
						}
#ifdef _DEBUG
						std::cout << "DBG: read " << nnzsToRead << " nonzeroes.\n"; // DBG
#endif
						// increment incs, set new pos
						if( nnzsToRead > 0 ) {
							++incs;
							pos = nnzsToRead - 1;
						} else {
#ifdef _DEBUG
							std::cout << "DBG: ended flag set on iterator " << this << ".\n"; // DBG
#endif
							ended = true;
						}
					} else {
						// simply increment
						--pos;
					}

					// re-bind coordinates
					if( started && ! ended ) {
						updateCoordinates();
					}

					// done
					return *this;
				}

				/** Standard dereferencing of iterator. */
				const Coordinates & operator*() {
#ifdef _DEBUG
					// std::cout << "MatrixFileIterator: star dereference operator called on " << this << "\n"; //DBG
#endif
					if( ended ) {
						throw std::runtime_error( "Attempt to dereference (via "
												  "operator*) "
												  "MatrixFileIterator in end "
												  "position." );
					}
					if( ! started ) {
						// start the iterator
						assert( hpparser == NULL );
						start();
						// and fill the buffer
						this->operator++();
					}
					return coordinates;
				}

				/** Standard pointer request of iterator. */
				const Coordinates * operator->() {
#ifdef _DEBUG
					// std::cout << "MatrixFileIterator: arrow dereference operator called on " << this << "\n"; //DBG
#endif
					if( ended ) {
						throw std::runtime_error( "Attempt to dereference (via "
												  "operator->) "
												  "MatrixFileIterator in end "
												  "position." );
					}
					if( ! started ) {
						// start the iterator
						assert( hpparser == NULL );
						start();
						// and fill the buffer
						this->operator++();
					}
					return coordinates;
				}

				/** Returns the current row index. */
				const S & i() const {
#ifdef _DEBUG
					// std::cout << "MatrixFileIterator: row dereference operator called on " << this << "\n";
#endif
					if( ended ) {
						throw std::runtime_error( "Attempt to dereference (via "
												  "i()) MatrixFileIterator in "
												  "end position." );
					}
					if( ! started ) {
						// start the iterator
						assert( hpparser == NULL );
						start();
						// and fill the buffer
						const_cast< MatrixFileIterator< S, void > * >( this )->operator++();
					}
					return coordinates.first;
				}

				/** Returns the current column index. */
				const S & j() const {
#ifdef _DEBUG
					// std::cout << "MatrixFileIterator: column dereference operator called on " << this << "\n";
#endif
					if( ended ) {
						throw std::runtime_error( "Attempt to dereference (via "
												  "j()) MatrixFileIterator in "
												  "end position." );
					}
					if( ! started ) {
						// start the iterator
						assert( hpparser == NULL );
						start();
						// and fill the buffer
						const_cast< MatrixFileIterator< S, void > * >( this )->operator++();
					}
					return coordinates.second;
				}
			};

			/** Prints iterator \a it to output stream \a os. */
			template< typename S >
			std::ostream & operator<<( std::ostream & os, const MatrixFileIterator< S, void > & it ) {
				if( ! it.started ) {
					assert( it.hpparser == NULL );
					it.start();
					(void)const_cast< MatrixFileIterator< S, void > * >( &it )->operator++();
				}
				if( it.ended ) {
					os << "iterator in end position";
				} else {
					os << it.coordinates.first << ", " << it.coordinats.second;
				}
				return os;
			}

			/** Prints this iterator to an output stream. */
			template< typename S, typename T >
			std::ostream & operator<<( std::ostream & os, const MatrixFileIterator< S, T > & it ) {
				if( it.started ) {
					const_cast< MatrixFileIterator< S, T > * >( &it )->preprocess();
					const_cast< MatrixFileIterator< S, T > * >( &it )->started = false;
					(void)const_cast< MatrixFileIterator< S, T > * >( &it )->operator++();
				}
				if( it.ended ) {
					os << "iterator in end position";
				} else {
					os << it.buffer[ it.pos ].first.first << ", " << it.buffer[ it.pos ].first.second << ", " << it.buffer[ it.pos ].second;
				}
				return os;
			}

		} // namespace internal
	}     // namespace utils
} // namespace grb

#endif // end ``_H_GRB_UTILS_MATRIXFILEITERATOR''

