
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

namespace alp {
	namespace utils {
		namespace internal {

			template< typename T, bool data_reflect = false,  typename S = size_t >
			class MatrixFileIterator {

			private:
				/** The output type of the base iterator. */
				typedef T OutputType;

				/** The underlying MatrixReader. */
				MatrixFileProperties & properties;

				/** The input stream. */
				std::ifstream infile;

				/** The input stream position. */
				std::streampos spos;

				/** The curent value */
				OutputType val;

				/** The i index counter. */
				size_t colidx;

				/** The j index counter. */
				size_t rowidx;

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
					properties( prop ), infile( properties._fn ), spos(),
					colidx( 0 ), rowidx( 0 ), ended( end ),
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
					properties( other.properties ), infile( properties._fn ), spos( other.spos ),
					colidx( other.colidx ), rowidx( other.rowidx ), ended( other.ended ), started( other.started ),
					converter( other.converter ) {
					// set latest stream position
					(void)infile.seekg( spos );
				}

				/** Base destructor. */
				~MatrixFileIterator() {
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

					if( ! ( infile >> val ) ) {
						ended = true;
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

						if(
							properties._symmetry == MatrixFileProperties::MMsymmetries::SYMMETRIC ||
							properties._symmetry == MatrixFileProperties::MMsymmetries::HERMITIAN
						) {
							++rowidx;
							if( rowidx  == properties._n + 1 ) {
								++colidx;
								rowidx = colidx + 1;
							}
						} else if (
							properties._symmetry == MatrixFileProperties::MMsymmetries::SKEWSYMMETRIC
						) {
							throw std::runtime_error(
								"Not implemented i,j: SKEWSYMMETRIC."
							);
						} else if (
							properties._symmetry == MatrixFileProperties::MMsymmetries::GENERAL
						) {
							++rowidx;
							if( rowidx == properties._m + 1 ) {
								rowidx = 1;
								++colidx;
							}

						}

					}

#ifdef _DEBUG
					std::cout << "MatrixFileIterator::operator++ "
						": buffer at index "
						  << i << " now contains " << val << "\n";
#endif

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
					return val;
				}

				/** Standard pointer request of iterator. */
				const OutputType * operator->() {
					if( started ) {
						preprocess();
						started = false;
						(void)operator++();
					}
					if( ended ) {
						throw std::runtime_error(
							"Attempt to dereference (via "
							"operator->) "
							"MatrixFileIterator in end "
							"position."
						);
					}
					return &( val );
				}

				/** Returns the current col index. */
				const S j() const {
					if( started ) {
						const_cast< MatrixFileIterator< T > * >( this )->preprocess();
						const_cast< MatrixFileIterator< T > * >( this )->started = false;
						(void)const_cast< MatrixFileIterator< T > * >( this )->operator++();
					}
					if( ended ) {
						throw std::runtime_error(
							"Attempt to dereference (via "
							"operator*) "
							"MatrixFileIterator in end "
							"position."
						);
					}
					return colidx;
				}

				/** Returns the current row index. */
				const S i() const {
					if( started ) {
						const_cast< MatrixFileIterator< T > * >( this )->preprocess();
						const_cast< MatrixFileIterator< T > * >( this )->started = false;
						(void)const_cast< MatrixFileIterator< T > * >( this )->operator++();
					}
					if( ended ) {
						throw std::runtime_error(
							"Attempt to dereference (via "
							"operator*) "
							"MatrixFileIterator in end "
							"position."
						);
					}
					return rowidx - 1;
				}

				/** Returns the current nonzero value. */
				const T & v() const {
					if( started ) {
						const_cast< MatrixFileIterator< T > * >( this )->preprocess();
						const_cast< MatrixFileIterator< T > * >( this )->started = false;
						(void)const_cast< MatrixFileIterator< T > * >( this )->operator++();
					}
					if( ended ) {
						throw std::runtime_error(
							"Attempt to dereference (via "
							"operator*) "
							"MatrixFileIterator in end "
							"position."
						);
					}

					return val;
				}
			};


		} // namespace internal
	}     // namespace utils
} // namespace alp

#endif // end ``_H_MATRIXFILEITERATOR''
