
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

#ifndef _H_GRB_UTILS_MATRIXFILEREADER
#define _H_GRB_UTILS_MATRIXFILEREADER

#include <string>
#include <iostream>
#include <algorithm> //std::max
#include <type_traits>

#include "matrixFileIterator.hpp"
#include "matrixFileProperties.hpp"
#include "matrixFileReaderBase.hpp"


namespace grb {

	namespace utils {

		template< typename T, typename S = size_t >
		class MatrixFileReader : public internal::MatrixFileReaderBase< T, S > {

			static_assert( std::is_integral< S >::value,
				"The template parameter S to MatrixFileReader must be integral." );

			template< typename U, typename V >
			friend std::ostream & operator<<(
				std::ostream &out, const MatrixFileReader< U, V > &A );


			private:

				/**
				 * In case we are reading pattern matrices, which value to substitute for
				 * nonzeroes.
				 */
				const T patternValue;


			public:

				/** Public typedef of the iterator type. */
				typedef typename internal::MatrixFileIterator< S, T > iterator;

				/** Public typedef of the iterator type. */
				typedef iterator const_iterator;

				/**
				 * Constructs a matrix reader using minimal information.
				 *
				 * This constructor will parse the file in its entirety once. The use of an
				 * iterator will parse the file \em again.
				 *
				 * @param[in] filename Which file to read.
				 * @param[in] direct   (Optional) Whether the file uses direct indexing.
				 *                     If not, new indices will be automatically inferred.
				 *                     Default value is \a true.
				 *
				 * @param[in] symmetricmap    (Optional) In case \a direct is \a false,
				 *                            whether the row map should equal the column
				 *                            map.
				 * @param[in] patternValueSub (Optional) Which value to substitute for
				 *                            nonzeroes when reading in from a pattern
				 *                            matrix.
				 *
				 * Defaults for \a direct and \a symmetricmap are <tt>true</tt>.
				 * Default for \a patternValueSub is <tt>static_cast< T >(1)</tt>
				 *
				 * @throws std::runtime_error If the given file does not exist.
				 *
				 * \note Auto-detecting the correct value for \a pattern only can happen
				 *       successfully in case of MatrixMarket.
				 */
				MatrixFileReader(
					const std::string filename,
					const bool direct = true, const bool symmetricmap = true,
					const T patternValueSub = 1
				) : patternValue( patternValueSub ) {
					internal::MatrixFileProperties &properties = this->properties;
					// set properties
					properties._fn = filename;
					properties._direct = direct;
					properties._symmetricmap = symmetricmap;
					// check for existance of file
					this->exists();
					// open up file stream to infer remainder properties
					std::ifstream infile( properties._fn );
					// try and find header
					if( !this->findHeader( infile ) ) {
#ifdef _DEBUG
						std::cout << "MatrixFileReader: couldn't parse header, inferring SNAP-"
							<< "based defaults; i.e., no pattern matrix, not symmetric, and"
							<< "0-based.\n";
#endif
						// not found, so we have to infer matrix properties
						// we assume the input is not pattern, since \a T is not \a void
						properties._pattern = false;
						// assume unsymmetric
						properties._symmetric = internal::General;
						// assume zero-based (SNAP-like)
						properties._oneBased = false;
						// record we assume SNAP
						properties._type = internal::MatrixFileProperties::Type::SNAP;
						// ignore comments
						this->ignoreComments( infile );
						// initialise default values, declare buffers
						properties._m = properties._n = properties._nz = properties._entries = 0;
						S row, col;
						T val;
						// read until we drop
						while( (infile >> row >> col >> val) ) {
							(void) ++properties._entries;
							(void) ++properties._nz;
							// if symmetric, count non-diagonal entries twice
							if( properties._symmetric && row != col ) {
								(void) ++properties._nz;
							}
							(void) val;
							if( !direct ) {
								const auto row_it = properties._row_map.find( row );
								if( row_it != properties._row_map.end() ) {
									row = row_it->second;
								} else {
									const size_t new_row_index = properties._row_map.size();
									properties._row_map[ row ] = new_row_index;
									row = new_row_index;
								}
								if( properties._symmetricmap ) {
									const auto col_it = properties._row_map.find( col );
									if( col_it != properties._row_map.end() ) {
										col = col_it->second;
									} else {
										const size_t new_col_index = properties._col_map.size();
										properties._row_map[ col ] = new_col_index;
										col = new_col_index;
									}
								} else {
									const auto col_it = properties._col_map.find( col );
									if( col_it != properties._col_map.end() ) {
										col = col_it->second;
									} else {
										const size_t new_col_index = properties._col_map.size();
										properties._col_map[ col ] = new_col_index;
										col = new_col_index;
									}
								}
							}
							// if symmetric, count non-diagonal entries twice
							if( properties._symmetric && row != col ) {
								++properties._nz;
							}
							// update dimensions
							if( row > properties._m ) {
								properties._m = row;
							}
							if( col > properties._n ) {
								properties._n = col;
							}
						}
						// correct _m and _n
						if( properties._symmetricmap ) {
							properties._m = std::max( properties._m, properties._n );
							properties._n = properties._m;
						}
						if( properties._nz > 0 ) {
							(void) ++properties._m;
							(void) ++properties._n;
						}
					}
					// print info to stdout
					this->coda();
#ifdef _DEBUG
					std::cout << *this << "\n";
#endif
				}

				// we do not provide non-const iterators, as we will not modify files
				// all other iterators follow the MatrixFileIterator codes:

				/**
				 * This is an alias to cbegin() -- we only allow read-only access to the
				 * underlying matrix.
				 */
				internal::MatrixFileIterator< S, T > begin(
					const IOMode mode = SEQUENTIAL,
					const std::function< void( T & ) > valueConverter = []( T & ) {} ) {
					return cbegin( mode, valueConverter );
				}

				/**
				 * This is an alias to cbegin() -- we only allow read-only access to the
				 * underlying matrix.
				 */
				internal::MatrixFileIterator< S, T > end(
					const IOMode mode = SEQUENTIAL,
					const std::function< void( T & ) > valueConverter = []( T & ) {} ) {
					return cend( mode, valueConverter );
				}

				/**
				 * Reads out the nonzeroes from the underlying matrix file. The returned
				 * iterator points to the first nonzero in the collection. No order of
				 * iteration is defined.
				 *
				 * @param[in] mode           Which I/O mode to employ.
				 * @param[in] valueConverter Optional value conversion lambda.
				 *
				 * The valueConverter must be a lambda function with signature
				 *   <tt> void f( T& ); </tt>
				 *
				 * The default value for \a mode is #SEQUENTIAL.
				 * The default for \a valueConverter is a no-op.
				 */
				internal::MatrixFileIterator< S, T > cbegin(
					const IOMode mode = SEQUENTIAL,
					const std::function< void( T & ) > valueConverter = []( T & ) {} ) {
					return internal::MatrixFileIterator< S, T >(
						internal::MatrixFileReaderBase< T, S >::properties, mode,
						valueConverter, patternValue, false
					);
				}

				/** Matching end iterator to cbegin(). */
				internal::MatrixFileIterator< S, T > cend(
					const IOMode mode = SEQUENTIAL,
					const std::function< void( T & ) > valueConverter = []( T & ) {} ) {
					return internal::MatrixFileIterator< S, T >(
						internal::MatrixFileReaderBase< T, S >::properties, mode,
						valueConverter, patternValue, true
					);
				}

		};

		template< typename S >
		class MatrixFileReader< void, S > :
			public internal::MatrixFileReaderBase< void, S >
		{
			static_assert( std::is_integral< S >::value,
				"The template parameter S to MatrixFileReader must be integral." );

			template< typename U, typename V >
			friend std::ostream & operator<<(
				std::ostream &out, const MatrixFileReader< U, V > &A );


			public:

				/** Public typedef of the iterator type. */
				typedef typename internal::MatrixFileIterator< S, void > iterator;

				/** Public typedef of the iterator type. */
				typedef iterator const_iterator;

				/**
				 * Constructs a matrix reader using minimal information.
				 *
				 * This constructor will parse the file in its entirety once. The use of an
				 * iterator will parse the file \em again.
				 *
				 * @param[in] filename Which file to read.
				 * @param[in] direct   (Optional) Whether the file uses direct indexing.
				 *                     If not, new indices will be automatically inferred.
				 *                     Default value is \a true.
				 * @param[in] symmetricmap (Optional) In case \a direct is \a false, whether
				 *                         the row map should equal the column map.
				 *
				 * @throws std::runtime_error If the given file does not exist.
				 *
				 * \note Auto-detecting the correct value for \a pattern only can happen
				 *       successfully in case of MatrixMarket.
				 */
				MatrixFileReader(
					const std::string filename,
					const bool direct = true,
					const bool symmetricmap = true
				) {
					internal::MatrixFileProperties &properties = this->properties;
					// set properties
					properties._fn = filename;
					properties._direct = direct;
					properties._symmetricmap = symmetricmap;
					// check for existance of file
					this->exists();
					// open up file stream to infer remainder properties
					std::ifstream infile( properties._fn );
					// try and find header
					if( !this->findHeader( infile ) ) {
						// not found, so we have to infer values for _n, _m, and _nz
						// we first assume the input is pattern and unsymmetric
						properties._pattern = true;
						properties._symmetric = internal::General;
						// assume 0-based input (SNAP-like)
						properties._oneBased = false;
						// record we assume SNAP
						properties._type = internal::MatrixFileProperties::Type::SNAP;
						// ignore comments
						this->ignoreComments( infile );
						// initialise default values, declare buffers
						properties._m = properties._n = properties._nz = properties._entries = 0;
						S row, col;
						// read until we drop
						while( (infile >> row >> col) ) {
							(void) ++properties._entries;
							(void) ++properties._nz;
							if( !direct ) {
								const auto row_it = properties._row_map.find( row );
								if( row_it != properties._row_map.end() ) {
									row = row_it->second;
								} else {
									const size_t new_row_index = properties._row_map.size();
									properties._row_map[ row ] = new_row_index;
									row = new_row_index;
								}
								if( properties._symmetricmap ) {
									const auto col_it = properties._row_map.find( col );
									if( col_it != properties._row_map.end() ) {
										col = col_it->second;
									} else {
										const size_t new_col_index = properties._row_map.size();
										properties._row_map[ col ] = new_col_index;
										col = new_col_index;
									}
								} else {
									const auto col_it = properties._col_map.find( col );
									if( col_it != properties._col_map.end() ) {
										col = col_it->second;
									} else {
										const size_t new_col_index = properties._col_map.size();
										properties._col_map[ col ] = new_col_index;
										col = new_col_index;
									}
								}
							}
							// if symmetric, count non-diagonal entries twice
							if( properties._symmetric && row != col ) {
								(void) ++properties._nz;
							}
							// update dimensions
							if( row > properties._m ) {
								properties._m = row;
							}
							if( col > properties._n ) {
								properties._n = col;
							}
						}
						// correct _m and _n
						if( properties._symmetricmap ) {
							properties._m = std::max( properties._m, properties._n );
							properties._n = properties._m;
						}
						if( properties._nz > 0 ) {
							(void) ++properties._m;
							(void) ++properties._n;
						}
					}
					// print info to stdout
					this->coda();
#ifdef _DEBUG
					std::cout << *this << "\n";
#endif
				}

				// we do not provide non-const iterators, as we will not modify files
				// all other iterators follow the MatrixFileIterator codes:

				internal::MatrixFileIterator< S, void > begin( const IOMode mode = SEQUENTIAL ) {
					return cbegin( mode );
				}

				internal::MatrixFileIterator< S, void > end( const IOMode mode = SEQUENTIAL ) {
					return cend( mode );
				}

				internal::MatrixFileIterator< S, void > cbegin( const IOMode mode = SEQUENTIAL ) {
					return internal::MatrixFileIterator< S, void >(
						internal::MatrixFileReaderBase< void, S >::properties, mode );
				}

				internal::MatrixFileIterator< S, void > cend( const IOMode mode = SEQUENTIAL ) {
					return internal::MatrixFileIterator< S, void >(
						internal::MatrixFileReaderBase< void, S >::properties, mode, true );
				}

		};

		/** Pretty printing function. */
		template< typename T, typename S >
		std::ostream & operator<<( std::ostream & out, const MatrixFileReader< T, S > & A ) {
			size_t nnz;
			try {
				nnz = A.nz();
			} catch( ... ) { nnz = -1; }
			out << A.filename() << " < ";
			if( nnz == static_cast< size_t >( -1 ) ) {
				out << "m: " << A.m() << ", n: " << A.n() << ", nz: "
					<< "<unknown>"
					<< ", entries: " << A.entries();
			} else {
				out << "m: " << A.m() << ", n: " << A.n() << ", nz: " << nnz << ", "
					<< "entries: " << A.entries();
			}
			out << ", pattern: " << ( A.isPattern() ? "yes" : "no" );
			out << ", symmetric: " << ( A.isSymmetric() ? "yes" : "no" );
			out << ", uses direct addressing: " <<
				(A.usesDirectAddressing() ? "yes" : "no");
			out << " >\n";
			return out;
		}

	} // namespace utils

} // namespace grb

#endif //``_H_GRB_UTILS_MATRIXFILEREADER''

