
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

#ifndef _H_MATRIXFILEREADERBASE
#define _H_MATRIXFILEREADERBASE

#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>

#include <sys/stat.h> //C-style stat, used to check file existance

#include "MatrixFileProperties.hpp"

namespace alp {
	namespace utils {
		namespace internal {

			/**
			 * Parses SNAP files & Matrix Market files.
			 *
			 * @tparam T The type a nonzero value iterator should return. Can be set to \a void in case the values are not of interest.
			 * @tparam S (Optional) The type an nonzero index iterator should return. Default value: \a size_t.
			 */
			template< typename T, typename S = size_t >
			class MatrixFileReaderBase {

			protected:
				/** Properties, including filename etc. */
				MatrixFileProperties properties;

				/** Checks whether \a fn exists on the file system. If not, throws a runtime error. */
				void exists() {
					// declare buffer (see man 2 stat)
					struct stat buf;
					// try and fill buffer
					const int rc = stat( properties._fn.c_str(), &buf );
					// if call was successful then the file exists (we ignore contents of buf)
					if( rc != 0 ) {
						throw std::runtime_error( "The given file " + properties._fn + " does not exist." );
					}
				}

				/** Forwards the stream until we hit a non-comment line. */
				void ignoreComments( std::ifstream & infile ) {
					char peek = infile.peek();
					while( infile.good() && ( peek == '%' || peek == '#' ) ) {
						(void)infile.ignore( std::numeric_limits< std::streamsize >::max(), '\n' );
						peek = infile.peek();
					}
				}

				/** Checks whether we have MatrixMarket input. If yes, use that to set \a _m, \a _n, and \a _nz. If yes, returns \a true. If not, returns \a false. */
				bool findHeader( std::ifstream & infile ) {
					// check if first header indicates MatrixMarket
					const std::streampos start = infile.tellg();
					// assume input is matrix market until we detect otherwise
					bool mmfile = true;
					std::string line;

					// try and parse header
					if( ! std::getline( infile, line ) ) {
						// some type of error occurred-- rewind and let a non-mmfile parse try
						mmfile = false;
						(void)infile.seekg( start );
					}

					std::stringstream streamline(line);
					std::string wordinline;

					if( mmfile && ( streamline >> wordinline ) && wordinline != "%%MatrixMarket" ) {
						// some type of error occurred-- rewind and let a non-mmfile parse try
						mmfile = false;
						(void)infile.seekg( start );
					}

					if( mmfile ) {
						std::cerr << "Info: MatrixMarket file detected. Header line: ``"
							  << line << "''\n";
						// matrix market files are always 1-based
						properties._oneBased = true;
						properties._direct = true;
						// parse header: object type
						if( !( streamline >> wordinline ) || wordinline != "matrix" ) {
							throw std::runtime_error(
								"MatrixMarket file does "
								"not describe a "
								"matrix."
							);
						}
						// parse header: format type
						if ( streamline >> wordinline ) {
							if( wordinline == "coordinate" ) {
								properties._mmformat = MatrixFileProperties::MMformats::COORDINATE;
								throw std::runtime_error(
									"Matrix Market Coordinate format "
									"should not be used in ALP."
									"Please use GRB parser for sparse matrices (Coordinate format)."
								);
							} else if ( wordinline == "array" ) {
								properties._mmformat = MatrixFileProperties::MMformats::ARRAY;
								properties._direct = false;
							} else {
								throw std::runtime_error( "This parser only "
											  "understands coordinate and array "
											  "matrix storage." );
							}
						} else {
							std::cout << "wordinline = " << wordinline << "\n";
							throw std::runtime_error(
								"Cannot parse matrix file header file."
							);
						}
						// parse header: nonzero value type
						if ( streamline >> wordinline ) {
							if( wordinline != "real" ) {
								throw std::runtime_error(
									"This parser only "
									"understands real matrices."
								);
							}
						} else {
							std::cout << "wordinline = " << wordinline << "\n";
							throw std::runtime_error(
								"Cannot parse matrix file header file."
							);
						}

						// parse header: structural information
						if ( streamline >> wordinline ) {
							if( wordinline == "symmetric" ) {
								properties._symmetric = true;
							} else {
								if( wordinline == "general" ) {
									properties._symmetric = false;
								} else {
									throw std::runtime_error(
										"This parser only "
										"understands "
										"symmetric or "
										"general matrices."
									);
								}
							}
						} else {
							std::cout << "wordinline = " << wordinline << "\n";
							throw std::runtime_error(
								"Cannot parse matrix file header file."
							);
						}

						// ignore all comment lines
						ignoreComments( infile );
						// parse first header line
						std::streampos start = infile.tellg();
						if( ! std::getline( infile, line ) ) {
							// could not read first non-comment line-- let a non-mtx parser try
							mmfile = false;
						} else {
							// parse first non-comment non-header line
							std::istringstream iss( line );
							// set defaults
							properties._m = properties._n = properties._nz = properties._entries = 0;
							if ( properties._mmformat == MatrixFileProperties::MMformats::COORDINATE ) {
								if( ! ( iss >> properties._m >> properties._n >> properties._entries ) ) {
									// could not read length line-- let a non-mtx parser try
									mmfile = false;
								} else {
									// header parse OK, set nonzeroes field if we can:
									if( ! properties._symmetric ) {
										properties._nz = properties._entries;
									} else {
										properties._nz = static_cast< size_t >( -1 );
									}
								}
							} else if ( properties._mmformat == MatrixFileProperties::MMformats::ARRAY ) {
								if( ! ( iss >> properties._m >> properties._n ) ) {
									// could not read length line-- let a non-mtx parser try
									mmfile = false;
								} else {
									properties._nz = properties._m * properties._n;
									// header parse OK, set nonzeroes field if we can:
									if( ! properties._symmetric ) {
										properties._entries = properties._nz;
									} else {
										if( properties._n != properties._m ) {
											throw std::runtime_error(
												"Matrix market Symmetric should be square: N x N."
											);
											properties._nz = static_cast< size_t >( -1 );
										}
										properties._entries = properties._n * ( properties._n + 1 ) / 2;
									}
								}
							}

						}
						// if we found a matrix market header but found non-standard lines
						if( ! mmfile ) {
							// rewind to let other parser try
							(void)infile.seekg( start );
							// and print warning
							std::cerr << "Warning: first line of file "
								"indicated MatrixMarket format-- "
								"however, no valid header line after "
								"comment block was found. ";
							std::cerr << "Attempting to continue as though "
								"this is *not* a MatrixMarket file.\n";
						}
					}
					// if header was successfully parsed, record type of file
					if( mmfile ) {
						properties._type = MatrixFileProperties::Type::MATRIX_MARKET;
					}
					// done
					return mmfile;
				}

				/** Prints info to stdout, to be called after successful construction. */
				void coda() const noexcept {
					std::cerr << "Info: MatrixFileReader constructed for "
						  << properties._fn << ": an " << properties._m << " times "
						  << properties._n << " matrix holding " << properties._entries
						  << " entries. ";
					if( properties._type == internal::MatrixFileProperties::Type::MATRIX_MARKET ) {
						std::cerr << "Type is MatrixMarket";
					} else {
						std::cerr << "Type is SNAP";
					}
					if( properties._symmetric ) {
						std::cerr << " and the input is symmetric";
					}
					std::cerr << ".\n";
				}

				/** Base construtor, does not initialise anything. */
				MatrixFileReaderBase() {}


			public:

				/**
				 * Constructs a matrix reader using maximal information.
				 *
				 * @param[in] filename  Which file to read.
				 * @param[in[ m         The number of rows to expect.
				 * @param[in] n         The number of columns to expect.
				 * @param[in] nz        The number of nonzeroes to expect.
				 * @param[in] symmetric Whether the input is symmetric.
				 * @param[in] direct    Whether the file uses direct indexing. If not, new
				 *                      indices will be automatically inferred.
				 * @param[in] symmetricmap Whether, in case \a direct is \a false, the row
				 *                         map should exactly correspond to the column map.
				 *                         If not, the row and column maps are computed
				 *                         independently of each other.
				 *
				 * @throws std::runtime_error If the given file does not exist.
				 *
				 * This constructor will \em not parse the file completely (only the use of an
				 * iterator such as begin() will do so). This constructor completes in
				 * \f$ \mathcal{O}(1) \f$ time.
				 */
				MatrixFileReaderBase( const std::string filename,
					const size_t m,
					const size_t n,
					const size_t nz,
					const size_t entries,
					const bool symmetric,
					const bool direct,
					const bool symmetricmap ) {
					// set all properties
					properties._fn = filename;
					properties._m = m;
					properties._n = n;
					properties._nz = nz;
					properties._entries = entries;
					properties._symmetric = symmetric;
					properties._direct = direct;
					properties._symmetricmap = symmetricmap;
					// check for existance of file
					exists();
				}

				/** Returns the underlying file name. */
				std::string filename() const noexcept {
					return properties._fn;
				}

				/** Returns the number of rows in the matrix file. */
				size_t m() const noexcept {
					return properties._m;
				}

				/** Returns the number of columns in the matrix file. */
				size_t n() const noexcept {
					return properties._n;
				}

				/**
				 * If known, returns the number of nonzeroes contained in the matrix file.
				 *
				 * \warning If the number is not known, this function will throw an
				 *          exception. Therefore, only use this function inside of a try-
				 *          catch.
				 *
				 * @returns The number of nonzeroes in the matrix file.
				 *
				 * @thows runtime_error In case the number of nonzeroes was not known a
				 *                      priori.
				 */
				size_t nz() const {
					if( properties._nz == static_cast< size_t >( -1 ) ) {
						throw std::runtime_error( "File header or parse mode "
												  "does not allow for an "
												  "a-priori count of "
												  "nonzeroes." );
					}
					return properties._nz;
				}

				/** Returns the number of entries in the underlying file. */
				size_t entries() const noexcept {
					return properties._entries;
				}

				/** Returns whether the matrix is symmetric. */
				bool isSymmetric() const noexcept {
					return properties._symmetric;
				}

				/** Returns whether the matrix uses direct indexing. */
				bool usesDirectAddressing() const noexcept {
					return properties._direct;
				}

				/**
				 * Returns the current row map.
				 *
				 * Will always be empty when \a usesDirectAddressing is \a true. Will only
				 * contain a mapping for those row coordinates that have been encountered
				 * during parsing. This means any iterator associated to this instance
				 * must have been exhausted before the map returned here is complete.
				 *
				 * Multiple iterators derived from this instance will share the same maps.
				 */
				const std::map< size_t, size_t > & rowMap() const noexcept {
					return properties._row_map;
				}

				/**
				 * Returns the current column map.
				 *
				 * Will always be empty when \a usesDirectAddressing is \a true. Will only
				 * contain a mapping for those row coordinates that have been encountered
				 * during parsing. This means any iterator associated to this instance
				 * must have been exhausted before the map returned here is complete.
				 *
				 * Multiple iterators derived from this instance will share the same maps.
				 */
				const std::map< size_t, size_t > & colMap() const noexcept {
					if( properties._symmetricmap ) {
						return properties._row_map;
					} else {
						return properties._col_map;
					}
				}
			};

		} // namespace internal
	}     // namespace utils
} // namespace alp

#endif // end ``_H_MATRIXFILEREADERBASE''
