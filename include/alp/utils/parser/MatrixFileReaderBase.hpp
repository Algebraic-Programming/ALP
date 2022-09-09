
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
						// properties._oneBased = true;

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
							if( wordinline == "real" ) {
								properties._datatype = MatrixFileProperties::MMdatatype::REAL;
							} else if ( wordinline == "complex" ) {
								properties._datatype = MatrixFileProperties::MMdatatype::COMPLEX;
								throw std::runtime_error( "Complex  matrices still not supported." );
							} else {
								throw std::runtime_error(
									"This parser only "
									"understands real or complex  matrices."
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
								properties._symmetry = MatrixFileProperties::MMsymmetries::SYMMETRIC;
							} else if ( wordinline == "general" ) {
								properties._symmetry = MatrixFileProperties::MMsymmetries::GENERAL;
							} else {
								throw std::runtime_error(
									"This parser only understands "
									"symmetric or general matrices."
								);
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
								throw std::runtime_error( "Matrix market Coordinate format not supported." );
							} else if ( properties._mmformat == MatrixFileProperties::MMformats::ARRAY ) {
								if( ! ( iss >> properties._m >> properties._n ) ) {
									// could not read length line-- let a non-mtx parser try
									mmfile = false;
								} else {
									properties._nz = properties._m * properties._n;
									// header parse OK, set nonzeroes field if we can:
									if( properties._symmetry == MatrixFileProperties::MMsymmetries::GENERAL ) {
										properties._entries = properties._nz;
									} else if ( properties._symmetry == MatrixFileProperties::MMsymmetries::SYMMETRIC ) {
										if( properties._n != properties._m ) {
											throw std::runtime_error( "Matrix market Symmetric should be square: N x N." );
											properties._nz = static_cast< size_t >( -1 );
										}
										properties._entries = properties._n * ( properties._n + 1 ) / 2;
									} else {
										throw std::runtime_error( "Not implemented." );
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
					std::cerr << " type  = " << properties._type << " " ;
					std::cerr << " symmetry  = " << properties._symmetry << " " ;
					std::cerr << ".\n";
				}

				/** Base construtor, does not initialise anything. */
				MatrixFileReaderBase() {}


			public:

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

			};

		} // namespace internal
	}     // namespace utils
} // namespace alp

#endif // end ``_H_MATRIXFILEREADERBASE''
