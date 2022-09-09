
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

#ifndef _H_MATRIXFILEREADER
#define _H_MATRIXFILEREADER

#include <algorithm> //std::max
#include <iostream>
#include <string>
#include <type_traits>

#include "MatrixFileIterator.hpp"
#include "MatrixFileProperties.hpp"
#include "MatrixFileReaderBase.hpp"

namespace alp {
	namespace utils {

		template< typename T, typename S = size_t >
		class MatrixFileReader : public internal::MatrixFileReaderBase< T, S > {

			static_assert( std::is_integral< S >::value, "The template parameter S to MatrixFileReader must be integral." );

			template< typename U, typename V >
			friend std::ostream & operator<<( std::ostream & out, const MatrixFileReader< U, V > & A );

		private:


		public:

			/**
			 * Constructs a matrix reader using minimal information.
			 *
			 * This constructor will parse the file in its entirety once. The use of an iterator will parse the file \em again.
			 *
			 * @param[in] filename Which file to read.
			 * @param[in] direct   (Optional) Whether the file uses direct indexing.
			 *                     If not, new indices will be automatically inferred.
			 *                     Default value is \a true.
			 * @param[in] symmetricmap (Optional) In case \a direct is \a false, whether
			 *                         the row map should equal the column map.
			 *
			 * Defaults for \a direct and \a symmetricmap are <tt>true</tt>.
			 *
			 * @throws std::runtime_error If the given file does not exist.
			 *
			 */
			MatrixFileReader( const std::string filename ) {
				internal::MatrixFileProperties & properties = this->properties;
				// set properties
				properties._fn = filename;
				// check for existance of file
				this->exists();
				// open up file stream to infer remainder properties
				std::ifstream infile( properties._fn );
				// try and find header
				this->findHeader( infile ) ;
				this->ignoreComments( infile ) ;
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
			internal::MatrixFileIterator< T > begin(
				const IOMode mode = SEQUENTIAL,
				const std::function< void( T & ) > valueConverter = []( T & ) {} ) {
				return cbegin( mode, valueConverter );
			}

			/**
			 * This is an alias to cbegin() -- we only allow read-only access to the
			 * underlying matrix.
			 */
			internal::MatrixFileIterator< T > end(
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
			internal::MatrixFileIterator< T > cbegin(
				const IOMode mode = SEQUENTIAL,
				const std::function< void( T & ) > valueConverter = []( T & ) {} ) {
				return internal::MatrixFileIterator< T >( internal::MatrixFileReaderBase< T, S >::properties, mode, valueConverter, false );
			}

			/** Matching end iterator to cbegin(). */
			internal::MatrixFileIterator< T > cend(
				const IOMode mode = SEQUENTIAL,
				const std::function< void( T & ) > valueConverter = []( T & ) {} ) {
				return internal::MatrixFileIterator< T >( internal::MatrixFileReaderBase< T, S >::properties, mode, valueConverter, true );
			}
		};

		// /** Pretty printing function. */
		// template< typename T, typename S >
		// std::ostream & operator<<( std::ostream & out, const MatrixFileReader< T, S > & A ) {
		// 	size_t nnz;
		// 	try {
		// 		nnz = A.nz();
		// 	} catch( ... ) { nnz = -1; }
		// 	out << A.filename() << " < ";
		// 	if( nnz == static_cast< size_t >( -1 ) ) {
		// 		out << "m: " << A.m() << ", n: " << A.n() << ", nz: "
		// 			<< "<unknown>"
		// 			<< ", entries: " << A.entries();
		// 	} else {
		// 		out << "m: " << A.m() << ", n: " << A.n() << ", nz: " << nnz << ", entries: " << A.entries();
		// 	}
		// 	out << ", symmetric: " << ( A.isSymmetric() ? "yes" : "no" );
		// 	out << ", uses direct addressing: " << ( A.usesDirectAddressing() ? "yes" : "no" );
		// 	out << " >\n";
		// 	return out;
		// }

	} // namespace utils
} // namespace alp

#endif //``_H_MATRIXFILEREADER''
