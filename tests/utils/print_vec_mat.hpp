
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

#ifndef _H_TEST_UTILS_PRINT_VEC_MAT
#define _H_TEST_UTILS_PRINT_VEC_MAT

/**
 * @file print_vec_mat.hpp
 * @authors
 * - Alberto Scolari (alberto.scolari@huawei.com)
 * - Benjamin Lozes (benjamin.lozes@huawei.com)
 *
 * @brief Utilities to print grb containers and objects.
 *
 * @details
 * Routines to print:
 * - grb::Vector, grb::Matrix & grb::PinnedVector: These primitives are in
 *   templated form to be generic w.r.t. stored data type and
 *   backend implementation.
 * - reference/CompressedStorage (CRS & CCS): These primitives are in
 *   templated form to be generic w.r.t. stored data type, but only for
 *   reference and reference_omp backends.
 *
 * @version 0.2
 * @date 25th of August 2023
 */

#include <algorithm>
#include <utility>
#include <iostream>
#include <iomanip>

#include <graphblas.hpp>

using namespace grb;


/**
 * Prints the first \p _limit items (including zeroes) of vector \p x
 * with optional heading \p head.
 *
 * @tparam T vector data type
 * @tparam B GraphBLAS backend storing the vector
 * @param x vector to print
 * @param _limit max number of elements to print; 0 for the entire vector
 * @param head optional heading to print \b before the vector
 */
template< typename T, enum grb::Backend B >
void print_vector(
	const grb::Vector< T, B > &x,
	size_t _limit = 10UL,
	const char * head = nullptr
) {
	// const T * const raw{grb::internal::getRaw(x)};
	size_t x_size { grb::size( x ) };
	size_t limit { _limit == 0 ? x_size : std::min( x_size, _limit ) };

	if( head != nullptr ) {
		std::cout << "<<< " << head << " >>>" << std::endl;
	}
	std::cout << "=== VECTOR ===" << std::endl;
	if( x_size == 0 ) {
		std::cout << "(size 0 vector)";
	}

	typename grb::Vector< T, B >::const_iterator it { x.cbegin() };
	typename grb::Vector< T, B >::const_iterator end { x.cend() };

	size_t previous_nnz { it == end ? limit : it->first };
	if( previous_nnz == 0 ) {
		std::cout << it->second;
		++it;
	} else if( x_size > 0 ) {
		std::cout << 0;
	}
	size_t next_nnz { it == end ? limit : it->first }, position { 1 };
	while( position < limit ) {
		size_t zero_streak { std::min( next_nnz, limit ) };
		// print sequence of zeroes
		for( ; position < zero_streak; ++position ) {
			std::cout << ", ";
			std::cout << 0;
		}
		if( position < limit ) {
			std::cout << ", ";
			std::cout << it->second;
			++position;
			++it;
			next_nnz = it->first;
		}
	}
	std::cout << std::endl << "==============" << std::endl << std::endl;
}

/**
 * Prints the first \p limit items of pinned vector \p x with optional
 * heading \p head.
 *
 * @tparam T vector data type
 * @tparam B GraphBLAS backend storing the vector
 *
 * @param[in] v pinned vector to print
 * @param[in] _limit max number of elements to print; 0 for the entire vector
 * @param[in]  head optional heading to print \b before the vector
 */
template< typename T, enum grb::Backend B >
void print_vector(
	const grb::PinnedVector< T, B > &v,
	const size_t limit = 10UL,
	const char * const head = nullptr
) {
	static_assert( !std::is_same< T, void >::value,
		"Cannot print the values of a void vector"
	);
	if( head != nullptr ) {
		std::cout << "<<< " << head << " >>>" << std::endl;
	}
	std::cout << "First " << limit << " nonzeroes of x are: ( ";
	size_t k { 0 };
	if( k < v.nonzeroes() && limit > 0 ) {
		std::cout << v.getNonzeroValue( k++ );
	}
	for( size_t nnzs { 1 }; nnzs < limit && k < v.nonzeroes(); k++ ) {
		std::cout << ", " << v.getNonzeroValue( k );
		++nnzs;
	}
	std::cout << " )" << std::endl;
}

/**
 * Easy matrix container to store a matrix in a \b dense format, thus <b>also
 * zeroes are stored</b> and the memory occupation is <b>proportional to the
 * full size of the matrix</b>; hence, use with case!
 *
 * @tparam T the type of the matrix values
 */
template< typename T >
struct dense_mat {
	const size_t rows, cols; ///< matrix dimensions
	T * const dense;              ///< pointer to data, stored in a linear format (row-wise)

	/**
	 * @brief Construct a new dense_mat object of given rows and columns, <b>allocating the necessary
	 *          physical memory for dense storage</b>.
	 */
	dense_mat( size_t _nrows, size_t _ncols, T initial_value = T( 0 ) ) :
		rows( _nrows ), cols( _ncols ), dense( new T[ rows * cols ] ) // we assume new throws if not enough memory
	{
		assert( rows != 0 );
		assert( cols != 0 );
		std::fill( dense, dense + rows * cols, initial_value );
	}

	~dense_mat() {
		delete[] dense;
	}

	/**
	 * @brief Operator to access an entire row, which simply returns the pointer to the first row element;
	 *          this way, one can conveniently write \code mat[i][j]] \endcode to access each element.
	 */
	inline T * operator[]( size_t row ) {
		return dense + row * cols;
	}

	/**
	 * @brief Operator to access an entire row, which simply returns the const pointer to the first row element;
	 *          this way, one can conveniently write \code mat[i][j]] \endcode to access each element.
	 */
	inline const T * operator[]( size_t row ) const {
		return dense + row * cols;
	}
};

/**
 * Prints up to \p _limit rows and columns of matrix \p mat with optional
 * heading \p head.
 *
 * @tparam T matrix data type
 * @tparam B GraphBLAS backend storing the matrix
 * @param mat matrix to print
 * @param _limit max number of rows and columns to print (0 for all)
 * @param head optional heading to print \b before the matrix
 */
template<
	typename T,
	enum grb::Backend B,
	typename std::enable_if< !std::is_void< T >::value >::type * = nullptr
>
void print_matrix(
	const grb::Matrix< T, B > &mat,
	size_t _limit = 0,
	const char * head = nullptr
) {
	const size_t rows = grb::nrows( mat );
	const size_t cols = grb::ncols( mat );
	size_t row_limit = _limit == 0 ? rows : std::min( _limit, rows );
	size_t col_limit = _limit == 0 ? cols : std::min( _limit, cols );
	// create and dump only relevant portion
	dense_mat< std::pair< bool, T> > dump(
		row_limit, col_limit, std::make_pair( false, static_cast< T >( 0 ) )
	);
	for( const std::pair< std::pair< size_t, size_t >, T > & t : mat ) {
		size_t row = t.first.first;
		size_t col = t.first.second;
		if( row < row_limit && col < col_limit ) {
			dump[ row ][ col ] = std::make_pair( true, t.second );
		}
	}

	if( head != nullptr ) {
		std::cout << "<<< " << head << " >>>" << std::endl;
	}
	std::cout << "=== MATRIX ===" << std::endl;
	std::cout << "Size: " << rows << " x " << cols << std::endl;
	for( size_t i = 0; i < row_limit; ++i ) {
		for( size_t j = 0; j < col_limit; ++j ) {
			bool assigned = dump[ i ][ j ].first;
			auto val = dump[ i ][ j ].second;
			if( assigned ) {
				std::cout << val;
			} else {
				std::cout << "_";
			}
			std::cout << " ";
		}
		std::cout << std::endl;
	}
	std::cout << "==============" << std::endl << std::endl;
}


/**
 * Prints up to \p _limit rows and columns of matrix \p mat with optional
 * heading \p head.
 *
 * @tparam T matrix data type
 * @tparam B GraphBLAS backend storing the matrix
 * @param mat matrix to print
 * @param _limit max number of rows and columns to print (0 for all)
 * @param head optional heading to print \b before the matrix
 */
template<
	typename T,
	enum grb::Backend B,
	typename std::enable_if< std::is_void< T >::value >::type * = nullptr
>
void print_matrix(
	const grb::Matrix< T, B > &mat,
	size_t _limit = 0,
	const char * head = nullptr
) {
	const size_t rows = grb::nrows( mat );
	const size_t cols = grb::ncols( mat );
	size_t row_limit = _limit == 0 ? rows : std::min( _limit, rows );
	size_t col_limit = _limit == 0 ? cols : std::min( _limit, cols );
	// create and dump only relevant portion
	dense_mat< bool > assigned( row_limit, col_limit, false );
	for( const auto &t : mat ) {
		auto row = t.first;
		auto col = t.second;
		assigned[ row ][ col ] = ( row < row_limit && col < col_limit );
	}

	if( head != nullptr ) {
		std::cout << "<<< " << head << " >>>" << std::endl;
	}
	std::cout << "=== PATTERN-MATRIX ===" << std::endl;
	std::cout << "Size: " << rows << " x " << cols << std::endl;
	for( size_t i = 0; i < row_limit; ++i ) {
		for( size_t j = 0; j < col_limit; ++j ) {
			if( assigned[ i ][ j ] ) {
				std::cout << "X";
			} else {
				std::cout << "_";
			}
			std::cout << " ";
		}
		std::cout << std::endl;
	}
	std::cout << "==============" << std::endl << std::endl;
}

namespace
{
	template<
		typename D,
		class Storage,
		typename std::enable_if< std::is_void< D >::value >::type * = nullptr
	>
	void printCompressedStorage(
		const Storage &storage,
		size_t n,
		size_t nnz,
		std::ostream &os = std::cout
	) {
		os << "  col_start (" << n + 1 << "): [ ";
		for( size_t i = 0; i <= n; ++i ) {
			os << storage.col_start[ i ] << " ";
		}
		os << "]" << std::endl;
		os << "  row_index (" << nnz << "): \n[\n";
		for( size_t i = 0; i < n; ++i ) {
			os << " " << std::setfill( '0' ) << std::setw( 2 ) << i << ":  ";
			for( auto t = storage.col_start[ i ]; t < storage.col_start[ i + 1 ]; t++ )
				os << std::setfill( '0' ) << std::setw( 2 ) << storage.row_index[ t ] << " ";
			os << std::endl;
		}
		os << "]" << std::endl;
	}

	template<
		typename D,
		class Storage,
		typename std::enable_if< !std::is_void< D >::value>::type * = nullptr
	>
	void printCompressedStorage(
		const Storage &storage,
		size_t n,
		size_t nnz,
		std::ostream &os
	) {
		printCompressedStorage< void >( storage, n, nnz, os );
		os << "  values    (" << nnz << "): [ ";
		for( size_t i = 0; i < nnz; ++i ) {
			os << storage.values[ i ] << " ";
		}
		os << "]" << std::endl << std::flush;
	}

} // namespace


template<
	bool Enabled = true,
	typename D, typename RIT, typename CIT, typename NIT,
	Backend implementation,
	typename = void
>
void printCRS(
	const Matrix< D, implementation, RIT, CIT, NIT > &,
	const std::string & = "",
	std::ostream & = std::cout
) {}

template<
	bool Enabled = true,
	typename D, typename RIT, typename CIT, typename NIT,
	Backend implementation,
	typename std::enable_if<
		implementation == reference ||
		implementation == reference_omp
	>::type = true
>
void printCRS(
	const Matrix< D, implementation, RIT, CIT, NIT > &mat,
	const std::string &label = "",
	std::ostream &os = std::cout
) {
	if( !Enabled ) return;

	if( nrows(mat) > 100 || ncols(mat) > 100 ) return;

	grb::wait( mat );
	os << "CRS \"" << label
		<< "\" (" << nrows( mat ) << "x" << ncols( mat ) << "):\n";
	printCompressedStorage< D >(
		internal::getCRS( mat ),
		grb::nrows( mat ),
		grb::nnz( mat ),
		os
	);
}


template<
	bool Enabled = true,
	typename D, typename RIT, typename CIT, typename NIT,
	Backend implementation = grb::config::default_backend,
	typename = void
>
void printCCS(
	const Matrix< D, implementation, RIT, CIT, NIT > &,
	const std::string & = "",
	std::ostream & = std::cout
) {}

template<
	bool Enabled = true,
	typename D, typename RIT, typename CIT, typename NIT,
	Backend implementation = grb::config::default_backend,
	typename std::enable_if<
		implementation == reference ||
		implementation == reference_omp
	>::type = true
>
void printCCS(
	const Matrix< D, implementation, RIT, CIT, NIT > &mat,
	const std::string &label = "",
	std::ostream &os = std::cout
) {
	if( !Enabled ) return;

	if( nrows(mat) > 100 || ncols(mat) > 100 ) return;

	grb::wait( mat );
	os << "CCS \"" << label
		<< "\" (" << nrows( mat ) << "x" << ncols( mat ) << "):\n" ;
	printCompressedStorage< D >(
		internal::getCCS( mat ),
		grb::ncols( mat ),
		grb::nnz( mat ),
		os
	);
}

#endif // _H_TEST_UTILS_PRINT_VEC_MAT

