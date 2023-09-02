
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
 *
 * Utilities to print grb containers and objects.
 *
 * @authors
 * - Alberto Scolari (alberto.scolari@huawei.com)
 * - Benjamin Lozes (benjamin.lozes@huawei.com)
 *
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
 * Prints the first \p limit items (including zeroes) of vector \p x
 * with optional heading \p head.
 *
 * Contents will be printed to the standard output stream.
 *
 * @tparam T Vector data type.
 * @tparam B Vector backend.
 *
 * @param[in] x     The vector to print
 * @param[in] limit Max. number of elements to print; 0 for the entire vector
 * @param head optional heading to print \b before the vector
 *
 * \warning Assumes iterators over \a x are ordered.
 */
template< typename T, enum grb::Backend B >
void print_vector(
	const grb::Vector< T, B > &x,
	const size_t limit = 10UL,
	const char * const head = nullptr
) {
	size_t x_size = grb::size( x );
	size_t limit = limit == 0 ? x_size : std::min( x_size, limit );

	if( head != nullptr ) {
		std::cout << "<<< " << head << " >>>" << std::endl;
	}
	std::cout << "=== VECTOR ===" << std::endl;
	if( x_size == 0 ) {
		std::cout << "(size 0 vector)";
	}

	typename grb::Vector< T, B >::const_iterator it = x.cbegin();
	typename grb::Vector< T, B >::const_iterator end = x.cend();

	size_t previous_nnz = it == end ? limit : it->first;
	if( previous_nnz == 0 ) {
		std::cout << it->second;
		(void) ++it;
	} else if( x_size > 0 ) {
		std::cout << 0;
	}
	size_t next_nnz, position;
	next_nnz = it == end ? limit : it->first;
	position = 1;
	while( position < limit ) {
		size_t zero_streak = std::min( next_nnz, limit );
		// print sequence of zeroes
		for( ; position < zero_streak; ++position ) {
			std::cout << ", ";
			std::cout << 0;
		}
		if( position < limit ) {
			std::cout << ", ";
			std::cout << it->second;
			(void) ++position;
			(void) ++it;
			next_nnz = it->first;
		}
	}
	std::cout << std::endl << "==============" << std::endl << std::endl;
}

/**
 * Prints the first \p limit items of pinned vector \p x with optional
 * heading \p head.
 *
 * Contents will be printed to the standard output stream.
 *
 * @tparam T vector data type
 * @tparam B GraphBLAS backend storing the vector
 *
 * @param[in] v      Pinned vector to print
 * @param[in] limit Max number of elements to print; 0 for the entire vector
 * @param[in]  head optional heading to print \b before the vector
 *
 * \warning Nonzero values will be printed in an undefined order.
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
	size_t k = 0;
	if( k < v.nonzeroes() && limit > 0 ) {
		std::cout << v.getNonzeroValue( k++ );
	}
	for( size_t nnzs = 1; nnzs < limit && k < v.nonzeroes(); k++ ) {
		std::cout << ", " << v.getNonzeroValue( k );
		(void) ++nnzs;
	}
	std::cout << " )" << std::endl;
}

/**
 * Easy matrix container to store a matrix in a \b dense format.
 *
 * \warning Thus, <b>also zeroes are stored</b> and the memory occupation is
 *          <b>proportional to the full size of the matrix</b>. Hence, use this
 *          function with care!
 *
 * @tparam T the type of the matrix values.
 *
 */
template< typename T >
struct dense_mat {

	/** The number of rows in the matrix. */
	const size_t rows;

	/** The number of columns in the matrix. */
	const size_t cols;

	/** Pointer to the raw data, row-major storage. */
	T * const dense;

	/**
	 * Construct a new dense_mat object of given rows and columns.
	 *
	 * This function <b>allocates the necessary physical memory for dense
	 * storage</b>.
	 *
	 * @param[in] rows          The number of matrix rows.
	 * @param[in] cols          The number of matrix columns.
	 * @param[in] initial_value Optional; by default equal to zero.
	 *
	 * \warning This function assumes that zero maps to the literal <tt>0</tt>.
	 *
	 * @throws Out of memory errors in case #::dense cannot be allocated.
	 */
	dense_mat(
		const size_t _nrows, const size_t _ncols,
		const T initial_value = T( 0 )
	) :
		rows( _nrows ), cols( _ncols ),
		dense( new T[ rows * cols ] )
	{
		assert( rows != 0 );
		assert( cols != 0 );
		std::fill( dense, dense + rows * cols, initial_value );
	}

	/**
	 * Releases the resources corresponding to this instance.
	 */
	~dense_mat() {
		delete[] dense;
	}

	/**
	 * Operator to access an entire row.
	 *
	 * @param[in] row The row to access.
	 *
	 * Simply returns the pointer to the first row element; this way, one can
	 * conveniently write \code mat[i][j]] \endcode to access each element.
	 */
	inline T * operator[]( const size_t row ) {
		return dense + row * cols;
	}

	/**
	 * Operator to access an entire row.
	 *
	 * @param[in] row The row to access.
	 *
	 * Simply returns the const pointer to the first row element; this way, one can
	 * conveniently write \code mat[i][j]] \endcode to access each element.
	 */
	inline const T * operator[]( const size_t row ) const {
		return dense + row * cols;
	}
};

/**
 * Prints up to \p limit rows and columns of matrix \p mat with optional
 * heading \p head.
 *
 * @tparam T matrix data type
 * @tparam B ALP/GraphBLAS backend storing the matrix
 *
 * @param[in] mat   Matrix to print
 * @param[in] limit Max. number of rows and columns to print (0 for all)
 * @param[in] head  Optional heading to print \b before the matrix
 *
 * \warning This first casts \a mat to a dense matrix.
 *
 * \warning This function does not guard against iterators over \a mat
 *          (erroneously) returning an element at the same coordinate more
 *          than once.
 */
template<
	typename T,
	enum grb::Backend B,
	typename std::enable_if< !std::is_void< T >::value >::type * = nullptr
>
void print_matrix(
	const grb::Matrix< T, B > &mat,
	const size_t limit = 0,
	const char * const head = nullptr
) {
	const size_t rows = grb::nrows( mat );
	const size_t cols = grb::ncols( mat );
	size_t row_limit = limit == 0 ? rows : std::min( limit, rows );
	size_t col_limit = limit == 0 ? cols : std::min( limit, cols );
	// create and dump only relevant portion
	dense_mat< std::pair< bool, T> > dump(
		row_limit, col_limit, std::make_pair( false, static_cast< T >( 0 ) )
	);
	for( const std::pair< std::pair< size_t, size_t >, T > &t : mat ) {
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
 * Prints up to \p limit rows and columns of matrix \p mat with optional header
 * \p head.
 *
 * Specialisation for void matrices.
 *
 * @tparam T matrix data type
 * @tparam B GraphBLAS backend storing the matrix
 *
 * @param[in] mat   Matrix to print
 * @param[in] limit Max. number of rows and columns to print (0 for all)
 * @param[in] head  Optional heading to print \b before the matrix
 *
 * \warning This first casts \a mat to a dense matrix.
 *
 * \warning This function does not guard against iterators over \a mat
 *          (erroneously) returning an element at the same coordinate more
 *          than once.
 */
template<
	typename T,
	enum grb::Backend B,
	typename std::enable_if< std::is_void< T >::value >::type * = nullptr
>
void print_matrix(
	const grb::Matrix< T, B > &mat,
	size_t limit = 0,
	const char * head = nullptr
) {
	const size_t rows = grb::nrows( mat );
	const size_t cols = grb::ncols( mat );
	size_t row_limit = limit == 0 ? rows : std::min( limit, rows );
	size_t col_limit = limit == 0 ? cols : std::min( limit, cols );
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

namespace {

	/**
	 * \internal
	 * Helper function for printing a void reference CompressedStorage object.
	 * \endinternal
	 */
	template< typename D, class Storage >
	void printCompressedStorage(
		const Storage &storage,
		const size_t n,
		const size_t nnz,
		std::ostream &os = std::cout,
		const typename std::enable_if<
			std::is_void< D >::value, void
		>::type * const = nullptr
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

	/**
	 * \internal
	 * Helper function for printing a general reference CompressedStorage object.
	 * \endinternal
	 */
	template< typename D, class Storage >
	void printCompressedStorage(
		const Storage &storage,
		const size_t n,
		const size_t nnz,
		std::ostream &os,
		const typename std::enable_if<
			!std::is_void< D >::value, void
		>::type * const = nullptr
	) {
		printCompressedStorage< void >( storage, n, nnz, os );
		os << "  values    (" << nnz << "): [ ";
		for( size_t i = 0; i < nnz; ++i ) {
			os << storage.values[ i ] << " ";
		}
		os << "]" << std::endl << std::flush;
	}

} // namespace

/**
 * Print the CRS structure of a grb::Matrix.
 *
 * @tparam Enabled boolean flag to enable/disable the function.
 *
 * @param[in]     mat   Matrix CRS to print.
 * @param[in]     label Label to print before the matrix.
 * @param[in]     limit Max number of rows and columns to print (-1 for all).
 * @param[in,out] os    Output stream (optional; default is <tt>std::cout</tt>).
 *
 * \warning This function does \em not convert to CRS; if the implementing
 *          backend is not back by a CRS-like format, calling this function will
 *          not compile.
 */
template<
	bool Enabled = true,
	typename D, typename RIT, typename CIT, typename NIT,
	Backend implementation
>
void printCRS(
	const Matrix< D, implementation, RIT, CIT, NIT > &,
	const std::string & = "",
	const size_t limit = 128,
	std::ostream & = std::cout,
	const typename std::enable_if<
		implementation != reference &&
		implementation != reference_omp,
	void >::type * const  = nullptr
) {
	static_assert(
		implementation != reference &&
		implementation != reference_omp,
		"printCRS() is only available for reference and reference_omp backends"
	);
}

/**
 * Print the CRS structure of a grb::Matrix.
 *
 * This is the specialisation for the reference and reference_omp backends.
 *
 * @tparam Enabled boolean flag to enable/disable the function
 *
 * @param[in]     mat   Matrix CRS to print.
 * @param[in]     label Label to print before the matrix.
 * @param[in]     limit Max number of rows and columns to print (-1 for all).
 * @param[in,out] os    Output stream (optional; default is <tt>std::cout</tt>).
 *
 * \note The value -1 for \a limit refers to SIZE_MAX.
 */
template<
	bool Enabled = true,
	typename D, typename RIT, typename CIT, typename NIT,
	Backend implementation
>
void printCRS(
	const Matrix< D, implementation, RIT, CIT, NIT > &mat,
	const std::string &label = "",
	const size_t limit = 128,
	std::ostream &os = std::cout,
	const typename std::enable_if<
		implementation == reference ||
		implementation == reference_omp,
	void >::type * const  = nullptr
) {
	constexpr const size_t SIZE_MAX = std::numeric_limits< size_t >::max();
	if( !Enabled ) { return; }

	const long rows = static_cast< long >( nrows( mat ) );
	const long cols = static_cast< long >( ncols( mat ) );
	if( limit < SIZE_MAX && (rows > limit || cols > limit) ) { return; }

	const grb::RC rc = grb::wait( mat );
	if( rc != grb::SUCCESS ) {
		throw std::runtime_error( grb::toString( rc ) );
	}
	os << "CRS \"" << label
		<< "\" (" << nrows( mat ) << "x" << ncols( mat ) << "):\n";
	printCompressedStorage< D >(
		internal::getCRS( mat ),
		grb::nrows( mat ),
		grb::nnz( mat ),
		os
	);
}

/**
 * Print the CCS structure of a grb::Matrix.
 *
 * @tparam Enabled boolean flag to enable/disable the function.
 *
 * @param[in]     mat   Matrix CCS to print.
 * @param[in]     label Label to print before the matrix.
 * @param[in]     limit Max number of rows and columns to print (-1 for all).
 * @param[in,out] os    Output stream (optional, default is <tt>std::cout</tt>.
 *
 * \note The value -1 for \a limit refers to SIZE_MAX.
 *
 * \warning This function does \em not convert to CCS; if the implementing
 *          backend is not back by a CCS-like format, calling this function will
 *          not compile.
 */
template<
	bool Enabled = true,
	typename D, typename RIT, typename CIT, typename NIT,
	Backend implementation
>
void printCCS(
	const Matrix< D, implementation, RIT, CIT, NIT > &mat,
	const std::string &label = "",
	const size_t limit = 128,
	std::ostream &os = std::cout,
	const typename std::enable_if<
		implementation != reference &&
		implementation != reference_omp,
	void >::type * const  = nullptr
) {
	static_assert(
		implementation != reference &&
		implementation != reference_omp,
		"printCCS() is only available for reference and reference_omp backends"
	);
}

/**
 * Print the CCS structure of a grb::Matrix.
 *
 * This is the specialisation for the reference and reference_omp backends.
 *
 * @tparam Enabled boolean flag to enable/disable the function.
 *
 * @param[in]     mat   Matrix CCS to print.
 * @param[in]     label Label to print before the matrix.
 * @param[in]     limit Max number of rows and columns to print (-1 for all).
 * @param[in,out] os    Output stream (optional, default is <tt>std::cout</tt>.
 *
 * \note The value -1 for \a limit refers to SIZE_MAX.
 */
template<
	bool Enabled = true,
	typename D, typename RIT, typename CIT, typename NIT,
	Backend implementation
>
void printCCS(
	const Matrix< D, implementation, RIT, CIT, NIT > &mat,
	const std::string &label = "",
	const size_t limit = 128,
	std::ostream &os = std::cout,
	const typename std::enable_if<
		implementation == reference ||
		implementation == reference_omp,
	void >::type * const = nullptr
) {
	constexpr const size_t SIZE_MAX = std::numeric_limits< size_t >::max();
	if( !Enabled ) { return; }

	const long rows = static_cast< long >( nrows( mat ) );
	const long cols = static_cast< long >( ncols( mat ) );
	if( limit < SIZE_MAX && (rows > limit || cols > limit) ) { return; }

	const grb::RC rc = grb::wait( mat );
	if( rc != grb::SUCCESS ) {
		throw std::runtime_error( grb::toString( rc ) );
	}
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

