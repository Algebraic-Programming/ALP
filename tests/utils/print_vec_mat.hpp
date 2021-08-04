
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
 * @author Alberto Scolari (alberto.scolari@huawei.com)
 * @brief Routines to print a grb::Vector, a grb::Matrix and a grb::PinnedVector; they are in templated form
 *          to be generic w.r.t. stored data type and backend implementation.
 * @version 0.1
 * @date 2021-04-30
 */
#include <algorithm>
#include <utility>

#include <graphblas.hpp>

/**
 * @brief Prints the first \p _limit items (including zeroes) of vector \p x with optional heading \p head.
 *
 * @tparam T vector data type
 * @tparam B GraphBLAS backend storing the vector
 * @param x vector to print
 * @param _limit max number of elements to print; 0 for the entire vector
 * @param head optional heading to print \b before the vector
 */
template< typename T, enum grb::Backend B >
void print_vector( const grb::Vector< T, B > & x, std::size_t _limit = 10UL, const char * head = nullptr ) {
	// const T * const raw{grb::internal::getRaw(x)};
	std::size_t x_size { grb::size( x ) };
	std::size_t limit { _limit == 0 ? x_size : std::min( x_size, _limit ) };

	if( head != nullptr ) {
		std::cout << "<<< " << head << " >>>" << std::endl;
	}
	std::cout << "=== VECTOR ===" << std::endl;
	if( x_size == 0 ) {
		std::cout << "(size 0 vector)";
	}

	typename grb::Vector< T, B >::const_iterator it { x.cbegin() };
	typename grb::Vector< T, B >::const_iterator end { x.cend() };

	std::size_t previous_nnz { it == end ? limit : it->first };
	if( previous_nnz == 0 ) {
		std::cout << it->second;
		++it;
	} else if( x_size > 0 ) {
		std::cout << 0;
	}
	std::size_t next_nnz { it == end ? limit : it->first }, position { 1 };
	while( position < limit ) {
		std::size_t zero_streak { std::min( next_nnz, limit ) };
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
 * @brief Prints the first \p limit items of pinned vector \p x with optional heading \p head.
 *
 * @tparam T vector data type
 * @tparam B GraphBLAS backend storing the vector
 * @param v pinned vector to print
 * @param _limit max number of elements to print; 0 for the entire vector
 * @param head optional heading to print \b before the vector
 */
template< typename T, enum grb::Backend B >
void print_vector( const grb::PinnedVector< T, B > & v, std::size_t limit = 10UL, const char * head = nullptr ) {
	if( head != nullptr ) {
		std::cout << "<<< " << head << " >>>" << std::endl;
	}
	std::cout << "First " << limit << " elements are: ( ";
	std::cout << ( v.mask( 0 ) ? v[ 0 ] : 0.0 );
	for( std::size_t i { 1 }; i < v.length() && i < limit; ++i ) {
		std::cout << ", " << ( v.mask( i ) ? v[ i ] : 0.0 );
	}
	std::cout << " )" << std::endl;
	std::cout << "First " << limit << " nonzeroes of x are: ( ";

	size_t i { 0 };
	for( ; i < v.length() && ! v.mask( i ); i++ )
		; // go to first non-zero
	if( i < v.length() ) {
		std::cout << v[ i++ ];
	}
	for( std::size_t nnzs { 1 }; nnzs < limit && i < v.length(); i++ ) {
		if( v.mask( i ) ) {
			std::cout << ", " << v[ i ];
			++nnzs;
		}
	}
	std::cout << " )" << std::endl;
}

/**
 * @brief Easy matrix container to store a matrix in a \b dense format, thus <b>also zeroes are stored</b>
 *          and the memory occupation is <b>proportional to the full size of the matrix</b>; hence, use with case!
 *
 * @tparam T the type of the matrix values
 */
template< typename T >
struct dense_mat {
	const std::size_t rows, cols; ///< matrix dimensions
	T * const dense;              ///< pointer to data, stored in a linear format (row-wise)

	/**
	 * @brief Construct a new dense_mat object of given rows and columns, <b>allocating the necessary
	 *          physical memory for dense storage</b>.
	 */
	dense_mat( std::size_t _nrows, std::size_t _ncols ) :
		rows( _nrows ), cols( _ncols ), dense( new T[ rows * cols ] ) // we assume new throws if not enough memory
	{
		assert( rows != 0 );
		assert( cols != 0 );
		memset( dense, T( 0 ), rows * cols * ( sizeof( T ) ) );
	}

	~dense_mat() {
		delete[] dense;
	}

	/**
	 * @brief Operator to access an entire row, which simply returns the pointer to the first row element;
	 *          this way, one can conveniently write \code mat[i][j]] \endcode to access each element.
	 */
	inline T * operator[]( std::size_t row ) {
		return dense + row * cols;
	}

	/**
	 * @brief Operator to access an entire row, which simply returns the const pointer to the first row element;
	 *          this way, one can conveniently write \code mat[i][j]] \endcode to access each element.
	 */
	inline const T * operator[]( std::size_t row ) const {
		return dense + row * cols;
	}
};

/**
 * @brief Prints up to \p _limit rows and columns of matrix \p mat with optional heading \p head.
 *
 * @tparam T matrix data type
 * @tparam B GraphBLAS backend storing the matrix
 * @param mat matrix to print
 * @param _limit max number of rows and columns to print (0 for all)
 * @param head optional heading to print \b before the matrix
 */
template< typename T, enum grb::Backend B >
void print_matrix( const grb::Matrix< T, B > & mat, std::size_t _limit = 0, const char * head = nullptr ) {
	const std::size_t rows = grb::nrows( mat );
	const std::size_t cols = grb::ncols( mat );
	std::size_t row_limit = _limit == 0 ? rows : std::min( _limit, rows );
	std::size_t col_limit = _limit == 0 ? cols : std::min( _limit, cols );
	// create and dump only relevant portion
	dense_mat< T > dump( row_limit, col_limit );
	for( const std::pair< std::pair< size_t, size_t >, T > & t : mat ) {
		std::size_t row { t.first.first };
		std::size_t col { t.first.second };
		if( row < row_limit && col < col_limit ) {
			dump[ row ][ col ] = t.second;
		}
	}

	if( head != nullptr ) {
		std::cout << "<<< " << head << " >>>" << std::endl;
	}
	std::cout << "=== MATRIX ===" << std::endl;
	std::cout << "Size: " << rows << " x " << cols << std::endl;
	for( std::size_t i = 0; i < row_limit; ++i ) {
		for( std::size_t j = 0; j < col_limit; ++j ) {
			double val = dump[ i ][ j ];
			std::cout << val;
			if( val == 0.0 ) {
				std::cout << "  ";
			} else {
				std::cout << " ";
			}
		}
		std::cout << std::endl;
	}
	std::cout << "==============" << std::endl << std::endl;
}

#endif // _H_TEST_UTILS_PRINT_VEC_MAT
