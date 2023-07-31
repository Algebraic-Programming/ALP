
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
 * @file
 *
 * Tests for the buildMatrixUnique API call.
 *
 * @author Benjamin Lozes
 * @date 17/05/2023
 */

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <vector>

#include <graphblas.hpp>

using namespace grb;

#define _DEBUG

template< class Iterator >
void printSparseMatrixIterator( size_t rows, size_t cols, const Iterator begin, const Iterator end, const std::string & name = "", std::ostream & os = std::cout ) {
	(void)rows;
	(void)cols;
	(void)begin;
	(void)end;
	(void)name;
	(void)os;
#ifndef _DEBUG
	return;
#endif
	std::cout << "Matrix \"" << name << "\" (" << rows << "x" << cols << "):" << std::endl << "[" << std::endl;
	if( rows > 50 || cols > 50 ) {
		os << "   Matrix too large to print" << std::endl;
	} else {
		for( size_t y = 0; y < rows; y++ ) {
			os << std::string( 3, ' ' );
			for( size_t x = 0; x < cols; x++ ) {
				auto nnz_val = std::find_if( begin, end, [ y, x ]( const typename std::iterator_traits< Iterator >::value_type & a ) {
					return a.first.first == y && a.first.second == x;
				} );
				if( nnz_val != end )
					os << std::fixed << ( *nnz_val ).second;
				else
					os << '_';
				os << " ";
			}
			os << std::endl;
		}
	}
	os << "]" << std::endl;
	std::flush( os );
}

template< class Iterator >
void printSparseVoidMatrixIterator( size_t rows, size_t cols, const Iterator begin, const Iterator end, const std::string & name = "", std::ostream & os = std::cout ) {
	(void)rows;
	(void)cols;
	(void)begin;
	(void)end;
	(void)name;
	(void)os;
#ifndef _DEBUG
	return;
#endif
	std::cout << "Matrix \"" << name << "\" (" << rows << "x" << cols << "):" << std::endl << "[" << std::endl;
	if( rows > 100 || cols > 100 ) {
		os << "   Matrix too large to print" << std::endl;
	} else {
		// os.precision( 3 );
		for( size_t y = 0; y < rows; y++ ) {
			os << std::string( 3, ' ' );
			for( size_t x = 0; x < cols; x++ ) {
				auto nnz_val = std::find_if( begin, end, [ y, x ]( const typename std::iterator_traits< Iterator >::value_type & a ) {
					return a.first == y && a.second == x;
				} );
				if( nnz_val != end )
					os << 'X';
				else
					os << '_';
				os << " ";
			}
			os << std::endl;
		}
	}
	os << "]" << std::endl;
	std::flush( os );
}

template< typename D >
void printSparseVoidMatrix( const Matrix< D > & mat, const std::string & name = "", std::ostream & os = std::cout ) {
	(void)mat;
	(void)name;
	(void)os;
#ifndef _DEBUG
	return;
#endif
	grb::wait( mat );
	printSparseVoidMatrixIterator( nrows( mat ), ncols( mat ), mat.cbegin(), mat.cend(), name, os );
}

template< typename D >
void printSparseMatrix( const Matrix< D > & mat, const std::string & name = "", std::ostream & os = std::cout ) {
	(void)mat;
	(void)name;
	(void)os;
#ifndef _DEBUG
	return;
#endif
	grb::wait( mat );
	printSparseMatrixIterator( nrows( mat ), ncols( mat ), mat.cbegin(), mat.cend(), name, os );
}

template< typename D, class Storage >
void printVoidCompressedStorage( const Storage & storage, size_t n, size_t nnz, std::ostream & os = std::cout ) {
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
	os << "  values    (" << nnz << "): [ ]" << std::endl << std::flush;
}

template< typename D, class Storage >
void printCompressedStorage( const Storage & storage, size_t n, size_t nnz, std::ostream & os = std::cout ) {
	printVoidCompressedStorage< D >( storage, n, nnz, os );
	os << "  values    (" << nnz << "): [ ";
	for( size_t i = 0; i < nnz; ++i ) {
		os << storage.values[ i ] << " ";
	}
	os << "]" << std::endl << std::flush;
}

template< typename D >
void printCRS( const Matrix< D > & mat, const std::string & label = "", std::ostream & os = std::cout ) {
	(void)mat;
	(void)label;
	(void)os;
#ifndef _DEBUG
	return;
#endif
	grb::wait( mat );
	os << "CRS \"" << label << "\" (" << nrows( mat ) << "x" << ncols( mat ) << "):" << std::endl;
	printCompressedStorage< D >( internal::getCRS( mat ), grb::nrows( mat ), grb::nnz( mat ), os );
}

template< typename D >
void printCCS( const Matrix< D > & mat, const std::string & label = "", std::ostream & os = std::cout ) {
	(void)mat;
	(void)label;
	(void)os;
#ifndef _DEBUG
	return;
#endif
	grb::wait( mat );
	os << "CCS \"" << label << "\" (" << nrows( mat ) << "x" << ncols( mat ) << "):" << std::endl;
	printCompressedStorage< D >( internal::getCCS( mat ), grb::ncols( mat ), grb::nnz( mat ), os );
}

template< typename D >
void printVoidCRS( const Matrix< D > & mat, const std::string & label = "", std::ostream & os = std::cout ) {
	(void)mat;
	(void)label;
	(void)os;
#ifndef _DEBUG
	return;
#endif
	grb::wait( mat );
	os << "CRS \"" << label << "\" (" << nrows( mat ) << "x" << ncols( mat ) << "):" << std::endl;
	printVoidCompressedStorage< D >( internal::getCRS( mat ), grb::nrows( mat ), grb::nnz( mat ), os );
}

template< typename D >
void printVoidCCS( const Matrix< D > & mat, const std::string & label = "", std::ostream & os = std::cout ) {
	(void)mat;
	(void)label;
	(void)os;
#ifndef _DEBUG
	return;
#endif
	grb::wait( mat );
	os << "CCS \"" << label << "\" (" << nrows( mat ) << "x" << ncols( mat ) << "):" << std::endl;
	printVoidCompressedStorage< D >( internal::getCCS( mat ), grb::ncols( mat ), grb::nnz( mat ), os );
}

template< class Storage >
bool check_storage_sorting( const size_t m, const size_t n, const Storage & storage ) {
	(void)n;
	for( size_t i = 0; i < m; ++i ) {
		for( auto t = storage.col_start[ i ] + 1; t < storage.col_start[ i + 1 ]; t++ ) {
			if( storage.row_index[ t - 1 ] < storage.row_index[ t ] )
				return false;
		}
	}
	return true;
}

template< typename D >
bool check_crs_sorting( const Matrix< D > & mat ) {
	grb::wait( mat );
	return check_storage_sorting( nrows( mat ), ncols( mat ), internal::getCRS( mat ) );
}

template< typename D >
bool check_ccs_sorting( const Matrix< D > & mat ) {
	grb::wait( mat );
	return check_storage_sorting( ncols( mat ), nrows( mat ), internal::getCCS( mat ) );
}

template< typename D >
RC check_sorting( const Matrix< D > & mat ) {
	bool crs_valid = check_crs_sorting( mat );
	bool ccs_valid = check_ccs_sorting( mat );
	RC rc = SUCCESS;
	if( not crs_valid ) {
		std::cerr << "CRS sorting check: FAILED\n" << std::flush;
		rc = FAILED;
	} else {
		std::cout << "CRS sorting check: OK\n" << std::flush;
	}
	if( not ccs_valid ) {
		std::cerr << "CCS sorting check: FAILED\n";
		rc = FAILED;
	} else {
		std::cout << "CCS sorting check: OK\n" << std::flush;
	}
	return rc;
}

void grb_program( const size_t & n, grb::RC & rc ) {
	{
		std::vector< int > rows( n * n ), cols( n * n );
		for( size_t i = 0; i < n * n; ++i ) {
			rows[ i ] = i / n;
			cols[ i ] = i % n;
		}
		Matrix< void > dense_void( n, n );
		if( SUCCESS != grb::buildMatrixUnique( dense_void, rows.begin(), cols.begin(), n * n, SEQUENTIAL ) ) {
			std::cerr << "buildMatrixUnique failed\n";
			rc = FAILED;
			return;
		}
		printVoidCRS( dense_void, "dense_void" );
		printVoidCCS( dense_void, "dense_void" );
		RC local_rc = check_sorting( dense_void );
		rc = rc ? rc : local_rc;
	}

	{
		std::vector< int > rows( n );
		std::iota( rows.begin(), rows.end(), 0 );
		Matrix< void > M( n, n, n );
		if( SUCCESS != grb::buildMatrixUnique( M, rows.begin(), rows.begin(), rows.size(), SEQUENTIAL ) ) {
			std::cerr << "buildMatrixUnique failed\n";
			rc = FAILED;
			return;
		}
		printVoidCRS( M, "identity_void" );
		printVoidCCS( M, "identity_void" );
		RC local_rc = check_sorting( M );
		rc = rc ? rc : local_rc;
	}

	{
		Matrix< void > M( n, n );
		std::vector< int > rows( 2 * n ), cols( 2 * n );

		std::iota( rows.begin(), rows.begin() + n, 0 );
		std::reverse( rows.begin(), rows.begin() + n );
		std::iota( rows.begin() + n, rows.end(), 0 );
		std::reverse( rows.begin() + n, rows.end() );
		std::fill( cols.begin(), cols.begin() + n, 1 );
		std::fill( cols.begin() + n, cols.end(), 0 );
		if( SUCCESS != grb::buildMatrixUnique( M, rows.begin(), cols.begin(), rows.size(), SEQUENTIAL ) ) {
			std::cerr << "buildMatrixUnique failed\n";
			rc = FAILED;
			return;
		}
		printVoidCRS( M, "2cols_void" );
		printVoidCCS( M, "2cols_void" );
		RC local_rc = check_sorting( M );
		rc = rc ? rc : local_rc;
	}
}

int main( int argc, char ** argv ) {
	// defaults
	size_t n = 100;

	// error checking
	if( argc == 2 ) {
		n = std::atol( argv[ 1 ] );
	}
	if( argc > 2 ) {
		std::cerr << "Usage: " << argv[ 0 ] << " [n]\n";
		std::cerr << "  -n (optional, default is 100): an even integer, the test "
				  << "size.\n";
		return 1;
	}

	std::cout << "This is functional test " << argv[ 0 ] << "\n";
	RC rc = SUCCESS;

	grb::Launcher< AUTOMATIC > launcher;

	if( launcher.exec( &grb_program, n, rc, true ) != SUCCESS ) {
		std::cerr << "Launching test FAILED\n";
		return 255;
	}

	if( rc != SUCCESS ) {
		std::cout << "Test FAILED (" << grb::toString( rc ) << ")" << std::endl;
		return rc;
	} else {
		std::cout << "Test OK" << std::endl;
		return 0;
	}
}
