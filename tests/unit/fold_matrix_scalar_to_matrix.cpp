
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
 * Tests for the foldl+r( Matrix<D>[in,out], T[in], Operator ) API call
 *
 * @author Benjamin Lozes
 * @date 26/05/2023
 *
 * Tests whether the foldl and foldr API calls produce the expected results.
 *
 * The test cases are focused on the following aspects:
 *   * The types of the result, the matrix values and the operator
 * 	 * The initial value of the reduction result
 * 	 * The order of the operands (foldr, foldl)
 */

#include <algorithm>
#include <chrono>
#include <iostream>
#include <numeric>
#include <sstream>
#include <vector>

#include <graphblas.hpp>

using namespace grb;

constexpr bool SKIP_FOLDL = false;
constexpr bool SKIP_FOLDR = false;
constexpr bool SKIP_UNMASKED = false;
constexpr bool SKIP_MASKED = false;

#define _DEBUG

template< class Iterator >
void printSparseMatrixIterator( size_t rows, size_t cols, Iterator begin, Iterator end, const std::string & name = "", std::ostream & os = std::cout ) {
#ifndef _DEBUG
	return;
#endif
	std::cout << "Matrix \"" << name << "\" (" << rows << "x" << cols << "):" << std::endl << "[" << std::endl;
	if( rows > 50 || cols > 50 ) {
		os << "   Matrix too large to print" << std::endl;
	} else {
		// os.precision( 3 );
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

template< typename D >
void printSparseMatrix( const grb::Matrix< D > & mat, const std::string & name = "", std::ostream & os = std::cout ) {
	grb::wait( mat );
	printSparseMatrixIterator( grb::nrows( mat ), grb::ncols( mat ), mat.cbegin(), mat.cend(), name, os );
}

template< typename D >
bool are_matrices_equals( const grb::Matrix< D > & A, const grb::Matrix< D > & B ) {
	if( grb::nrows( A ) != grb::nrows( B ) || grb::ncols( A ) != grb::ncols( B ) )
		return false;
	grb::wait( A );
	grb::wait( B );
	std::vector< std::pair< std::pair< size_t, size_t >, D > > A_vec( A.cbegin(), A.cend() );
	std::vector< std::pair< std::pair< size_t, size_t >, D > > B_vec( B.cbegin(), B.cend() );
	return std::is_permutation( A_vec.cbegin(), A_vec.cend(), B_vec.cbegin() );
}

/**
 * Structure for testing
 *
 */
template< typename T, typename M, typename S, class OpFoldl, class OpFoldr >
struct input {
	const char * test_label;
	const char * test_description;
	const grb::Matrix< T > & initial;
	const grb::Matrix< M > & mask;
	const S scalar;
	const grb::Matrix< T > & expected;
	const OpFoldl & opFoldl;
	const OpFoldr & opFoldr = OpFoldr();

	input( const char * test_label,
		const char * test_description,
		const grb::Matrix< T > & initial,
		const grb::Matrix< M > & mask,
		const S scalar,
		const grb::Matrix< T > & expected,
		const OpFoldl & opFoldl = OpFoldl(),
		const OpFoldr & opFoldr = OpFoldr() ) :
		test_label( test_label ),
		test_description( test_description ), initial( initial ), mask( mask ), scalar( scalar ), expected( expected ), opFoldl( opFoldl ), opFoldr( opFoldr ) {}
};

template< typename T, typename M, typename S, class OpFoldl, class OpFoldr >
void grb_program( const input< T, M, S, OpFoldl, OpFoldr > & in, grb::RC & rc ) {
	rc = RC::SUCCESS;

	printSparseMatrix( in.initial, "initial" );
	printSparseMatrix( in.expected, "expected" );

	if( not SKIP_FOLDL && not SKIP_UNMASKED && rc == RC::SUCCESS ) { // Unmasked foldl
		grb::Matrix< T > result = in.initial;
		foldl( result, in.scalar, in.opFoldl );
		std::cout << "foldl (unmasked) \"" << in.test_label << "\": ";
		rc = rc ? rc : ( are_matrices_equals( result, in.expected ) ? RC::SUCCESS : RC::FAILED );
		if( rc == RC::SUCCESS )
			std::cout << "OK" << std::flush << std::endl;
		else
			std::cerr << "Failed" << std::endl << in.test_description << std::endl;
		printSparseMatrix( result, "foldl (unmasked) result" );
	}

	if( not SKIP_FOLDL && not SKIP_MASKED && rc == RC::SUCCESS ) { // Masked foldl
		grb::Matrix< T > result = in.initial;
		foldl( result, in.mask, in.scalar, in.opFoldl );
		std::cout << "foldl (masked) \"" << in.test_label << "\": ";
		rc = rc ? rc : ( are_matrices_equals( result, in.expected ) ? RC::SUCCESS : RC::FAILED );
		if( rc == RC::SUCCESS )
			std::cout << "OK" << std::endl;
		else
			std::cerr << "Failed" << std::endl << in.test_description << std::endl;
		printSparseMatrix( result, "foldl (masked) result" );
	}

	if( not SKIP_FOLDR && not SKIP_UNMASKED && rc == RC::SUCCESS ) { // Unmasked foldr
		grb::Matrix< T > result = in.initial;
		foldr( result, in.scalar, in.opFoldr );
		std::cout << "foldr (unmasked) \"" << in.test_label << "\": ";
		rc = rc ? rc : ( are_matrices_equals( result, in.expected ) ? RC::SUCCESS : RC::FAILED );
		if( rc == RC::SUCCESS )
			std::cout << "OK" << std::endl;
		else
			std::cerr << "Failed" << std::endl << in.test_description << std::endl;
		printSparseMatrix( result, "foldr (unmasked) result" );
	}

	if( not SKIP_FOLDR && not SKIP_MASKED && rc == RC::SUCCESS ) { // Masked foldr
		grb::Matrix< T > result = in.initial;
		foldr( result, in.mask, in.scalar, in.opFoldr );
		std::cout << "foldr (masked) \"" << in.test_label << "\": ";
		rc = rc ? rc : ( are_matrices_equals( result, in.expected ) ? RC::SUCCESS : RC::FAILED );
		if( rc == RC::SUCCESS )
			std::cout << "OK" << std::endl;
		else
			std::cerr << "Failed" << std::endl << in.test_description << std::endl;
		printSparseMatrix( result, "foldr (masked) result" );
	}
}

int main( int argc, char ** argv ) {
	// defaults
	bool printUsage = false;
	size_t n = 10;

	// error checking
	if( argc > 2 ) {
		printUsage = true;
	}
	if( argc == 2 ) {
		n = std::atol( argv[ 1 ] );
	}
	if( printUsage ) {
		std::cerr << "Usage: " << argv[ 0 ] << " [n]\n";
		std::cerr << "  -n (optional, default is 10): an even integer, the test "
				  << "size.\n";
		return 1;
	}

	std::cout << "This is functional test " << argv[ 0 ] << "\n";
	grb::Launcher< AUTOMATIC > launcher;
	grb::RC rc = RC::SUCCESS;

	if( ! rc ) { // Identity square * 2
		const int k = 2;
		const std::string label( "Test 01" );
		const std::string description( "Initial: Identity int [" + std::to_string( n ) + ";" + std::to_string( n ) +
			"]\n"
			"Mask: Identity void matrix.\n"
			"k = 2\n"
			"Operator: mul\n"
			"Expected: Identity int [" +
			std::to_string( n ) + ";" + std::to_string( n ) + "] * 2" );
		// Initial matrix
		Matrix< int > initial( n, n );
		std::vector< size_t > initial_rows( n ), initial_cols( n );
		std::vector< int > initial_values( n, 1 );
		std::iota( initial_rows.begin(), initial_rows.end(), 0 );
		std::iota( initial_cols.begin(), initial_cols.end(), 0 );
		buildMatrixUnique( initial, initial_rows.data(), initial_cols.data(), initial_values.data(), initial_values.size(), SEQUENTIAL );
		// Mask
		Matrix< void > mask( n, n );
		buildMatrixUnique( mask, initial_rows.data(), initial_cols.data(), initial_rows.size(), SEQUENTIAL );
		// Expected matrix
		Matrix< int > expected( n, n );
		std::vector< int > expected_values( n, 2 );
		buildMatrixUnique( expected, initial_rows.data(), initial_cols.data(), expected_values.data(), expected_values.size(), SEQUENTIAL );
		std::cout << "-- Running " << label << " --" << std::endl;
		input< int, void, int, grb::operators::mul< int >, grb::operators::mul< int > > in { label.c_str(), description.c_str(), initial, mask, k, expected };
		if( launcher.exec( &grb_program, in, rc, true ) != SUCCESS ) {
			std::cerr << "Launching " << label << " failed" << std::endl;
			return 255;
		}
		std::cout << std::endl << std::flush;
	}

	if( rc != SUCCESS ) {
		std::cout << "Test FAILED (" << grb::toString( rc ) << ")" << std::endl;
		return rc;
	} else {
		std::cout << "Test OK" << std::endl;
		return 0;
	}
}
