
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
 * @date 27/05/2023
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
constexpr bool SKIP_MASKED = false; // Not implemented yet

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
template< typename T, typename M, typename S, class MonoidFoldl, class MonoidFoldr >
struct input {
	const char * test_label;
	const char * test_description;
	const grb::Matrix< T > & initial;
	const grb::Matrix< M > & mask;
	const grb::Matrix< T > & B;
	const grb::Matrix< T > & expected;
	const bool skip_masked, skip_unmasked;
	const MonoidFoldl & monoidFoldl;
	const MonoidFoldr & monoidFoldr;

	input( const char * test_label = "",
		const char * test_description = "",
		const grb::Matrix< T > & initial = { 0, 0 },
		const grb::Matrix< M > & mask = { 0, 0 },
		const grb::Matrix< T > & B = { 0, 0 },
		const grb::Matrix< T > & expected = { 0, 0 },
		bool skip_masked = false,
		bool skip_unmasked = false,
		const MonoidFoldl & monoidFoldl = MonoidFoldl(),
		const MonoidFoldr & monoidFoldr = MonoidFoldr() ) :
		test_label( test_label ),
		test_description( test_description ), initial( initial ), mask( mask ), B( B ), expected( expected ), skip_masked( skip_masked ), skip_unmasked( skip_unmasked ), monoidFoldl( monoidFoldl ),
		monoidFoldr( monoidFoldr ) {}
};

template< typename T, typename M, typename S, class MonoidFoldl, class MonoidFoldr >
void grb_program( const input< T, M, S, MonoidFoldl, MonoidFoldr > & in, grb::RC & rc ) {
	rc = RC::SUCCESS;

	printSparseMatrix( in.initial, "initial" );
	printSparseMatrix( in.B, "B" );
	printSparseMatrix( in.expected, "expected" );

	if( not in.skip_unmasked && not SKIP_FOLDL && not SKIP_UNMASKED && rc == RC::SUCCESS ) { // Unmasked foldl
		grb::Matrix< T > result = in.initial;
		foldl( result, in.B, in.monoidFoldl );
		std::cout << "foldl (unmasked) \"" << in.test_label << "\": ";
		rc = rc ? rc : ( are_matrices_equals( result, in.expected ) ? RC::SUCCESS : RC::FAILED );
		if( rc == RC::SUCCESS )
			std::cout << "OK" << std::flush << std::endl;
		else
			std::cerr << "Failed" << std::endl << in.test_description << std::endl;
		printSparseMatrix( result, "foldl (unmasked) result" );
	}

	if( not in.skip_masked && not SKIP_FOLDL && not SKIP_MASKED && rc == RC::SUCCESS ) { // Masked foldl
		grb::Matrix< T > result = in.initial;
		foldl( result, in.mask, in.B, in.monoidFoldl );
		std::cout << "foldl (masked) \"" << in.test_label << "\": ";
		rc = rc ? rc : ( are_matrices_equals( result, in.expected ) ? RC::SUCCESS : RC::FAILED );
		if( rc == RC::SUCCESS )
			std::cout << "OK" << std::endl;
		else
			std::cerr << "Failed" << std::endl << in.test_description << std::endl;
		printSparseMatrix( result, "foldl (masked) result" );
	}

	if( not in.skip_unmasked && not SKIP_FOLDR && not SKIP_UNMASKED && rc == RC::SUCCESS ) { // Unmasked foldr
		grb::Matrix< T > result = in.initial;
		foldr( result, in.B, in.monoidFoldr );
		std::cout << "foldr (unmasked) \"" << in.test_label << "\": ";
		rc = rc ? rc : ( are_matrices_equals( result, in.expected ) ? RC::SUCCESS : RC::FAILED );
		if( rc == RC::SUCCESS )
			std::cout << "OK" << std::endl;
		else
			std::cerr << "Failed" << std::endl << in.test_description << std::endl;
		printSparseMatrix( result, "foldr (unmasked) result" );
	}

	if( not in.skip_masked && not SKIP_FOLDR && not SKIP_MASKED && rc == RC::SUCCESS ) { // Masked foldr
		grb::Matrix< T > result = in.initial;
		foldr( result, in.mask, in.B, in.monoidFoldr );
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

	// Identity matrix: I
	Matrix< int > I( n, n );
	std::vector< size_t > I_coords( n );
	std::vector< int > I_vals( n, 1 );
	std::iota( I_coords.begin(), I_coords.end(), 0 );
	buildMatrixUnique( I, I_coords.data(), I_coords.data(), I_vals.data(), I_vals.size(), SEQUENTIAL );

	{ // Test 01:  I *. I -> I
		const std::string label( "Test 01" );
		const std::string description( "A: Identity int [" + std::to_string( n ) + ";" + std::to_string( n ) + "]\n" + "Mask: Identity void matrix (matching the input).\n" + "B: Identity int [" +
			std::to_string( n ) + ";" + std::to_string( n ) + "]\n" + "Operator: mul\n" + "Expected: Identity int [" + std::to_string( n ) + ";" + std::to_string( n ) + "]" );
		// Mask: Pattern identity
		Matrix< void > mask( n, n );
		buildMatrixUnique( mask, I_coords.data(), I_coords.data(), I_coords.size(), SEQUENTIAL );
		// B: Identity
		Matrix< int > B = I;
		// Expected matrix: Identity
		Matrix< int > expected = I;
		// Run test
		std::cout << "-- Running " << label << " --" << std::endl;
		input< int, void, int, grb::Monoid< grb::operators::mul< int >, grb::identities::one >, grb::Monoid< grb::operators::mul< int >, grb::identities::one > > in { label.c_str(),
			description.c_str(), I, mask, B, expected };
		if( launcher.exec( &grb_program, in, rc, true ) != SUCCESS ) {
			std::cerr << "Launching " << label << " failed" << std::endl;
			return 255;
		}
		std::cout << std::endl << std::flush;
	}

	{ // Test 01:  I +. I -> 2 * I
		const std::string label( "Test 01" );
		const std::string description( "A: Identity int [" + std::to_string( n ) + ";" + std::to_string( n ) + "]\n" + "Mask: Identity void matrix (matching the input).\n" + "B: Identity int [" +
			std::to_string( n ) + ";" + std::to_string( n ) + "]\n" + "Operator: add\n" + "Expected: Identity int [" + std::to_string( n ) + ";" + std::to_string( n ) + "] * 2" );
		// Mask: Pattern identity
		Matrix< void > mask( n, n );
		buildMatrixUnique( mask, I_coords.data(), I_coords.data(), I_coords.size(), SEQUENTIAL );
		// B: Identity
		Matrix< int > B = I;
		// Expected matrix: Identity * 2
		Matrix< int > expected( n, n );
		std::vector< int > expected_vals( n, 2 );
		buildMatrixUnique( expected, I_coords.data(), I_coords.data(), expected_vals.data(), expected_vals.size(), SEQUENTIAL );
		// Run test
		std::cout << "-- Running " << label << " --" << std::endl;
		input< int, void, int, grb::Monoid< grb::operators::add< int >, grb::identities::zero >, grb::Monoid< grb::operators::add< int >, grb::identities::zero > > in { label.c_str(),
			description.c_str(), I, mask, B, expected };
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
