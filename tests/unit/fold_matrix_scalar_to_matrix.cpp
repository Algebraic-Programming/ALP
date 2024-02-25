
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
 * Tests for:
 * - foldl+r( Matrix<D>[in,out], T[in], Operator )
 * - foldl+r( Matrix<D>[in,out], Mask[in], T[in], Operator )
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

// #define _DEBUG

template< typename D, typename T >
bool are_matrices_equals(
	const Matrix< D > & A,
	const Matrix< T > & B )
{
	if( nrows( A ) != nrows( B ) || ncols( A ) != ncols( B ) || nnz( A ) != nnz( B ) )
		return false;
	return std::is_permutation( A.cbegin(), A.cend(), B.cbegin() );
}

/**
 * Structure for testing
 *
 */
template< typename T, typename M, typename S, class OpFoldl, class OpFoldr >
struct input {
	const char * test_label;
	const char * test_description;
	const Matrix< T > & initial;
	const Matrix< M > & mask;
	const S scalar;
	const Matrix< T > & expected;
	const bool skip_masked, skip_unmasked;
	const OpFoldl & opFoldl;
	const OpFoldr & opFoldr = OpFoldr();

	input( const char * test_label = "",
		const char * test_description = "",
		const Matrix< T > & initial = {0,0},
		const Matrix< M > & mask = {0,0},
		const S scalar = 0,
		const Matrix< T > & expected = {0,0},
		bool skip_masked = false,
		bool skip_unmasked = false,
		const OpFoldl & opFoldl = OpFoldl(),
		const OpFoldr & opFoldr = OpFoldr() ) :
			test_label( test_label ),
			test_description( test_description ),
			initial( initial ),
			mask( mask ),
			scalar( scalar ),
			expected( expected ),
			skip_masked( skip_masked ),
			skip_unmasked( skip_unmasked ),
			opFoldl( opFoldl ),
			opFoldr( opFoldr ) {}
};

template< typename T, typename M, typename S, class OpFoldl, class OpFoldr >
void grb_program( const input< T, M, S, OpFoldl, OpFoldr > & in, RC & rc ) {
	rc = SUCCESS;

	// Unmasked variant foldl
	if( !in.skip_unmasked && !SKIP_FOLDL && !SKIP_UNMASKED && rc == SUCCESS ) {
		std::cout << "foldl( unmasked ) \"" << in.test_label << "\": ";

		Matrix< T > result = in.initial;
		rc = rc ? rc : foldl( result, in.scalar, in.opFoldl );
		if( rc != SUCCESS ) {
			std::cerr << "Execution failed - " << std::endl << in.test_description << std::endl;
		}

		rc = rc ? rc : ( are_matrices_equals( result, in.expected ) ? SUCCESS : FAILED );
		if( rc != SUCCESS ) {
			std::cerr << "Check failed - " << std::endl << in.test_description << std::endl;
		}

		std::cout << "OK" << std::endl;
	}

	// Masked variant foldl
	if( !in.skip_masked && !SKIP_FOLDL && !SKIP_MASKED && rc == SUCCESS ) {
		std::cout << "foldl( masked ) \"" << in.test_label << "\": ";

		Matrix< T > result = in.initial;
		rc = rc ? rc : foldl( result, in.mask, in.scalar, in.opFoldl );
		if( rc != SUCCESS ) {
			std::cerr << "Execution failed - " << std::endl << in.test_description << std::endl;
		}

		rc = rc ? rc : ( are_matrices_equals( result, in.expected ) ? SUCCESS : FAILED );
		if( rc != SUCCESS ) {
			std::cerr << "Check failed - " << std::endl << in.test_description << std::endl;
		}

		std::cout << "OK" << std::endl;
	}

	// Unmasked variant foldr
	if( !in.skip_unmasked && !SKIP_FOLDR && !SKIP_UNMASKED && rc == SUCCESS ) {
		std::cout << "foldr( unmasked ) \"" << in.test_label << "\": ";

		Matrix< T > result = in.initial;
		rc = rc ? rc : foldr( result, in.scalar, in.opFoldr );
		if( rc != SUCCESS ) {
			std::cerr << "Execution failed - " << std::endl << in.test_description << std::endl;
		}

		rc = rc ? rc : ( are_matrices_equals( result, in.expected ) ? SUCCESS : FAILED );
		if( rc != SUCCESS ) {
			std::cerr << "Check failed - " << std::endl << in.test_description << std::endl;
		}

		std::cout << "OK" << std::endl;
	}

	// Masked variant foldr
	if( !in.skip_masked && !SKIP_FOLDR && !SKIP_MASKED && rc == SUCCESS ) {
		std::cout << "foldr( masked ) \"" << in.test_label << "\": ";

		Matrix< T > result = in.initial;
		rc = rc ? rc : foldr( result, in.mask, in.scalar, in.opFoldr );
		if( rc != SUCCESS ) {
			std::cerr << "Execution failed - " << std::endl << in.test_description << std::endl;
		}

		rc = rc ? rc : ( are_matrices_equals( result, in.expected ) ? SUCCESS : FAILED );
		if( rc != SUCCESS ) {
			std::cerr << "Check failed - " << std::endl << in.test_description << std::endl;
		}

		std::cout << "OK" << std::endl;
	}
}

struct main_input_t {
	const int k;
	const size_t n;
};

void main_grb(const main_input_t &in, RC& rc) {
	const size_t n = in.n;
	const auto k = in.k;

	// Initial matrix
	Matrix< int > initial( n, n );
	std::vector< size_t > initial_rows( n ), initial_cols( n );
	std::vector< int > initial_values( n, 1 );
	std::iota( initial_rows.begin(), initial_rows.end(), 0 );
	std::iota( initial_cols.begin(), initial_cols.end(), 0 );
	if( SUCCESS !=
		buildMatrixUnique( initial, initial_rows.data(), initial_cols.data(), initial_values.data(), initial_values.size(), SEQUENTIAL )
	) { throw std::runtime_error("[Line " + std::to_string(__LINE__) + "]: Building initial matrix failed"); }

	{
		const std::string label( "Test 01" );
		const std::string description(
			"Initial: Identity int [" + std::to_string( n ) + ";" + std::to_string( n ) +
			"]\n"
			"Mask: Identity void matrix (matching the input).\n"
			"k = 2\n"
			"Operator: mul\n"
			"Expected: Identity int [" +
			std::to_string( n ) + ";" + std::to_string( n ) + "] * 2"
		);
		// Mask (matching the input matrix)
		Matrix< void > mask( n, n );
		if( SUCCESS !=
			buildMatrixUnique( mask, initial_rows.data(), initial_cols.data(), initial_rows.size(), SEQUENTIAL )
		) { throw std::runtime_error("[Line " + std::to_string(__LINE__) + "]: Building mask matrix failed"); }
		// Expected matrix
		Matrix< int > expected( n, n );
		std::vector< int > expected_values( n, 2 );
		if( SUCCESS !=
			buildMatrixUnique( expected, initial_rows.data(), initial_cols.data(), expected_values.data(), expected_values.size(), SEQUENTIAL )
		) { throw std::runtime_error("[Line " + std::to_string(__LINE__) + "]: Building mask matrix failed"); }

		std::cout << "-- Running " << label << " --" << std::endl;
		input< int, void, int, operators::mul< int >, operators::mul< int > > in {
			label.c_str(), description.c_str(), initial, mask, k, expected
		};
		grb_program( in, rc );
		std::cout << std::endl << std::flush;
	}

	{
		const std::string label( "Test 02" );
		const std::string description(
			"Initial: Identity int [" + std::to_string( n ) + ";" + std::to_string( n ) +
			"]\n"
			"Mask: Identity void matrix (empty).\n"
			"k = 2\n"
			"Operator: mul\n"
			"Expected: Identity int [" +
			std::to_string( n ) + ";" + std::to_string( n ) + "]"
		);
		// Mask (matching the input matrix)
		Matrix< void > mask( n, n );
		if( SUCCESS !=
			buildMatrixUnique( mask, initial_rows.data(), initial_cols.data(), 0, SEQUENTIAL )
		) { throw std::runtime_error("[Line " + std::to_string(__LINE__) + "]: Building mask matrix failed"); }
		// Expected matrix
		Matrix< int > expected( n, n );
		if( SUCCESS !=
			buildMatrixUnique( expected, initial_rows.data(), initial_cols.data(), initial_values.data(), initial_values.size(), SEQUENTIAL )
		) { throw std::runtime_error("[Line " + std::to_string(__LINE__) + "]: Building mask matrix failed"); }

		std::cout << "-- Running " << label << " --" << std::endl;
		input< int, void, int, operators::mul< int >, operators::mul< int > > in {
			label.c_str(), description.c_str(), initial, mask, k, expected, false, true
		};
		grb_program( in, rc );
		std::cout << std::endl << std::flush;
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
	const int k = 2;

	const main_input_t in = {k, n};
	RC rc = grb::Launcher< AUTOMATIC >().exec( &main_grb, in, rc );


	if( rc != SUCCESS ) {
		std::cout << "Test FAILED (" << grb::toString( rc ) << ")" << std::endl;
		return rc;
	}

	std::cout << "Test OK" << std::endl;
	return 0;
}
