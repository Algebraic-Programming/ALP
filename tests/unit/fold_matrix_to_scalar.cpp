
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
 * Tests for the reduce( Matrix<D>, T, Operator<T,D,T> ) API call
 *
 * @author Benjamin Lozes
 * @date 17/05/2023
 *
 * Tests whether the foldl and foldl API calls produce the expected results.
 *
 * The test cases are focused on the following aspects:
 *   * The types of the result, the matrix values and the operator
 * 	 * The initial value of the reduction result
 * 	 * The order of the operands (foldr, foldl)
 */

#include <chrono>
#include <iostream>
#include <numeric>
#include <sstream>
#include <vector>

#include <graphblas.hpp>

using namespace grb;

using NzType = double;

constexpr bool SKIP_FOLDL = false;
constexpr bool SKIP_FOLDR = false;
constexpr bool SKIP_UNMASKED = false;
constexpr bool SKIP_MASKED = false;
// Benchmarking
constexpr bool PRINT_TIMERS = false;
constexpr size_t ITERATIONS = 1;

//#define _DEBUG

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


template< typename T, typename V, typename M, class Monoid >
RC foldl_test( const char * test_label,
	const char * test_description,
	const grb::Matrix< V > & A,
	const grb::Matrix< M > & mask,
	T initial,
	T expected,
	const Monoid & monoid,
	bool skip_masked = false,
	bool skip_unmasked = false ) {
	if( SKIP_FOLDL )
		return RC::SUCCESS;
	RC rc = RC::SUCCESS;

	if( not skip_unmasked && rc == RC::SUCCESS && ! SKIP_UNMASKED ) { // Unmasked
		T value = initial;
		auto start_chrono = std::chrono::high_resolution_clock::now();
		for( size_t _ = 0; _ < ITERATIONS; _++ ) {
			value = initial;
			foldl( value, A, monoid );
		}
		auto end_chrono = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast< std::chrono::nanoseconds >( end_chrono - start_chrono ) / ITERATIONS;
		if( PRINT_TIMERS )
			std::cout << "foldl (unmasked) \"" << test_label << "\" took " << duration.count() << " ns" << std::endl;

		std::cout << "foldl (unmasked) \"" << test_label << "\": ";
		if( value == expected )
			std::cout << "OK" << std::endl;
		else
			std::cerr << "Failed" << std::endl
					  << test_description << std::endl
					  << std::string( 3, ' ' ) << "Initial value: " << initial << std::endl
					  << std::string( 3, ' ' ) << "Expected value: " << expected << std::endl
					  << std::string( 3, ' ' ) << "Actual value: " << value << std::endl;

		rc = rc ? rc : ( value == expected ? RC::SUCCESS : RC::FAILED );
	}

	if( not skip_masked && rc == RC::SUCCESS && ! SKIP_MASKED ) { // Masked
		T value = initial;
		auto start_chrono = std::chrono::high_resolution_clock::now();
		for( size_t _ = 0; _ < ITERATIONS; _++ ) {
			value = initial;
			foldl( value, A, mask, monoid );
		}
		auto end_chrono = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast< std::chrono::nanoseconds >( end_chrono - start_chrono ) / ITERATIONS;
		if( PRINT_TIMERS )
			std::cout << "foldl (masked) \"" << test_label << "\" took " << duration.count() << " ns" << std::endl;

		std::cout << "foldl (masked) \"" << test_label << "\": ";
		if( value == expected )
			std::cout << "OK" << std::endl;
		else
			std::cerr << "Failed" << std::endl
					  << test_description << std::endl
					  << std::string( 3, ' ' ) << "Initial value: " << initial << std::endl
					  << std::string( 3, ' ' ) << "Expected value: " << expected << std::endl
					  << std::string( 3, ' ' ) << "Actual value: " << value << std::endl;

		rc = rc ? rc : ( value == expected ? RC::SUCCESS : RC::FAILED );
	}

	return rc;
}

template< typename T, typename V, typename M, class Monoid >
RC foldr_test( const char * test_label,
	const char * test_description,
	const grb::Matrix< V > & A,
	const grb::Matrix< M > & mask,
	T initial,
	T expected,
	const Monoid & monoid,
	bool skip_masked = false,
	bool skip_unmasked = false ) {
	if( SKIP_FOLDR )
		return RC::SUCCESS;
	RC rc = RC::SUCCESS;

	if( not skip_unmasked && rc == RC::SUCCESS && ! SKIP_UNMASKED ) { // Unmasked
		T value = initial;
		auto start_chrono = std::chrono::high_resolution_clock::now();
		for( size_t _ = 0; _ < ITERATIONS; _++ ) {
			value = initial;
			foldr( value, A, monoid );
		}
		auto end_chrono = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast< std::chrono::nanoseconds >( end_chrono - start_chrono ) / ITERATIONS;
		if( PRINT_TIMERS )
			std::cout << "foldr (unmasked) \"" << test_label << "\" took " << duration.count() << " ns" << std::endl;

		std::cout << "foldr (unmasked) \"" << test_label << "\": ";
		if( value == expected )
			std::cout << "OK" << std::endl;
		else
			std::cerr << "Failed" << std::endl
					  << test_description << std::endl
					  << std::string( 3, ' ' ) << "Initial value: " << initial << std::endl
					  << std::string( 3, ' ' ) << "Expected value: " << expected << std::endl
					  << std::string( 3, ' ' ) << "Actual value: " << value << std::endl;

		rc = rc ? rc : ( value == expected ? RC::SUCCESS : RC::FAILED );
	}

	if( not skip_masked && rc == RC::SUCCESS && ! SKIP_MASKED ) { // Masked
		T value = initial;
		auto start_chrono = std::chrono::high_resolution_clock::now();
		for( size_t _ = 0; _ < ITERATIONS; _++ ) {
			value = initial;
			foldr( value, A, mask, monoid );
		}
		auto end_chrono = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast< std::chrono::nanoseconds >( end_chrono - start_chrono ) / ITERATIONS;
		if( PRINT_TIMERS )
			std::cout << "foldr (masked) \"" << test_label << "\" took " << duration.count() << " ns" << std::endl;

		std::cout << "foldr (masked) \"" << test_label << "\": ";
		if( value == expected )
			std::cout << "OK" << std::endl;
		else
			std::cerr << "Failed" << std::endl
					  << test_description << std::endl
					  << std::string( 3, ' ' ) << "Initial value: " << initial << std::endl
					  << std::string( 3, ' ' ) << "Expected value: " << expected << std::endl
					  << std::string( 3, ' ' ) << "Actual value: " << value << std::endl;

		rc = rc ? rc : ( value == expected ? RC::SUCCESS : RC::FAILED );
	}

	return rc;
}

template< typename T, typename V, typename M, class Monoid >
RC foldLR_test( const char * test_label,
	const char * test_description,
	const grb::Matrix< V > & A,
	const grb::Matrix< M > & mask,
	T initial,
	T expected,
	const Monoid & monoid,
	bool skip_masked = false,
	bool skip_unmasked = false ) {
	RC rc = foldl_test( test_label, test_description, A, mask, initial, expected, monoid, skip_masked, skip_unmasked );
	return rc ? rc : foldr_test( test_label, test_description, A, mask, initial, expected, monoid, skip_masked, skip_unmasked );
}

template< typename T, typename M >
struct input {
	const grb::Matrix< T > & A;
	const grb::Matrix< M > & mask;
};

template< typename T, typename M >
void grb_program( const input< T, M > & in, grb::RC & rc ) {
	const grb::Matrix< T > & I = in.A;
	const grb::Matrix< M > & mask = in.mask;

	const long n = grb::nnz( I );

	/**    Test case 1:
	 *  A simple additive reduction with the same types for the nzs and the reduction result.
	 *  * Initial value is 0
	 *  * Expected unmasked result: n
	 *  * Expected masked result: 0
	 */
	rc = foldLR_test( "1", "A simple reduction(+) with the same types for the nzs and the reduction result.", I, mask, (NzType)0, (NzType)n, Monoid< operators::add< NzType >, identities::zero >() );
	if( rc )
		return;

	/**     Test case 2:
	 *  A simple additive reduction with the same types for the nzs and the reduction result.
	 *  * Initial value is n
	 *  * Expected result: 2*n
	 */
	rc = foldLR_test(
		"2", "A simple reduction(+) with the same types for the nzs and the reduction result.", I, mask, (NzType)n, (NzType)( 2 * n ), Monoid< operators::add< NzType >, identities::zero >() );
	if( rc )
		return;

	/**     Test case 3:
	 *  A simple additive reduction with different types for the nzs and the reduction result (int <- int + NzType).
	 *  * Initial value is 0
	 *  * Expected result: n
	 */
	rc = foldl_test( "3", "A simple reduction(+) with different types for the nzs and the reduction result (int <- int + NzType).", I, mask, (int)0, (int)n,
		Monoid< operators::add< int, NzType, int >, identities::zero >() );
	if( rc )
		return;
	rc = foldr_test( "3", "A simple reduction(+) with different types for the nzs and the reduction result (int <- NzType + int).", I, mask, (int)0, (int)n,
		Monoid< operators::add< NzType, int, int >, identities::zero >() );
	if( rc )
		return;

	/**     Test case 4:
	 *  A simple additive reduction with different types for the nzs and the reduction result (int <- int + NzType).
	 *  * Initial value is n
	 *  * Expected result: 2*n
	 */
	rc = foldl_test( "4", "A simple reduction(+) with different types for the nzs and the reduction result (int <- int + NzType).", I, mask, (int)n, (int)( 2 * n ),
		Monoid< operators::add< int, NzType, int >, identities::zero >() );
	if( rc )
		return;
	rc = foldr_test( "4", "A simple reduction(+) with different types for the nzs and the reduction result (int <- NzType + int).", I, mask, (int)n, (int)( 2 * n ),
		Monoid< operators::add< NzType, int, int >, identities::zero >() );
	if( rc )
		return;

	/**     Test case 5:
	 * A simple multiplicative reduction with the same types for the nzs and the reduction result.
	 * * Initial value is 0
	 * * Expected result: 0
	 */
	rc = foldLR_test( "5", "A simple reduction(*) with the same types for the nzs and the reduction result.", I, mask, (NzType)0, (NzType)0, Monoid< operators::mul< NzType >, identities::one >() );
	if( rc )
		return;

	/**     Test case 6:
	 * A simple multiplicative reduction with the same types for the nzs and the reduction result.
	 * * Initial value is 1
	 * * Expected result: 1
	 */
	rc = foldLR_test( "6", "A simple reduction(*) with the same types for the nzs and the reduction result.", I, mask, (NzType)1, (NzType)1, Monoid< operators::mul< NzType >, identities::one >() );
	if( rc )
		return;

	/**     Test case 7:
	 * A simple multiplicative reduction with different types for the nzs and the reduction result (size_t <- size_t * NzType).
	 * * Initial value is 0
	 * * Expected result: 0
	 */
	rc = foldl_test( "7", "A simple reduction(*) with different types for the nzs and the reduction result (int <- int * NzType).", I, mask, (size_t)0, (size_t)0,
		Monoid< operators::mul< size_t, NzType, size_t >, identities::one >() );
	if( rc )
		return;
	rc = foldr_test( "7", "A simple reduction(*) with different types for the nzs and the reduction result (int <- int * NzType).", I, mask, (size_t)0, (size_t)0,
		Monoid< operators::mul< NzType, size_t, size_t >, identities::one >() );
	if( rc )
		return;

	/**     Test case 8:
	 * A simple multiplicative reduction with different types for the nzs and the reduction result (size_t <- size_t * NzType).
	 * * Initial value is 1
	 * * Expected result: 1
	 */
	rc = foldl_test( "8", "A simple reduction(*) with different types for the nzs and the reduction result (int <- int * NzType).", I, mask, (size_t)1, (size_t)1,
		Monoid< operators::mul< size_t, NzType, size_t >, identities::one >() );
	if( rc )
		return;
	rc = foldr_test( "8", "A simple reduction(*) with different types for the nzs and the reduction result (int <- int * NzType).", I, mask, (size_t)1, (size_t)1,
		Monoid< operators::mul< NzType, size_t, size_t >, identities::one >() );
	if( rc )
		return;

	/**     Test case 9:
	 * A simple binary equal reduction with different types for the nzs and the reduction result (bool <- bool == NzType).
	 * * Initial value is true
	 * * Expected result: true
	 */
	rc = foldl_test( "9", "A simple reduction(==) with different types for the nzs and the reduction result (bool <- bool == NzType).", I, mask, (bool)true, (bool)true,
		Monoid< operators::equal< bool, NzType, bool >, identities::logical_true >() );
	if( rc )
		return;
	rc = foldr_test( "9", "A simple reduction(==) with different types for the nzs and the reduction result (bool <- bool == NzType).", I, mask, (bool)true, (bool)true,
		Monoid< operators::equal< NzType, bool, bool >, identities::logical_true >() );
	if( rc )
		return;

	/**     Test case 10:
	 * A simple binary logical_or reduction with different types for the nzs and the reduction result (bool <- bool || NzType).
	 * * Initial value is false
	 * * Expected result: true
	 */
	rc = foldl_test( "10", "A simple reduction(||) with different types for the nzs and the reduction result (bool <- bool || NzType).", I, mask, (bool)false, (bool)true,
		Monoid< operators::logical_or< bool, NzType, bool >, identities::logical_false >() );
	if( rc )
		return;
	rc = foldr_test( "10", "A simple reduction(||) with different types for the nzs and the reduction result (bool <- bool || NzType).", I, mask, (bool)false, (bool)true,
		Monoid< operators::logical_or< NzType, bool, bool >, identities::logical_false >() );
	if( rc )
		return;

	/**     Test case 11:
	 * Reduction with an empty mask.
	 * * Initial value is 4
	 * * Expected result: 4
	 */
	Matrix< void > empty_mask( grb::nrows( I ), grb::ncols( I ), 0 );
	rc = foldLR_test( "11", "Reduction with an empty mask.", I, empty_mask, (NzType)4, (NzType)4, Monoid< operators::add< NzType >, identities::zero >(), false, true );
	if( rc )
		return;

	/**     Test case 12:
	 * Reduction with a dense mask.
	 * * Initial value is 0
	 * * Expected result: n
	 */
	Matrix< bool > dense_mask( grb::nrows( I ), grb::ncols( I ), grb::nrows( I ) * grb::ncols( I ) );
	std::vector< size_t > rows( grb::nrows( I ) * grb::ncols( I ) ), cols( grb::nrows( I ) * grb::ncols( I ) );
	for( size_t x = 0; x < grb::nrows( I ); x++ ) {
		std::fill( rows.begin() + x * grb::ncols( I ), rows.begin() + ( x + 1 ) * grb::ncols( I ), x );
		std::iota( cols.begin() + x * grb::ncols( I ), cols.begin() + ( x + 1 ) * grb::ncols( I ), 0 );
	}
	std::vector< int > dense_mask_vals( grb::nrows( I ) * grb::ncols( I ), 1 );
	buildMatrixUnique( dense_mask, rows.data(), cols.data(), dense_mask_vals.data(), grb::nrows( I ) * grb::ncols( I ), SEQUENTIAL );
	printSparseMatrix( dense_mask, "dense_mask" );
	rc = foldLR_test( "11", "Reduction with a dense mask.", I, dense_mask, (NzType)0, (NzType)n, Monoid< operators::add< NzType >, identities::zero >(), false, true );
	if( rc )
		return;
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

	if( ! rc ) { // Identity square-matrix
		Matrix< NzType > I( n, n );
		std::vector< size_t > I_rows( n ), I_cols( n );
		std::vector< NzType > I_vals( n, 1.f );
		std::iota( I_rows.begin(), I_rows.end(), 0 );
		std::iota( I_cols.begin(), I_cols.end(), 0 );
		buildMatrixUnique( I, I_rows.data(), I_cols.data(), I_vals.data(), I_vals.size(), PARALLEL );
		Matrix< void > mask( n, n );
		buildMatrixUnique( mask, I_rows.data(), I_cols.data(), I_rows.size(), PARALLEL );
		std::cout << "-- Running test 01: Identity square matrix of size n = " << n << std::endl;
		input< NzType, void > input = { I, mask };
		if( launcher.exec( &grb_program, input, rc, true ) != SUCCESS ) {
			std::cerr << "Launching test 01 FAILED\n";
			return 255;
		}
		std::cout << std::endl << std::flush;
	}

	if( ! rc ) { // Build a square-matrix with n 1s on the first row
		Matrix< NzType > I( n, n );
		std::vector< size_t > I_rows( n, 0 ), I_cols( n );
		std::vector< NzType > I_vals( n, 1.f );
		std::iota( I_cols.begin(), I_cols.end(), 0 );
		buildMatrixUnique( I, I_rows.data(), I_cols.data(), I_vals.data(), I_vals.size(), PARALLEL );
		Matrix< void > mask( n, n );
		buildMatrixUnique( mask, I_rows.data(), I_cols.data(), I_rows.size(), PARALLEL );
		std::cout << "-- Running test 02: Square matrix of size n = " << n << ", with n 1s on the first row" << std::endl;
		input< NzType, void > input = { I, mask };
		if( launcher.exec( &grb_program, input, rc, true ) != SUCCESS ) {
			std::cerr << "Launching test 02 FAILED\n";
			return 255;
		}
		std::cout << std::endl << std::flush;
	}

	if( ! rc ) { // Square-matrix with n 1s on the first column
		Matrix< NzType > I( n, n );
		std::vector< size_t > I_rows( n ), I_cols( n, 0 );
		std::vector< NzType > I_vals( n, 1.f );
		std::iota( I_rows.begin(), I_rows.end(), 0 );
		buildMatrixUnique( I, I_rows.data(), I_cols.data(), I_vals.data(), I_vals.size(), PARALLEL );
		Matrix< void > mask( n, n );
		buildMatrixUnique( mask, I_rows.data(), I_cols.data(), I_rows.size(), PARALLEL );
		std::cout << "-- Running test 03: Square matrix of size n = " << n << ", with n 1s on the first column" << std::endl;
		input< NzType, void > input = { I, mask };
		if( launcher.exec( &grb_program, input, rc, true ) != SUCCESS ) {
			std::cerr << "Launching test 03 FAILED\n";
			return 255;
		}
		std::cout << std::endl << std::flush;
	}

	if( ! rc ) { // Building a square-matrix with n 1s on the first row and column
		Matrix< NzType > I( n, n );
		std::vector< size_t > I_rows( 2 * n - 1, 0 ), I_cols( 2 * n - 1, 0 );
		std::vector< NzType > I_vals( 2 * n - 1, 1.f );
		std::iota( I_rows.begin() + n, I_rows.end(), 1 );
		std::iota( I_cols.begin(), I_cols.begin() + n, 0 );
		buildMatrixUnique( I, I_rows.data(), I_cols.data(), I_vals.data(), I_vals.size(), PARALLEL );
		Matrix< void > mask( n, n );
		buildMatrixUnique( mask, I_rows.data(), I_cols.data(), I_rows.size(), PARALLEL );
		std::cout << "-- Running test 04: Square matrix of size n = " << n << ", with n 1s on the first row and column" << std::endl;
		input< NzType, void > input = { I, mask };
		if( launcher.exec( &grb_program, input, rc, true ) != SUCCESS ) {
			std::cerr << "Launching test 04 FAILED\n";
			return 255;
		}
		std::cout << std::endl << std::flush;
	}

	if( ! rc ) { // Building a [1 row, n columns] matrix filled with 1s
		Matrix< NzType > I( 1, n );
		std::vector< size_t > I_rows( n, 0 ), I_cols( n, 0 );
		std::vector< NzType > I_vals( n, 1.f );
		std::iota( I_cols.begin(), I_cols.end(), 0 );
		buildMatrixUnique( I, I_rows.data(), I_cols.data(), I_vals.data(), I_vals.size(), PARALLEL );
		Matrix< void > mask( 1, n );
		buildMatrixUnique( mask, I_rows.data(), I_cols.data(), I_rows.size(), PARALLEL );
		std::cout << "-- Running test 05: [1-row, n = " << n << " columns] matrix, filled with 1s" << std::endl;
		input< NzType, void > input = { I, mask };
		if( launcher.exec( &grb_program, input, rc, true ) != SUCCESS ) {
			std::cerr << "Launching test 04 FAILED\n";
			return 255;
		}
		std::cout << std::endl << std::flush;
	}

	if( ! rc ) { // Building a [n rows, 1 column] matrix filled with 1s
		Matrix< NzType > I( n, 1 );
		std::vector< size_t > I_rows( n, 0 ), I_cols( n, 0 );
		std::vector< NzType > I_vals( n, 1.f );
		std::iota( I_rows.begin(), I_rows.end(), 0 );
		buildMatrixUnique( I, I_rows.data(), I_cols.data(), I_vals.data(), I_vals.size(), PARALLEL );
		Matrix< void > mask( n, 1 );
		buildMatrixUnique( mask, I_rows.data(), I_cols.data(), I_rows.size(), PARALLEL );
		std::cout << "-- Running test 06: [n = " << n << " rows, 1 column] matrix, filled with 1s" << std::endl;
		input< NzType, void > input = { I, mask };
		if( launcher.exec( &grb_program, input, rc, true ) != SUCCESS ) {
			std::cerr << "Launching test 06 FAILED\n";
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
