
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

constexpr bool PRINT_TIMERS = false;
constexpr bool SKIP_FOLDL = false;
constexpr bool SKIP_FOLDR = false;

using nz_t = float;

template< typename T, typename V, class Monoid >
RC foldl_test( const char * test_label, const char * test_description, const grb::Matrix< V > & A, T initial, T expected, const Monoid & monoid ) {
	if( SKIP_FOLDL )
		return RC::SUCCESS;

	T value = initial;
	auto start_chrono = std::chrono::high_resolution_clock::now();
	foldl( value, A, monoid );
	auto end_chrono = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast< std::chrono::nanoseconds >( end_chrono - start_chrono );
	if( PRINT_TIMERS )
		std::cout << "foldl_test \"" << test_label << "\" took " << duration.count() << " ns" << std::endl;

	std::cout << "foldl_test \"" << test_label << "\": ";
	if( value == expected )
		std::cout << "OK" << std::endl;
	else
		std::cerr << "Failed" << std::endl
				  << test_description << std::endl
				  << std::string( 3, ' ' ) << "Initial value: " << initial << std::endl
				  << std::string( 3, ' ' ) << "Expected value: " << expected << std::endl
				  << std::string( 3, ' ' ) << "Actual value: " << value << std::endl;

	return value == expected ? RC::SUCCESS : RC::FAILED;
}

template< typename T, typename V, class Monoid >
RC foldr_test( const char * test_label, const char * test_description, const grb::Matrix< V > & A, T initial, T expected, const Monoid & monoid ) {
	if( SKIP_FOLDR )
		return RC::SUCCESS;

	T value = initial;
	auto start_chrono = std::chrono::high_resolution_clock::now();
	foldr( value, A, monoid );
	auto end_chrono = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast< std::chrono::nanoseconds >( end_chrono - start_chrono );
	if( PRINT_TIMERS )
		std::cout << "foldr_test \"" << test_label << "\" took " << duration.count() << " ns" << std::endl;

	std::cout << "foldr_test \"" << test_label << "\": ";
	if( value == expected )
		std::cout << "OK" << std::endl;
	else
		std::cerr << "Failed" << std::endl
				  << test_description << std::endl
				  << std::string( 3, ' ' ) << "Initial value: " << initial << std::endl
				  << std::string( 3, ' ' ) << "Expected value: " << expected << std::endl
				  << std::string( 3, ' ' ) << "Actual value: " << value << std::endl;

	return value == expected ? RC::SUCCESS : RC::FAILED;
}

template< typename T, typename V, class Monoid >
RC foldLR_test( const char * test_label, const char * test_description, const grb::Matrix< V > & A, T initial, T expected, const Monoid & monoid ) {
	RC rc = foldl_test( test_label, test_description, A, initial, expected, monoid );
	return rc ? rc : foldr_test( test_label, test_description, A, initial, expected, monoid );
}

void grb_program( const long & n, grb::RC & rc ) {
	// Build an identity matrix
	Matrix< nz_t > I( n, n );
	std::vector< size_t > I_rows( n ), I_cols( n );
	std::vector< nz_t > I_vals( n, 1 );
	std::iota( I_rows.begin(), I_rows.end(), 0 );
	std::iota( I_cols.begin(), I_cols.end(), 0 );
	buildMatrixUnique( I, I_rows.data(), I_cols.data(), I_vals.data(), n, PARALLEL );

	/**    Test case 1:
	 *  A simple additive reduction with the same types for the nnzs and the reduction result.
	 *  * Initial value is 0
	 *  * Expected result: n
	 */
	rc = foldLR_test( "1", "A simple reduction(+) with the same types for the nnzs and the reduction result.", I, (nz_t)0, (nz_t)n, Monoid< operators::add< nz_t >, identities::zero >() );
	if( rc )
		return;

	/**     Test case 2:
	 *  A simple additive reduction with the same types for the nnzs and the reduction result.
	 *  * Initial value is n
	 *  * Expected result: 2*n
	 */
	rc = foldLR_test( "2", "A simple reduction(+) with the same types for the nnzs and the reduction result.", I, (nz_t)n, (nz_t)( 2 * n ), Monoid< operators::add< nz_t >, identities::zero >() );
	if( rc )
		return;

	/**     Test case 3:
	 *  A simple additive reduction with different types for the nnzs and the reduction result (size_t <- size_t + float).
	 *  * Initial value is 0
	 *  * Expected result: n
	 */
	rc = foldl_test( "3", "A simple reduction(+) with different types for the nnzs and the reduction result (int <- int * float).", I, (size_t)0, (size_t)n,
		Monoid< operators::add< size_t, nz_t, size_t >, identities::zero >() );
	if( rc )
		return;
	rc = foldr_test( "3", "A simple reduction(+) with different types for the nnzs and the reduction result (int <- int * float).", I, (size_t)0, (size_t)n,
		Monoid< operators::add< nz_t, size_t, size_t >, identities::zero >() );
	if( rc )
		return;

	/**     Test case 4:
	 *  A simple additive reduction with different types for the nnzs and the reduction result (size_t <- size_t + float).
	 *  * Initial value is n
	 *  * Expected result: 2*n
	 */
	rc = foldl_test( "4", "A simple reduction(+) with different types for the nnzs and the reduction result (int <- int * float).", I, (size_t)n, (size_t)( 2 * n ),
		Monoid< operators::add< size_t, nz_t, size_t >, identities::zero >() );
	if( rc )
		return;
	rc = foldr_test( "4", "A simple reduction(+) with different types for the nnzs and the reduction result (int <- int * float).", I, (size_t)n, (size_t)( 2 * n ),
		Monoid< operators::add< nz_t, size_t, size_t >, identities::zero >() );
	if( rc )
		return;

	/**     Test case 5:
	 * A simple multiplicative reduction with the same types for the nnzs and the reduction result.
	 * * Initial value is 0
	 * * Expected result: 0
	 */
	rc = foldLR_test( "5", "A simple reduction(*) with the same types for the nnzs and the reduction result.", I, (nz_t)0, (nz_t)0, Monoid< operators::mul< nz_t >, identities::one >() );
	if( rc )
		return;

	/**     Test case 6:
	 * A simple multiplicative reduction with the same types for the nnzs and the reduction result.
	 * * Initial value is 1
	 * * Expected result: 1
	 */
	rc = foldLR_test( "6", "A simple reduction(*) with the same types for the nnzs and the reduction result.", I, (nz_t)1, (nz_t)1, Monoid< operators::mul< nz_t >, identities::one >() );
	if( rc )
		return;

	/**     Test case 7:
	 * A simple multiplicative reduction with different types for the nnzs and the reduction result (size_t <- size_t * float).
	 * * Initial value is 0
	 * * Expected result: 0
	 */
	rc = foldl_test( "7", "A simple reduction(*) with different types for the nnzs and the reduction result (int <- int * float).", I, (size_t)0, (size_t)0,
		Monoid< operators::mul< size_t, nz_t, size_t >, identities::one >() );
	if( rc )
		return;
	rc = foldr_test( "7", "A simple reduction(*) with different types for the nnzs and the reduction result (int <- int * float).", I, (size_t)0, (size_t)0,
		Monoid< operators::mul< nz_t, size_t, size_t >, identities::one >() );
	if( rc )
		return;

	/**     Test case 8:
	 * A simple multiplicative reduction with different types for the nnzs and the reduction result (size_t <- size_t * float).
	 * * Initial value is 1
	 * * Expected result: 1
	 */
	rc = foldl_test( "8", "A simple reduction(*) with different types for the nnzs and the reduction result (int <- int * float).", I, (size_t)1, (size_t)1,
		Monoid< operators::mul< size_t, nz_t, size_t >, identities::one >() );
	if( rc )
		return;
	rc = foldr_test( "8", "A simple reduction(*) with different types for the nnzs and the reduction result (int <- int * float).", I, (size_t)1, (size_t)1,
		Monoid< operators::mul< nz_t, size_t, size_t >, identities::one >() );
	if( rc )
		return;

	/**     Test case 9:
	 * A simple binary equal reduction with different types for the nnzs and the reduction result (bool <- bool == float).
	 * * Initial value is true
	 * * Expected result: true
	 */
	rc = foldl_test( "9", "A simple reduction(==) with different types for the nnzs and the reduction result (bool <- bool == float).", I, (bool)true, (bool)true,
		Monoid< operators::equal< bool, nz_t, bool >, identities::logical_true >() );
	if( rc )
		return;
	rc = foldr_test( "9", "A simple reduction(==) with different types for the nnzs and the reduction result (bool <- bool == float).", I, (bool)true, (bool)true,
		Monoid< operators::equal< nz_t, bool, bool >, identities::logical_true >() );
	if( rc )
		return;

	/**     Test case 10:
	 * A simple binary logical_or reduction with different types for the nnzs and the reduction result (bool <- bool || float).
	 * * Initial value is false
	 * * Expected result: true
	 */
	rc = foldl_test( "10", "A simple reduction(||) with different types for the nnzs and the reduction result (bool <- bool || float).", I, (bool)false, (bool)true,
		Monoid< operators::logical_or< bool, nz_t, bool >, identities::logical_false >() );
	if( rc )
		return;
	rc = foldr_test( "10", "A simple reduction(||) with different types for the nnzs and the reduction result (bool <- bool || float).", I, (bool)false, (bool)true,
		Monoid< operators::logical_or< nz_t, bool, bool >, identities::logical_false >() );
	if( rc )
		return;
}

int main( int argc, char ** argv ) {
	// defaults
	bool printUsage = false;
	size_t in = 10;

	// error checking
	if( argc > 2 ) {
		printUsage = true;
	}
	if( argc == 2 ) {
		in = std::atol( argv[ 1 ] );
	}
	if( printUsage ) {
		std::cerr << "Usage: " << argv[ 0 ] << " [n]\n";
		std::cerr << "  -n (optional, default is 10): an even integer, the test "
				  << "size.\n";
		return 1;
	}

	std::cout << "This is functional test " << argv[ 0 ] << "\n";
	grb::Launcher< AUTOMATIC > launcher;
	grb::RC out = RC::SUCCESS;
	if( launcher.exec( &grb_program, (long)in, out, true ) != SUCCESS ) {
		std::cerr << "Launching test FAILED\n";
		return 255;
	}
	if( out != SUCCESS ) {
		std::cout << "Test FAILED (" << grb::toString( out ) << ")" << std::endl;
		return out;
	} else {
		std::cout << "Test OK" << std::endl;
		return 0;
	}
}
