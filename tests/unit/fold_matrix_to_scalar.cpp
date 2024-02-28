
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

template<
	Descriptor descr = descriptors::no_operation,
	typename T, typename V, typename M, class Monoid
>
RC foldl_test(
	const char * test_label,
	const char * test_description,
	const Matrix< V > &A,
	const Matrix< M > &mask,
	T initial,
	T expected,
	const Monoid &monoid,
	bool skip_masked = false,
	bool skip_unmasked = false,
	RC expected_unmasked_rc = SUCCESS,
	RC expected_masked_rc = SUCCESS
) {
	if( SKIP_FOLDL ) { return SUCCESS; }
	RC rc = SUCCESS;

	 // Unmasked variant test
	if( !skip_unmasked && rc == SUCCESS && !SKIP_UNMASKED ) {
		T value = initial;
		RC local_rc = SUCCESS;
		const auto start_chrono = std::chrono::high_resolution_clock::now();
		for( size_t _ = 0; _ < ITERATIONS; _++ ) {
			value = initial;
			local_rc = local_rc ? local_rc : foldl< descr >( value, A, monoid );
		}
		const auto end_chrono = std::chrono::high_resolution_clock::now();
		const auto duration = std::chrono::duration_cast< std::chrono::nanoseconds >(
				end_chrono - start_chrono
			) / ITERATIONS;
		if( PRINT_TIMERS ) {
			std::cout << "foldl (unmasked) \"" << test_label
				<< "\" took " << duration.count() << " ns" << std::endl;
		}

		std::cout << "foldl (unmasked) \"" << test_label << "\": ";
		if( local_rc == expected_unmasked_rc ) {
			std::cout << "OK" << std::endl;
			rc = rc ? rc : SUCCESS;
		} else if( value == expected ) {
			std::cout << "OK" << std::endl;
			rc = rc ? rc : SUCCESS;
		} else {
			std::cerr << "Failed" << std::endl
				<< test_description << std::endl
				<< std::string( 3, ' ' ) << "Initial value: " << initial << std::endl
				<< std::string( 3, ' ' ) << "Expected value: " << expected << std::endl
				<< std::string( 3, ' ' ) << "Actual value: " << value << std::endl
				<< std::string( 3, ' ' ) << "Expected rc: " << expected_unmasked_rc << std::endl
				<< std::string( 3, ' ' ) << "Actual rc: " << local_rc << std::endl;
			rc = rc ? rc : FAILED;
		}
	}

	// Masked variant test
	if( !skip_masked && rc == SUCCESS && !SKIP_MASKED ) {
		T value = initial;
		const auto start_chrono = std::chrono::high_resolution_clock::now();
		RC local_rc = SUCCESS;
		for( size_t _ = 0; _ < ITERATIONS; _++ ) {
			value = initial;
			local_rc = local_rc ? local_rc : foldl< descr >( value, A, mask, monoid );
		}
		const auto end_chrono = std::chrono::high_resolution_clock::now();
		const auto duration = std::chrono::duration_cast< std::chrono::nanoseconds >(
				end_chrono - start_chrono
			) / ITERATIONS;
		if( PRINT_TIMERS ) {
			std::cout << "foldl (masked) \"" << test_label
				<< "\" took " << duration.count() << " ns" << std::endl;
		}

		std::cout << "foldl (masked) \"" << test_label << "\": ";
		if( local_rc == expected_masked_rc ) {
			std::cout << "OK" << std::endl;
			rc = rc ? rc : SUCCESS;
		} else if( value == expected ) {
			std::cout << "OK" << std::endl;
			rc = rc ? rc : SUCCESS;
		} else {
			std::cerr << "Failed" << std::endl
				<< test_description << std::endl
				<< std::string( 3, ' ' ) << "Initial value: " << initial << std::endl
				<< std::string( 3, ' ' ) << "Expected value: " << expected << std::endl
				<< std::string( 3, ' ' ) << "Actual value: " << value << std::endl
				<< std::string( 3, ' ' ) << "Expected rc: " << expected_masked_rc << std::endl
				<< std::string( 3, ' ' ) << "Actual rc: " << local_rc << std::endl;
			rc = rc ? rc : FAILED;
		}
	}

	return rc;
}


template<
	Descriptor descr = descriptors::no_operation,
	typename T, typename V, typename M, class Monoid
>
RC foldr_test(
	const char * test_label,
	const char * test_description,
	const Matrix< V > &A,
	const Matrix< M > &mask,
	T initial,
	T expected,
	const Monoid &monoid,
	bool skip_masked = false,
	bool skip_unmasked = false,
	RC expected_unmasked_rc = SUCCESS,
	RC expected_masked_rc = SUCCESS
) {
	if( SKIP_FOLDR ) { return SUCCESS; }
	RC rc = SUCCESS;

	// Unmasked variant test
	if( !skip_unmasked && rc == SUCCESS && !SKIP_UNMASKED ) {
		T value = initial;
		const auto start_chrono = std::chrono::high_resolution_clock::now();
		RC local_rc = SUCCESS;
		for( size_t _ = 0; _ < ITERATIONS; _++ ) {
			value = initial;
			local_rc = local_rc ? local_rc : foldr< descr >( A, value, monoid );
		}
		const auto end_chrono = std::chrono::high_resolution_clock::now();
		const auto duration = std::chrono::duration_cast< std::chrono::nanoseconds >(
				end_chrono - start_chrono
			) / ITERATIONS;
		if( PRINT_TIMERS ) {
			std::cout << "foldr (unmasked) \"" << test_label
				<< "\" took " << duration.count() << " ns" << std::endl;
		}

		std::cout << "foldr (unmasked) \"" << test_label << "\": ";
		if( local_rc == expected_unmasked_rc ) {
			std::cout << "OK" << std::endl;
			rc = rc ? rc : SUCCESS;
		} else if( value == expected ) {
			std::cout << "OK" << std::endl;
			rc = rc ? rc : SUCCESS;
		} else {
			std::cerr << "Failed" << std::endl
				<< test_description << std::endl
				<< std::string( 3, ' ' ) << "Initial value: " << initial << std::endl
				<< std::string( 3, ' ' ) << "Expected value: " << expected << std::endl
				<< std::string( 3, ' ' ) << "Actual value: " << value << std::endl
				<< std::string( 3, ' ' ) << "Expected rc: " << expected_unmasked_rc << std::endl
				<< std::string( 3, ' ' ) << "Actual rc: " << local_rc << std::endl;
			rc = rc ? rc : FAILED;
		}
	}

	// Masked variant test
	if( !skip_masked && rc == SUCCESS && !SKIP_MASKED ) {
		T value = initial;
		const auto start_chrono = std::chrono::high_resolution_clock::now();
		RC local_rc = SUCCESS;
		for( size_t _ = 0; _ < ITERATIONS; _++ ) {
			value = initial;
			local_rc = local_rc ? local_rc : foldr< descr >( A, mask, value, monoid );
		}
		const auto end_chrono = std::chrono::high_resolution_clock::now();
		const auto duration = std::chrono::duration_cast< std::chrono::nanoseconds >(
				end_chrono - start_chrono
			) / ITERATIONS;
		if( PRINT_TIMERS ) {
			std::cout << "foldr (masked) \"" << test_label
				<< "\" took " << duration.count() << " ns" << std::endl;
		}

		std::cout << "foldr (masked) \"" << test_label << "\": ";
		if( local_rc == expected_masked_rc ) {
			std::cout << "OK" << std::endl;
			rc = rc ? rc : SUCCESS;
		} else if( value == expected ) {
			std::cout << "OK" << std::endl;
			rc = rc ? rc : SUCCESS;
		} else {
			std::cerr << "Failed" << std::endl
				<< test_description << std::endl
				<< std::string( 3, ' ' ) << "Initial value: " << initial << std::endl
				<< std::string( 3, ' ' ) << "Expected value: " << expected << std::endl
				<< std::string( 3, ' ' ) << "Actual value: " << value << std::endl
				<< std::string( 3, ' ' ) << "Expected rc: " << expected_masked_rc << std::endl
				<< std::string( 3, ' ' ) << "Actual rc: " << local_rc << std::endl;
			rc = rc ? rc : FAILED;
		}
	}

	return rc;
}

template<
	Descriptor descr = descriptors::no_operation,
	typename T, typename V, typename M, class Monoid
>
RC foldLR_test(
	const char * test_label,
	const char * test_description,
	const Matrix< V > &A,
	const Matrix< M > &mask,
	T initial,
	T expected,
	const Monoid &monoid,
	bool skip_masked = false,
	bool skip_unmasked = false,
	RC expected_unmasked_rc = SUCCESS,
	RC expected_masked_rc = SUCCESS
) {
	RC rc = foldl_test< descr >(
		test_label, test_description,
		A, mask,
		initial, expected,
		monoid,
		skip_masked, skip_unmasked,
		expected_unmasked_rc, expected_masked_rc
	);
	return rc ? rc : foldr_test< descr >(
		test_label, test_description,
		A, mask,
		initial, expected,
		monoid,
		skip_masked, skip_unmasked,
		expected_unmasked_rc, expected_masked_rc
	);
}

template< typename T, typename M >
struct input {
	const Matrix< T > &A;
	const Matrix< M > &mask;

	// Default constructor for distributed backends
	input(
		const Matrix< T > &A = {0,0},
		const Matrix< M > &mask = {0,0}
	) : A( A ), mask( mask ) {}
};

template< typename T, typename M >
void grb_program( const input< T, M > &in, RC &rc ) {

	const Matrix< T > &I = in.A;
	const Matrix< M > &mask = in.mask;

	const long n = nnz( I );

	/**
	 * Test case 1:
	 *
	 * A simple additive reduction with the same types for the nzs and the reduction result.
	 * * Initial value is 0
	 * * Expected unmasked result: n
	 * * Expected masked result: 0
	 */
	{
		rc = foldLR_test(
			"1",
			"A simple reduction(+) with the same types for the nzs and the reduction "
			"result.",
			I, mask,
			static_cast< NzType >( 0 ), static_cast< NzType >( n ),
			Monoid< operators::add< NzType >, identities::zero >()
		);
		if( rc ) { return; }
	}

	/**
	 * Test case 2:
	 *
	 * A simple additive reduction with the same types for the nzs and the reduction result.
	 * * Initial value is n
	 * * Expected result: 2*n
	 */
	{
		rc = foldLR_test(
			"2",
			"A simple reduction(+) with the same types for the nzs and the reduction "
			"result.",
			I, mask,
			static_cast< NzType >( n ), static_cast< NzType >( 2 * n ),
			Monoid< operators::add< NzType >, identities::zero >()
		);
		if( rc ) { return; }
	}

	/**
	 * Test case 3:
	 *
	 * A simple additive reduction with different types for
	 * the nzs and the reduction result (int <- int + NzType).
	 * * Initial value is 0
	 * * Expected result: n
	 */
	{
		rc = foldl_test(
			"3",
			"A simple reduction(+) with different types for the nzs and the reduction "
			"result (int <- int + NzType).",
			I, mask,
			static_cast< int >( 0 ), static_cast< int >( n ),
			Monoid< operators::add< int, NzType, int >, identities::zero >()
		);
		if( rc ) { return; }
		rc = foldr_test(
			"3",
			"A simple reduction(+) with different types for the nzs and the reduction "
			"result (int <- NzType + int).",
			I, mask,
			static_cast< int >( 0 ), static_cast< int >( n ),
			Monoid< operators::add< NzType, int, int >, identities::zero >()
		);
		if( rc ) { return; }
	}

	/**
	 * Test case 4:
	 *
	 * A simple additive reduction with different types for
	 * the nzs and the reduction result (int <- int + NzType).
	 * * Initial value is n
	 * * Expected result: 2*n
	 */
	{
		rc = foldl_test(
			"4",
			"A simple reduction(+) with different types for the nzs and the reduction "
			"result (int <- int + NzType).",
			I, mask,
			static_cast< int >( n ), static_cast< int >( 2 * n ),
			Monoid< operators::add< int, NzType, int >, identities::zero >()
		);
		if( rc ) { return; }
		rc = foldr_test(
			"4",
			"A simple reduction(+) with different types for the nzs and the reduction "
			"result (int <- NzType + int).",
			I, mask,
			static_cast< int >( n ), static_cast< int >( 2 * n ),
			Monoid< operators::add< NzType, int, int >, identities::zero >()
		);
		if( rc ) { return; }
	}

	/**
	 * Test case 5:
	 *
	 * A simple multiplicative reduction with the same types for
	 * the nzs and the reduction result.
	 * * Initial value is 0
	 * * Expected result: 0
	 */
	{
		rc = foldLR_test(
			"5",
			"A simple reduction(*) with the same types for the nzs and the reduction "
			"result.",
			I, mask,
			static_cast< NzType >( 0 ), static_cast< NzType >( 0 ),
			Monoid< operators::mul< NzType >, identities::one >()
		);
		if( rc ) { return; }
	}

	/**
	 * Test case 6:
	 *
	 * A simple multiplicative reduction with the same types for
	 * the nzs and the reduction result.
	 * * Initial value is 1
	 * * Expected result: 1
	 */
	{
		rc = foldLR_test(
			"6",
			"A simple reduction(*) with the same types for the nzs and the reduction "
			"result.",
			I, mask,
			static_cast< NzType >( 1 ), static_cast< NzType >( 1 ),
			Monoid< operators::mul< NzType >, identities::one >()
		);
		if( rc ) { return; }
	}

	/**
	 * Test case 7:
	 *
	 * A simple multiplicative reduction with different types for
	 * the nzs and the reduction result (size_t <- size_t * NzType).
	 * * Initial value is 0
	 * * Expected result: 0
	 */
	{
		rc = foldl_test(
			"7",
			"A simple reduction(*) with different types for the nzs and the reduction "
			"result (int <- int * NzType).",
			I, mask,
			static_cast< size_t >( 0 ), static_cast< size_t >( 0 ),
			Monoid< operators::mul< size_t, NzType, size_t >, identities::one >()
		);
		if( rc ) { return; }
		rc = foldr_test(
			"7",
			"A simple reduction(*) with different types for the nzs and the reduction "
			"result (int <- int * NzType).",
			I, mask,
			static_cast< size_t >( 0 ), static_cast< size_t >( 0 ),
			Monoid< operators::mul< NzType, size_t, size_t >, identities::one >()
		);
		if( rc ) { return; }
	}

	/**
	 * Test case 8:
	 *
	 * A simple multiplicative reduction with different types for
	 * the nzs and the reduction result (size_t <- size_t * NzType).
	 * * Initial value is 1
	 * * Expected result: 1
	 */
	{
		rc = foldl_test(
			"8",
			"A simple reduction(*) with different types for the nzs and the reduction "
			"result (int <- int * NzType).",
			I, mask,
			static_cast< size_t >( 1 ), static_cast< size_t >( 1 ),
			Monoid< operators::mul< size_t, NzType, size_t >, identities::one >()
		);
		if( rc ) { return; }
		rc = foldr_test(
			"8",
			"A simple reduction(*) with different types for the nzs and the reduction "
			"result (int <- int * NzType).",
			I, mask,
			static_cast< size_t >( 1 ), static_cast< size_t >( 1 ),
			Monoid< operators::mul< NzType, size_t, size_t >, identities::one >()
		);
		if( rc ) { return; }
	}

	/**
	 * Test case 9:
	 *
	 * A simple binary equal reduction with different types for
	 * the nzs and the reduction result (bool <- bool == NzType).
	 * * Initial value is true
	 * * Expected result: true
	 */
	{
		rc = foldl_test(
			"9",
			"A simple reduction(==) with different types for the nzs and the reduction "
			"result (bool <- bool == NzType).",
			I, mask,
			static_cast< bool >( true ), static_cast< bool >( true ),
			Monoid< operators::equal< bool, NzType, bool >, identities::logical_true >()
		);
		if( rc ) { return; }
		rc = foldr_test(
			"9",
			"A simple reduction(==) with different types for the nzs and the reduction "
			"result (bool <- bool == NzType).",
			I, mask,
			static_cast< bool >( true ), static_cast< bool >( true ),
			Monoid< operators::equal< NzType, bool, bool >, identities::logical_true >()
		);
		if( rc ) { return; }
	}

	/**
	 * Test case 10:
	 *
	 * A simple binary logical_or reduction with different types for
	 * the nzs and the reduction result (bool <- bool || NzType).
	 * * Initial value is false
	 * * Expected result: true
	 */
	{
		rc = foldl_test(
			"10",
			"A simple reduction(||) with different types for the nzs and the reduction "
			"result (bool <- bool || NzType).",
			I, mask,
			static_cast< bool >( false ), static_cast< bool >( true ),
			Monoid< operators::logical_or< bool, NzType, bool >, identities::logical_false >()
		);
		if( rc ) { return; }
		rc = foldr_test(
			"10",
			"A simple reduction(||) with different types for the nzs and the reduction "
			"result (bool <- bool || NzType).",
			I, mask,
			static_cast< bool >( false ), static_cast< bool >( true ),
			Monoid< operators::logical_or< NzType, bool, bool >, identities::logical_false >()
		);
		if( rc ) { return; }
	}

	/**
	 * Test case 11:
	 *
	 * Reduction with an empty mask.
	 * * Initial value is 4
	 * * Expected result: 4
	 */
	{
		Matrix< void > empty_mask( nrows( I ), ncols( I ), 0 );
		rc = foldLR_test(
			"11",
			"Reduction with an empty mask.",
			I, empty_mask,
			static_cast< NzType >( 4 ), static_cast< NzType >( 4 ),
			Monoid< operators::add< NzType >, identities::zero >(),
			false, true
		);
		if( rc ) { return; }
	}

	/**
	 * Test case 12:
	 *
	 * Reduction with a dense void mask.
	 * * Initial value is 0
	 * * Expected result: n
	 */
	{
		Matrix< void > dense_mask( nrows( I ), ncols( I ), nrows( I ) * ncols( I ) );
		std::vector< size_t > rows( nrows( I ) * ncols( I ) ), cols( nrows( I ) * ncols( I ) );
		for( size_t x = 0; x < nrows( I ); x++ ) {
			std::fill( rows.begin() + x * ncols( I ),
				rows.begin() + ( x + 1 ) * ncols( I ), x );
			std::iota( cols.begin() + x * ncols( I ),
				cols.begin() + ( x + 1 ) * ncols( I ), 0 );
		}
		if( SUCCESS !=
			buildMatrixUnique( dense_mask, rows.data(), cols.data(),
				nrows( I ) * ncols( I ), SEQUENTIAL )
		) {
			std::cerr << "Failed to build dense mask" << std::endl;
			rc = FAILED;
			return;
		}
		rc = foldLR_test(
			"12",
			"Reduction with a dense void mask.",
			I, dense_mask,
			static_cast< NzType >( 0 ), static_cast< NzType >( n ),
			Monoid< operators::add< NzType >, identities::zero >(),
			false, true
		);
		if( rc ) { return; }
	}

	/**
	 * Test case 13:
	 *
	 * Reduction with a dense int mask.
	 * * Initial value is 0
	 * * Expected result: n
	 */
	{
		Matrix< int > dense_mask( nrows( I ), ncols( I ), nrows( I ) * ncols( I ) );
		std::vector< size_t > rows( nrows( I ) * ncols( I ) ),
			cols( nrows( I ) * ncols( I ) );
		for( size_t x = 0; x < nrows( I ); x++ ) {
			std::fill( rows.begin() + x * ncols( I ),
				rows.begin() + ( x + 1 ) * ncols( I ), x );
			std::iota( cols.begin() + x * ncols( I ),
				cols.begin() + ( x + 1 ) * ncols( I ), 0 );
		}
		std::vector< int > vals( nrows( I ) * ncols( I ), 1 );
		if( SUCCESS !=
			buildMatrixUnique( dense_mask, rows.data(), cols.data(), vals.data(),
				vals.size(), SEQUENTIAL )
		) {
			std::cerr << "Failed to build dense mask" << std::endl;
			rc = FAILED;
			return;
		}
		rc = foldLR_test(
			"13",
			"Reduction with a dense int mask.",
			I, dense_mask,
			static_cast< NzType >( 0 ), static_cast< NzType >( n ),
			Monoid< operators::add< NzType >, identities::zero >(),
			false, true
		);
		if( rc ) { return; }
	}

	/**
	 * Test case 14:
	 *
	 * Reduction with a dense int mask, full of zero, except for the first nz.
	 * * Initial value is 0
	 * * Expected result: 1
	 */
	{
		Matrix< int > dense_mask( nrows( I ), ncols( I ), nrows( I ) * ncols( I ) );
		std::vector< size_t > rows( nrows( I ) * ncols( I ) ),
			cols( nrows( I ) * ncols( I ) );
		for( size_t x = 0; x < nrows( I ); x++ ) {
			std::fill( rows.begin() + x * ncols( I ),
				rows.begin() + ( x + 1 ) * ncols( I ), x );
			std::iota( cols.begin() + x * ncols( I ),
				cols.begin() + ( x + 1 ) * ncols( I ), 0 );
		}
		std::vector< int > vals( nrows( I ) * ncols( I ), 0 );
		for( const auto &e : I ) {
			vals[ e.first.first * ncols( I ) + e.first.second ] = 1;
			break;
		}
		if( SUCCESS !=
			buildMatrixUnique( dense_mask, rows.data(), cols.data(), vals.data(),
				vals.size(), SEQUENTIAL )
		) {
			std::cerr << "Failed to build dense mask" << std::endl;
			rc = FAILED;
			return;
		}
		rc = foldLR_test(
			"14",
			"Reduction with a dense int mask, matching only the first nz.",
			I, dense_mask,
			static_cast< NzType >( 0 ), static_cast< NzType >( 1 ),
			Monoid< operators::add< NzType >, identities::zero >(),
			false, true
		);
		if( rc ) { return; }
	}

	/**
	 * Test case 15:
	 *
	 * Reduction with a dense int mask, full of zero, except for the last nz.
	 * * Initial value is 0
	 * * Expected result: 1
	 */
	{
		Matrix< int > dense_mask( nrows( I ), ncols( I ), nrows( I ) * ncols( I ) );
		std::vector< size_t > rows( nrows( I ) * ncols( I ) ),
			cols( nrows( I ) * ncols( I ) );
		for( size_t x = 0; x < nrows( I ); x++ ) {
			std::fill( rows.begin() + x * ncols( I ),
				rows.begin() + ( x + 1 ) * ncols( I ), x );
			std::iota( cols.begin() + x * ncols( I ),
				cols.begin() + ( x + 1 ) * ncols( I ), 0 );
		}
		std::vector< int > vals( nrows( I ) * ncols( I ), 0 );
		size_t previous_idx = 0;
		for( const auto e : I )
			previous_idx = e.first.first * ncols( I ) + e.first.second;
		vals[ previous_idx ] = 1;
		if( SUCCESS !=
			buildMatrixUnique( dense_mask, rows.data(), cols.data(), vals.data(),
				vals.size(), SEQUENTIAL )
		) {
			std::cerr << "Failed to build dense mask" << std::endl;
			rc = FAILED;
			return;
		}
		rc = foldLR_test(
			"15",
			"Reduction with a dense int mask, matching only the last nz.",
			I, dense_mask,
			static_cast< NzType >( 0 ), static_cast< NzType >( 1 ),
			Monoid< operators::add< NzType >, identities::zero >(),
			false, true
		);
		if( rc ) { return; }
	}

	/**
	 * Test case 16:
	 *
	 * Reduction with a dense void mask, with the descriptors::add_identity.
	 * * Initial value is 0
	 * * Expected result: 2*n
	 */
	{
		size_t nnz =  nrows( I ) * ncols( I );
		Matrix< void > dense_mask( nrows( I ), ncols( I ), nnz);
		std::vector< size_t > rows( nnz ), cols( nnz );
		Semiring<
			operators::add< NzType >, operators::mul< NzType >,
			identities::zero, identities::one
		> semiring;
		for( size_t x = 0; x < nrows( I ); x++ ) {
			std::fill( rows.begin() + x * ncols( I ),
				rows.begin() + ( x + 1 ) * ncols( I ), x );
			std::iota( cols.begin() + x * ncols( I ),
				cols.begin() + ( x + 1 ) * ncols( I ), 0 );
		}
		if( SUCCESS !=
			buildMatrixUnique( dense_mask, rows.data(), cols.data(), rows.size(),
				SEQUENTIAL )
		) {
			std::cerr << "Failed to build dense mask" << std::endl;
			rc = FAILED;
			return;
		}
		rc = foldLR_test< descriptors::add_identity >(
			"16",
			"Reduction with a dense void mask, with the descriptors::add_identity.",
			I, dense_mask,
			static_cast< NzType >( 0 ), static_cast< NzType >( n + std::min( nrows( I ),
				ncols( I ) ) ),
			semiring,
			false, false
		);
		if( rc ) { return; }
	}

	/**
	 * Test case 17:
	 *
	 * Reduction with mismatching dimensions between
	 * an empty void-mask and the input matrix.
	 * * Expected RC: MISMATCH (masked only)
	 * * Initial value is 4 (unmasked only)
	 * * Expected result: 4 (unmasked only)
	 */
	{
		Matrix< void > void_mask( nrows( I ) + 1, ncols( I ) + 1, 0 );
		rc = foldLR_test(
			"17",
			"Reduction with an empty void mask. Mismatching dimensions, should fail.",
			I, void_mask,
			static_cast< NzType >( 4 ), static_cast< NzType >( 4 ),
			Monoid< operators::add< NzType >, identities::zero >(),
			false, false,
			SUCCESS, MISMATCH
		);
		if( rc ) { return; }
	}

	/**
	 * Test case 18:
	 *
	 * Reduction with mismatching dimensions between an empty
	 * int-mask and the input matrix.
	 * * Expected RC: MISMATCH (masked only)
	 * * Initial value is 4 (unmasked only)
	 * * Expected result: 4 (unmasked only)
	 */
	{
		Matrix< int > void_mask( nrows( I ) + 1, ncols( I ) + 1, 0 );
		rc = foldLR_test(
			"18",
			"Reduction with an empty int mask. Mismatching dimensions, should fail..",
			I, void_mask,
			static_cast< NzType >( 4 ), static_cast< NzType >( 4 ),
			Monoid< operators::add< NzType >, identities::zero >(),
			false, false,
			SUCCESS, MISMATCH
		);
		if( rc ) { return; }
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
		std::cerr << "Usage: " << argv[ 0 ] << " [ n ]\n";
		std::cerr << "  -n (optional, default is 10): an even integer, the test "
				  << "size.\n";
		return 1;
	}

	std::cout << "This is functional test " << argv[ 0 ] << "\n";
	Launcher< AUTOMATIC > launcher;
	RC rc = SUCCESS;

	if( !rc ) { // Identity square-matrix
		Matrix< NzType > I( n, n );
		std::vector< size_t > I_rows( n ), I_cols( n );
		std::vector< NzType > I_vals( n, 1.f );
		std::iota( I_rows.begin(), I_rows.end(), 0 );
		std::iota( I_cols.begin(), I_cols.end(), 0 );
		if( SUCCESS !=
			buildMatrixUnique( I, I_rows.data(), I_cols.data(), I_vals.data(),
				I_vals.size(), SEQUENTIAL )
		) {
			std::cerr << "Failed to build identity matrix" << std::endl;
			rc = FAILED;
			return 2;
		}
		Matrix< void > mask( n, n );
		if( SUCCESS  !=
			buildMatrixUnique( mask, I_rows.data(), I_cols.data(), I_rows.size(),
				SEQUENTIAL )
		) {
			std::cerr << "Failed to build identity mask" << std::endl;
			rc = FAILED;
			return 3;
		}
		std::cout << "-- Running test 01: Identity square matrix of size n = "
			<< n << std::endl;
		input< NzType, void > input(I, mask);
		if( launcher.exec( &grb_program, input, rc, true ) != SUCCESS ) {
			std::cerr << "Launching test 01 FAILED\n";
			return 4;
		}
		std::cout << std::endl << std::flush;
	}

	if( !rc ) { // Build a square-matrix with n 1s on the first row
		Matrix< NzType > I( n, n );
		std::vector< size_t > I_rows( n, 0 ), I_cols( n );
		std::vector< NzType > I_vals( n, 1.f );
		std::iota( I_cols.begin(), I_cols.end(), 0 );
		if( SUCCESS !=
			buildMatrixUnique( I, I_rows.data(), I_cols.data(), I_vals.data(),
				I_vals.size(), SEQUENTIAL )
		) {
			std::cerr << "Failed to build matrix with n 1s on the first row" << std::endl;
			rc = FAILED;
			return 5;
		}
		Matrix< void > mask( n, n );
		if( SUCCESS !=
			buildMatrixUnique( mask, I_rows.data(), I_cols.data(), I_rows.size(),
				SEQUENTIAL )
		) {
			std::cerr << "Failed to build mask with n 1s on the first row" << std::endl;
			rc = FAILED;
			return 6;
		}
		std::cout << "-- Running test 02: Square matrix of size n = "
			<< n << ", with n 1s on the first row" << std::endl;
		input< NzType, void > input(I, mask);
		if( launcher.exec( &grb_program, input, rc, true ) != SUCCESS ) {
			std::cerr << "Launching test 02 FAILED\n";
			return 7;
		}
		std::cout << std::endl << std::flush;
	}

	if( !rc ) { // Square-matrix with n 1s on the first column
		Matrix< NzType > I( n, n );
		std::vector< size_t > I_rows( n ), I_cols( n, 0 );
		std::vector< NzType > I_vals( n, 1.f );
		std::iota( I_rows.begin(), I_rows.end(), 0 );
		if( SUCCESS !=
			buildMatrixUnique( I, I_rows.data(), I_cols.data(), I_vals.data(),
				I_vals.size(), SEQUENTIAL )
		) {
			std::cerr << "Failed to build matrix with n 1s on the first column"
				<< std::endl;
			rc = FAILED;
			return 8;
		}
		Matrix< void > mask( n, n );
		if( SUCCESS !=
			buildMatrixUnique( mask, I_rows.data(), I_cols.data(), I_rows.size(),
				SEQUENTIAL )
		) {
			std::cerr << "Failed to build mask with n 1s on the first column"
				<< std::endl;
			rc = FAILED;
			return 9;
		}
		std::cout << "-- Running test 03: Square matrix of size n = "
			<< n << ", with n 1s on the first column" << std::endl;
		input< NzType, void > input(I, mask);
		if( launcher.exec( &grb_program, input, rc, true ) != SUCCESS ) {
			std::cerr << "Launching test 03 FAILED\n";
			return 10;
		}
		std::cout << std::endl << std::flush;
	}

	if( !rc ) { // Building a square-matrix with n 1s on the first row and column
		Matrix< NzType > I( n, n );
		std::vector< size_t > I_rows( 2 * n - 1, 0 ), I_cols( 2 * n - 1, 0 );
		std::vector< NzType > I_vals( 2 * n - 1, 1.f );
		std::iota( I_rows.begin() + n, I_rows.end(), 1 );
		std::iota( I_cols.begin(), I_cols.begin() + n, 0 );
		if( SUCCESS !=
			buildMatrixUnique( I, I_rows.data(), I_cols.data(), I_vals.data(),
				I_vals.size(), SEQUENTIAL )
		) {
			std::cerr << "Failed to build matrix with n 1s on the first row and column"
				<< std::endl;
			rc = FAILED;
			return 11;
		}
		Matrix< void > mask( n, n );
		if( SUCCESS !=
			buildMatrixUnique( mask, I_rows.data(), I_cols.data(), I_rows.size(),
				SEQUENTIAL )
		) {
			std::cerr << "Failed to build mask with n 1s on the first row and column"
				<< std::endl;
			rc = FAILED;
			return 12;
		}
		std::cout << "-- Running test 04: Square matrix of size n = "
			<< n << ", with n 1s on the first row and column" << std::endl;
		input< NzType, void > input( I, mask );
		if( launcher.exec( &grb_program, input, rc, true ) != SUCCESS ) {
			std::cerr << "Launching test 04 FAILED\n";
			return 13;
		}
		std::cout << std::endl << std::flush;
	}

	if( !rc ) { // Building a [1 row, n columns] matrix filled with 1s
		Matrix< NzType > I( 1, n );
		std::vector< size_t > I_rows( n, 0 ), I_cols( n, 0 );
		std::vector< NzType > I_vals( n, 1.f );
		std::iota( I_cols.begin(), I_cols.end(), 0 );
		if( SUCCESS !=
			buildMatrixUnique( I, I_rows.data(), I_cols.data(), I_vals.data(),
				I_vals.size(), SEQUENTIAL )
		) {
			std::cerr << "Failed to build matrix with n 1s on the first row"
				<< std::endl;
			rc = FAILED;
			return 14;
		}
		Matrix< void > mask( 1, n );
		if( SUCCESS !=
			buildMatrixUnique( mask, I_rows.data(), I_cols.data(), I_rows.size(),
				SEQUENTIAL )
		) {
			std::cerr << "Failed to build mask with n 1s on the first row" << std::endl;
			rc = FAILED;
			return 11;
		}
		std::cout << "-- Running test 05: [1-row, n = "
			<< n << " columns] matrix, filled with 1s" << std::endl;
		input< NzType, void > input(I, mask);
		if( launcher.exec( &grb_program, input, rc, true ) != SUCCESS ) {
			std::cerr << "Launching test 04 FAILED\n";
			return 15;
		}
		std::cout << std::endl << std::flush;
	}

	if( !rc ) { // Building a [n rows, 1 column] matrix filled with 1s
		Matrix< NzType > I( n, 1 );
		std::vector< size_t > I_rows( n, 0 ), I_cols( n, 0 );
		std::vector< NzType > I_vals( n, 1.f );
		std::iota( I_rows.begin(), I_rows.end(), 0 );
		if( SUCCESS !=
			buildMatrixUnique( I, I_rows.data(), I_cols.data(), I_vals.data(),
				I_vals.size(), SEQUENTIAL )
		) {
			std::cerr << "Failed to build matrix with n 1s on the first column"
				<< std::endl;
			rc = FAILED;
			return 16;
		}
		Matrix< void > mask( n, 1 );
		if( SUCCESS !=
			buildMatrixUnique( mask, I_rows.data(), I_cols.data(), I_rows.size(),
				SEQUENTIAL )
		) {
			std::cerr << "Failed to build mask with n 1s on the first column" << std::endl;
			rc = FAILED;
			return 17;
		}
		std::cout << "-- Running test 06: [n = "
					<< n << " rows, 1 column] matrix, filled with 1s" << std::endl;
		input< NzType, void > input(I, mask);
		if( launcher.exec( &grb_program, input, rc, true ) != SUCCESS ) {
			std::cerr << "Launching test 06 FAILED\n";
			return 18;
		}
		std::cout << std::endl << std::flush;
	}

	std::cerr << std::flush;
	if( rc != SUCCESS ) {
		std::cout << std::flush << "Test FAILED (rc = " << toString( rc ) << ")"
			<< std::endl;
		return 19;
	}

	std::cout << std::flush << "Test OK" << std::endl;
	return 0;
}

