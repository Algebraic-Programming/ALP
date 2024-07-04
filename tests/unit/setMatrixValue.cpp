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

/**
 * @file
 *
 * Tests the grb::set( matrix, mask, value ) primitive under all permissable
 * descriptors.
 *
 * @author Benjamin Lozes
 * @author A. N. Yzelman
 * @date August 31st, 2023
 */

#include <iostream>
#include <sstream>
#include <numeric>
#include <algorithm>
#include <vector>

#include <graphblas.hpp>

#include <graphblas/algorithms/matrix_factory.hpp>


using namespace grb;

template< typename T >
class Expected {

	private:

		T value;


	public:

		Expected() : value() {}

		template< typename U >
		void set( const U &in ) {
			value = static_cast< T >( in );
		}

		const T * getPointer() const noexcept {
			return &value;
		}

};

template<>
class Expected< void > {

	public:

		Expected() {}

		template< typename U >
		void set( const U &in ) {
			(void) in;
		}

		inline const void * getPointer() const noexcept {
			return nullptr;
		}

};

/**
 * Checks whether all entries in a given matrix \a A have a given value,
 * and checks whether all those entries are at an off-diagonal position
 * defined by \a row_offset.
 */
template< bool no_even_rows, bool no_odd_rows, typename T >
RC check_all(
	grb::Matrix< T > &A,
	Expected< T > expected_value_even, Expected< T > expected_value_odd,
	const size_t row_offset
) {
	const T * const even_value_p = expected_value_even.getPointer();
	const T * const odd_value_p  = expected_value_odd.getPointer();
	return std::all_of( A.cbegin(), A.cend(),
		[&even_value_p,&odd_value_p,&row_offset](
			const std::pair< std::pair< size_t, size_t >, T > &entry
		) {
			const T expected_value = entry.first.first % 2 == 0
				? *even_value_p
				: *odd_value_p;
			const bool unexpected = (entry.first.first % 2 == 0 && no_even_rows) ||
				(entry.first.first % 2 == 1 && no_odd_rows);
			const bool valMatch = entry.second == expected_value;
			const bool coorMatch = entry.first.first + row_offset == entry.first.second;
			const bool ret = valMatch && coorMatch && !unexpected;
			if( !ret ) {
				if( unexpected ) {
					std::cerr << "Value " << entry.second << " at coordinates ( "
						<< entry.first.first << ", " << entry.first.second << " ) while no "
						<< "value was expected at this coordinate.\n";
				} else {
					std::cerr << "Value = " << entry.second << ", expected " << expected_value
						<< " at coordinates ( " << entry.first.first << ", " << entry.first.second
						<< " ), expected diagonal with row offset " << row_offset << "\n";
				}
			}
			return ret;
		}
	) ? SUCCESS : FAILED;
}

/**
 * This variant is for pattern (void) matrices, and only checks for the
 * off-diagonal position of the entries. The interface matches that of the
 * generic check_all.
 */
template< bool no_even_rows, bool no_odd_rows >
RC check_all(
	grb::Matrix< void > &A,
	Expected< void > expected_value_even, Expected< void > expected_value_odd,
	const size_t row_offset
) {
	(void) expected_value_even;
	(void) expected_value_odd;
	return std::all_of( A.cbegin(), A.cend(),
		[&row_offset](
			const std::pair< size_t, size_t > &entry
		) {
			const bool unexpected = (entry.first % 2 == 0 && no_even_rows) ||
				(entry.first % 2 == 1 && no_odd_rows);
			const bool coorMatch = entry.first + row_offset == entry.second;
			const bool ret = coorMatch && !unexpected;
			if( !ret ) {
				if( unexpected ) {
					std::cerr << "Value " << entry.second << " at coordinates ( "
						<< entry.first << ", " << entry.second << " ) while no "
						<< "value was expected at this coordinate.\n";
				} else {
					std::cerr << "Value at coordinates ( " << entry.first << ", "
						<< entry.second << " ), expected diagonal with row offset "
						<< row_offset << "\n";
				}
			}
			return ret;
		}
	) ? SUCCESS : FAILED;
}

/**
 * @tparam left Whether the output matrix is off_diagonal (left) or identity
 *              (right). The difference in test is that for if selecting left,
 *              then the capacity of the output may have to resize. If the
 *              other way around, no resizing is ever needed.
 */
template< grb::Descriptor descr, typename T, bool left >
void identity_test( const size_t &n, grb::RC &rc ) {
	rc = SUCCESS;
	if( n < 2 ) {
		std::cout << "\t test does not apply for n smaller than 2\n";
		return;
	}

	// construct containers
	grb::Matrix< int > off_diagonal( 0, 0, 0 );
	grb::Matrix< T > identity( 0, 0, 0 );
	try {
		grb::Matrix< int > off_diagonal_alloc =
			algorithms::matrices< int >::eye( n, n, 7, 1 );
		grb::Matrix< T > identity_alloc = algorithms::matrices< T >::identity( n );
		off_diagonal = std::move( off_diagonal_alloc );
		identity = std::move( identity_alloc );
	} catch( ... ) {
		std::cerr << "\t error during construction of the identity matrix\n";
		rc = FAILED;
		return;
	}
	rc = grb::eWiseLambda(
		[&off_diagonal](const size_t i, const size_t j, int &v ) {
			(void) j;
			if( i % 2 == 0 ) { v = 0; }
		}, off_diagonal );
	if( rc != SUCCESS ) {
		std::cerr << "\t error during application of element-wise lambda-- test "
			<< "could not initialise\n";
		return;
	}

	// at this point:
	//  - identity is an n by n identity matrix
	//  - off_diagonal is an n by n matrix with values at coordinates above its
	//    main diagonal. On even-numbered rows, the value at corresponding
	//    coordinates is 7. On odd-numbered rows, the value is 0. This helps detect
	//    differing behaviour for structural vs. non-structural masking.

	if( nnz( off_diagonal ) != n - 1 ) {
		std::cerr << "\t verification of off-diagonal construction failed; "
			"expected " << n << " elements, got " << nnz( off_diagonal ) << "\n";
		rc = FAILED;
	}
	if( nnz( identity ) != n ) {
		std::cerr << "\t verification of off-diagonal construction failed; "
			"expected " << n << " elements, got " << nnz( off_diagonal ) << "\n";
		rc = FAILED;
	}

	// set expected values for validating construction
	Expected< int > expected_left_odd, expected_left_even;
	Expected< T > expected_right_odd, expected_right_even;
	expected_left_odd.set( 7 );
	expected_left_even.set( 0 );
	expected_right_odd.set( 1 );
	expected_right_even.set( 1 );

	// validate construction
	{
		RC local_rc = check_all< false, false >(
			off_diagonal, expected_left_even, expected_left_odd, 1 );
		if( local_rc != SUCCESS ) {
			std::cerr << "\t verification of off-diagonal construction failed: "
				<< "at least one unexpected matrix element found\n";
			if( rc == SUCCESS ) { rc = local_rc; }
		}
		local_rc = check_all< false, false >(
			identity, expected_right_even, expected_right_odd, 0 );
			if( local_rc != SUCCESS ) {
			std::cerr << "\t verification of identity construction failed: "
				<< "at least one unexpected matrix element found\n";
			if( rc == SUCCESS ) { rc = local_rc; }
		}
	}
	if( rc != SUCCESS ) { return; }

	// perform the set, resize phase
	if( left ) {
		rc = grb::set< descr >( off_diagonal, identity, 3, RESIZE );
	} else {
		rc = grb::set< descr >( identity, off_diagonal, 3, RESIZE );
	}
	if( rc != SUCCESS ) {
		std::cerr << "\t resize failed: " << grb::toString( rc ) << "\n";
		return;
	}

	// check capacity
	const size_t expected_nnz = left
		? ((descr & descriptors::invert_mask)
			? 0
			: n
		)
		: ((descr & descriptors::structural)
			? (n-1)
			: ((descr & descriptors::invert_mask)
				? (n/2)
				: ((n-1)/2)
			  )
		);
	{
		const size_t cap = left
		       ? grb::capacity( off_diagonal )
		       : grb::capacity( identity );
		if( cap < expected_nnz ) {
			std::cerr << "\t resize failed to achieve correct capacity for "
				<< "off_diagonal: got " << cap << " but require at least " << expected_nnz
				<< "\n";
			rc = FAILED;
			return;
		}
	}

	// perform the set, execute phase
	if( left ) {
		rc = grb::set< descr >( off_diagonal, identity, 3, EXECUTE );
	} else {
		rc = grb::set< descr >( identity, off_diagonal, 3, EXECUTE );
	}
	if( rc != SUCCESS ) {
		std::cerr << "\t execute failed\n";
		return;
	}

	// set expected value for tests, accounting for possible void T
	expected_left_odd.set( 3 );
	expected_left_even.set( 3 );
	if( descr & descriptors::invert_mask ) {
		expected_right_even.set( 3 );
		expected_right_odd.set( 17 );
		// we use 17 above here as it is a never-encountered value. There should be no
		// nnzs on odd-numbered rows in this case
	} else if( descr & descriptors::structural ) {
		expected_right_even.set( 3 );
		expected_right_odd.set( 3 );
	} else {
		expected_right_even.set( 17 ); // (see above)
		expected_right_odd.set( 3 );
	}

	// check output
	{
		const size_t actual_nnz = left ? nnz( off_diagonal ) : nnz( identity );
		if( actual_nnz != expected_nnz ) {
			std::cerr << "\t unexpected number of nonzeroes: got " << actual_nnz << ", "
				<< "expected " << expected_nnz << "\n";
			rc = FAILED;
		}
		RC local_rc = PANIC;
		if( left ) {
			local_rc = check_all< false, false >(
				off_diagonal, expected_left_even, expected_left_odd, 0 );
		} else {
			constexpr bool flag = (descr & descriptors::invert_mask);
			constexpr bool nstr = !(descr & descriptors::structural);
			local_rc = check_all< nstr && !flag, nstr && flag >(
				identity, expected_right_even, expected_right_odd, 1 );
		}
		if( local_rc == FAILED ) {
			std::cerr << "\t at least one unexpected output entry found\n";
			if( rc == SUCCESS ) { rc = local_rc; }
		}
	}

	// done
}

template< grb::Descriptor descr >
void self_identity_test( const size_t &n, grb::RC &rc ) {
	rc = SUCCESS;

	grb::Matrix< int > Identity( 0, 0, 0 );
	try {
		grb::Matrix< int > Identity_alloc =
			algorithms::matrices< int >::identity( n );
		Identity = std::move( Identity_alloc );
	} catch( ... ) {
		std::cerr << "\t error during construction of the identity matrix\n";
		rc = FAILED;
		return;
	}
	rc = grb::eWiseLambda(
		[&Identity](const size_t i, const size_t j, int &v ) {
#ifdef NDEBUG
			(void) j;
#endif
			assert( i == j );
			assert( v == 1 );
			if( i % 2 == 0 ) {
				v = 0;
			}
		}, Identity );
	if( rc != SUCCESS ) {
		std::cerr << "\t error during application of element-wise lambda-- test "
			<< "could not initialise\n";
		return;
	}

	if( nnz( Identity ) != n ) {
		std::cerr << "\t diagonal has " << nnz( Identity ) << " elements, expected "
			<< n << "\n";
		rc = FAILED;
		return;
	}

	// Check first if the matrix is correctly initialised
	Expected< int > expected_even, expected_odd;
	expected_even.set( 0 );
	expected_odd.set( 1 );
	rc = check_all< false, false >(
		Identity, expected_even, expected_odd, 0 );
	if( rc != SUCCESS ) {
		std::cerr << "\t initialisation FAILED: rc is "
			<< grb::toString(rc) << "\n";
		return;
	}

	// Try to set the matrix to 2s ( RESIZE )
	size_t expected_nnz;
	if( descr & descriptors::structural ) {
		expected_nnz = n;
		expected_even.set( 2 );
		expected_odd.set( 2 );
	} else if( descr & descriptors::invert_mask ) {
		expected_nnz = ( n + 1 ) / 2;
		expected_even.set( 2 );
		expected_odd.set( 17 ); // never-encountered value
	} else {
		expected_nnz = n / 2;
		expected_even.set( 17 );
		expected_odd.set( 2 );
	}
	rc = grb::set< descr >( Identity, Identity, 2UL, grb::Phase::RESIZE );
	if( rc != SUCCESS ) {
		std::cerr << "\t set identity matrix diagonal to 2 (RESIZE phase) FAILED: "
			<< "rc is " << grb::toString(rc) << "\n";
		return;
	}
	// As the RESIZE phase is useless, the matrix should not be resized.
	if( grb::capacity( Identity ) < n ) {
		std::cerr << "\t unexpected matrix capacity: " << grb::capacity( Identity )
			<< ", expected at least " << n << "\n";
		rc = FAILED;
		return;
	}

	// Try to set the matrix to 2s ( EXECUTE )
	rc = grb::set< descr >( Identity, Identity, 2UL, grb::Phase::EXECUTE );
	if( rc != SUCCESS ) {
		std::cerr << "\t set identity matrix diagonal to 2s ( EXECUTE ) FAILED: "
			<< "rc is " << grb::toString(rc) << "\n";
		return;
	}

	// Now all values should be 2s
	if( grb::nnz( Identity ) != expected_nnz ) {
		std::cerr << "\t Expected " << expected_nnz << " nonzeroes, "
			<< "got " << nnz( Identity ) << "\n";
		rc = FAILED;
	}
	{
		constexpr bool nstr = !(descr & descriptors::structural);
		constexpr bool invm = descr & descriptors::invert_mask;
		const RC local_rc = check_all< nstr && !invm, nstr && invm >(
			Identity, expected_even, expected_odd, 0 );
		if( local_rc ) {
			std::cout << "\t Entry verification failed\n";
		}
		if( rc == SUCCESS ) {
			rc = local_rc;
		}
	}
	if( rc != SUCCESS ) {
		std::cerr << "\t Check of set identity matrix diagonal to 2s ( VERIFY ) "
			<< "FAILED\n";
		return;
	}

	// done
}

template< grb::Descriptor descr >
void self_void_test( const size_t &n, grb::RC &rc ) {
	rc = SUCCESS;

	grb::Matrix< void > Identity( 0, 0, 0 );
	try {
		grb::Matrix< void > Identity_alloc =
			algorithms::matrices< void >::identity( n );
		Identity = std::move( Identity_alloc );
	} catch( ... ) {
		std::cerr << "\t error during construction of the identity matrix\n";
		rc = FAILED;
		return;
	}
	if( rc != SUCCESS ) {
		std::cerr << "\t error during application of element-wise lambda-- test "
			<< "could not initialise\n";
		return;
	}

	if( nnz( Identity ) != n ) {
		std::cerr << "\t diagonal has " << nnz( Identity ) << " elements, expected "
			<< n << "\n";
		rc = FAILED;
		return;
	}

	for( const auto &pair : Identity ) {
		if( pair.first != pair.second ) {
			std::cerr << "Identity matrix has a non-diagonal entry\n";
			rc = FAILED;
			return;
		}
	}

	rc = grb::set( Identity, Identity, 2UL, grb::Phase::RESIZE );
	rc = rc ? rc : grb::set( Identity, Identity, 2UL, grb::Phase::EXECUTE );
	if( rc != SUCCESS ) {
		std::cerr << "Error during call to grb::set: " << grb::toString( rc )
			<< "\n";
		return;
	}

	if( nnz( Identity ) != n ) {
		std::cerr << "\t Result has " << nnz( Identity ) << " elements, expected "
			<< n << "\n";
		rc = FAILED;
		return;
	}

	for( const auto &pair : Identity ) {
		if( pair.first != pair.second ) {
			std::cerr << "Result has a non-diagonal entry, expected diagonal only\n";
			rc = FAILED;
			return;
		}
	}
}

int main( int argc, char ** argv ) {
	// defaults
	bool printUsage = false;
	size_t in = 1000;

	// error checking
	if( argc > 2 ) {
		printUsage = true;
	}
	if( argc == 2 ) {
		size_t read;
		std::istringstream ss( argv[ 1 ] );
		if( ! ( ss >> read ) ) {
			std::cerr << "Error parsing first argument\n";
			printUsage = true;
		} else if( ! ss.eof() ) {
			std::cerr << "Error parsing first argument\n";
			printUsage = true;
		} else {
			// all OK
			in = read;
		}
	}
	if( printUsage ) {
		std::cerr << "Usage: " << argv[ 0 ] << " [n]\n";
		std::cerr << "  -n (optional, default is 1000): an integer test size.\n";
		return 1;
	}

	std::cout << "This is functional test " << argv[ 0 ] << "\n";
	grb::Launcher< AUTOMATIC > launcher;
	grb::RC out = PANIC, last_error = SUCCESS;
	bool failed = false;

	// these next two tests are called test 0 since the requested operation, while
	// legal, actually translate to no-ops
	std::cout << "\t test 0A (self-masked, void output and void mask)\n";
	if( launcher.exec(
			&(self_void_test< grb::descriptors::no_operation >), in, out, true
		) != SUCCESS
	) {
		std::cerr << "Launching test FAILED\n" << std::endl;
		return 255;
	}
	if( out != SUCCESS ) {
		std::cout << "\t\t FAILED\n";
		last_error = out;
		failed = true;
	} else {
		std::cout << "\t\t OK\n";
	}

	std::cout << "\t test 0B (self-masked, void output and void mask, "
		<< "explicit structural descriptor)\n";
	if( launcher.exec(
			&(self_void_test< grb::descriptors::structural>), in, out, true
		) != SUCCESS
	) {
		std::cerr << "Launching test FAILED\n" << std::endl;
		return 255;
	}
	if( out != SUCCESS ) {
		std::cout << "\t\t FAILED\n";
		last_error = out;
		failed = true;
	} else {
		std::cout << "\t\t OK\n";
	}

	std::cout << "\t (test 0C does not exist: "
		<< "void inverted masks are not allowed)\n";

	std::cout << "\t test 1A (self-masked)\n";
	if( launcher.exec(
			&(self_identity_test< grb::descriptors::no_operation >), in, out, true
		) != SUCCESS
	) {
		std::cerr << "Launching test FAILED\n" << std::endl;
		return 255;
	}
	if( out != SUCCESS ) {
		std::cout << "\t\t FAILED\n";
		last_error = out;
		failed = true;
	} else {
		std::cout << "\t\t OK\n";
	}

	std::cout << "\t test 1B (self-masked, structural)\n";
	if( launcher.exec(
			&(self_identity_test< grb::descriptors::structural >), in, out, true
		) != SUCCESS
	) {
		std::cerr << "Launching test FAILED\n" << std::endl;
		return 255;
	}
	if( out != SUCCESS ) {
		std::cout << "\t\t FAILED\n";
		last_error = out;
		failed = true;
	} else {
		std::cout << "\t\t OK\n";
	}

	std::cout << "\t test 1C (self-masked, inverted mask)\n";
	if( launcher.exec(
			&(self_identity_test< grb::descriptors::invert_mask >), in, out, true
		) != SUCCESS
	) {
		std::cerr << "Launching test FAILED\n" << std::endl;
		return 255;
	}
	if( out != SUCCESS ) {
		std::cout << "\t\t FAILED\n";
		last_error = out;
		failed = true;
	} else {
		std::cout << "\t\t OK\n";
	}

	std::cout << "\t test 2A (matching domains, no-op resize)\n";
	if( launcher.exec(
			&(identity_test< grb::descriptors::no_operation, int, false >), in, out, true
		) != SUCCESS
	) {
		std::cerr << "Launching test FAILED\n" << std::endl;
		return 255;
	}
	if( out != SUCCESS ) {
		std::cout << "\t\t FAILED\n";
		last_error = out;
		failed = true;
	} else {
		std::cout << "\t\t OK\n";
	}

	std::cout << "\t test 2B (matching domains, no-op resize, structural)\n";
	if( launcher.exec(
			&(identity_test< grb::descriptors::structural, int, false >), in, out, true
		) != SUCCESS
	) {
		std::cerr << "Launching test FAILED\n" << std::endl;
		return 255;
	}
	if( out != SUCCESS ) {
		std::cout << "\t\t FAILED\n";
		last_error = out;
		failed = true;
	} else {
		std::cout << "\t\t OK\n";
	}

	std::cout << "\t test 2C (matching domains, no-op resize, inverted mask)\n";
	if( launcher.exec(
			&(identity_test< grb::descriptors::invert_mask, int, false >), in, out, true
		) != SUCCESS
	) {
		std::cerr << "Launching test FAILED\n" << std::endl;
		return 255;
	}
	if( out != SUCCESS ) {
		std::cout << "\t\t FAILED\n";
		last_error = out;
		failed = true;
	} else {
		std::cout << "\t\t OK\n";
	}

	std::cout << "\t test 3A (matching domains, resize)\n";
	if( launcher.exec(
			&(identity_test< grb::descriptors::no_operation, int, true >),
			in, out, true
		) != SUCCESS
	) {
		std::cerr << "Launching test FAILED\n" << std::endl;
		return 255;
	}
	if( out != SUCCESS ) {
		std::cout << "\t\t FAILED\n";
		last_error = out;
		failed = true;
	} else {
		std::cout << "\t\t OK\n";
	}

	std::cout << "\t test 3B (matching domains, resize, structural)\n";
	if( launcher.exec(
			&(identity_test< grb::descriptors::structural, int, true >), in, out, true )
		!= SUCCESS
	) {
		std::cerr << "Launching test FAILED\n" << std::endl;
		return 255;
	}
	if( out != SUCCESS ) {
		std::cout << "\t\t FAILED\n";
		last_error = out;
		failed = true;
	} else {
		std::cout << "\t\t OK\n";
	}

	std::cout << "\t test 3C (matching domains, resize, inverted mask)\n";
	if( launcher.exec(
			&(identity_test< grb::descriptors::invert_mask, int, true >), in, out, true )
		!= SUCCESS
	) {
		std::cerr << "Launching test FAILED\n" << std::endl;
		return 255;
	}
	if( out != SUCCESS ) {
		std::cout << "\t\t FAILED\n";
		last_error = out;
		failed = true;
	} else {
		std::cout << "\t\t OK\n";
	}

	std::cout << "\t test 4A (mismatching domains, no-op resize)\n";
	if( launcher.exec(
			&(identity_test< grb::descriptors::no_operation, size_t, false >),
			in, out, true
		) != SUCCESS
	) {
		std::cerr << "Launching test FAILED\n" << std::endl;
		return 255;
	}
	if( out != SUCCESS ) {
		std::cout << "\t\t FAILED\n";
		last_error = out;
		failed = true;
	} else {
		std::cout << "\t\t OK\n";
	}

	std::cout << "\t test 4B (mismatching domains, no-op resize, structural)\n";
	if( launcher.exec(
			&(identity_test< grb::descriptors::structural, size_t, false >),
			in, out, true
		) != SUCCESS
	) {
		std::cerr << "Launching test FAILED\n" << std::endl;
		return 255;
	}
	if( out != SUCCESS ) {
		std::cout << "\t\t FAILED\n";
		last_error = out;
		failed = true;
	} else {
		std::cout << "\t\t OK\n";
	}

	std::cout << "\t test 4C (mismatching domains, no-op resize, inverted mask)\n";
	if( launcher.exec(
			&(identity_test< grb::descriptors::invert_mask, size_t, false >),
			in, out, true
		) != SUCCESS
	) {
		std::cerr << "Launching test FAILED\n" << std::endl;
		return 255;
	}
	if( out != SUCCESS ) {
		std::cout << "\t\t FAILED\n";
		last_error = out;
		failed = true;
	} else {
		std::cout << "\t\t OK\n";
	}

	std::cout << "\t test 5A (mismatching domains, resize)\n";
	if( launcher.exec(
			&(identity_test< grb::descriptors::no_operation, double, true >),
			in, out, true
		) != SUCCESS
	) {
		std::cerr << "Launching test FAILED\n" << std::endl;
		return 255;
	}
	if( out != SUCCESS ) {
		std::cout << "\t\t FAILED\n";
		last_error = out;
		failed = true;
	} else {
		std::cout << "\t\t OK\n";
	}

	std::cout << "\t test 5B (mismatching domains, resize, structural)\n";
	if( launcher.exec(
			&(identity_test< grb::descriptors::structural, double, true >),
			in, out, true
		) != SUCCESS
	) {
		std::cerr << "Launching test FAILED\n" << std::endl;
		return 255;
	}
	if( out != SUCCESS ) {
		std::cout << "\t\t FAILED\n";
		last_error = out;
		failed = true;
	} else {
		std::cout << "\t\t OK\n";
	}

	std::cout << "\t test 5C (mismatching domains, resize, inverted mask)\n";
	if( launcher.exec(
			&(identity_test< grb::descriptors::invert_mask, double, true >),
			in, out, true
		) != SUCCESS
	) {
		std::cerr << "Launching test FAILED\n" << std::endl;
		return 255;
	}
	if( out != SUCCESS ) {
		std::cout << "\t\t FAILED\n";
		last_error = out;
		failed = true;
	} else {
		std::cout << "\t\t OK\n";
	}

	std::cout << "\t test 6A (void mask, no-op resize)\n";
	if( launcher.exec(
			&(identity_test< grb::descriptors::no_operation, void, false >),
			in, out, true
		) != SUCCESS
	) {
		std::cerr << "Launching test FAILED\n" << std::endl;
		return 255;
	}
	if( out != SUCCESS ) {
		std::cout << "\t\t FAILED\n";
		last_error = out;
		failed = true;
	} else {
		std::cout << "\t\t OK\n";
	}

	std::cout << "\t test 6B (void mask, no-op resize, structural)\n";
	if( launcher.exec(
			&(identity_test< grb::descriptors::structural, void, false >),
			in, out, true
		) != SUCCESS
	) {
		std::cerr << "Launching test FAILED\n" << std::endl;
		return 255;
	}
	if( out != SUCCESS ) {
		std::cout << "\t\t FAILED\n";
		last_error = out;
		failed = true;
	} else {
		std::cout << "\t\t OK\n";
	}

	std::cout << "\t (test 6C does not exist: "
		<< "void masks with inversion is not supported)\n";

	std::cout << "\t test 7A (void mask, resize)\n";
	if( launcher.exec(
			&(identity_test< grb::descriptors::no_operation, void, true >),
			in, out, true
		) != SUCCESS
	) {
		std::cerr << "Launching test FAILED\n" << std::endl;
		return 255;
	}
	if( out != SUCCESS ) {
		std::cout << "\t\t FAILED\n";
		last_error = out;
		failed = true;
	} else {
		std::cout << "\t\t OK\n";
	}

	std::cout << "\t test 7B (void mask, resize, structural)\n";
	if( launcher.exec(
			&(identity_test< grb::descriptors::structural, void, true >),
			in, out, true
		) != SUCCESS
	) {
		std::cerr << "Launching test FAILED\n" << std::endl;
		return 255;
	}
	if( out != SUCCESS ) {
		std::cout << "\t\t FAILED\n";
		last_error = out;
		failed = true;
	} else {
		std::cout << "\t\t OK\n";
	}

	std::cout << "\t (test 7C does not exist: "
		<< "void masks with inversion is not supported)\n";

	if( failed ) {
		std::cerr << std::flush;
		std::cout << "Test FAILED (last error: " << grb::toString( last_error )
			<< ")\n" << std::endl;
		return static_cast< int >(last_error);
	} else {
		std::cout << "Test OK\n" << std::endl;
	}

	return 0;
}

