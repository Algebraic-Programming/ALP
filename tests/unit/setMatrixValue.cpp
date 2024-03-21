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
template< typename T >
RC check_all(
	grb::Matrix< T > &A,
	Expected< T > expected_value, const size_t row_offset
) {
	const T * const value_p = expected_value.getPointer();
	return std::all_of( A.cbegin(), A.cend(),
		[&value_p,&row_offset](
			const std::pair< std::pair< size_t, size_t >, T > &entry
		) {
			return entry.second == *value_p &&
				entry.first.first + row_offset == entry.first.second;
		}
	) ? SUCCESS : FAILED;
}

/**
 * This variant is for pattern (void) matrices, and only checks for the
 * off-diagonal position of the entries. The interface matches that of the
 * generic check_all.
 */
RC check_all(
	grb::Matrix< void > &A,
	Expected< void > expected_value, const size_t row_offset
) {
	(void) expected_value;
	return std::all_of( A.cbegin(), A.cend(),
		[&row_offset](
			const std::pair< size_t, size_t > &entry
		) {
			return entry.first + row_offset == entry.second;
		}
	) ? SUCCESS : FAILED;
}

/**
 * @tparam left Whether the output matrix is off_diagonal (left) or identity
 *              (right). The difference in test is that for if selecting left,
 *              then the capacity of the output may have to resize. If the
 *              other way around, no resizing is ever needed.
 */
template< typename T, bool left >
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
	Expected< int > expected_left;
	Expected< T > expected_right;
	expected_left.set( 7 );
	expected_right.set( 1 );

	// validate construction
	{
		RC local_rc = check_all( off_diagonal, expected_left, 1 );
		if( local_rc != SUCCESS ) {
			std::cerr << "\t verification of off-diagonal construction failed: "
				<< "at least one unexpected matrix element found\n";
			if( rc == SUCCESS ) { rc = local_rc; }
		}
		local_rc = check_all( identity, expected_right, 0 );
			if( local_rc != SUCCESS ) {
			std::cerr << "\t verification of identity construction failed: "
				<< "at least one unexpected matrix element found\n";
			if( rc == SUCCESS ) { rc = local_rc; }
		}
	}
	if( rc != SUCCESS ) { return; }

	// perform the set, resize phase
	if( left ) {
		rc = grb::set( off_diagonal, identity, 3, RESIZE );
	} else {
		rc = grb::set( identity, off_diagonal, 3, RESIZE );
	}
	if( rc != SUCCESS ) {
		std::cerr << "\t resize failed\n";
		return;
	}

	// set expected value for tests, accounting for possible void T
	expected_left.set( 3 );
	expected_right.set( 3 );

	// check capacity
	if( left ) {
		if( grb::capacity( off_diagonal ) < n ) {
			std::cerr << "\t resize failed to achieve correct capacity for "
				<< "off_diagonal\n";
			rc = FAILED;
		}
		return;
	}

	// perform the set, execute phase
	if( left ) {
		rc = grb::set( off_diagonal, identity, 3, EXECUTE );
	} else {
		rc = grb::set( identity, off_diagonal, 3, EXECUTE );
	}
	if( rc != SUCCESS ) {
		std::cerr << "\t execute failed\n";
		return;
	}

	// check output
	{
		const size_t expected_nnz = left ? n : n - 1;
		const size_t actual_nnz   = left ? nnz( off_diagonal ) : nnz( identity );
		if( actual_nnz != expected_nnz ) {
			std::cerr << "\t unexpected number of nonzeroes: got " << actual_nnz << ", "
				<< "expected " << expected_nnz << "\n";
			rc = FAILED;
		}
		RC local_rc = PANIC;
		if( left ) {
			local_rc = check_all( off_diagonal, expected_left, 0 );
		} else {
			local_rc = check_all( identity, expected_right, 1 );
		}
		if( local_rc == FAILED ) {
			std::cerr << "\t at least one unexpected output entry found\n";
			if( rc == SUCCESS ) { rc = local_rc; }
		}
	}

	// done
}

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

	if( nnz( Identity ) != n ) {
		std::cerr << "\t diagonal has " << nnz( Identity ) << " elements, expected "
			<< n << "\n";
		rc = FAILED;
		return;
	}

	// Check first if the matrix is correctly initialised with 1s
	rc = std::all_of( Identity.cbegin(), Identity.cend(),
		[]( const std::pair< std::pair< size_t, size_t >, int > &entry ) {
			return entry.second == 1 && entry.first.first == entry.first.second;
		} ) ? SUCCESS : FAILED;
	if( rc != SUCCESS ) {
		std::cerr << "\t initialisation (buildMatrixUnique check) FAILED: rc is "
			<< grb::toString(rc) << "\n";
		return;
	}

	// Try to set the matrix to 2s ( RESIZE )
	rc = grb::set( Identity, Identity, 2UL, Phase::RESIZE );
	if( rc != SUCCESS ) {
		std::cerr << "\t set identity matrix diagonal to 2s ( RESIZE ) FAILED: "
			<< "rc is " << grb::toString(rc) << "\n";
		return;
	}
	// As the RESIZE phase is useless, the matrix should not be resized.
	if( capacity( Identity ) < n ) {
		std::cerr << "\t unexpected matrix capacity: " << capacity( Identity )
			<< ", expected at least " << n << "\n";
		rc = FAILED;
		return;
	}

	// Try to set the matrix to 2s ( EXECUTE )
	rc = grb::set( Identity, Identity, 2UL, Phase::EXECUTE );
	if( rc != SUCCESS ) {
		std::cerr << "\t set identity matrix diagonal to 2s ( EXECUTE ) FAILED: "
			<< "rc is " << grb::toString(rc) << "\n";
		return;
	}

	// Now all values should be 2s
	if( nnz( Identity ) != n ) {
		std::cerr << "\t Expected " << n << " nonzeroes, got " << nnz( Identity )
			<< "\n";
		rc = FAILED;
	}
	{
		const RC local_rc = std::all_of( Identity.cbegin(), Identity.cend(),
			[]( const std::pair< std::pair< size_t, size_t >, int > &entry ) {
				return entry.second == 2 && entry.first.first == entry.first.second;
			} ) ? SUCCESS : FAILED;
		if( rc == SUCCESS ) {
			rc = local_rc;
		} else {
			std::cout << "\t Entry verification failed\n";
		}
	}
	if( rc != SUCCESS ) {
		std::cerr << "\t Check of set identity matrix diagonal to 2s ( VERIFY ) "
			<< "FAILED\n";
		return;
	}

	// done
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

	std::cout << "\t test 1 (self-masked)\n";
	if( launcher.exec( &self_identity_test, in, out, true ) != SUCCESS ) {
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

	std::cout << "\t test 2 (matching domains, no-op resize)\n";
	if( launcher.exec( &identity_test< int, false >, in, out, true ) != SUCCESS ) {
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

	std::cout << "\t test 3 (matching domains, resize)\n";
	if( launcher.exec( &identity_test< int, true >, in, out, true ) != SUCCESS ) {
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

	std::cout << "\t test 4 (mismatching domains, no-op resize)\n";
	if( launcher.exec( &identity_test< size_t, false >, in, out, true )
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

	std::cout << "\t test 5 (mismatching domains, resize)\n";
	if( launcher.exec( &identity_test< double, true >, in, out, true )
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

	std::cout << "\t test 6 (void mask, no-op resize)\n";
	if( launcher.exec( &identity_test< void, false >, in, out, true )
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

	std::cout << "\t test 7 (void output, resize)\n";
	if( launcher.exec( &identity_test< void, true >, in, out, true )
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

	if( failed ) {
		std::cerr << std::flush;
		std::cout << "Test FAILED (last error: " << grb::toString( last_error )
			<< ")\n" << std::endl;
	} else {
		std::cout << "Test OK\n" << std::endl;
	}
	return 0;
}

