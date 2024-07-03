
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

#include <graphblas.hpp>

#include <graphblas/algorithms/matrix_factory.hpp>

#include <utils/matrix_values_check.hpp>


using namespace grb;
using namespace grb::algorithms;

void grb_program( const size_t &n, grb::RC &rc ) {
	grb::Semiring<
		grb::operators::add< double >, grb::operators::mul< double >,
		grb::identities::zero, grb::identities::one
	> ring;

	// initialize test
	grb::Matrix< double > A = matrices< double >::eye( n, n, 1, 1 );
	grb::Matrix< double > B = matrices< double >::eye( n, n, 2, 2 );
	grb::Matrix< double > C( n, n );
	grb::Matrix< double > C_expected = matrices< double >::eye( n, n, 2, 3 );

	// compute with the semiring mxm
	std::cout << "\tVerifying the semiring version of mxm\n";

	rc = grb::mxm( C, A, B, ring, RESIZE );
	if( rc == SUCCESS ) {
		rc = grb::mxm( C, A, B, ring );
		if( rc != SUCCESS ) {
			std::cerr << "Call to grb::mxm( ..., RESIZE ) I FAILED\n";
		}
	} else {
		std::cerr << "Call to grb::mxm( ..., EXECUTE ) I FAILED\n";
	}
	if( rc != SUCCESS ) { return; }

	// check CRS output
	if( utils::compare_crs( C, C_expected ) != SUCCESS ) {
		std::cerr << "Error detected while comparing output to ground-truth CRS\n";
		rc = FAILED;
	}

	// check CCS output
	if( utils::compare_ccs( C, C_expected ) != SUCCESS ) {
		std::cerr << "Error detected while comparing output to ground-truth CCS\n";
		rc = FAILED;
	}
	if( rc != SUCCESS ) { return; }

	// compute with the operator-monoid mxm
	std::cout << "\tVerifying the operator-monoid version of mxm\n";

	rc = grb::clear( C );
	rc = rc ? rc : grb::mxm(
		C, A, B,
		ring.getAdditiveMonoid(),
		ring.getMultiplicativeOperator(),
		RESIZE
	);
	if( rc == SUCCESS ) {
		rc = grb::mxm(
				C, A, B,
				ring.getAdditiveMonoid(),
				ring.getMultiplicativeOperator()
			);
		if( rc != SUCCESS ) {
			std::cerr << "Call to grb::mxm( ..., EXECUTE ) II FAILED\n";
		}
	} else {
		std::cerr << "Call to grb::mxm( ..., RESIZE ) II FAILED\n";
	}
	if( rc != SUCCESS ) { return; }

	// check CRS output
	if( utils::compare_crs( C, C_expected ) != SUCCESS ) {
		std::cerr << "Error detected while comparing output to ground-truth CRS\n";
		rc = FAILED;
	}

	// check CCS output
	if( utils::compare_ccs( C, C_expected ) != SUCCESS ) {
		std::cerr << "Error detected while comparing output to ground-truth CCS\n";
		rc = FAILED;
	}
	if( rc != SUCCESS ) { return; }

	// check in-place behaviour using the semiring
	std::cout << "\tVerifying in-place behaviour of mxm (using semirings)\n"
		<< "\t\tin this test, the output nonzero structure is unchanged\n"
		<< "\t\talso in this test, we skip RESIZE as we know a priori the capacity "
		<< "is sufficient\n";

	{
		grb::Matrix< double > replace = matrices< double >::eye( n, n, 4, 3 );
		std::swap( replace, C_expected );
	}

	rc = grb::mxm( C, A, B, ring, EXECUTE );
	if( rc != SUCCESS ) {
		std::cerr << "Call to grb::mxm( .., EXECUTE ) III FAILED\n";
	}
	if( rc != SUCCESS ) { return; }

	// check CRS and CCS output
	if( utils::compare_crs( C, C_expected ) != SUCCESS ) {
		std::cerr << "Error detected while comparing output to ground-truth CRS\n";
		rc = FAILED;
	}
	if( utils::compare_ccs( C, C_expected ) != SUCCESS ) {
		std::cerr << "Error detected while comparing output to ground-truth CCS\n";
		rc = FAILED;
	}
	if( rc != SUCCESS ) { return; }

	// check in-place behaviour using the monoid-operator variant
	std::cout << "\tVerifying in-place behaviour of mxm (using monoid-op)\n"
		<< "\t\tin this test, the output nonzero structure changes\n";
	size_t expected_nz = grb::nnz( C ) + n;

	// replace A, B with (scaled) identities
	{
		grb::Matrix< double > replace = matrices< double >::eye( n, n, 3, 0 );
		std::swap( A, replace );
	}
	{
		grb::Matrix< double > replace = matrices< double >::identity( n );
		std::swap( B, replace );
	}

	rc = grb::mxm(
			C, A, B,
			ring.getAdditiveMonoid(),
			ring.getMultiplicativeOperator(),
			RESIZE
		);
	if( rc == SUCCESS ) {
		rc = grb::mxm(
				C, A, B,
				ring.getAdditiveMonoid(),
				ring.getMultiplicativeOperator(),
				EXECUTE
			);
		if( rc != SUCCESS ) {
			std::cerr << "Call to grb::mxm( ..., EXECUTE ) IV FAILED\n";
		}
	} else {
		std::cerr << "Call to grb::mxm( ..., RESIZE ) IV FAILED\n";
	}
	if( rc != SUCCESS ) {
		std::cerr << "Call to grb::mxm( ..., EXECUTE ) IV FAILED: "
			<< grb::toString( rc ) << "\n";
		return;
	}

	// ``manual'' check
	if( expected_nz != grb::nnz( C ) ) {
		std::cerr << "Expected " << expected_nz << " nonzeroes, got "
			<< grb::nnz( C ) << "\n";
		rc = FAILED;
	}
	for( const auto &pair : C ) {
		const size_t &i = pair.first.first;
		const size_t &j = pair.first.second;
		const size_t &v = pair.second;
		if( i == j ) {
			if( v != 3 ) {
				std::cerr << "\t expected value 3 at position ( " << i << ", " << j
					<< " ), got " << v << "\n";
				rc = FAILED;
			}
		} else if( i + 3 == j ) {
			if( v != 4 ) {
				std::cerr << "\t expected value 4 at position ( " << i << ", " << j
					<< " ), got " << v << "\n";
				rc = FAILED;
			}
		} else {
			std::cerr << "\t expected no entry at position ( " << i << ", " << j
				<< " ), but got one with value " << v << "\n";
			rc = FAILED;
		}
	}
	if( rc != SUCCESS ) {
		std::cerr << "Test IV did not pass verification\n";
		return;
	}

	std::cout << "\tVerifying in-place behaviour of mxm (using semiring)\n"
		<< "\t\tin this test, both nonzero structure and existing nonzeroes change\n";
	{
		const size_t i[ 2 ] = { 0, n - 1 };
		const size_t j[ 2 ] = { 0, 0 };
		const double v[ 2 ] = { 2.0, 2.0 };
		grb::Matrix< double > replaces_A( n, n );
		rc = grb::buildMatrixUnique( replaces_A, i, i + 2, j, j + 2, v, v + 2,
			grb::PARALLEL );
		if( rc == grb::SUCCESS ) { std::swap( A, replaces_A ); }
	}
	if( rc != grb::SUCCESS ) {
		std::cerr << "Experiment setup FAILED\n";
		return;
	}

	rc = grb::mxm( C, A, B, ring, grb::RESIZE );
	if( rc != grb::SUCCESS ) {
		std::cerr << "Call to grb::mxm( ..., RESIZE ) V FAILED\n";
		return;
	}
	rc = grb::mxm( C, A, B, ring );
	if( rc != SUCCESS ) {
		std::cerr << "Call to grb::mxm( ..., EXECUTE ) V FAILED\n";
		return;
	}

	// ``manual'' check
	(void) ++expected_nz;
	if( expected_nz != grb::nnz( C ) ) {
		std::cerr << "Expected " << expected_nz << " nonzeroes, got "
			<< grb::nnz( C ) << "\n";
		rc = FAILED;
	}
	for( const auto &pair : C ) {
		const size_t &i = pair.first.first;
		const size_t &j = pair.first.second;
		const size_t &v = pair.second;
		if( i == 0 && j == 0 ) {
			// note: this branch checks existing nonzero value mutation
			if( v != 5 ) {
				std::cerr << "\t expected value 5 at position ( 0, 0 ), got " << v << "\n";
				rc = FAILED;
			}
		} else if( i == j ) {
			// note: this branch checks unchanged nonzero value mutation
			if( v != 3 ) {
				std::cerr << "\t expected value 3 at position ( " << i << ", " << j
					<< " ), got " << v << "\n";
				rc = FAILED;
			}
		} else if( i + 3 == j ) {
			// note: this branch checks unchanged nonzero value mutation
			if( v != 4 ) {
				std::cerr << "\t expected value 4 at position ( " << i << ", " << j
					<< " ), got " << v << "\n";
				rc = FAILED;
			}
		} else if( i == n - 1 && j == 0 ) {
			// note: this branch checks nonzero structure mutation
			if( v != 2 ) {
				std::cerr << "\t expected value 2 at position ( " << (n - 1 ) << ", 0 ), "
					<< "got " << v << "\n";
				rc = FAILED;
			}
		} else {
			std::cerr << "\t expected no entry at position ( " << i << ", " << j
				<< " ), but got one with value " << v << "\n";
			rc = FAILED;
		}
	}
	if( rc != SUCCESS ) {
		std::cerr << "Test V did not pass verification\n";
		return;
	}

}

int main( int argc, char ** argv ) {
	// defaults
	bool printUsage = false;
	size_t in = 100;

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
		} else if( read % 2 != 0 ) {
			std::cerr << "Given value for n is odd\n";
			printUsage = true;
		} else {
			// all OK
			in = read;
		}
	}
	if( printUsage ) {
		std::cerr << "Usage: " << argv[ 0 ] << " [n]\n";
		std::cerr << "  -n (optional, default is 100): an even integer, the "
					 "test size.\n";
		return 1;
	}

	std::cout << "This is functional test " << argv[ 0 ] << "\n";
	grb::Launcher< AUTOMATIC > launcher;
	grb::RC out;
	if( launcher.exec( &grb_program, in, out, true ) != SUCCESS ) {
		std::cerr << "Launching test FAILED\n";
		return 255;
	}
	if( out != SUCCESS ) {
		std::cerr << "Test FAILED (" << grb::toString( out ) << ")" << std::endl;
	} else {
		std::cout << "Test OK" << std::endl;
	}
	return 0;
}

