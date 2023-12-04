
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


using namespace grb;

void grb_program( const size_t &n, grb::RC &rc ) {
	grb::Semiring<
		grb::operators::add< double >, grb::operators::mul< double >,
		grb::identities::zero, grb::identities::one
	> ring;

	// initialize test
	grb::Matrix< double > A( n, n );
	grb::Matrix< double > B( n, n );
	grb::Matrix< double > C( n, n );
	size_t I[ n ], J[ n ];
	double V[ n ];
	for( size_t k = 0; k < n; ++k ) {
		I[ k ] = J[ k ] = k;
		V[ k ] = 2.0;
	}
	rc = grb::resize( B, n );
	if( rc == SUCCESS ) {
		rc = grb::buildMatrixUnique( B, I, J, V, n, SEQUENTIAL );
	}
	if( rc == SUCCESS ) {
		rc = grb::resize( A, n );
	}
	if( rc == SUCCESS ) {
		V[ 0 ] = 1.0;
		for( size_t k = 1; k < n; ++k ) {
			J[ k - 1 ] = k;
			V[ k ] = 1.0;
		}
		J[ n - 1 ] = 0;
		rc = grb::buildMatrixUnique( A, I, J, V, n, SEQUENTIAL );
	}
	if( rc != SUCCESS ) {
		std::cerr << "\tinitialisation FAILED\n";
		return;
	}

	// compute with the semiring mxm
	std::cout << "\tVerifying the semiring version of mxm\n";

	rc = grb::mxm( C, A, B, ring, RESIZE );
	if( rc == SUCCESS ) {
		rc = grb::mxm( C, A, B, ring );
		if( rc != SUCCESS ) {
			std::cerr << "Call to grb::mxm FAILED\n";
		}
	} else {
		std::cerr << "Call to grb::resize FAILED\n";
	}
	if( rc != SUCCESS ) {
		return;
	}

	// check CRS output
	const auto &crs1 = internal::getCRS( C );
	for( size_t i = 0; i < n; ++i ) {
		const size_t entries = crs1.col_start[ i + 1 ] - crs1.col_start[ i ];
		if( entries != 1 ) {
			std::cerr << "Error: unexpected number of entries " << entries << ", "
				<< "expected 1 (CRS).\n";
			rc = FAILED;
		}
		for( size_t k = crs1.col_start[ i ]; k < crs1.col_start[ i + 1 ]; ++k ) {
			const size_t expect = i == n - 1 ? 0 : i + 1;
			if( crs1.row_index[ k ] != expect ) {
				std::cerr << "Error: unexpected entry at ( " << i << ", "
					<< crs1.row_index[ k ] << " ), expected one at ( "
					<< i << ", " << ( ( i + 1 ) % n ) << " ) instead (CRS).\n";
				rc = FAILED;
			}
			if( crs1.values[ k ] != 2.0 ) {
				std::cerr << "Error: unexpected value " << crs1.values[ k ]
					<< "; expected 2 (CRS).\n";
				rc = FAILED;
			}
		}
	}

	// check CCS output
	const auto &ccs1 = internal::getCCS( C );
	for( size_t j = 0; j < n; ++j ) {
		const size_t entries = ccs1.col_start[ j + 1 ] - ccs1.col_start[ j ];
		if( entries != 1 ) {
			std::cerr << "Error: unexpected number of entries " << entries
				<< ", expected 1 (CCS).\n";
			rc = FAILED;
		}
		for( size_t k = ccs1.col_start[ j ]; k < ccs1.col_start[ j + 1 ]; ++k ) {
			const size_t expect = j == 0 ? n - 1 : j - 1;
			if( ccs1.row_index[ k ] != expect ) {
				std::cerr << "Error: unexpected entry at ( " << ccs1.row_index[ k ] << ", "
					<< j << " ), expected one at ( " << expect << ", " << j
					<< " ) instead (CCS).\n";
				rc = FAILED;
			}
			if( ccs1.values[ k ] != 2.0 ) {
				std::cerr << "Error: unexpected value " << ccs1.values[ k ]
					<< "; expected 2 (CCS).\n";
				rc = FAILED;
			}
		}
	}

	// compute with the operator-monoid mxm
	std::cout << "\tVerifying the operator-monoid version of mxm\n";

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
			ring.getMultiplicativeOperator()
		);
		if( rc != SUCCESS ) {
			std::cerr << "Call to grb::mxm FAILED\n";
		}
	} else {
		std::cerr << "Call to grb::resize FAILED\n";
	}
	if( rc != SUCCESS ) {
		return;
	}

	// check CRS output
	const auto &crs2 = internal::getCRS( C );
	for( size_t i = 0; i < n; ++i ) {
		const size_t entries = crs2.col_start[ i + 1 ] - crs2.col_start[ i ];
		if( entries != 1 ) {
			std::cerr << "Error: unexpected number of entries " << entries
				<< ", expected 1 (CRS).\n";
			rc = FAILED;
		}
		for( size_t k = crs2.col_start[ i ]; k < crs2.col_start[ i + 1 ]; ++k ) {
			const size_t expect = i == n - 1 ? 0 : i + 1;
			if( crs2.row_index[ k ] != expect ) {
				std::cerr << "Error: unexpected entry at ( " << i << ", "
					<< crs2.row_index[ k ] << " ), expected one at ( " << i << ", "
					<< ( ( i + 1 ) % n ) << " ) instead (CRS).\n";
				rc = FAILED;
			}
			if( crs2.values[ k ] != 2.0 ) {
				std::cerr << "Error: unexpected value " << crs2.values[ k ]
					<< "; expected 2 (CRS).\n";
				rc = FAILED;
			}
		}
	}

	// check CCS output
	const auto &ccs2 = internal::getCCS( C );
	for( size_t j = 0; j < n; ++j ) {
		const size_t entries = ccs2.col_start[ j + 1 ] - ccs2.col_start[ j ];
		if( entries != 1 ) {
			std::cerr << "Error: unexpected number of entries " << entries
				<< ", expected 1 (CCS).\n";
			rc = FAILED;
		}
		for( size_t k = ccs2.col_start[ j ]; k < ccs2.col_start[ j + 1 ]; ++k ) {
			const size_t expect = j == 0 ? n - 1 : j - 1;
			if( ccs2.row_index[ k ] != expect ) {
				std::cerr << "Error: unexpected entry at ( " << ccs2.row_index[ k ] << ", "
					<< j << " ), expected one at ( " << expect << ", " << j
					<< " ) instead (CCS).\n";
				rc = FAILED;
			}
			if( ccs2.values[ k ] != 2.0 ) {
				std::cerr << "Error: unexpected value " << ccs2.values[ k ]
					<< "; expected 2 (CCS).\n";
				rc = FAILED;
			}
		}
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

