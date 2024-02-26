
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

#include <utility>
#include <iostream>

#include "graphblas.hpp"


using namespace grb;

// sample data

static const double m1[ 4 ] = { 0.5, 3.4, 5, 0 };

static const size_t I1[ 4 ] = { 0, 1, 2, 0 };
static const size_t J1[ 4 ] = { 0, 1, 2, 2 };

static const double m2[ 8 ] = { 1, 1, 0, 0, 0, 0, 0, 0 };

static const size_t I2[ 8 ] = { 0, 2, 0, 0, 1, 1, 2, 2 };
static const size_t J2[ 8 ] = { 0, 2, 1, 2, 0, 2, 0, 1 };

static const double mask_test1_expect[ 3 ] = { 4, 10, 18 };
static const double mask_test2_expect[ 3 ] = { 11, 20, 27 };

void grbProgram( const void *, const size_t in_size, int &error ) {
	error = 0;

	if( in_size != 0 ) {
		(void)fprintf( stderr, "Unit tests called with unexpected input\n" );
		error = 1;
		return;
	}

	// allocate
	grb::Vector< double > u = { 1, 2, 3 };
	grb::Vector< double > v = { 4, 5, 6 };
	grb::Matrix< double > Result1( 3, 3 );
	grb::Matrix< double > Result2( 3, 3 );
	grb::Matrix< double > Mask1( 3, 3 );
	grb::Matrix< double > Mask2( 3, 3 );
	grb::Vector< double > test1 = { 1, 1, 1 };
	grb::Vector< double > out1( 3 );
	grb::Vector< double > test2 = { 1, 1, 1 };
	grb::Vector< double > out2( 3 );

	// semiring
	grb::Semiring<
		grb::operators::add< double >, grb::operators::mul< double >,
		grb::identities::zero, grb::identities::one
	> ring;

	grb::RC rc = grb::buildMatrixUnique( Mask1, &( I1[ 0 ] ), &( J1[ 0 ] ), m1, 3, SEQUENTIAL );

	if( rc != SUCCESS ) {
		std::cerr << "\t first mask buildMatrix FAILED\n";
		error = 5;
	}

	if( !error ) {
		rc = grb::buildMatrixUnique( Mask2, &( I2[ 0 ] ), &( J2[ 0 ] ), m2, 8, SEQUENTIAL );
		if( rc != SUCCESS ) {
			std::cerr << "\t second mask buildMatrix FAILED\n";
			error = 10;
		}
	}

	

	if( !error ) {
		rc = grb::outer( Result1, Mask1, u, v, ring.getMultiplicativeOperator(), RESIZE );
		rc = rc ? rc : grb::outer( Result1, Mask1, u, v, ring.getMultiplicativeOperator() );
		if( rc != grb::SUCCESS ) {
			std::cerr << "Unexpected return code from grb::outer: "
				<< toString( rc ) << ".\n";
			error = 15;
		}
	}


	if( !error && grb::nnz( Result1 ) != 3 ) {
		std::cerr << "\t Unexpected number of nonzeroes in matrix: "
			<< grb::nnz( Result1 ) << ", expected 3.\n";
		error = 20;
	}

	if( !error ) {
		rc = grb::outer< descriptors::force_row_major | descriptors::invert_mask >( Result2, Mask2, u, v, ring.getMultiplicativeOperator(), RESIZE );
		rc = rc ? rc : grb::outer< descriptors::force_row_major | descriptors::invert_mask >( Result2, Mask2, u, v, ring.getMultiplicativeOperator() );
		if( rc != grb::SUCCESS ) {
			std::cerr << "Unexpected return code from grb::outer: "
				<< toString( rc ) << ".\n";
			error = 25;
		}
	}


	if( !error && grb::nnz( Result2 ) != 6 ) {
		std::cerr << "\t Unexpected number of nonzeroes in matrix: "
			<< grb::nnz( Result2 ) << ", expected 6.\n";
		error = 30;
	}

	if( !error ) {
		rc = grb::vxm( out1, test1, Result1, ring );
		if( rc != grb::SUCCESS ) {
			std::cerr << "Unexpected return code from premultiplying Result1 by a vector (vxm): "
				<< toString( rc ) << ".\n";
			error = 35;
		}
	}

	if( !error ) {
		if( grb::nnz( out1 ) != 3 ) {
			std::cerr << "\t Unexpected number of nonzeroes (premultiply): "
				<< grb::nnz( out1 ) << ", expected 3\n";
			error = 40;
		}
		for( const auto &pair : out1 ) {
			size_t i = pair.first;
			if( pair.second != mask_test1_expect[ i ] ) {
				std::cerr << "Premultiplying Result1 by a vector of all ones, "
					<< "unexpected value " << pair.second << " "
					<< "at coordinate " << i << ", expected "
					<< mask_test1_expect[ i ] << ".\n";
				error = 45;
				break;
			}
		}
	}

	if( !error ) {
		rc = grb::vxm< grb::descriptors::transpose_matrix >( out2, test2, Result2, ring );
		if( rc != grb::SUCCESS ) {
			std::cerr << "Unexpected return code from postmultiplying Result2 by a vector (vxm): "
				<< toString( rc ) << ".\n";
			error = 60;
		}
	}

	if( !error ) {
		if( grb::nnz( out2 ) != 3 ) {
			std::cerr << "\t Unexpected number of nonzeroes (postmultiply): "
				<< grb::nnz( out1 ) << ", expected 3\n";
			error = 65;
		}
		for( const auto &pair : out2 ) {
			size_t i = pair.first;
			if( pair.second != mask_test2_expect[ i ] ) {
				std::cerr << "Postmultiplying Result2 by a vector of all ones, "
					<< "unexpected value " << pair.second << " "
					<< "at coordinate " << i << ", "
					<< "expected " << mask_test2_expect[ i ] << ".\n";
				error = 70;
				break;
			}
		}
	}
}

void grb_program_custom_size( const size_t &n, int &error ) {
	grb::Semiring<
		grb::operators::add< double >, grb::operators::mul< double >,
		grb::identities::zero, grb::identities::one
	> ring;

	// initialize test
	grb::Matrix< double > Result( n, n );
	grb::Matrix< double > Mask( n, n );
	size_t I[ 2 * n - 1 ], J[ 2 * n - 1 ];
	double mask_in [ 2 * n - 1 ];
	double u_in[ n ];
	double v_in[ n ];
	double test_in[ n ];
	double expect[ n ];
	grb::Vector< double > u( n );
	grb::Vector< double > v( n );
	grb::Vector< double > test( n );
	grb::Vector< double > out( n );
	for( size_t k = 0; k < n; ++k ) {
		I[ k ] = J[ k ] = k;
		mask_in[ k ] = 1;
		u_in[ k ] = k + 1;
		v_in[ k ] = k + 1;
		test_in[ k ] = 1;
		if( k < n - 1 ) {
			I[ n + k ] = k;
			J[ n + k ] = k + 1;
			mask_in [ n + k ] = 1;
		}
		if( k == 0 ) {
			expect [ k ] = 1;
		}
		else {
			expect [ k ] = ( k + 1 ) * ( 2 * k + 1 );
		}
	}
	
	const double * vec_iter = &(u_in[ 0 ]);
	grb::RC rc = grb::buildVector( u, vec_iter, vec_iter + n, SEQUENTIAL );
	if( rc != SUCCESS ) {
		std::cerr << "\t buildVector of u vector FAILED\n";
		error = 5;
	}

	if( !error ) {
		vec_iter = &(v_in[ 0 ]);
		rc = grb::buildVector( v, vec_iter, vec_iter + n, SEQUENTIAL );
		if( rc != SUCCESS ) {
			std::cerr << "\t buildVector of v vector FAILED\n";
			error = 10;
		}
	}

	if( !error ) {
		vec_iter = &(test_in[ 0 ]);
		rc = grb::buildVector( test, vec_iter, vec_iter + n, SEQUENTIAL );
		if( rc != SUCCESS ) {
			std::cerr << "\t buildVector of test input vector FAILED\n";
			error = 15;
		}
	}

	if( !error ) {
		rc = grb::resize( Mask, 2 * n - 1 );
		if( rc != SUCCESS ) {
			std::cerr << "\t mask matrix resize FAILED\n";
			error = 20;
		}
	}

	if( !error ) {
		rc = grb::buildMatrixUnique( Mask, I, J, mask_in, 2 * n - 1, SEQUENTIAL );
		if( rc != SUCCESS ) {
			std::cerr << "\t buildMatrixUnique of mask matrix FAILED\n";
			error = 25;
		}
	}

	if( !error ) {
		rc = grb::outer< descriptors::structural >( Result, Mask, u, v, ring.getMultiplicativeOperator(), RESIZE );
		rc = rc ? rc : grb::outer< descriptors::structural >( Result, Mask, u, v, ring.getMultiplicativeOperator() );
		if( rc != grb::SUCCESS ) {
			std::cerr << "Unexpected return code from grb::outer: "
				<< toString( rc ) << ".\n";
			error = 30;
		}
	}


	if( !error && grb::nnz( Result ) != 2 * n -1 ) {
		std::cerr << "\t Unexpected number of nonzeroes in matrix: "
			<< grb::nnz( Result ) << ", expected "
			<< 2 * n - 1 <<".\n";
		error = 35;
	}

	if( !error ) {
		rc = grb::vxm( out, test, Result, ring );
		if( rc != grb::SUCCESS ) {
			std::cerr << "Unexpected return code from premultiplying Result by a vector (vxm): "
				<< toString( rc ) << ".\n";
			error = 40;
		}
	}

	if( !error ) {
		if( grb::nnz( out ) != n ) {
			std::cerr << "\t Unexpected number of nonzeroes (premultiply): "
				<< grb::nnz( out ) << ", expected "
				<< n << ".\n";
			error = 45;
		}
		for( const auto &pair : out ) {
			size_t i = pair.first;
			if( pair.second != expect[ i ] ) {
				std::cerr << "Premultiplying Result by a vector of all ones, "
					<< "unexpected value " << pair.second << " "
					<< "at coordinate " << i << ", expected "
					<< expect[ i ] << ".\n";
				error = 50;
				break;
			}
		}
	}

}

int main( int argc, char ** argv ) {
	(void)argc;
	std::cout << "Functional test executable: " << argv[ 0 ] << "\n";

	bool printUsage = false;
	size_t n = 100;

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
			n = read;
		}
	}
	if( printUsage ) {
		std::cerr << "Usage: " << argv[ 0 ] << " [n]\n";
		std::cerr << "  -n (optional, default is 100): an integer, the "
					 "test size.\n";
		return 1;
	}

	

	int error = 0;
	grb::Launcher< AUTOMATIC > launcher;

	if( launcher.exec( &grbProgram, nullptr, 0, error ) != SUCCESS ) {
		std::cerr << "Test 1 failed to launch\n";
		error = 255;
	}
	if( error != 0 ) {
		std::cerr << std::flush;
		std::cout << "Test 1 FAILED\n" << std::endl;
		return 0;
	}

	if( launcher.exec( &grb_program_custom_size, n, error ) != SUCCESS ) {
		std::cerr << "Launching test 2 FAILED\n";
		error = 255;
	}
	if( error == 0 ) {
		std::cout << "Test OK\n" << std::endl;
	} else {
		std::cerr << std::flush;
		std::cout << "Test 2 FAILED\n" << std::endl;
	}

	// done
	return error;
}

