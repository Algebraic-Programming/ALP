
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
static const double vec1_vals[ 3 ] = { 1, 2, 3 };
static const double vec2_vals[ 3 ] = { 4, 5, 6 };

static const double m1[ 3 ] = { 0.5, 3.4, 5 };

static const size_t I1[ 3 ] = { 0, 1, 2 };
static const size_t J1[ 3 ] = { 0, 1, 2 };

static const double m2[ 6 ] = { 0.5, -1, 23, 34, -4.4, 3.14 };

static const size_t I2[ 6 ] = { 0, 0, 1, 1, 2, 2 };
static const size_t J2[ 6 ] = { 1, 2, 0, 2, 0, 1 };

static const double mask_test1_in[ 3 ] = { 1, 1, 1 };
static const double mask_test1_expect[ 3 ] = { 4, 10, 18 };

static const double mask_test2_in[ 3 ] = { 1, 1, 1 };
static const double mask_test2_expect[ 3 ] = { 11, 20, 27 };

void grbProgram( const void *, const size_t in_size, int &error ) {
	error = 0;

	if( in_size != 0 ) {
		(void)fprintf( stderr, "Unit tests called with unexpected input\n" );
		error = 1;
		return;
	}

	// allocate
	grb::Vector< double > u( 3 );
	grb::Vector< double > v( 3 );
	grb::Matrix< double > Result1( 3, 3 );
	grb::Matrix< double > Result2( 3, 3 );
	grb::Matrix< double > mask1( 3, 3 );
	grb::Matrix< double > mask2( 3, 3 );
	grb::Vector< double > test1( 3 );
	grb::Vector< double > out1( 3 );
	grb::Vector< double > test2( 3 );
	grb::Vector< double > out2( 3 );

	// semiring
	grb::Semiring<
		grb::operators::add< double >, grb::operators::mul< double >,
		grb::identities::zero, grb::identities::one
	> ring;

	// initialise vec
	const double * vec_iter = &(vec1_vals[ 0 ]);
	grb::RC rc = grb::buildVector( u, vec_iter, vec_iter + 3, SEQUENTIAL );
	if( rc != SUCCESS ) {
		std::cerr << "\t initial buildVector FAILED\n";
		error = 5;
	}

	if( !error ) {
		vec_iter = &(vec2_vals[ 0 ]);
		rc = grb::buildVector( v, vec_iter, vec_iter + 3, SEQUENTIAL );
	}
	if( rc != SUCCESS ) {
		std::cerr << "\t initial buildVector FAILED\n";
		error = 10;
	}

	if( !error ) {
		rc = grb::buildMatrixUnique( mask1, &( I1[ 0 ] ), &( J1[ 0 ] ), m1, 3, SEQUENTIAL );
	}

	if( rc != SUCCESS ) {
		std::cerr << "\t first buildMatrix FAILED\n";
		error = 15;
	}

	if( !error ) {
		rc = grb::buildMatrixUnique( mask2, &( I2[ 0 ] ), &( J2[ 0 ] ), m2, 6, SEQUENTIAL );
	}

	if( rc != SUCCESS ) {
		std::cerr << "\t second buildMatrix FAILED\n";
		error = 20;
	}

	if( !error ) {
		rc = grb::maskedOuter( Result1, mask1, u, v, ring.getMultiplicativeOperator(), RESIZE );
		rc = rc ? rc : grb::maskedOuter( Result1, mask1, u, v, ring.getMultiplicativeOperator() );
	}
	if( rc != grb::SUCCESS ) {
		std::cerr << "Unexpected return code from grb::maskedOuter: "
			<< toString( rc ) << ".\n";
		error = 25;
	}

	if( !error && grb::nnz( Result1 ) != 3 ) {
		std::cerr << "\t Unexpected number of nonzeroes in matrix: "
			<< grb::nnz( Result1 ) << ", expected 3.\n";
		error = 30;
	}

	if( !error ) {
		rc = grb::maskedOuter( Result2, mask2, u, v, ring.getMultiplicativeOperator(), RESIZE );
		rc = rc ? rc : grb::maskedOuter( Result2, mask2, u, v, ring.getMultiplicativeOperator() );
	}
	if( rc != grb::SUCCESS ) {
		std::cerr << "Unexpected return code from grb::maskedOuter: "
			<< toString( rc ) << ".\n";
		error = 35;
	}

	if( !error && grb::nnz( Result2 ) != 6 ) {
		std::cerr << "\t Unexpected number of nonzeroes in matrix: "
			<< grb::nnz( Result2 ) << ", expected 6.\n";
		error = 40;
	}

	if( !error ) {
		const double * const test1_iter = &( mask_test1_in[ 0 ] );
		rc = grb::buildVector( test1, test1_iter, test1_iter + 3, SEQUENTIAL );
		if( rc == grb::SUCCESS ) {
			rc = grb::vxm( out1, test1, Result1, ring );
		}
		if( rc != grb::SUCCESS ) {
			std::cerr << "Unexpected return code from premultiplying M by a vector (vxm): "
				<< toString( rc ) << ".\n";
			error = 45;
		}
	}

	if( !error ) {
		if( grb::nnz( out1 ) != 3 ) {
			std::cerr << "\t Unexpected number of nonzeroes (premultiply): "
				<< grb::nnz( out1 ) << ", expected 3\n";
			error = 50;
		}
		for( const auto &pair : out1 ) {
			size_t i = pair.first;
			if( pair.second != mask_test1_expect[ i ] ) {
				std::cerr << "Premultiplying Result1 by a vector of all ones, "
					<< "unexpected value " << pair.second << " "
					<< "at coordinate " << i << ", expected "
					<< mask_test1_expect[ i ] << ".\n";
				error = 55;
				break;
			}
		}
	}

	if( !error ) {
		const double * const test2_iter = &( mask_test2_in[ 0 ] );
		rc = grb::buildVector( test2, test2_iter, test2_iter + 3, SEQUENTIAL );
		if( rc == grb::SUCCESS ) {
			rc = grb::vxm< grb::descriptors::transpose_matrix >( out2, test2, Result2, ring );
		}
		if( rc != grb::SUCCESS ) {
			std::cerr << "Unexpected return code from postmultiplying M by a vector (vxm): "
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
				std::cerr << "Postmultiplying M by a vector of all ones, "
					<< "unexpected value " << pair.second << " "
					<< "at coordinate " << i << ", "
					<< "expected " << mask_test2_expect[ i ] << ".\n";
				error = 70;
				break;
			}
		}
	}
}

int main( int argc, char ** argv ) {
	(void)argc;
	std::cout << "Functional test executable: " << argv[ 0 ] << "\n";

	int error;
	grb::Launcher< AUTOMATIC > launcher;

	if( launcher.exec( &grbProgram, nullptr, 0, error ) != SUCCESS ) {
		std::cerr << "Test failed to launch\n";
		error = 255;
	}
	if( error == 0 ) {
		std::cout << "Test OK\n" << std::endl;
	} else {
		std::cerr << std::flush;
		std::cout << "Test FAILED\n" << std::endl;
	}

	// done
	return error;
}

