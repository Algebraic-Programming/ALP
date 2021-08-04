
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

#include <cstdio>
#include <utility>

#include "graphblas.hpp"

// sample data
static const double vec1_vals[ 3 ] = { 1, 2, 3 };
static const double vec2_vals[ 3 ] = { 4, 5, 6 };

static const size_t I[ 3 ] = { 0, 1, 2 };

static const double test1_in[ 3 ] = { 1, 1, 1 };
static const double test1_expect[ 3 ] = { 24, 30, 36 };

static const double test2_in[ 3 ] = { 1, 1, 1 };
static const double test2_expect[ 3 ] = { 15, 30, 45 };

// graphblas program
void grbProgram( const void *, const size_t in_size, int & error ) {
	error = 0;

	if( in_size != 0 ) {
		(void)fprintf( stderr, "Unit tests called with unexpected input\n" );
		error = 1;
		return;
	}

	// allocate
	grb::Vector< double > u( 3 );
	grb::Vector< double > v( 3 );
	grb::Matrix< double > M( 3, 3 );
	grb::Vector< double > test1( 3 );
	grb::Vector< double > out1( 3 );
	grb::Vector< double > test2( 3 );
	grb::Vector< double > out2( 3 );

	// semiring
	grb::Semiring< grb::operators::add< double >, grb::operators::mul< double >, grb::identities::zero, grb::identities::one > ring;

	// initialise vec
	const double * vec_iter = &( vec1_vals[ 0 ] );
	grb::RC rc = grb::buildVector( u, vec_iter, vec_iter + 3, SEQUENTIAL );
	if( rc != SUCCESS ) {
		std::cerr << "\t initial buildVector FAILED\n";
		error = 5;
	}

	if( ! error ) {
		vec_iter = &( vec2_vals[ 0 ] );
		rc = grb::buildVector( v, vec_iter, vec_iter + 3, SEQUENTIAL );
	}
	if( rc != SUCCESS ) {
		std::cerr << "\t initial buildVector FAILED\n";
		error = 10;
	}

	if( ! error ) {
		rc = grb::outerProduct( M, u, v, ring.getMultiplicativeOperator() );
	}
	if( rc != grb::SUCCESS ) {
		(void)fprintf( stderr, "Unexpected return code from grb::outerProduct: %d.\n", (int)rc );
		error = 15;
	}

	if( ! error && grb::nnz( M ) != 9 ) {
		std::cerr << "\t Unexpected number of nonzeroes in matrix: " << grb::nnz( M ) << ", expected 9.\n";
		error = 20;
	}

	if( ! error ) {
		const double * test1_iter = &( test1_in[ 0 ] );
		rc = grb::buildVector( test1, test1_iter, test1_iter + 3, SEQUENTIAL );
		if( rc == grb::SUCCESS ) {
			rc = grb::vxm< grb::descriptors::in_place >( out1, test1, M, ring );
		}
		if( rc != grb::SUCCESS ) {
			(void)fprintf( stderr,
				"Unexpected return code from premultiplying M by a vector (vxm): "
				"%d.\n",
				(int)rc );
			error = 25;
		}
	}

	if( ! error ) {
		if( grb::nnz( out1 ) != 3 ) {
			std::cerr << "\t Unexpected number of nonzeroes (premultiply): " << grb::nnz( out1 ) << ", expected 3\n";
			error = 30;
		}
		for( const auto & pair : out1 ) {
			size_t i = pair.first;
			if( pair.second != test1_expect[ i ] ) {
				(void)fprintf( stderr,
					"Premultiplying M by a vector of all ones, unexpected value %d "
					"at coordinate %zd, expected %d.\n",
					(int)pair.second, i, (int)test1_expect[ i ] );
				error = 35;
				break;
			}
		}
	}

	if( ! error ) {
		const double * test2_iter = &( test2_in[ 0 ] );
		rc = grb::buildVector( test2, test2_iter, test2_iter + 3, SEQUENTIAL );
		if( rc == grb::SUCCESS ) {
			rc = grb::vxm< grb::descriptors::in_place | grb::descriptors::transpose_matrix >( out2, test2, M, ring );
		}
		if( rc != grb::SUCCESS ) {
			(void)fprintf( stderr,
				"Unexpected return code from postmultiplying M by a vector (vxm): "
				"%d.\n",
				(int)rc );
			error = 40;
		}
	}

	if( ! error ) {
		if( grb::nnz( out2 ) != 3 ) {
			std::cerr << "\t Unexpected number of nonzeroes (postmultiply): " << grb::nnz( out1 ) << ", expected 3\n";
			error = 45;
		}
		for( const auto & pair : out2 ) {
			size_t i = pair.first;
			if( pair.second != test2_expect[ i ] ) {
				(void)fprintf( stderr,
					"Postmultiplying M by a vector of all ones, unexpected value "
					"%d at coordinate %zd, expected %d.\n",
					(int)pair.second, i, (int)test2_expect[ i ] );
				error = 50;
				break;
			}
		}
	}
}

int main( int argc, char ** argv ) {
	(void)argc;
	(void)printf( "Functional test executable: %s\n", argv[ 0 ] );

	int error;
	grb::Launcher< AUTOMATIC > launcher;
	if( launcher.exec( &grbProgram, NULL, 0, error ) != SUCCESS ) {
		(void)fprintf( stderr, "Test failed to launch\n" );
		error = 255;
	}
	if( error == 0 ) {
		(void)printf( "Test OK.\n\n" );
	} else {
		fflush( stderr );
		(void)printf( "Test FAILED.\n\n" );
	}

	// done
	return error;
}
