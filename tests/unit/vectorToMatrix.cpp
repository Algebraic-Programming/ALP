
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

#include "graphblas/utils/MatrixVectorIterator.hpp"

#include "graphblas.hpp"

using namespace grb;

// sample data
static const size_t vec_vals[ 15 ] = { 0, 0, 1, 1, 1, 1, 0, 0, 0, 2, 2, 2, 1, 2, 0 };

static const double test1_in[ 15 ] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
static const double test1_expect_arr[ 3 ] = { 6, 10, 20 };
static const double test1_expect_void_arr[ 3 ] = { 6, 5, 4 };

static const double test2_in[ 3 ] = { 1, 1, 1 };
static const double test2_expect_arr[ 15 ] = { 1, 1, 2, 2, 2, 2, 1, 1, 1, 5, 5, 5, 2, 5, 1 };
static const double test2_expect_void_arr[ 15 ] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };

// converter functions
std::pair< std::pair< size_t, size_t >, double > converter_function(
	const size_t &ind,
	const size_t &val
) {
	return std::make_pair(
		std::make_pair( ind, val ),
		static_cast< double >( 1 + val * val )
	);
};

std::pair< size_t, size_t > converter_function_void(
	const size_t &ind,
	const size_t &val
) {
	return std::make_pair( ind, val );
}

// tests a converter
template< typename OutputType >
void testIterator( int &error,
	grb::utils::VectorToMatrixConverter< OutputType, size_t > &converter,
	grb::Matrix< OutputType > &M,
	grb::Vector< double > &test1,
	const double * const test1_expect,
	grb::Vector< double > &out1,
	grb::Vector< double > &test2,
	const double * const test2_expect,
	grb::Vector< double > &out2
) {
	grb::Semiring<
		grb::operators::add< double >, grb::operators::mul< double >,
		grb::identities::zero, grb::identities::one
	> ring;

	grb::RC rc = buildMatrixUnique( M,
		converter.begin(), converter.end(),
		PARALLEL
	);

	if( rc != grb::SUCCESS ) {
		std::cerr << "Unexpected return code from Matrix build (M): "
			<< grb::toString( rc ) << ".\n";
		error = 10;
		return;
	}

	if( grb::nnz( M ) != 15 ) {
		std::cerr << "\t Unexpected number of nonzeroes in matrix: " << grb::nnz( M ) << ", expected 15.\n";
		error = 15;
	}

	// test that the matrix is correct by premultiplying by a vector of all ones
	if( !error ) {
		const double * const test1_iter = &( test1_in[ 0 ] );
		rc = grb::buildVector( test1, test1_iter, test1_iter + 15, SEQUENTIAL );
		if( rc == grb::SUCCESS ) {
			rc = grb::clear( out1 );
		}
		if( rc == grb::SUCCESS ) {
			rc = grb::vxm( out1, test1, M, ring );
		}
		if( rc != grb::SUCCESS ) {
			std::cerr << "nexpected return code from premultiplying M by a vector (vxm): "
				<< grb::toString( rc ) << ".\n";
			error = 20;
		}
	}

	if( !error ) {
		if( nnz( out1 ) != 3 ) {
			std::cerr << "\t Unexpected number of nonzeroes (premultiply): "
				<< grb::nnz( out1 ) << ", expected 3\n";
			error = 30;
		}
		for( const auto &pair : out1 ) {
			size_t i = pair.first;
			if( pair.second != test1_expect[ i ] ) {
				std::cerr << "Premultiplying M by a vector of all ones, unexpected value "
					<< pair.second << " at coordinate " << i << ", expected "
					<< test1_expect[ i ] << ".\n";
				error = 35;
				break;
			}
		}
	}

	// test that the matrix is correct by postmultiplying by a vector of all ones

	if( !error ) {
		const double * const test2_iter = &( test2_in[ 0 ] );
		rc = grb::buildVector( test2, test2_iter, test2_iter + 3, SEQUENTIAL );
		if( rc == grb::SUCCESS ) {
			rc = grb::clear( out2 );
		}
		if( rc == grb::SUCCESS ) {
			rc = grb::vxm< grb::descriptors::transpose_matrix >( out2, test2, M, ring );
		}
		if( rc != grb::SUCCESS ) {
			std::cerr << "Unexpected return code from postmultiplying M by a vector "
			       << "(vxm): " << grb::toString( rc ) << ".\n";
			error = 40;
		}
	}

	if( ! error ) {
		if( grb::nnz( out2 ) != 15 ) {
			std::cerr << "\t Unexpected number of nonzeroes (postmultiply): "
				<< grb::nnz( out2 ) << ", expected 15.\n";
			error = 50;
		}
		for( const auto &pair : out2 ) {
			const size_t i = pair.first;
			if( pair.second != test2_expect[ i ] ) {
				std::cerr << "Postmultiplying M by a vector of all ones, unexpected value "
					<< pair.second << " at coordinate " << i << ", expected "
					<< test2_expect[ i ] << ".\n";
				error = 55;
				break;
			}
		}
	}
}

// graphblas program
void grbProgram( const void *, const size_t in_size, int &error ) {
	error = 0;

	if( in_size != 0 ) {
		(void)fprintf( stderr, "Unit tests called with unexpected input\n" );
		error = 1;
		return;
	}

	// allocate
	grb::Vector< size_t > vec( 15 );
	grb::Matrix< double > M( 15, 3 );
	grb::Matrix< void > V( 15, 3 );
	grb::Vector< double > test1( 15 );
	grb::Vector< double > out1( 3 );
	grb::Vector< double > test2( 3 );
	grb::Vector< double > out2( 15 );

	// initialise vec
	const size_t * vec_iter = &( vec_vals[ 0 ] );
	const grb::RC grb_rc = grb::buildVector( vec,
		vec_iter, vec_iter + 15,
		SEQUENTIAL
	);
	if( grb_rc != SUCCESS ) {
		std::cerr << "\t initial buildVector FAILED\n";
		error = 5;
	}

	// test 1
	if( grb::spmd<>::pid() == 0 ) {
		std::cout << "\t Test 1: InputType size_t, OutputType double, direct "
			"construction...\n";
	}
	if( error == 0 ) {
		auto converter = grb::utils::VectorToMatrixConverter< double, size_t >(
			vec,
			converter_function
		);
		testIterator(
			error, converter, M,
			test1, &(test1_expect_arr[ 0 ]), out1,
			test2, &(test2_expect_arr[ 0 ]), out2
		);
	}

	// test 2
	if( error == 0 ) {
		if( grb::spmd<>::pid() == 0 ) {
			std::cout << "\t Test 2: InputType size_t, OutputType void, direct "
				"construction...\n";
		}
		auto converter = grb::utils::VectorToMatrixConverter< void, size_t >(
			vec,
			converter_function_void
		);
		testIterator( error, converter, V,
			test1, &(test1_expect_void_arr[ 0 ]), out1,
			test2, &(test2_expect_void_arr[ 0 ]), out2
		);
		if( error != 0 ) {
			error += 100;
		}
	}

	// test 3
	if( error == 0 ) {
		if( grb::spmd<>::pid() == 0 ) {
			std::cout << "\t Test 3: InputType size_t, OutputType double, factory "
				"construction...\n";
		}
		grb::utils::VectorToMatrixConverter< double, size_t > converter =
			grb::utils::makeVectorToMatrixConverter< double >( vec, converter_function );
		testIterator( error, converter, M,
			test1, &(test1_expect_arr[ 0 ]), out1,
			test2, &(test2_expect_arr[ 0 ]), out2
		);
		if( error != 0 ) {
			error += 200;
		}
	}

	// test 4
	if( error == 0 ) {
		if( grb::spmd<>::pid() == 0 ) {
			std::cout << "\t Test 4: InputType size_t, OutputType void, factory "
				"construction...\n";
		}
		auto converter = grb::utils::makeVectorToMatrixConverter< void >(
			vec,
			converter_function_void
		);
		testIterator( error, converter, V,
			test1, &(test1_expect_void_arr[ 0 ]), out1,
			test2, &(test2_expect_void_arr[ 0 ]), out2
		);
		if( error != 0 ) {
			error += 300;
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

