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
#include <vector>

#ifdef _COMPLEX
#include <complex>
#include <cmath>
#include <iomanip>
#endif

#include <alp.hpp>
#include <alp/algorithms/backsubstitution.hpp>
#include <graphblas/utils/iscomplex.hpp> // use from grb
#ifdef DEBUG
#include "../utils/print_alp_containers.hpp"
#endif

using namespace alp;

using BaseScalarType = double;
using Orthogonal = structures::Orthogonal;
using General = structures::General;

#ifdef _COMPLEX
using ScalarType = std::complex< BaseScalarType >;
#else
using ScalarType = BaseScalarType;
#endif

constexpr BaseScalarType tol = 1.e-10;
constexpr size_t RNDSEED = 1;

//** generate real upper triangular positive definite matrix data */
template<
	typename T,
	const typename std::enable_if<
		!grb::utils::is_complex< T >::value,
		void
	>::type * const = nullptr
>
std::vector<T> generate_upd_matrix( size_t N ) {
	std::vector<T> data( ( N * ( N + 1 ) ) / 2 );
	size_t k = 0;
	for( size_t i = 0; i < N; ++i ) {
		for( size_t j = i; j < N; ++j ) {
			data[ k ] = static_cast< T >( std::rand() ) / static_cast< T >( RAND_MAX );
			if( i == j ) {
				data[ k ] = data[ k ] + static_cast< T >( N );
			}
			++k;
		}
	}
	return( data );
}

//** generate complex upper triangular positive definite matrix data */
template<
	typename T,
	const typename std::enable_if<
		grb::utils::is_complex< T >::value,
		void
	>::type * const = nullptr
>
std::vector<T> generate_upd_matrix( size_t N  ) {
	std::vector<T> data( ( N * ( N + 1 ) ) / 2 );
	size_t k = 0;
	for( size_t i = 0; i < N; ++i ) {
		for( size_t j = i; j < N; ++j ) {
			T val( std::rand(), std::rand() );
			data[ k ] = val / std::abs( val );
			if( i == j ) {
				data[ k ] = data[ k ] + static_cast< T >( N );
			}
			++k;
		}
	}
	return ( data );
}

//** check if rows/columns or matrix Q are orthogonal */
template<
	typename D = double,
	typename Ring = Semiring< operators::add< D >, operators::mul< D >, identities::zero, identities::one >,
	typename Minus = operators::subtract< D >,
	typename Divide = operators::divide< D > >
RC check_solution(
	Matrix< D, structures::UpperTriangular, Dense > &A,
	Vector< D > &x,
	Vector< D > &b,
	const Ring &ring = Ring(),
	const Minus &minus = Minus(),
	const Divide &divide = Divide()
) {
	(void) divide;
	const Scalar< D > zero( ring.template getZero< D >() );
	const Scalar< D > one( ring.template getOne< D >() );

	RC rc = SUCCESS;

	const size_t n = nrows( A );

	alp::Vector< D > lhs( n );
	rc = rc ? rc : alp::set( lhs, zero );
	auto lhs_matview =  get_view< alp::view::matrix >( lhs );
	rc = rc ? rc : alp::mxm( lhs_matview, A, x, ring );
	rc = rc ? rc : alp::foldl( lhs_matview, b, minus );

	D alpha = ring.template getZero< D >();
	rc = rc ? rc : alp::norm2( alpha, lhs, ring );
	if(  std::abs( alpha ) > tol ) {
		std::cout << "Numerical error too large: |Ax-b| = " << alpha << ".\n";
		return FAILED;
	}

	return rc;
}


void alp_program( const size_t &unit, alp::RC &rc ) {
	rc = SUCCESS;

	alp::Semiring<
		alp::operators::add< ScalarType >,
		alp::operators::mul< ScalarType >,
		alp::identities::zero,
		alp::identities::one
	> ring;

	// dimensions of upper triangular matrix
	const size_t N = unit;

	alp::Vector< ScalarType > b( N );
	alp::Vector< ScalarType > x( N );
	alp::Matrix< ScalarType, structures::UpperTriangular > A( N );
	{
		std::srand( RNDSEED );
		auto matrix_data = generate_upd_matrix< ScalarType >( N );
		rc = rc ? rc : alp::buildMatrix( A, matrix_data.begin(), matrix_data.end() );
	}
	rc = rc ? rc : alp::set( b, Scalar< ScalarType >( ring.template getOne< ScalarType >() ) );
	rc = rc ? rc : alp::set( x, Scalar< ScalarType >( ring.template getZero< ScalarType >() ) );

#ifdef DEBUG
	print_matrix( " input matrix A ", A );
	print_vector( " input vector b ", b );
#endif

	rc = rc ? rc : algorithms::backsubstitution( A, x, b, ring );

#ifdef DEBUG
	print_vector( " output vector x ", x );
#endif
	rc = rc ? rc : check_solution( A, x, b );

}

int main( int argc, char **argv ) {
	// defaults
	bool printUsage = false;
	size_t in = 5;

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
	alp::Launcher< AUTOMATIC > launcher;
	alp::RC out;
	if( launcher.exec( &alp_program, in, out, true ) != SUCCESS ) {
		std::cerr << "Launching test FAILED\n";
		return 255;
	}
	if( out != SUCCESS ) {
		std::cerr << "Test FAILED (" << alp::toString( out ) << ")" << std::endl;
	} else {
		std::cout << "Test OK" << std::endl;
	}
	return 0;
}
