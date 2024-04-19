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

#include <graphblas/utils/Timer.hpp>
#include <alp.hpp>
#include <alp/algorithms/backsubstitution.hpp>
#include <graphblas/utils/iscomplex.hpp> // use from grb
#ifdef DEBUG
#include "../utils/print_alp_containers.hpp"
#endif

using namespace alp;

using BaseScalarType = double;

#ifdef _COMPLEX
using ScalarType = std::complex< BaseScalarType >;
#else
using ScalarType = BaseScalarType;
#endif

constexpr BaseScalarType tol = 1.e-10;
constexpr size_t RNDSEED = 1;

struct inpdata {
	size_t N = 0;
	size_t repeat = 1;
};

template< typename T >
T random_value();

template<>
BaseScalarType random_value< BaseScalarType >() {
	return static_cast< BaseScalarType >( rand() ) / RAND_MAX;
}

template<>
std::complex< BaseScalarType > random_value< std::complex< BaseScalarType > >() {
	const BaseScalarType re = random_value< BaseScalarType >();
	const BaseScalarType im = random_value< BaseScalarType >();
	return std::complex< BaseScalarType >( re, im );
}


/** generate data */
template< typename T >
std::vector< T > generate_data( size_t N ) {
	std::vector< T > data( N );
	for( size_t i = 0; i < N; ++i ) {
		data[ i ] = random_value< T >();
	}
	return( data );
}

/** generate real upper triangular positive definite matrix data */
template<
	typename T,
	const typename std::enable_if<
		!grb::utils::is_complex< T >::value,
		void
	>::type * const = nullptr
>
std::vector< T > generate_upd_matrix( size_t N ) {
	std::vector< T > data( ( N * ( N + 1 ) ) / 2 );
	size_t k = 0;
	for( size_t i = 0; i < N; ++i ) {
		for( size_t j = i; j < N; ++j ) {
			data[ k ] = random_value< T >();
			if( i == j ) {
				data[ k ] = data[ k ] + static_cast< T >( N );
			}
			++k;
		}
	}
	return( data );
}

/** generate complex upper triangular positive definite matrix data */
template<
	typename T,
	const typename std::enable_if<
		grb::utils::is_complex< T >::value,
		void
	>::type * const = nullptr
>
std::vector< T > generate_upd_matrix( size_t N  ) {
	std::vector< T > data( ( N * ( N + 1 ) ) / 2 );
	size_t k = 0;
	for( size_t i = 0; i < N; ++i ) {
		for( size_t j = i; j < N; ++j ) {
			data[ k ] = random_value< T >();
			if( i == j ) {
				data[ k ] = data[ k ] + static_cast< T >( N );
			}
			++k;
		}
	}
	return ( data );
}

/** check if Ax == b */
template<
	typename D = double,
	typename Ring = Semiring< operators::add< D >, operators::mul< D >, identities::zero, identities::one >,
	typename Minus = operators::subtract< D >
>
RC check_solution(
	Matrix< D, structures::UpperTriangular, Dense > &A,
	Vector< D > &x,
	Vector< D > &b,
	const Ring &ring = Ring(),
	const Minus &minus = Minus()
) {
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
	if( std::abs( alpha ) > tol ) {
		std::cout << "Numerical error too large: |Ax-b| = " << alpha << ".\n";
		return FAILED;
	}

	return rc;
}

/** check if AX == B */
template<
	typename D = double,
	typename StructX,
	typename StructB,
	typename Ring = Semiring< operators::add< D >, operators::mul< D >, identities::zero, identities::one >,
	typename Minus = operators::subtract< D >,
	typename Divide = operators::divide< D >
>
RC check_solution(
	Matrix< D, structures::UpperTriangular, Dense > &A,
	Matrix< D, StructX, Dense > &X,
	Matrix< D, StructB, Dense > &B,
	const Ring &ring = Ring(),
	const Minus &minus = Minus(),
	const Divide &divide = Divide()
) {
	(void) divide;
	const Scalar< D > zero( ring.template getZero< D >() );
	const Scalar< D > one( ring.template getOne< D >() );

	RC rc = SUCCESS;

	if( ncols( A ) != nrows( X ) ){
		std::cerr << "Asked to check incompatible structures.\n";
		return FAILED;
	}

	const size_t n = nrows( A );
	const size_t m = ncols( X );

	alp::Matrix< D, StructB > LHS( n, m );
	rc = rc ? rc : alp::set( LHS, zero );
	rc = rc ? rc : alp::mxm( LHS, A, X, ring );
	rc = rc ? rc : alp::foldl( LHS, B, minus );

	//Frobenius norm
	D fnorm = 0;
	rc = rc ? rc : alp::eWiseLambda(
		[ &fnorm ]( const size_t i, const size_t j, D &val ) {
			(void) i;
			(void) j;
			fnorm += val * val;
		},
		LHS
	);
	fnorm = std::sqrt( fnorm );
	if( tol < std::abs( fnorm ) ) {
		std::cout << " FrobeniusNorm(AX-B) = " << fnorm << " is too large\n";
		return FAILED;
	}

	return rc;
}

void alp_program( const inpdata &unit, alp::RC &rc ) {
	rc = SUCCESS;

	grb::utils::Timer timer;
	timer.reset();
	double times = 0;

	for( size_t j = 0; j < unit.repeat; ++j ) {

		alp::Semiring<
			alp::operators::add< ScalarType >,
			alp::operators::mul< ScalarType >,
			alp::identities::zero,
			alp::identities::one
			> ring;

		// dimensions of upper triangular matrix
		const size_t N = unit.N;

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

		const size_t M = N / 2;
		// version with matrices
		alp::Matrix< ScalarType, structures::General > X( N, M );
		alp::Matrix< ScalarType, structures::General > B( N, M );
		rc = rc ? rc : alp::set( X, Scalar< ScalarType >( ring.template getZero< ScalarType >() ) );
		{
			auto matrix_data = generate_data< ScalarType >( N * M );
			rc = rc ? rc : alp::buildMatrix( B, matrix_data.begin(), matrix_data.end() );
		}
#ifdef DEBUG
		print_matrix( " input matrix B ", B );
#endif

		timer.reset();

		rc = rc ? rc : algorithms::backsubstitution( A, X, B, ring );

		times += timer.time();

		rc = rc ? rc : check_solution( A, X, B );

	}

	std::cout << " times(total) = " << times << "\n";
	std::cout << " times(per repeat) = " << times / unit.repeat  << "\n";

}

int main( int argc, char **argv ) {
	// defaults
	bool printUsage = false;
	inpdata in;

	// error checking
	if(
		( argc == 3 ) || ( argc == 5 )
	) {
		std::string readflag;
		std::istringstream ss1( argv[ 1 ] );
		std::istringstream ss2( argv[ 2 ] );
		if( ! ( ( ss1 >> readflag ) &&  ss1.eof() ) ) {
			std::cerr << "Error parsing\n";
			printUsage = true;
		} else if(
			readflag != std::string( "-n" )
		) {
			std::cerr << "Given first argument is unknown\n";
			printUsage = true;
		} else {
			if( ! ( ( ss2 >> in.N ) &&  ss2.eof() ) ) {
				std::cerr << "Error parsing\n";
				printUsage = true;
			}
		}

		if( argc == 5 ) {
			std::string readflag;
			std::istringstream ss1( argv[ 3 ] );
			std::istringstream ss2( argv[ 4 ] );
			if( ! ( ( ss1 >> readflag ) &&  ss1.eof() ) ) {
				std::cerr << "Error parsing\n";
				printUsage = true;
			} else if(
				readflag != std::string( "-repeat" )
			) {
				std::cerr << "Given third argument is unknown\n";
				printUsage = true;
			} else {
				if( ! ( ( ss2 >> in.repeat ) &&  ss2.eof() ) ) {
					std::cerr << "Error parsing\n";
					printUsage = true;
				}
			}
		}

	} else {
		std::cout << "Wrong number of arguments\n";
		printUsage = true;
	}

	if( printUsage ) {
		std::cerr << "Usage: \n";
		std::cerr << "       " << argv[ 0 ] << " -n N \n";
		std::cerr << "      or  \n";
		std::cerr << "       " << argv[ 0 ] << " -n N   -repeat N \n";
		return 1;
	}

	alp::RC rc = alp::SUCCESS;
	alp_program( in, rc );
	if( rc == alp::SUCCESS ) {
		std::cout << "Test OK\n";
		return 0;
	} else {
		std::cout << "Test FAILED\n";
		return 1;
	}
}
