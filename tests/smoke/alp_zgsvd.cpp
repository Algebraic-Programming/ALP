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
#include <alp/algorithms/householder_bidiag.hpp>
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

//** generate random rectangular matrix data: complex version */
template< typename T >
std::vector< T > generate_rectangular_matrix_data(
	size_t N,
	size_t M,
	const typename std::enable_if<
		grb::utils::is_complex< T >::value,
		void
	>::type * const = nullptr
) {
	std::vector< T > data( N * M );
	for( size_t i = 0; i < N; ++i ) {
		for( size_t j = 0; j < M; ++j ) {
			T val( std::rand(), std::rand() );
			data[ i * M + j ] = val / std::abs( val );
		}
	}
	return data;
}

//** generate random rectangular matrix data: real version */
template<
	typename T
>
std::vector< T >  generate_rectangular_matrix_data(
	size_t N,
	size_t M,
	const typename std::enable_if<
		!grb::utils::is_complex< T >::value,
		void
	>::type * const = nullptr
) {
	std::vector< T > data( N * M );
	for( size_t i = 0; i < N; ++i ) {
		for( size_t j = 0; j < M; ++j ) {
			data[ i * M + j ] = static_cast< T >( std::rand() ) / RAND_MAX;
		}
	}
	return data;
}

template<
	typename D,
	typename GenType,
	typename GenView,
	typename GenImfR,
	typename GenImfC,
	typename UType,
	typename UView,
	typename UImfR,
	typename UImfC,
	typename BType,
	typename BView,
	typename BImfR,
	typename BImfC,
	typename VType,
	typename VView,
	typename VImfR,
	typename VImfC,
	class Ring = Semiring< operators::add< D >, operators::mul< D >, identities::zero, identities::one >,
	class Minus = operators::subtract< D >
	>
RC check_svd_solution(
	Matrix< D, GenType, alp::Dense, GenView, GenImfR, GenImfC > &H,
	Matrix< D, UType, alp::Dense, UView, UImfR, UImfC > &U,
	Matrix< D, BType, alp::Dense, BView, BImfR, BImfC > &B,
	Matrix< D, VType, alp::Dense, VView, VImfR, VImfC > &V,
	const Ring &ring = Ring(),
	const Minus &minus = Minus()
) {
	RC rc = SUCCESS;
	const Scalar< D > zero( ring.template getZero< D >() );
	const Scalar< D > one( ring.template getOne< D >() );

	const size_t m = nrows( H );
	const size_t n = ncols( H );

#ifdef DEBUG
	std::cout << " ********************\n";
	std::cout << " ** check_solution **\n";
	std::cout << " input:\n";
	print_matrix( "  H  ", H );
	print_matrix( "  U  ", U );
	print_matrix( "  B  ", B );
	print_matrix( "  V  ", V );
	std::cout << " ********************\n";
#endif

 	alp::Matrix< D, GenType, alp::Density::Dense > UB( m, n );
	// UB = U * B
	rc = rc ? rc : set( UB, zero );
	rc = rc ? rc : mxm( UB, U, B, ring );

 	alp::Matrix< D, GenType, alp::Density::Dense > UBV( m, n );
	// UBV = U * B * V
	rc = rc ? rc : set( UBV, zero );
	rc = rc ? rc : mxm( UBV, UB, V, ring );


#ifdef DEBUG
	print_matrix( " UBV ", UBV );
#endif

	rc = foldl( UBV, H, minus );

	//Frobenius norm
	D fnorm = ring.template getZero< D >();
	rc = rc ? rc : alp::eWiseLambda(
		[ &fnorm, &ring ]( const size_t i, const size_t j, D &val ) {
			(void) i;
			(void) j;
			internal::foldl( fnorm, val * val, ring.getAdditiveOperator() );
		},
		UBV
	);
	fnorm = std::sqrt( fnorm );

#ifdef DEBUG
	std::cout << " FrobeniusNorm(UBV-H) = " << std::abs( fnorm ) << "\n";
#endif
	if( tol < std::abs( fnorm ) ) {
		std::cout << "The Frobenius norm is too large.\n";
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

	const Scalar< ScalarType > zero( ring.template getZero< ScalarType >() );
	const Scalar< ScalarType > one( ring.template getOne< ScalarType >() );

	// test thin, square and flat
	std::vector< size_t > m_arr { unit, unit, 2* unit };
	std::vector< size_t > n_arr { 2* unit, unit, unit, };
	for( size_t i = 0; i < 3; ++i ) {
		// dimensions of sqare matrices H, Q and R
		const size_t M = m_arr[ i ];
		const size_t N = n_arr[ i ];
		//const size_t K = std::min( N, M );

		alp::Matrix< ScalarType, General > H( M, N );
		alp::Matrix< ScalarType, General > B( M, N );
		alp::Matrix< ScalarType, structures::Orthogonal > U( M, M );
		alp::Matrix< ScalarType, structures::Orthogonal > V( N, N );
		{
			std::srand( RNDSEED );
			auto matrix_data = generate_rectangular_matrix_data< ScalarType >( M, N );
			rc = rc ? rc : alp::buildMatrix( H, matrix_data.begin(), matrix_data.end() );
		}
#ifdef DEBUG
		print_matrix( " input matrix H ", H );
#endif

		rc = rc ? rc : set( U, zero );
		rc = rc ? rc : set( B, H );
		rc = rc ? rc : set( V, zero );
		rc = rc ? rc : algorithms::householder_bidiag( U, B, V, ring );


#ifdef DEBUG
		print_matrix( "  U(out) ", U );
		print_matrix( "  B(out) ", B );
		print_matrix( "  V(out) ", V );
#endif

		rc = check_svd_solution( H, U, B, V, ring );
		if( rc != SUCCESS ) {
			std::cout << "Error: solution numerically wrong\n";
			return;
		}
	}
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
