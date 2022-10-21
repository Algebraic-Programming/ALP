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
#include <alp/algorithms/householder_qr.hpp>
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

//** check if rows/columns or matrix Q are orthogonal */
template<
	typename T,
	typename Structure,
	typename ViewType,
	typename ImfR,
	typename ImfC,
	class Ring = Semiring< operators::add< T >, operators::mul< T >, identities::zero, identities::one >,
	class Minus = operators::subtract< T >
>
RC check_overlap(
	alp::Matrix< T, Structure, alp::Density::Dense, ViewType, ImfR, ImfC > &Q,
	const Ring &ring = Ring(),
	const Minus &minus = Minus()
) {
	const Scalar< T > zero( ring.template getZero< T >() );
	const Scalar< T > one( ring.template getOne< T >() );

	RC rc = SUCCESS;
	const size_t n = nrows( Q );

	// check if QxQt == I
	alp::Matrix< T, Structure, alp::Density::Dense, ViewType > Qtmp( n );
	rc = rc ? rc : set( Qtmp, zero );
	rc = rc ? rc : mxm(
		Qtmp,
		Q,
		conjugate( alp::get_view< alp::view::transpose >( Q ) ),
		ring
	);
	Matrix< T, Structure, Dense > Identity( n );
	rc = rc ? rc : alp::set( Identity, zero );
	auto id_diag = alp::get_view< alp::view::diagonal >( Identity );
	rc = rc ? rc : alp::set( id_diag, one );
	rc = rc ? rc : foldl( Qtmp, Identity, minus );

	//Frobenius norm
	T fnorm = ring.template getZero< T >();
	rc = rc ? rc : alp::eWiseLambda(
		[ &fnorm, &ring ]( const size_t i, const size_t j, T &val ) {
			(void) i;
			(void) j;
			internal::foldl( fnorm, val * val, ring.getAdditiveOperator() );
		},
		Qtmp
	);
	fnorm = std::sqrt( fnorm );

#ifdef DEBUG
	std::cout << " FrobeniusNorm(QQt - I) = " << std::abs( fnorm ) << "\n";
#endif
	if( tol < std::abs( fnorm ) ) {
		std::cout << "The Frobenius norm is too large: " << std::abs( fnorm ) << ".\n";
		return FAILED;
	}

	return rc;
}

//** check solution by calculating H-QR */
template<
	typename D,
	typename StructureGen,
	typename GenView,
	typename GenImfR,
	typename GenImfC,
	typename StructureOrth,
	typename OrthogonalView,
	typename OrthogonalImfR,
	typename OrthogonalImfC,
	class Minus = operators::subtract< D >,
	class Ring = Semiring< operators::add< D >, operators::mul< D >, identities::zero, identities::one >
>
RC check_solution(
	alp::Matrix< D, StructureGen, alp::Density::Dense, GenView, GenImfR, GenImfC > &H,
	alp::Matrix< D, StructureOrth, alp::Density::Dense, OrthogonalView, OrthogonalImfR, OrthogonalImfC > &Q,
	alp::Matrix< D, StructureGen, alp::Density::Dense, GenView, GenImfR, GenImfC > &R,
	const Ring &ring = Ring(),
	const Minus &minus = Minus()
) {
	RC rc = SUCCESS;
	const size_t n = nrows( H );
	const size_t m = ncols( H );

#ifdef DEBUG
	std::cout << " ** check_solution **\n";
	std::cout << " input matrices:\n";
	print_matrix( " << H >> ", H );
	print_matrix( " << Q >> ", Q );
	print_matrix( " << R >> ", R );
	std::cout << " ********************\n";
#endif

 	alp::Matrix< D, StructureGen, alp::Density::Dense > QR( n, m );
	// QR = Q * R
	const Scalar< D > zero( ring.template getZero< D >() );
	rc = rc ? rc : set( QR, zero );
	rc = rc ? rc : mxm( QR, Q, R, ring );
	// QR = QR - H
	rc = foldl( QR, H, minus );

#ifdef DEBUG
	print_matrix( " << QR - H >> ", QR );
#endif

	//Frobenius norm
	D fnorm = ring.template getZero< D >();
	rc = rc ? rc : alp::eWiseLambda(
		[ &fnorm, &ring ]( const size_t i, const size_t j, D &val ) {
			(void) i;
			(void) j;
			internal::foldl( fnorm, val * val, ring.getAdditiveOperator() );
		},
		QR
	);
	fnorm = std::sqrt( fnorm );

#ifdef DEBUG
	std::cout << " FrobeniusNorm(H-QR) = " << std::abs( fnorm ) << "\n";
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

	// dimensions of sqare matrices H, Q and R
	const size_t N = unit;
	const size_t M = 2 * unit;

	alp::Matrix< ScalarType, Orthogonal > Q( N );
	alp::Matrix< ScalarType, General > R( N, M );
	alp::Matrix< ScalarType, General > H( N, M );
	{
		std::srand( RNDSEED );
		auto matrix_data = generate_rectangular_matrix_data< ScalarType >( N, M );
		rc = rc ? rc : alp::buildMatrix( H, matrix_data.begin(), matrix_data.end() );
	}
#ifdef DEBUG
	print_matrix( " input matrix H ", H );
#endif

	rc = rc ? rc : algorithms::householder_qr( H, Q, R, ring );


#ifdef DEBUG
	print_matrix( " << Q >> ", Q );
	print_matrix( " << R >> ", R );
#endif

	rc = check_overlap( Q );
	if( rc != SUCCESS ) {
		std::cout << "Error: mratrix Q is not orthogonal\n";
	}

	rc = check_solution( H, Q, R );
	if( rc != SUCCESS ) {
		std::cout << "Error: solution numerically wrong\n";
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
