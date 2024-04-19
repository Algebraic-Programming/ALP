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

#include <graphblas/utils/timer.hpp>
#include <alp.hpp>
#include <alp/algorithms/householder_qr.hpp>
#include <alp/utils/iscomplex.hpp> // use from grb
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

struct inpdata {
	size_t N = 0;
	size_t repeat = 1;
};

//** generate random rectangular matrix data: complex version */
template< typename T >
std::vector< T > generate_rectangular_matrix_data(
	size_t N,
	size_t M,
	const typename std::enable_if<
		alp::utils::is_complex< T >::value,
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
		!alp::utils::is_complex< T >::value,
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

		// dimensions of sqare matrices H, Q and R
		const size_t N = unit.N;
		const size_t M = 2 * unit.N;

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

		timer.reset();

		rc = rc ? rc : algorithms::householder_qr( H, Q, R, ring );

		times += timer.time();

#ifdef DEBUG
		print_matrix( " << Q >> ", Q );
		print_matrix( " << R >> ", R );
#endif

		rc = check_overlap( Q );
		if( rc != SUCCESS ) {
			std::cout << "Error: mratrix Q is not orthogonal\n";
			return;
		}

		rc = check_solution( H, Q, R );
		if( rc != SUCCESS ) {
			std::cout << "Error: solution numerically wrong\n";
			return;
		}

	}

	std::cout << " time (ms, total) = " << times << "\n";
	std::cout << " time (ms, per repeat) = " << times / unit.repeat  << "\n";
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
