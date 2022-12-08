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
#include <alp/algorithms/svd.hpp>
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
	typename MatH,
	typename MatU,
	typename MatS,
	typename MatV,
	typename D = typename MatH::value_type,
	class Ring = Semiring< operators::add< D >, operators::mul< D >, identities::zero, identities::one >,
	class Minus = operators::subtract< D >,
	std::enable_if_t<
		is_matrix< MatH >::value &&
		is_matrix< MatU >::value &&
		is_matrix< MatS >::value &&
		is_matrix< MatV >::value &&
		structures::is_a< typename MatH::structure, structures::General >::value &&
		structures::is_a< typename MatU::structure, structures::Orthogonal >::value &&
		structures::is_a< typename MatS::structure, structures::RectangularDiagonal >::value &&
		structures::is_a< typename MatV::structure, structures::Orthogonal >::value &&
		is_semiring< Ring >::value &&
		is_operator< Minus >::value
	> * = nullptr
>
RC check_svd_solution(
	MatH &H,
	MatU &U,
	MatS &S,
	MatV &V,
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
	print_matrix( "  S  ", S );
	print_matrix( "  V  ", V );
	std::cout << " ********************\n";
#endif

	MatH US( m, n );
	// UB = U * S
	rc = rc ? rc : set( US, zero );
	rc = rc ? rc : mxm( US, U, S, ring );

	MatH USV( m, n );
	// USV = U * S * V
	rc = rc ? rc : set( USV, zero );
	rc = rc ? rc : mxm( USV, US, V, ring );


#ifdef DEBUG
	print_matrix( " USV ", USV );
#endif

	rc = foldl( USV, H, minus );

	//Frobenius norm
	D fnorm = ring.template getZero< D >();
	rc = rc ? rc : alp::eWiseLambda(
		[ &fnorm, &ring ]( const size_t i, const size_t j, D &val ) {
			(void) i;
			(void) j;
			internal::foldl( fnorm, val * val, ring.getAdditiveOperator() );
		},
		USV
	);
	fnorm = std::sqrt( fnorm );

#ifdef DEBUG
	std::cout << " FrobeniusNorm(USV-H) = " << std::abs( fnorm ) << "\n";
#endif
	if( tol < std::abs( fnorm ) ) {
		std::cout << "The Frobenius norm is too large.\n";
		return FAILED;
	}

	return rc;
}



void alp_program( const inpdata &unit, alp::RC &rc ) {
	rc = SUCCESS;

	// test thin, square and flat
	std::vector< size_t > m_arr { unit.N, unit.N, 2 * unit.N };
	std::vector< size_t > n_arr { 2 * unit.N, unit.N, unit.N };

	grb::utils::Timer timer;
	timer.reset();
	double times[] = { 0, 0, 0 };

	for( size_t j = 0; j < unit.repeat; ++j ) {

		alp::Semiring<
			alp::operators::add< ScalarType >,
			alp::operators::mul< ScalarType >,
			alp::identities::zero,
			alp::identities::one
			> ring;

		const Scalar< ScalarType > zero( ring.template getZero< ScalarType >() );
		const Scalar< ScalarType > one( ring.template getOne< ScalarType >() );

		for( size_t i = 0; i < 3; ++i ) {
			// dimensions of sqare matrices H, Q and R
			const size_t M = m_arr[ i ];
			const size_t N = n_arr[ i ];
			//const size_t K = std::min( N, M );

			alp::Matrix< ScalarType, General > H( M, N );
			alp::Matrix< ScalarType, structures::RectangularDiagonal > S( M, N );
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

			timer.reset();

			rc = rc ? rc : algorithms::svd( H, U, S, V, ring );

			times[ i ] += timer.time();

#ifdef DEBUG
			print_matrix( "  U(out) ", U );
			print_matrix( "  S(out) ", S );
			print_matrix( "  V(out) ", V );
#endif

			rc = check_svd_solution( H, U, S, V, ring );
			if( rc != SUCCESS ) {
				std::cout << "Error: solution numerically wrong\n";
				return;
			}
		}
	}
	for( size_t i = 0; i < 3; ++i ) {
		std::cout << " Matrix " << m_arr[ i ] << " x " << n_arr[ i ] << "\n";
		std::cout << " time (ms, total) = " << times[ i ] << "\n";
		std::cout << " time (ms, per repeat) = " << times[ i ] / unit.repeat  << "\n";
	}
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
