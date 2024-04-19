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
#include <alp/algorithms/householder_lu.hpp>
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

/** generate random rectangular matrix data: complex version */
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

/** generate random rectangular matrix data: real version */
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
	typename GeneralType,
	typename GenView,
	typename GenImfR,
	typename GenImfC,
	typename UType,
	typename UView,
	typename UImfR,
	typename UImfC,
	typename LType,
	typename LView,
	typename LImfR,
	typename LImfC,
	class Ring = Semiring< operators::add< D >, operators::mul< D >, identities::zero, identities::one >,
	class Minus = operators::subtract< D >
>
RC check_lu_solution(
	Matrix< D, GeneralType, alp::Dense, GenView, GenImfR, GenImfC > &H,
	Matrix< D, LType, alp::Dense, LView, LImfR, LImfC > &L,
	Matrix< D, UType, alp::Dense, UView, UImfR, UImfC > &U,
	Vector< size_t > &p,
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
	print_matrix( "  L  ", L );
	print_matrix( "  U  ", U );
	print_vector( "  p  ", p );
	std::cout << " ********************\n";
#endif

 	alp::Matrix< D, GeneralType, alp::Density::Dense > LU( m, n );
	// LU = L * U
	rc = rc ? rc : set( LU, zero );
	rc = rc ? rc : mxm( LU, L, U, ring );

	// until #591 is implemented we use no_permutation_vec explicitly
	alp::Vector< size_t > no_permutation_vec( n );
	alp::set< alp::descriptors::use_index >( no_permutation_vec, alp::Scalar< size_t >( 0 ) );

	// LU = LU - [p]H // where p are row permutations
	auto pH = alp::get_view< alp::structures::General >( H, p, no_permutation_vec );
	rc = foldl( LU, pH, minus );

#ifdef DEBUG
	print_matrix( " LU - [p]H >> ", LU );
#endif

	//Frobenius norm
	D fnorm = ring.template getZero< D >();
	rc = rc ? rc : alp::eWiseLambda(
		[ &fnorm, &ring ]( const size_t i, const size_t j, D &val ) {
			(void) i;
			(void) j;
			internal::foldl( fnorm, val * val, ring.getAdditiveOperator() );
		},
		LU
	);
	fnorm = std::sqrt( fnorm );

#ifdef DEBUG
	std::cout << " FrobeniusNorm(LU-[p]H) = " << std::abs( fnorm ) << "\n";
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
			const size_t K = std::min( N, M );

			alp::Matrix< ScalarType, General > H( M, N );
			alp::Matrix< ScalarType, structures::LowerTrapezoidal > L( M, K );
			alp::Matrix< ScalarType, structures::UpperTrapezoidal > U( K, N );
			alp::Vector< size_t > permutation_vec( M );
			{
				std::srand( RNDSEED );
				auto matrix_data = generate_rectangular_matrix_data< ScalarType >( M, N );
				rc = rc ? rc : alp::buildMatrix( H, matrix_data.begin(), matrix_data.end() );
			}
#ifdef DEBUG
			print_matrix( " input matrix H ", H );
#endif

			rc = rc ? rc : set( L, zero );
			rc = rc ? rc : set( U, zero );

			timer.reset();

			rc = rc ? rc : algorithms::householder_lu( H, L, U, permutation_vec, ring );
			times[ i ] += timer.time();


#ifdef DEBUG
			print_matrix( "  H(out) ", H );
			print_matrix( "  L(out) ", L );
			print_matrix( "  U(out) ", U );
#endif

			rc = check_lu_solution( H, L, U, permutation_vec, ring );
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
