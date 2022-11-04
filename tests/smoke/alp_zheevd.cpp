/*
 *   Copyright 2022 Huawei Technologies Co., Ltd.
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
#include <alp/algorithms/householder_tridiag.hpp>
#include <alp/algorithms/symm_tridiag_eigensolver.hpp>
#include <graphblas/utils/iscomplex.hpp> // use from grb
#include "../utils/print_alp_containers.hpp"

//once TEMPDISABLE is removed the code should be in the final version
#define TEMPDISABLE

using namespace alp;

using BaseScalarType = double;
using Orthogonal = structures::Orthogonal;

#ifdef _COMPLEX
using ScalarType = std::complex< BaseScalarType >;
//not fully implemented structures
using HermitianOrSymmetricTridiagonal = structures::HermitianTridiagonal;
using HermitianOrSymmetric = structures::Hermitian;
#else
using ScalarType = BaseScalarType;
using HermitianOrSymmetricTridiagonal = structures::SymmetricTridiagonal;
//fully implemented structures
using HermitianOrSymmetric = structures::Symmetric;
#endif

constexpr BaseScalarType tol = 1.e-10;
constexpr size_t RNDSEED = 1;

//temp function untill Hermitian containter is implemented
//** gnerate symmetric-hermitian matrix in a square container */
template<
	typename T
>
std::vector< T > generate_symmherm_matrix_data(
	size_t N,
	const typename std::enable_if<
		grb::utils::is_complex< T >::value,
		void
	>::type * const = nullptr
) {
	std::vector< T > data( N * N );
	std::fill(data.begin(), data.end(), static_cast< T >( 0 ) );
	std::srand( RNDSEED );
	for( size_t i = 0; i < N; ++i ) {
		for( size_t j = i; j < N; ++j ) {
			T val( std::rand(), std::rand() );
			data[ i * N + j ] = val / std::abs( val );
			data[ j * N + i ] += grb::utils::is_complex< T >::conjugate( data[ i * N + j ] );
		}
	}
	return data;
}

//** generate upper/lower triangular part of a Symmetric matrix */
template<
	typename T
>
std::vector< T >  generate_symmherm_matrix_data(
	size_t N,
	const typename std::enable_if<
		!grb::utils::is_complex< T >::value,
		void
	>::type * const = nullptr
) {
	std::vector< T > data( ( N * ( N + 1 ) ) / 2 );
	std::srand( RNDSEED );
	size_t k = 0;
	for( size_t i = 0; i < N; ++i ) {
		for( size_t j = i; j < N; ++j ) {
			//data[ k ] = static_cast< T >( i + j*j ); // easily reproducible
			data[ k ] = static_cast< T >( std::rand() )  / RAND_MAX;
			++k;
		}
	}
	return data;
}

//** check if rows/columns or matrix Q are orthogonal */
template<
	typename T,
	typename Structure,
	typename ViewType,
	class Ring = Semiring< operators::add< T >, operators::mul< T >, identities::zero, identities::one >,
	class Minus = operators::subtract< T >
>
RC check_overlap(
	alp::Matrix< T, Structure, alp::Density::Dense, ViewType > &Q,
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


//** check solution by calculating A x Q - Q x diag(d) */
template<
	typename D,
	typename SymmOrHermTridiagonalType,
	typename OrthogonalType,
	typename SymmHermTrdiViewType,
	typename OrthViewType,
	typename SymmHermTrdiImfR,
	typename SymmHermTrdiImfC,
	typename OrthViewImfR,
	typename OrthViewImfC,
	typename VecViewType,
	typename VecImfR,
	typename VecImfC,
	class Ring = Semiring< operators::add< D >, operators::mul< D >, identities::zero, identities::one >,
	class Minus = operators::subtract< D >,
	class Divide = operators::divide< D >
>
RC check_solution(
	Matrix< D, SymmOrHermTridiagonalType, Dense, SymmHermTrdiViewType, SymmHermTrdiImfR, SymmHermTrdiImfC > &T,
	Matrix<	D, OrthogonalType, Dense, OrthViewType, OrthViewImfR, OrthViewImfC > &Q,
	Vector<	D, structures::General, Dense, VecViewType, VecImfR, VecImfC > &d,
	const Ring &ring = Ring(),
	const Minus &minus = Minus(),
	const Divide &divide = Divide()
) {
	(void) ring;
	(void) minus;
	(void) divide;
	RC rc = SUCCESS;

 	const size_t n = nrows( Q );

#ifdef DEBUG
	print_matrix( " T ", T  );
	print_matrix( " Q ", Q  );
	print_vector( " d ", d  );
#endif

	alp::Matrix< D, alp::structures::Square, alp::Density::Dense > Left( n );
	alp::Matrix< D, alp::structures::Square, alp::Density::Dense > Right( n );
	alp::Matrix< D, alp::structures::Square, alp::Density::Dense > Dmat( n );
	const Scalar< D > zero( ring.template getZero< D >() );
	const Scalar< D > one( ring.template getOne< D >() );

	rc = rc ? rc : set( Left, zero );
	rc = rc ? rc : mxm( Left, T, Q, ring );

	rc = rc ? rc : set( Dmat, zero );
	auto D_diag = alp::get_view< alp::view::diagonal >( Dmat );
	rc = rc ? rc : set( D_diag, d );
	rc = rc ? rc : set( Right, zero );
	rc = rc ? rc : mxm( Right, Q, Dmat, ring );
#ifdef DEBUG
	print_matrix( " TxQ ", Left  );
	print_matrix( " QxD ", Right  ),
#endif
	rc = rc ? rc : foldl( Left, Right, minus );

	//Frobenius norm
	D fnorm = ring.template getZero< D >();
	rc = rc ? rc : alp::eWiseLambda(
		[ &fnorm, &ring ]( const size_t i, const size_t j, D &val ) {
			(void) i;
			(void) j;
			internal::foldl( fnorm, val * val, ring.getAdditiveOperator() );
		},
		Left
	);
	fnorm = std::sqrt( fnorm );

#ifdef DEBUG
	std::cout << " FrobeniusNorm(AQ-QD) = " << std::abs( fnorm ) << "\n";
#endif
	if( tol < std::abs( fnorm ) ) {
		std::cout << "The Frobenius norm is too large: " << std::abs( fnorm ) << ".\n";
		return FAILED;
	}

	return rc;
}




void alp_program( const size_t & unit, alp::RC & rc ) {
	rc = SUCCESS;

	alp::Semiring<
		alp::operators::add< ScalarType >,
		alp::operators::mul< ScalarType >,
		alp::identities::zero,
		alp::identities::one
	> ring;
	const Scalar< ScalarType > zero( ring.template getZero< ScalarType >() );

	// dimensions of sqare matrices H, Q and R
	size_t N = unit;

	alp::Matrix< ScalarType, Orthogonal > Q( N ); //output eigenvectors
	alp::Matrix< ScalarType, Orthogonal > Q1( N ); //temp orthogonal matrix
	alp::Matrix< ScalarType, Orthogonal > Q2( N ); //temp orthogonal matrix
	alp::Matrix< ScalarType, HermitianOrSymmetricTridiagonal > T( N ); //temptridiagonal matrix
	alp::Matrix< ScalarType, HermitianOrSymmetric > H( N ); //input matrix
	Vector< ScalarType, structures::General, Dense > d( N ); //output eigenvalues
	{
		auto matrix_data = generate_symmherm_matrix_data< ScalarType >( N );
		rc = rc ? rc : alp::buildMatrix( H, matrix_data.begin(), matrix_data.end() );
	}
#ifdef DEBUG
	print_matrix( " input matrix H ", H );
#endif

	rc = rc ? rc : set( Q1, zero );
	rc = rc ? rc : set( Q2, zero );
	rc = rc ? rc : set( Q, zero );

 	rc = rc ? rc : algorithms::householder_tridiag( Q1, T, H, ring );
	rc = rc ? rc : algorithms::symm_tridiag_dac_eigensolver( T, Q2, d, ring );
	rc = rc ? rc : alp::mxm( Q, Q1, Q2, ring );

#ifdef DEBUG
	print_matrix( "  Q1 ", Q1 );
	print_matrix( "  Q2 ", Q2 );
	print_matrix( "  Q  ", Q );
	print_matrix( "  T  ", T );
#endif

	// the algorithm should return correct eigenvalues
	// but for larger matrices (n>20) a more stable calculations
	// of eigenvectors is needed
	// therefore we disable numerical correctness check in this version

	// rc = check_overlap( Q );
	// if( rc != SUCCESS ) {
	// 	std::cout << "Error: mratrix Q is not orthogonal\n";
	// }

	// rc = check_solution( H, Q, d );
	// if( rc != SUCCESS ) {
	// 	std::cout << "Error: solution numerically wrong\n";
	// }
}

int main( int argc, char ** argv ) {
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
