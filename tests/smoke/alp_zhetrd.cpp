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
#include <alp/algorithms/householder_tridiag.hpp>
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
	class Ring = Semiring< operators::add< T >, operators::mul< T >, identities::zero, identities::one >
>
RC check_overlap( alp::Matrix< T, Structure, alp::Density::Dense, ViewType > &Q, const Ring & ring = Ring() ) {
	RC rc = SUCCESS;
	const size_t n = nrows( Q );
#ifdef DEBUG
	std::cout << "Overlap matrix for Q:\n";
#endif
	for ( size_t i = 0; i < n; ++i ) {
		auto vi = get_view( Q, i, utils::range( 0, n ) );
		for ( size_t j = 0; j < n; ++j ) {
			auto vj = get_view( Q, j, utils::range( 0, n ) );
			Scalar< T > alpha( ring.template getZero< T >() );
			rc = dot( alpha, vi, vj, ring );
			if( rc != SUCCESS ) {
				std::cerr << " dot( alpha, vi, vj, ring ) failed\n";
				return PANIC;
			}
			if( i == j ) {
				if( std::abs( *alpha - ring.template getOne< T >() ) > tol ) {
					std::cerr << " vector " << i << " not normalized\n";
					return PANIC;
				}
			} else {
				if( std::abs( *alpha ) > tol ) {
					std::cerr << " vector " << i << " and vctor " << j << " are note orthogonal\n";
					return PANIC;
				}
			}
#ifdef DEBUG
			std::cout << "\t" << std::abs( *alpha );
#endif
		}
#ifdef DEBUG
		std::cout << "\n";
#endif
	}
#ifdef DEBUG
	std::cout << "\n";
#endif
	return rc;
}


//** check solution by calculating H-QTQh */
template<
	typename D,
	typename StructureSymm,
	typename StructureOrth,
	typename StructureTrDg,
	class Minus = operators::subtract< D >,
	class Ring = Semiring< operators::add< D >, operators::mul< D >, identities::zero, identities::one >
>
RC check_solution(
	alp::Matrix< D, StructureSymm, alp::Density::Dense > &H,
	alp::Matrix< D, StructureOrth, alp::Density::Dense > &Q,
	alp::Matrix< D, StructureTrDg, alp::Density::Dense > &T,
	const Ring &ring = Ring(),
	const Minus &minus = Minus()
) {
	RC rc = SUCCESS;
	const size_t n = nrows( Q );

#ifdef DEBUG
	std::cout << " ** check_solution **\n";
	std::cout << " input matrices:\n";
	print_matrix( " << H >> ", H );
	print_matrix( " << Q >> ", Q );
	print_matrix( " << T >> ", T );
	std::cout << " ********************\n";
#endif

	alp::Matrix< D, alp::structures::Square, alp::Density::Dense > QTQh( n );
	alp::Matrix< D, alp::structures::Square, alp::Density::Dense > QTQhmH( n );
	const Scalar< D > zero( ring.template getZero< D >() );

	rc = rc ? rc : set( QTQh, zero );
	rc = rc ? rc : mxm( QTQh, T, conjugate( alp::get_view< alp::view::transpose >( Q ) ), ring );
	rc = rc ? rc : set( QTQhmH, zero );
	rc = rc ? rc : mxm( QTQhmH, Q, QTQh, ring );
	rc = rc ? rc : set( QTQh, QTQhmH );
#ifdef DEBUG
	print_matrix( " << QTQhmH >> ", QTQhmH );
	print_matrix( " << H >> ", H );
	std::cout << "call foldl( mat, mat, minus )\n";
#endif

#ifndef TEMPDISABLE
	rc = foldl( QTQhmH, H, minus );
#else
	rc = rc ? rc : alp::eWiseLambda(
		[ &H, &minus, &zero ]( const size_t i, const size_t j, D &val ) {
			if ( j >= i ) {
				internal::foldl(
					val,
					internal::access( H, internal::getStorageIndex( H, i, j ) ),
					minus
				);
			} else {
				val = *zero;
			}
		},
		QTQhmH
	);
#endif

#ifdef DEBUG
	print_matrix( " << QTQhmH >> ", QTQhmH );
	print_matrix( " << H >> ", H );
#endif

	//Frobenius norm
	D fnorm = ring.template getZero< D >();
	rc = rc ? rc : alp::eWiseLambda(
		[ &fnorm, &ring ]( const size_t i, const size_t j, D &val ) {
			(void) i;
			(void) j;
			internal::foldl( fnorm, val * val, ring.getAdditiveOperator() );
		},
		QTQhmH
	);
	fnorm = std::sqrt( fnorm );

#ifdef DEBUG
	std::cout << " FrobeniusNorm(H-QTQh) = " << std::abs( fnorm ) << "\n";
#endif
	if( tol < std::abs( fnorm ) ) {
#ifdef DEBUG
		std::cout << " ----------------------\n";
		std::cout << " compare matrices\n";
		print_matrix( " << H >> ", H );
		print_matrix( " << QTQh >> ", QTQh );
		std::cout << " ----------------------\n";
#endif
		std::cout << "The Frobenius norm is too large.\n";
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

	// dimensions of sqare matrices H, Q and R
	size_t N = unit;

	alp::Matrix< ScalarType, Orthogonal > Q( N );
	alp::Matrix< ScalarType, HermitianOrSymmetricTridiagonal > T( N );
	alp::Matrix< ScalarType, HermitianOrSymmetric > H( N );
	{
		auto matrix_data = generate_symmherm_matrix_data< ScalarType >( N );
		rc = rc ? rc : alp::buildMatrix( H, matrix_data.begin(), matrix_data.end() );
	}
#ifdef DEBUG
	print_matrix( " input matrix H ", H );
#endif


 	rc = rc ? rc : algorithms::householder_tridiag( Q, T, H, ring );


#ifdef DEBUG
	print_matrix( " << Q >> ", Q );
	print_matrix( " << T >> ", T );
#endif

	rc = check_overlap( Q );
	if( rc != SUCCESS ) {
		std::cout << "Error: mratrix Q is not orthogonal\n";
	}

	rc = check_solution( H, Q, T );
	if( rc != SUCCESS ) {
		std::cout << "Error: solution numerically wrong\n";
	}
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