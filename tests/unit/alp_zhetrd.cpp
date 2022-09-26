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
#include <alp/utils/iscomplex.hpp> // tmp copy from grb, change after rebase
#include "../utils/print_alp_containers.hpp"

//once TEMPDISABLE is remnoved the code should be in the final version
#define TEMPDISABLE

using namespace alp;

using BaseScalarType = double;
#ifdef _COMPLEX
using ScalarType = std::complex< BaseScalarType >;
#else
using ScalarType = BaseScalarType;
#endif

constexpr BaseScalarType tol = 1.e-10;
constexpr size_t RNDSEED = 1;

//temp function untill Hermitian containter is implemented
//** gnerate symmetric-hermitian matrix in square container */
template<
	typename T
>
void generate_symmherm_matrix(
	size_t N,
	std::vector<T> &data
) {
	std::srand( RNDSEED );
	for( size_t i = 0; i < N; ++i ) {
		for( size_t j = i; j < N; ++j ) {
#ifdef _COMPLEX
			T val( std::rand(), std::rand() );
			data[ i * N + j ] = val / std::abs( val );
			//data[ i * N + j ] = std::complex< double >( i + 1 , j * j + 1 );
			if( j != i ) {
				data[ j * N + i ] = std::conj( data[ i * N + j ]  );
			}
			if( j == i ) {
				data[ i * N + j ] += std::conj( data[ i * N + j ]  );
			}
#endif
		}
	}
}

//** gnerate upper/lower triangular part of a Symmetric matrix */
template<
	typename T
>
void generate_symm_matrix(
	size_t N,
	std::vector<T> &data
) {
	std::srand( RNDSEED );
	size_t k = 0;
	for( size_t i = 0; i < N; ++i ) {
		for( size_t j = i; j < N; ++j ) {
			//data[ k ] = static_cast< T >( i + j*j ) ;
#ifdef _COMPLEX
			T val( std::rand(), std::rand() );
			data[ k ] = val / std::abs( val );
#else
			data[ k ] = static_cast< T >( std::rand() )  / RAND_MAX;
#endif
			++k;
		}
	}
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
			if( grb::utils::is_complex< T >::value ) {
				//someting like this should be implemented
				//rc = rc ? rc : dot_complex( alpha, vi, vj, ring );
				Vector< T, structures::General, Dense > vj_star( n );
				rc = set( vj_star, vj );
				rc = rc ? rc : eWiseLambda(
					[ ]( const size_t i, T &val ) {
						(void) i;
						val = grb::utils::is_complex< T >::conjugate( val );
					},
					vj_star
				);
				rc = rc ? rc : dot( alpha, vi, vj_star, ring );
			} else {
				rc = dot( alpha, vi, vj, ring );
			}
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
	//Q=conjugate(Q)
	if( grb::utils::is_complex< D >::value ) {
		rc = rc ? rc : alp::eWiseLambda(
			[ ]( const size_t i, const size_t j, D &val ) {
				(void) i;
				(void) j;
				val = grb::utils::is_complex< D >::conjugate( val );
			},
			Q
		);
	}
	rc = rc ? rc : mxm( QTQh, T, alp::get_view< alp::view::transpose >( Q ), ring );
	//Q=conjugate(Q)
	if( grb::utils::is_complex< D >::value ) {
		rc = rc ? rc : alp::eWiseLambda(
			[ ]( const size_t i, const size_t j, D &val ) {
				(void) i;
				(void) j;
				val = grb::utils::is_complex< D >::conjugate( val );
			},
			Q
		);
	}
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

#ifdef TEMPDISABLE
	//not fully implemented structures
	using Orthogonal = structures::Square;
#ifdef _COMPLEX
	using HermitianTridiagonal = structures::Square;
	using Hermitian = structures::Square;
#else
	using SymmetricTridiagonal = structures::Symmetric;
	//fully implemented structures
	using Symmetric = structures::Symmetric;
#endif
#endif

#ifdef _COMPLEX
	alp::Matrix< ScalarType, Orthogonal > Q( N );
	alp::Matrix< ScalarType, HermitianTridiagonal > T( N );
	alp::Matrix< ScalarType, Hermitian > H( N );
	std::vector< ScalarType > matrix_data( N * N );
	generate_symmherm_matrix( N, matrix_data );
#else
	alp::Matrix< ScalarType, Orthogonal > Q( N );
	alp::Matrix< ScalarType, SymmetricTridiagonal > T( N );
	alp::Matrix< ScalarType, Symmetric > H( N );
	std::vector< ScalarType > matrix_data( ( N * ( N + 1 ) ) / 2 );
	generate_symm_matrix( N, matrix_data );
#endif

	{
		rc = rc ? rc : alp::buildMatrix( H, matrix_data.begin(), matrix_data.end() );
#ifdef DEBUG
		print_matrix( " input matrix H ", H );
#endif
	}

 	rc = algorithms::householder_tridiag( Q, T, H, ring );


#ifdef DEBUG
	print_matrix( " << Q >> ", Q );
	print_matrix( " << T >> ", T );
#endif

	rc = check_overlap( Q );
	auto Qt = alp::get_view< alp::view::transpose >( Q );
	rc = rc ? rc : check_overlap( Qt );
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
