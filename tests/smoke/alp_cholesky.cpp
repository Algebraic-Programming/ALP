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

#include <cmath>
#include <iostream>
#include <sstream>
#include <vector>

#include <alp.hpp>
#include <alp/algorithms/cholesky.hpp>
#include <alp/utils/parser/MatrixFileReader.hpp>
#include "../utils/print_alp_containers.hpp"

using namespace alp;

typedef double ScalarType;
constexpr ScalarType tol = 1.e-10;
constexpr size_t RNDSEED = 1;

struct inpdata {
	std::string fname="";
	size_t N=0;
};

//** gnerate upper/lower triangular part of a SPD matrix */
template< typename T >
void generate_spd_matrix_full( size_t N, std::vector<T> &data ) {
	for( size_t i = 0; i < N; ++i ) {
		for( size_t j = 0; j < N; ++j ) {
			size_t k = i * N + j;
			if( i <= j ) {
				data[ k ] = static_cast< T >( std::rand() ) / static_cast< T >( RAND_MAX );
			}
			if( i == j ) {
				data[ k ] = data[ k ] + static_cast< T >( N );
			}
			if( i > j ) {
				data[ i * N + j ] = data[ j * N + i ];
			}

		}
	}
}


//** gnerate upper/lower triangular part of a SPD matrix */
template< typename T >
void generate_spd_matrix( size_t N, std::vector<T> &data ) {
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
}


//** check the solution by calculating the Frobenius norm of (H-LL^T) */
template<
	typename MatSymmType,
	typename MatUpTriangType,
	typename T = typename MatSymmType::value_type,
	typename Ring = Semiring< operators::add< T >, operators::mul< T >, identities::zero, identities::one >,
	typename Minus = operators::subtract< T >
>
alp::RC check_cholesky_solution(
	const MatSymmType &H,
	MatUpTriangType &L,
	const Ring &ring = Ring(),
	const Minus &minus = Minus()
) {
	alp::RC rc = SUCCESS;
	const Scalar< T > zero( ring.template getZero< T >() );
	const Scalar< T > one( ring.template getOne< T >() );
	const size_t N = nrows( H );
	MatSymmType LLT( N, N );
	rc = rc ? rc : alp::set( LLT, zero );
	auto LT = alp::get_view< alp::view::transpose >( L );
#ifdef DEBUG
	print_matrix( " << LLT >> ", LLT );
	print_matrix( " << LT >>  ", LT );
#endif
	rc = rc ? rc : alp::mxm( LLT, LT, L, ring );
#ifdef DEBUG
	print_matrix( " << LLT >> ", LLT );
#endif

	MatSymmType HminsLLt( N, N );
	rc = rc ? rc : alp::set( HminsLLt, zero );

	// LLT = -LLT
	Scalar< T > alpha( *zero );
	rc = rc ? rc : foldl( alpha, one, minus );
	rc = rc ? rc : foldl( LLT, alpha, ring.getMultiplicativeOperator() );

#ifdef DEBUG
	print_matrix( " << -LLT  >> ", LLT );
#endif

	// HminsLLt = H - LLT
	rc = rc ? rc : alp::eWiseApply(
		HminsLLt, H, LLT,
		ring.getAdditiveMonoid()
	);
#ifdef DEBUG
	print_matrix( " << H - LLT  >> ", HminsLLt );
#endif

	//Frobenius norm
	T fnorm = 0;
	rc = rc ? rc : alp::eWiseLambda(
		[ &fnorm ]( const size_t i, const size_t j, T &val ) {
			(void) i;
			(void) j;
			fnorm += val * val;
		},
		HminsLLt
	);
	fnorm = std::sqrt( fnorm );
#ifdef DEBUG
	std::cout << " FrobeniusNorm(H-LL^T) = " << fnorm << "\n";
#endif
	if( tol < fnorm ) {
		std::cout << "The Frobenius norm is too large. "
			"Make sure that you have used SPD matrix as input.\n";
		return FAILED;
	}

	return rc;
}

void alp_program( const inpdata &unit, alp::RC &rc ) {
	rc = SUCCESS;

	alp::Semiring<
		alp::operators::add< ScalarType >,
		alp::operators::mul< ScalarType >,
		alp::identities::zero,
		alp::identities::one
	> ring;
	const alp::Scalar< ScalarType > zero_scalar( ring.getZero< ScalarType >() );

	size_t N = 0;
	if( !unit.fname.empty() ) {
		alp::utils::MatrixFileReader< ScalarType > parser_A( unit.fname );
		N = parser_A.n();
		if( !parser_A.isSymmetric() ) {
			std::cout << "Symmetric matrix epxected as input!\n";
			rc = ILLEGAL;
			return;
		}
	} else if( unit.N != 0 )  {
		N = unit.N;
	}

	alp::Matrix< ScalarType, structures::UpperTriangular, Dense > L( N, N );
	alp::Matrix< ScalarType, structures::Symmetric, Dense > H( N, N );

	if( !unit.fname.empty() ) {
		alp::utils::MatrixFileReader< ScalarType > parser_A( unit.fname );
		rc = rc ? rc : alp::buildMatrix( H, parser_A.begin(), parser_A.end() );
	} else if( unit.N != 0 )  {
		std::vector< ScalarType > matrix_data( ( N * ( N + 1 ) ) / 2 );
		std::srand( RNDSEED );
		generate_spd_matrix( N, matrix_data );
		rc = rc ? rc : alp::buildMatrix( H, matrix_data.begin(), matrix_data.end() );
	}

	if( !internal::getInitialized( H ) ) {
		std::cout << " Matrix H is not initialized\n";
		return;
	}

#ifdef DEBUG
	print_matrix( std::string(" << H >> "), H );
	print_matrix( std::string(" << L >> "), L );
#endif

	rc = rc ? rc : alp::set( L, zero_scalar	);

	if( !internal::getInitialized( L ) ) {
		std::cout << " Matrix L is not initialized\n";
		return;
	}

	rc = rc ? rc : algorithms::cholesky_uptr( L, H, ring );
#ifdef DEBUG
	print_matrix( std::string(" << L >> "), L );
#endif
	rc = rc ? rc : check_cholesky_solution( H, L, ring );

	rc = rc ? rc : alp::set( L, zero_scalar	);
	// test blocked version, for bs = 1, N / 2 and N
	for( size_t bs = 1; bs <= N; bs = std::min( bs + N / 2, N ) ) {
		rc = rc ? rc : algorithms::cholesky_uptr_blk( L, H, bs, ring );
		rc = rc ? rc : check_cholesky_solution( H, L, ring );
		if( bs == N ) {
			break;
		}
	}

	// test non-blocked inplace version
	alp::Matrix< ScalarType, structures::Square, Dense > LL_original( N );
	alp::Matrix< ScalarType, structures::Square, Dense > LL( N );
	std::vector< ScalarType > matrix_data( N * N );
	std::srand( RNDSEED );
	generate_spd_matrix_full( N, matrix_data );
	rc = rc ? rc : alp::buildMatrix( LL, matrix_data.begin(), matrix_data.end() );
	rc = rc ? rc : alp::set( LL_original, LL );
#ifdef DEBUG
	print_matrix( " LL(input) ", LL );
#endif
	rc = rc ? rc : algorithms::cholesky_uptr( LL, ring );
#ifdef DEBUG
	print_matrix( " LL(output) ", LL );
#endif
	auto LLUT = get_view< structures::UpperTriangular >( LL );
	rc = rc ? rc : check_cholesky_solution( LL_original, LLUT, ring );

	// test non-blocked inplace version, for bs = 1, N / 2 and N
	for( size_t bs = 1; bs <= N; bs = std::min( bs + N / 2, N ) ) {
		rc = rc ? rc : alp::set( LL, LL_original );
		rc = rc ? rc : algorithms::cholesky_uptr_blk( LL, bs, ring );
		rc = rc ? rc : check_cholesky_solution( LL_original, LLUT, ring );
		if( bs == N ) {
			break;
		}
	}
}

int main( int argc, char ** argv ) {
	// defaults
	bool printUsage = false;
	inpdata in;

	// error checking
	if( argc == 3 ) {
		std::string readflag;
		std::istringstream ss1( argv[ 1 ] );
		std::istringstream ss2( argv[ 2 ] );
		if( ! ( ( ss1 >> readflag ) &&  ss1.eof() ) ) {
			std::cerr << "Error parsing first argument\n";
			printUsage = true;
		} else if(
			( readflag != std::string( "-fname" ) ) &&
			( readflag != std::string( "-n" ) )
		) {
			std::cerr << "Given first argument is unknown\n";
			printUsage = true;
		} else {
			if( readflag == std::string( "-fname" ) ) {
				if( ! ( ( ss2 >> in.fname ) &&  ss2.eof() ) ) {
					std::cerr << "Error parsing second argument\n";
					printUsage = true;
				}
			}

			if( readflag == std::string( "-n" ) ) {
				if( ! ( ( ss2 >> in.N ) &&  ss2.eof() ) ) {
					std::cerr << "Error parsing second argument\n";
					printUsage = true;
				}
			}

		}
	} else {
		std::cout << "Wrong number of arguments\n" ;
		printUsage = true;
	}

	if( printUsage ) {
		std::cerr << "Usage: \n";
		std::cerr << "       " << argv[ 0 ] << " -fname FILENAME.mtx \n";
		std::cerr << "      or  \n";
		std::cerr << "       " << argv[ 0 ] << " -n N \n";
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
