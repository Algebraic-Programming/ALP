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
constexpr ScalarType tol = 1.e-12;

struct inpdata {
	std::string fname;
};

//** check the solution by calculating the Frobenius norm of (H-LL^T) **//
template< typename T, typename RingType, typename ZeroType >
alp::RC check_cholesky_solution(
	const  alp::Matrix< T, structures::Symmetric, Dense > &H,
	alp::Matrix< T, structures::UpperTriangular, Dense > &L,
	const ZeroType &zero_scalar,
	const RingType &ring
) {
	alp::RC rc = SUCCESS;
	const size_t N = nrows( H );
	alp::Matrix< T, alp::structures::Symmetric > LLT( N, N );
	alp::set( LLT, zero_scalar );
	auto LT = alp::get_view< alp::view::transpose >( L );
#ifdef DEBUG
	print_matrix( " << LLT >> ", LLT );
	print_matrix( " << LT >>  ", LT );
#endif
	alp::mxm( LLT, LT, L, ring );
#ifdef DEBUG
	print_matrix( " << LLT >> ", LLT );
#endif

	alp::Matrix< T, alp::structures::Symmetric > TMP( N, N );
	alp::set( TMP, zero_scalar );

	// LLT = -LLT
	alp::Scalar< T > alpha( -1 );
	rc = rc ? rc : alp::eWiseLambda(
		[ ]( const size_t i, const size_t j, T &val ) {
			(void)i;
			(void)j;
			val = -val;
		},
		LLT
	);
#ifdef DEBUG
	print_matrix( " << -LLT  >> ", LLT );
#endif

	// TMP = H - LLT
	rc = rc ? rc : alp::eWiseApply(
		TMP, H, LLT,
		ring.getAdditiveMonoid()
	);
#ifdef DEBUG
	print_matrix( " << H - LLT  >> ", TMP );
#endif

	//Frobenius norm
	T fnorm = 0;
	rc = rc ? rc : alp::eWiseLambda(
		[ &fnorm ]( const size_t i, const size_t j, T &val ) {
			(void)i;
			(void)j;
			fnorm += val*val;
		},
		TMP
	);
	fnorm = std::sqrt(fnorm);
	std::cout << " FrobeniusNorm(H-LL^T) = " << fnorm << "\n";
	if ( tol < fnorm ) {
		throw std::runtime_error(
			"The Frobenius norm is too large. "
			"Make sure that you have used SPD matrix as input."
		);
	}

	return rc;
}

void alp_program( const inpdata &unit, alp::RC &rc ) {
	rc = SUCCESS;
	auto fname = unit.fname;
	alp::utils::MatrixFileReader< ScalarType > parser_A( fname );
	size_t N = parser_A.n();

	alp::Semiring<
		alp::operators::add< ScalarType >,
		alp::operators::mul< ScalarType >,
		alp::identities::zero,
		alp::identities::one
	> ring;
	alp::Scalar< ScalarType > zero_scalar( ring.getZero< ScalarType >() );

	std::cout << "\tTesting ALP cholesky\n"
		"\tH = L * L^T\n";

	alp::Matrix< ScalarType, structures::Symmetric, Dense > H( N, N );
	alp::Matrix< ScalarType, structures::UpperTriangular, Dense > L( N, N );

	if( !parser_A.isSymmetric() ) {
		std::cout << "Symmetric matrix epxected as input!\n";
		rc = ILLEGAL;
		return;
	}

	rc = rc ? rc : alp::buildMatrix( H, parser_A.begin(), parser_A.end() );

	if( !internal::getInitialized( H ) ) {
		std::cout << " Matrix H is not initialized\n";
	}

#ifdef DEBUG
	print_matrix( std::string(" << H >> "), H);
	print_matrix( std::string(" << L >> "), L);
#endif

	rc = rc ? rc : alp::set( L, zero_scalar	);

	if( !internal::getInitialized( L ) ) {
		std::cout << " Matrix L is not initialized\n";
	}

	rc = rc ? rc : algorithms::cholesky_lowtr( L, H, ring );
#ifdef DEBUG
	print_matrix( std::string(" << L >> "), L);
#endif
	rc = rc ? rc : check_cholesky_solution( H, L, zero_scalar, ring );
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
		} else if( readflag != std::string("-fname") ) {
			std::cerr << "Given first argument is unknown\n";
			printUsage = true;
		} else {
			if( ! ( ( ss2 >> in.fname ) &&  ss2.eof() ) ) {
				std::cerr << "Error parsing second argument\n";
				printUsage = true;
			} else {
				// all fine
			}
		}
	} else {
		std::cout << "Wrong number of arguments\n" ;
		printUsage = true;
	}

	if( printUsage ) {
		std::cerr << "Usage: " << argv[ 0 ] << " -fname FILENAME \n";
		std::cerr << "  FILENAME .mtx file.\n";
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
