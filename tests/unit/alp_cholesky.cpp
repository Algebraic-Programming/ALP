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

// an attempt to make more general parser
struct inpdata {
	struct fname{
		bool initialized = false;
		const std::string flag="-fname";
		std::string value;
	};

	struct n{
		bool initialized = false;
		const std::string flag="-n";
		double value;
	};

	typedef std::tuple< fname, n  > pack;
	pack packdata;

	template< typename X >
	typename std::enable_if< std::is_same< X, double >::value , X>::type
	stoX ( const std::string &s ) {
		return( std::stod( s ) );
	};

	template< typename X >
	typename std::enable_if< std::is_same< X, std::string >::value , X>::type
	stoX ( const std::string &s ) {
		return( s );
	};

	template< std::size_t I = 0, typename... Tp, typename F >
	inline typename std::enable_if<I == sizeof...(Tp), void>::type
	iterate_lambda(std::tuple<Tp...>& t, const F &func) {
		(void)func;
		(void)t;
	};

	template< std::size_t I = 0, typename... Tp, typename F >
	inline typename std::enable_if<I < sizeof...(Tp), void>::type
	iterate_lambda(std::tuple<Tp...>& t, const F &func) {
		//std::cout << " " << std::get<I>(t).flag;
		func( std::get<I>(t) );
		iterate_lambda<I + 1, Tp...>(t, func);
	};

	void print_flags () {
		std::cout << "available flags:\n";
		auto lambda = [](auto &x){
			std::cout << " " << x.flag << " " << x.initialized << " " << x.value << "\n";
		};
		iterate_lambda(packdata, lambda);
		std::cout << "\n";
	};

	void set_flags ( int argc, char ** argv ) {
		for( int i = 1 ; i < argc ; i += 2 ) {
			std::string key = argv[ i ];
			std::string val = argv[ i + 1 ];
			auto lambda = [ &key, &val, this ] (auto &x){
					if( x.flag ==  key ) {
						x.value = stoX< decltype(x.value) >( val );
						x.initialized = true;
					}
			};
			iterate_lambda(packdata, lambda);
		}
	};

	std::string get_fname ( ) const {
		return (std::get<0>(packdata)).value;
	}



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
	auto fname = unit.get_fname();
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
	// bool printUsage = false;
	inpdata in;

	// // error checking
	// if( argc > 2 ) {
	// 	printUsage = true;
	// }
	// if( argc == 2 ) {
	// 	size_t read;
	// 	std::istringstream ss( argv[ 1 ] );
	// 	if( ! ( ss >> read ) ) {
	// 		std::cerr << "Error parsing first argument\n";
	// 		printUsage = true;
	// 	} else if( ! ss.eof() ) {
	// 		std::cerr << "Error parsing first argument\n";
	// 		printUsage = true;
	// 	} else if( read % 2 != 0 ) {
	// 		std::cerr << "Given value for n is odd\n";
	// 		printUsage = true;
	// 	} else {
	// 		// all OK
	// 		in.n = read;
	// 	}
	// }
	// if( printUsage ) {
	// 	std::cerr << "Usage: " << argv[ 0 ] << " [n]\n";
	// 	std::cerr << "  -n (optional, default is 100): an even integer, the "
	// 				 "test size.\n";
	// 	return 1;
	// }

	in.set_flags( argc, argv );
	in.print_flags();

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
