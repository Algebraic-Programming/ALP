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
#include <utility>

#include <alp.hpp>
#include <alp/algorithms/cholesky.hpp>
#include <alp/utils/parser/MatrixFileReader.hpp>
#include "../utils/print_alp_containers.hpp"

using namespace alp;


struct inpdata {
	size_t n = 100;
	struct fname{
		bool initialized = false;
		const std::string flag="-fname";
		std::string value;
	};

	struct nn{
		bool initialized = false;
		const std::string flag="-n";
		std::string value;
	};

	typedef std::tuple< fname, nn  > pack;
	pack packdata;

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
			auto lambda = [ &key, &val ] (auto &x){
					if( x.flag ==  key ) {
						x.value = val;
						x.initialized = true;
					}
			};
			iterate_lambda(packdata, lambda);
		}
	};

	std::string get_fname () {
		return (std::get<0>(packdata)).value;
	}



};

void alp_program( const inpdata & unit, alp::RC & rc ) {
	rc = SUCCESS;

	auto fname = std::get<0>(unit.packdata).value;

	alp::utils::MatrixFileReader<
		double
	> parser_A( fname );

	size_t N = parser_A.n();
	std::cout << "parser_A.n() = " << parser_A.n() << "\n";
	std::vector< double > A_data_full( N * N , 0 );
	std::vector< double > A_data( N * ( N + 1 ) / 2, 0 );

	for ( auto it = parser_A.begin() ; it != parser_A.end() ; ++it  ) {
		//std::cout << " i,j,v= " << it.i() << " " << it.j() << " " << it.v() << "\n";
		auto i = it.i();
		auto j = it.j();
		auto v = it.v();
		A_data_full[ i * N + j  ] = v;
	}

	size_t k = 0;
	for( size_t i = 0; i < N ; ++i ){
		for( size_t j = i; j < N ; ++j ){
			A_data[ k ] =  A_data_full[ i * N + j  ];
			k++;
		}
	}


	alp::Semiring< alp::operators::add< double >, alp::operators::mul< double >, alp::identities::zero, alp::identities::one > ring;

	std::cout << "\tTesting ALP cholesky\n"
	             "\tH = L * L^T\n";

	// dimensions of sqare matrices H, L

	alp::Matrix< double, structures::Symmetric, Dense > H( N, N );
	alp::Matrix< double, structures::UpperTriangular, Dense > L( N, N );

        alp::Scalar< double > zero_scalar( ring.getZero< double >() );
        alp::Scalar< double > one_scalar( ring.getOne< double >() );
	if( !internal::getInitialized( zero_scalar ) ) {
		std::cout << " zero_scalar is not initialized\n";
	}

        // rc = alp::set( H, one_scalar );
	rc = rc ? rc : alp::buildMatrix( H, A_data.begin(), A_data.end() );
	if( !internal::getInitialized( H ) ) {
		std::cout << " Matrix H is not initialized\n";
	}

	print_matrix( std::string(" << H >> "), H);
	print_matrix( std::string(" << L >> "), L);

	rc = rc ? rc : alp::set( L, zero_scalar );
	if( !internal::getInitialized( L ) ) {
		std::cout << " Matrix L is not initialized\n";
	}

	rc = rc ? rc : algorithms::cholesky_lowtr( L, H, ring );

	print_matrix( std::string(" << L >> "), L);

	alp::Matrix< double, alp::structures::Symmetric > LLT( N, N );
	alp::set( LLT, zero_scalar );
	print_matrix( " LLT ", LLT );
	auto LT = alp::get_view< alp::view::transpose >( L );
	print_matrix( " LT ", LT );
	alp::mxm( LLT, LT, L, ring );
	print_matrix( " LLT ", LLT );
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
