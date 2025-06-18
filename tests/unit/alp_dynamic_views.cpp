
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

#include <utility>
#include <iostream>

#include <alp.hpp>
#include "../utils/print_alp_containers.hpp"

template< typename T >
void init_matrix( std::vector< T > &A, const size_t rows, const size_t cols ) {

	size_t multiplier;
	for( multiplier = 1; multiplier < rows; multiplier *= 10 );

	for( size_t row = 0; row < rows; ++row ) {
		for( size_t col = 0; col < cols; ++col ) {
			A[ row * cols + col ] = multiplier * row + col;
		}
	}
}

// alp program
void alpProgram( const size_t &n, alp::RC &rc ) {

	typedef double T;

	alp::Semiring< alp::operators::add< T >, alp::operators::mul< T >, alp::identities::zero, alp::identities::one > ring;

	T zero = ring.getZero< T >();

	// allocate
	const size_t m = 2 * n;
	std::vector< T > M_data( m * n, zero );
	init_matrix( M_data, m, n );

	alp::Matrix< T, alp::structures::General > M( m, n );
	alp::buildMatrix( M, M_data.begin(), M_data.end() );
	print_matrix( "M", M );
	std::cout << "------------" << std::endl;

	std::vector< size_t > sel_r_data{ 3, 1, 5 };
	std::vector< size_t > sel_c_data{ 2, 4, 0 };
	alp::Vector< size_t > sel_r( sel_r_data.size() );
	alp::Vector< size_t > sel_c( sel_c_data.size() );
	alp::buildVector( sel_r, sel_r_data.begin(), sel_r_data.end() );
	alp::buildVector( sel_c, sel_c_data.begin(), sel_c_data.end() );

	// select view
	auto Ms = alp::get_view< alp::structures::General >( M, sel_r, sel_c );
	print_matrix( "Ms", Ms );
	std::cout << "------------" << std::endl;

	// apply another select view to test telescopic view select->select
	std::vector< size_t > sel1_data{ 2, 1, 0 };
	alp::Vector< size_t > sel1( sel1_data.size() );
	alp::buildVector( sel1, sel1_data.begin(), sel1_data.end() );
	auto Mss = alp::get_view< alp::structures::General >( Ms, sel1, sel1 );
	print_matrix( "Mss", Mss );
	std::cout << "------------" << std::endl;

	// transposed view
	auto MsT = alp::get_view< alp::view::Views::transpose >( Ms );
	print_matrix( "Ms^T", MsT );
	std::cout << "------------" << std::endl;

	// gather view
	auto MsTg = alp::get_view( MsT, alp::utils::range( 0, 2 ), alp::utils::range( 0, 2 ) );
	print_matrix( "Ms^Tg", MsTg );
	std::cout << "------------" << std::endl;

	// another select view to test telescopic view dynamic->static->dynamic
	std::vector< size_t > sel2_r_data{ 1, 0 };
	std::vector< size_t > sel2_c_data{ 0, 1 };
	alp::Vector< size_t > sel2_r( sel2_r_data.size() );
	alp::Vector< size_t > sel2_c( sel2_c_data.size() );
	alp::buildVector( sel2_r, sel2_r_data.begin(), sel2_r_data.end() );
	alp::buildVector( sel2_c, sel2_c_data.begin(), sel2_c_data.end() );
	auto MsTgs = alp::get_view< alp::structures::General >( MsTg, sel2_r, sel2_c );
	print_matrix( "Ms^Tgs", MsTgs );
	std::cout << "------------" << std::endl;

	// Vector views
	// allocate vector
	std::vector< T > v_data( m, zero );
	init_matrix( v_data, m, 1 );
	alp::Vector< T > v( m );
	alp::buildMatrix( static_cast< decltype( v )::base_type & >( v ), v_data.begin(), v_data.end() );
	print_vector( "v", v );

	// select view over a vector
	auto v_view = alp::get_view< alp::structures::General >( v, sel_r );
	print_vector( "v_view", v_view );

	// select view over select view
	std::vector< size_t > sel2_v_data{ 2, 0, 1 };
	alp::Vector< size_t > sel2_v( sel2_v_data.size() );
	alp::buildVector( sel2_v, sel2_v_data.begin(), sel2_v_data.end() );
	auto v_view_2 = alp::get_view< alp::structures::General >( v_view, sel2_v );
	print_vector( "v_view_2", v_view_2 );

	// matrix view over select x select view
	auto v_mat = alp::get_view< alp::view::matrix >( v_view_2 );
	print_matrix( "v_mat", v_mat );

	rc = alp::SUCCESS;

}

int main( int argc, char ** argv ) {
	// defaults
	bool printUsage = false;
	size_t in = 100;

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
		} else if( read == 0 ) {
			std::cerr << "n must be a positive number\n";
			printUsage = true;
		} else {
			// all OK
			in = read;
		}
	}
	if( printUsage ) {
		std::cerr << "Usage: " << argv[ 0 ] << " [n]\n";
		std::cerr << "  -n (optional, default is 100): an integer, the "
					 "test size.\n";
		return 1;
	}
	std::cout << "Functional test executable: " << argv[ 0 ] << "\n";

	std::cout << "This is functional test " << argv[ 0 ] << "\n";
	alp::Launcher< alp::AUTOMATIC > launcher;
	alp::RC out;
	if( launcher.exec( &alpProgram, in, out, true ) != alp::SUCCESS ) {
		std::cerr << "Launching test FAILED\n";
		return 255;
	}
	if( out != alp::SUCCESS ) {
		std::cerr << "Test FAILED (" << alp::toString( out ) << ")" << std::endl;
	} else {
		std::cout << "Test OK" << std::endl;
	}
	return 0;
}

