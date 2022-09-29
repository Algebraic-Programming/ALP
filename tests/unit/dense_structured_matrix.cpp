
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
#include <string>
#include <type_traits>
#include <vector>
#include <memory>

#include <alp.hpp>

template< typename StructuredMat >
void ask_questions( const StructuredMat & M, std::string name ) {

	using M_type = typename std::remove_const< typename std::remove_reference< decltype( M ) >::type >::type;

	std::cout << name << "( " << alp::nrows( M ) << ", " << alp::ncols( M ) << " )" << std::endl;
	std::cout << "Is " << name << ":" << std::endl;
	std::cout << "\ta structured Matrix? " << alp::is_structured_matrix< M_type >::value << std::endl;
	std::cout << "\tgeneral? " << alp::structures::is_a< typename M_type::structure, alp::structures::General >::value << std::endl;
	std::cout << "\tsquare? " << alp::structures::is_a< typename M_type::structure, alp::structures::Square >::value << std::endl;
	std::cout << "\tfull rank? " << alp::structures::is_a< typename M_type::structure, alp::structures::FullRank >::value << std::endl;
	std::cout << "\tnon-singular? " << alp::structures::is_a< typename M_type::structure, alp::structures::NonSingular >::value << std::endl;
	std::cout << "\tsymmetric? " << alp::structures::is_in< alp::structures::Symmetric, typename M_type::structure::inferred_structures >::value << std::endl;
}

void alp_program( const size_t & n, alp::RC & rc ) {

	std::cout << "\tStarting structured matrices test with size: " << n << "\n";
	rc = alp::SUCCESS;

	// initialize test
	alp::Matrix< float, alp::structures::General > M( n, n );
	alp::Matrix< float, alp::structures::Square > A( n );
	alp::Matrix< float, alp::structures::Orthogonal > Orth( n );
	alp::Matrix< float, alp::structures::SymmetricTridiagonal > SymmTridiag( n );
	alp::Matrix< float, alp::structures::Hermitian > Hermit( n );
	// TODO: temporarily comented until containers are ready
	//alp::Matrix< float, alp::structures::NonSingular > B( n, n );
	//alp::Matrix< float, alp::structures::FullRank > C( n, 2 * n );
	auto At = alp::get_view< alp::view::transpose >( A );
	auto Mt = alp::get_view< alp::view::transpose >( M );
	auto Mview = alp::get_view( M, alp::utils::range(0,4), alp::utils::range(0,4) );
	auto Sq_Mref = alp::get_view< alp::structures::Square > ( M );

	ask_questions( M, "M" );
	ask_questions( A, "A" );
	ask_questions( Orth, "Orth" );
	ask_questions( SymmTridiag, "SymmTridiag" );
	ask_questions( Hermit, "Hermit" );
	// TODO: temporarily comented until containers are ready
	//ask_questions( B, "B" );
	//ask_questions( C, "C" );

	ask_questions( At, "At" );
	ask_questions( Mt, "Mt" );
	ask_questions( Mview, "Mview" );
	ask_questions( Sq_Mref, "Sq_Mref" );

	auto v_diag = alp::get_view< alp::view::diagonal >( M );
	auto v_view1 = alp::get_view( v_diag );
	//auto v_view2 = alp::get_view( v_diag, alp::utils::range(1,2) );
	std::cout << "v_diag( " << alp::getLength( v_diag ) << " )" << std::endl;
	std::cout << "v_view1( " << alp::getLength( v_view1 ) << " )" << std::endl;
	//std::cout << "v_view2( " << alp::getLength( v_view2 ) << " )" << std::endl;

	// TODO: temporarily comented until containers are ready
	//alp::Matrix< float, alp::structures::Band< alp::Interval<-2, 5> > > BM0( n, n );
	//alp::Matrix< float, alp::structures::Band< alp::RightOpenInterval<-2> > > BM1( n, n );
	//alp::Matrix< float, alp::structures::Band< alp::LeftOpenInterval<-2> > > BM2( n, n );
	//alp::Matrix< double, alp::structures::Band< alp::Interval<-2>, alp::Interval<1>, alp::Interval<3> > > BM3( n, n );
	rc = alp::SUCCESS;
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
	alp::Launcher< alp::AUTOMATIC > launcher;
	alp::RC out;
	if( launcher.exec( &alp_program, in, out, true ) != alp::SUCCESS ) {
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
