
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

#include <alp.hpp>

using namespace alp;

template< typename MatrixType >
void ask_questions( const MatrixType & M, std::string name ) {

	using M_type = typename std::remove_const< typename std::remove_reference< decltype( M ) >::type >::type;

	std::cout << name << "( " << alp::nrows( M ) << ", " << alp::ncols( M ) << " )" << std::endl;
	std::cout << "Is " << name << ":" << std::endl;
	std::cout << "\tan ALP Matrix? " << alp::is_matrix< M_type >::value << std::endl;
	std::cout << "\tan ALP Vector? " << alp::is_vector< M_type >::value << std::endl;
	std::cout << "\ta structured Matrix? " << alp::is_structured_matrix< M_type >::value << std::endl;
	std::cout << "\ta container-based Matrix? " << alp::is_concrete< M_type >::value << std::endl;
	std::cout << "\ta functor-based Matrix? " << !alp::is_concrete< M_type >::value << std::endl;
	std::cout << "\tan original Matrix? " << alp::is_original< M_type >::value << std::endl;
	std::cout << "\ta view over another Matrix? " << !alp::is_original< M_type >::value << std::endl;
	//std::cout << name << " has the following static properties:" << std::endl;
	//std::cout << "\tstructure: " << typeid(typename alp::inspect_structure< M_type >::type).name() << std::endl;
	//std::cout << "\tview type: " << typeid(typename alp::inspect_view< M_type >::type).name() << std::endl;
	//std::cout << "\tApplied to: " << typeid(typename alp::inspect_view< M_type >::type::applied_to).name() << std::endl;
}


void alp_program( const size_t & n, alp::RC & rc ) {

	alp::Matrix< float, alp::structures::General > M( n, n );
	alp::Matrix< float, alp::structures::Square > A( n );
	auto At = alp::get_view< alp::view::transpose >( A );
	auto Mt = alp::get_view< alp::view::transpose >( M );
	auto Mview = alp::get_view( M, alp::utils::range(0,4), alp::utils::range(0,4) );
	auto Sq_Mref = alp::get_view< alp::structures::Square > ( M );

	ask_questions( M, "M" );
	ask_questions( A, "A" );

	ask_questions( At, "At" );
	ask_questions( Mt, "Mt" );
	ask_questions( Mview, "Mview" );
	ask_questions( Sq_Mref, "Sq_Mref" );

	auto v_diag = alp::get_view< alp::view::diagonal >( M );
	auto v_view1 = alp::get_view( v_diag );
	auto v_view2 = alp::get_view( v_diag, alp::utils::range(1,2) );
	std::cout << "v_diag( " << alp::getLength( v_diag ) << " )" << std::endl;
	std::cout << "v_view1( " << alp::getLength( v_view1 ) << " )" << std::endl;
	std::cout << "v_view2( " << alp::getLength( v_view2 ) << " )" << std::endl;

	ask_questions( v_diag, "v_diag" );
	ask_questions( v_view1, "v_view1" );
	ask_questions( v_view2, "v_view2" );

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

