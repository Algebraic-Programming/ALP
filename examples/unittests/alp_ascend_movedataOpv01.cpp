
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
#define DEBUG

#include <alpAscend.hpp>

using namespace alp;


// alp::Grid< 1, 3 > note:
//   - Thread dimensionality = 1, means that the 1D thread grid maps to first
//     axis of the problem grid. A refinement of this API may make this
//     configurable.
template < typename GridType >
void ascend_code( const GridType &grid, RC &rc ) { // max shape = ( Tr, Br, d )
	rc = RC::FAILED;

	Tensor Sin(Datatype::FP16, make_axes( "i", "j", "k" ) );
	Tensor Sout(Datatype::FP16, make_axes( "i", "j", "k" ) );

	rc = grid.forEach( make_axes( "i" ), [ & ] () {

		auto S_block_in = getView( Sin );
		auto S_block_out = getView( Sout );

		Tensor localTensor(Datatype::FP16, make_axes( "j", "k" ) );

		set( localTensor, S_block_in);
		set( S_block_out, localTensor);

		store( S_block_out );

	} );
}

int main( int argc, char ** argv ) {

	// default options
	bool printUsage = false;

	// input error checking
	if( argc > 1 ) {
		printUsage = true;
	}

	// print help on error
	if( printUsage ) {
		std::cerr << "Usage: " << argv[ 0 ] << "\n";
		return 10;
	}

	// start opgen
	std::cout << "//This is AscendOpGen example " << argv[ 0 ] << "\n";
	RC error_code = RC::SUCCESS;
	try {
		error_code = alp::compile< 1, 3 >( ascend_code, "movedataOpv01" );
	} catch( std::exception &e ) {
		std::cerr << "alp::compile threw error: " << e.what() << "\n";
		return 20;
	}
	if( error_code != RC::SUCCESS ) {
		std::cerr << std::flush;
		std::cout << "Codegen FAILED (" << toString( error_code ) << ")"
			<< std::endl;
		return 30;
	} else {
		std::cout << "//Codegen OK" << std::endl;
		return 0;
	}

}

