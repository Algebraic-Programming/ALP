
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


// alp::Grid< 1, 4 > note:
//   - Thread dimensionality = 1, means that the 1D thread grid maps to first
//     axis of the problem grid. A refinement of this API may make this
//     configurable.
template < typename GridType >
void ascend_code( const GridType &grid, RC &rc ) {
	rc = alp::RC::FAILED;

	Tensor Sin( alp::Datatype::FP16, make_axes( "i", "j", "k", "l" ) );
	Tensor Sout( alp::Datatype::FP16, make_axes( "i", "j", "k", "l" ) );

	rc = grid.forEach( make_axes( "i" ), [ & ] () {

		rc = grid.forEach( make_axes( "j" ), [ & ] () {

			auto S_block_in  = getView( Sin );  // T(2,3)
			auto S_block_out = getView( Sout ); // T(2,3)
			Tensor localTensor( alp::Datatype::FP16, make_axes( "k" ) ); // T(2)

			//     T(2)         T(2,3)
			// apply( localTensor, S_block_in, "max", make_axes( "l" ) );
			localTensor( "k" ) = max( S_block_in("k", "l" ), "l" );

			//     T(2,3)       T(2,3)        T(2)
			// apply( S_block_out, S_block_in, localTensor, "minus", make_axes( "l" ) );
			S_block_out( "k", "l" ) = minus( S_block_in("k", "l" ), localTensor( "k" ) , "l" );

			//     T(2,3)
			foldl( S_block_out, "exp" );

			//     T(2)         T(2,3)
			// apply( localTensor, S_block_out, "add", make_axes( "l" ) );
			localTensor( "k" ) = add( S_block_out("k", "l" ), "l" );

			//     T(2,3)       T(2)
			foldl( S_block_out, localTensor, "divide", make_axes( "k" ) );

			//     T(2,3)
			store( S_block_out );

		} );

	} );

	return;
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
	alp::RC error_code = alp::RC::SUCCESS;
	try {
		error_code = alp::compile< 1, 4 >( ascend_code, "softmaxOpv3" );
	} catch( std::exception &e ) {
		std::cerr << "alp::compile threw error: " << e.what() << "\n";
		return 20;
	}
	if( error_code != alp::RC::SUCCESS ) {
		std::cerr << std::flush;
		std::cout << "Codegen FAILED (" << alp::toString( error_code ) << ")"
			<< std::endl;
		return 30;
	} else {
		std::cout << "//Codegen OK" << std::endl;
		return 0;
	}

}

