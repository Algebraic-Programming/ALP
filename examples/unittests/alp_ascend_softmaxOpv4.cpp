
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


// alp::Grid< 1, 6 > note:
//   - Thread dimensionality = 1, means that the 1D thread grid maps to first
//     axis of the problem grid. A refinement of this API may make this
//     configurable.
template < typename GridType >
void ascend_code( const GridType &grid, RC &rc ) {
	rc = RC::FAILED;

	Tensor Sin(Datatype::FP16, make_axes( "i", "j", "k", "l", "m", "n" ) );
	Tensor Sout(Datatype::FP16, make_axes( "i", "j", "k", "l", "m", "n" ) );

	rc = grid.forEach( make_axes( "i" ), [ & ] () {

		rc = grid.forEach( make_axes( "j" ), [ & ] () {

			rc = grid.forEach( make_axes( "l" ), [ & ] () {

				rc = grid.forEach( make_axes( "m" ), [ & ] () {

					auto S_block_in  = getView( Sin );  // T(2,5)
					auto S_block_out = getView( Sout ); // T(2,5)
					Tensor localTensor(Datatype::FP16, make_axes( "k" ) ); // T(2)

					//     T(2)          T(2,5)
					// apply( localTensor, S_block_in, "max", make_axes( "n" ) );
					localTensor( "k" ) = max( S_block_in("k", "n" ), "n" );

					//     T(2,5)         T(2,5)        T(2)
					// apply( S_block_out, S_block_in, localTensor, "minus", make_axes( "n" ) );
					S_block_out( "k", "n" ) = minus( S_block_in("k", "n" ), localTensor( "k" ) , "n" );

					//     T(2,5)
					foldl( S_block_out, "exp" );

					//     T(2)          T(2,5)
					// apply( localTensor, S_block_out, "add", make_axes( "n" ) );
					localTensor( "k" ) = add( S_block_out("k", "n" ), "n" );

					//     T(2,5)         T(2)
					foldl( S_block_out, localTensor, "divide", make_axes( "n" ) );

					//     T(2,5)
					store( S_block_out );

				} );
			} );
		} );
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
		error_code = alp::compile< 1, 6 >( ascend_code, "softmaxOpv4" );
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

