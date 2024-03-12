
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

#include <alpAscend.hpp>

using namespace alp;

template < typename GridType >
void ascend_code( const GridType &grid, RC&rc ) {
	rc = RC::FAILED;

	Tensor x_global(Datatype::FP16, make_axes( "i" ) ); // 0 is default
	Tensor y_global(Datatype::FP16, make_axes( "i" ) );
	Tensor z_global(Datatype::FP16, make_axes( "i" ) );

	rc = grid.forEach( make_axes( "i" ), [ & ] () {
		auto x_block = getView( x_global );
		auto y_block = getView( y_global );
		auto z_block = getView( z_global );

		apply( z_block, x_block, y_block, "add" ); // z = x + y
//		z_block( "j" ) = add( x_block( "j" ), y_block( "j" ), "j" ); // z = x + y

		store( z_block );
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
	RC error_code = RC::SUCCESS;
	try {
		error_code = alp::compile< 1, 1 >( ascend_code, "addOpv1" );
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

