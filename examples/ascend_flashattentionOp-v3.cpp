
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


// alp::Grid< 1, 5 > note:
//   - Thread dimensionality = 1, means that the 1D thread grid maps to first
//     axis of the problem grid. A refinement of this API may make this
//     configurable.
void ascend_code( const Grid< 1, 5 > &grid, RC &rc ) { // max shape = ( Tr, Br, Tc, Bc, d )
	rc = alp::RC::FAILED;

	// input          // Q and O are 'canonically' aligned.
	Tensor Q( grid, type::FP16, axes( 0, 1, 4 ) ); //  shape = (Tr, Br, d)
	Tensor K( grid, type::FP16, axes( 4, 2, 3 ) ); //  shape = (d, Tc, Bc)  // transposed shape compared to Q
	Tensor V( grid, type::FP16, axes( 4, 2, 3 ) ); //  shape = (d, Tc, Bc)  // transposed shape compared to Q

	// temp
	Tensor m( grid, type::FP16, axes( 0, 1 ) );    //  shape = (Tr, Br) =  (Tr, Br , 1)  = ( Tr, Br, 1, 1, .. )
	                                               //  scalar shape = (1, 1, 1)
	// output
	Tensor l( grid, type::FP16, axes( 0, 1 ) );    //  shape = (Tr, Br)
	Tensor O( grid, type::FP16, axes( 0, 1, 2 ) ); //  shape = (Tr, Br, d)

	set( O, 0 );
	set( l, values::zero );           // values::zero is equivalent to 0
	set( m, values::minus_infinity );

	// forEach cuts the grid into small pieces that are processed concurrently
	rc = grid.forEach( [ &grid, &Q, &K, &V, &l, &m ] () {
		// a view gets the local part to be processed
		// e.g. axes( O_block ) = alp::axes( threadID(), 1, 4 )
		auto O_block = O.getView( grid );
		auto Q_block = Q.getView( grid );
		auto K_block = K.getView( grid );
		auto V_block = V.getView( grid );
		auto l_block = l.getView( grid );
		auto m_block = m.getView( grid );

		// tensor version of Stmp = mxm( Q_block, K_block )
		//  - tensor contraction along one axis
		//  - 2 is the contraction axis
		Tensor Stmp( grid, type::FP16, axes( 0, 2, 3 ) );
		Stmp = Q_block( "i", "j", "k" ) * K_block( "l", "m", "k" );
		// not contracted and non-stored index imply loop, e.g. loop over "j" here


		Tensor tmp( grid, type::FP16, axes( 0, 1 ) );
		set( tmp, m_block );

		// row-wise max
		// do this operation for all l indices
		m_block( "i", "j" ) = max( m_block( "i", "j" ), Stmp( "i", "k", "l" ) , "l");

		// row-wise Stmp -= m_block
		// do this operation for all l indices
		Stmp( "i", "k", "l" ) = minus( Stmp( "i",  "k", "l" ),  m_block( "i", "j" ), "l" );

		// if no axes are specified then apply along all axes
		// This is equivalent to reduction with scalar, just inplace
		// Stmp = exp(Stmp)
		Stmp = exp( Stmp );

		// tmp=exp(tmp-m_block)
		tmp = exp( tmp - m_block );

		// l_block += rowsum(Stmp)
		l_block += sum( Stmp( "i", "j", "k" ), "k" );

		// 'row-wise' O_block *= tmp
		O_block *= tmp;

		// tensor version of O_block = mxm( Stmp,  V_block ), i.e., contraction
		O_block( "i", "j", "k" ) +=  Stmp( "i", "l", "m" ) * V_block( "k", "r", "j" );

		// 'row-wise' O_block *=  1/l_block
		O_block /= l_block;

		// l_block = log(m_block) + m_block
		l_block = log( m_block ) + m_block;

		// store output
		alp::store( O_block );
		alp::store( l_block );
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
		error_code = alp::compile< 1, 5 >( ascend_code );
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

