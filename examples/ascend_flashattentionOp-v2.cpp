
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


// alp::Grid< 1, 3 > note:
//   - Thread dimensionality = 1, means that the 1D thread grid maps to first
//     axis of the problem grid. A refinement of this API may make this
//     configurable.
void ascend_code( const Grid< 1, 3 > &grid, RC &rc ) { // max shape = (  m,  Tr,  N )
	rc = alp::RC::FAILED;

	// input          // Q and O are 'canonically' aligned.
	Tensor Q( grid, type::FP16, axes( 0, 1, 2 ) ); //  shape = (m, Tr, N)
	Tensor K( grid, type::FP16, axes( 2, 0, 1 ) ); //  shape = (N, m, Tr)  // transposed shape compared to Q
	Tensor V( grid, type::FP16, axes( 2, 0, 1 ) ); //  shape = (N, m, Tr)  // transposed shape compared to Q

	// temp
	Tensor m( grid, type::FP16, axes( 0, 1 ) );    //  shape = (m, Tr) =  (m, Tr , 1)  = ( m, Tr, 1, 1, .. )
	                                               //  scalar shape = (1, 1, 1)
	// output
	Tensor l( grid, type::FP16, axes( 0, 1 ) );    //  shape = (m, Tr)
	Tensor O( grid, type::FP16, axes( 0, 1, 2 ) ); //  shape = (m, Tr, N)

	set( O, 0 );
	set( l, values::zero );           // values::zero is equivalent to 0
	set( m, values::minus_infinity );
	
	// forEach cuts the grid into small pieces that are processed concurrently
	rc = grid.forEach( [ &grid, &Q, &K, &V, &l, &m ] () {
		// a view gets the local part to be processed
		// e.g. axes( O_block ) = alp::axes( threadID(), 1, 2 )
		auto O_block = O.getView( grid );

		auto Q_block = Q.getView( grid );

		// if tensors are permuted, the "cut" dimension still refers to that defined
		// by the grid. E.g.  axes( K_block ) = alp::axes( 2, threadID(), 1 )
		auto K_block = K.getView( grid );
		auto V_block = V.getView( grid );
		auto l_block = l.getView( grid );
		auto m_block = m.getView( grid );

		// tensor version of Stmp = mxm( Q_block, K_block )
		//  - tensor contraction along one axis
		//  - 2 is the contraction axis
		Tensor Stmp( grid, type::FP16, axes( 0, 1, 1 ) ); // AJ2D: I think this should have been 1, 1? Or the below mxm ex. is wrong?
		Stmp = Q_block( "i", "m", "k" ) * K_block( "k", "j", "m" ); // AJ2D: is this correct in Einstein notation?
		                                                            //       It seems to me to match the below code
									    //       (although I don't get foldl with a semiring)

		// tensor contraction in one axis:
		// alp::semiring multiplication and accumualtion operators
		// e.g. Stmp[ : , : ] = mxm( Q_block[ threadID(), :, : ], K_block[ :, threadID(), : ] )
		// set( Stmp, values::zero );
		// alp::foldl( Stmp,  Q_block, K_block, alp::semiring(), alp::axes( 2 ) );
		// NOTE:  in general multiple axes needed with proper reduction rules:
		// here, Dim(Stmp) + 2*Dim(axes) = Dim(Q_block) + Dim(Q_block)


		Tensor tmp( grid, type::FP16, axes( 1 ) );
		set( tmp, m_block ); // AJ2D: here tmp is one-dimensional but m_block is two-dimensional?
		                     //       I think this means the parallelised dimension has only one fiber,
				     //       not a block of fibers, perhaps? That could work (though the codegen
				     //       would have to coalesce them back). I had assumed we got back a block
				     //       of some size close to n/p. If we have a block then the following
				     //       seems correct and perhaps more clear?
		// Tensor tmp( grid, type::FP16, axes( 0, 1 ) )
		// set( tmp, m_block )
		
		// two was the "contraction" axis, e.g. row-wise reduction
		max( m_block, Stmp ); // AJ2D: I think here the axes become confusing. If the axes of Stmp are correct
		                      //       (I modified it), then the "axes(2)" which used to be here do not match
				      //       any axes in m_bock and Stmp. Translating it into matrix land, Stmp is
				      //       n x n while m_block is m x n. If m = 1 (see above comment block), then
				      //       indeed what max is a reduction, but it remains ambiguous over what
				      //       dimension the reduction should go (rows or columns -- both are the same
				      //       mode). If m > 1, then the semantics I suppose are to broadcast the
				      //       result of max( Stmp ) into m_block?
				      //
				      //       Would the following perhaps be clearer?
		// tmp = max( Stmp( "i", "j" ), "j" );
		// m_block( "i", "j" ) = tmp( "j" ); // broadcast tmp to m_block

		// AJ2D: in the below, I will just assume Einstein notation while simplifying the code

		// 'row-wise' Stmp -= m_block
		Stmp( "i", "j" ) -= m_block( "j" );

		// if no axes are specified apply along all axes
		// This is equivalent to reduction with scalar, just inplace
		// Stmp = exp(Stmp)
		Stmp = exp( Stmp );

		// tmp=exp(tmp-m_block)
		tmp = exp( tmp - m_block );
		
		// l_block += rowsum(Stmp)
		l_block += sum( Stmp( "i", "j" ), "j" );

		// 'row-wise' O_block *= tmp
		O_block *= tmp;

		// tensor version of O_block = mxm( Stmp,  V_block ), i.e., contraction
		Oblock( "i", "j", "k" ) +=  Stmp( "i", "r" ) * V_block( "k", "r", "j" );

		// 'row-wise' O_block *=  1/l_block
		O_block /= l_block;
		// or div( O_block, l_block );

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
		error_code = alp::compile< 1, 1 >( ascend_code );
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

