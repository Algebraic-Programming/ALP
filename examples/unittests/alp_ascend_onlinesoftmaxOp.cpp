
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

	// max shape = ( Tr,Tc,Br,Bc )
	// Tr = number for row-blocks, Br = row-length of rowblocks;  Tr*Tc = N
	// Tc = number for column-blocks, Bc = column-length of rowblocks;  Tr*Tc = M
	// for softmax N == M, i.e. Sin and Sout are square matrices
	rc = alp::RC::FAILED;

	Tensor mtensorout( alp::Datatype::FP16, make_axes( 0, 2 ) );	//  shape = ( Tr,Br )
	Tensor ltensorout( alp::Datatype::FP16, make_axes( 0, 2 ) );	//  shape = ( Tr,Br )clear
	Tensor Sin( alp::Datatype::FP16, make_axes( 0, 1, 2, 3 ) );	//  shape = ( Tr,Tc,Br,Bc )
	Tensor Sout( alp::Datatype::FP16, make_axes( 0, 1, 2, 3 ) );	//  shape = ( Tr,Tc,Br,Bc )

	rc = grid.forEach( make_axes( 0 ), [ & ] () {

		auto m_block_out = getView( mtensorout );
		auto l_block_out = getView( ltensorout );

//-->
		set( m_block_out, -alp::Infinity< double > );	// TODO the double should re replaced by alp::Datatype::FP16
//-->
		set( l_block_out, alp::Zero< double > ); //TODO

		grid.forEach(
			make_axes( 1 ),		// prallel loop- > for(i0=0; i0<n0; ++i0 ) { ...
			[ & ] () {

				// these tensors will have original axes with axes 0 and 2 removed
				// S_block_in=S[i0,i1,:,:]
				auto S_block_in = getView( Sin );
				auto S_block_out = getView( Sout );

				Tensor rowmaxS( alp::Datatype::FP16, make_axes( 2 ) );
				Tensor mi_old( alp::Datatype::FP16, make_axes( 2 ) );
				Tensor expmidiff( alp::Datatype::FP16, make_axes( 2 ) );

				// mi_old=cp.copy(mtensor[i,:])
				//   T(2)    T(2)

				set( mi_old, m_block_out); //TODO: use VectorSet(vec,vec,n2);

				// rowmaxS=np.max(Si,axis=-1)
				//      T(2)    T(2,3)
				// foldl( rowmaxS, S_block_in, "max", make_axes( 3 ) );
				rowmaxS( 2 ) = max( S_block_in( 2, 3 ), 3 );

				// mtensor[i,:]=np.maximum(mtensor[i,:],rowmaxS)
				//     T(2)         T(2)

				foldl( m_block_out, rowmaxS, "max" ); //TODO use VectorEwiseMinus

				// Si=Si-np.expand_dims(mtensor[i,:], axis=-1)

				//     T(2,3)      T(2)

				// apply( S_block_out, S_block_in, m_block_out, "minus", make_axes( 3 ) );
				S_block_out( 2, 3 ) = minus( S_block_in( 2, 3 ), m_block_out( 2 ), 3 );

				// Si=np.exp(Si)
				foldl( S_block_out, "exp" );

				// expmidiff=np.exp(mi_old-mtensor[i,:])
				//     T(2)       T(2)    T(2)
				// apply( expmidiff, mi_old, m_block_out, "minus" );
				expmidiff( 2 ) = minus( mi_old( 2 ), m_block_out( 2 ), 2 );

				foldl( expmidiff, "exp" );

				// ltensor[i,:]*=expmidiff
				//     T(2)         T(2)

				foldl( l_block_out, expmidiff, "times" ); //TODO use VectorEwiseMultiply()

				// ltensor[i,:]+= np.sum(Si,axis=-1)
				//     T(2)         T(2,3)

				// foldl( l_block_out, S_block_out, "add", make_axes( 3 ) );
				foldl( rowmaxS, S_block_out, "add", make_axes( 3 ) );
				foldl( l_block_out, rowmaxS, "add", make_axes( 3 ) );

				//TODO use
				// BlockReduceSum(work0, S_block_out, workhidden, .. );
				// VectorEwiseSum(l_block_out,l_block_out,work0, ..);

				// Otensor[i,:,:,:]*=np.expand_dims(expmidiff, axis=(-2,-1))
				// foldl( S_block_out, expmidiff, "times", make_axes( 3 ) );

				store( S_block_out );
		} );

		// Otensor[i,:,j,:]=Si
		// already output

		// Otensor[i,:,:,:]/=np.expand_dims(ltensor[i,:], axis=(-2,-1))
		// foldl( S_block_out, l_block_out, "divide", make_axes( 3 ) );

		store( l_block_out );
		store( m_block_out );

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
		error_code = alp::compile< 1, 4 >( ascend_code, "onlinesoftmaxOp" );
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

