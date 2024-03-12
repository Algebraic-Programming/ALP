
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

// alp::Grid< 1, 5 > note:
//   - Thread dimensionality = 1, means that the 1D thread grid maps to first
//     axis of the problem grid. A refinement of this API may make this
//     configurable.
template < typename GridType >
void ascend_code( const GridType &grid, alp::RC &rc ) {

	// shape = ( Tr,Tc,Br,Bc,d )
	// Tr = number for row-blocks, Br = row-length of rowblocks;  Tr*Tc = N
	// Tc = number for column-blocks, Bc = column-length of rowblocks;  Tr*Tc = M
	// for softmax N == M, i.e. Sin and Sout are square matrices
	rc = alp::RC::SUCCESS;

	// input
	alp::Tensor Qtensorin( alp::Datatype::FP16, alp::make_axes( 0, 2, 4 ) );  //  shape = ( Tr,Br,d )
	alp::Tensor Ktensorin( alp::Datatype::FP16, alp::make_axes( 1, 3, 4 ) );  //  shape = ( Tc,Bc,d )
	alp::Tensor Vtensorin( alp::Datatype::FP16, alp::make_axes( 1, 3, 4 ) );  //  shape = ( Tc,Bc,d )

	// temp
	alp::Tensor Otensorout( alp::Datatype::FP16, alp::make_axes( 0, 2, 4 ) ); //  shape = ( Tr,Br,d )
	alp::Tensor mtensorout( alp::Datatype::FP16, alp::make_axes( 0, 2 ) );    //  shape = ( Tr,Br )
	alp::Tensor ltensorout( alp::Datatype::FP16, alp::make_axes( 0, 2 ) );    //  shape = ( Tr,Br )

	rc = !rc ? rc : grid.forEach( alp::make_axes( 0 ), [ & ] () {

		auto Q_block_in  = Qtensorin.getView();  // T(2,4)

		auto O_block_out = Otensorout.getView(); // T(2,4)
		auto m_block_out = mtensorout.getView(); // T(2)
		auto l_block_out = ltensorout.getView(); // T(2)

		 //TODO: fix, i.e. double replace with  half
		alp::set( m_block_out, -alp::Infinity<double> );
		alp::set( l_block_out, alp::Zero<double>  );

		rc = !rc ? rc : grid.forEach( alp::make_axes( 1 ), [ & ] () {

				// these tensors will have original axes with axes 0 and 1 removed
				// Sij=S[i0,i1,:,:]

				auto K_block_in = Ktensorin.getView(); // T(3,4)
				auto V_block_in = Vtensorin.getView(); // T(3,4)

				alp::Tensor Sij(       alp::Datatype::FP16, alp::make_axes( 2, 3 ) );
				alp::Tensor Temp(      alp::Datatype::FP16, alp::make_axes( 2, 3 ) );
				alp::Tensor rowmaxS(   alp::Datatype::FP16, alp::make_axes( 2 ) );
				alp::Tensor mi_old(    alp::Datatype::FP16, alp::make_axes( 2 ) );
				alp::Tensor expmidiff( alp::Datatype::FP16, alp::make_axes( 2 ) );

				//          T(2,3)  T(2,4)     T(3,4)
				alp::apply( Sij, Q_block_in, K_block_in, "mxm", alp::make_axes( 4 ) );

				// mi_old=cp.copy(mtensor[i,:])
				//        T(2)    T(2)
				alp::set( mi_old, m_block_out);

				// rowmaxS=np.max(Si,axis=-1)
				//          T(2)    T(2,3)
				alp::apply( rowmaxS, Sij, "max", alp::make_axes( 3 ) );

				// mtensor[i,:]=np.maximum(mtensor[i,:],rowmaxS)
				//          T(2)         T(2)
				alp::foldl( m_block_out, rowmaxS, "max" );

				// Si=Si-np.expand_dims(mtensor[i,:], axis=-1)
				//          T(2,3)      T(2)
				alp::foldl( Sij, m_block_out, "minus", alp::make_axes( 3 ) );

				// Si=np.exp(Si)
				alp::foldl( Sij, "exp" );

				// expmidiff=np.exp(mi_old-mtensor[i,:])
				//          T(2)       T(2)    T(2)
				alp::apply( expmidiff, mi_old, m_block_out, "minus" );

				alp::foldl( expmidiff, "exp" );

				// ltensor[i,:]*=expmidiff
				//          T(2)         T(2)
				alp::foldl( l_block_out, expmidiff, "times" );

				// ltensor[i,:]+= np.sum(Si,axis=-1)
				//          T(2)         T(2,3)
				alp::foldl( l_block_out, Sij, "add", alp::make_axes( 3 ) );

				// Otensor[i,:,:]*=np.expand_dims(expmidiff, axis=(-2,-1))
				//          T(2,4)       T(2)
				alp::foldl( O_block_out, expmidiff, "times", alp::make_axes( 4 ) );

				//          T(2,3)      T(2,4)   T(3,4)
				alp::apply( Temp, Sij, V_block_in,  "mxm", alp::make_axes( 4 ) );
				//          T(2,3)       T(2,3)
				alp::foldl( O_block_out, Temp , "add" );

		} );

		// Otensor[i,:,:]/=np.expand_dims(ltensor[i,:], axis=(-2,-1))
		//     T(2,3)       T(2)
		alp::foldl( O_block_out, l_block_out, "divide", alp::make_axes( 3 ) );

		//ltensor[i,:] = mtensor[i,:] + log(ltensor[i,:])
		// skip for now

		alp::store( O_block_out );
		alp::store( l_block_out );
		alp::store( m_block_out );

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
		error_code = alp::compile< 1, 5 >( ascend_code, "KernelFlashattention" );
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

