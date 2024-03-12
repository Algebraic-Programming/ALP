#include <functional>
#include <limits>
#include <cstddef>

#include <graphblas/ascend/operators.hpp>
#include <graphblas/ascend/tensor.hpp>
#include <graphblas/ascend/stage.hpp>
#include <graphblas/ascend/lazy_evaluation.hpp>
#include <graphblas/ascend/opgen.hpp>

namespace alp
{
	namespace internal
	{
		extern AscendLazyEvaluation ale;
	}
}

namespace alp
{
	Tensor getView( const Tensor &parent ) {

		std::vector< int > forEachAxes = internal::vectorOfVectorsToVector( internal::OpGen::forEachAxes );
		std::vector< int > difference_axes = internal::vectorDifference( parent.getAxes(), forEachAxes );

		Tensor ret_view( parent, difference_axes );

		internal::Rule rule = internal::Rule::NONE;

		alp::internal::ale.addStage( alp::internal::Stagetype::GET_VIEW, rule, parent, difference_axes, forEachAxes );

		return ret_view;
	}

	// TODO extend to multiple containers
	void store( const Tensor &output ) {

		const alp::Tensor &parent = alp::internal::ale.store( output );

		std::vector< int > forEachAxes = internal::vectorOfVectorsToVector( internal::OpGen::forEachAxes );
		std::vector< int > difference_axes = internal::vectorDifference( parent.getAxes(), forEachAxes );

		internal::Rule rule = internal::Rule::NONE;

		alp::internal::ale.addStage( alp::internal::Stagetype::STORE, rule, parent, difference_axes, forEachAxes );
	}

	void set(
		Tensor &tout,
		Tensor &tin,
		const std::vector< int > &activeAxes
	) {
		std::vector< int > forEachAxes = internal::vectorOfVectorsToVector( internal::OpGen::forEachAxes );

		internal::Rule rule = internal::Rule::NONE;

		alp::internal::ale.addStage( alp::internal::Stagetype::SET_TENSOR, rule, tout, tin, activeAxes, forEachAxes );
	}

	void set(
		Tensor &tout,
		double alpha		//TODO perhaps use a templated datatype instead of double
	) {
		std::vector< int > forEachAxes = internal::vectorOfVectorsToVector( internal::OpGen::forEachAxes );

		internal::Rule rule = internal::Rule::NONE;

		alp::internal::ale.addStage( alp::internal::Stagetype::SET_SCALAR, rule, tout, alpha, forEachAxes );
	}

	void apply(
		Tensor &tout,
		Tensor &tin1,
		Tensor &tin2,
		const std::string &opName,
		const std::vector< int > &activeAxes
	) {
		std::vector< int > forEachAxes = internal::vectorOfVectorsToVector( internal::OpGen::forEachAxes );
/*
		std::vector< int > union_axes = internal::vectorUnion( tout.getAxes(), tin1.getAxes() );
		union_axes = internal::vectorUnion( union_axes, tin2.getAxes() );

		assert( union_axes.size() < 3 );

		std::vector< int > temp_axes;
		if( union_axes.size() == 1 ) {
			temp_axes.push_back( union_axes[ 0 ] );
		} else if ( union_axes.size() == 2 ) {
			temp_axes.push_back( union_axes[ 1 ] );
		}

		// create a temporary Tensor
		Tensor temp( temp_axes, tout.getType() );
*/
		internal::Rule rule = internal::Rule::NONE;

		//TODO the current design does not make a distinction between the different cases
		//		of BCAST and REDUCE, this should be fixed in a later version
		if( tin1.getAxes() == tin2.getAxes() && tout.getAxes() == tin1.getAxes() ) {
			rule = internal::Rule::EWISE;
		} else if ( tin1.getAxes() == tin2.getAxes() && internal::vectorSubset( tout.getAxes(), tin1.getAxes() ) == true ) {
			rule = internal::Rule::REDUCE;
		} else if ( tin1.getAxes() == tin2.getAxes() && internal::vectorSubset( tin1.getAxes(), tout.getAxes() ) == true ) {
			rule = internal::Rule::BCAST;
		} else if ( tin1.getAxes() == tout.getAxes() && internal::vectorSubset( tout.getAxes(), tin2.getAxes() ) == true ) {
			rule = internal::Rule::REDUCE;
		} else if ( tin1.getAxes() == tout.getAxes() && internal::vectorSubset( tin2.getAxes(), tout.getAxes() ) == true ) {
			rule = internal::Rule::BCAST;
		} else if ( tin2.getAxes() == tout.getAxes() && internal::vectorSubset( tout.getAxes(), tin1.getAxes() ) == true ) {
			rule = internal::Rule::REDUCE;
		} else if ( tin2.getAxes() == tout.getAxes() && internal::vectorSubset( tin1.getAxes(), tout.getAxes() ) == true ) {
			rule = internal::Rule::BCAST;
		} else if ( tin1.getAxes() != tin2.getAxes() && tin1.getAxes() != tout.getAxes() && tin2.getAxes() != tout.getAxes() ) {
			if( internal::vectorSubset( tout.getAxes(), tin1.getAxes() ) == true && internal::vectorSubset( tout.getAxes(), tin2.getAxes() ) == true ) {
				rule = internal::Rule::BCAST;
			} else if( internal::vectorSubset( tin1.getAxes(), tout.getAxes() ) == true && internal::vectorSubset( tin2.getAxes(), tout.getAxes() ) == true ) {
				rule = internal::Rule::REDUCE;
			} else {
		        std::cerr << "The axes of the output tensor cannot be subset of the axes of one input and superset of the axes of the other input: apply " << opName << std::endl;
		        std::abort();
			}
		}

		if( opName == "minus" ) {
			alp::internal::ale.addStage( alp::internal::Stagetype::APPLY_MINUS, rule, tout, tin1, tin2, activeAxes, forEachAxes );
		} else if( opName == "add" ) {
			alp::internal::ale.addStage( alp::internal::Stagetype::APPLY_ADD, rule, tout, tin1, tin2, activeAxes, forEachAxes );
		}
		else {

		}
	}

	void foldl(
		Tensor &tinout,
		Tensor &tin,
		const std::string &opName,
		const std::vector< int > &activeAxes
	) {
		std::vector< int > forEachAxes = internal::vectorOfVectorsToVector( internal::OpGen::forEachAxes );
/*
		std::vector< int > union_axes = internal::vectorUnion( tinout.getAxes(), tin.getAxes() );

		assert( union_axes.size() < 3 );

		std::vector< int > temp_axes;
		if( union_axes.size() == 1 ) {
			temp_axes.push_back( union_axes[ 0 ] );
		} else if ( union_axes.size() == 2 ) {
			temp_axes.push_back( union_axes[ 1 ] );
		}

		// create a temporary Tensor
		Tensor temp( temp_axes, tinout.getType() );
*/
		internal::Rule rule = internal::Rule::NONE;

		if( tinout.getAxes() == tin.getAxes() ) {
			rule = internal::Rule::EWISE;
		} else if ( internal::vectorSubset( tinout.getAxes(), tin.getAxes() ) == true ) {
			rule = internal::Rule::REDUCE;
		} else if ( internal::vectorSubset( tin.getAxes(), tinout.getAxes() ) == true ) {
			rule = internal::Rule::BCAST;
		} else {

		}

		if( opName == "divide" ) {
			alp::internal::ale.addStage( alp::internal::Stagetype::FOLDL_DIVIDE, rule, tinout, tin, activeAxes, forEachAxes );
		} else if( opName == "max" ) {
			alp::internal::ale.addStage( alp::internal::Stagetype::FOLDL_MAX, rule, tinout, tin, activeAxes, forEachAxes );
		} else if( opName == "times" ) {
			alp::internal::ale.addStage( alp::internal::Stagetype::FOLDL_TIMES, rule, tinout, tin, activeAxes, forEachAxes );
		} else if( opName == "add" ) {
			alp::internal::ale.addStage( alp::internal::Stagetype::FOLDL_ADD, rule, tinout, tin, activeAxes, forEachAxes );
		} else {

		}
	}

//	template< size_t sm, size_t pm >
	void foldl(
//		const Grid< sm, pm > &grid,
		Tensor &tinout,
		const std::string &opName,
		const std::vector< int > &activeAxes
	) {
		std::vector< int > forEachAxes = internal::vectorOfVectorsToVector( internal::OpGen::forEachAxes );

		internal::Rule rule = internal::Rule::NONE;

		if( opName == "exp" ) {
			alp::internal::ale.addStage( alp::internal::Stagetype::FOLDL_EXP, rule, tinout, activeAxes, forEachAxes );
		} else {

		}
	}

}

