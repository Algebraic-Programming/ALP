#include <graphblas/ascend/stage.hpp>
#include <graphblas/ascend/symbolTable.hpp>
#include <graphblas/ascend/semantics.hpp>
#include <graphblas/ascend/grid.hpp>

namespace alp
{
	namespace internal
	{
		extern iGrid *igrid;
		extern SymbolTable symbols;
	}
}

//TODO double should be replaced by alp::Scalar
alp::internal::Stage::Stage( const AscendPipeline &parent,
							 Stagetype _enum_op_type, Rule _rule,
							 const alp::Tensor &_tensor0,
							const double _alpha,
							 const std::vector< int > &_forEachAxes )
		: pipeline( parent ),
		  enum_op_type( _enum_op_type ),
		  rule( _rule ),
		  tensor0( _tensor0 ),
		  alpha( _alpha ),
		  forEachAxes( _forEachAxes )
{
	semanticsCheks();
	computeMemoryOffsets();
}

alp::internal::Stage::Stage( const AscendPipeline &parent,
							 Stagetype _enum_op_type, Rule _rule,
							 const alp::Tensor &_tensor0,
							 const std::vector< int > &_activeAxes,
							 const std::vector< int > &_forEachAxes )
		: pipeline( parent ),
		  enum_op_type( _enum_op_type ),
		  rule( _rule ),
		  tensor0( _tensor0 ),
		  activeAxes( _activeAxes ),
		  forEachAxes( _forEachAxes )
{
	semanticsCheks();
	computeMemoryOffsets();
}

alp::internal::Stage::Stage( const AscendPipeline &parent,
							 Stagetype _enum_op_type, Rule _rule,
							 const alp::Tensor &_tensor0,
							 const alp::Tensor &_tensor1,
							 const std::vector< int > &_activeAxes,
							 const std::vector< int > &_forEachAxes )
		: pipeline( parent ),
		  enum_op_type( _enum_op_type ),
		  rule( _rule ),
		  tensor0( _tensor0 ),
		  tensor1( _tensor1 ),
		  activeAxes( _activeAxes ),
		  forEachAxes( _forEachAxes )
{
	semanticsCheks();
	computeMemoryOffsets();
}

alp::internal::Stage::Stage( const AscendPipeline &parent,
							 Stagetype _enum_op_type, Rule _rule,
							 const alp::Tensor &_tensor0,
							 const alp::Tensor &_tensor1,
							 const alp::Tensor &_tensor2,
							 const std::vector< int > &_activeAxes,
							 const std::vector< int > &_forEachAxes )
		: pipeline( parent ),
		  enum_op_type( _enum_op_type ),
		  rule( _rule ),
		  tensor0( _tensor0 ),
		  tensor1( _tensor1 ),
		  tensor2( _tensor2 ),
		  activeAxes( _activeAxes ),
		  forEachAxes( _forEachAxes )
{
	semanticsCheks();
	computeMemoryOffsets();
}
/*
alp::internal::Stage::Stage( const AscendPipeline &parent,
							 Stagetype _enum_op_type, Rule _rule,
							 const alp::Tensor &_tensor0,
							 const alp::Tensor &_tensor1,
							 const alp::Tensor &_tensor2,
							 const alp::Tensor &_tensor3,
							 const std::vector< int > &_activeAxes,
							 const std::vector< int > &_forEachAxes )
		: pipeline( parent ),
		  enum_op_type( _enum_op_type ),
		  rule( _rule ),
		  tensor0( _tensor0 ),
		  tensor1( _tensor1 ),
		  tensor2( _tensor2 ),
		  tensor3( _tensor3 ),
		  activeAxes( _activeAxes ),
		  forEachAxes( _forEachAxes )
{
	semanticsCheks();
	computeMemoryOffsets();
}
*/
alp::internal::Stagetype alp::internal::Stage::getOpType() const {

	return enum_op_type;
}

alp::internal::Rule alp::internal::Stage::getRule() const {

	return rule;
}

const alp::Tensor & alp::internal::Stage::getTensor0() const {

	return tensor0;
}

const std::vector< int > & alp::internal::Stage::getAxes() const {

	return activeAxes;
}

const std::vector< int >& alp::internal::Stage::getForEachAxes() const {

	return forEachAxes;
}

std::string alp::internal::Stage::getOp( const std::string &tabs ) const {

	switch (enum_op_type) {
		case alp::internal::Stagetype::APPLY_MINUS:
			return generateApplyMinusOp( tabs );
		case alp::internal::Stagetype::APPLY_ADD:
			return generateApplyAddOp( tabs );
		case alp::internal::Stagetype::FOLDL_DIVIDE:
			return generateFoldlDivideOp( tabs );
		case alp::internal::Stagetype::FOLDL_MAX:
			return generateFoldlMaxOp( tabs );
		case alp::internal::Stagetype::FOLDL_TIMES:
			return generateFoldlTimesOp( tabs );
		case alp::internal::Stagetype::FOLDL_ADD:
			return generateFoldlAddOp( tabs );
		case alp::internal::Stagetype::FOLDL_EXP:
			return generateFoldlExpOp( tabs );
		case alp::internal::Stagetype::SET_TENSOR:
			return generateSetTensorOp( tabs );
		case alp::internal::Stagetype::SET_SCALAR:
			return generateSetScalarOp( tabs );
		case alp::internal::Stagetype::GET_VIEW:
			return generateGetViewOp( tabs );
		case alp::internal::Stagetype::STORE:
			return generateStoreOp( tabs );
		case alp::internal::Stagetype::IMPLICIT_FREE:
			return generateImplicitFreeOp( tabs );
		default:
			return generateToDoOp( tabs );
	}
}

std::string alp::internal::Stage::generateApplyMinusOp( const std::string &tabs ) const {

	const std::string arg1 = tensor0.getAccessedElement( pipeline.getID() );
	const std::string arg2 = tensor1.getAccessedElement( pipeline.getID() );
	const std::string arg3 = tensor2.getAccessedElement( pipeline.getID() );
//	const std::string arg4 = tensor3.getAccessedElement( pipeline.getID() );

	const std::vector< int > op_axes = computeOperatorAxes();
	std::stringstream stage;

	switch ( rule ) {
		case Rule::EWISE:
		{
			if( op_axes.size() == 1) {
				stage << tabs << "\t\t\talp::VectorEwiseMinus( " << arg1 << ", " << arg2 << ", "
					  << arg3 << ", " << igrid->problemSize( op_axes[ 0 ] ) << " );\n";
			} else if( op_axes.size() == 2) {
				stage << tabs << "\t\t\talp::BlockEwiseMinus( " << arg1 << ", " << arg2 << ", "
					  << arg3 << ", " << igrid->problemSize( op_axes[ 0 ] ) << ", "
					  << igrid->problemSize( op_axes[ 1 ] ) << " );\n";
			}
			break;
		}
		case Rule::BCAST:
		{
			if( op_axes.size() == 1) {
				stage << tabs << "\t\t\talp::VectorBcastMinus( " << arg1 << ", " << arg2 << ", " << arg3 << ", "
					  << alp::internal::symbols.getLocalTempTensorBuffer( tensor0.getType() ) << "[ 0 ], "
					  << pipeline.getTilingAxes() << igrid->problemSize( op_axes[ 0 ] ) << " );\n";
			} else if( op_axes.size() == 2) {
				stage << tabs << "\t\t\talp::BlockBcastMinus( " << arg1 << ", " << arg2 << ", " << arg3 << ", "
					  << alp::internal::symbols.getLocalTempTensorBuffer( tensor0.getType() ) << "[ 0 ], "
					  << pipeline.getTilingAxes() << igrid->problemSize( op_axes[ 0 ] ) << ", "
					  << igrid->problemSize( op_axes[ 1 ] ) << " );\n";
			}
			break;
		}
		case Rule::REDUCE:
		{
			break;
		}
		default:
		{
		        std::cerr << "Invalid rule: apply minus" << std::endl;
		        std::abort();
		}
	}

	return stage.str();
}

std::string alp::internal::Stage::generateApplyAddOp( const std::string &tabs ) const {

	const std::string arg1 = tensor0.getAccessedElement( pipeline.getID() );
	const std::string arg2 = tensor1.getAccessedElement( pipeline.getID() );
	const std::string arg3 = tensor2.getAccessedElement( pipeline.getID() );

	const std::vector< int > op_axes = computeOperatorAxes();
	std::stringstream stage;

	switch ( rule ) {
		case Rule::EWISE:
		{
			if( op_axes.size() == 0) {
				stage << tabs << "\t\t\tAdd( " << arg1 << ", " << arg2 << ", " << arg3 << ", "
					  << pipeline.getTilingAxes() << "1" << " );\n";
			} else if( op_axes.size() == 1) {
				stage << tabs << "\t\t\tAdd( " << arg1 << ", " << arg2 << ", " << arg3 << ", "
					  << pipeline.getTilingAxes() << igrid->problemSize( op_axes[ 0 ] ) << " );\n";
			} else if( op_axes.size() == 2) {
				stage << tabs << "\t\t\tAdd( " << arg1 << ", " << arg2 << ", " << arg3 << ", "
					  << igrid->problemSize( op_axes[ 0 ] ) << " * "
					  << igrid->problemSize( op_axes[ 1 ] ) << " );\n";
			}
			break;
		}
		case Rule::BCAST:
		{
			break;
		}
		case Rule::REDUCE:
		{
			break;
		}
		default:
		{
		        std::cerr << "Invalid rule: apply add" << std::endl;
		        std::abort();
		}
	}

	return stage.str();
}

std::string alp::internal::Stage::generateFoldlDivideOp( const std::string &tabs ) const {

	const std::string arg1 = tensor0.getAccessedElement( pipeline.getID() );
	const std::string arg2 = tensor1.getAccessedElement( pipeline.getID() );
//	const std::string arg3 = tensor2.getAccessedElement( pipeline.getID() );

	const std::vector< int > op_axes = computeOperatorAxes();
	std::stringstream stage;

	switch ( rule ) {
		case Rule::EWISE:
		{
			break;
		}
		case Rule::BCAST:
		{
			if( op_axes.size() == 1) {
				stage << tabs << "\t\t\talp::VectorBcastDivide( " << arg1 << ", " << arg1 << ", " << arg2 << ", "
					  << alp::internal::symbols.getLocalTempTensorBuffer( tensor0.getType() ) << "[ 0 ], "
					  << pipeline.getTilingAxes() << igrid->problemSize( op_axes[ 0 ] ) << " );\n";
			} else if( op_axes.size() == 2) {
				stage << tabs << "\t\t\talp::BlockBcastDivide( " << arg1 << ", " << arg1 << ", " << arg2 << ", "
					  << alp::internal::symbols.getLocalTempTensorBuffer( tensor0.getType() ) << "[ 0 ], "
					  << pipeline.getTilingAxes() << igrid->problemSize( op_axes[ 0 ] ) << ", "
					  << igrid->problemSize( op_axes[ 1 ] ) << " );\n";
			}
			break;
		}
		case Rule::REDUCE:
		{
			break;
		}
		default:
		{
		        std::cerr << "Invalid rule: foldl divide" << std::endl;
		        std::abort();
		}
	}

	return stage.str();
}

std::string alp::internal::Stage::generateFoldlMaxOp( const std::string &tabs ) const {

	const std::string arg1 = tensor0.getAccessedElement( pipeline.getID() );
	const std::string arg2 = tensor1.getAccessedElement( pipeline.getID() );

	const std::vector< int > op_axes = computeOperatorAxes();
	std::stringstream stage;

	switch ( rule ) {
		case Rule::EWISE:
		{
			if( op_axes.size() == 1) {
				stage << tabs << "\t\t\talp::VectorEwiseMax( " << arg1 << ", " << arg1 << ", "
					  << arg2 << ", " << igrid->problemSize( op_axes[ 0 ] ) << " );\n";
			} else if( op_axes.size() == 2) {
				stage << tabs << "\t\t\talp::BlockEwiseMax( " << arg1 << ", " << arg1 << ", "
					  << arg2 << ", " << igrid->problemSize( op_axes[ 0 ] ) << ", "
					  << igrid->problemSize( op_axes[ 1 ] ) << " );\n";
			}
			break;
		}
		case Rule::BCAST:
		{
			break;
		}
		case Rule::REDUCE:
		{
			if( op_axes.size() == 1) {
				stage << tabs << "\t\t\talp::VectorReduceMax( " << arg1 << ", " << arg2 << ", "
					  << alp::internal::symbols.getLocalTempTensorBuffer( tensor0.getType() ) << "[ 0 ], "
					  << pipeline.getTilingAxes() << igrid->problemSize( op_axes[ 0 ] ) << " );\n";
			} else if( op_axes.size() == 2) {
				stage << tabs << "\t\t\talp::BlockReduceMax( " << arg1 << ", " << arg2 << ", "
					  << alp::internal::symbols.getLocalTempTensorBuffer( tensor0.getType() ) << "[ 0 ], "
					  << pipeline.getTilingAxes() << igrid->problemSize( op_axes[ 0 ] ) << ", "
					  << igrid->problemSize( op_axes[ 1 ] ) << " );\n";
			}
			break;
		}
		default:
		{
		        std::cerr << "Invalid rule: foldl max" << std::endl;
		        std::abort();
		}
	}

	return stage.str();
}

std::string alp::internal::Stage::generateFoldlTimesOp( const std::string &tabs ) const {

	const std::string arg1 = tensor0.getAccessedElement( pipeline.getID() );
	const std::string arg2 = tensor1.getAccessedElement( pipeline.getID() );

	const std::vector< int > op_axes = computeOperatorAxes();
	std::stringstream stage;

	switch ( rule ) {
		case Rule::EWISE:
		{
			if( op_axes.size() == 1) {
				stage << tabs << "\t\t\talp::VectorEwiseMultiply( " << arg1 << ", " << arg1 << ", "
					  << arg2 << ", " << igrid->problemSize( op_axes[ 0 ] ) << " );\n";
			} else if( op_axes.size() == 2) {
				stage << tabs << "\t\t\talp::BlockEwiseMultiply( " << arg1 << ", " << arg1 << ", "
					  << arg2 << ", " << igrid->problemSize( op_axes[ 0 ] ) << ", "
					  << igrid->problemSize( op_axes[ 1 ] ) << " );\n";
			}
			break;
		}
		case Rule::BCAST:
		{
			break;
		}
		case Rule::REDUCE:
		{
			break;
		}
		default:
		{
		        std::cerr << "Invalid rule: foldl times" << std::endl;
		        std::abort();
		}
	}

	return stage.str();
}

std::string alp::internal::Stage::generateFoldlAddOp( const std::string &tabs ) const {

	const std::string arg1 = tensor0.getAccessedElement( pipeline.getID() );
	const std::string arg2 = tensor1.getAccessedElement( pipeline.getID() );

	const std::vector< int > op_axes = computeOperatorAxes();
	std::stringstream stage;

	switch ( rule ) {
		case Rule::EWISE:
		{
			if( op_axes.size() == 1) {
				stage << tabs << "\t\t\talp::VectorEwiseSum( " << arg1 << ", " << arg1 << ", "
						<< arg2 << ", " << igrid->problemSize( op_axes[ 0 ] ) << " );\n";
			} else if( op_axes.size() == 2) {
				stage << tabs << "\t\t\talp::BlockEwiseSum( " << arg1 << ", " << arg1 << ", "
						<< arg2 << ", " << igrid->problemSize( op_axes[ 0 ] ) << ", "
						<< igrid->problemSize( op_axes[ 1 ] ) << " );\n";
			}
			break;
		}
		case Rule::BCAST:
		{
			break;
		}
		case Rule::REDUCE:
		{
			if( op_axes.size() == 1) {
				stage << tabs << "\t\t\talp::VectorReduceSum( " << arg1 << ", " << arg2 << ", "
					  << alp::internal::symbols.getLocalTempTensorBuffer( tensor0.getType() ) << "[ 0 ], "
					  << pipeline.getTilingAxes() << igrid->problemSize( op_axes[ 0 ] ) << " );\n";
			} else if( op_axes.size() == 2) {
				stage << tabs << "\t\t\talp::BlockReduceSum( " << arg1 << ", " << arg2 << ", "
					  << alp::internal::symbols.getLocalTempTensorBuffer( tensor0.getType() ) << "[ 0 ], "
					  << pipeline.getTilingAxes() << igrid->problemSize( op_axes[ 0 ] ) << ", "
					  << igrid->problemSize( op_axes[ 1 ] ) << " );\n";
			}
			break;
		}
		default:
		{
		        std::cerr << "Invalid rule: foldl add" << std::endl;
		        std::abort();
		}
	}

	return stage.str();
}

std::string alp::internal::Stage::generateFoldlExpOp( const std::string &tabs ) const {

	const std::string arg1 = tensor0.getAccessedElement( pipeline.getID() );

	const std::vector< int > op_axes = computeOperatorAxes();
	std::stringstream stage;

	if( op_axes.size() == 1) {
		stage << tabs << "\t\t\talp::VectorExp( " << arg1 << ", " << arg1
			  << ", " << pipeline.getTilingAxes() << igrid->problemSize( op_axes[ 0 ] ) << " );\n";
	} else if( op_axes.size() == 2) {
		stage << tabs << "\t\t\talp::BlockExp( " << arg1 << ", " << arg1
			  << ", " << pipeline.getTilingAxes() << igrid->problemSize( op_axes[ 0 ] ) << ", "
			  << igrid->problemSize( op_axes[ 1 ] ) << " );\n";
	}

	return stage.str();
}

std::string alp::internal::Stage::generateSetTensorOp( const std::string &tabs ) const {
	const std::string arg1 = tensor0.getAccessedElement( pipeline.getID() );
	const std::string arg2 = tensor1.getAccessedElement( pipeline.getID() );

	const std::vector< int > op_axes = computeOperatorAxes();
	std::stringstream stage;

	if( op_axes.size() == 1) {
		stage << tabs << "\t\t\talp::VectorSet( " << arg1 << ", " << arg2 << ", "
			  << igrid->problemSize( op_axes[ 0 ] ) << " );\n";
	} else if( op_axes.size() == 2) {
		stage << tabs << "\t\t\talp::BlockSet( " << arg1 << ", " << arg2 << ", "
			  << pipeline.getTilingAxes() << igrid->problemSize( op_axes[ 0 ] ) << ", "
			  << igrid->problemSize( op_axes[ 1 ] ) << " );\n";
	}

	return stage.str();
}

std::string alp::internal::Stage::generateSetScalarOp( const std::string &tabs ) const {

	const std::string arg1 = tensor0.getAccessedElement( pipeline.getID() );
	const std::string scalar = ( alpha == std::numeric_limits< double >::infinity() ) ? "65504.0"
							 : ( alpha == -std::numeric_limits< double >::infinity() ) ? "-65504.0"
							 : std::to_string( alpha );

	const std::vector< int > op_axes = computeOperatorAxes();
	std::stringstream stage;

	if( op_axes.size() == 1) {
		stage << tabs << "\t\t\talp::VectorSet( " << arg1 << ", " << scalar << ", "
			  << igrid->problemSize( op_axes[ 0 ] ) << " );\n";
	} else if( op_axes.size() == 2) {
		stage << tabs << "\t\t\talp::BlockSet( " << arg1 << ", " << scalar << ", "
			  << igrid->problemSize( op_axes[ 0 ] ) << ", "
			  << igrid->problemSize( op_axes[ 1 ] ) << " );\n";
	}

	return stage.str();
}

std::string alp::internal::Stage::generateGetViewOp( const std::string &tabs ) const {

	const std::string arg1 = tensor0.getAccessedElement( pipeline.getID() );

	const std::vector< int > op_axes = computeOperatorAxes();
	std::stringstream stage;

	const size_t id = pipeline.getID();

	if( pipeline.isOutput( tensor0 ) == true ) {
		stage << tabs << "\t\t\t// Initializing data for an output global Tensor\n";
		stage << tabs << "\t\t\t" << tensor0.getAscendName( id ) << " = "
			  << tensor0.getTQueBufName( id ) << ".AllocTensor< "
			  << internal::getDataType( tensor0.getType() ) << " >();\n";
	} else {
		stage << tabs << "\t\t\t// Initializing data for an input global Tensor\n";
		stage << tabs << "\t\t\t" << tensor0.getAscendName( id ) << " = "
			  << tensor0.getTQueBufName( id ) << ".AllocTensor< "
			  << internal::getDataType( tensor0.getType() ) << " >();\n";

		if( op_axes.size() == 0 ) {
			stage << tabs << "\t\t\talp::DataMove( " << tensor0.getAscendName( id )
				  << "[ " << "0" << " ], "
				  << tensor0.getAscendGlobalName( id ) << "[ " << tensor0_offset << " ], "
				  << pipeline.getTilingAxes() << "1" << " );\n";

		}else if( op_axes.size() == 1 ) {
//			stage << tabs << "\t\t\tDataCopy( " << tensor0.getAscendName( id ) << ", "
//				  << tensor0.getAscendGlobalName( id ) << "[ " << tensor0_offset << " ], "
//				  << igrid->problemSize( op_axes[ 0 ] ) << " );\n";

//			stage << tabs << "\t\t\talp::DataMove( " << tensor0.getAscendName( id ) << ", "
//				  << tensor0.getAscendGlobalName( id ) << "[ " << tensor0_offset << " ], "

			stage << tabs << "\t\t\talp::DataMove( " << tensor0.getAscendName( id )
				  << "[ " << "0" << " ], "
				  << tensor0.getAscendGlobalName( id ) << "[ " << tensor0_offset << " ], "
				  << pipeline.getTilingAxes() <<  igrid->problemSize( op_axes[ 0 ] ) << " );\n";

		} else if( op_axes.size() == 2) {
//			stage << tabs << "\t\t\tfor( uint32_t k = 0; k < "
//				  << igrid->problemSize( op_axes[ 0 ] ) << "; k++ ) {\n";

//			stage << tabs << "\t\t\t\tDataCopy( " << tensor0.getAscendName( id )
//				  << "[ k * " << igrid->problemSize( op_axes[ 1 ] ) << " ], "
//				  << tensor0.getAscendGlobalName( id )
//				  << "[ " << tensor0_offset << " + k" << stride << " ], "
//				  << igrid->problemSize( op_axes[ 1 ] ) << " );\n";

//			stage << tabs << "\t\t\t}\n";

			stage << tabs << "\t\t\talp::DataMove( " << tensor0.getAscendName( id )
				  << "[ " << "0" << " ], "
				  << tensor0.getAscendGlobalName( id ) << "[ " << tensor0_offset << " ], "
				  << pipeline.getTilingAxes() << igrid->problemSize( op_axes[ 0 ] ) << ", "
				  << igrid->problemSize( op_axes[ 1 ] ) << ", "
				  << stride << ", "
				  << igrid->problemSize( op_axes[ 1 ] ) << " );\n";
		}

		stage << tabs << "\t\t\t" << tensor0.getTQueBufName( id )
			  << ".EnQue( " << tensor0.getAscendName( id ) << " );\n";

		stage << tabs << "\t\t\t" << tensor0.getAscendName( id ) << " = "
			  << tensor0.getTQueBufName( id )
			  << ".DeQue< " << internal::getDataType( tensor0.getType() ) << " >();\n";
	}

	return stage.str();
}

std::string alp::internal::Stage::generateStoreOp( const std::string &tabs ) const {

	//TODO I should use the arg1
	const std::string arg1 = tensor0.getAccessedElement( pipeline.getID() );

	const std::vector< int > op_axes = computeOperatorAxes();
	std::stringstream stage;

	const size_t id = pipeline.getID();

	stage << tabs << "\t\t\t// Copying data of an output Tensor back to the global memory\n";
	stage << tabs << "\t\t\t" << tensor0.getTQueBufName( id )
		  << ".EnQue< " << internal::getDataType( tensor0.getType() )
		  << " >( " << tensor0.getAscendName( id ) << " );\n";
	stage << tabs << "\t\t\t" << tensor0.getAscendName( id ) << " = "
		  << tensor0.getTQueBufName( id ) << ".DeQue< "
		  << internal::getDataType( tensor0.getType() ) << " >();\n";

	if( op_axes.size() == 0) {
		stage << tabs << "\t\t\talp::DataMove( "
			<< tensor0.getAscendGlobalName( id ) << "[ " << tensor0_offset << " ], "
			<< tensor0.getAscendName( id ) << "[ " << "0" << " ], "
			<< pipeline.getTilingAxes() << "1" << " );\n";
	} else if( op_axes.size() == 1) {
//		stage << tabs << "\t\t\tDataCopy( " << tensor0.getAscendGlobalName( id )
//			  << "[ " << tensor0_offset << " ], " << tensor0.getAscendName( id ) << ", "
//			  << igrid->problemSize( op_axes[ 0 ] ) << " );\n";

		stage << tabs << "\t\t\talp::DataMove( "
			<< tensor0.getAscendGlobalName( id ) << "[ " << tensor0_offset << " ], "
			<< tensor0.getAscendName( id ) << "[ " << "0" << " ], "
			<< pipeline.getTilingAxes() <<  igrid->problemSize( op_axes[ 0 ] ) << " );\n";

	} else if( op_axes.size() == 2) {
/*
		stage << tabs << "\t\t\tfor( uint32_t k = 0; k < "
			  << igrid->problemSize( op_axes[ 0 ] ) << "; k++ ) {\n";

		stage << tabs << "\t\t\t\tDataCopy( " << tensor0.getAscendGlobalName( id )
			  << "[ " << tensor0_offset << " + k" << stride << " ], "
			  << tensor0.getAscendName( id ) << "[ k * " << igrid->problemSize( op_axes[ 1 ] ) << " ], "
			  << igrid->problemSize( op_axes[ 1 ] ) << " );\n";

		stage << tabs << "\t\t\t}\n";
*/
		stage << tabs << "\t\t\talp::DataMove( "
			  << tensor0.getAscendGlobalName( id ) << "[ " << tensor0_offset << " ], "
			  << tensor0.getAscendName( id ) << "[ " << "0" << " ], "
			  << pipeline.getTilingAxes() << igrid->problemSize( op_axes[ 0 ] ) << ", "
			  << igrid->problemSize( op_axes[ 1 ] ) << ", "
			  << igrid->problemSize( op_axes[ 1 ] ) << ", "
			  << stride << " );\n";
	}

	stage << tabs << "\t\t\t" << tensor0.getTQueBufName( id )
		  << ".FreeTensor( " << tensor0.getAscendName( id ) << " );\n";

	return stage.str();
}

std::string alp::internal::Stage::generateImplicitFreeOp( const std::string &tabs ) const {

	//TODO I should use the arg1
	const std::string arg1 = tensor0.getAccessedElement( pipeline.getID() );

	std::stringstream stage;

	const size_t id = pipeline.getID();

	stage << tabs << "\t\t\t// Freeing data of a Tensor that is not output\n";
	stage << tabs << "\t\t\t" << tensor0.getTQueBufName( id )
		  << ".FreeTensor( " << tensor0.getAscendName( id ) << " );\n";

	return stage.str();
}

std::string alp::internal::Stage::generateToDoOp( const std::string &tabs ) const {

	return tabs + std::string("");
}

//TODO: perhaps rename it to computeUnionAxes

std::vector< int > alp::internal::Stage::computeOperatorAxes() const {

	// initializing the union with the axes of tensor0 used by all operators
	std::vector< int > union_axes = tensor0.getAxes();

	switch ( enum_op_type ) {

		case alp::internal::Stagetype::APPLY_MINUS:		// 3 Tensors
		case alp::internal::Stagetype::APPLY_ADD:		// 3 Tensors
		case alp::internal::Stagetype::FOLDL_DIVIDE:	// 2 Tensors
		case alp::internal::Stagetype::SET_TENSOR:		// 2 Tensors
		case alp::internal::Stagetype::FOLDL_MAX:		// 2 Tensors
		case alp::internal::Stagetype::FOLDL_TIMES:		// 2 Tensors
		case alp::internal::Stagetype::FOLDL_ADD:		// 2 Tensors
		{
			const std::vector< int > &tensor1_axes = tensor1.getAxes();
			union_axes = internal::vectorUnion( union_axes, tensor1_axes );
			break;
		}
		case alp::internal::Stagetype::FOLDL_EXP:		// 1 Tensor
		case alp::internal::Stagetype::SET_SCALAR:		// 1 Tensor
		case alp::internal::Stagetype::GET_VIEW:		// 1 Tensor
		case alp::internal::Stagetype::STORE:			// 1 Tensor
		case alp::internal::Stagetype::IMPLICIT_FREE:	// 1 Tensor
//		default:
		break;
	}

	switch ( enum_op_type ) {

		case alp::internal::Stagetype::APPLY_MINUS:		// 3 Tensors
		case alp::internal::Stagetype::APPLY_ADD:		// 3 Tensors
		{
			const std::vector< int > &tensor2_axes = tensor2.getAxes();
			union_axes = internal::vectorUnion( union_axes, tensor2_axes );
			break;
		}
		case alp::internal::Stagetype::FOLDL_DIVIDE:	// 2 Tensors
		case alp::internal::Stagetype::SET_TENSOR:		// 2 Tensors
		case alp::internal::Stagetype::FOLDL_MAX:		// 2 Tensors
		case alp::internal::Stagetype::FOLDL_TIMES:		// 2 Tensors
		case alp::internal::Stagetype::FOLDL_ADD:		// 2 Tensors
		case alp::internal::Stagetype::FOLDL_EXP:		// 1 Tensor
		case alp::internal::Stagetype::SET_SCALAR:		// 1 Tensor
		case alp::internal::Stagetype::GET_VIEW:		// 1 Tensor
		case alp::internal::Stagetype::STORE:			// 1 Tensor
		case alp::internal::Stagetype::IMPLICIT_FREE:	// 1 Tensor
//		default:
		break;
	}

	// only in the case of GET_VIEW and STORE
	// we need to remove the axes of the loops
	// because the stored axes are those of the parent
	// FIXME: perhaps we should change this design and handle views
	// as different objects added to the symbol table
	switch ( enum_op_type ) {

		case alp::internal::Stagetype::GET_VIEW:		// 1 Tensor
		case alp::internal::Stagetype::STORE:			// 1 Tensor
		{
			union_axes = internal::vectorDifference( union_axes, forEachAxes );
			break;
		}
		// IMPLICIT_FREE is created based on STORE
		// and this step is already done except that
		// this function is not used by IMPLICIT_FREE
		case alp::internal::Stagetype::IMPLICIT_FREE:	// 1 Tensor
		case alp::internal::Stagetype::APPLY_MINUS:		// 3 Tensors
		case alp::internal::Stagetype::APPLY_ADD:		// 3 Tensors
		case alp::internal::Stagetype::FOLDL_DIVIDE:	// 2 Tensors
		case alp::internal::Stagetype::SET_TENSOR:		// 2 Tensors
		case alp::internal::Stagetype::FOLDL_MAX:		// 2 Tensors
		case alp::internal::Stagetype::FOLDL_TIMES:		// 2 Tensors
		case alp::internal::Stagetype::FOLDL_ADD:		// 2 Tensors
		case alp::internal::Stagetype::FOLDL_EXP:		// 1 Tensor
		case alp::internal::Stagetype::SET_SCALAR:		// 1 Tensor
//		default:
		break;
	}

	return union_axes;
}

void alp::internal::Stage::computeMemoryOffsets(){

	switch ( enum_op_type ) {

		case alp::internal::Stagetype::GET_VIEW:		// 1 Tensor
		case alp::internal::Stagetype::STORE:			// 1 Tensor
		case alp::internal::Stagetype::IMPLICIT_FREE:	// 1 Tensor
		{
			// for the GET_VIEW and STORE it's necessary to compute the expression for the stride
			// we compute the stride only if the axes of the view are two
			// more than two axes are not supported
			// one axis does not require the stride
			if( activeAxes.size() == 2 ) {
				bool first = true;
				for( int i = activeAxes[ 0 ] + 1; i <= activeAxes[ 1 ]; ++i ) {
					if( first == true ) {
						first = false;
						stride.append( igrid->problemSize( i ) ); // n3 * n4 * n5
					} else {
						stride.append( " * " + igrid->problemSize( i ) ); // n3 * n4 * n5
					}
				}
			}
			break;
		}
		default:
			break;
	}

	switch ( enum_op_type ) {

//		case alp::internal::Stagetype::APPLY_MINUS:		// 4 Tensors
		case alp::internal::Stagetype::APPLY_MINUS:		// 3 Tensors
		case alp::internal::Stagetype::APPLY_ADD:		// 3 Tensors
		case alp::internal::Stagetype::FOLDL_DIVIDE:	// 2 Tensors
		case alp::internal::Stagetype::SET_TENSOR:		// 2 Tensors
		case alp::internal::Stagetype::FOLDL_MAX:		// 2 Tensors
		case alp::internal::Stagetype::FOLDL_TIMES:		// 2 Tensors
		case alp::internal::Stagetype::FOLDL_ADD:		// 2 Tensors
		case alp::internal::Stagetype::GET_VIEW:		// 1 Tensor
		case alp::internal::Stagetype::STORE:			// 1 Tensor
		case alp::internal::Stagetype::IMPLICIT_FREE:	// 1 Tensor
		case alp::internal::Stagetype::FOLDL_EXP:		// 1 Tensor
		case alp::internal::Stagetype::SET_SCALAR:		// 1 Tensor
		{
			const std::vector< int > &view_parent_0_axes = alp::internal::symbols.getTensorFromView( tensor0 ).getAxes();

			bool first = true;

			for( auto it = forEachAxes.begin(); it != forEachAxes.end(); ++it ) {

				if( std::find( view_parent_0_axes.begin(), view_parent_0_axes.end(), *it ) != view_parent_0_axes.end() ) {

					if( !first ) {
						tensor0_offset.append( " + " );
					} else {
						first = false;
					}

					tensor0_offset.append( igrid->problemMainMode( *it ) ); // z0
					for( auto jt = view_parent_0_axes.begin(); jt != view_parent_0_axes.end(); ++jt ) {
						if( *jt > *it ) {
							tensor0_offset.append( " * " + igrid->problemSize( *jt ) ); // n1 * n2 * n3
						}
					}
				}
			}
			break;
		}
//		default:
	}

	switch ( enum_op_type ) {

//		case alp::internal::Stagetype::APPLY_MINUS:		// 4 Tensors
		case alp::internal::Stagetype::APPLY_MINUS:		// 3 Tensors
		case alp::internal::Stagetype::APPLY_ADD:		// 3 Tensors
		case alp::internal::Stagetype::FOLDL_DIVIDE:	// 2 Tensors
		case alp::internal::Stagetype::SET_TENSOR:		// 2 Tensors
		case alp::internal::Stagetype::FOLDL_MAX:		// 2 Tensors
		case alp::internal::Stagetype::FOLDL_TIMES:		// 2 Tensors
		case alp::internal::Stagetype::FOLDL_ADD:		// 2 Tensors
		{
			const std::vector< int > &view_parent_1_axes = alp::internal::symbols.getTensorFromView( tensor1 ).getAxes();

			bool first = true;

			for( auto it = forEachAxes.begin(); it != forEachAxes.end(); ++it ) {

				if( std::find( view_parent_1_axes.begin(), view_parent_1_axes.end(), *it ) != view_parent_1_axes.end() ) {

					if( !first ) {
						tensor1_offset.append( " + " );
					} else {
						first = false;
					}

					tensor1_offset.append( igrid->problemMainMode( *it ) ); // z0
					for( auto jt = view_parent_1_axes.begin(); jt != view_parent_1_axes.end(); ++jt ) {
						if( *jt > *it ) {
							tensor1_offset.append( " * " + igrid->problemSize( *jt ) ); // n1 * n2 * n3
						}
					}
				}
			}
			break;
		}
		case alp::internal::Stagetype::GET_VIEW:		// 1 Tensor
		case alp::internal::Stagetype::STORE:			// 1 Tensor
		case alp::internal::Stagetype::IMPLICIT_FREE:	// 1 Tensor
		case alp::internal::Stagetype::FOLDL_EXP:		// 1 Tensor
		case alp::internal::Stagetype::SET_SCALAR:		// 1 Tensor
//		default:
		break;
	}

	switch ( enum_op_type ) {

//		case alp::internal::Stagetype::APPLY_MINUS:		// 4 Tensors
		case alp::internal::Stagetype::APPLY_MINUS:		// 3 Tensors
		case alp::internal::Stagetype::APPLY_ADD:		// 3 Tensors
		{
			const std::vector< int > &view_parent_2_axes = alp::internal::symbols.getTensorFromView( tensor2 ).getAxes();

			bool first = true;

			for( auto it = forEachAxes.begin(); it != forEachAxes.end(); ++it ) {

				if( std::find( view_parent_2_axes.begin(), view_parent_2_axes.end(), *it ) != view_parent_2_axes.end() ) {

					if( !first ) {
						tensor2_offset.append( " + " );
					} else {
						first = false;
					}

					tensor2_offset.append( igrid->problemMainMode( *it ) ); // z0
					for( auto jt = view_parent_2_axes.begin(); jt != view_parent_2_axes.end(); ++jt ) {
						if( *jt > *it ) {
							tensor2_offset.append( " * " + igrid->problemSize( *jt ) ); // n1 * n2 * n3
						}
					}
				}
			}
			break;
		}
		case alp::internal::Stagetype::FOLDL_DIVIDE:	// 2 Tensors
		case alp::internal::Stagetype::SET_TENSOR:		// 2 Tensors
		case alp::internal::Stagetype::FOLDL_MAX:		// 2 Tensors
		case alp::internal::Stagetype::FOLDL_TIMES:		// 2 Tensors
		case alp::internal::Stagetype::FOLDL_ADD:		// 2 Tensors
		case alp::internal::Stagetype::GET_VIEW:		// 1 Tensor
		case alp::internal::Stagetype::STORE:			// 1 Tensor
		case alp::internal::Stagetype::IMPLICIT_FREE:	// 1 Tensor
		case alp::internal::Stagetype::FOLDL_EXP:		// 1 Tensor
		case alp::internal::Stagetype::SET_SCALAR:		// 1 Tensor
//		default:
		break;
	}
}

void alp::internal::Stage::semanticsCheks(){

	switch ( enum_op_type ) {

//		case alp::internal::Stagetype::APPLY_MINUS:		// 4 Tensors
		case alp::internal::Stagetype::APPLY_MINUS:		// 3 Tensors
		case alp::internal::Stagetype::APPLY_ADD:		// 3 Tensors
		case alp::internal::Stagetype::FOLDL_DIVIDE:	// 2 Tensors
		case alp::internal::Stagetype::SET_TENSOR:		// 2 Tensors
		case alp::internal::Stagetype::FOLDL_MAX:		// 2 Tensors
		case alp::internal::Stagetype::FOLDL_TIMES:		// 2 Tensors
		case alp::internal::Stagetype::FOLDL_ADD:		// 2 Tensors
		case alp::internal::Stagetype::FOLDL_EXP:		// 1 Tensor
		case alp::internal::Stagetype::SET_SCALAR:		// 1 Tensor
		{
			if( internal::invalidAxes( tensor0.getAxes() ) == true ) {
				std::cerr << "The axes of the Tensor must not be included in the axes of the forEach." << std::endl;
				std::abort();
			}
			break;
		}
		case alp::internal::Stagetype::GET_VIEW:		// 1 Tensor
		case alp::internal::Stagetype::STORE:			// 1 Tensor
		case alp::internal::Stagetype::IMPLICIT_FREE:	// 1 Tensor
		{
			//TODO this semantics check cannot be done on the parent tensor
			break;
		}
//		default:
	}

	switch ( enum_op_type ) {

//		case alp::internal::Stagetype::APPLY_MINUS:		// 4 Tensors
		case alp::internal::Stagetype::APPLY_MINUS:		// 3 Tensors
		case alp::internal::Stagetype::APPLY_ADD:		// 3 Tensors
		case alp::internal::Stagetype::FOLDL_DIVIDE:	// 2 Tensors
		case alp::internal::Stagetype::SET_TENSOR:		// 2 Tensors
		case alp::internal::Stagetype::FOLDL_MAX:		// 2 Tensors
		case alp::internal::Stagetype::FOLDL_TIMES:		// 2 Tensors
		case alp::internal::Stagetype::FOLDL_ADD:		// 2 Tensors
		{
			if( internal::invalidAxes( tensor1.getAxes() ) == true ) {
				std::cerr << "The axes of the Tensor must not be included in the axes of the forEach." << std::endl;
				std::abort();
			}
			break;
		}
		case alp::internal::Stagetype::FOLDL_EXP:		// 1 Tensor
		case alp::internal::Stagetype::SET_SCALAR:		// 1 Tensor
		case alp::internal::Stagetype::GET_VIEW:		// 1 Tensor
		case alp::internal::Stagetype::STORE:			// 1 Tensor
		case alp::internal::Stagetype::IMPLICIT_FREE:	// 1 Tensor
		{
			//TODO this semantics check cannot be done on the parent tensor
			break;
		}
//		default:
	}

	switch ( enum_op_type ) {

//		case alp::internal::Stagetype::APPLY_MINUS:		// 4 Tensors
		case alp::internal::Stagetype::APPLY_MINUS:		// 3 Tensors
		case alp::internal::Stagetype::APPLY_ADD:		// 3 Tensors
		{
			if( internal::invalidAxes( tensor2.getAxes() ) == true ) {
				std::cerr << "The axes of the Tensor must not be included in the axes of the forEach." << std::endl;
				std::abort();
			}
			break;
		}
		case alp::internal::Stagetype::FOLDL_DIVIDE:	// 2 Tensors
		case alp::internal::Stagetype::SET_TENSOR:		// 2 Tensors
		case alp::internal::Stagetype::FOLDL_MAX:		// 2 Tensors
		case alp::internal::Stagetype::FOLDL_TIMES:		// 2 Tensors
		case alp::internal::Stagetype::FOLDL_ADD:		// 2 Tensors
		case alp::internal::Stagetype::FOLDL_EXP:		// 1 Tensor
		case alp::internal::Stagetype::SET_SCALAR:		// 1 Tensor
		case alp::internal::Stagetype::GET_VIEW:		// 1 Tensor
		case alp::internal::Stagetype::STORE:			// 1 Tensor
		case alp::internal::Stagetype::IMPLICIT_FREE:	// 1 Tensor
		{
			//TODO this semantics check cannot be done on the parent tensor
			break;
		}
//		default:
	}
}
