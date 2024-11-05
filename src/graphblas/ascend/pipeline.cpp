#include <vector>

#include <graphblas/ascend/pipeline.hpp>
#include <graphblas/ascend/tensor.hpp>
#include <graphblas/ascend/stage.hpp>
#include <graphblas/ascend/symbolTable.hpp>
#include <graphblas/ascend/grid.hpp>

namespace alp
{
	namespace internal
	{
		extern iGrid *igrid;
		extern SymbolTable symbols;
	}
}

alp::internal::AscendPipeline::AscendPipeline( size_t _id ) : id( _id )
{

}

void alp::internal::AscendPipeline::insertTensorToInputs( const alp::Tensor &tensor )
{
	accessed.insert( alp::internal::symbols.getTensorFromView( tensor ) );
}

void alp::internal::AscendPipeline::insertFreeInputTensorStages( const std::vector< int > &forEachAxes )
{
	std::vector< alp::internal::Stage * > st;

	// search for all GET_VIEW stages in the pipeline and store them
	for( auto it = stages.begin(); it != stages.end(); ++it ) {
		if (it->getOpType() == internal::Stagetype::GET_VIEW ) {
			st.push_back( &(*it) );
		}
	}

	// search for all STORE stages in the pipeline and delete
	// the corresponding GET_VIEW from those stored above
	for( auto it = stages.begin(); it != stages.end(); ++it ) {
		if (it->getOpType() == internal::Stagetype::STORE ) {
			for( auto jt = st.begin(); jt != st.end(); ) {
				if( (*jt)->getTensor0().getID() == it->getTensor0().getID() ) {
					jt = st.erase( jt );
				} else {
					++jt;
				}
			}
		}
	}

	// for the remaining GET_VIEW stages that are still stored
	// insert a new stage in the end of the pipeline that
	// corresponds to all input tensors for which store
	// is not explicitly invoked by the user
	for( auto it = st.begin(); it != st.end(); ++it ) {
		if( (*it)->getForEachAxes() == forEachAxes ) {
			addStage( alp::internal::Stagetype::IMPLICIT_FREE, (*it)->getRule(), (*it)->getTensor0(), (*it)->getAxes(), (*it)->getForEachAxes() );
		}
	}
}

std::set< int > alp::internal::AscendPipeline::getIteratedAxes() const {
	std::vector< int > union_iterated_axes;

	for( auto it = stages.begin(); it != stages.end(); ++it ) {
		union_iterated_axes = internal::vectorUnion( union_iterated_axes, it->getForEachAxes() );
	}

	// convert the std::vector to std::set
	std::set< int > ret;
	ret.insert( union_iterated_axes.begin(), union_iterated_axes.end() );
	return ret;
}

const alp::Tensor &alp::internal::AscendPipeline::store( const alp::Tensor &output_tensor ) {

	//FIXME I should check here that this is indeed a VIEW

	const alp::Tensor &parent = alp::internal::symbols.getTensorFromView( output_tensor );
	outputs.insert( parent );

	alp::internal::symbols.addOutputTensor( parent );

	return parent;
}

bool alp::internal::AscendPipeline::isOutput( const alp::Tensor &tensor ) const {

	//FIXME I should check here that this is indeed a VIEW

	return outputs.find( tensor ) != outputs.end();
}

void alp::internal::AscendPipeline::clear() {

	stages.clear();
	accessed.clear();
	outputs.clear();
}

size_t alp::internal::AscendPipeline::getID() const {

	return id;
}

std::string alp::internal::AscendPipeline::getTilingAxes() const {

	std::string tiling_init_numerator;

	const std::set< int > iterated_axes = getIteratedAxes();

	for( auto it = iterated_axes.begin(); it != iterated_axes.end(); ++it ) {
		tiling_init_numerator.append( igrid->tileSize( *it ) );
		tiling_init_numerator.append( " * " );
	}

	return tiling_init_numerator;
}

void alp::internal::AscendPipeline::addStage( alp::internal::Stagetype op_type, alp::internal::Rule rule,
											  const alp::Tensor &tensor1, const double alpha, const std::vector< int > &forEachAxes ) {

	// insert the Tensor to the set of accessed data
	insertTensorToInputs( tensor1 );
	// get the name of the Tensor object that exists behind this view or tensor

	switch ( op_type ) {

		case alp::internal::Stagetype::SET_SCALAR:
		{
			stages.push_back( std::move( alp::internal::Stage( *this,
												alp::internal::Stagetype::SET_SCALAR, rule, tensor1, alpha, forEachAxes ) ) );

			break;
		}
		case alp::internal::Stagetype::GET_VIEW:
		case alp::internal::Stagetype::STORE:
		case alp::internal::Stagetype::IMPLICIT_FREE:
		{
            std::cerr << "Stage: " << (int) op_type << " has only one tensor argument" << std::endl;
            std::abort();
			break;
		}
		case alp::internal::Stagetype::FOLDL_EXP:
		case alp::internal::Stagetype::SET_TENSOR:
		case alp::internal::Stagetype::APPLY_ADD:
		case alp::internal::Stagetype::APPLY_MINUS:
		case alp::internal::Stagetype::FOLDL_DIVIDE:
		case alp::internal::Stagetype::FOLDL_MAX:
		case alp::internal::Stagetype::FOLDL_TIMES:
		case alp::internal::Stagetype::FOLDL_ADD:
		{
            std::cerr << "Stage: " << (int) op_type << " has more than one tensor arguments" << std::endl;
            std::abort();
			break;
		}
	}
}

void alp::internal::AscendPipeline::addStage( alp::internal::Stagetype op_type, alp::internal::Rule rule,
											  const alp::Tensor &tensor1, const std::vector< int > &activeAxes, const std::vector< int > &forEachAxes ) {

	// insert the Tensor to the set of accessed data
	insertTensorToInputs( tensor1 );
	// get the name of the Tensor object that exists behind this view or tensor

	switch ( op_type ) {

		case alp::internal::Stagetype::FOLDL_EXP:
		case alp::internal::Stagetype::GET_VIEW:
		case alp::internal::Stagetype::STORE:
		case alp::internal::Stagetype::IMPLICIT_FREE:
		{
			stages.push_back( std::move( alp::internal::Stage( *this,
												op_type, rule, tensor1, activeAxes, forEachAxes ) ) );

			break;
		}
		case alp::internal::Stagetype::SET_SCALAR:
		case alp::internal::Stagetype::SET_TENSOR:
		case alp::internal::Stagetype::APPLY_ADD:
		case alp::internal::Stagetype::APPLY_MINUS:
		case alp::internal::Stagetype::FOLDL_DIVIDE:
		case alp::internal::Stagetype::FOLDL_MAX:
		case alp::internal::Stagetype::FOLDL_TIMES:
		case alp::internal::Stagetype::FOLDL_ADD:
		{
            std::cerr << "Stage: " << (int) op_type << " has more than one tensor arguments" << std::endl;
            std::abort();
			break;
		}
	}
}

void alp::internal::AscendPipeline::addStage( alp::internal::Stagetype op_type, alp::internal::Rule rule,
											  const alp::Tensor &tensor1, const alp::Tensor &tensor2,
											  const std::vector< int > &activeAxes, const std::vector< int > &forEachAxes ) {

	// insert the Tensors to the set of accessed data
	insertTensorToInputs( tensor1 ); //TODO pass the string
	insertTensorToInputs( tensor2 );

	switch ( op_type ) {

		case alp::internal::Stagetype::SET_TENSOR:
		case alp::internal::Stagetype::FOLDL_MAX:
		case alp::internal::Stagetype::FOLDL_TIMES:
		case alp::internal::Stagetype::FOLDL_ADD:
		case alp::internal::Stagetype::FOLDL_DIVIDE:
		{
			stages.push_back( std::move( alp::internal::Stage( *this,
												op_type, rule, tensor1, tensor2, activeAxes, forEachAxes ) ) );
			break;
		}
		case alp::internal::Stagetype::APPLY_ADD:
		case alp::internal::Stagetype::APPLY_MINUS:
		case alp::internal::Stagetype::FOLDL_EXP:
		case alp::internal::Stagetype::SET_SCALAR:
		case alp::internal::Stagetype::GET_VIEW:
		case alp::internal::Stagetype::STORE:
		case alp::internal::Stagetype::IMPLICIT_FREE:
		{
            std::cerr << "Stage: " << (int) op_type << " does not have two tensor arguments" << std::endl;
            std::abort();
			break;
		}
	}
}

void alp::internal::AscendPipeline::addStage( alp::internal::Stagetype op_type, alp::internal::Rule rule,
											  const alp::Tensor &tensor1, const alp::Tensor &tensor2,
											  const alp::Tensor &tensor3, const std::vector< int > &activeAxes, const std::vector< int > &forEachAxes ) {

	// insert the Tensors to the set of accessed data
	insertTensorToInputs( tensor1 ); //TODO pass the string
	insertTensorToInputs( tensor2 );
	insertTensorToInputs( tensor3 );

	switch ( op_type ) {

		case alp::internal::Stagetype::APPLY_MINUS:
		case alp::internal::Stagetype::APPLY_ADD:
		{
			stages.push_back( std::move( alp::internal::Stage( *this,
												op_type, rule, tensor1, tensor2, tensor3, activeAxes, forEachAxes ) ) );
			break;
		}
		case alp::internal::Stagetype::FOLDL_DIVIDE:
		case alp::internal::Stagetype::SET_TENSOR:
		case alp::internal::Stagetype::FOLDL_MAX:
		case alp::internal::Stagetype::FOLDL_TIMES:
		case alp::internal::Stagetype::FOLDL_ADD:
		case alp::internal::Stagetype::FOLDL_EXP:
		case alp::internal::Stagetype::SET_SCALAR:
		case alp::internal::Stagetype::GET_VIEW:
		case alp::internal::Stagetype::STORE:
		case alp::internal::Stagetype::IMPLICIT_FREE:
		{
            std::cerr << "Stage: " << (int) op_type << " does not have three tensor arguments" << std::endl;
            std::abort();
			break;
		}
	}
}
/*
void alp::internal::AscendPipeline::addStage( alp::internal::Stagetype op_type, alp::internal::Rule rule,
											  const alp::Tensor &tensor1, const alp::Tensor &tensor2,
											  const alp::Tensor &tensor3, const alp::Tensor &tensor4,
											  const std::vector< int > &activeAxes, const std::vector< int > &forEachAxes ) {

	// insert the Tensors to the set of accessed data
	insertTensorToInputs( tensor1 ); //TODO pass the string
	insertTensorToInputs( tensor2 );
	insertTensorToInputs( tensor3 );
	insertTensorToInputs( tensor4 );

	switch ( op_type ) {

		case alp::internal::Stagetype::APPLY_MINUS:
		{
			//TODO tensor4 is a temporary variable

			stages.push_back( std::move( alp::internal::Stage( *this,
												op_type, rule, tensor1, tensor2, tensor3, tensor4, activeAxes, forEachAxes ) ) );
			break;
		}
		case alp::internal::Stagetype::FOLDL_DIVIDE:
		case alp::internal::Stagetype::APPLY_ADD:
		case alp::internal::Stagetype::SET_TENSOR:
		case alp::internal::Stagetype::FOLDL_MAX:
		case alp::internal::Stagetype::FOLDL_TIMES:
		case alp::internal::Stagetype::FOLDL_ADD:
		case alp::internal::Stagetype::FOLDL_EXP:
		case alp::internal::Stagetype::SET_SCALAR:
		case alp::internal::Stagetype::GET_VIEW:
		case alp::internal::Stagetype::STORE:
		case alp::internal::Stagetype::IMPLICIT_FREE:
		{
            std::cerr << "Stage: " << (int) op_type << " does not have four tensor arguments" << std::endl;
            std::abort();
			break;
		}
	}
}
*/
void alp::internal::AscendPipeline::generateDeclarations(
	std::stringstream &declarations
) {

//	declarations << "\t\tuint32_t " << "block_length" << id << ";\n";
//	declarations << "\t\tuint32_t " << "tile_length" << id << ";\n";
//	declarations << "\n";

	for( auto it = accessed.cbegin(); it != accessed.cend(); ++it ) {
		if( it->isGlobalDecl() ) {
			declarations << "\t\t// Global Tensor declaration\n";
			if( outputs.find( *it ) != outputs.end() ) {
				// TQue< QuePosition::VECOUT, BUFFER_NUM > globalQue_tensor1_0;
				declarations << "\t\tTQue< QuePosition::VECOUT, BUFFER_NUM > "
							 << it->getTQueBufName( id ) << ";\n";
			} else {
				// TQue< QuePosition::VECIN, BUFFER_NUM > globalQue_tensor0_0;
				declarations << "\t\tTQue< QuePosition::VECIN, BUFFER_NUM > "
							 << it->getTQueBufName( id ) << ";\n";
			}
			// GlobalTensor< half > Gm_tensor0_0;
			declarations << "\t\tGlobalTensor< " << internal::getDataType( it->getType() ) << " > "
						 << it->getAscendGlobalName( id ) << ";\n";
			// LocalTensor< half > Gm_local_tensor0_0;
			declarations << "\t\tLocalTensor< " << internal::getDataType( it->getType() ) << " > "
						 << it->getAscendName( id ) << ";\n";
		} else if( it->isLocalDecl() ) {
/*			declarations << "\t\t// Local Tensor declaration\n";
			// TBuf< QuePosition::VECCALC > localBuf_tensor4_0;
			declarations << "\t\tTBuf< QuePosition::VECCALC > "
						 << it->getTQueBufName( id ) << ";\n";
			// LocalTensor< half > local_tensor4_0;
			declarations << "\t\tLocalTensor< " << internal::getDataType( it->getType() ) << " > "
						 << it->getAscendName( id ) << ";\n";
*/
			declarations << "\t\t// Offset for local Tensor declaration\n";
			declarations << "\t\tint32_t " << it->getAscendName( id ) << ";\n";
		} else if( it->isTempDecl() ) {
/*			declarations << "\t\t// Temporary Tensor declaration\n";
			// TBuf< QuePosition::VECCALC > tempBuf_tensor5_0;
			declarations << "\t\tTBuf< QuePosition::VECCALC > "
						 << it->getTQueBufName( id ) << ";\n";
			// LocalTensor< half > temp_tensor5_0;
			declarations << "\t\tLocalTensor< " << internal::getDataType( it->getType() ) << " > "
						 << it->getAscendName( id ) << ";\n";
*/
			declarations << "\t\t// Offset for temporary Tensor declaration\n";
			declarations << "\t\tint32_t " << it->getAscendName( id ) << ";\n";
		}
		declarations << "\n";
	}
/*
	if( temp_or_local_found == true ) {
		declarations << "\t\t// Declaration of memory used for Local and Temporary tensor\n";
		declarations << "\t\tTBuf< QuePosition::VECCALC > " << "_temp_local;\n";
		declarations << "\t\tLocalTensor< " << "half" << " > " << "_temp_local_Buf;\n";
		declarations << "\n";
	}
*/
}

//void alp::internal::AscendPipeline::generateConstructor( std::stringstream &constructor ) {
/*
	constructor << "\n";
	constructor << "\t\t\tblock_length" << id << " = ( ";
	constructor << igrid->problemSize( 0 );
	for( size_t i = 1; i < igrid->getProblemOrder(); ++i ) {
		constructor << " * " << igrid->problemSize( i );
	}
	constructor << " ) / ( ";
	constructor << igrid->processSize( 0 );
	for( size_t i = 1; i < igrid->getProblemOrder(); ++i ) {
		constructor << " * " << igrid->processSize( i );
	}
	constructor << " );\n";
	constructor << "\t\t\ttile_length" << id << " = ( ";
	bool first = true;
	for( size_t i = 0; i < igrid->getProblemOrder(); ++i ) {
		//TODO this solution assumes that there is only one parallel axis, which is not true
		// omit the problemSize variables for which the corresponding axes is defined in the parallel forEach
		// we use the parallel axes of the first stage, any other stage can be used as well
		// since all stages of the same pipeline have the same outer loop
		if( stages.begin()->getForEachAxes()[ 0 ] != ( int ) i ) {
			if( !first ) {
				constructor << " * ";
			}
			constructor << igrid->problemSize( i );
			first = false;
		}
	}
	constructor << " ) / " << "BUFFER_NUM;\n";
*/
//}

void alp::internal::AscendPipeline::generateHostBody( std::stringstream &os, std::stringstream &analyticModelArgs,
							std::stringstream &analyticModelFormalParams, std::stringstream &analyticModelDecls,
							std::stringstream &analyticModelConstrBody ) {
	// analytic model codeblock
	constexpr size_t ub_size = grb::config::ASCEND_CACHE_HIERARCHY<>::UB_SIZE;

	// This is a symbolic analysis to find what the largest global tensors are.
	// After this symbolic analysis, we will have generally identified multiple
	// global tensors as candidates for being the largest. We're generally still
	// not sure which of these will be the largest tensor at run-time. Therefore,
	// there is still a final run-time component to find the largest tensor(s).
	std::set< std::set< int > > largestGlobals;
	std::vector< Tensor > minorTensors;
	bool differingDynamicAxesPresent = false;
	for( const auto &tensor : accessed ) {
		if( tensor.getScope() == internal::Scope::GLOBAL ) {
			// TODO FIXME think about a cheaper algorithm for computing this check
			const auto &current = tensor.getAxes();
			assert( current.size() > 0 );
			// by default, register the current tensor (don't register only if symbolic
			// analysis is sure it is smaller)
			bool insert = true;
			for( const auto &existing : largestGlobals ) {
				if( existing.size() <= current.size() ) {
					bool larger = true;
					for( const unsigned int &axis : existing ) {
						if( std::find( current.cbegin(), current.cend(), axis ) != current.cend() ) {
							// in this case, static analysis cannot conclude that the current tensor
							// is larger than this entry in largestGlobal -- check the next entry of
							// largestGlobals instead
							larger = false;
							break;
						} else {
							// check if the differing axis is a dynamic one
							if( getIteratedAxes().find( axis ) != getIteratedAxes().cend() ) {
								differingDynamicAxesPresent = true;
							}
						}
					}
					if( larger ) {
						// in this case, the current tensor is guaranteed larger than this entry
						// in largestGlobals -- so remove this entry, then flag the current axes
						// for insertion.
						(void) largestGlobals.erase( existing );
						insert = true;
						// By induction, furthermore, there are no other entries in largestGlobals
						// that could contain the current tensor. So we terminate the check as
						// well.
						break;
					}
				} else {
					bool smaller = true;
					for( const unsigned int &axis : current ) {
						if( existing.find( axis ) == existing.cend() ) {
							// check if the differing axis is a dynamic one
							if( getIteratedAxes().find( axis ) != getIteratedAxes().cend() ) {
								differingDynamicAxesPresent = true;
							}
							// in this case, current is not a subset of this entry in largestGlobals,
							// so we cannot that current is smaller-- check next one
							smaller = false;
							break;
						}
					}
					if( smaller ) {
						// in this case, current is a subset of this entry in largestGlobals, and
						// so we can ignore the current tensor and move to the next one
						insert = false;
						// for allowing the analytic model to compute the exact buffer usage, we
						// still record the tensor
						minorTensors.push_back( tensor );
						break;
					}
				}
			}
			if( insert ) {
				std::set< int > tempSet( current.cbegin(), current.cend() );
				(void) largestGlobals.insert( tempSet );
			}
		}
	}

	// start codegen: constructor
	os << "\tasc::AnalyticModel< " << igrid->getProcessOrder() << ", "
		<< igrid->getProblemOrder() << ", "
		<< (differingDynamicAxesPresent ? "true" : "false")
		<< " > am( " << ub_size << ", { ";
	os << "_" << igrid->processSize( 0 );
	for( size_t i = 1; i < igrid->getProcessOrder(); ++i ) {
		os << ", _" << igrid->processSize( i );
	}
	os << " }, { ";
	os << "_" << igrid->problemSize( 0 );
	for( size_t i = 1; i < igrid->getProblemOrder(); ++i ) {
		os << ", _" << igrid->problemSize( i );
	}
	os << " }, { ";
	{
		const auto &axes = getIteratedAxes();
		if( axes.find( 0 ) != axes.cend() ) {
			os << "true";
		} else {
			os << "false";
		}
		for( size_t i = 1; i < igrid->getProblemOrder(); ++i ) {
			if( axes.find( i ) != axes.cend() ) {
				os << ", true";
			} else {
				os << ", false";
			}
		}
	}
	os << " } );\n";

	// add minor tensors
	for( const auto &tensor : minorTensors ) {
		const auto &current = tensor.getAxes();
		os << "\tam.addMinorTensor( sizeof( "
			<< internal::getDataType( tensor.getType() )
			<< " ), { ";
		if( std::find( current.cbegin(), current.cend(), 0 ) == current.cend() ) {
			os << "false";
		} else {
			os << "true";
		}
		for( size_t i = 1; i < igrid->getProblemOrder(); ++i ) {
			if( std::find( current.cbegin(), current.cend(), i ) == current.end() ) {
				os << ", false";
			} else {
				os << ", true";
			}
		}
		os << " } );\n";
	}

	// add global non-minor tensors
	for( const auto &tensor : accessed ) {
		const auto &axes = tensor.getAxes();
		std::set< int > tempSet( axes.cbegin(), axes.cend() ); // TODO FIXME not the most performant code
		if( tensor.getScope() != internal::Scope::GLOBAL ) { continue; }
		if( largestGlobals.find( tempSet ) == largestGlobals.cend() ) { continue; }
		assert( axes.size() > 0 );
		os << "\tam.addGlobalTensor( sizeof( "
			<< internal::getDataType( tensor.getType() )
			<< " ), { ";
		size_t k = 0;
		if( std::find( axes.cbegin(), axes.cend(), 0 ) != axes.cend() ) {
			os << "true";
		} else {
			os << "false";
		}
		(void) ++k;
		for( ; k < igrid->getProblemOrder(); ++k ) {
			if( std::find( axes.cbegin(), axes.cend(), k ) != axes.cend() ) {
				os << ", false";
			} else {
				os << ", true";
			}
		}
		os << " } );\n";
	}

	// add buffers
	for( const auto &tensor : accessed ) {
		if( tensor.getScope() != internal::Scope::GLOBAL ) {
			const auto &axes = tensor.getAxes();
			os << "\tam.addBuffer( sizeof( "
				<< internal::getDataType( tensor.getType() )
				<< " ), { ";
			if( std::find( axes.cbegin(), axes.cend(), 0 ) == axes.cend() ) {
				os << "false";
			} else {
				os << "true";
			}
			for( size_t i = 1; i < igrid->getProblemOrder(); ++i ) {
				if( std::find( axes.cbegin(), axes.cend(), i ) == axes.cend() ) {
					os << ", false";
				} else {
					os << ", true";
				}
			}
			os << " } );\n";
		}
	}

	// add stages
	// TODO ideally, all AscendC functions have a unique identifier and an array of
	//      those are passed to the analytic model. For now, we just transfer a
	//      count instead.
	os << "\tam.setNumStages( " << stages.size() << " );\n";

	// Now, finally, the analytic model has all info it needs -- get the blocksizes!
	for( auto axes : getIteratedAxes() ) {
		os << "\tconst uint32_t _tile_size" << axes << " = am.getBlockSize( "
			<< axes << " );\n";
	}
	os << "\n";


	// done: move this to the host
	// for( auto axes : getIteratedAxes() ) {
	// 	os << "\tconst uint32_t _tile_size" << axes << " = 1;\n";
	// }
	// os << "\n";

	analyticModelConstrBody << "\n";
	for( const auto &axes : getIteratedAxes() ) {
		analyticModelConstrBody << "\t\t\ttile_size" << axes << " = _tile_size" << axes << ";\n";
		analyticModelFormalParams << ", const uint32_t _tile_size" << axes;
		analyticModelDecls << "\t\tuint32_t tile_size" << axes << ";\n\n";
		analyticModelArgs << ", _tile_size" << axes;
	}
	// end analytic model code block
}

void alp::internal::AscendPipeline::generateInit( std::stringstream &init ) {

	for( auto it = accessed.cbegin(); it != accessed.cend(); ++it ) {
		if( it->isGlobalDecl() ) {

			assert( it->getAxes().size() > 0 );

			// n0 * n1 * n2 * n3 ...
			std::string set_numerator( igrid->problemSize( *( it->getAxes().begin() ) ) );
			// p0 * p1 * p2 * n3 ...
			std::string set_denominator( igrid->processSize( *( it->getAxes().begin() ) ) );
			for( auto jt = ++it->getAxes().begin(); jt != it->getAxes().end(); ++jt ) {

				set_numerator.append( " * " + igrid->problemSize( *jt ) );
				set_denominator.append( " * " + igrid->processSize( *jt ) );
			}

			// n2 * n3 ... (e.g., n0 and n1 are excluded since they are the loop axes)
			std::string non_parallel_init_numerator;

			for( auto jt = stages.cbegin(); jt != stages.cend(); ++jt ) {

				if( jt->getOpType() == internal::Stagetype::GET_VIEW && jt->getTensor0().getID() == it->getID() ) {

					bool first = true;
					for( auto kt = it->getAxes().cbegin(); kt != it->getAxes().cend(); ++kt ) {
						if( std::find( jt->getForEachAxes().begin(), jt->getForEachAxes().end(), ( int ) *kt ) == jt->getForEachAxes().end() ) {
							if( !first ) {
								non_parallel_init_numerator.append( " * " );
							}
							non_parallel_init_numerator.append( igrid->problemSize( *kt ) );
							first = false;
						}
					}
					break;
				}
			}

			if( non_parallel_init_numerator.empty() ) {
				non_parallel_init_numerator.assign( "1" );
			}

			std::string tiling_init_numerator = getTilingAxes();

			init << "\n";
			init << "\t\t\t// Initializing data for a Global Tensor\n";
			init << "\t\t\t" << it->getAscendGlobalName( id )
					<< ".SetGlobalBuffer( ( __gm__ "
					<< internal::getDataType( it->getType() ) << " * )"
					<< it->getName() << " + ( " << set_numerator << " ) / ( " << set_denominator << " ) * GetBlockIdx(), ( "
					<< set_numerator << " ) / ( " << set_denominator << " ) );\n";
			init << "\t\t\tpipe.InitBuffer( " << it->getTQueBufName( id ) << ", BUFFER_NUM, "
					<< tiling_init_numerator
					<< "( ( " << non_parallel_init_numerator << " ) / BUFFER_NUM ) * sizeof( "
					<< internal::getDataType( it->getType() ) << " ) );\n";
		}
	}
/*
	//TODO these two loops can be fused, the only reason are written that way
	//		was to solve or avoid a bug regarding the order of the init
	for( auto it = accessed.cbegin(); it != accessed.cend(); ++it ) {
		if( it->isTempDecl() ) {
			init << "\n";
			init << "\t\t\t// Initializing data for a temporary Tensor\n";
			init << "\t\t\tpipe.InitBuffer( " << it->getTQueBufName( id )
				<< ", " << "totWorkSpaceSize" << " );\n";
			init << "\t\t\t" << it->getAscendName( id ) << " = "
				<< it->getTQueBufName( id )
				<< ".Get< " << internal::getDataType( it->getType() ) << " >();\n";
		}
	}

	for( auto it = accessed.cbegin(); it != accessed.cend(); ++it ) {
		if( it->isLocalDecl() ) {
			init << "\n";
			//TODO fix that
			std::vector< int > forEachParallelAxes = internal::vectorOfVectorsToVector( internal::OpGen::forEachAxes );
			const std::vector< int > &axes = it->getAxes();
			std::vector< int > local_iterated_axes;

			//TODO: is this set correct? is it necessary, if yes, perhaps a sort is required
			local_iterated_axes = internal::vectorDifference( axes, forEachParallelAxes );

			std::string product_dim("");
			bool first = true;
			for( auto it = local_iterated_axes.cbegin(); it != local_iterated_axes.cend(); ++it ) {
				if( first == true ) {
					first = false;
				} else {
					product_dim.append(" * ");
				}
				product_dim.append( igrid->problemSize( *it ) );
			}

			if( product_dim.empty() == true ) {
				product_dim.append( "1" );
			}
			init << "\t\t\t// Initializing data for a local Tensor\n";
			init << "\t\t\tpipe.InitBuffer( " << it->getTQueBufName( id )
				<< ", " << product_dim << " );\n";
			init << "\t\t\t" << it->getAscendName( id ) << " = "
				<< it->getTQueBufName( id )
				<< ".Get< " << internal::getDataType( it->getType() ) << " >();\n";
		}
	}
*/
	init << "\n";

	std::string prev("totWorkSpaceSize");
	std::string prev_dim("");
	for( auto it = accessed.cbegin(); it != accessed.cend(); ++it ) {
		if( it->isLocalDecl() || it->isTempDecl() ) {
			init << "\t\t\t" << it->getAscendName( id ) << " = " << prev << ( prev_dim.empty() ? ";\n" : ( " + " + prev_dim + ";\n" ) );

			if( it->getAxes().size() > 0 ) {
				// n0 * n1 * n2 ...
				std::string set_numerator( igrid->problemSize( *( it->getAxes().begin() ) ) );
				for( auto jt = ++it->getAxes().begin(); jt != it->getAxes().end(); ++jt ) {

					set_numerator.append( " * " + igrid->problemSize( *jt ) );
				}
				prev_dim = set_numerator;
			} else {
				prev_dim = "16";
			}
			prev = it->getAscendName( id );
		}
	}
}

void alp::internal::AscendPipeline::generateProcess( std::stringstream &process,
													 std::stringstream &processCall ) {

	processCall << "\t\t\tProcess" << id << "();\n";

	// generate the Process function
	// TODO here we should use the grid info and symbolic analytic model

	process << "\n";
	process << "\t\t__aicore__ inline void Process" << id << "() {\n";
	process << "\n";

	std::string tabs("");

	// use a stack to keep track of the for loops that are already generated
	std::vector< int > stack;
//	std::vector< std::pair< std::string, std::pair< std::string, std::string > > > tiling_stack;

	int parallel_axe = *( stages.cbegin()->getForEachAxes().begin() );
	// initialize the stack with the axe of the outer forEach
	// which is the parallel loop and thus can be omitted
	stack.push_back(  parallel_axe );

//	bool new_nested_level = true;

	// declare variables for the upper bound of the extra loops that are introduced
	std::set< int > iterated_axes = getIteratedAxes();
	for( auto it = iterated_axes.cbegin(); it != iterated_axes.cend(); ++it ) {
		process << tabs << "\t\t\tuint32_t upper_" << igrid->problemTileMode( *it ) << ";\n";
	}

	process << "\n";

	process << tabs << "\t\t\tfor( uint32_t " << igrid->problemMainMode( parallel_axe )
			<< " = 0; " << igrid->problemMainMode( parallel_axe ) << " < ( "
			<< igrid->problemSize( parallel_axe ) << " / " << igrid->processSize( parallel_axe ) << " ); "
			<< igrid->problemMainMode( parallel_axe ) << " += tile_size" << parallel_axe << " ) {\n";

	tabs.append("\t");

/*	std::stringstream tiling_loop, tiling_condition, tiling_var;

	tiling_condition.str("");
	tiling_condition << "\t\t\tupper_" << igrid->problemTileMode( parallel_axe ) << " = ( "
					 << "( " << igrid->problemSize( parallel_axe ) << " / " << igrid->processSize( parallel_axe )
					 << " ) < ( " << igrid->problemMainMode( parallel_axe ) << " + tile_size" << parallel_axe << " ) ) ? "
					 << "( (" << igrid->problemSize( parallel_axe ) << " / " << igrid->processSize( parallel_axe ) << " ) - "
					 << igrid->problemMainMode( parallel_axe ) << " ) : ( tile_size" << parallel_axe << " );\n";

	// the tiling loop is not added in the stack of generated loops
	tiling_loop.str("");
	tiling_loop << "\t\t\tfor( uint32_t " << igrid->problemTileMode( parallel_axe )
				<< " = " << "0" << "; " << igrid->problemTileMode( parallel_axe )
				<< " < upper_" << igrid->problemTileMode( parallel_axe ) << "; " << igrid->problemTileMode( parallel_axe ) << "++ ) {\n";

	tiling_var.str("");
	tiling_var << "\t\t\t\tconst uint32_t " << igrid->problemMode( parallel_axe ) << " = " << igrid->problemMainMode( parallel_axe )
			   << " + " << igrid->problemTileMode( parallel_axe ) << ";\n";


	tiling_stack.push_back( std::make_pair( tiling_condition.str(), std::make_pair( tiling_loop.str(), tiling_var.str() ) ) );
*/
	std::vector< int > prev_stage_axes;

	// generate AscendC code for the operators of the pipeline
	for( auto it = stages.cbegin(); it != stages.cend(); ++it ) {

		// get the axes of the current stage
		const std::vector< int > &forEachAxes = it->getForEachAxes();

		// iterator of the stack
		auto st = stack.begin();
		// iterator of the axes for the current stage
		auto at = forEachAxes.begin();

		// the number of axes that are currently in the stack
		// and match the corresponding axes of the current stage
		size_t match_axes = 0;

		// iterate over all the axes of the stack that match
		// those of the stage, which implies that if the current
		// stage goes into the current for loop, i.e., no loop
		// needs to be created and no loop needs to be closed,
		// all the axes should match
		while( st != stack.end() ) {

			if( at == forEachAxes.end() || *st != *at ) {
				break;
			}

			++match_axes;
			++st;
			++at;
		}

		// if there was a mismatch on the axes between the
		// already generated loops (stack) and the axes of the stage
		// then the axes of the stack that do not match should be popped
		// which implies that the generated loops should close
		size_t to_pop_axes = stack.size() - match_axes;
/*
		if( to_pop_axes > 0 ) {
			// close the loops for tiling first
			for( auto jt = tiling_stack.begin(); jt != tiling_stack.end(); ++jt ) {
				process << tabs << "\t\t}\n";
				tabs.pop_back();
			}
		}
*/
		for( size_t i = 0; i < to_pop_axes; ++i ) {

//			tiling_stack.pop_back();

			process << tabs << "\t\t}\n";
			stack.pop_back();
			tabs.pop_back();
		}
/*
		// generate tiling loops if at least one loop of axes was closed
		if( to_pop_axes > 0 ) {
			for( auto jt = tiling_stack.begin(); jt != tiling_stack.end(); ++jt ) {
				process << "\n";
				process << tabs << jt->first;
				process << tabs << jt->second.first;
				process << "\n";
				process << tabs << jt->second.second;
				tabs.append("\t");
			}
		}
*/
		// iterator of the stack
		st = stack.begin();
		// iterator of the axes for the current stage
		at = forEachAxes.begin();

		// iterate over all the axes of the stage as long as the
		// corresponding axes are already in the stack, which implies
		// the for loops are already generated
		while( at != forEachAxes.end() ) {

			if( st == stack.end() ) {
				break;
			}

			// as long as the end of the stack was not reached
			// the axes should match those of the current stage
			// since all the elements did not match were popped
			assert( *st != *at );

			++st;
			++at;
		}
/*
		// close tiling loops provides that
		// a) no loop was already closed, otherwise the corresponding loops are closed
		// b) this is not the first stage
		// c) the axes of the previous stage are different than those of the current stage
		//		a situation that indicates these two stages are not nested in the same level
		if( to_pop_axes == 0 && it != stages.cbegin() && prev_stage_axes != forEachAxes ) {
			for( auto jt = tiling_stack.begin(); jt != tiling_stack.end(); ++jt ) {
				process << tabs << "\t\t}\n";
				tabs.pop_back();
			}

		}
*/
		// iterate over the rest of the axes of the stage, i.e., those
		// that are not included in the stack and lead to generation of for loops
		while( at != forEachAxes.end() ) {

//			new_nested_level = true;

			process << "\n";

			process << tabs << "\t\t\tfor( uint32_t " << igrid->problemMainMode( *at )
					<< " = 0; " << igrid->problemMainMode( *at ) << " < "
					<< igrid->problemSize( *at ) << "; "
					<< igrid->problemMainMode( *at ) << " += tile_size" << *at << " ) {\n";

			tabs.append("\t");
			stack.push_back( *at );
/*
			tiling_condition.str("");
			tiling_condition << "\t\t\tupper_" << igrid->problemTileMode( *at ) << " = ( "
							 << igrid->problemSize( *at ) << " < ( " << igrid->problemMainMode( *at ) << " + tile_size" << *at << " ) ) ? "
							 << igrid->problemSize( *at ) << " - " << igrid->problemMainMode( *at ) << " : ( tile_size" << *at << " );\n";

			// the tiling loop is not added in the stack of generated loops
			tiling_loop.str("");
			tiling_loop << "\t\t\tfor( uint32_t " << igrid->problemTileMode( *at )
						<< " = " << "0" << "; " << igrid->problemTileMode( *at ) << " < upper_"
						<< igrid->problemTileMode( *at ) << "; " << igrid->problemTileMode( *at ) << "++ ) {\n";

			tiling_var.str("");
			tiling_var << "\t\t\t\tconst uint32_t " << igrid->problemMode( *at ) << " = " << igrid->problemMainMode( *at )
					   << " + " << igrid->problemTileMode( *at ) << ";\n";

			tiling_stack.push_back( std::make_pair( tiling_condition.str(), std::make_pair( tiling_loop.str(), tiling_var.str() ) ) );
*/ 			++at;
		}
/*
		if( new_nested_level ) {
			for( auto jt = tiling_stack.begin(); jt != tiling_stack.end(); ++jt ) {
				process << "\n";
				process << tabs << jt->first;
				process << tabs << jt->second.first;
				process << "\n";
				process << tabs << jt->second.second;
				tabs.append("\t");
			}
		}
*/
		process << "\n";
		process << it->getOp( tabs );

		// reset the flag to false
//		new_nested_level = false;

		// set the axes of the previous stage to those of the current one
		prev_stage_axes = forEachAxes;
	}
/*
	if( stack.size() > 0 ) {
		// before closing a loop, all the generated loops for tiling should close as well
		for( auto jt = tiling_stack.begin(); jt != tiling_stack.end(); ++jt ) {
			process << tabs << "\t\t}\n";
			tabs.pop_back();
		}
	}
*/
	// close all the generated for loops
	// starting from 0 to generate the parallel/outer loop
	// starting from 1 if the outer parallel/loop is not generated
	for( size_t i = 0; i < stack.size(); ++i ) {

		process << tabs << "\t\t}\n";
		tabs.pop_back();
	}

	// the curly bracket for the process function
	process << "\t\t}\n";
}

void alp::internal::AscendPipeline::debug_print() const {

	std::cerr << "ACCESSED: ";
	for (auto it = accessed.cbegin(); it != accessed.cend(); ++it ) {
		std::cerr << it->getName() << ", ";
	}

	std::cerr << std::endl << std::endl << std::endl;
}
