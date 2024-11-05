#include <graphblas/ascend/utils.hpp>
#include <graphblas/ascend/tensor.hpp>
//#include <graphblas/ascend/grid.hpp>
#include <graphblas/ascend/opgen.hpp>		//TODO forEachLevel
#include <graphblas/ascend/symbolTable.hpp>
#include <graphblas/ascend/operators.hpp>


namespace alp
{
	namespace internal
	{
		extern SymbolTable symbols;
	}
}

size_t alp::Tensor::tensor_id = 0;

bool alp::Tensor::operator==( const Tensor &t ) const {
	return this->name == t.name;
}

void alp::Tensor::operator=( const ReductionOperation& op ) {
	foldl( *this, op.input, op.opName, op.axes );
}

void alp::Tensor::operator=( const ApplyOperation& op ) {
	apply( *this, op.input1, op.input2, op.opName, op.axes );
}

alp::Tensor::Tensor( const Datatype _type, const std::vector< int > &_axes ) noexcept
			: id( tensor_id++ ),
			  name( std::string("tensor") + std::to_string( id ) ),
			  type( _type ),
			  scope( internal::OpGen::forEachLevel > 0 ? internal::Scope::LOCAL : internal::Scope::GLOBAL ),
			  axes( _axes ) {
	if( internal::OpGen::forEachLevel > 0 ) {
		internal::symbols.addLocalTensor( *this );
	} else {
		/*
		for( auto it = axes.begin(); it != axes.end(); ++it ) {
			if( it != axes.begin() ) {
				internal::OpGen::output_host_log << ",";
			}
			internal::OpGen::output_host_log << *it;
		}
		*/
		internal::symbols.addGlobalTensor( *this );
	}
}

alp::Tensor::Tensor( const Tensor&parent, const std::vector< int > &_axes ) noexcept
			: id( tensor_id++ ),
			  name( "view_" + std::to_string( id ) + "_of_" + parent.getName() ),
			  type( parent.getType() ),
			  scope( internal::Scope::VIEW ),
			  axes( _axes ) {
		// TODO Is it okay to have a view with empty Axes?
		internal::symbols.addTensorView( name, parent.getName() );
}

alp::Tensor::Tensor( const Tensor &t ) noexcept
			: id( t.id ),
			  name( t.name ),
			  type( t.type ),
			  scope( t.scope ),
			  axes( t.axes ) {

}

alp::Tensor::Tensor( const std::vector< int > &_axes, const Datatype _type ) noexcept
			: id( tensor_id++ ),
			  name( std::string("tensor") + std::to_string( id ) ),
			  type( _type ),
			  scope( internal::Scope::TEMP ),
			  axes( _axes ) {
	internal::symbols.addTempTensor( *this );
}

size_t alp::Tensor::getID() const {
	return id;
}

const std::string &alp::Tensor::getName() const {
	return name;
}

alp::Datatype alp::Tensor::getType() const {
	return type;
}

alp::internal::Scope alp::Tensor::getScope() const {
	return scope;
}

const std::vector< int > &alp::Tensor::getAxes() const {
	return axes;
}

bool alp::Tensor::isGlobalDecl() const {

	const Tensor tensor = internal::symbols.getTensorFromView( *this );

	return tensor.scope == internal::Scope::GLOBAL;
}

bool alp::Tensor::isLocalDecl() const {
	return scope == internal::Scope::LOCAL;
}

bool alp::Tensor::isTempDecl() const {
	return scope == internal::Scope::TEMP;
}

std::string alp::Tensor::getAccessedElement( size_t id ) const {

	// if this tensor is a view, find its parent tensor
	const Tensor tensor = internal::symbols.getTensorFromView( *this );

	// make a decision based on the scope of the parent tensor
	switch( tensor.scope ) {
		case internal::Scope::GLOBAL:
			return "Gm_local_" + tensor.name + "_" + std::to_string( id );
		case internal::Scope::LOCAL:
			return internal::getDataType( type ) + "_temp_local[ local_" + tensor.name + "_" + std::to_string( id ) + " ]";
		case internal::Scope::TEMP:
			return internal::getDataType( type ) + "_temp_local[ temp_" + tensor.name + "_" + std::to_string( id ) + " ]";
		case internal::Scope::VIEW:
		default:
			std::cerr << "ERROR in the declaration " << name << " of getAccessedElement" << std::endl;
			std::abort();
			break;
	}
}

std::string alp::Tensor::getAscendName( size_t id ) const {

	switch( scope ) {
		case internal::Scope::GLOBAL:
			return "Gm_local_" + name + "_" + std::to_string( id );
		case internal::Scope::LOCAL:
			return "local_" + name + "_" + std::to_string( id );
		case internal::Scope::TEMP:
			return "temp_" + name + "_" + std::to_string( id );
		case internal::Scope::VIEW:
		default:
			std::cerr << "ERROR in the symbol table, the declaration " << name << " was not found" << std::endl;
			std::abort();
	}
}

std::string alp::Tensor::getAscendGlobalName( size_t id ) const {

	switch( scope ) {
		case internal::Scope::GLOBAL:
			return "Gm_" + name + "_" + std::to_string( id );
		case internal::Scope::LOCAL:
		case internal::Scope::TEMP:
		case internal::Scope::VIEW:
		default:
			std::cerr << "ERROR: declaration " << name << " is not global" << std::endl;
			std::abort();

	}
}

std::string alp::Tensor::getTQueBufName( size_t id ) const {

	switch( scope ) {
		case internal::Scope::GLOBAL:
			return "globalQue_" + name + "_" + std::to_string( id );
		case internal::Scope::LOCAL:
			return "localBuf_" + name + "_" + std::to_string( id );
		case internal::Scope::TEMP:
			return "tempBuf_" + name + "_" + std::to_string( id );
		case internal::Scope::VIEW:
		default:
			std::cerr << "ERROR in the declaration " << name << " of getTQueBufName" << std::endl;
			std::abort();
			break;
	}
}

