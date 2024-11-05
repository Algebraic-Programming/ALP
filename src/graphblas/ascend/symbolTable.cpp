#include <graphblas/ascend/symbolTable.hpp>
#include <graphblas/ascend/tensor.hpp>
#include <graphblas/ascend/semantics.hpp>
#include <graphblas/ascend/grid.hpp>

namespace alp
{
	namespace internal
	{
		extern iGrid *igrid;
		SymbolTable symbols;
	}
}

alp::internal::SymbolTable::SymbolTable() {

	TBuf_decl = false;
	temp_scalar_id = 0;
}

bool alp::internal::SymbolTable::existsTBufTensorDecl() const {

	return TBuf_decl;
}

void alp::internal::SymbolTable::clearAll() {

	global_tensor_declarations.clear();
	local_tensor_declarations.clear();
	temp_tensor_declarations.clear();

	// assuming that views are created locally in a forEach
	// or is it possible to have views in a global scope?
	viewToTensor.clear();
}

void alp::internal::SymbolTable::addGlobalTensor( const alp::Tensor &t ) {

	// TODO this semantics check is essentially unnecessary
	// since global Tensors are not declared within forEach
	if( internal::invalidAxes( t.getAxes() ) == true ) {
		std::cerr << "The axes of the global Tensor must not be included in the axes of the forEach." << std::endl;
		std::abort();
	}

	global_tensor_declarations.emplace( t.getName() , t );

	all_global_tensors.emplace_back( t );
}

void alp::internal::SymbolTable::addLocalTensor( const alp::Tensor &t ) {

	if( internal::invalidAxes( t.getAxes() ) == true ) {
		std::cerr << "The axes of the local Tensor must not be included in the axes of the forEach." << std::endl;
		std::abort();
	}

	TBuf_decl = true;
	local_tensor_declarations.emplace( t.getName(), t );

	reuseLocalTempTensorBuffer( t );
}

void alp::internal::SymbolTable::addTempTensor( const alp::Tensor &t ) {

	// TODO this semantics check is essentially unnecessary
	// since temporary Tensors are declared internally
	if( internal::invalidAxes( t.getAxes() ) == true ) {
		std::cerr << "The axes of the temporary Tensor must not be included in the axes of the forEach." << std::endl;
		std::abort();
	}

	TBuf_decl = true;
	temp_tensor_declarations.emplace( t.getName(), t );

	reuseLocalTempTensorBuffer( t );
}

void alp::internal::SymbolTable::addTensorView( const std::string &view_name, const std::string &parent_name ) {

	viewToTensor[ view_name ] = parent_name;
}

/*
std::string alp::internal::SymbolTable::newTempScalar() {

	return "temp_scalar_" + std::to_string( temp_scalar_id++ );
}
*/

void alp::internal::SymbolTable::addOutputTensor( const alp::Tensor &t ) {

	outputs_global_tensors.emplace_back( t );
}

void alp::internal::SymbolTable::printHostLogFile( std::stringstream &listOfGlobalTensors ) {

	bool first = true;

	for( auto it = all_global_tensors.begin(); it != all_global_tensors.end(); ++it ) {

		std::vector< int > axes = it->getAxes();
		for( auto jt = axes.begin(); jt != axes.end(); ++jt ) {

			if( first == true ) {
				first = false;
			} else {
				listOfGlobalTensors << ",";
			}
			listOfGlobalTensors << *jt;
		}
		if ( std::find( outputs_global_tensors.cbegin(), outputs_global_tensors.cend(), *it ) == outputs_global_tensors.cend() ) {
			listOfGlobalTensors << ",in";
		} else {
			listOfGlobalTensors << ",out";
		}
	}
}

void alp::internal::SymbolTable::generateGlobalSymbols( std::stringstream &initFormalParam,
								std::stringstream &customFormalParam, std::stringstream &allAccessedArg,
								std::stringstream &allTempLocalDecl ) const {

	for( auto it = global_tensor_declarations.cbegin(); it != global_tensor_declarations.cend(); ++it ) {
		if( it->first != global_tensor_declarations.cbegin()->first ) {
			initFormalParam  << ", ";
			customFormalParam  << ", ";
			allAccessedArg  << ", ";
		}
		initFormalParam << "GM_ADDR " << it->first;
		// TODO data type needs to be parametrised
		// TODO or MAYBE NOT?
		customFormalParam << "uint8_t" << " * " << it->first;
		allAccessedArg << it->first;
	}

	for( auto it = temp_local_buffer_declarations.begin(); it != temp_local_buffer_declarations.end(); ++it ) {
		allTempLocalDecl << "\t\t// Declaration of memory used for Local and Temporary tensor\n";
		allTempLocalDecl << "\t\tTBuf< QuePosition::VECCALC > " << it->first << "_temp_local_Buf;\n";
		allTempLocalDecl << "\t\tLocalTensor< " << it->first << " > " << it->first << "_temp_local;\n";
		allTempLocalDecl << "\n";
	}
}

void alp::internal::SymbolTable::generateTempLocalInit( std::stringstream &allTempLocalInit ) const {

	for( auto it = temp_local_buffer_declarations.begin(); it != temp_local_buffer_declarations.end(); ++it ) {
		allTempLocalInit << "\n";
		allTempLocalInit << "\t\t\t// Initialization of memory used for Local and Temporary tensor\n";
		allTempLocalInit << "\t\t\tpipe.InitBuffer( " << it->first
						 << "_temp_local_Buf, ( totWorkSpaceSize + " << it->second << " ) * sizeof( " << it->first << " ) );\n";
		allTempLocalInit << "\t\t\t" << it->first << "_temp_local = "
						 << it->first << "_temp_local_Buf.Get< " << it->first << " >();\n";
	}
}

const alp::Tensor &alp::internal::SymbolTable::getTensorFromView( const alp::Tensor &tensor ) const {

	auto it = viewToTensor.find( tensor.getName() );
	// TODO: assume we have only one level of views, otherwise a loop is required
	if( it != viewToTensor.cend() ) {
		auto jt = global_tensor_declarations.find( it->second );
		if( jt != global_tensor_declarations.cend() ) {
			return jt->second;
		} else {
            std::cerr << "Cannot handle a view of a non-global declaration" << std::endl;
            std::abort();
		}
	} else {
		return tensor;
	}
}

std::string alp::internal::SymbolTable::getLocalTempTensorBuffer( Datatype type, const std::string &size ) {

	std::string datatype = internal::getDataType( type );

	auto it = temp_local_buffer_declarations.find( datatype );
	if( it == temp_local_buffer_declarations.cend() ) {
		temp_local_buffer_declarations.emplace( datatype, size );
	} else if ( size.empty() == false ) {
		it->second.append( std::string( " + " ) + std::string( size ) );
	}
	return datatype + "_temp_local";
}

void alp::internal::SymbolTable::reuseLocalTempTensorBuffer( const alp::Tensor &t ) {

	std::string datatype = internal::getDataType( t.getType() );
	const std::vector< int > &axes = t.getAxes();

	assert( axes.size() < 3 );

	std::string size;
	if( axes.size() == 0 ) {
		size = "( 32 / sizeof( ";
		size.append( datatype );
		size.append( " ) )" );
	} else if( axes.size() == 1 ) {
		size = igrid->problemSize( axes[ 0 ] );
	} else if( axes.size() == 2) {
		size = igrid->problemSize( axes[ 0 ] ) + " * " + igrid->problemSize( axes[ 1 ] );
	}
/*
	auto it = temp_local_buffer_declarations.find( datatype );
	if( it == temp_local_buffer_declarations.cend() ) {
		temp_local_buffer_declarations.emplace( datatype, size );
	} else {
		it->second.append( std::string( " + " ) + std::string( size ) );
	}
*/
	( void ) getLocalTempTensorBuffer( t.getType(), size );
}

void alp::internal::SymbolTable::debug_print() const {

	std::cerr << "\nGLOBAL: ";
	for( auto it = global_tensor_declarations.cbegin(); it != global_tensor_declarations.cend(); ++it ) {
		std::cerr << it->first << "(" << alp::internal::getScope( it->second.getScope() ) << "), ";

	}

	std::cerr << "\nLOCAL: ";
	for( auto it = local_tensor_declarations.cbegin(); it != local_tensor_declarations.cend(); ++it ) {
		std::cerr << it->first << "(" << alp::internal::getScope( it->second.getScope() ) << "), ";
	}

	std::cerr << "\nTEMP: ";
	for( auto it = temp_tensor_declarations.cbegin(); it != temp_tensor_declarations.cend(); ++it ) {
		std::cerr << it->first << "(" << alp::internal::getScope( it->second.getScope() ) << "), ";
	}

	std::cerr << "\nVIEW: ";
	for( auto it = viewToTensor.cbegin(); it != viewToTensor.cend(); ++it ) {
		std::cerr << it->first << "( of " << it->second << "), ";

	}

	std::cerr << std::endl << std::endl << std::endl;
}
