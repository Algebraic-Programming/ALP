
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

/*
 * @author A. N. Yzelman
 * @date 1st of February, 2022
 */

#include <graphblas/hyperdags/hyperdags.hpp>

std::string grb::internal::hyperdags::toString(
	const enum SourceVertexType type
) noexcept {
	switch( type ) {

		case SCALAR:
			return "input scalar";

		case CONTAINER:
			return "ALP/GraphBLAS container";

		case ITERATOR:
			return "input iterator";

		case USER_INT:
			return "input integer";

	}
	assert( false );
	return "unidentified source vertex type";
}

std::string grb::internal::hyperdags::toString(
	const enum OutputVertexType type
) noexcept {
	switch( type ) {

		case CONTAINER_OUTPUT:
			return "output container";

	}
	assert( false );
	return "unidentified output vertex type";
}

std::string grb::internal::hyperdags::toString(
	const enum OperationVertexType type
) noexcept {
	switch( type ) {

		case NNZ_VECTOR:
			return "nnz( vector )";

		case NNZ_MATRIX:
			return "nnz( matrix )";

		case CLEAR_VECTOR:
			return "clear( vector )";

		case SET_VECTOR_ELEMENT:
			return "setElement( vector )";

		case DOT:
			return "dot( scalar, vector, vector )";

		case SET_USING_VALUE:
			return "set( vector, scalar )";

		case SET_USING_MASK_AND_VECTOR:
			return "set( vector, vector, vector )";

		case SET_USING_MASK_AND_SCALAR:
			return "set( vector, vector, scalar )";

		case SET_FROM_VECTOR:
			return "set( vector, vector )";

		case ZIP:
			return "zip( vector, vector, vector )";

		case E_WISE_APPLY_VECTOR_VECTOR_VECTOR_OP:
			return "eWiseApply( vector, vector, vector, op )";

		case FOLDR_VECTOR_SCALAR_MONOID:
			return "foldr( vector, scalar, monoid )";

		case FOLDR_VECTOR_MASK_SCALAR_MONOID:
			return "foldr( vector, vector, scalar, monoid )";

		case FOLDL_SCALAR_VECTOR_MONOID:
			return "foldl( scalar, vector, monoid )";

		case FOLDL_SCALAR_VECTOR_MASK_MONOID:
			return "foldl( calar, vector, vector, monoid )";

		case EWISELAMBDA:
			return "eWiseLambda( f, vector )";

		case BUILD_VECTOR:
			return "buildVector( vector, scalar, scalar, scalar, scalar )";

		case BUILD_VECTOR_WITH_VALUES:
			return "buildVector( vector, scalar, scalar, scalar, scalar, scalar )";

		case SIZE:
			return "size( vector )";

		case NROWS:
			return "nrows( matrix )";

		case NCOLS:
			return "ncols( matrix )";

		case EWISEAPPLY_VECTOR_ALPHA_BETA_OP:
			return "eWiseApply( vector, scalar, scalar, operation)";

		case EWISEAPPLY_VECTOR_ALPHA_VECTOR_OP:
			return "eWiseApply( vector, scalar, vector, operation)";

		case EWISEAPPLY_VECTOR_VECTOR_BETA_OP:
			return "eWiseApply( vector, vector, scalar, operation)";

		case EWISEAPPLY_VECTOR_VECTOR_VECTOR_OP:
			return "eWiseApply( vector, vector, vector, operation)";

		case EWISEAPPLY_VECTOR_MASK_ALPHA_BETA_OP:
			return "eWiseApply( vector, vector, scalar, scalar, operation)";

		case EWISEAPPLY_VECTOR_MASK_ALPHA_VECTOR_OP:
			return "eWiseApply( vector, vector, scalar, vector, operation)";

		case EWISEAPPLY_VECTOR_MASK_VECTOR_BETA_OP:
			return "eWiseApply( vector, vector, vector, scalar, operation)";

		case EWISEAPPLY_VECTOR_MASK_VECTOR_VECTOR_OP:
			return "eWiseApply( vector, vector, vector, vector, operation)";

		case EWISEAPPLY_VECTOR_ALPHA_BETA_MONOID:
			return "eWiseApply( vector, scalar, scalar, monoid)";

		case EWISEAPPLY_VECTOR_ALPHA_VECTOR_MONOID:
			return "eWiseApply( vector, scalar, vector, monoid)";

		case EWISEAPPLY_VECTOR_VECTOR_BETA_MONOID:
			return "eWiseApply( vector, vector, scalar, monoid)";

		case EWISEAPPLY_VECTOR_VECTOR_VECTOR_MONOID:
			return "eWiseApply( vector, vector, vector, monoid)";

		case EWISEAPPLY_VECTOR_MASK_ALPHA_BETA_MONOID:
			return "eWiseApply( vector, vector, scalar, scalar, monoid)";

		case EWISEAPPLY_VECTOR_MASK_ALPHA_VECTOR_MONOID:
			return "eWiseApply( vector, vector, scalar, vector, monoid)";

		case EWISEAPPLY_VECTOR_MASK_VECTOR_BETA_MONOID:
			return "eWiseApply( vector, vector, vector, scalar, monoid)";

		case EWISEAPPLY_VECTOR_MASK_VECTOR_VECTOR_MONOID:
			return "eWiseApply( vector, vector, vector, vector, monoid)";

		case EWISE_MUL_ADD:
			return "eWiseMulAdd( vector, vector, vector, vector, vector, ring )";

		case EWISE_MUL_ADD_FOUR_VECTOR:
			return "eWiseMulAdd( vector, vector, vector, vector, scalar, ring )";

		case EWISE_MUL_ADD_THREE_VECTOR_ALPHA:
			return "eWiseMulAdd( vector, scalar, vector, scalar, ring )";

		case EWISE_MUL_ADD_THREE_VECTOR_CHI:
			return "eWiseMulAdd( vector, vector, scalar, vector, ring )";

		case EWISE_MUL_ADD_FOUR_VECTOR_CHI:
			return "eWiseMulAdd( vector, vector, vector, scalar, vector, ring )";

		case EWISE_MUL_ADD_FOUR_VECTOR_CHI_RING:
			return "eWiseMulAdd( vector, vector, vector, scalar, vector, ring )";

		case EWISE_MUL_ADD_THREE_VECTOR_BETA:
			return "eWiseMulAdd( vector, vector, vector, scalar, scalar, ring )";

		case EWISE_MUL_ADD_THREE_VECTOR_ALPHA_GAMMA:
			return "eWiseMulAdd( vector, vector, vector, scalar, ring )";

		case EWISE_MUL_ADD_TWO_VECTOR_ALPHA_BETA:
			return "eWiseMulAdd( vector, vector, scalar, scalar, vector, ring )";

		case EWISE_MUL_ADD_TWO_VECTOR_ALPHA_BETA_GAMMA:
			return "eWiseMulAdd( vector, vector, scalar, scalar, scalar, ring )";

		case EWISEAPPLY_MATRIX_MATRIX_MATRIX_OPERATOR_PHASE:
			return "eWiseApply( matrix, matrix, matrix, scalar, scalar )";

		case EWISEAPPLY_MATRIX_MATRIX_MATRIX_MULMONOID_PHASE:
			return "eWiseApply( matrix, matrix, matrix, scalar, scalar )";

		case SET_MATRIX_MATRIX:
			return "set( matrix, matrix )";

		case SET_MATRIX_MATRIX_INPUT2:
			return "set( matrix, matrix, scalar )";

		case MXM_MATRIX_MATRIX_MATRIX_MONOID:
			return "mxm( matrix, matrix, matrix, monoid, scalar, scalar )";

		case OUTER:
			return "outer( matrix, vector, vector, scalar, scalar )";

		case MXV_VECTOR_VECTOR_MATRIX_VECTOR_VECTOR_R:
			return "mxv( vector, vector, matrix, vector, vector, ring )";

		case ZIP_MATRIX_VECTOR_VECTOR_VECTOR:
			return "zip( matrix, vector, vector, vector )";

		case ZIP_MATRIX_VECTOR_VECTOR:
			return "zip( matrix, vector, vector )";

		case UNZIP_VECTOR_VECTOR_VECTOR:
			return "unzip( matrix, vector, vector )";

		case EWISEMULADD_VECTOR_VECTOR_VECTOR_GAMMA_RING:
			return "eWiseMulAdd( vector, vector, vector, scalar, ring )";

		case EWISEMULADD_VECTOR_VECTOR_BETA_GAMMA_RING:
			return "eWiseMulAdd( vector, vector, scalar, scalar, ring )";

		case EWISEMULADD_VECTOR_ALPHA_VECTOR_GAMMA_RING:
			return "eWiseMulAdd( vector, vector, scalar, ring )";

		case EWISEMULADD_VECTOR_ALPHA_BETA_VECTOR_RING:
			return "eWiseMulAdd( vector, scalar, scalar, vector, ring )";

		case EWISEMULADD_VECTOR_ALPHA_BETA_GAMMA_RING:
			return "eWiseMulAdd( vector, scalar, scalar, scalar, ring )";

		case EWISEMULADD_VECTOR_VECTOR_VECTOR_VECTOR_RING:
			return "eWiseMulAdd( vector, vector, vector, vector, ring )";

		case VXM_VECTOR_VECTOR_VECTOR_MATRIX:
			return "vxm( vector, vector, vector, matrix, ring )";

		case VXM_VECTOR_VECTOR_VECTOR_MATRIX_ADD_MUL:
			return "vxm( vector, vector, vector, matrix, scalar, scalar )";

		case VXM_VECTOR_VECTOR_MATRIX_RING:
			return "vxm( vector, vector, matrix, ring )";

		case MXV_VECTOR_VECTOR_MATRIX_VECTOR_RING:
			return "mxv( vector, vector, matrix, vector, ring )";

		case MXV_VECTOR_VECTOR_MATRIX_VECTOR_VECTOR_A:
			return "mxv( vector, vector, matrix, vector, vector, scalar, scalar )";

		case MXV_VECTOR_MATRIX_VECTOR_RING:
			return "mxv( vector, matrix, vector, ring )";

		case MXV_VECTOR_MATRIX_VECTOR_ADD_MUL:
			return "mxv( vector, matrix, vector, scalar, scalar )";

		case EWISELAMBDA_FUNC_MATRIX:
			return "eWiseLambda( function, matrix )";

		case VXM_GENERIC_VECTOR_VECTOR_VECTOR_VECTOR_MATRIX_ADD_MUL:
			return "vxm( vector, vector, vector, vector, matrix, scalar, scalar )";

		case VXM_VECTOR_VECTOR_VECTOR_VECTOR_MATRIX_ADD_MUL:
			return "vxm( vector, vector, vector, matrix, scalar, scalar )";

		case VXM_VECTOR_VECTOR_MATRIX_ADD_MUL:
			return "vxm( vector, vector, matrix, scalar, scalar )";

		case FOLDL_VECTOR_BETA_OP:
			return "foldl( vector, scalar, scalar )";

		case FOLDL_VECTOR_VECTOR_BETA_OP:
			return "foldl( vector, vector, scalar, scalar )";

		case FOLDL_VECTOR_BETA_MONOID:
			return "foldl( vector, scalar, monoid)";

		case FOLDL_VECTOR_VECTOR_BETA_MONOID:
			return "foldl( vector, vector, scalar, monoid)";

		case FOLDL_VECTOR_VECTOR_MONOID:
			return "foldl( vector, vector, monoid)";

		case FOLDL_VECTOR_VECTOR_VECTOR_MONOID:
			return "foldl( vector, vector, vector, monoid)";

		case FOLDL_VECTOR_VECTOR_VECTOR_OP:
			return "foldl( vector, vector, vecotr, scalar )";

		case FOLDL_VECTOR_VECTOR_OP:
			return "foldl( vector, vector, scalar )";

		case FOLDR_APLHA_VECTOR_MONOID:
			return "foldr( scalar, vector, monoid)";

		case FOLDR_APLHA_VECTOR_OPERATOR:
			return "foldr( scalar, vector, scalar )";

		case FOLDR_VECTOR_VECTOR_OPERATOR:
			return "foldr( vector, vector, scalar )";

		case FOLDR_VECTOR_VECTOR_VECTOR_OPERATOR:
			return "foldr( vector, vector, vector, scalar )";

		case FOLDR_VECTOR_VECTOR_MONOID:
			return "foldr( vector, vector, monoid)";

		case FOLDR_VECTOR_VECTOR_VECTOR_MONOID:
			return "foldr( vector, vector, vector, monoid)";

		case EWISEMUL_VECTOR_VECTOR_VECTOR_RING:
			return "eWiseMul( vector, vector, vector )";

		case EWISEMUL_VECTOR_ALPHA_VECTOR_RING:
			return "eWiseMul( vector, scalar, vector )";

		case EWISEMUL_VECTOR_VECTOR_BETA_RING:
			return "eWiseMul( vector, vector, scalar )";

		case EWISEMUL_VECTOR_ALPHA_BETA_RING:
			return "eWiseMul( vector, scalar, scalar )";

		case EWISEMUL_VECTOR_VECTOR_VECTOR_VECTOR_RING:
			return "eWiseMul( vector, vector, vector, vector )";

		case EWISEMUL_VECTOR_VECTOR_ALPHA_VECTOR_RING:
			return "eWiseMul( vector, vector, scalar, vector )";

		case EWISEMUL_VECTOR_VECTOR_VECTOR_BETA_RING:
			return "eWiseMul( vector, vector, vector, scalar )";

		case EWISEMUL_VECTOR_VECTOR_ALPHA_BETA_RING:
			return "eWiseMul( vector, vector, scalar, scalar )";

		case EWISELAMBDA_FUNC_VECTOR:
			return "eWiseLambda( function, vector )";

		case MXM_MATRIX_MATRIX_MATRIX_SEMIRING:
			return "mxm( matrix, matrix, matrix, semiring, scalar )";

		case CLEAR_MATRIX:
			return "clear( matrix )";

		case BUILDMATRIXUNIQUE_MATRIX_START_END_MODE:
			return "buildMatrixUnique( matrix, scalar, scalar, scalar )";

		case CAPACITY_VECTOR:
			return "capacity( vector )";

		case CAPACITY_MATRIX:
			return "capacity( matrix )";

		case RESIZE:
			return "resize( vector, scalar )";

		case RESIZE_MATRIX:
			return "resize( matrix, scalar )";

		case GETID_VECTOR:
			return "getID( vector )";

		case GETID_MATRIX:
			return "getID( matrix )";

		case SELECT_MATRIX_MATRIX_OP:
			return "select( matrix, matrix, selection_operator )";

		case SELECT_MATRIX_MATRIX_LAMBDA:
			return "selectLambda( matrix, matrix, lambda )";

	}
	assert( false );
	return "unknown operation";
}

grb::internal::hyperdags::SourceVertex::SourceVertex(
	const enum grb::internal::hyperdags::SourceVertexType _type,
	const size_t _local_id, const size_t _global_id
) noexcept : type( _type ), local_id( _local_id ), global_id( _global_id ) {}

enum grb::internal::hyperdags::SourceVertexType
grb::internal::hyperdags::SourceVertex::getType() const noexcept {
	return type;
}

size_t grb::internal::hyperdags::SourceVertex::getLocalID() const noexcept {
	return local_id;
}

size_t grb::internal::hyperdags::SourceVertex::getGlobalID() const noexcept {
	return global_id;
}

grb::internal::hyperdags::SourceVertexGenerator::SourceVertexGenerator() {
	for( size_t i = 0; i < numSourceVertexTypes; ++i ) {
		nextID[ allSourceVertexTypes[ i ] ] = 0;
	}
}

grb::internal::hyperdags::SourceVertex
grb::internal::hyperdags::SourceVertexGenerator::create(
	const grb::internal::hyperdags::SourceVertexType type,
	const size_t global_id
) {
	const size_t local_id = (nextID[ type ])++;
	grb::internal::hyperdags::SourceVertex ret( type, local_id, global_id );
	return ret;
}

size_t grb::internal::hyperdags::SourceVertexGenerator::size() const {
	size_t ret = 0;
	for( auto &pair : nextID ) {
		ret += pair.second;
	}
	return ret;
}

grb::internal::hyperdags::OutputVertex::OutputVertex(
	const size_t _lid,
	const size_t _gid
) noexcept : local_id( _lid ), global_id( _gid ) {
	type = OutputVertexType::CONTAINER_OUTPUT;
}

enum grb::internal::hyperdags::OutputVertexType
grb::internal::hyperdags::OutputVertex::getType() const noexcept {
	return type;
}

size_t grb::internal::hyperdags::OutputVertex::getLocalID() const noexcept {
	return local_id;
}

size_t grb::internal::hyperdags::OutputVertex::getGlobalID() const noexcept {
	return global_id;
}

grb::internal::hyperdags::OutputVertexGenerator::OutputVertexGenerator()
	noexcept : nextID( 0 )
{}

grb::internal::hyperdags::OutputVertex
grb::internal::hyperdags::OutputVertexGenerator::create(
	const size_t global_id
) {
	grb::internal::hyperdags::OutputVertex ret( nextID++, global_id );
	return ret;
}

size_t grb::internal::hyperdags::OutputVertexGenerator::size() const noexcept {
	return nextID;
}

grb::internal::hyperdags::OperationVertex::OperationVertex(
	const enum grb::internal::hyperdags::OperationVertexType _type,
	const size_t _lid, const size_t _gid
) noexcept : type( _type ), local_id( _lid ), global_id( _gid ) {}

enum grb::internal::hyperdags::OperationVertexType
grb::internal::hyperdags::OperationVertex::getType() const noexcept {
	return type;
}

size_t grb::internal::hyperdags::OperationVertex::getLocalID() const noexcept {
	return local_id;
}

size_t grb::internal::hyperdags::OperationVertex::getGlobalID() const noexcept {
	return global_id;
}

grb::internal::hyperdags::OperationVertexGenerator::OperationVertexGenerator() {
	for( size_t i = 0; i < numOperationVertexTypes; ++i ) {
		nextID[ allOperationVertexTypes[ i ] ] = 0;
	}
}

grb::internal::hyperdags::OperationVertex
grb::internal::hyperdags::OperationVertexGenerator::create(
	const grb::internal::hyperdags::OperationVertexType type,
	const size_t global_id
) {
	const size_t local_id = (nextID[ type ])++;
	grb::internal::hyperdags::OperationVertex ret( type, local_id, global_id );
	return ret;
}

size_t grb::internal::hyperdags::OperationVertexGenerator::size() const {
	size_t ret = 0;
	for( auto &pair : nextID ) {
		ret += pair.second;
	}
	return ret;
}

grb::internal::hyperdags::DHypergraph::DHypergraph() noexcept :
	num_vertices( 0 ), num_pins( 0 )
{}

size_t grb::internal::hyperdags::DHypergraph::createVertex() noexcept {
	return num_vertices++;
}

size_t grb::internal::hyperdags::DHypergraph::numVertices() const noexcept {
	return num_vertices;
}

size_t grb::internal::hyperdags::DHypergraph::numHyperedges() const noexcept {
	return hyperedges.size();
}

size_t grb::internal::hyperdags::DHypergraph::numPins() const noexcept {
	return num_pins;
}

void grb::internal::hyperdags::DHypergraph::render(
	std::ostream &out
) const {
	size_t net_num = 0;
	for( const auto &pair : hyperedges ) {
		out << net_num << " " << pair.first << "\n";
		const std::set< size_t > &net = pair.second;
		for( const auto &id : net ) {
			out << net_num << " " << id << "\n";
		}
		(void) ++net_num;
	}
	out << std::flush;
}

grb::internal::hyperdags::HyperDAG::HyperDAG(
	grb::internal::hyperdags::DHypergraph _hypergraph,
	const std::vector< grb::internal::hyperdags::SourceVertex > &_srcVec,
	const std::vector< grb::internal::hyperdags::OperationVertex > &_opVec,
	const std::vector< grb::internal::hyperdags::OutputVertex > &_outVec
) : hypergraph( _hypergraph ),
	num_sources( 0 ), num_operations( 0 ), num_outputs( 0 ),
	sourceVertices( _srcVec ),
	operationVertices( _opVec ),
	outputVertices( _outVec )
{
	// first add sources
	for( const auto &src : sourceVertices ) {
		const size_t local_id = src.getLocalID();
		const size_t global_id = src.getGlobalID();
		source_to_global_id[ local_id ] = global_id;
		global_to_type[ global_id ] = SOURCE;
		global_to_local_id[ global_id ] = local_id;
		(void) ++num_sources;
	}

	// second, add operations
	for( const auto &op : operationVertices ) {
		const size_t local_id = op.getLocalID();
		const size_t global_id = op.getGlobalID();
		operation_to_global_id[ local_id ] = global_id;
		global_to_type[ global_id ] = OPERATION;
		global_to_local_id[ global_id ] = local_id;
		(void) ++num_operations;
	}

	// third, add outputs
	for( const auto &out : outputVertices ) {
		const size_t local_id = out.getLocalID();
		const size_t global_id = out.getGlobalID();
		output_to_global_id[ local_id ] = global_id;
		global_to_type[ global_id ] = OUTPUT;
		global_to_local_id[ global_id ] = local_id;
		(void) ++num_outputs;
	}

	// final sanity check
	assert(
		num_sources + num_operations + num_outputs == hypergraph.numVertices()
	);
}

grb::internal::hyperdags::DHypergraph
grb::internal::hyperdags::HyperDAG::get() const noexcept {
	return hypergraph;
}

size_t grb::internal::hyperdags::HyperDAG::numSources() const noexcept {
	return num_sources;
}

size_t grb::internal::hyperdags::HyperDAG::numOperations() const noexcept {
	return num_operations;
}

size_t grb::internal::hyperdags::HyperDAG::numOutputs() const noexcept {
	return num_outputs;
}

std::vector< grb::internal::hyperdags::SourceVertex >::const_iterator
grb::internal::hyperdags::HyperDAG::sourcesBegin() const {
	return sourceVertices.cbegin();
}

std::vector< grb::internal::hyperdags::SourceVertex >::const_iterator
grb::internal::hyperdags::HyperDAG::sourcesEnd() const {
	return sourceVertices.cend();
}

std::vector< grb::internal::hyperdags::OperationVertex >::const_iterator
grb::internal::hyperdags::HyperDAG::operationsBegin() const {
	return operationVertices.cbegin();
}

std::vector< grb::internal::hyperdags::OperationVertex >::const_iterator
grb::internal::hyperdags::HyperDAG::operationsEnd() const {
	return operationVertices.cend();
}

std::vector< grb::internal::hyperdags::OutputVertex >::const_iterator
grb::internal::hyperdags::HyperDAG::outputsBegin() const {
	return outputVertices.cbegin();
}

std::vector< grb::internal::hyperdags::OutputVertex >::const_iterator
grb::internal::hyperdags::HyperDAG::outputsEnd() const {
	return outputVertices.cend();
}

grb::internal::hyperdags::HyperDAGGenerator::HyperDAGGenerator() noexcept {}

void grb::internal::hyperdags::HyperDAGGenerator::addContainer(
	const uintptr_t id
) {
	(void) addAnySource(
		grb::internal::hyperdags::SourceVertexType::CONTAINER,
		nullptr,
		id
	);
}

void grb::internal::hyperdags::HyperDAGGenerator::addSource(
	const enum grb::internal::hyperdags::SourceVertexType type,
	const void * const pointer
) {
	assert( type != grb::internal::hyperdags::SourceVertexType::CONTAINER );
	(void) addAnySource( type, pointer, 0 );
}

size_t grb::internal::hyperdags::HyperDAGGenerator::addAnySource(
	const enum grb::internal::hyperdags::SourceVertexType type,
	const void * const pointer,
	const uintptr_t id
) {
	size_t global_id;
	if( type == CONTAINER ) {
#ifdef _DEBUG
		std::cerr << "\t entering HyperDAGGen::addAnySource for container " << id
			<< " and type " << toString( type ) << ". "
			<< "Current source pointers:\n";
		for( const auto &pair : sourceVerticesP ) {
			std::cerr << "\t\t " << pair.first << "\n";
		}
		std::cerr << "Current source containers:\n";
		for( const auto &pair : sourceVerticesC ) {
			std::cerr << "\t\t " << pair.first << "\n";
		}
#endif
		const auto &find = sourceVerticesC.find( id );
		if( find != sourceVerticesC.end() ) {
#ifdef _DEBUG
			std::cerr << "\t\t entry already existed, removing it\n";
#endif
			sourceVerticesC.erase( find );
		}
		assert( sourceVerticesC.find( id ) == sourceVerticesC.end() );
		global_id = hypergraph.createVertex();
		const auto sourceVertex = sourceGen.create( type, global_id );
#ifdef _DEBUG
		std::cerr << "\t\t created a source vertex with global ID " << global_id
			<< " and local ID " << sourceVertex.getLocalID()
			<< " that will be associated to the unique identifier "
			<< id << "\n";
#endif
		assert( sourceVertex.getGlobalID() == global_id );
		sourceVerticesC.insert( std::make_pair( id, sourceVertex ) );
		sourceVec.push_back( sourceVertex );
	} else {
		assert( type != CONTAINER );
#ifdef _DEBUG
		std::cerr << "\t entering HyperDAGGen::addAnySource for auxiliary data at "
			<< pointer << " of type " << toString( type ) << ". "
			<< "Current source pointers:\n";
		for( const auto &pair : sourceVerticesP ) {
			std::cerr << "\t\t " << pair.first << "\n";
		}
		std::cerr << "Current source containers:\n";
		for( const auto &pair : sourceVerticesC ) {
			std::cerr << "\t\t " << pair.first << "\n";
		}
#endif
		const auto &find = sourceVerticesP.find( pointer );
		if( find != sourceVerticesP.end() ) {
#ifdef _DEBUG
			std::cerr << "\t\t entry already existed, removing it\n";
#endif
			sourceVerticesP.erase( find );
		}
		assert( sourceVerticesP.find( pointer ) == sourceVerticesP.end() );
		global_id = hypergraph.createVertex();
		const auto sourceVertex = sourceGen.create( type, global_id );
#ifdef _DEBUG
		std::cerr << "\t\t created a source vertex with global ID " << global_id
			<< " and local ID " << sourceVertex.getLocalID()
			<< " that will be associated to the unique identifier "
			<< pointer << "\n";
#endif
		assert( sourceVertex.getGlobalID() == global_id );
		sourceVerticesP.insert( std::make_pair( pointer, sourceVertex ) );
		sourceVec.push_back( sourceVertex );
	}
#ifdef _DEBUG
	std::cerr << "\t\t Sizes of sourceVertices and sourceVec: "
		<< sourceVerticesP.size() << " + "
		<< sourceVerticesC.size() << ", resp., "
		<< sourceVec.size() << ".\n\t\t Contents of sourceVertices:\n";
	for( const auto &pair : sourceVerticesP ) {
		std::cerr << "\t\t\t " << pair.first << "\n";
	}
	for( const auto &pair : sourceVerticesC ) {
		std::cerr << "\t\t\t " << pair.first << "\n";
	}
#endif
	return global_id;
}

grb::internal::hyperdags::HyperDAG
grb::internal::hyperdags::HyperDAGGenerator::finalize() const {
#ifdef _DEBUG
	std::cout << "HyperDAGGenerator::finalize called.\n"
		<< "\t there are presently "
		<< operationOrOutputVertices.size()
		<< "output vertices\n";
#endif
	std::vector< grb::internal::hyperdags::OutputVertex > outputVec;

	// generate outputVertices
	{
		grb::internal::hyperdags::OutputVertexGenerator outputGen;
		for( const auto &pair : operationOrOutputVertices ) {
			grb::internal::hyperdags::OutputVertex toAdd =
				outputGen.create( pair.second.first );
			outputVec.push_back( toAdd );
		}
	}

	// generate HyperDAG
	grb::internal::hyperdags::HyperDAG ret(
		hypergraph,
		sourceVec, operationVec, outputVec
	);

	// done
	return ret;
}

