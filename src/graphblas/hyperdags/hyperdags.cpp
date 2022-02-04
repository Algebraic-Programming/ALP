
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
			return "user-initialised container";

		case SET:
			return "container initialised by a call to set";

	}
	assert( false );
	return "unidentified source vertex type";
}

std::string grb::internal::hyperdags::toString(
	const enum OperationVertexType type
) noexcept {
	switch( type ) {

		case NNZ_VECTOR:
			return "nnz (vector)";

		case CLEAR_VECTOR:
			return "clear (vector)";

		case SET_VECTOR_ELEMENT:
			return "setElement (vector)";

		case DOT:
			return "dot";

	}
	assert( false );
	return "unknown operation";
}

grb::internal::hyperdags::SourceVertex::SourceVertex(
	const enum grb::internal::hyperdags::SourceVertexType _type,
	const size_t _local_id, const size_t _global_id
) noexcept : type( _type ), local_id( _local_id ), global_id( _global_id ) {}

enum grb::internal::hyperdags::SourceVertexType grb::internal::hyperdags::SourceVertex::getType() const noexcept {
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

grb::internal::hyperdags::SourceVertex grb::internal::hyperdags::SourceVertexGenerator::create(
	const grb::internal::hyperdags::SourceVertexType type,
	const size_t global_id
) {
	const size_t local_id = (nextID[ type ])++;
	grb::internal::hyperdags::SourceVertex ret( type, local_id, global_id );
	return ret;
}

size_t grb::internal::hyperdags::SourceVertexGenerator::size() const {
	size_t ret = 0;
	{
		const auto &it = nextID.find( grb::internal::hyperdags::SourceVertexType::CONTAINER );
		assert( it != nextID.end() );
		ret += it->second;
	}
	{
		const auto &it = nextID.find( grb::internal::hyperdags::SourceVertexType::SET );
		assert( it != nextID.end() );
		ret += it->second;
	}
	return ret;
}

grb::internal::hyperdags::OutputVertex::OutputVertex(
	const size_t _lid,
	const size_t _gid
) noexcept : local_id( _lid ), global_id( _gid ) {}

size_t grb::internal::hyperdags::OutputVertex::getLocalID() const noexcept {
	return local_id;
}

size_t grb::internal::hyperdags::OutputVertex::getGlobalID() const noexcept {
	return global_id;
}

grb::internal::hyperdags::OutputVertexGenerator::OutputVertexGenerator() noexcept : nextID( 0 ) {}

grb::internal::hyperdags::OutputVertex grb::internal::hyperdags::OutputVertexGenerator::create(
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

enum grb::internal::hyperdags::OperationVertexType grb::internal::hyperdags::OperationVertex::getType() const noexcept {
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
	{
		const auto &it = nextID.find( grb::internal::hyperdags::OperationVertexType::NNZ_VECTOR );
		assert( it != nextID.end() );
		ret += it->second;
	}
	return ret;
}

grb::internal::hyperdags::Hypergraph::Hypergraph() noexcept : num_vertices( 0 ), num_pins( 0 ) {}

size_t grb::internal::hyperdags::Hypergraph::createVertex() noexcept {
	return num_vertices++;
}

size_t grb::internal::hyperdags::Hypergraph::numVertices() const noexcept {
	return num_vertices;
}

size_t grb::internal::hyperdags::Hypergraph::numHyperedges() const noexcept {
	return hyperedges.size();
}

size_t grb::internal::hyperdags::Hypergraph::numPins() const noexcept {
	return num_pins;
}

void grb::internal::hyperdags::Hypergraph::render(
	std::ostream &out
) const {
	size_t net_num = 0;
	for( const std::set< size_t > &net : hyperedges ) {
		for( const auto &id : net ) {
			out << net_num << " " << id << "\n";
		}
		(void) ++net_num;
	}
	out << std::flush;
}

grb::internal::hyperdags::HyperDAG::HyperDAG(
	grb::internal::hyperdags::Hypergraph _hypergraph,
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
	assert( num_sources + num_operations + num_outputs == hypergraph.numVertices() );
}

grb::internal::hyperdags::Hypergraph grb::internal::hyperdags::HyperDAG::get() const noexcept {
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

grb::internal::hyperdags::HyperDAGGenerator::HyperDAGGenerator() noexcept {}

void grb::internal::hyperdags::HyperDAGGenerator::addSource(
	const enum grb::internal::hyperdags::SourceVertexType type,
	const void * const pointer
) {
	assert( type != grb::internal::hyperdags::SourceVertexType::CONTAINER );
	(void) addAnySource( type, pointer );
}

size_t grb::internal::hyperdags::HyperDAGGenerator::addAnySource(
	const enum grb::internal::hyperdags::SourceVertexType type,
	const void * const pointer
) {
#ifdef _DEBUG
	std::cerr << "\t entering HyperDAGGen::addAnySource for " << pointer << "\n";
#endif
	const auto &find = sourceVertices.find( pointer );
	if( find != sourceVertices.end() ) {
#ifdef _DEBUG
		std::cerr << "\t\t entry already existed, removing it\n";
#endif
		sourceVertices.erase( find );
	}
	const size_t global_id = hypergraph.createVertex();
	const auto &sourceVertex = sourceGen.create( type, global_id );
#ifdef _DEBUG
	std::cerr << "\t\t created a source vertex with global ID " << global_id
		<< " and local ID " << sourceVertex.getLocalID() << "\n";
#endif
	assert( sourceVertex.getGlobalID() == global_id );
	sourceVertices.insert( std::make_pair( pointer, sourceVertex ) );
	sourceVec.push_back( sourceVertex );
#ifdef _DEBUG
	std::cerr << "\t\t sourceVertices and sourceVec sizes: "
		<< sourceVertices.size() << ", resp., "
		<< sourceVec.size() << "\n";
#endif
	return global_id;
}

grb::internal::hyperdags::HyperDAG grb::internal::hyperdags::HyperDAGGenerator::finalize() const {
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

