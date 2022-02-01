
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
	nextID[ grb::internal::hyperdags::SourceVertexType::CONTAINER ] = 0;
	nextID[ grb::internal::hyperdags::SourceVertexType::SET ] = 0;
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
	nextID[ grb::internal::hyperdags::OperationVertexType::NNZ_VECTOR ] = 0;
}

grb::internal::hyperdags::OperationVertex grb::internal::hyperdags::OperationVertexGenerator::create(
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

grb::internal::hyperdags::Hypergraph::Hypergraph() noexcept : num_vertices( 0 ) {}

size_t grb::internal::hyperdags::Hypergraph::createVertex() noexcept {
	return num_vertices++;
}

size_t grb::internal::hyperdags::Hypergraph::numVertices() const noexcept {
	return num_vertices;
}

void grb::internal::hyperdags::Hypergraph::render(
	std::ostream &out
) const {
	size_t net_num = 0;
	for( const auto &net : hyperedges ) {
		for( const auto &id : net ) {
			out << net_num << " " << id << "\n";
		}
		(void) ++net_num;
	}
	out << std::flush;
}

const grb::internal::hyperdags::Hypergraph & grb::internal::hyperdags::HyperDAG::get() const noexcept {
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

grb::internal::hyperdags::HyperDAGGenerator::HyperDAGGenerator() noexcept {}

void grb::internal::hyperdags::HyperDAGGenerator::addSource(
	const enum grb::internal::hyperdags::SourceVertexType type,
	const void * const pointer
) {
	assert( type != grb::internal::hyperdags::SourceVertexType::CONTAINER );
	const auto &find = sourceVertices.find( pointer );
	if( find != sourceVertices.end() ) {
		sourceVertices.erase( find );
	}
	const size_t global_id = hypergraph.createVertex();
	const auto &sourceVertex = sourceGen.create( type, global_id );
	sourceVertices.insert( std::make_pair( pointer, sourceVertex ) );
}

grb::internal::hyperdags::HyperDAG grb::internal::hyperdags::HyperDAGGenerator::finalize() const {
	std::vector< grb::internal::hyperdags::SourceVertex > sourceVec;
	std::vector< grb::internal::hyperdags::OperationVertex > operationVec;
	std::vector< grb::internal::hyperdags::OutputVertex > outputVec;

	// generate sourceVertices
	{
		for( const auto &pair : sourceVertices ) {
			sourceVec.push_back( pair.second );
		}
	}

	// generate operationVertices
	{
		for( const auto &pair : operationVertices ) {
			operationVec.push_back( pair.second );
		}
	}

	// generate outputVertices
	{
		grb::internal::hyperdags::OutputVertexGenerator outputGen;
		for( const auto &pair : operationOrOutputVertices ) {
			grb::internal::hyperdags::OutputVertex toAdd = outputGen.create( pair.second );
			outputVec.push_back( toAdd );
		}
	}

	// generate HyperDAG
	grb::internal::hyperdags::HyperDAG ret(
		hypergraph,
		sourceVec.begin(), sourceVec.end(),
		operationVec.begin(), operationVec.end(),
		outputVec.begin(), outputVec.end()
	);

	// done
	return ret;	
}

