
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
 * @date 31st of January, 2022
 */

#include <iostream>
#include <fstream>

#include <graphblas/init.hpp>
#include <graphblas/hyperdags/hyperdags.hpp>


namespace grb {

	namespace internal {

		namespace hyperdags {

			HyperDAGGenerator generator;

		}

	}

}

template<>
grb::RC grb::init< grb::hyperdags >(
	const size_t s, const size_t P, void * const
) {
	std::cerr << "Info: grb::init (hyperdags) called.\n";
	return grb::init< grb::_GRB_WITH_HYPERDAGS_USING >( s, P, nullptr );
}

static size_t src2int( const grb::internal::hyperdags::SourceVertexType type ) {
	return static_cast< size_t >( type );
}

static size_t op2int( const grb::internal::hyperdags::OperationVertexType type ) {
	size_t ret = static_cast< size_t >( type );
	ret += grb::internal::hyperdags::numSourceVertexTypes;
	return ret;
}

static size_t out2int( const grb::internal::hyperdags::OutputVertexType type ) {
	size_t ret = static_cast< size_t >( type );
	ret += grb::internal::hyperdags::numSourceVertexTypes;
	ret += grb::internal::hyperdags::numOperationVertexTypes;
	return ret;
}

template<>
grb::RC grb::finalize< grb::hyperdags >() {
	std::cerr << "Info: grb::finalize (hyperdags) called.\n";
	std::cerr << "\t dumping HyperDAG to stdout" << std::endl;
	const grb::internal::hyperdags::HyperDAG &hyperdag =
		grb::internal::hyperdags::generator.finalize();
	const grb::internal::hyperdags::DHypergraph &hypergraph =
		hyperdag.get();

	// Try to get the env variable HYPERDAGS_OUTPUTH_PATH
	// If it exists, use it as the output path
	// Otherwise, use stdout
	const char *outputPath = std::getenv( "HYPERDAGS_OUTPUT_PATH" );
	std::ofstream fileStream;
	if( outputPath != nullptr ) {
		std::cerr << "\t dumping HyperDAG to " << outputPath << std::endl;
		fileStream.open( outputPath );
	} else {
		std::cerr << "\t dumping HyperDAG to stdout" << std::endl;
	}
	std::ostream &ostream = ( outputPath != nullptr ) ? fileStream : std::cout;

	ostream << "%%MatrixMarket weighted-matrix coordinate pattern general\n";
	// print source vertex types as comments
	{
		std::set< grb::internal::hyperdags::SourceVertexType > present;
		for( auto it = hyperdag.sourcesBegin(); it != hyperdag.sourcesEnd(); ++it ) {
			present.insert( it->getType() );
		}
		ostream << "%\t There are " << present.size() << " "
			<< "unique source vertex types present in this graph. "
			<< "An index of source type ID and their description follows:\n";
		for( const auto &type : present ) {
			ostream << "%\t\t " << src2int( type ) << ": " << toString( type ) << "\n";
		}
	}
	// print operation vertex types as comments
	{
		std::set< grb::internal::hyperdags::OperationVertexType > present;
		for(
			auto it = hyperdag.operationsBegin();
			it != hyperdag.operationsEnd();
			++it
		) {
			present.insert( it->getType() );
		}
		ostream << "%\t There are " << present.size() << " "
			<< "unique operation vertex types present in this graph. "
			<< "An index of vertex type ID and their description follows:\n";
		for( const auto &type : present ) {
			ostream << "%\t\t " << op2int( type ) << ": " << toString( type ) << "\n";
		}
	}
	// print output vertex types as comments
	{
		std::set< grb::internal::hyperdags::OutputVertexType > present;
		for( auto it = hyperdag.outputsBegin(); it != hyperdag.outputsEnd(); ++it ) {
			present.insert( it->getType() );
		}
		ostream << "%\t There are " << present.size() << " "
			<< "unique output vertex types present in this graph. "
			<< "An index of output vertex type ID and their description follows:\n";
		for( const auto &type : present ) {
			ostream << "%\t\t " << out2int( type ) << ": " << toString( type ) << "\n";
		}
	}
	// print HyperDAG size
	const size_t numEdges = hypergraph.numHyperedges();
	ostream << numEdges << " " << hypergraph.numVertices() << " "
		<< hypergraph.numPins() << "\n";
	// print all hyperedge IDs
	for( size_t i = 0; i < numEdges; ++i ) {
		ostream << i << " % no additional data on hyperedges at present\n";
	}
	// print all vertex IDs, their types, and their local IDs
	for( auto it = hyperdag.sourcesBegin(); it != hyperdag.sourcesEnd(); ++it ) {
		ostream << it->getGlobalID() << " " << src2int( it->getType() ) << " "
			<< it->getLocalID() << " "
			<< "% source vertex of type "
			<< grb::internal::hyperdags::toString( it->getType() ) << " no. "
			<< it->getLocalID() << "\n";
	}
	for(
		auto it = hyperdag.operationsBegin();
		it != hyperdag.operationsEnd();
		++it
	) {
		ostream << it->getGlobalID() << " " << op2int( it->getType() ) << " "
			<< it->getLocalID() << " "
			<< "% operation vertex of type "
			<< grb::internal::hyperdags::toString( it->getType() ) << " "
			<< "no. " << it->getLocalID() << "\n";
	}
	for( auto it = hyperdag.outputsBegin(); it != hyperdag.outputsEnd(); ++it ) {
		ostream << it->getGlobalID() << " " << out2int( it->getType() ) << " "
			<< it->getLocalID() << " "
			<< "% output vertex of type "
			<< grb::internal::hyperdags::toString( it->getType() ) << " "
			<< "no. " << it->getLocalID() << "\n";
	}
	// print HyperDAG structure
	hypergraph.render( ostream );
	// done
	return grb::finalize< grb::_GRB_WITH_HYPERDAGS_USING >();
}

