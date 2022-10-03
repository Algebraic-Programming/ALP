
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

template<>
grb::RC grb::finalize< grb::hyperdags >() {
	std::cerr << "Info: grb::finalize (hyperdags) called.\n";
	std::cerr << "\t dumping HyperDAG to stdout" << std::endl;
	const grb::internal::hyperdags::HyperDAG &hyperdag =
		grb::internal::hyperdags::generator.finalize();
	const grb::internal::hyperdags::Hypergraph &hypergraph =
		hyperdag.get();
	std::cout << "%%MatrixMarket matrix coordinate pattern general\n";
	std::cout << "%\t Source vertices:\n";
	for( auto it = hyperdag.sourcesBegin(); it != hyperdag.sourcesEnd(); ++it ) {
		std::cout << "%\t\t " << it->getGlobalID() << ": "
			<< grb::internal::hyperdags::toString( it->getType() ) << " "
			<< "no. " << it->getLocalID()
			<< "\n";
	}
	std::cout << "%\t Output vertices:\n";
	for( auto it = hyperdag.outputsBegin(); it != hyperdag.outputsEnd(); ++it ) {
		std::cout << "%\t\t " << it->getGlobalID() << ": "
			<< grb::internal::hyperdags::toString( it->getType() ) << " "
			<< "no. " << it->getLocalID()
			<< "\n";
	}
	std::cout << hypergraph.numHyperedges() << " "
		<< hypergraph.numVertices() << " "
		<< hypergraph.numPins() << "\n";
	hypergraph.render( std::cout );
	return grb::finalize< grb::_GRB_WITH_HYPERDAGS_USING >();
}
