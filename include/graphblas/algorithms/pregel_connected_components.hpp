
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

/**
 * @file
 *
 * Implements the (strongly) connected components algorithm over undirected
 * graphs using the ALP/Pregel interface.
 *
 * @author: A. N. Yzelman.
 */

#ifndef _H_GRB_PREGEL_CONNECTEDCOMPONENTS
#define _H_GRB_PREGEL_CONNECTEDCOMPONENTS

#include <graphblas/interfaces/pregel.hpp>


namespace grb {

	namespace algorithms {

		namespace pregel {

			/**
			 * A vertex-centric Connected Components algorithm.
			 *
			 * @tparam VertexIDType A type large enough to assign an ID to each vertex
			 *                      in the graph the algorithm is to run on.
			 *
			 * \ingroup Pregel
			 */
			template< typename VertexIDType >
			struct ConnectedComponents {

				/**
				 * This vertex-centric Connected Components algorithm does not require any
				 * algorithm parameters.
				 */
				struct Data {};

				/**
				 * The vertex-centric program for computing connected components. On
				 * termination, the number of individual IDs in \a current_max_ID signifies
				 * the number of components, while the value at each entry signifies which
				 * component the vertex corresponds to.
				 *
				 * @param[in,out] current_max_ID On input: each entry is set to an unique
				 *                               ID, corresponding to a unique ID for each
				 *                               vertex. On output: the ID of the component
				 *                               the corresponding vertex belongs to.
				 * @param[in]   incoming_message A buffer for incoming messages to a vertex
				 *                               program.
				 * @param[in]   outgoing_message A buffer for outgoing messages to a vertex
				 *                               program.
				 * @param[in]         parameters Global algorithm parameters, currently an
				 *                               instance of an empty struct (no
				 *                               parameters).
				 * @param[in,out]         pregel The Pregel state the program may refer to.
				 *
				 * This program 1) broadcasts its current ID to its neighbours, 2) checks
				 * if any received IDs are larger than the current ID, then 3a) if not,
				 * votes to halt; 3b) if yes, replaces the current ID with the received
				 * maximum. It is meant to be executed using a max monoid as message
				 * aggregator.
				 */
				static void program(
					VertexIDType &current_max_ID,
					const VertexIDType &incoming_message,
					VertexIDType &outgoing_message,
					const Data &parameters,
					grb::interfaces::PregelState &pregel
				) {
					(void) parameters;
					if( pregel.round > 0 ) {
						if( pregel.indegree == 0 ) {
							pregel.voteToHalt = true;
						} else if( current_max_ID < incoming_message ) {
							current_max_ID = incoming_message;
						} else {
							pregel.voteToHalt = true;
						}
					}
					if( pregel.outdegree > 0 ) {
						outgoing_message = current_max_ID;
					} else {
						pregel.voteToHalt = true;
					}
				}

				/**
				 * A convenience function that, given a Pregel instance, executes the
				 * #program.
				 *
				 * @param[in,out] pregel A Pregel instance over which to execute the
				 *                       program.
				 * @param[out] group_ids The ID of the component the corresponding vertex
				 *                       belongs to.
				 * @param[in]  max_steps A maximum number of rounds the program is allowed
				 *                       to run. If \a 0, no maximum number of rounds will
				 *                       be in effect.
				 *
				 * On succesful termination, the number of rounds is optionally written
				 * out:

				 * @param[out] steps_taken A pointer to where the number of rounds should
				 *                         be recorded. Will not be used if equal to
				 *                         <tt>nullptr</tt>.
				 */
				template< typename PregelType >
				static grb::RC execute(
					grb::interfaces::Pregel< PregelType > &pregel,
					grb::Vector< VertexIDType > &group_ids,
					const size_t max_steps = 0,
					size_t * const steps_taken = nullptr
				) {
					const size_t n = pregel.numVertices();
					if( grb::size( group_ids ) != n ) {
						return MISMATCH;
					}

					grb::RC ret = grb::set< grb::descriptors::use_index >( group_ids, 1 );
					if( ret != SUCCESS ) {
						return ret;
					}

					grb::Vector< VertexIDType > in( n );
					grb::Vector< VertexIDType > out( n );
					grb::Vector< VertexIDType > out_buffer = interfaces::config::out_sparsify
						? grb::Vector< VertexIDType >( n )
						: grb::Vector< VertexIDType >( 0 );

					size_t steps;

					ret = pregel.template execute<
						grb::operators::max< VertexIDType >,
						grb::identities::negative_infinity
					> (
						program,
						group_ids,
						Data(),
						in, out,
						steps,
						out_buffer,
						max_steps
					);

					if( ret == grb::SUCCESS && steps_taken != nullptr ) {
						*steps_taken = steps;
					}

					return ret;
				}

			};

		} //end namespace `grb::algorithms::pregel'

	} // end namespace ``grb::algorithms''

} // end namespace ``grb''

#endif

