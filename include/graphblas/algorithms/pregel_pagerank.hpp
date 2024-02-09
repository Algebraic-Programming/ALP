
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
 * Implements a traditional vertex-centric page ranking algorithm using
 * ALP/Pregel.
 *
 * @author A. N. Yzelman
 */

#ifndef _H_GRB_PREGEL_PAGERANK
#define _H_GRB_PREGEL_PAGERANK

#include <graphblas/interfaces/pregel.hpp>


namespace grb {

	namespace algorithms {

		namespace pregel {

			/**
			 * A Pregel-style PageRank-like algorithm.
			 *
			 * This vertex-centric program does not correspond to the canonical PageRank
			 * algorithm by Brin and Page. In particular, it misses corrections for
			 * dangling nodes and does not perform convergence checks in any norm.
			 *
			 * @tparam IOType The type of the PageRank scores (e.g., <tt>double</tt>).
			 * @tparam localConverge Whether vertices become inactive once their local
			 *                       scores have converged, or whether to terminate only
			 *                       when all vertices have converged.
			 *
			 * \ingroup Pregel
			 */
			template< typename IOType, bool localConverge >
			struct PageRank {

				/**
				 * The algorithm parameters.
				 */
				struct Data {

					/**
					 * The probability of jumping to a random page instead of a linked page.
					 */
					IOType alpha = 0.15;

					/**
					 * The local convergence criterion.
					 */
					IOType tolerance = 0.00001;

				};

				/**
				 * The vertex-centric PageRank-like program.
				 *
				 * @param[out] current_score    The current rank corresponding to this
				 *                              vertex.
				 * @param[in]  incoming_message Neighbour contributions to our score.
				 * @param[out] outgoing_message The score contribution to send to our
				 *                              neighbours.
				 * @param[in]     parameters    The algorithm parameters.
				 * @param[in,out]     pregel    The state of the Pregel interface.
				 *
				 * The Pregel program expects incoming messages to be aggregated using a
				 * plus monoid over elements of \a IOType.
				 */
				static void program(
					IOType &current_score,
					const IOType &incoming_message,
					IOType &outgoing_message,
					const Data &parameters,
					grb::interfaces::PregelState &pregel
				) {
					// initialise
					if( pregel.round == 0 ) {
						current_score = static_cast< IOType >( 1 );
					}

#ifdef _DEBUG
					// when in debug mode, probably one does not wish to track the state of
					// each vertex individually, hence we include a simple guard by default:
					const bool dbg = pregel.vertexID == 0;
					if( dbg ) {
						std::cout << "ID: " << pregel.vertexID << "\n"
							<< "\t active: " << pregel.active << "\n"
							<< "\t round: " << pregel.round << "\n"
							<< "\t previous score: " << current_score << "\n"
							<< "\t incoming message: " << incoming_message << "\n";
					}
#endif

					// compute
					if( pregel.round > 0 ) {
						const IOType old_score = current_score;
						current_score = parameters.alpha +
							(static_cast< IOType >(1) - parameters.alpha) * incoming_message;
						if( fabs(current_score-old_score) < parameters.tolerance ) {
#ifdef _DEBUG
							std::cout << "\t\t vertex " << pregel.vertexID << " converged\n";
#endif
							if( localConverge ) {
								pregel.active = false;
							} else {
								pregel.voteToHalt = true;
							}
						}
					}

					// broadcast
					if( pregel.outdegree > 0 ) {
						outgoing_message =
							current_score /
							static_cast< IOType >(pregel.outdegree);
					}

#ifdef _DEBUG
					if( dbg ) {
						std::cout << "\t current score: " << current_score << "\n"
							<< "\t voteToHalt: " << pregel.voteToHalt << "\n"
							<< "\t outgoing message: " << outgoing_message << "\n";
					}
#endif

				}

				/**
				 * A convenience function for launching a PageRank algorithm over a given
				 * Pregel instance.
				 *
				 * @tparam PregelType The nonzero type of an edge in the Pregel instance.
				 *
				 * This convenience function materialises the buffers expected to be passed
				 * into the Pregel instance, and selects the expected monoid for executing
				 * this program.
				 *
				 * \warning In performance-critical code, one may want to pre-allocate the
				 *          buffers instead of having this convenience function allocate
				 *          those. In such cases, please call manually the Pregel execute
				 *          function, i.e., #grb::interfaces::Pregel< PregelType >::execute.
				 *
				 * The following arguments are mandatory:
				 *
				 * @param[in]  pregel      The Pregel instance that this program should
				 *                         execute on.
				 * @param[out] scores      A vector that corresponds to the scores
				 *                         corresponding to each vertex. It must be of size
				 *                         equal to the number of vertices \f$ n \f$ in the
				 *                         \a pregel instance, and must have \f$ n \f$
				 *                         capacity \em and values. The initial contents are
				 *                         ignored by this algorithm.
				 * @param[out] steps_taken How many rounds the program took until
				 *                         termination.
				 *
				 * The following arguments are optional:
				 *
				 * @param[in] parameters The algorithm parameters. If not given, default
				 *                       values will be substituted.
				 * @param[in] max_steps  The maximum number of rounds this program may take.
				 *                       If not given, the number of rounds will be
				 *                       unlimited.
				 */
				template< typename PregelType >
				static grb::RC execute(
					grb::interfaces::Pregel< PregelType > &pregel,
					grb::Vector< IOType > &scores,
					size_t &steps_taken,
					const Data &parameters = Data(),
					const size_t max_steps = 0
				) {
					const size_t n = pregel.numVertices();
					if( grb::size( scores ) != n ) {
						return MISMATCH;
					}

					grb::Vector< IOType > in( n );
					grb::Vector< IOType > out( n );
					grb::Vector< IOType > out_buffer = interfaces::config::out_sparsify
						? grb::Vector< IOType >( n )
						: grb::Vector< IOType >( 0 );

					return pregel.template execute<
							grb::operators::add< IOType >,
							grb::identities::zero
						> (
							program,
							scores,
							parameters,
							in, out,
							steps_taken,
							out_buffer,
							max_steps
						);
				}

			};

		} //end namespace `grb::algorithms::pregel'

	} // end namespace ``grb::algorithms''

} // end namespace ``grb''

#endif

