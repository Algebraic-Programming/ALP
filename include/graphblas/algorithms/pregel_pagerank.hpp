
/*
 * Copyright Huawei Technologies Switzerland AG
 * All rights reserved.
 *
 * @author A. N. Yzelman
 */

#ifndef _H_GRB_PREGEL_PAGERANK
#define _H_GRB_PREGEL_PAGERANK

#include <graphblas/interfaces/pregel.hpp>


namespace grb {

	namespace algorithms {

		namespace pregel {

			template< typename IOType, bool localConverge >
			struct PageRank {

				struct Data {
					IOType alpha = 0.15;
					IOType tolerance = 0.00001;
				};

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
					const bool dbg = pregel.vertexID == 17;
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

				template< typename PregelType >
				static grb::RC execute(
					grb::interfaces::Pregel< PregelType > &pregel,
					grb::Vector< IOType > &scores,
					size_t &steps_taken,
					const Data &parameters = Data(),
					const size_t max_steps = 0
				) {
					const size_t n = pregel.num_vertices();
					if( grb::size( scores ) != n ) {
						return MISMATCH;
					}

					grb::Vector< IOType > in( n );
					grb::Vector< IOType > out( n );

					return pregel.template execute<
							grb::operators::add< PregelType >,
							grb::identities::zero
						> (
							program,
							scores,
							parameters,
							in, out,
							steps_taken,
							max_steps
						);
				}

			};

		} //end namespace `grb::algorithms::pregel'

	} // end namespace ``grb::algorithms''

} // end namespace ``grb''

#endif

