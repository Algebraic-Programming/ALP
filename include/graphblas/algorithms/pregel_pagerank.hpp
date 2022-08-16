
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

			template< typename IOType >
			struct PageRank {

				struct Data {
					IOType alpha = 0.85;
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
						current_score = 1 /
							static_cast< double >(
								pregel.num_vertices
							);
					}

					// compute
					if( pregel.round > 0 ) {
						const double old_score = current_score;
						current_score = (1-parameters.alpha) +
							parameters.alpha * incoming_message;
						if( fabs(current_score-old_score) <
							parameters.tolerance
						) {
							pregel.active = false;
						}
					}

					// broadcast
					if( pregel.outdegree > 0 ) {
						outgoing_message =
							current_score /
							static_cast< double >(pregel.outdegree);
					}
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
							grb::operators::add< double >,
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

