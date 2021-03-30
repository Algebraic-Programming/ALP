
/*
 * Copyright Huawei Technologies Switzerland AG
 * All rights reserved.
 *
 * @author A. N. Yzelman
 */

#ifndef _H_GRB_PREGEL_PAGERANK
#define _H_GRB_PREGEL_PAGERANK

#include <graphblas/interfaces/pregel.hpp>

/*namespace grb {

	namespace algorithms {

		namespace pregel {*/

					typedef typename grb::interfaces::Pregel<
						double,
						grb::operators::add,
						grb::identities::zero
					> PageRankPregelType;

					typedef typename PageRankPregelType::PregelData PageRankPregelData;


					struct PageRankData {
						const double alpha = 0.85;
						const double tolerance = 0.1;
					};

					void PageRankProgram(
						double &current_score,
						const PageRankData &parameters,
						PageRankPregelData &pregel
					) {
						if( pregel.round == 0 ) {
							current_score = 1 /
								static_cast< double >(
									pregel.num_vertices
								);
						}
						if( pregel.round > 0 ) {
							const double old_score = current_score;
							current_score = (1-parameters.alpha) +
								parameters.alpha * pregel.incoming_message;
							if( old_score < parameters.tolerance ) {
								pregel.active = false;
							}
						}
						pregel.outgoing_message =
							current_score /
							static_cast< double >(pregel.outdegree);
					}

/*		} //end namespace `grb::algorithms::pregel'
	} // end namespace ``grb::algorithms''
} // end namespace ``grb''
*/

#endif

