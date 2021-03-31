
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

			struct PageRankData {
				const double alpha = 0.85;
				const double tolerance = 0.001;
			};

			void pageRank(
				double &current_score,
				const double &incoming_message,
				double &outgoing_message,
				const PageRankData &parameters,
				grb::interfaces::PregelData &pregel
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
						parameters.alpha * incoming_message;
					if( fabs(current_score-old_score) <
						parameters.tolerance
					) {
						pregel.active = false;
					}
				}
				if( pregel.outdegree > 0 ) {
					outgoing_message =
						current_score /
						static_cast< double >(pregel.outdegree);
				}
			}

		} //end namespace `grb::algorithms::pregel'

	} // end namespace ``grb::algorithms''

} // end namespace ``grb''

#endif

