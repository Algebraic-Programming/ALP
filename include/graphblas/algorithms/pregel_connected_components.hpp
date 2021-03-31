
/*
 * Copyright Huawei Technologies Switzerland AG
 * All rights reserved.
 *
 * @author A. N. Yzelman
 */

#ifndef _H_GRB_PREGEL_CONNECTEDCOMPONENTS
#define _H_GRB_PREGEL_CONNECTEDCOMPONENTS

#include <graphblas/interfaces/pregel.hpp>

namespace grb {

	namespace algorithms {

		namespace pregel {

			template< typename VertexIDType >
			struct ConnectedComponents {

				struct Data {};

				static void program(
					VertexIDType &current_max_ID,
					const VertexIDType &incoming_message,
					VertexIDType &outgoing_message,
					const Data &parameters,
					grb::interfaces::PregelData &pregel
				) {
					if( pregel.round > 0 ) {
						if( pregel.indegree == 0 ) {
							pregel.voteToHalt = true;
						} else if( current_max_ID < incoming_message ) {
std::cout << "I found a larger ID, going from " << current_max_ID << " to " << incoming_message << "!\n";
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

				template< typename PregelType >
				static grb::RC execute(
					grb::interfaces::Pregel< PregelType > &pregel,
					grb::Vector< VertexIDType > &group_ids,
					const size_t max_steps = 1000
				) {
					const size_t n = pregel.num_vertices();
					if( grb::size( group_ids ) != n ) {
						return MISMATCH;
					}

					const grb::RC ret =
						grb::set< grb::descriptors::use_index >( group_ids, 1 );
					if( ret != SUCCESS ) {
						return ret;
					}

					grb::Vector< VertexIDType > in( n );
					grb::Vector< VertexIDType > out( n );

					return pregel.execute(
						group_ids,
						in, out,
						program,
						Data(),
						grb::operators::max< VertexIDType >(),
						grb::identities::negative_infinity(),
						max_steps
					);
				}

			};

		} //end namespace `grb::algorithms::pregel'

	} // end namespace ``grb::algorithms''

} // end namespace ``grb''

#endif

