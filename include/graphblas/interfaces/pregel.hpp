
/*
 * Copyright Huawei Technologies Switzerland AG
 * All rights reserved.
 *
 * @author A. N. Yzelman
 */

#ifndef _H_GRB_INTERFACES_PREGEL
#define _H_GRB_INTERFACES_PREGEL

#include <graphblas.hpp>
#include <graphblas/utils/parser.hpp>

#include <stdexcept>   // std::runtime_error

namespace grb {

	namespace interfaces {

		struct PregelData {
			bool &active;
			const size_t &num_vertices;
			const size_t &outdegree;
			const size_t &indegree;
			const size_t &round;
		};

		template<
			typename MatrixEntryType
		>
		class Pregel {

			private:

				/** The number of vertices. */
				const size_t n;

				/** The graph */
				grb::Matrix< MatrixEntryType > graph;

				/** Which vertices are still active */
				grb::Vector< bool > activeVertices;

				/** Pre-computed outdegrees. */
				grb::Vector< size_t > outdegrees;

				/** Pre-cominputed indegrees. */
				grb::Vector< size_t > indegrees;

				/** 
				 * Initialises the following fields:
				 *   -# outdegrees
				 */
				void initialize() {
					grb::Semiring<
						size_t, size_t, size_t, size_t,
						grb::operators::add,
						grb::operators::right_assign,
						grb::identities::zero,
						grb::identities::zero
					> patternRing;
					grb::Vector< size_t > ones( n );
					if( grb::set( ones, 1 ) != SUCCESS ) {
						throw std::runtime_error( "Could not set vector ones" );
					}
					if( grb::set( outdegrees, 0 ) != SUCCESS ) {
						throw std::runtime_error( "Could not initialise outdegrees" );
					}
					if( grb::vxm< grb::descriptors::in_place >(
						outdegrees, ones, graph, patternRing
					) != SUCCESS ) {
						throw std::runtime_error( "Could not compute outdegrees" );
					}
					if( grb::set( indegrees, 0 ) != SUCCESS ) {
						throw std::runtime_error( "Could not initialise indegrees" );
					}
					if( grb::mxv< grb::descriptors::in_place >(
						indegrees, graph, ones, patternRing
					) != SUCCESS ) {
						throw std::runtime_error( "Could not compute indegrees" );
					}
				}


			public:

				template< typename fwd_it >
				Pregel(
					const size_t _m, const size_t _n,
					fwd_it _start, const fwd_it _end
				) :
					n( _m ),
					graph( _m, _n ),
					activeVertices( _n ),
					outdegrees( _n ),
					indegrees( _n )
				{
					if( grb::ncols( graph ) != grb::nrows( graph ) ) {
						throw std::runtime_error( "Input graph is bipartite" );
					}
					if( grb::buildMatrixUnique(
						graph, _start, _end, SEQUENTIAL
					) != SUCCESS ) {
						throw std::runtime_error( "Could not build graph" );
					}
					initialize();
				}

				/**
				 * @param[in] in  Where incoming messages are stored. This can hold default
				 *                messages on function entry valid for the first round. If
				 *                \a in does not contain #num_vertices entries it will
				 *                instead be reset to the given identity.
				 */
				template<
					typename IOType,
					typename IncomingMessageType,
					typename OutgoingMessageType,
					typename GlobalProgramData,
					template< typename, typename, typename > class Combiner,
					class CombinerIdentity,
					class Program
				>
				grb::RC execute(
					grb::Vector< IOType > &x,
					grb::Vector< IncomingMessageType > &in,
					grb::Vector< OutgoingMessageType > &out,
					Program program,
					const GlobalProgramData &data,
					const Combiner<
						IncomingMessageType,
						OutgoingMessageType,
						OutgoingMessageType
					> &combiner = Combiner<
						IncomingMessageType,
						OutgoingMessageType,
						OutgoingMessageType
					>(),
					const CombinerIdentity &identity = CombinerIdentity(),
					const size_t max_steps = 1000
				) {
					static_assert( grb::is_operator< Combiner<
							IncomingMessageType,
							OutgoingMessageType,
							OutgoingMessageType
						> >::value,
						"The combiner must be a GraphBLAS operator"
					);

					static_assert( grb::is_associative< Combiner<
							IncomingMessageType,
							OutgoingMessageType,
							OutgoingMessageType
						> >::value,
						"The combiner must be associative."
					);

					// sanity checks
					if( grb::size(x) != n ) {
						return MISMATCH;
					}
					if( grb::size(in) != n ) {
						return MISMATCH;
					}
					if( grb::size(out) != n ) {
						return MISMATCH;
					}

					// define some monoids and semirings
					grb::Monoid< bool, bool, bool,
						grb::operators::logical_or,
						grb::identities::logical_false
					> orMonoid;

					grb::Semiring<
						bool,
						IncomingMessageType,
						IncomingMessageType,
						OutgoingMessageType,
						Combiner,
						//grb::operators::right_assign_if_left,
						grb::operators::right_assign,
						CombinerIdentity,
						//grb::identities::logical_true
						grb::identities::zero
					> ring;

					// set initial round ID
					size_t step = 0;

					// activate all vertices
					grb::RC ret = grb::set( activeVertices, true );

					// set default incoming message
					if( ret == SUCCESS && grb::nnz(in) < n ) {
#ifdef _DEBUG
						if( grb::nnz(in) > 0 ) {
							std::cerr << "Warning: overwriting initial incoming messages because it was not a dense vector\n";
						}
#endif
						ret = grb::set( in, ring.template getZero< IncomingMessageType >() );
					}

					// set default outgoing message
					if( ret == SUCCESS ) {
						ret = grb::set( out, ring.template getZero< OutgoingMessageType >() );
					}

					// check if there are active vertices
					bool thereAreActiveVertices = false;
					if( ret == SUCCESS ) {
						ret = grb::foldl(
							thereAreActiveVertices,
							activeVertices,
							orMonoid
						);
					}

					// return if initialisation failed
					if( ret != SUCCESS ) { return ret; }

					// while there are active vertices, execute
					while( thereAreActiveVertices && ret == SUCCESS && step < max_steps ) {
						// run one step of the program
						ret = grb::eWiseLambda(
							[
								this,
								&x,
								&in,
								&out,
								&program,
								&step,
								&data
							]( const size_t i ) {
								// create Pregel struct
								PregelData pregel = {
									activeVertices[ i ],
									n,
									outdegrees[ i ],
									indegrees[ i ],
									step
								};
								// only execute program on active vertices
								if( activeVertices[ i ] ) {
#ifdef _DEBUG
									std::cout << "Vertex " << i << " remains active in step " << step << "\n";
#endif
									program(
										x[ i ],
										in[ i ],
										out[ i ],
										data,
										pregel
									);
#ifdef _DEBUG
									std::cout << "Vertex " << i << " sends out message " << out[ i ] << "\n";
#endif
								}
							}, x, activeVertices, in, out, outdegrees
						);
						// reset incoming buffer
						if( ret == SUCCESS ) {
							ret = grb::set(
								in,
								//activeVertices,
								ring.template getZero< IncomingMessageType >()
							);
						}
						// execute communication
						if( ret == SUCCESS ) {
							ret = grb::vxm( in, out, graph, ring );
						}
						// reset outgoing buffer
						if( ret == SUCCESS ) {
							ret = grb::set( out, ring.template getZero< OutgoingMessageType >() );
						}
						// check if there is a next round
						if( ret == SUCCESS ) {
							thereAreActiveVertices = false;
							ret = grb::foldl(
								thereAreActiveVertices,
								activeVertices,
								orMonoid
							);
						}
						(void) ++step;
					}

					std::cout << "Info: Pregel exits after " << step << " steps with error code " << ret << " ( " << grb::toString(ret) << " )\n";

					// done
					return ret;
				}

				size_t num_vertices() const noexcept { return n; }

				const grb::Matrix< MatrixEntryType >&
				get_matrix() const noexcept { return graph; }

		};

	} // end namespace ``grb::interfaces''

} // end namespace ``grb''

#endif // end ``_H_GRB_INTERFACES_PREGEL''

