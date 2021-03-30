
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
#include <functional>  // std::function

namespace grb {

	namespace interfaces {

		template<
			typename MessageType,
			template< typename, typename, typename > class CombinerOp,
			class CombinerIdentity,
			typename CombinedType = MessageType
		>
		class Pregel {

			private:

				typedef CombinerOp< MessageType, CombinedType, CombinedType > Combiner;

				static_assert( grb::is_operator< Combiner >::value,
					"The combiner must be a GraphBLAS operator"
				);

				static_assert( grb::is_associative< Combiner >::value,
					"The combiner must be associative."
				);

				/** The number of vertices. */
				const size_t n;

				/** The graph */
				grb::Matrix< void > graph;

				/** Incoming messages after broadcast */
				grb::Vector< MessageType > in;

				/** Outgoing messages for broadcast */
				grb::Vector< MessageType > out;

				/** Which vertices are still active */
				grb::Vector< bool > activeVertices;

				/** Pre-computed outdegrees. */
				grb::Vector< size_t > outdegrees;

				/** 
				 * Initialises the following fields:
				 *   -# outdegrees
				 */
				void initialize() {
					grb::Semiring< size_t > intRing;
					grb::Vector< size_t > ones( n );
					if( grb::set( ones, 1 ) != SUCCESS ) {
						throw std::runtime_error( "Could not set vector ones" );
					}
					if( grb::set( outdegrees, 1 ) != SUCCESS ) {
						throw std::runtime_error( "Could not initialise outdegrees" );
					}
					if( grb::vxm< grb::descriptors::in_place >(
						outdegrees, ones, graph, intRing
					) != SUCCESS ) {
						throw std::runtime_error( "Could not compute outdegrees" );
					}
				}


			public:

				struct PregelData {
					const CombinedType &incoming_message;
					MessageType &outgoing_message;
					bool &active;
					const size_t &num_vertices;
					const size_t &outdegree;
					const size_t &round;
				};

				template< typename InputType >
				Pregel( const grb::Matrix< InputType > &_g ) :
					n( grb::nrows( _g ) ),
					graph( n, grb::ncols( _g ) ),
					in( n ), out( n ),
					activeVertices( n ),
					outdegrees( n )
				{
					if( grb::ncols( graph ) != grb::nrows( graph ) ) {
						throw std::runtime_error( "Input graph is bipartite" );
					}
					if( grb::buildMatrixUnique(
						graph,
						_g.begin(),
						_g.end()
					) != SUCCESS ) {
						throw std::runtime_error( "Could not copy input graph" );
					}
					initialize();
				}

				Pregel( const std::string &filename ) : Pregel(
					grb::utils::MatrixFileReader< void >( filename )
				) {};

				Pregel( grb::utils::MatrixFileReader< void > file ) :
					n( file.n() ),
					graph( n, file.m() ),
					in( n ), out( n ),
					activeVertices( n ),
					outdegrees( n )
				{
					if( grb::ncols( graph ) != grb::nrows( graph ) ) {
						throw std::runtime_error( "Input graph is bipartite" );
					}
					if( grb::buildMatrixUnique(
						graph,
						file.begin(),
						file.end(),
						SEQUENTIAL
					) != SUCCESS ) {
						throw std::runtime_error( "Error during ingestion of input graph" );
					}
					initialize();
				}

				template<
					typename IOType,
					typename GlobalProgramData
				>
				grb::RC execute(
					grb::Vector< IOType > &x,
					std::function< void(
						IOType&,
						const GlobalProgramData&,
						PregelData&
					)> program,
					const GlobalProgramData &data,
					const size_t max_steps = 1000
				) {
					// sanity checks
					if( grb::size(x) != n ) {
						return MISMATCH;
					}

					// define some monoids and semirings
					grb::Monoid< bool, bool, bool,
						grb::operators::logical_or,
						grb::identities::logical_false
					> orMonoid;

					grb::Semiring<
						MessageType, MessageType, MessageType, MessageType,
						CombinerOp,
						grb::operators::right_assign,
						CombinerIdentity,
						CombinerIdentity
					> ring;

					// set initial round ID
					size_t step = 0;

					// activate all vertices
					grb::RC ret = grb::set( activeVertices, true );

					// set default incoming message
					if( ret == SUCCESS ) {
						ret = grb::set( in, ring.template getZero< CombinedType >() );
					}

					// set default outgoing message
					if( ret == SUCCESS ) {
						ret = grb::set( out, ring.template getZero< MessageType >() );
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
								&program,
								&step,
								&data
							]( const size_t i ) {
								// create Pregel struct
								PregelData pregel = {
									in[ i ],
									out[ i ],
									activeVertices[ i ],
									n,
									outdegrees[ i ],
									step
								};
								// only execute program on active vertices
								if( activeVertices[ i ] ) {
									std::cout << "Vertex " << i << " remains active in step " << step << "\n";
									program(
										x[ i ],
										data,
										pregel
									);
								}
							}, x, activeVertices, in, out, outdegrees
						);
						// reset incoming buffer
						if( ret == SUCCESS ) {
							ret = grb::set(
								in,
								//activeVertices,
								ring.template getZero< CombinedType >()
							);
						}
						// execute communication
						if( ret == SUCCESS ) {
							ret = grb::vxm( in, out, graph, ring );
						}
						// reset outgoing buffer
						if( ret == SUCCESS ) {
							ret = grb::set( out, ring.template getZero< MessageType >() );
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

					std::cout << "Pregel exits after " << step << " steps with error code " << ret << " ( " << grb::toString(ret) << " )\n";

					// done
					return ret;
				}

				size_t num_vertices() const noexcept { return n; }

		};

	} // end namespace ``grb::interfaces''

} // end namespace ``grb''

#endif // end ``_H_GRB_INTERFACES_PREGEL''

