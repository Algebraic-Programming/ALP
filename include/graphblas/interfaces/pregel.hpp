
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
 * @author A. N. Yzelman
 */

#ifndef _H_GRB_INTERFACES_PREGEL
#define _H_GRB_INTERFACES_PREGEL

#include <graphblas.hpp>
#include <graphblas/utils/parser.hpp>

#include <stdexcept>   // std::runtime_error


namespace grb {

	namespace interfaces {

		namespace config {

			/**
			 * The set of sparsification strategies supported by the ALP/Pregel
			 * interface.
			 */
			enum SparsificationStrategy {

				/** No sparsification. */
				NONE = 0,

				/**
				 * Always applies the sparsification procedure.
				 *
				 * Does not consider whether the resulting operation would reduce the number
				 * of vertex entries. This variant was tested against #NONE for
				 * #out_sparsify, and found to be slower always.
				 *
				 * This strategy necessarily always applied on the Pregel::ActiveVertices
				 * vector.
				 */
				ALWAYS,

				/**
				 * Sparsify only when the resulting vector would indeed be sparser.
				 *
				 * \note This strategy should \em not be applied to #Pregel::ActiveVertices
				 *       since doing so requires computing the number of active vertices,
				 *       which has the same complexity as actually sparsifying that vector.
				 *
				 * \todo This variant has never been tested for \a out_sparsify.
				 */
				WHEN_REDUCED,

				/**
				 * Sparsify only when the resulting vector would have half (or less) its
				 * current number of nonzeroes.
				 *
				 * \note This strategy should \em not be applied to #Pregel::ActiveVertices
				 *       since doing so requires computing the number of active vertices,
				 *       which has the same complexity as actually sparsifying that vector.
				 *
				 * \todo This variant has never been tested for \a out_sparsify.
				 */
				WHEN_HALVED

			};

			/**
			 * What sparsification strategy should be applied to the outgoing
			 * messages.
			 *
			 * Only #NONE and #ALWAYS have been tested, with #NONE being faster on all
			 * test cases.
			 */
			constexpr const SparsificationStrategy out_sparsify = NONE;

		} // end namespace grb::interfaces::config

		/**
		 * The state of the vertex-center Pregel program that the user  may interface
		 * with.
		 *
		 * The state includes global data as well as vertex-centric state. The global
		 * state is umodifiable and includes:
		 *  - #PregelState::num_vertices,
		 *  - #PregelState::num_edges, and
		 *  - #PregelState::round.
		 *
		 * Vertex-centric state can be either constant or modiable:
		 *  - static vertex-centric state: #PregelState::indegree,
		 *    #PregelState::outdegree, and #PregelState::vertexID.
		 *  - modifiable vertex-centric state: #PregelState::voteToHalt, and
		 *    #PregelState::active.
		 */
		struct PregelState {

			/**
			 * Represents whether the current vertex is active.
			 *
			 * Since this struct is only to-be used within the computational phase of a
			 * vertex-centric program, this always reads <tt>true</tt> on the start of a
			 * round.
			 *
			 * The program may set this field to <tt>false</tt> which will cause this
			 * vertex to no longer trigger computational steps during subsequent rounds.
			 *
			 * An inactive vertex will no longer broadcast messages.
			 *
			 * If all vertices are inactive the program terminates.
			 */
			bool &active;

			/**
			 * Represents whether this (active) vertex votes to terminate the program.
			 *
			 * On start of a round, this entry is set to <tt>false</tt>. If all active
			 * vertices set this to <tt>true</tt>, the program will terminate after the
			 * current round.
			 */
			bool &voteToHalt;

			/**
			 * The number of vertices in the global graph.
			 */
			const size_t &num_vertices;

			/**
			 * The number of edges in the global graph.
			 */
			const size_t &num_edges;

			/**
			 * The out-degree of this vertex.
			 */
			const size_t &outdegree;

			/**
			 * The in-degree of this vertex.
			 */
			const size_t &indegree;

			/**
			 * The current round the vertex-centric program is currently executing.
			 */
			const size_t &round;

			/**
			 * A unique ID of this vertex.
			 *
			 * This number is an unsigned integer between 0 (inclusive) and
			 * the number of vertices the underlying graph holds (exclusive).
			 */
			const size_t &vertexID;

		};

		/**
		 * A Pregel run-time instance.
		 *
		 * Pregel wraps around graph data and executes computations on said graph. A
		 * runtime thus is constructed from graph, and enables running any Pregel
		 * algorithm on said graph.
		 */
		template<
			typename MatrixEntryType
		>
		class Pregel {

			private:

				/** \internal The number of vertices of the underlying #graph. */
				const size_t n;

				/** \internal The number of edges of the underlying #graph. */
				size_t nz;

				/** \internal The graph to run vertex-centric programs over. */
				grb::Matrix< MatrixEntryType > graph;

				/** \internal Which vertices are still active. */
				grb::Vector< bool > activeVertices;

				/** \internal Which vertices voted to halt. */
				grb::Vector< bool > haltVotes;

				/** \internal A buffer used to sparsify #activeVertices. */
				grb::Vector< bool > buffer;

				/** \internal Pre-computed outdegrees. */
				grb::Vector< size_t > outdegrees;

				/** \internal Pre-cominputed indegrees. */
				grb::Vector< size_t > indegrees;

				/** \internal Global vertex IDs. */
				grb::Vector< size_t > IDs;

				/**
				 * \internal
				 * Initialises the following fields:
				 *   -# outdegrees
				 *   -# indegrees
				 *   -# IDs
				 * Other fields are set on program start.
				 * \endinternal
				 */
				void initialize() {
					grb::Semiring<
						grb::operators::add< size_t >,
						grb::operators::right_assign_if< bool, size_t, size_t >,
						grb::identities::zero,
						grb::identities::logical_true
					> ring;
					grb::Vector< size_t > ones( n );
					if( grb::set( ones, 1 ) != SUCCESS ) {
						throw std::runtime_error( "Could not set vector ones" );
					}
					if( grb::set( outdegrees, 0 ) != SUCCESS ) {
						throw std::runtime_error( "Could not initialise outdegrees" );
					}
					if( grb::mxv< grb::descriptors::dense >(
							outdegrees, graph, ones, ring
						) != SUCCESS
					) {
						throw std::runtime_error( "Could not compute outdegrees" );
					}
					if( grb::set( indegrees, 0 ) != SUCCESS ) {
						throw std::runtime_error( "Could not initialise indegrees" );
					}
					if( grb::mxv<
						grb::descriptors::dense | grb::descriptors::transpose_matrix
					>(
						indegrees, graph, ones, ring
					) != SUCCESS ) {
						throw std::runtime_error( "Could not compute indegrees" );
					}
					if( grb::set< grb::descriptors::use_index >(
							IDs, 0
						) != SUCCESS
					) {
						throw std::runtime_error( "Could not compute vertex IDs" );
					}
				}


			protected:

				/**
				 * \internal
				 * Internal constructor for the cases where the number of vertix IDs,
				 * \a _n, is already known.
				 * \endinternal
				 */
				template< typename IType >
				Pregel(
					const size_t _n,
					IType _start, const IType _end,
					const grb::IOMode _mode
				) :
					n( _n ),
					graph( _n, _n ),
					activeVertices( _n ),
					haltVotes( _n ),
					buffer( _n ),
					outdegrees( _n ),
					indegrees( _n ),
					IDs( _n )
				{
					if( grb::ncols( graph ) != grb::nrows( graph ) ) {
						throw std::runtime_error( "Input graph is bipartite" );
					}
					if( grb::buildMatrixUnique(
						graph, _start, _end, _mode
					) != SUCCESS ) {
						throw std::runtime_error( "Could not build graph" );
					}
					nz = grb::nnz( graph );
					initialize();
				}


			public:

				/**
				 * Constructs a Pregel instance from input iterators over some graph.
				 *
				 * @tparam IType The type of the input iterator.
				 *
				 * @param[in] _m The maximum vertex ID for excident edges.
				 * @param[in] _n The maximum vertex ID for incident edges.
				 *
				 * \note This is equivalent to the row- and column- size of an input matrix
				 *       which represents the input graph.
				 *
				 * \note If these values are not known, please scan the input iterators to
				 *       derive these values prior to calling this constructor. On
				 *       compelling reasons why such functionality would be useful to
				 *       provide as a standard factory method, please feel welcome to submit
				 *       an issue.
				 *
				 * \warning The graph is assumed to have contiguous IDs -- i.e., every
				 *          vertex ID in the range of 0 (inclusive) to the maximum of \a m
				 *          and \a n (exclusive) has at least one excident or at least one
				 *          incident edge.
				 *
				 * @param[in] _start An iterator pointing to the start element of an
				 *                   a collection of edges.
				 * @param[in] _end   An iterator matching \a _start in end position.
				 *
				 * All edges to be ingested thus are contained within \a _start and \a end.
				 *
				 * @param[in] _mode Whether sequential or parallel I/O is to be used.
				 *
				 * The value of \a _mode only takes effect when there are multiple user
				 * processes, such as for example when executing over a distributed-memory
				 * cluster. The choice between sequential and parallel I/O should be thus:
				 *  - If the edges pointed to by \a _start and \a _end correspond to the
				 *    \em entire set of edges on \em each process, then the I/O mode should
				 *    be #grb::SEQUENTIAL;
				 *  - If the edges pointed to by \a _start and \a _end correspond to
				 *    \em different sets of edges on each different process while their
				 *    union represents the graph to be ingested, then the I/O mode should be
				 *    #grb::PARALLEL.
				 *
				 * On errors during ingestion, this constructor throws exceptions.
				 */
				template< typename IType >
				Pregel(
					const size_t _m, const size_t _n,
					IType _start, const IType _end,
					const grb::IOMode _mode
				) : Pregel( std::max( _m, _n ), _start, _end, _mode ) {}

				/**
				 * Executes a given vertex-centric \a program on this graph.
				 *
				 * The program must be a static function that returns void and takes five
				 * input arguments:
				 *  - a reference to a vertex-defined state. The type of this reference may
				 *    be defined by the program, but has to match the element type of
				 *    \a vertex_state passed to this function.
				 *  - a const-reference to an incoming message. The type of this reference
				 *    may be defined by the program, but has to match the element type of
				 *    \a in passed to this function. It must furthermore be compatible with
				 *    the domains of \a Op (see below).
				 *  - a reference to an outgoing message. The type of this reference may be
				 *    defined by the program, but has to match the element type of \a out
				 *    passed to this function. It must furthermore be compatible with the
				 *    domains of \a Op (see below).
				 *  - a const-reference to a program-defined type. The function of this
				 *    argument is to collect global read-only algorithm parameters.
				 *  - a reference to an instance of #grb::interfaces::PregelState. The
				 *    function of this argument is two-fold: 1) make available global read-
				 *    only statistics of the graph the algorithm is executing on, and to 2)
				 *    control algorithm termination conditions.
				 *
				 * The program will be called during each round of a Pregel computation. The
				 * program is expected to compute something based on the incoming message
				 * and vertex-local state, and (optionally) generate an outgoing message.
				 * After each round, the outgoing message at all vertices are broadcast to
				 * all its neighbours. The Pregel runtime, again for each vertex, reduces
				 * all incoming messages into a single message, after which the next round
				 * of computation starts, after which the procedure is repeated.
				 *
				 * The program terminates in one of two ways:
				 *  1. there are no more active vertices; or
				 *  2. all active vertices vote to halt.
				 *
				 * On program start, i.e., during the first round, all vertices are active.
				 * During the computation phase, any vertex can set itself inactive for
				 * subsequent rounds by setting #grb::interfaces::PregelState::active to
				 * <tt>false</tt>. Similarly, any active vertex can vote to halt by setting
				 * #grb::interfaces::PregelState::voteToHalt to <tt>true</tt>.
				 *
				 * Reduction of incoming messages to a vertex will occur through an user-
				 * defined monoid given by:
				 *
				 * @tparam Op The binary operation of the monoid. This includes its domain.
				 * @tparam Id The identity element of the monoid.
				 *
				 * The following template arguments will be automatically inferred:
				 *
				 * @tparam Program             The type of the program to-be executed.
				 * @tparam IOType              The type of the state of a single vertex.
				 * @tparam GlobalProgramData   The type of globally accessible read-only
				 *                             program data.
				 * @tparam IncomingMessageType The type of an incoming message.
				 * @tparam OutgoingMessageType The type of an outgoing message.
				 *
				 * The arguments to this function are as follows:
				 *
				 * @param[in] program The vertex-centric program to execute.
				 *
				 * The same Pregel runtime instance hence can be re-used to execute multiple
				 * algorithms on the same graph.
				 *
				 * Vertex-centric programs have both vertex-local and global state:
				 *
				 * @param[in] vertex_state A vector that contains the state of each vertex.
				 * @param[in] global_data  Global read-only state for the given \a program.
				 *
				 * The capacity, size, and number of nonzeroes of \a vertex_state must equal
				 * the maximum vertex ID.
				 *
				 * Finally, in the ALP spirit which aims to control all relevant performance
				 * aspects, the workspace required by the Pregel runtime must be pre-
				 * allocated and passed in:
				 *
				 * @param[in] in  Where incoming messages are stored. Any initial values may
				 *                or may not be ignored, depending on the \a program
				 *                behaviour during the first round of computation.
				 *
				 * @param[in] out Where outgoing messages are stored. Any initial values
				 *                will be ignored.
				 *
				 * The capacities and sizes of \a in and \a out must equal the maximum vertex
				 * ID. For sparse vectors \a in with more than zero nonzeroes, all initial
				 * contents will be overwritten by the identity of the reduction monoid. Any
				 * initial contents for \a out will always be ignored as every round of
				 * computation starts with the outgoing message set to the monoid identity.
				 *
				 * \note Thus if the program requires some initial incoming messages to be
				 *       present during the first round of computation, those may be passed
				 *       as part of a dense vectors \a in.
				 *
				 * The contents of \a in and \a out after termination of a vertex-centric
				 * function are undefined, including when this function returns
				 * #grb::SUCCESS. Output of the program should be part of the vertex-centric
				 * state recorded in \a vertex_state.
				 *
				 * Some statistics are returned after a vertex-centric program terminates:
				 *
				 * @param[out] rounds The number of rounds the Pregel program has executed.
				 *                    The initial value to \a rounds will be ignored.
				 *
				 * The contents of this field shall be undefined when this function does not
				 * return #grb::SUCCESS.
				 *
				 * Vertex-programs execute in rounds and could, if the given program does
				 * not infer proper termination conditions, run forever. To curb the number
				 * of rounds, the following \em optional parameter may be given:
				 *
				 * @param[in] out_buffer An optional buffer area that should only be set
				 *                       whenever the #config::out_sparsify configuration
				 *                       parameter is not set to #NONE. If that is the case,
				 *                       then \a out_buffer should have size and capacity
				 *                       equal to the maximum vertex ID.
				 *
				 * @param[in] max_rounds The maximum number of rounds the \a program may
				 *                       execute. Once reached and not terminated, the
				 *                       program will forcibly terminate.
				 *
				 * To turn off termination after a maximum number of rounds, \a max_rounds
				 * may be set to zero. This is also the default.
				 *
				 * Executing a Pregel function returns one of the following error codes:
				 *
				 * @returns #grb::SUCCESS  The \a program executed (and terminated)
				 *                         successfully.
				 * @returns #grb::MISMATCH At least one of \a vertex_state, \a in, or \a out
				 *                         is not of the required size.
				 * @returns #grb::ILLEGAL  At least one of \a vertex_state, \a in, or \a out
				 *                         does not have the required capacity.
				 * @returns #grb::ILLEGAL  If \a vertex_state is not dense.
				 * @returns #grb::PANIC    In case an unrecoverable error was encountered
				 *                         during execution.
				 */
				template<
					class Op,
					template< typename > class Id,
					class Program,
					typename IOType,
					typename GlobalProgramData,
					typename IncomingMessageType,
					typename OutgoingMessageType
				>
				grb::RC execute(
					const Program program,
					grb::Vector< IOType > &vertex_state,
					const GlobalProgramData &data,
					grb::Vector< IncomingMessageType > &in,
					grb::Vector< OutgoingMessageType > &out,
					size_t &rounds,
					grb::Vector< OutgoingMessageType > &out_buffer =
						grb::Vector< OutgoingMessageType >(0),
					const size_t max_rounds = 0
				) {
					static_assert( grb::is_operator< Op >::value &&
							grb::is_associative< Op >::value,
						"The combiner must be an associate operator"
					);
					static_assert( std::is_same< typename Op::D1, IncomingMessageType >::value,
						"The combiner left-hand input domain should match the incoming message type." );
					static_assert( std::is_same< typename Op::D1, IncomingMessageType >::value,
						"The combiner right-hand input domain should match the incoming message type." );
					static_assert( std::is_same< typename Op::D1, IncomingMessageType >::value,
						"The combiner output domain should match the incoming message type." );

					// set default output
					rounds = 0;

					// sanity checks
					if( grb::size(vertex_state) != n ) {
						return MISMATCH;
					}
					if( grb::size(in) != n ) {
						return MISMATCH;
					}
					if( grb::size(out) != n ) {
						return MISMATCH;
					}
					if( grb::capacity(vertex_state) != n ) {
						return ILLEGAL;
					}
					if( grb::capacity(in) != n ) {
						return ILLEGAL;
					}
					if( grb::capacity(out) != n ) {
						return ILLEGAL;
					}
					if( config::out_sparsify && grb::capacity(out_buffer) != n ) {
						return ILLEGAL;
					}
					if( grb::nnz(vertex_state) != n ) {
						return ILLEGAL;
					}

					// define some monoids and semirings
					grb::Monoid<
						grb::operators::logical_or< bool >,
						grb::identities::logical_false
					> orMonoid;

					grb::Monoid<
						grb::operators::logical_and< bool >,
						grb::identities::logical_true
					> andMonoid;

					grb::Semiring<
						Op,
						grb::operators::left_assign_if<
							IncomingMessageType, bool, IncomingMessageType
						>,
						Id,
						grb::identities::logical_true
					> ring;

					// set initial round ID
					size_t step = 0;

					// activate all vertices
					grb::RC ret = grb::set( activeVertices, true );

					// initialise halt votes to all-false
					if( ret == SUCCESS ) {
						ret = grb::set( haltVotes, false );
					}

					// set default incoming message
					if( ret == SUCCESS && grb::nnz(in) < n ) {
#ifdef _DEBUG
						if( grb::nnz(in) > 0 ) {
							std::cerr << "Overwriting initial incoming messages since it was not a "
								<< "dense vector\n";
						}
#endif
						ret = grb::set( in, Id< IncomingMessageType >::value() );
					}

					// reset outgoing buffer
					size_t out_nnz = n;
					if( ret == SUCCESS ) {
						ret = grb::set( out, Id< OutgoingMessageType >::value() );
					}

					// return if initialisation failed
					if( ret != SUCCESS ) {
						assert( ret == FAILED );
						std::cerr << "Error: initialisation failed, but if workspace holds full "
							<< "capacity, initialisation should never fail. Please submit a bug "
							<< "report.\n";
						return PANIC;
					}

					// while there are active vertices, execute
					while( ret == SUCCESS ) {

						assert( max_rounds == 0 || step < max_rounds );
						// run one step of the program
						ret = grb::eWiseLambda(
							[
								this,
								&vertex_state,
								&in,
								&out,
								&program,
								&step,
								&data
							]( const size_t i ) {
								// create Pregel struct
								PregelState pregel = {
									activeVertices[ i ],
									haltVotes[ i ],
									n,
									nz,
									outdegrees[ i ],
									indegrees[ i ],
									step,
									IDs[ i ]
								};
								// only execute program on active vertices
								assert( activeVertices[ i ] );
#ifdef _DEBUG
								std::cout << "Vertex " << i << " remains active in step " << step
									<< "\n";
#endif
								program(
									vertex_state[ i ],
									in[ i ],
									out[ i ],
									data,
									pregel
								);
#ifdef _DEBUG
								std::cout << "Vertex " << i << " sends out message " << out[ i ]
									<< "\n";
#endif
							}, activeVertices, vertex_state, in, out, outdegrees, haltVotes
						);

						// increment counter
						(void) ++step;

						// check if everyone voted to halt
						if( ret == SUCCESS ) {
							bool halt = true;
							ret = grb::foldl< grb::descriptors::structural >(
								halt, haltVotes, activeVertices, andMonoid
							);
							assert( ret == SUCCESS );
							if( ret == SUCCESS && halt ) {
#ifdef _DEBUG
								std::cout << "\t All active vertices voted to halt; "
									<< "terminating Pregel program.\n";
#endif
								break;
							}
						}

						// update active vertices
						if( ret == SUCCESS ) {
#ifdef _DEBUG
							std::cout << "\t Number of active vertices was "
								<< grb::nnz( activeVertices ) << ", and ";
#endif
							ret = grb::clear( buffer );
							ret = ret ? ret : grb::set( buffer, activeVertices, true );
							std::swap( buffer, activeVertices );
#ifdef _DEBUG
							std::cout << " has now become " << grb::nnz( activeVertices ) << "\n";
#endif
						}

						// check if there is a next round
						const size_t curActive = grb::nnz( activeVertices );
						if( ret == SUCCESS && curActive == 0 ) {
#ifdef _DEBUG
							std::cout << "\t All vertices are inactive; "
								<< "terminating Pregel program.\n";
#endif
							break;
						}

						// check if we exceed the maximum number of rounds
						if( max_rounds > 0 && step > max_rounds ) {
#ifdef _DEBUG
							std::cout << "\t Maximum number of Pregel rounds met "
								<< "without the program returning a valid termination condition. "
								<< "Exiting prematurely with a FAILED error code.\n";
#endif
							ret = FAILED;
							break;
						}

#ifdef _DEBUG
						std::cout << "\t Starting message exchange\n";
#endif

						// reset halt votes
						if( ret == SUCCESS ) {
							ret = grb::clear( haltVotes );
							ret = ret ? ret : grb::set< grb::descriptors::structural >(
								haltVotes, activeVertices, false
							);
						}

						// reset incoming buffer
						if( ret == SUCCESS ) {
							ret = grb::clear( in );
							ret = ret ? ret : grb::set< grb::descriptors::structural >(
								in, activeVertices, Id< IncomingMessageType >::value()
							);
						}

						// execute communication
						if( ret == SUCCESS ) {
							ret = grb::vxm< grb::descriptors::structural >(
								in, activeVertices, out, graph, ring
							);
						}

						// sparsify and reset outgoing buffer
						if( config::out_sparsify && ret == SUCCESS ) {
							if( config::out_sparsify == config::ALWAYS ||
								(config::out_sparsify == config::WHEN_REDUCED && out_nnz > curActive) ||
								(config::out_sparsify == config::WHEN_HALVED && curActive <= out_nnz/2)
							) {
								ret = grb::clear( out_buffer );
								ret = ret ? ret : grb::set< grb::descriptors::structural >(
										out_buffer, activeVertices, Id< OutgoingMessageType >::value()
									);
								std::swap( out, out_buffer );
								out_nnz = curActive;
							}
						}

#ifdef _DEBUG
						std::cout << "\t Resetting outgoing message fields and "
							<< "starting next compute round\n";
#endif

					}

#ifdef _DEBUG
					if( grb::spmd<>::pid() == 0 ) {
						std::cout << "Info: Pregel exits after " << step
							<< " rounds with error code " << ret
							<< " ( " << grb::toString(ret) << " )\n";
					}
#endif

					// done
					rounds = step;
					return ret;
				}

				/**
				 * Queries the maximum vertex ID for programs running on this Pregel
				 * instance.
				 *
				 * @returns The maximum vertex ID.
				 */
				size_t num_vertices() const noexcept { return n; }

				/**
				 * Queries the number of edges of the graph this Pregel instance has been
				 * constructed over.
				 *
				 * @returns The number of edges within the underlying graph.
				 */
				size_t num_edges() const noexcept { return nz; }

				/**
				 * Returns the ALP/GraphBLAS matrix representation of the underlying
				 * graph.
				 *
				 * This is useful when an application prefers to sometimes use vertex-
				 * centric algorithms and other times prefers direct ALP/GraphBLAS
				 * algorithms.
				 *
				 * @returns The underlying ALP/GraphBLAS matrix corresponding to the
				 *          underlying graph.
				 */
				const grb::Matrix< MatrixEntryType > & get_matrix() const noexcept {
					return graph;
				}

		};

	} // end namespace ``grb::interfaces''

} // end namespace ``grb''

#endif // end ``_H_GRB_INTERFACES_PREGEL''

