
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

/*
 * @author A. N. Yzelman
 * @date 1st of February, 2022
 */

#ifndef _H_GRB_HYPERDAGS_STATE
#define _H_GRB_HYPERDAGS_STATE

#include <map>
#include <set>
#include <vector>
#include <ostream>
#include <type_traits>

#include <assert.h>


namespace grb {

	namespace internal {

		namespace hyperdags {

			/** \internal The three vertex types in a HyperDAG */
			enum VertexType {
				SOURCE,
				OPERATION,
				OUTPUT
			};

			// 1: all source vertex definitions

			/** \internal The types of source vertices that may be generated. */
			enum SourceVertexType {

				/**
				 * \internal The source is a container with contents that are not generated
				 *           by ALP.
				 */
				CONTAINER,

				/**
				 * \internal The source is a container with contents initialised by a call
				 *           to set.
				 */
				SET

			};

			/** \internal A source vertex. */
			class SourceVertex {

				private:

					/** \internal The type of source */
					enum SourceVertexType type;

					/** \internal The type-wise ID of the vertex */
					size_t local_id;

					/** \internal The global ID of the vertex */
					size_t global_id;


				public:

					SourceVertex(
						const enum SourceVertexType,
						const size_t, const size_t
					) noexcept;

					enum SourceVertexType getType() const noexcept;

					size_t getLocalID() const noexcept;

					size_t getGlobalID() const noexcept;

			};

			/** \internal Helps create a new source vertex */
			class SourceVertexGenerator {

				private:

					/** \internal Map of next local IDs. */
					std::map< enum SourceVertexType, size_t > nextID;


				public:

					SourceVertexGenerator();

					/**
					 * \internal
					 *
					 * @param[in] type the type of source vertex
					 * @param[in] id   a unique global ID
					 *
					 * @returns a new source vertex with an unique local ID
					 *
					 * \endinternal
					 */
					SourceVertex create( const SourceVertexType type, const size_t id );

					/**
					 * \internal
					 *
					 * @returns The total number of source vertex generated of any type.
					 *
					 * \endinternal
					 */
					size_t size() const;

			};

			// 2: everything related to output vertices

			/* TODO maybe not needed-- so far, only one output type
			enum OutputVertexType {
				CONTAINER
			}*/

			/** \internal An output vertex. */
			class OutputVertex {

				private:

					/** \internal The output vertex ID */
					const size_t local_id;

					/** \internal The global vertex ID */
					const size_t global_id;


				public:

					OutputVertex( const size_t, const size_t ) noexcept;

					size_t getLocalID() const noexcept;

					size_t getGlobalID() const noexcept;

			};

			class OutputVertexGenerator {

				private:

					size_t nextID;


				public:

					OutputVertexGenerator() noexcept;

					/**
					 * \internal
					 *
					 * @param[in] id a unique global ID
					 *
					 * @returns a new output vertex with an unique local ID
					 *
					 * \endinternal
					 */
					OutputVertex create( const size_t id );

					/**
					 * \internal
					 *
					 * @returns The total number of output vertices generated.
					 *
					 * \endinternal
					 */
					size_t size() const noexcept;

			};

			// 3: everything related to operation vertices

			/** \internal Which operation an OperationVertex encodes. */
			enum OperationVertexType {

				NNZ_VECTOR

			};

			/** \internal An operation vertex */
			class OperationVertex {

				private:

					const enum OperationVertexType type;

					const size_t local_id;

					const size_t global_id;


				public:

					OperationVertex(
						const enum OperationVertexType,
						const size_t, const size_t
					) noexcept;

					enum OperationVertexType getType() const noexcept;

					size_t getLocalID() const noexcept;

					size_t getGlobalID() const noexcept;

			};


			class OperationVertexGenerator {

				private:

					std::map< enum OperationVertexType, size_t > nextID;


				public:

					OperationVertexGenerator();

					/**
					 * \internal
					 *
					 * @param[in] type type of the new operation vertex
					 * @param[in] id   a unique global ID
					 *
					 * @returns a new output vertex with an unique local ID
					 *
					 * \endinternal
					 */
					OperationVertex create(
						const OperationVertexType type,
						const size_t id
					);

					/**
					 * \internal
					 *
					 * @returns The total number of output vertices generated.
					 *
					 * \endinternal
					 */
					size_t size() const;

			};

			/**
			 * \internal
			 * 
			 * Encodes any hypergraph
			 *
			 * \endinternal
			 */
			class Hypergraph {

				private:

					/** \internal The total number of vertices in the hypergraph. */
					size_t num_vertices;

					/** \internal All hyperedges in the hypergraph. */
					std::vector< std::set< size_t > > hyperedges;


				public:

					Hypergraph() noexcept;

					/**
					 * \internal
					 *
					 * @param[in] start The iterator over vertex IDs that need be added into
					 *                  the hypergraph.
					 * @param[in] end   The end iterator over the vertex IDs to be added.
					 *
					 * There must be at least one vertex ID added, or undefined behaviour will
					 * occur.
					 *
					 * Non-unique elements in the IDs to be added will be filtered out.
					 *
					 * Performance is log-linear in the number of IDs to be added.
					 * \endinternal
					 */
					template< typename FwdIt >
					void createHyperedge( FwdIt start, const FwdIt &end ) {
						static_assert( std::is_unsigned<
							typename std::iterator_traits< FwdIt >::value_type
						>::value, "Expected an iterator over positive integral values" );
						std::set< size_t > toAdd;
						assert( start != end );
						for( ; start != end; ++start ) {
							assert( *start < num_vertices );
							if( toAdd.find(
								static_cast< size_t >( *start )
							) == toAdd.end() ) {
								toAdd.insert( *start );
							}
						}
						hyperedges.push_back( std::move(toAdd) );
					}

					/**
					 * \internal
					 *
					 * Creates a new vertex and returns its global ID.
					 *
					 * \endinternal
					 */
					size_t createVertex() noexcept;

					size_t numVertices() const noexcept;

					/**
					 * \internal
					 *
					 * Prints the hypergraph to a given output stream as a series of
					 * hyperedges. The output format is MatrixMarket-like, where every
					 * hyperedge is assigned a unique ID, and every hyperedge-to-vertex pair
					 * then is printed to \a out.
					 *
					 * @param[in,out] out Where to print the hypergraph to.
					 *
					 * \endinternal
					 */
					void render( std::ostream &out ) const;

			};

			/** \internal Represents a finalised HyperDAG */
			class HyperDAG {

				friend class HyperDAGGenerator;

				private:

					Hypergraph hypergraph;

					size_t num_sources;

					size_t num_operations;

					size_t num_outputs;

					std::vector< SourceVertex > sourceVertices;

					std::vector< OperationVertex > operationVertices;

					std::vector< OutputVertex > outputVertices;

					std::map< size_t, size_t > source_to_global_id;

					std::map< size_t, size_t > operation_to_global_id;

					std::map< size_t, size_t > output_to_global_id;

					std::map< size_t, enum VertexType > global_to_type;

					std::map< size_t, size_t > global_to_local_id;

					template< typename SrcIt, typename OpIt, typename OutIt >
					HyperDAG(
						Hypergraph _hypergraph,
						SrcIt src_start, const SrcIt &src_end,
						OpIt op_start, const OpIt &op_end,
						OutIt out_start, const OutIt &out_end
					) : hypergraph( _hypergraph ),
						num_sources( 0 ), num_operations( 0 ), num_outputs( 0 )
					{
						// static checks
						static_assert( std::is_same< SourceVertex,
								typename std::iterator_traits< SrcIt >::value_type
							>::value,
							"src_start must iterate over elements of type SourceVertex"
						);
						static_assert( std::is_same< OperationVertex,
								typename std::iterator_traits< OpIt >::value_type
							>::value,
							"op_start must iterate over elements of type OperationVertex"
						);
						static_assert( std::is_same< OutputVertex,
								typename std::iterator_traits< OutIt >::value_type
							>::value,
							"out_start must iterate over elements of type OutputVertex"
						);

						// first add sources
						for( ; src_start != src_end; ++src_start ) {
							const size_t local_id = src_start->getLocalID();
							const size_t global_id = src_start->getGlobalID();
							source_to_global_id[ local_id ] = global_id;
							global_to_type[ global_id ] = SOURCE;
							global_to_local_id[ global_id ] = local_id;
							sourceVertices.push_back( *src_start );
							(void) ++num_sources;
						}

						// second, add operations
						for( ; op_start != op_end; ++op_start ) {
							const size_t local_id = op_start->getLocalID();
							const size_t global_id = op_start->getGlobalID();
							operation_to_global_id[ local_id ] = global_id;
							global_to_type[ global_id ] = OPERATION;
							global_to_local_id[ global_id ] = local_id;
							operationVertices.push_back( *op_start );
							(void) ++num_operations;
						}

						// third, add outputs
						for( ; out_start != out_end; ++out_start ) {
							const size_t local_id = out_start->getLocalID();
							const size_t global_id = out_start->getGlobalID();
							output_to_global_id[ local_id ] = global_id;
							global_to_type[ global_id ] = OUTPUT;
							global_to_local_id[ global_id ] = local_id;
							outputVertices.push_back( *out_start );
							(void) ++num_outputs;
						}

						// final sanity check
						assert( num_sources + num_operations + num_outputs == hypergraph.numVertices() );
					}


				public:

					/** \internal @returns The hypergraph representation of the HyperDAG. */
					const Hypergraph & get() const noexcept;

					size_t numSources() const noexcept;

					size_t numOperations() const noexcept;

					size_t numOutputs() const noexcept;

			};

			/** \internal Builds a HyperDAG representation of an ongoing computation. */
			class HyperDAGGenerator {

				private:

					/** \internal The hypergraph under construction. */
					Hypergraph hypergraph;

					/** \internal Map of pointers to source vertices. */
					std::map< const void *, SourceVertex > sourceVertices;

					/** \internal Map of pointers to operation vertices. */
					std::map< const void *, OperationVertex > operationVertices;

					// note: there is no map of OutputVertices because only at the point we
					//       finalize to generate the final HyperDAG do we know for sure what
					//       the output vertices are

					/**
					 * \internal
					 *
					 * During a computation, once an operation executes, its output container
					 * may be an intermediate result or an output. For as long as it is unknown
					 * which it is, those pointers are registered here. Each vertex here must
					 * be assigned a global ID, which are stored as values in this map.
					 *
					 * \endinternal
					 */
					std::map< const void *, size_t > operationOrOutputVertices;

					SourceVertexGenerator sourceGen;

					OperationVertexGenerator operationGen;

					// OutputVertexGenerator is a local field of #finalize()


				public:

					HyperDAGGenerator() noexcept;

					/**
					 * \internal
					 * Sometimes, but not always, do we know for sure that a given operation
					 * generates a source vertex-- for example, #SourceVertexType::SET.
					 *
					 * In such cases, this function should be called to register the source
					 * vertex.
					 *
					 * @param[in] type    The type of source vertex
					 * @param[in] pointer A pointer to the source vertex
					 *
					 * \warning \a type cannot be #SourceVertexType::CONTAINER-- such source
					 *          vertices should be automatically resolved via #addOperation.
					 *
					 * \endinternal
					 */
					void addSource(
						const SourceVertexType type,
						const void * const pointer
					);

					/**
					 * \internal
					 * Registers a new operation with the HyperDAG.
					 *
					 * @param[in] type The type of operation being registered.
					 * @param[in] src_start, src_end Iterators to a set of source pointers.
					 * @param[in] dst_start, dst_end Iterators to a set of destination pointers.
					 *
					 * This function proceeds as follows:
					 *    1. for source pointers in #operationOrOutputVertices, a) upgrade them
					 *       to #OperationVertex, and b) add them to #operationVertices.
					 *    2. for remaining source pointers that are not in #sourceVertices,
					 *       upgrade them to #SourceVertex and add them to #sourceVertices.
					 *    3. for every source pointer k, build an hyperedge. Each hyperedge
					 *       contains only one entry at this point, namely the global ID
					 *       corresponding to each of the k source pointers.
					 *    4. if destination pointers already existed within this HyperDAG, the
					 *       current operation does not correspond to the same ones-- we need
					 *       to create new ones for them. Therefore, we first remove old
					 *       copies. Note that destinations that also dubbed as sources are now
					 *       safe to remove, because we already processed the source pointers.
					 *    5. Assign all destination pointers a new global ID, and add them to
					 *       #operationOrOutputVertices.
					 *    6. Assign all these new global IDs to each of the k hyperedges that
					 *       step 3 started to construct. Thus if there are l destination,
					 *       pointers, we now have k hyperedges with l+1 entries each.
					 *    7. Store those k hyperedges and exit.
					 *
					 * \warning For in-place operations, the output container must be given
					 *          both as a source \em and destination pointer.
					 *
					 * \endinternal
					 */
					template< typename SrcIt, typename DstIt >
					void addOperation(
						const OperationVertexType type,
						SrcIt src_start, const DstIt &src_end,
						DstIt dst_start, const DstIt &dst_end
					);

					/**
					 * \internal
					 *
					 * Assumes that all remaining vertices in #operationVertexOrOutputVertex
					 * are of type #OutputVertex. It then generates a finalised HyperDAG.
					 *
					 * @returns The resulting HyperDAG.
					 *
					 * The current generator instance is left unmodified; this function takes
					 * a snapshot of the current state, and allows its further extension.
					 *
					 * \endinternal
					 */
					HyperDAG finalize() const;

			};

		} // end namespace grb::internal::hyperdags

	} // end namespace grb::internal

} // end namespace grb

#endif // end _H_GRB_HYPERDAGS_STATE

