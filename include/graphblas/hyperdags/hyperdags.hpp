
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

namespace grb {

	namespace internal {

		namespace hyperdags {

			// 1: all source vertex definition

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
				SET;

			}

			/** \internal A source vertex. */
			class SourceVertex {

				private:

					/** \internal The type of source */
					const enum SourceVertexType type;

					/** \internal The type-wise ID of the vertex */
					const size_t local_id;

					/** \internal The global ID of the vertex */
					const size_t global_id;


				public:

					SourceVertex(
						const enum SourceVertexType,
						const size_t, const size_t
					);

					enum SourceVertexType getType();

					size_t getLocalID();

					size_t getGlobalID();

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
				CONTAINER;
			}*/

			/** \internal An output vertex. */
			class OutputVertex {

				private:

					/** \internal The output vertex ID */
					const size_t local_id;

					/** \internal The global vertex ID */
					const size_t global_id;


				public:

					OutputVertex( const size_t, const size_t );

					size_t getLocalID();

					size_t getGlobalID();

			};

			class OutputVertexGenerator {

				private:

					size_t nextID;


				public:

					OutputVertexGenerator();

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
					size_t size() const;

			};

			// 3: everything related to operation vertices

			/** \internal Which operation an OperationVertex encodes. */
			enum OperationVertexType {

				NNZ_VECTOR;

			}

			/** \internal An operation vertex */
			class OperationVertex {

				private:

					const enum OperationType type;

					const size_t local_id;

					const size_t global_id;


				public:

					OperationVertex(
						const enum OperationType,
						const size_t, const size_t
					);

					enum OperationType getType();

					size_t getLocalID();

					size_t getGlobalID();

			};


			class OperationVertexTranslator {

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
						const void * const id
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

					Hypergraph();

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
							typename std::iterator_traits< FwdIt >::value_type,
						>::value, "Expected an iterator over positive integral values" );
						std::set< size_t > toAdd;
						assert( start != end );
						for( ; start != end; ++start ) {
							assert( *it < num_vertices );
							if( toAdd.find(
								static_cast< size_t >( *it )
							) == toAdd.end() ) {
								toAdd.insert( *it );
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
					size_t createVertex();

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
					void render( std::ostream &out ) {
						const size_t net_num = 0;
						for( const auto &net : hyperedges ) {
							for( const auto &id : net ) {
								out << net_num << " " << id << "\n";
							}
							(void) ++net_num;
						}
						out << std::flush;
					}

			};

			/** \internal Represents a finalised HyperDAG. */
			class HyperDAG {

				friend class HyperDAGGenerator;

				private:

					Hypergraph hypergraph;

					const size_t num_sources;

					const size_t num_operations;

					const size_t num_outputs;

					std::map< size_t, size_t > source_to_global_id;

					std::map< size_t, size_t > operation_to_global_id;

					std::map< size_t, size_t > output_to_global_id;

					std::map< size_t, enum VertexType > global_to_type;

					std::map< size_t, size_t > global_to_local_id;


				public:

					/** \internal @returns The hypergraph representation of the HyperDAG. */
					const Hypergraph & get() const;

			};

			/** \internal Builds a HyperDAG representation of an ongoing computation. */
			class HyperDAGGenerator {

				private:

					/** \internal The hypergraph under construction. */
					Hypergraph hypergraph;

					/** \internal Map of pointers to source vertices. */
					std::map< void *, SourceVertex > sourceVertices;

					/** \internal Map of pointers to operation vertices. */
					std::map< void *, OperationVertex > operationVertices;

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
					std::map< void *, size_t > operationOrOutputVertices;

					SourceVertexGenerator;

					OperationVertexGenerator;

					// OutputVertexGenerator is a local field of #finalize()


				public:

					HyperDAGGenerator();

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
					 * \endinternal
					 */
					HyperDAG finalize();

			};

		} // end namespace grb::internal::hyperdags

	} // end namespace grb::internal

} // end namespace grb

#endif // end _H_GRB_HYPERDAGS_STATE

