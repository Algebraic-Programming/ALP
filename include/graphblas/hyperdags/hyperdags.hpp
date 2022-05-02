
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
#include <iostream>
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
				 * \internal Scalars are always handled as a new source. We do not track
				 * whether the same scalars are re-used, because we cannot reliably do so
				 * (due to a lack of an alp::Scalar).
				 */
				SCALAR,

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

			const constexpr size_t numSourceVertexTypes = 3;

			const constexpr enum SourceVertexType
				allSourceVertexTypes[ numSourceVertexTypes ] =
			{
				SCALAR,
				CONTAINER,
				SET
			};

			std::string toString( const enum SourceVertexType type ) noexcept;

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

				NNZ_VECTOR,

				CLEAR_VECTOR,

				SET_VECTOR_ELEMENT,

				/** \internal The monoid-operator version, specifically */
				DOT,
				
				SET_USING_MASK_AND_VECTOR,
				
				SET_USING_MASK_AND_SCALAR,
				
				SET_FROM_VECTOR,
				
				ZIP,

				E_WISE_APPLY_VECTOR_VECTOR_VECTOR_OP,
				
				FOLDR_VECTOR_SCALAR_MONOID,
				
				FOLDL_SCALAR_VECTOR_MASK_MONOID,
				
				EWISELAMBDA,
				
				BUILD_VECTOR,
				
				BUILD_VECTOR_WITH_VALUES,
				
				SIZE,
				
				EWISEAPPLY_VECTOR_VECTOR,
				
				EWISEAPPLY_VECTOR_BETA,
				
				EWISEAPPLY_VECTOR_VECTOR_BETA,
				
				EWISEAPPLY_VECTOR_VECTOR_VECTOR_BETA,
				
				EWISEAPPLY_VECTOR_VECTOR_ALPHA_VECTOR,
				
				EWISEAPPLY_VECTOR_VECTOR_ALPHA_VECTOR_OP,
				
				EWISEAPPLY_VECTOR_MASK_VECTOR_VECTOR_OP,
				
				EWISEAPPLY_VECTOR_SCALAR_MONOID,
				
				EWISEAPPLY_SCALAR_VECTOR_MONOID,
				
				EWISEAPPLY_VECTOR_MASK_VECTOR_VECTOR_MONOID,
				
				EWISEAPPLY_VECTOR_VECTOR_VECTOR_MONOID,
				
				EWISEAPPLY_MUL_ADD,
				
				EWISEAPPLY_MUL_ADD_FOUR_VECTOR,
				
				EWISEAPPLY_MUL_ADD_THREE_VECTOR_ALPHA,
				
				EWISEAPPLY_MUL_ADD_THREE_VECTOR_CHI,
				
				EWISEAPPLY_MUL_ADD_FOUR_VECTOR_CHI,
				
				EWISEAPPLY_MUL_ADD_FOUR_VECTOR_CHI_RING,
				
				EWISEAPPLY_MUL_ADD_THREE_VECTOR_BETA,
				
				EWISEAPPLY_MUL_ADD_THREE_VECTOR_ALPHA_GAMMA,
				
				EWISEAPPLY_MUL_ADD_TWO_VECTOR_ALPHA_BETA,
				
				EWISEAPPLY_MUL_ADD_TWO_VECTOR_ALPHA_BETA_GAMMA,
				
				EWISEAPPLY_MATRIX_MATRIX_MATRIX_MULMONOID_PHASE,
				
				EWISEAPPLY_MATRIX_MATRIX_MATRIX_OPERATOR_PHASE,
				
				SET_MATRIX_MATRIX,
				
				SET_MATRIX_MATRIX_INPUT2,
				
				SET_MATRIX_MATRIX_DOUBLE,
				
				MXM_MATRIX_MATRIX_MATRIX_SEMIRING,
				
				MXM_MATRIX_MATRIX_MATRIX_MONOID,
				
				OUTER,
				
				MXV_VECTOR_VECTOR_MATRIX_VECTOR_VECTOR,
				
				MXV_VECTOR_VECTOR_MATRIX_VECTOR_VECTOR_RING,
				
				VXM_VECTOR_VECTOR_VECTOR_VECTOR_RING,
				
				VXM_VECTOR_VECTOR_VECTOR_VECTOR_MATRIX_ADD,
				
				UNZIP_VECTOR_VECTOR_VECTOR,
				
				ZIP_MATRIX_VECTOR_VECTOR,
				
				CLEAR_MATRIX,
				
				EWISEMULADD_VECTOR_VECTOR_VECTOR_GAMMA_RING,
				
				EWISEMULADD_VECTOR_VECTOR_BETA_GAMMA_RING,
				
				EWISEMULADD_VECTOR_ALPHA_VECTOR_GAMMA_RING,
				
				EWISEMULADD_VECTOR_ALPHA_BETA_VECTOR_RING,
				
				EWISEMULADD_VECTOR_ALPHA_BETA_GAMMA_RING,
				
				EWISEMULADD_VECTOR_VECTOR_VECTOR_VECTOR_RING,
				
				VXM_VECTOR_VECTOR_VECTOR_MATRIX,
				
				VXM_VECTOR_VECTOR_VECTOR_MATRIX_ADD_MUL,
				
				VXM_VECTOR_VECTOR_MATRIX_RING,
				
				MXV_VECTOR_VECTOR_MATRIX_VECTOR_RING,
				
				MXV_VECTOR_VECTOR_MATRIX_VECTOR_VECTOR_R,
				
				MXV_VECTOR_VECTOR_MATRIX_VECTOR_VECTOR_A,
				
				MXV_VECTOR_MATRIX_VECTOR_RING,
				
				MXV_VECTOR_MATRIX_VECTOR_ADD_MUL,
				
				BUILDMATRIXUNIQUE_MATRIX_START_END_MODE,
				
				CAPACITY_VECTOR,
				
				CAPACITY_MATRIX,
				
				RESIZE,
				
				GETID_VECTOR,
				
				GETID_MATRIX,
				
				EWISELAMBDA_FUNC_MATRIX,
				
				EWISELAMBDA_FUNC_MATRIX_VECTOR
				
				
				
				
			};

			const constexpr size_t numOperationVertexTypes = 73;

			const constexpr enum OperationVertexType
				allOperationVertexTypes[ numOperationVertexTypes ] =
			{
				NNZ_VECTOR,
				CLEAR_VECTOR,
				SET_VECTOR_ELEMENT,
				DOT,
				SET_USING_MASK_AND_VECTOR,
				SET_USING_MASK_AND_SCALAR,
				SET_FROM_VECTOR,
				ZIP,
				E_WISE_APPLY_VECTOR_VECTOR_VECTOR_OP,
				FOLDR_VECTOR_SCALAR_MONOID,
				FOLDL_SCALAR_VECTOR_MASK_MONOID,
				EWISELAMBDA,
				BUILD_VECTOR,
				BUILD_VECTOR_WITH_VALUES,
				SIZE,
				EWISEAPPLY_VECTOR_VECTOR,
				EWISEAPPLY_VECTOR_BETA,
				EWISEAPPLY_VECTOR_VECTOR_BETA,
				EWISEAPPLY_VECTOR_VECTOR_VECTOR_BETA,
				EWISEAPPLY_VECTOR_VECTOR_ALPHA_VECTOR,
				EWISEAPPLY_VECTOR_VECTOR_ALPHA_VECTOR_OP,
				EWISEAPPLY_VECTOR_MASK_VECTOR_VECTOR_OP,
				EWISEAPPLY_VECTOR_SCALAR_MONOID,
				EWISEAPPLY_SCALAR_VECTOR_MONOID,
				EWISEAPPLY_VECTOR_MASK_VECTOR_VECTOR_MONOID,
				EWISEAPPLY_VECTOR_VECTOR_VECTOR_MONOID,
				EWISEAPPLY_MUL_ADD,
				EWISEAPPLY_MUL_ADD_FOUR_VECTOR,
				EWISEAPPLY_MUL_ADD_THREE_VECTOR_ALPHA,
				EWISEAPPLY_MUL_ADD_THREE_VECTOR_CHI,
				EWISEAPPLY_MUL_ADD_FOUR_VECTOR_CHI,
				EWISEAPPLY_MUL_ADD_FOUR_VECTOR_CHI_RING,
				EWISEAPPLY_MUL_ADD_THREE_VECTOR_BETA,
				EWISEAPPLY_MUL_ADD_THREE_VECTOR_ALPHA_GAMMA,
				EWISEAPPLY_MUL_ADD_TWO_VECTOR_ALPHA_BETA,
				EWISEAPPLY_MUL_ADD_TWO_VECTOR_ALPHA_BETA_GAMMA,
				EWISEAPPLY_MATRIX_MATRIX_MATRIX_MULMONOID_PHASE,
				EWISEAPPLY_MATRIX_MATRIX_MATRIX_OPERATOR_PHASE,
				SET_MATRIX_MATRIX,
				SET_MATRIX_MATRIX_INPUT2,
				SET_MATRIX_MATRIX_DOUBLE,
				MXM_MATRIX_MATRIX_MATRIX_SEMIRING,
				MXM_MATRIX_MATRIX_MATRIX_MONOID,
				OUTER,
				MXV_VECTOR_VECTOR_MATRIX_VECTOR_VECTOR,
				MXV_VECTOR_VECTOR_MATRIX_VECTOR_VECTOR_RING,
				VXM_VECTOR_VECTOR_VECTOR_VECTOR_RING,
				VXM_VECTOR_VECTOR_VECTOR_VECTOR_MATRIX_ADD,
				UNZIP_VECTOR_VECTOR_VECTOR,
				ZIP_MATRIX_VECTOR_VECTOR,
				CLEAR_MATRIX,
				EWISEMULADD_VECTOR_VECTOR_VECTOR_GAMMA_RING,
				EWISEMULADD_VECTOR_VECTOR_BETA_GAMMA_RING,
				EWISEMULADD_VECTOR_ALPHA_VECTOR_GAMMA_RING,
				EWISEMULADD_VECTOR_ALPHA_BETA_VECTOR_RING,
				EWISEMULADD_VECTOR_ALPHA_BETA_GAMMA_RING,
				EWISEMULADD_VECTOR_VECTOR_VECTOR_VECTOR_RING,
				VXM_VECTOR_VECTOR_VECTOR_MATRIX,
				VXM_VECTOR_VECTOR_VECTOR_MATRIX_ADD_MUL,
				VXM_VECTOR_VECTOR_MATRIX_RING,
				MXV_VECTOR_VECTOR_MATRIX_VECTOR_RING,
				MXV_VECTOR_VECTOR_MATRIX_VECTOR_VECTOR_R,
				MXV_VECTOR_VECTOR_MATRIX_VECTOR_VECTOR_A,
				MXV_VECTOR_MATRIX_VECTOR_RING,
				MXV_VECTOR_MATRIX_VECTOR_ADD_MUL,
				BUILDMATRIXUNIQUE_MATRIX_START_END_MODE,
				CAPACITY_VECTOR,
				CAPACITY_MATRIX,
				RESIZE,
				GETID_VECTOR,
				GETID_MATRIX,
				EWISELAMBDA_FUNC_MATRIX,
				EWISELAMBDA_FUNC_MATRIX_VECTOR
				
			};

			std::string toString( const enum OperationVertexType ) noexcept;

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

					/** \internal The total number of pins in the hypergraph. */
					size_t num_pins;


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
#ifdef _DEBUG
						std::cerr << "in createHyperedge\n\t adding ( ";
						std::vector< size_t > warn;
#endif
						std::set< size_t > toAdd;
						assert( start != end );
						for( ; start != end; ++start ) {
							assert( *start < num_vertices );
							if( toAdd.find(
								static_cast< size_t >( *start )
							) == toAdd.end() ) {
								toAdd.insert( *start );
								(void) ++num_pins;
#ifdef _DEBUG
								std::cerr << *start << " ";
#endif
							} else {
#ifdef _DEBUG
								warn.push_back( *start );
#endif
							}
						}
						hyperedges.push_back( std::move(toAdd) );
#ifdef _DEBUG
						std::cerr << ")\n";
						if( warn.size() > 0 ) {
							std::cerr << "\t Warning: the following edges were multiply-defined: ( ";
							for( const auto &id : warn ) {
								std::cerr << id << " ";
							}
						}
						std::cerr << ")\n\t exiting\n";
#endif
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

					size_t numHyperedges() const noexcept;

					size_t numPins() const noexcept;

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

					HyperDAG(
						Hypergraph _hypergraph,
						const std::vector< SourceVertex > &_srcVec,
						const std::vector< OperationVertex > &_opVec,
						const std::vector< OutputVertex > &_outVec
					);


				public:


					/** \internal @returns The hypergraph representation of the HyperDAG. */
					Hypergraph get() const noexcept;

					size_t numSources() const noexcept;

					size_t numOperations() const noexcept;

					size_t numOutputs() const noexcept;

					std::vector< SourceVertex >::const_iterator sourcesBegin() const;

					std::vector< SourceVertex >::const_iterator sourcesEnd() const;

			};

			/** \internal Builds a HyperDAG representation of an ongoing computation. */
			class HyperDAGGenerator {

				private:

					/** \internal The hypergraph under construction. */
					Hypergraph hypergraph;

					/**
					 * \internal
					 *
					 * Once new source vertices are created, they are recorded here. This
					 * storage differs from #sourceVertices in that the latter only keeps
					 * track of currently active source vertices, and identifies them by
					 * a pointer.
					 *
					 * \endinternal
					 */
					std::vector< SourceVertex > sourceVec;

					/**
					 * \internal
					 *
					 * Once new operation vertices are created, they are recorded here. This
					 * storage differs from #operationVertices in that the latter only keeps
					 * track of currently active source vertices, and identifies them by
					 * a pointer.
					 *
					 * \endinternal
					 */
					std::vector< OperationVertex > operationVec;

					/** \internal Map of pointers to source vertices. */
					std::map< const void *, SourceVertex > sourceVertices;

					/** \internal Map of pointers to operation vertices. */
					std::map< const void *, OperationVertex > operationVertices;

					// note: there is no map of OutputVertices because only at the point we
					//       finalize to generate the final HyperDAG do we know for sure what
					//       the output vertices are. The same applies to an `outputVec`.

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
					std::map< const void *,
						std::pair< size_t, OperationVertexType >
					> operationOrOutputVertices;

					SourceVertexGenerator sourceGen;

					OperationVertexGenerator operationGen;

					// OutputVertexGenerator is a local field of #finalize()

					size_t addAnySource(
						const SourceVertexType type,
						const void * const pointer
					);


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
					 *       Otherwise, if already a source, add it from #sourceVertices
					 *       directly.
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
					) {
						static_assert( std::is_same< const void *,
								typename std::iterator_traits< SrcIt >::value_type
							>::value,
							"Sources should be given as const void pointers"
						);
						static_assert( std::is_same< const void *,
								typename std::iterator_traits< DstIt >::value_type
							>::value,
							"Destinations should be given as const void pointers"
						);
#ifdef _DEBUG
						std::cerr << "In HyperDAGGen::addOperation( "
							<< toString( type ) << ", ... )\n"
							<< "\t sourceVertices size: " << sourceVertices.size() << "\n"
							<< "\t sourceVec size: " << sourceVec.size() << "\n";
#endif

						// steps 1, 2, and 3
						std::vector< std::vector< size_t > > hyperedges;
						for( ; src_start != src_end; ++src_start ) {
#ifdef _DEBUG
							std::cerr << "\t processing source " << *src_start << "\n";
#endif
							std::vector< size_t > toPush;
							// step 1
							const auto &it = operationOrOutputVertices.find( *src_start );
							if( it == operationOrOutputVertices.end() ) {
								// step 2
								const auto alreadySource = sourceVertices.find( *src_start );
								if( alreadySource == sourceVertices.end() ) {
#ifdef _DEBUG
									std::cerr << "\t creating new entry in sourceVertices\n";
#endif
									toPush.push_back( addAnySource( CONTAINER, *src_start ) );
								} else {
#ifdef _DEBUG
									std::cerr << "\t found source in sourceVertices\n";
#endif
									toPush.push_back( alreadySource->second.getGlobalID() );
								}
							} else {
#ifdef _DEBUG
								std::cerr << "\t found source in operationOrOutputVertices\n";
#endif
								// step 2
								const auto &remove = operationVertices.find( it->first );
								if( remove != operationVertices.end() ) {
#ifdef _DEBUG
									std::cerr << "\t found source in operationVertices; removing it\n";
#endif
									operationVertices.erase( remove );
								}
#ifdef _DEBUG
								std::cerr << "\t creating new entry in operationOrOutputVertices\n";
#endif
								const size_t global_id = it->second.first;
								const auto &operationVertex = operationGen.create(
									it->second.second, global_id
								);
								operationVertices.insert( std::make_pair( it->first, operationVertex ) );
								operationVec.push_back( operationVertex );
								operationOrOutputVertices.erase( it );
								toPush.push_back( global_id );
							}
							// step 3
							assert( toPush.size() == 1 );
							hyperedges.push_back( toPush );
						}

						// step 4, 5, and 6
						for( ; dst_start != dst_end; ++dst_start ) {
#ifdef _DEBUG
							std::cerr << "\t processing destination " << *dst_start << "\n";
#endif
							// step 4
							{
								const auto &it = sourceVertices.find( *dst_start );
								if( it != sourceVertices.end() ) {
#ifdef _DEBUG
									std::cerr << "\t destination found in sources-- "
										<< "removing it from there\n";
#endif
									sourceVertices.erase( it );
								}
							}
							{
								const auto &it = operationVertices.find( *dst_start );
								if( it != operationVertices.end() ) {
#ifdef _DEBUG
									std::cerr << "\t destination found in operations-- "
										<< "removing it from there\n";
#endif
									operationVertices.erase( it );
								}
							}
							{
								const auto &it = operationOrOutputVertices.find( *dst_start );
								if( it != operationOrOutputVertices.end() ) {
									std::cerr << "WARNING (hyperdags::addOperation): an unconsumed output "
										<< "container was detected. This indicates the existance of "
										<< "an ALP primitive whose output is never used.\n";
#ifdef _DEBUG
									std::cerr << "\t destination found in operationsOrOutput-- "
										<< "removing it from there\n";
#endif
									operationOrOutputVertices.erase( it );
								}
							}
							// step 5
							const size_t global_id = hypergraph.createVertex();
							operationOrOutputVertices.insert( std::make_pair( *dst_start,
								std::make_pair( global_id, type )
							) );
#ifdef _DEBUG
							std::cerr << "\t created a new operation vertex with global ID "
								<< global_id << "\n";
#endif
							// step 6
							for( auto &hyperedge : hyperedges ) {
								hyperedge.push_back( global_id );
							}
						}

						// step 7
						for( const auto &hyperedge : hyperedges ) {
#ifdef _DEBUG
							std::cerr << "\t storing a hyperedge of size "
								<< hyperedge.size() << "\n";
#endif
							hypergraph.createHyperedge( hyperedge.begin(), hyperedge.end() );
						}
					}

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

