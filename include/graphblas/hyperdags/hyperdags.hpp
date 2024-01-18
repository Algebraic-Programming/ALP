
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
 * @file
 *
 * Provides mechanisms to track HyperDAG representations of ALP programs
 *
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
				 * due to a lack of an grb::Scalar.
				 */
				SCALAR,

				/**
				 * \internal The source is a container managed by ALP.
				 */
				CONTAINER,

				/**
				 * \internal The source is an iterator passed to ALP.
				 */
				ITERATOR,

				/**
				 * \internal The source is a user integer passed to ALP, usually signifying
				 *           an index or a size.
				 */
				USER_INT

			};

			/** \internal The number of source vertex types. */
			const constexpr size_t numSourceVertexTypes = 4;

			/** \internal An array of all source vertex types. */
			const constexpr enum SourceVertexType
				allSourceVertexTypes[ numSourceVertexTypes ] =
			{
				SCALAR,
				CONTAINER,
				ITERATOR,
				USER_INT
			};

			/** \internal @returns The type, as a string, of a source vertex. */
			std::string toString( const enum SourceVertexType type ) noexcept;

			/** \internal A source vertex. */
			class SourceVertex {

				private:

					/** \internal The type of source */
					enum SourceVertexType type;

					/** \internal The ID amongst vertices of the same type */
					size_t local_id;

					/** \internal The global ID of the vertex */
					size_t global_id;


				public:

					/**
					 * \internal The default source vertex constructor.
					 *
					 * @param[in] type The type of the vertex.
					 * @param[in] lid  The ID of vertices of the same type.
					 * @param[in] gid  The global ID of the vertex.
					 */
					SourceVertex(
						const enum SourceVertexType type,
						const size_t lid, const size_t gid
					) noexcept;

					/** \internal @returns The vertex type. */
					enum SourceVertexType getType() const noexcept;

					/** \internal @returns The type ID. */
					size_t getLocalID() const noexcept;

					/** \internal @returns The global ID. */
					size_t getGlobalID() const noexcept;

			};

			/** \internal Helps create a new source vertex */
			class SourceVertexGenerator {

				private:

					/** \internal Map of next local IDs. */
					std::map< enum SourceVertexType, size_t > nextID;


				public:

					/** \internal Default constructor. */
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

			/** \internal The types of output vertices that may be generated. */
			enum OutputVertexType {

				/**
				 * \internal The output is an ALP container.
				 */
				CONTAINER_OUTPUT

			};

			/** \internal The number of distinct output vertex types. */
			const constexpr size_t numOutputVertexTypes = 1;

			/** \internal An array of output vertex types. */
			const constexpr enum OutputVertexType
				allOutputVertexTypes[ numOutputVertexTypes ] =
			{
				CONTAINER_OUTPUT
			};

			/** \internal @returns A string form of a given output vertex type. */
			std::string toString( const enum OutputVertexType type ) noexcept;

			/** \internal An output vertex. */
			class OutputVertex {

				private:

					/** \internal The type of the output */
					enum OutputVertexType type;

					/** \internal The output vertex ID */
					const size_t local_id;

					/** \internal The global vertex ID */
					const size_t global_id;


				public:

					/**
					 * \internal Default constructor.
					 *
					 * @param[in] lid The ID within vertices of this type.
					 * @param[in] gid The global vertex ID.
					 *
					 * Recall there is only one output vertex type, hence the precise type is
					 * not a constructor argument.
					 */
					OutputVertex( const size_t lid, const size_t gid ) noexcept;

					/** \internal @returns The type of this output vertex. */
					enum OutputVertexType getType() const noexcept;

					/** \internal @returns The ID amongst vertices of the same type. */
					size_t getLocalID() const noexcept;

					/** \internal @returns The ID amongst all vertices. */
					size_t getGlobalID() const noexcept;

			};

			/** \internal Helps create output vertices. */
			class OutputVertexGenerator {

				private:

					/** \internal Keeps track of the next output vertex ID. */
					size_t nextID;


				public:

					/** \internal Default constructor. */
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

				NNZ_MATRIX,

				CLEAR_VECTOR,

				SET_VECTOR_ELEMENT,

				DOT,

				SET_USING_VALUE,

				SET_USING_MASK_AND_VECTOR,

				SET_USING_MASK_AND_SCALAR,

				SET_FROM_VECTOR,

				ZIP,

				E_WISE_APPLY_VECTOR_VECTOR_VECTOR_OP,

				FOLDR_VECTOR_SCALAR_MONOID,

				FOLDR_VECTOR_MASK_SCALAR_MONOID,

				FOLDL_SCALAR_VECTOR_MONOID,

				FOLDL_SCALAR_VECTOR_MASK_MONOID,

				EWISELAMBDA,

				BUILD_VECTOR,

				BUILD_VECTOR_WITH_VALUES,

				SIZE,

				NROWS,

				NCOLS,

				EWISEAPPLY_VECTOR_ALPHA_BETA_OP,

				EWISEAPPLY_VECTOR_ALPHA_VECTOR_OP,

				EWISEAPPLY_VECTOR_VECTOR_BETA_OP,

				EWISEAPPLY_VECTOR_VECTOR_VECTOR_OP,

				EWISEAPPLY_VECTOR_MASK_ALPHA_BETA_OP,

				EWISEAPPLY_VECTOR_MASK_ALPHA_VECTOR_OP,

				EWISEAPPLY_VECTOR_MASK_VECTOR_BETA_OP,

				EWISEAPPLY_VECTOR_MASK_VECTOR_VECTOR_OP,

				EWISEAPPLY_VECTOR_ALPHA_BETA_MONOID,

				EWISEAPPLY_VECTOR_ALPHA_VECTOR_MONOID,

				EWISEAPPLY_VECTOR_VECTOR_BETA_MONOID,

				EWISEAPPLY_VECTOR_VECTOR_VECTOR_MONOID,

				EWISEAPPLY_VECTOR_MASK_ALPHA_BETA_MONOID,

				EWISEAPPLY_VECTOR_MASK_ALPHA_VECTOR_MONOID,

				EWISEAPPLY_VECTOR_MASK_VECTOR_BETA_MONOID,

				EWISEAPPLY_VECTOR_MASK_VECTOR_VECTOR_MONOID,

				EWISE_MUL_ADD,

				EWISE_MUL_ADD_FOUR_VECTOR,

				EWISE_MUL_ADD_THREE_VECTOR_ALPHA,

				EWISE_MUL_ADD_THREE_VECTOR_CHI,

				EWISE_MUL_ADD_FOUR_VECTOR_CHI,

				EWISE_MUL_ADD_FOUR_VECTOR_CHI_RING,

				EWISE_MUL_ADD_THREE_VECTOR_BETA,

				EWISE_MUL_ADD_THREE_VECTOR_ALPHA_GAMMA,

				EWISE_MUL_ADD_TWO_VECTOR_ALPHA_BETA,

				EWISE_MUL_ADD_TWO_VECTOR_ALPHA_BETA_GAMMA,

				EWISEAPPLY_MATRIX_MATRIX_MATRIX_MULMONOID_PHASE,

				EWISEAPPLY_MATRIX_MATRIX_MATRIX_OPERATOR_PHASE,

				SET_MATRIX_MATRIX,

				SET_MATRIX_MATRIX_INPUT2,

				MXM_MATRIX_MATRIX_MATRIX_SEMIRING,

				MXM_MATRIX_MATRIX_MATRIX_MONOID,

				OUTER,

				UNZIP_VECTOR_VECTOR_VECTOR,

				ZIP_MATRIX_VECTOR_VECTOR_VECTOR,

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

				RESIZE_MATRIX,

				GETID_VECTOR,

				GETID_MATRIX,

				EWISELAMBDA_FUNC_MATRIX,

				VXM_GENERIC_VECTOR_VECTOR_VECTOR_VECTOR_MATRIX_ADD_MUL,

				VXM_VECTOR_VECTOR_VECTOR_VECTOR_MATRIX_ADD_MUL,

				VXM_VECTOR_VECTOR_MATRIX_ADD_MUL,

				FOLDL_VECTOR_BETA_OP,

				FOLDL_VECTOR_VECTOR_BETA_OP,

				FOLDL_VECTOR_BETA_MONOID,

				FOLDL_VECTOR_VECTOR_BETA_MONOID,

				FOLDL_VECTOR_VECTOR_MONOID,

				FOLDL_VECTOR_VECTOR_VECTOR_MONOID,

				FOLDL_VECTOR_VECTOR_VECTOR_OP,

				FOLDL_VECTOR_VECTOR_OP,

				FOLDR_APLHA_VECTOR_MONOID,

				FOLDR_APLHA_VECTOR_OPERATOR,

				FOLDR_VECTOR_VECTOR_OPERATOR,

				FOLDR_VECTOR_VECTOR_VECTOR_OPERATOR,

				FOLDR_VECTOR_VECTOR_MONOID,

				FOLDR_VECTOR_VECTOR_VECTOR_MONOID,

				EWISEMUL_VECTOR_VECTOR_VECTOR_RING,

				EWISEMUL_VECTOR_ALPHA_VECTOR_RING,

				EWISEMUL_VECTOR_VECTOR_BETA_RING,

				EWISEMUL_VECTOR_ALPHA_BETA_RING,

				EWISEMUL_VECTOR_VECTOR_VECTOR_VECTOR_RING,

				EWISEMUL_VECTOR_VECTOR_ALPHA_VECTOR_RING,

				EWISEMUL_VECTOR_VECTOR_VECTOR_BETA_RING,

				EWISEMUL_VECTOR_VECTOR_ALPHA_BETA_RING,

				EWISELAMBDA_FUNC_VECTOR,

				SELECT_MATRIX_MATRIX,

				SELECT_LAMBDA_MATRIX_MATRIX

			};

			/** \internal How many operation vertex types exist. */
			const constexpr size_t numOperationVertexTypes = 108;

			/** \internal An array of all operation vertex types. */
			const constexpr enum OperationVertexType
				allOperationVertexTypes[ numOperationVertexTypes ] =
			{
				NNZ_VECTOR,
				NNZ_MATRIX,
				CLEAR_VECTOR,
				SET_VECTOR_ELEMENT,
				DOT,
				SET_USING_VALUE,
				SET_USING_MASK_AND_VECTOR,
				SET_USING_MASK_AND_SCALAR,
				SET_FROM_VECTOR,
				ZIP,
				E_WISE_APPLY_VECTOR_VECTOR_VECTOR_OP,
				FOLDR_VECTOR_SCALAR_MONOID,
				FOLDR_VECTOR_MASK_SCALAR_MONOID,
				FOLDL_SCALAR_VECTOR_MONOID,
				FOLDL_SCALAR_VECTOR_MASK_MONOID,
				EWISELAMBDA,
				BUILD_VECTOR,
				BUILD_VECTOR_WITH_VALUES,
				SIZE,
				NROWS,
				NCOLS,
				EWISEAPPLY_VECTOR_ALPHA_BETA_OP,
				EWISEAPPLY_VECTOR_ALPHA_VECTOR_OP,
				EWISEAPPLY_VECTOR_VECTOR_BETA_OP,
				EWISEAPPLY_VECTOR_VECTOR_VECTOR_OP,
				EWISEAPPLY_VECTOR_MASK_ALPHA_BETA_OP,
				EWISEAPPLY_VECTOR_MASK_ALPHA_VECTOR_OP,
				EWISEAPPLY_VECTOR_MASK_VECTOR_BETA_OP,
				EWISEAPPLY_VECTOR_MASK_VECTOR_VECTOR_OP,
				EWISEAPPLY_VECTOR_ALPHA_BETA_MONOID,
				EWISEAPPLY_VECTOR_ALPHA_VECTOR_MONOID,
				EWISEAPPLY_VECTOR_VECTOR_BETA_MONOID,
				EWISEAPPLY_VECTOR_VECTOR_VECTOR_MONOID,
				EWISEAPPLY_VECTOR_MASK_ALPHA_BETA_MONOID,
				EWISEAPPLY_VECTOR_MASK_ALPHA_VECTOR_MONOID,
				EWISEAPPLY_VECTOR_MASK_VECTOR_BETA_MONOID,
				EWISEAPPLY_VECTOR_MASK_VECTOR_VECTOR_MONOID,
				EWISE_MUL_ADD,
				EWISE_MUL_ADD_FOUR_VECTOR,
				EWISE_MUL_ADD_THREE_VECTOR_ALPHA,
				EWISE_MUL_ADD_THREE_VECTOR_CHI,
				EWISE_MUL_ADD_FOUR_VECTOR_CHI,
				EWISE_MUL_ADD_FOUR_VECTOR_CHI_RING,
				EWISE_MUL_ADD_THREE_VECTOR_BETA,
				EWISE_MUL_ADD_THREE_VECTOR_ALPHA_GAMMA,
				EWISE_MUL_ADD_TWO_VECTOR_ALPHA_BETA,
				EWISE_MUL_ADD_TWO_VECTOR_ALPHA_BETA_GAMMA,
				EWISEAPPLY_MATRIX_MATRIX_MATRIX_MULMONOID_PHASE,
				EWISEAPPLY_MATRIX_MATRIX_MATRIX_OPERATOR_PHASE,
				SET_MATRIX_MATRIX,
				SET_MATRIX_MATRIX_INPUT2,
				MXM_MATRIX_MATRIX_MATRIX_SEMIRING,
				MXM_MATRIX_MATRIX_MATRIX_MONOID,
				OUTER,
				UNZIP_VECTOR_VECTOR_VECTOR,
				ZIP_MATRIX_VECTOR_VECTOR_VECTOR,
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
				RESIZE_MATRIX,
				GETID_VECTOR,
				GETID_MATRIX,
				EWISELAMBDA_FUNC_MATRIX,
				VXM_GENERIC_VECTOR_VECTOR_VECTOR_VECTOR_MATRIX_ADD_MUL,
				VXM_VECTOR_VECTOR_VECTOR_VECTOR_MATRIX_ADD_MUL,
				VXM_VECTOR_VECTOR_MATRIX_ADD_MUL,
				FOLDL_VECTOR_BETA_OP,
				FOLDL_VECTOR_VECTOR_BETA_OP,
				FOLDL_VECTOR_BETA_MONOID,
				FOLDL_VECTOR_VECTOR_BETA_MONOID,
				FOLDL_VECTOR_VECTOR_MONOID,
				FOLDL_VECTOR_VECTOR_VECTOR_MONOID,
				FOLDL_VECTOR_VECTOR_VECTOR_OP,
				FOLDL_VECTOR_VECTOR_OP,
				FOLDR_APLHA_VECTOR_MONOID,
				FOLDR_APLHA_VECTOR_OPERATOR,
				FOLDR_VECTOR_VECTOR_OPERATOR,
				FOLDR_VECTOR_VECTOR_VECTOR_OPERATOR,
				FOLDR_VECTOR_VECTOR_MONOID,
				FOLDR_VECTOR_VECTOR_VECTOR_MONOID,
				EWISEMUL_VECTOR_VECTOR_VECTOR_RING,
				EWISEMUL_VECTOR_ALPHA_VECTOR_RING,
				EWISEMUL_VECTOR_VECTOR_BETA_RING,
				EWISEMUL_VECTOR_ALPHA_BETA_RING,
				EWISEMUL_VECTOR_VECTOR_VECTOR_VECTOR_RING,
				EWISEMUL_VECTOR_VECTOR_ALPHA_VECTOR_RING,
				EWISEMUL_VECTOR_VECTOR_VECTOR_BETA_RING,
				EWISEMUL_VECTOR_VECTOR_ALPHA_BETA_RING,
				EWISELAMBDA_FUNC_VECTOR,
				SELECT_MATRIX_MATRIX,
				SELECT_LAMBDA_MATRIX_MATRIX
			};

			/** \internal @returns The operation vertex type as a string. */
			std::string toString( const enum OperationVertexType ) noexcept;

			/** \internal An operation vertex */
			class OperationVertex {

				private:

					/** \internal The type of the vertex. */
					const enum OperationVertexType type;

					/** \internal The ID amongst vertices of the same type. */
					const size_t local_id;

					/** \internal The ID amongst all vertices. */
					const size_t global_id;


				public:

					/**
					 * \internal
					 * Base constructor.
					 *
					 * @param[in] type The type of the new operation vertex.
					 * @param[in] lid  An ID amongst vertices of the same type.
					 * @param[in] gid  An ID unique amongst all vertices.
					 * \endinternal
					 */
					OperationVertex(
						const enum OperationVertexType type,
						const size_t lid, const size_t gid
					) noexcept;

					/** \internal @returns The type of this vertex. */
					enum OperationVertexType getType() const noexcept;

					/**
					 * \internal
					 * @returns An ID unique amongst all vertices of the same type.
					 * \endinternal
					 */
					size_t getLocalID() const noexcept;

					/**
					 * \internal
					 * @returns An ID unique amongst all vertices, regardless of type.
					 * \endinternal
					 */
					size_t getGlobalID() const noexcept;

			};

			/** \internal Helps generate operation vertices. */
			class OperationVertexGenerator {

				private:

					/**
					 * \internal
					 * A map that keeps track of the number of vertices of each type.
					 * \endinternal
					 */
					std::map< enum OperationVertexType, size_t > nextID;


				public:

					/** \internal Base constructor. */
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
			 * Encodes any directed hypergraph that may yet grow.
			 *
			 * \endinternal
			 */
			class DHypergraph {

				private:

					/** \internal The total number of vertices in the hypergraph. */
					size_t num_vertices;

					/**
					 * \internal
					 *
					 * All hyperedges in the hypergraph.
					 *
					 * \endinternal
					 */
					std::map< size_t, std::set< size_t > > hyperedges;

					/** \internal The total number of pins in the hypergraph. */
					size_t num_pins;


				public:

					DHypergraph() noexcept;

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
					void appendHyperedge(
						const size_t source,
						FwdIt start, const FwdIt &end
					) {
						static_assert( std::is_unsigned<
							typename std::iterator_traits< FwdIt >::value_type
						>::value, "Expected an iterator over positive integral values" );
#ifdef _DEBUG
						std::cerr << "in appendHyperedge\n\t source " << source
							<< "\n\t adds destinations ( ";
						std::vector< size_t > warn;
#endif
						const auto it = hyperedges.find( source );
						if( it == hyperedges.end() ) {
							hyperedges[ source ] = std::set< size_t >();
						}

						std::set< size_t > &toAdd = hyperedges[ source ];
						for( ; start != end; ++start ) {
							assert( *start < num_vertices );
							if( toAdd.find( static_cast< size_t >( *start ) ) == toAdd.end() ) {
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

					/** \internal @returns The number of vertices in the current graph. */
					size_t numVertices() const noexcept;

					/** \internal @returns The number of hyperedges in the current graph. */
					size_t numHyperedges() const noexcept;

					/** \internal @returns The total number of pins in the current graph. */
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

					/** \internal The underlying hypergraph. */
					DHypergraph hypergraph;

					/** \internal The number of source vertices. */
					size_t num_sources;

					/** \internal The number of operation vertices. */
					size_t num_operations;

					/** \internal The number of output vertices. */
					size_t num_outputs;

					/** \internal A vector of source vertices. */
					std::vector< SourceVertex > sourceVertices;

					/** \internal A vector of operation vertices. */
					std::vector< OperationVertex > operationVertices;

					/** \internal A vector of output vertices. */
					std::vector< OutputVertex > outputVertices;

					/** \internal A map from source vertex IDs to global IDs. */
					std::map< size_t, size_t > source_to_global_id;

					/** \internal A map from operation vertex IDs to global IDs. */
					std::map< size_t, size_t > operation_to_global_id;

					/** \internal A map from output vertex IDs to global IDs. */
					std::map< size_t, size_t > output_to_global_id;

					/** \internal A map from global IDs to their types. */
					std::map< size_t, enum VertexType > global_to_type;

					/** \internal A map from global IDs to their local IDs. */
					std::map< size_t, size_t > global_to_local_id;

					/**
					 * \internal
					 *
					 * Base constructor.
					 *
					 * @param[in] _hypergraph The base hypergraph.
					 * @param[in] _srcVec     Vector of source vertices.
					 * @param[in] _opVec      Vector of operation vertices.
					 * @param[in] _outVec     Vector of output vertices.
					 */
					HyperDAG(
						DHypergraph _hypergraph,
						const std::vector< SourceVertex > &_srcVec,
						const std::vector< OperationVertex > &_opVec,
						const std::vector< OutputVertex > &_outVec
					);


				public:


					/** @returns The hypergraph representation of the HyperDAG. */
					DHypergraph get() const noexcept;

					/** @returns The number of source vertices. */
					size_t numSources() const noexcept;

					/** @returns The number of operation vertices. */
					size_t numOperations() const noexcept;

					/** @returns The number of output vertices. */
					size_t numOutputs() const noexcept;

					/** @returns A start iterator to the source vertices. */
					std::vector< SourceVertex >::const_iterator sourcesBegin() const;

					/** @returns End iterator matching #sourcesBegin(). */
					std::vector< SourceVertex >::const_iterator sourcesEnd() const;

					/** @returns A start iterator to the output vertices. */
					std::vector< OperationVertex >::const_iterator operationsBegin() const;

					/** @returns End iterator matching #outputsBegin. */
					std::vector< OperationVertex >::const_iterator operationsEnd() const;

					/** @returns A start iterator to the output vertices. */
					std::vector< OutputVertex >::const_iterator outputsBegin() const;

					/** @returns End iterator matching #outputsBegin. */
					std::vector< OutputVertex >::const_iterator outputsEnd() const;

			};

			/** \internal Builds a HyperDAG representation of an ongoing computation. */
			class HyperDAGGenerator {

				private:

					/** \internal The hypergraph under construction. */
					DHypergraph hypergraph;

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
					std::map< const void *, SourceVertex > sourceVerticesP;

					/** \internal Map of IDs to source vertices. */
					std::map< uintptr_t, SourceVertex > sourceVerticesC;

					/** \internal Map of IDs to operation vertices. */
					std::map< uintptr_t, OperationVertex > operationVertices;

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
					std::map< uintptr_t,
						std::pair< size_t, OperationVertexType >
					> operationOrOutputVertices;

					/** \internal Source vertex generator. */
					SourceVertexGenerator sourceGen;

					/** \internal Operation vertex generator. */
					OperationVertexGenerator operationGen;

					// OutputVertexGenerator is a local field of #finalize()

					/**
					 * \internal
					 * Adds a source vertex to the hypergraph.
					 *
					 * @param[in] type    The type of source vertex.
					 * @param[in] pointer A unique identifier of the source.
					 * @param[in] id      A unique identifier of the source.
					 *
					 * If the \a type corresponds to an ALP/GraphBLAS container, then
					 * \a pointer is ignored; otherwise, \a id is ignored.
					 * \endinternal
					 */
					size_t addAnySource(
						const SourceVertexType type,
						const void * const pointer,
						const uintptr_t id
					);


				public:

					/**
					 * \internal Base constructor.
					 */
					HyperDAGGenerator() noexcept;

					/**
					 * \internal
					 *
					 * Sometimes a given \em operation generates a source vertex-- for example,
					 * the scalar input/output argument to grb::dot.
					 *
					 * In such cases, this function should be called to register the source
					 * vertex.
					 *
					 * @param[in] type    The type of source vertex
					 * @param[in] pointer A unique identifier corresponding to the source
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
					 *
					 * Registers a new source container with a given \a id.
					 *
					 * \endinternal
					 */
					void addContainer( const uintptr_t id );

					/**
					 * \internal
					 *
					 * Registers a new operation with the HyperDAG.
					 *
					 * @param[in] type The type of operation being registered.
					 * @param[in] src_start, src_end Iterators to a set of source pointers.
					 * @param[in] dst_start, dst_end Iterators to a set of destination pointers.
					 *
					 * This function proceeds as follows:
					 *    1. for source pointers in #operationOrOutputVertices, a) upgrade them
					 *       to #OperationVertex, and b) add them to #operationVertices. For
					 *       source pointers in #operationVertices, do nothing.
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
					template< typename SrcPIt, typename SrcCIt, typename DstIt >
					void addOperation(
						const OperationVertexType type,
						SrcPIt src_p_start, const SrcPIt &src_p_end,
						SrcCIt src_c_start, const SrcCIt &src_c_end,
						DstIt dst_start, const DstIt &dst_end
					) {
						static_assert( std::is_same< const void *,
								typename std::iterator_traits< SrcPIt >::value_type
							>::value,
							"Source pointers should be given as const void pointers"
						);
						static_assert( std::is_same< uintptr_t,
								typename std::iterator_traits< DstIt >::value_type
							>::value,
							"Destinations should be identified by their IDs"
						);
						static_assert( std::is_same< uintptr_t,
								typename std::iterator_traits< SrcCIt >::value_type
							>::value,
							"Source containers should be identified by their IDs"
						);

#ifdef _DEBUG
						std::cerr << "In HyperDAGGen::addOperation( "
							<< toString( type ) << ", ... )\n"
							<< "\t sourceVertices size: " << sourceVerticesP.size() << " pointers + "
							<< sourceVerticesC.size() << " containers\n"
							<< "\t sourceVec size: " << sourceVec.size() << "\n";
#endif

						// steps 1, 2, and 3
						std::vector< std::pair< size_t, std::set< size_t > > > hyperedges;
						for( ; src_p_start != src_p_end; ++src_p_start ) {
#ifdef _DEBUG
							std::cerr << "\t processing source pointer " << *src_p_start << "\n";
#endif
							// source pointers (input scalars, not input containers) are simple--
							// they will never appear as operation vertices, nor as output vertices.
							// Therefore step 1 does not apply.

							// step 2
							size_t sourceID;
							const auto alreadySource = sourceVerticesP.find( *src_p_start );
							if( alreadySource == sourceVerticesP.end() ) {
#ifndef NDEBUG
								const bool all_sources_should_already_be_added = false;
								assert( all_sources_should_already_be_added );
#endif
								std::cerr << "Warning: unidentified source " << *src_p_start << ". "
									<< "Adding it as an input scalar.\n";
								sourceID = addAnySource( SCALAR, *src_p_start, 0 );
							} else {
#ifdef _DEBUG
								std::cerr << "\t found source in sourceVertices\n";
#endif
								sourceID = alreadySource->second.getGlobalID();
							}
							// step 3
							hyperedges.push_back( std::make_pair( sourceID, std::set< size_t >() ) );
						}
						for( ; src_c_start != src_c_end; ++src_c_start ) {
#ifdef _DEBUG
							std::cerr << "\t processing source container " << *src_c_start << "\n";
#endif
							// step 1
							size_t sourceID;
							const auto &it = operationOrOutputVertices.find( *src_c_start );
							const auto &it2 = operationVertices.find( *src_c_start );
							if( it2 != operationVertices.end() ) {
								// operation vertices are fine as a source -- no additional operations
								// necessary
								assert( it == operationOrOutputVertices.end() );
#ifdef _DEBUG
								std::cerr << "\t source was previously an operation\n";
#endif
								sourceID = it2->second.getGlobalID();
							} else if( it == operationOrOutputVertices.end() ) {
								// step 2
								const auto alreadySource = sourceVerticesC.find( *src_c_start );
								if( alreadySource == sourceVerticesC.end() ) {
#ifndef NDEBUG
									const bool all_sources_should_already_be_added = false;
									assert( all_sources_should_already_be_added );
#endif
									std::cerr << "Warning: unidentified source " << *src_c_start << ". "
										<< "Adding it as a container.\n";
									sourceID = addAnySource( CONTAINER, nullptr, *src_c_start );
								} else {
#ifdef _DEBUG
									std::cerr << "\t found source in sourceVertices\n";
#endif
									sourceID = alreadySource->second.getGlobalID();
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
								sourceID = global_id;
							}
							// step 3
							hyperedges.push_back( std::make_pair( sourceID, std::set< size_t >() ) );
						}


						// step 4, 5, and 6
						for( ; dst_start != dst_end; ++dst_start ) {
#ifdef _DEBUG
							std::cerr << "\t processing destination " << *dst_start << "\n";
#endif
							// step 4
							{
								const auto &it = sourceVerticesC.find( *dst_start );
								if( it != sourceVerticesC.end() ) {
#ifdef _DEBUG
									std::cerr << "\t destination found in sources-- "
										<< "removing it from there\n";
#endif
									sourceVerticesC.erase( it );
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
							operationOrOutputVertices.insert(
								std::make_pair( *dst_start,
									std::make_pair( global_id, type )
								)
							);
#ifdef _DEBUG
							std::cerr << "\t created a new operation vertex with global ID "
								<< global_id << "\n";
#endif
							// step 6
							for( auto &hyperedge : hyperedges ) {
								hyperedge.second.insert( global_id );
							}
						}

						// step 7
						for( const auto &hyperedge : hyperedges ) {
#ifdef _DEBUG
							std::cerr << "\t storing a hyperedge of size "
								<< (hyperedge.second.size()+1) << "\n";
#endif
							hypergraph.appendHyperedge(
								hyperedge.first,
								hyperedge.second.begin(), hyperedge.second.end()
							);
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

