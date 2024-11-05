
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
 * Describes a pipeline.
 *
 * @author Aristeidis Mastoras
 * @date 16th of May, 2022
 */

#ifndef _H_GRB_NONBLOCKING_PIPELINE
#define _H_GRB_NONBLOCKING_PIPELINE

/**
 * To enable debugging information only for the nonblocking backend, the code
 * should be combiled with the _NONBLOCKING_DEBUG definition, without defining
 * _DEBUG. If the code is compiled with _DEBUG, the debugging information for
 * the nonblocking backend is enabled as well.
 */
#if !defined(_NONBLOCKING_DEBUG) && defined(_DEBUG)
 #define _NONBLOCKING_DEBUG
#endif

/**
 * The GRB_ALREADY_DENSE_OPTIMIZATION definition is used for easily enabling and
 * disabling the optimization for already dense vectors to avoid the overhead of
 * the local coordinates. This is very useful for comparing the performance
 * between the different versions. The optimization is enabled by default.
 */
#define GRB_ALREADY_DENSE_OPTIMIZATION

/**
 * The GRB_BOOLEAN_DISPATCHER definition is related to the optimization for
 * already dense vectors, and it is used to easily choose between two different
 * implementations:
 * - one that uses formal parameters for variables that indicate if a vector is
 *   dense; and
 * - another one that uses template parameters for those variables.
 * The first one implies runtime overhead, and the second one requires some
 * additional code, which selects the values for the template parameters, and it
 * is defined in boolean_dispatcher_io.hpp, boolean_dispatcher_blas1.hpp, and
 * boolean_dispatcher_blas2.hpp.
 * A preliminary evaluation does not confirm that the first implementation is
 * slower. Therefore, we temporarily maintain both implementations to conduct
 * further evaluation.
 */
#define GRB_BOOLEAN_DISPATCHER

#include <vector>
#include <set>
#include <algorithm>
#include <functional>

#include <graphblas/backends.hpp>

#include "coordinates.hpp"


// TODO ugly hack, fwd declare ALP::internal::OpGen
namespace alp {
	template< size_t process_order, size_t problem_order >
	class Grid;
	namespace internal {
		class OpGen;
	}
}

namespace grb {

	namespace internal {

		/** Operation codes of primitives that may enter a dynamic pipeline. */
		enum class Opcode {
			IO_SET_SCALAR,
			IO_SET_MASKED_SCALAR,
			IO_SET_VECTOR,
			IO_SET_MASKED_VECTOR,

			BLAS1_FOLD_VECTOR_SCALAR_GENERIC,
			BLAS1_FOLD_SCALAR_VECTOR_GENERIC,
			BLAS1_FOLD_MASKED_SCALAR_VECTOR_GENERIC,
			BLAS1_FOLD_VECTOR_VECTOR_GENERIC,
			BLAS1_FOLD_MASKED_VECTOR_VECTOR_GENERIC,
			BLAS1_EWISEAPPLY,
			BLAS1_MASKED_EWISEAPPLY,
			BLAS1_EWISEMULADD_DISPATCH,
			BLAS1_DOT_GENERIC,
			BLAS1_EWISELAMBDA,
			BLAS1_EWISEMAP,
			BLAS1_ZIP,
			BLAS1_UNZIP,

			BLAS2_VXM_GENERIC
		};

		/**
		 * Encodes a single pipeline that may be expanded, merged, or executed.
		 */
		class Pipeline {

			friend class alp::internal::OpGen;
	template< size_t process_order, size_t problem_order >
	friend
				class alp::Grid;

			public:

				// The pipeline is passed by reference such that an out-of-place operation
				// can disable the dense descriptor and remove the coordinates of the empty
				// vector from the list.
				typedef std::function<
						RC( Pipeline &, const size_t, const size_t )
					> stage_type;


			private:

				size_t containers_size;
				size_t size_of_data_type;

				// per-stage data
				std::vector< stage_type > stages;


			public: //DBG

				std::vector< Opcode > opcodes;


			private: //DBG

				std::vector< std::vector< size_t > > stage_inputs;
				std::vector< size_t > stage_output;

				// per-pipeline data
				std::set< Coordinates< nonblocking > * > accessed_coordinates;
				std::set< const void * > input_vectors;
				std::set< const void * > output_vectors;
				std::set< const void * > vxm_input_vectors;

				/**
				 * The following vectors are used temporarily by the execution method.
				 * They are declared as members of the class to pre-allocate memory once.
				 */
				std::vector< size_t > lower_bound;
				std::vector< size_t > upper_bound;
				std::vector< const void * > input_output_intersection;

				/**
				 * In the current implementation that supports level-1 and level-2
				 * operations, pointers to the input matrices are used only for triggering
				 * the pipeline execution, e.g., in the destructor of the class Matrix.
				 * \todo Once level-3 operations are supported, they will be used in a
				 *       similar way as vectors.
				 */
				std::set< const void * > input_matrices;

				/**
				 * Indicates that the pipeline contains an out-of-place operation, which
				 * may clear the output vector and break any guarantees of already dense
				 * vectors.
				 */
				bool contains_out_of_place_primitive;

				/**
				 * Stores the set of output vectors of the out-of-place operations executed
				 * in the pipeline. It's used by the execution method to ensure that an
				 * already dense vector will remain dense after the execution of the
				 * pipeline, i.e., the vector is not the output of an out-of-place
				 * operation.
				 */
				std::set< const Coordinates< nonblocking > * >
					out_of_place_output_coordinates;

				/**
				 * Indicates that all the vectors are already dense before the execution
				 * of the pipeline, and thus enabling runtime optimizations.
				 */
				bool all_already_dense_vectors;
#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
				/**
				 * Maintains the coordinates of vectors that are already dense to enable
				 * optimizations.
				 * The set is built explicitly before the execution of the pipeline in the
				 * execution method.
				 */
				std::set< const Coordinates< nonblocking > * > already_dense_coordinates;
#endif
				/**
				 * This set of vectors is used for the verification for correct usage of the
				 * dense descriptor that takes place after the execution of the pipeline.
				 * The set is built when stages are added into the pipeline.
				 */
				std::set< Coordinates< nonblocking > * > dense_descr_coordinates;

				/**
				 * Whether a warning on container capacities increased beyond their initial
				 * capacities has been emitted.
				 */
				bool no_warning_emitted_yet;

				/**
				 * Function that checks if current container capacities have exceeded their
				 * initial capacity.
				 */
				void warnIfExceeded();


			public:

				/**
				 * Constructs a pipeline with given initial container, stage, and tile
				 * capacities.
				 *
				 * If during pipeline construction these initial capacities are exceeded, a
				 * warning may be emitted (see #grb::config::PIPELINE::warn_if_exceeded).
				 */
				Pipeline();

				Pipeline( const Pipeline &pipeline );
				Pipeline( Pipeline &&pipeline ) noexcept;

				Pipeline &operator=( const Pipeline &pipeline );
				Pipeline &operator=( Pipeline &&pipeline );

#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
				bool allAlreadyDenseVectors() const;
#endif
				bool empty() const;

				typename std::vector< stage_type >::iterator pbegin();
				typename std::vector< stage_type >::iterator pend();
				typename std::set< Coordinates< nonblocking > * >::iterator vbegin();
				typename std::set< Coordinates< nonblocking > * >::iterator vend();

				size_t accessedCoordinatesSize() const;
				size_t getNumStages() const;
				size_t getContainersSize() const;

				/**
				 * @param[in]  func   The lambda function executed by this operation (stage)
				 * @param[in]  opcode The operation code used as an identifier
				 * @param[in]  n      The size of the containers handled by this operation
				 * @param[in]  data_type_size The size of the data used in this operation
				 *                         required by the analytic model
				 * @param[in]  dense_descr Indicates that all vectors are dense before the
				 *                         execution of the operation. Used for the
				 *                         already dense optimization and the dense
				 *                         descriptor verification.
				 * @param[in]  dense_mask  Used only by the masked out-of-place operations.
				 * @param[out] output_vector_ptr A pointer to the output vector, equal to
				 *                         nullptr for operations that return a scalar.
				 * @param[out] output_aux_vector_ptr A pointer to the second output vector,
				 *                         equal to nullptr except for the unzip operation.
				 * @param[out] coor_output_ptr A pointer to the coordinates of the output.
				 * @param[out] coor_output_aux_ptr A pointer to the coordinates of the
				 *                         second output.
				 * @param[in]  input_a_ptr A pointer to the first input vector, it may be
				 *                         equal to nullptr if the input is a scalar.
				 * @param[in]  input_b_ptr A pointer to the second input vector, equal to
				 *                         nullptr if a second input does not exist.
				 * @param[in]  input_c_ptr A pointer to the third input vector, equal to
				 *                         nullptr if a third input does not exist.
				 * @param[in]  input_d_ptr A pointer to the fourth input vector, equal to
				 *                         nullptr if a fourth input does not exist.
				 * @param[in]  coor_a_ptr  A pointer to the coordinates of the first input
				 *                         vector, it may be equal to nullptr if the input
				 *                         is a scalar.
				 * @param[in]  coor_b_ptr  A pointer to the coordinates of the second input
				 *                         vector, equal to nullptr if a second input does
				 *                         not exist.
				 * @param[in]  coor_c_ptr  A pointer to the coordinates of the third input
				 *                         vector, equal to nullptr if a third input does
				 *                         not exist.
				 * @param[in]  coor_d_ptr  A pointer to the coordinates of the fourth input
				 *                         vector, equal to nullptr if a fourth input does
				 *                         not exist.
				 *
				 * \todo in the current implementation:
				 *
				 * @param[in]  input_matrix A pointer to the input matrix of SpMV.
				 */
				void addStage(
					const stage_type &&func,
					const Opcode opcode,
					const size_t n,
					const size_t data_type_size,
					const bool dense_descr,
					const bool dense_mask,
					const size_t output_vector_id,
					// TODO FIXME is there really a need for pointers?
					void * const output_vector_ptr,
					void * const output_aux_vector_ptr,
					Coordinates< nonblocking > * const coor_output_ptr,
					Coordinates< nonblocking > * const coor_output_aux_ptr,
					const size_t input_a_id,
					const size_t input_b_id,
					const size_t input_c_id,
					const size_t input_d_id,
					// TODO FIXME is there really a need for pointers?
					const void * const input_a_ptr,
					const void * const input_b_ptr,
					const void * const input_c_ptr,
					const void * const input_d_ptr,
					const Coordinates< nonblocking > * const coor_a_ptr,
					const Coordinates< nonblocking > * const coor_b_ptr,
					const Coordinates< nonblocking > * const coor_c_ptr,
					const Coordinates< nonblocking > * const coor_d_ptr,
					const size_t input_matrix_id,
					// TODO FIXME is there really a need for pointers?
					const void * const input_matrix
				);

				void addeWiseLambdaStage(
					const stage_type &&func,
					const Opcode opcode,
					const size_t n,
					const size_t data_type_size,
					const bool dense_descr,
					std::vector< const void * > all_vectors_ptr,
					const Coordinates< nonblocking > * const coor_a_ptr
				);

				bool accessesInputVector( const void * const vector ) const;
				bool accessesOutputVector( const void * const vector ) const;
				bool accessesVector( const void * const vector ) const;
				bool accessesMatrix( const void * const matrix ) const;

				bool overwritesVXMInputVectors( const void * const output_vector_ptr )
					const;

#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
				bool emptyAlreadyDenseVectors() const;
				bool containsAlreadyDenseVector(
					const Coordinates< nonblocking > * const vector_ptr
				) const;
				void markMaybeSparseVector(
					const Coordinates< nonblocking > * const vector_ptr
				);
#endif
				void markMaybeSparseDenseDescriptorVerification(
					Coordinates< nonblocking > * const vector_ptr
				);

				bool outOfPlaceOutput(
					const Coordinates< nonblocking > * const vector_ptr
				);

				void merge( Pipeline &pipeline );

				void clear();
#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
				void buildAlreadyDenseVectors();
#endif
				RC verifyDenseDescriptor();

				RC execution();

		};

	}

}

#endif //end `_H_GRB_NONBLOCKING_PIPELINE'

