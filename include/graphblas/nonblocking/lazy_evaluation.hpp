
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
 * Supporting constructs for lazy evaluation.
 *
 * @author Aristeidis Mastoras
 * @date 16th of May, 2022
 */

#ifndef _H_GRB_NONBLOCKING_LAZY_EVALUATION
#define _H_GRB_NONBLOCKING_LAZY_EVALUATION

#include <graphblas/backends.hpp>

#include "coordinates.hpp"
#include "pipeline.hpp"

namespace grb {

	namespace internal {

		/**
		 * Stores ALP primitives as stages in a set of pipelines maintained by this
		 * class.
		 */
		class LazyEvaluation {

		private:
			/** Multiple pipelines may be maintained at any time. */
			std::vector< Pipeline > pipelines;

			/** Stores the pipelines that share data with the new stage. */
			std::vector< std::vector< Pipeline >::iterator > shared_data_pipelines;

			/**
			 * Makes sure any warnings related to exceeding the initial number of
			 * pipelines are printed only once.
			 */
			bool warn_if_exceeded;

			/**
			 * Checks if the number of pipelines has been exceeded past the initial
			 * number of pipelines.
			 *
			 * The initial number is configurable via the following configuration
			 * field: #grb::config::PIPELINE::max_pipelines.
			 */
			void checkIfExceeded() noexcept;

		public:
			/** Default constructor. */
			LazyEvaluation();

			/**
			 * Adds a stage to an automatically determined pipeline.
			 *
			 * The following parameters are mandatory:
			 *
			 * @param[in]  func                     The function to be added.
			 * @param[in]  opcode                   The corresponding opcode.
			 * @param[in]  n                        The pipeline size.
			 * @param[in]  data_type_size           The output byte size.
			 * @param[in]  dense_descr              Whether the op is dense.
			 * @param[in]  dense_mask               Whether the mask is dense.
			 *
			 * The following parameters are optional and could be <tt>nullptr</tt> if
			 * not required:
			 *
			 * @param[out] output_container_ptr     Pointer to the output container.
			 * @param[out] output_aux_container_ptr Pointer to another output.
			 * @param[out] coor_output_ptr          Pointer to the coordinates that
			 *                                      correspond to
			 *                                      \a output_container_ptr
			 * @param[out] coor_output_aux_ptr      Pointer to the coordinates that
			 *                                      correspond to
			 *                                      \a output_aux_container_ptr
			 * @param[in]  input_a_ptr              Pointer to a first input container.
			 * @param[in]  input_b_ptr              Pointer to a second such container.
			 * @param[in]  input_c_ptr              Pointer to a third such container.
			 * @param[in]  input_d_ptr              Pointer to a fourth such container.
			 * @param[in]  coor_a_ptr               Pointer to coordinates that
			 *                                      correspond to \a input_a_ptr.
			 * @param[in]  coor_b_ptr               Pointer to coordinates that
			 *                                      correspond to \a input_b_ptr.
			 * @param[in]  coor_c_ptr               Pointer to coordinates that
			 *                                      correspond to \a input_c_ptr.
			 * @param[in]  coor_d_ptr               Pointer to coordinates that
			 *                                      correspond to \a input_d_ptr.
			 * @param[in]  input_matrix             Pointer to an input matrix.
			 */
			RC addStage(
				const Pipeline::stage_type &&func,
				const Opcode opcode,
				const size_t n,
				const size_t data_type_size,
				const bool dense_descr,
				const bool dense_mask,
				void * const output_container_ptr,
				void * const output_aux_container_ptr,
				Coordinates< nonblocking > * const coor_output_ptr,
				Coordinates< nonblocking > * const coor_output_aux_ptr,
				const void * const input_a_ptr,
				const void * const input_b_ptr,
				const void * const input_c_ptr,
				const void * const input_d_ptr,
				const Coordinates< nonblocking > * const coor_a_ptr,
				const Coordinates< nonblocking > * const coor_b_ptr,
				const Coordinates< nonblocking > * const coor_c_ptr,
				const Coordinates< nonblocking > * const coor_d_ptr,
				const void * const input_matrix
			);


			/**
			 * Adds a stage to an automatically determined pipeline. This is for 
			 * Level 3 operations
			 *
			 * The following parameters are mandatory:
			 *
			 * @param[in]  func                     The function to be added.
			 * @param[in]  opcode                   The corresponding opcode.
			 * @param[in]  n                        The pipeline size.
			 * @param[in]  data_type_size           The output byte size.
			 * @param[in]  dense_descr              Whether the op is dense.
			 * @param[in]  dense_mask               Whether the mask is dense.
			 *                                      correspond to \a input_d_ptr.
			 * @param[in]  input_matrix_A           Pointer to first input of C = AB
			 * @param[in]  input_matrix_B           Pointer to second input of C = AB
			 * @param[out]  input_matrix_B          Pointer to output matrix C = AB
			 * @param[out]  count_nonzeros          function to count the nnz in each tile of C = AB
			 */
			RC addStageLevel3( const Pipeline::stage_type && func,
				const Opcode opcode,
				const size_t n,
				const size_t data_type_size,
				const bool dense_descr,
				const bool dense_mask,			
				const void * const input_matrix_A,
				const void * const input_matrix_B,
				void * const output_matrix_C, 
				const void * const output_matrix_C_mask,
				const Pipeline::count_nnz_local_type && count_nonzeros,
				const Pipeline::prefix_sum_nnz_mxm_type && prefix_sum_nnz);
			
			/**
			 * Adds an eWiseLambda stage to an automatically-determined pipeline.
			 *
			 * The following parameters are mandatory:
			 *
			 * @param[in] func               The function to be added.
			 * @param[in] opcode             The corresponding opcode.
			 * @param[in] n                  The pipeline size.
			 * @param[in] data_type_size     The output byte size.
			 * @param[in] dense_descr        Whether the op is dense.
			 * @param[in] all_containers_ptr A container of all ALP containers that the
			 *                               \a func reads \em or writes
			 * @param[in] coor_a_ptr         A container of all coordinates that
			 *                               correspond to those in
			 *                               \a all_containers_ptr
			 */
			RC addeWiseLambdaStage( const Pipeline::stage_type && func,
				const Opcode opcode,
				const size_t n,
				const size_t data_type_size,
				const bool dense_descr,
				std::vector< const void * > all_containers_ptr,
				const Coordinates< nonblocking > * const coor_a_ptr );

			/**
			 * Executes the pipeline necessary to generate the output of the given
			 * \a container.
			 */
			RC execution( const void * container );

			/**
			 * Executes all pipelines.
			 */
			RC execution();

		}; // end class LazyEvaluation

	} // end namespace internal

} // end namespace grb

#endif // end `_H_GRB_NONBLOCKING_LAZY_EVALUATION'
