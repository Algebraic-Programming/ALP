
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

		class LazyEvaluation {

			private:

				std::vector< Pipeline > pipelines;

				// stores the pipelines that share data with the new stage
				std::vector< std::vector< Pipeline >::iterator > shared_data_pipelines;


			public:

				LazyEvaluation();

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

				RC addeWiseLambdaStage(
							const Pipeline::stage_type &&func,
							const Opcode opcode,
							const size_t n,
							const size_t data_type_size,
							const bool dense_descr,
							std::vector< const void * > all_containers_ptr,
							const Coordinates< nonblocking > * const coor_a_ptr
					);

				RC execution( const void *container );
				RC execution( );

		};

	}

}

#endif //end `_H_GRB_NONBLOCKING_LAZY_EVALUATION'

