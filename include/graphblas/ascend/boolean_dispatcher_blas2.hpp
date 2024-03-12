
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
 * Dispatchers for the level-2 primitives
 *
 * @author A. N. Yzelman
 * @date 12th of September, 2023
 */

#ifndef _H_GRB_ASCEND_BOOLEAN_DISPATCHER_BLAS2
#define _H_GRB_ASCEND_BOOLEAN_DISPATCHER_BLAS2

#include <graphblas/backends.hpp>
#include <graphblas/blas0.hpp>
#include <graphblas/descriptors.hpp>
#include <graphblas/internalops.hpp>
#include <graphblas/ops.hpp>
#include <graphblas/rc.hpp>
#include <graphblas/semiring.hpp>

#include <graphblas/nonblocking/lazy_evaluation.hpp>

#include "coordinates.hpp"
#include "vector.hpp"


namespace grb {

	namespace internal {

		template<
			Descriptor descr,
			bool masked,
			bool input_masked,
			bool left_handed,
			template< typename > class One,
			bool already_dense_destination_vector,
			bool already_dense_mask_vector,
			class AdditiveMonoid,
			class Multiplication,
			typename IOType,
			typename InputType1,
			typename InputType2,
			typename InputType3,
			typename InputType4,
			typename Coords,
			typename RowColType,
			typename NonzeroType
		>
		inline void vxm_inner_kernel_gather(
			RC &rc,
			const size_t lower_bound,
			Coords &local_destination_vector,
			const Coords &local_mask_vector,
			Vector< IOType, ascend, Coords > &destination_vector,
			IOType &destination_element,
			const size_t &destination_index,
			const Vector< InputType1, ascend, Coords > &source_vector,
			const InputType1 * __restrict__ const &source,
			const size_t &source_range,
			const internal::Compressed_Storage<
				InputType2, RowColType, NonzeroType
			> &matrix,
			const Vector< InputType3, ascend, Coords > &mask_vector,
			const InputType3 * __restrict__ const &mask,
			const Vector< InputType4, ascend, Coords > &source_mask_vector,
			const InputType4 * __restrict__ const &source_mask,
			const AdditiveMonoid &add,
			const Multiplication &mul,
			const std::function< size_t( size_t ) > &src_local_to_global,
			const std::function< size_t( size_t ) > &src_global_to_local,
			const std::function< size_t( size_t ) > &dst_local_to_global
		);

		template<
			Descriptor descr,
			bool masked,
			bool input_masked,
			bool left_handed,
			template< typename > class One,
			class AdditiveMonoid,
			class Multiplication,
			typename IOType,
			typename InputType1,
			typename InputType2,
			typename InputType3,
			typename InputType4,
			typename Coords,
			typename RowColType,
			typename NonzeroType
		>
		inline void boolean_dispatcher_vxm_inner_kernel_gather(
			const bool already_dense_destination_vector,
			const bool already_dense_mask_vector,
			RC &rc,
			const size_t lower_bound,
			Coords &local_destination_vector,
			const Coords &local_mask_vector,
			Vector< IOType, ascend, Coords > &destination_vector,
			IOType &destination_element,
			const size_t &destination_index,
			const Vector< InputType1, ascend, Coords > &source_vector,
			const InputType1 * __restrict__ const &source,
			const size_t &source_range,
			const internal::Compressed_Storage<
				InputType2, RowColType, NonzeroType
			> &matrix,
			const Vector< InputType3, ascend, Coords > &mask_vector,
			const InputType3 * __restrict__ const &mask,
			const Vector< InputType4, ascend, Coords > &source_mask_vector,
			const InputType4 * __restrict__ const &source_mask,
			const AdditiveMonoid &add,
			const Multiplication &mul,
			const std::function< size_t( size_t ) > &src_local_to_global,
			const std::function< size_t( size_t ) > &src_global_to_local,
			const std::function< size_t( size_t ) > &dst_local_to_global
		) {
			if( already_dense_destination_vector ) {
				if( already_dense_mask_vector ) {
					return internal::vxm_inner_kernel_gather<
							descr, masked, input_masked, left_handed, One,
							true, true
						>(
							rc, lower_bound, local_destination_vector, local_mask_vector,
							destination_vector, destination_element, destination_index,
							source_vector, source, source_range, matrix, mask_vector, mask,
							source_mask_vector, source_mask, add, mul,
								src_local_to_global, src_global_to_local, dst_local_to_global
						);
				} else {
					return internal::vxm_inner_kernel_gather<
							descr, masked, input_masked, left_handed, One,
							true, false
						>(
							rc, lower_bound, local_destination_vector, local_mask_vector,
							destination_vector, destination_element, destination_index,
							source_vector, source, source_range, matrix, mask_vector, mask,
							source_mask_vector, source_mask, add, mul,
							src_local_to_global, src_global_to_local, dst_local_to_global
						);
				}
			} else {
				if( already_dense_mask_vector ) {
					return internal::vxm_inner_kernel_gather<
							descr, masked, input_masked, left_handed, One,
							false, true
						>(
							rc, lower_bound, local_destination_vector, local_mask_vector,
							destination_vector, destination_element, destination_index,
							source_vector, source, source_range, matrix, mask_vector, mask,
							source_mask_vector, source_mask, add, mul,
							src_local_to_global, src_global_to_local, dst_local_to_global
						);
				} else {
					return internal::vxm_inner_kernel_gather<
							descr, masked, input_masked, left_handed, One,
							false, false
						>(
							rc, lower_bound, local_destination_vector, local_mask_vector,
							destination_vector, destination_element, destination_index,
							source_vector, source, source_range, matrix, mask_vector, mask,
							source_mask_vector, source_mask, add, mul,
							src_local_to_global, src_global_to_local, dst_local_to_global
						);
				}
			}
		}

	} // end namespace ``internal''

} // end namespace ``grb''

#endif

