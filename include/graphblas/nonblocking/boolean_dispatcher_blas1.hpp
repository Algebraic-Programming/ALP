
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
 * Dispatcher functions for the level-1 primitives.
 *
 * @author Aristeidis Mastoras
 * @date 24th of October, 2022
 */

#ifndef _H_GRB_NONBLOCKING_BOOLEAN_DISPATCHER_BLAS1
#define _H_GRB_NONBLOCKING_BOOLEAN_DISPATCHER_BLAS1

#include <graphblas/backends.hpp>
#include <graphblas/blas0.hpp>
#include <graphblas/descriptors.hpp>
#include <graphblas/internalops.hpp>
#include <graphblas/ops.hpp>
#include <graphblas/rc.hpp>
#include <graphblas/semiring.hpp>

#include "coordinates.hpp"
#include "vector.hpp"
#include "lazy_evaluation.hpp"
#include "vector_wrapper.hpp"


namespace grb {

	namespace internal {

		template<
			Descriptor descr,
			bool masked,
			bool left,
			bool already_dense_input_to_fold,
			bool already_dense_mask,
			class Monoid,
			typename InputType,
			typename MaskType,
			class Coords
		>
		RC fold_from_vector_to_scalar_vectorDriven(
			typename Monoid::D3 &thread_local_output,
			const size_t lower_bound,
			const size_t upper_bound,
			const Coords &local_to_fold,
			const Coords &local_mask,
			const Vector< InputType, nonblocking, Coords > &to_fold,
			const Vector< MaskType, nonblocking, Coords > &mask,
			const Monoid &monoid
		);

		template<
			Descriptor descr,
			bool masked,
			bool left,
			class Monoid,
			typename InputType,
			typename MaskType,
			class Coords
		>
		RC boolean_dispatcher_fold_from_vector_to_scalar_vectorDriven(
			const bool already_dense_input_to_fold,
			const bool already_dense_mask,
			typename Monoid::D3 &thread_local_output,
			const size_t lower_bound,
			const size_t upper_bound,
			const Coords &local_to_fold,
			const Coords &local_mask,
			const Vector< InputType, nonblocking, Coords > &to_fold,
			const Vector< MaskType, nonblocking, Coords > &mask,
			const Monoid &monoid
		) {
			if( already_dense_input_to_fold ) {
				if( already_dense_mask ) {
					return internal::fold_from_vector_to_scalar_vectorDriven<
							descr, masked, left, true, true
						>(
							thread_local_output, lower_bound, upper_bound,
							local_to_fold, local_mask, to_fold, mask, monoid
						);
				} else {
					return internal::fold_from_vector_to_scalar_vectorDriven<
							descr, masked, left, true, false
						>(
							thread_local_output, lower_bound, upper_bound,
							local_to_fold, local_mask, to_fold, mask, monoid
						);
				}
			} else {
				if( already_dense_mask ) {
					return internal::fold_from_vector_to_scalar_vectorDriven<
							descr, masked, left, false, true
						>(
							thread_local_output, lower_bound, upper_bound,
							local_to_fold, local_mask, to_fold, mask, monoid
						);
				} else {
					return internal::fold_from_vector_to_scalar_vectorDriven<
							descr, masked, left, false, false
						>(
							thread_local_output, lower_bound, upper_bound,
							local_to_fold, local_mask, to_fold, mask, monoid
						);
				}
			}
		}

		template<
			Descriptor descr,
			bool left,
			bool already_dense_input_to_fold,
			bool already_dense_mask,
			class Monoid,
			typename InputType,
			typename MaskType,
			class Coords
		>
		RC fold_from_vector_to_scalar_maskDriven(
			typename Monoid::D3 &thread_local_output,
			const size_t lower_bound,
			const size_t upper_bound,
			const Coords &local_to_fold,
			const Coords &local_mask,
			const Vector< InputType, nonblocking, Coords > &to_fold,
			const Vector< MaskType, nonblocking, Coords > &mask,
			const Monoid &monoid
		);

		template<
			Descriptor descr,
			bool left,
			class Monoid,
			typename InputType,
			typename MaskType,
			class Coords
		>
		RC boolean_dispatcher_fold_from_vector_to_scalar_maskDriven(
			const bool already_dense_input_to_fold,
			const bool already_dense_mask,
			typename Monoid::D3 &thread_local_output,
			const size_t lower_bound,
			const size_t upper_bound,
			const Coords &local_to_fold,
			const Coords &local_mask,
			const Vector< InputType, nonblocking, Coords > &to_fold,
			const Vector< MaskType, nonblocking, Coords > &mask,
			const Monoid &monoid
		) {
			if( already_dense_input_to_fold ) {
				if( already_dense_mask ) {
					return internal::fold_from_vector_to_scalar_maskDriven<
							descr, left, true, true
						>(
							thread_local_output, lower_bound, upper_bound,
							local_to_fold, local_mask, to_fold, mask, monoid
						);
				} else {
					return internal::fold_from_vector_to_scalar_maskDriven<
							descr, left, true, false
						>(
							thread_local_output, lower_bound, upper_bound,
							local_to_fold, local_mask, to_fold, mask, monoid
						);
				}
			} else {
				if( already_dense_mask ) {
					return internal::fold_from_vector_to_scalar_maskDriven<
							descr, left, false, true
						>(
							thread_local_output, lower_bound, upper_bound,
							local_to_fold, local_mask, to_fold, mask, monoid
						);
				} else {
					return internal::fold_from_vector_to_scalar_maskDriven<
							descr, left, false, false
						>(
							thread_local_output, lower_bound, upper_bound,
							local_to_fold, local_mask, to_fold, mask, monoid
						);
				}
			}
		}

		template<
			Descriptor descr,
			bool masked,
			bool left,
			bool already_dense_input_to_fold,
			bool already_dense_mask,
			class Monoid,
			typename InputType,
			typename MaskType,
			class Coords
		>
		RC fold_from_vector_to_scalar_fullLoopSparse(
			typename Monoid::D3 &thread_local_output,
			const size_t lower_bound,
			const size_t upper_bound,
			const Coords &local_to_fold,
			const Coords &local_mask,
			const Vector< InputType, nonblocking, Coords > &to_fold,
			const Vector< MaskType, nonblocking, Coords > &mask,
			const Monoid &monoid
		);

		template<
			Descriptor descr,
			bool masked,
			bool left,
			class Monoid,
			typename InputType,
			typename MaskType,
			class Coords
		>
		RC boolean_dispatcher_fold_from_vector_to_scalar_fullLoopSparse(
			const bool already_dense_input_to_fold,
			const bool already_dense_mask,
			typename Monoid::D3 &thread_local_output,
			const size_t lower_bound,
			const size_t upper_bound,
			const Coords &local_to_fold,
			const Coords &local_mask,
			const Vector< InputType, nonblocking, Coords > &to_fold,
			const Vector< MaskType, nonblocking, Coords > &mask,
			const Monoid &monoid
		) {
			if( already_dense_input_to_fold ) {
				if( already_dense_mask ) {
					return internal::fold_from_vector_to_scalar_fullLoopSparse<
							descr, masked, left, true, true
						>(
							thread_local_output, lower_bound, upper_bound,
							local_to_fold, local_mask, to_fold, mask, monoid
						);
				} else {
					return internal::fold_from_vector_to_scalar_fullLoopSparse<
							descr, masked, left, true, false
						>(
							thread_local_output, lower_bound, upper_bound,
							local_to_fold, local_mask, to_fold, mask, monoid
						);
				}
			} else {
				if( already_dense_mask ) {
					return internal::fold_from_vector_to_scalar_fullLoopSparse<
							descr, masked, left, false, true
						>(
							thread_local_output, lower_bound, upper_bound,
							local_to_fold, local_mask, to_fold, mask, monoid
						);
				} else {
					return internal::fold_from_vector_to_scalar_fullLoopSparse<
							descr, masked, left, false, false
						>(
							thread_local_output, lower_bound, upper_bound,
							local_to_fold, local_mask, to_fold, mask, monoid
						);
				}
			}
		}

		template< Descriptor descr,
			bool left,
			bool sparse,
			bool masked,
			bool monoid,
			bool already_dense_output,
			bool already_dense_mask,
			typename MaskType,
			typename IOType,
			typename InputType,
			typename Coords,
			class OP
		>
		RC fold_from_scalar_to_vector_generic(
			const size_t lower_bound,
			const size_t upper_bound,
			Coords &local_vector,
			const Coords * const local_mask_ptr,
			Vector< IOType, nonblocking, Coords > &vector,
			const Vector< MaskType, nonblocking, Coords > * const mask,
			const InputType &scalar,
			const OP &op,
			const Phase &phase
		);

		template< Descriptor descr,
			bool left,
			bool sparse,
			bool masked,
			bool monoid,
			typename MaskType,
			typename IOType,
			typename InputType,
			typename Coords,
			class OP
		>
		RC boolean_dispatcher_fold_from_scalar_to_vector_generic(
			const bool already_dense_output,
			const bool already_dense_mask,
			const size_t lower_bound,
			const size_t upper_bound,
			Coords &local_vector,
			const Coords * const local_mask_ptr,
			Vector< IOType, nonblocking, Coords > &vector,
			const Vector< MaskType, nonblocking, Coords > * const mask,
			const InputType &scalar,
			const OP &op,
			const Phase &phase
		) {
			if( already_dense_output ) {
				if( already_dense_mask ) {
					return internal::fold_from_scalar_to_vector_generic<
							descr, left, sparse, masked, monoid,
							true, true
						>(
							lower_bound, upper_bound, local_vector, local_mask_ptr,
							vector, mask, scalar, op, phase
						);
				} else {
					return internal::fold_from_scalar_to_vector_generic<
							descr, left, sparse, masked, monoid,
							true, false
						>(
							lower_bound, upper_bound, local_vector, local_mask_ptr,
							vector, mask, scalar, op, phase
						);
				}
			} else {
				if( already_dense_mask ) {
					return internal::fold_from_scalar_to_vector_generic<
							descr, left, sparse, masked, monoid,
							false, true
						>(
							lower_bound, upper_bound, local_vector, local_mask_ptr,
							vector, mask, scalar, op, phase
						);
				} else {
					return internal::fold_from_scalar_to_vector_generic<
							descr, left, sparse, masked, monoid,
							false, false
						>(
							lower_bound, upper_bound, local_vector, local_mask_ptr,
							vector, mask, scalar, op, phase
						);
				}
			}
		}

		template< Descriptor descr,
			bool left,
			bool sparse,
			bool masked,
			bool monoid,
			bool already_dense_output,
			bool already_dense_input_to_fold,
			bool already_dense_mask,
			typename MaskType,
			typename IOType,
			typename IType,
			typename Coords,
			class OP
		>
		RC fold_from_vector_to_vector_generic(
			const size_t lower_bound,
			const size_t upper_bound,
			Coords &local_fold_into,
			const Coords * const local_m_ptr,
			const Coords &local_to_fold,
			Vector< IOType, nonblocking, Coords > &fold_into,
			const Vector< MaskType, nonblocking, Coords > * const m,
			const Vector< IType, nonblocking, Coords > &to_fold,
			const OP &op,
			const Phase phase
		);

		template< Descriptor descr,
			bool left,
			bool sparse,
			bool masked,
			bool monoid,
			typename MaskType,
			typename IOType,
			typename IType,
			typename Coords,
			class OP
		>
		RC boolean_dispatcher_fold_from_vector_to_vector_generic(
			const bool already_dense_output,
			const bool already_dense_input_to_fold,
			const bool already_dense_mask,
			const size_t lower_bound,
			const size_t upper_bound,
			Coords &local_fold_into,
			const Coords * const local_m_ptr,
			const Coords &local_to_fold,
			Vector< IOType, nonblocking, Coords > &fold_into,
			const Vector< MaskType, nonblocking, Coords > * const m,
			const Vector< IType, nonblocking, Coords > &to_fold,
			const OP &op,
			const Phase phase
		) {
			if( already_dense_output ) {
				if( already_dense_input_to_fold ) {
					if( already_dense_mask ) {
						return internal::fold_from_vector_to_vector_generic<
								descr, left, sparse, masked, monoid,
								true, true, true
							>(
								lower_bound, upper_bound, local_fold_into, local_m_ptr,
								local_to_fold, fold_into, m, to_fold, op, phase
							);
					} else {
						return internal::fold_from_vector_to_vector_generic<
								descr, left, sparse, masked, monoid,
								true, true, false
							>(
								lower_bound, upper_bound, local_fold_into, local_m_ptr,
								local_to_fold, fold_into, m, to_fold, op, phase
							);
					}
				} else {
					if( already_dense_mask ) {
						return internal::fold_from_vector_to_vector_generic<
								descr, left, sparse, masked, monoid,
								true, false, true
							>(
								lower_bound, upper_bound, local_fold_into, local_m_ptr,
								local_to_fold, fold_into, m, to_fold, op, phase
							);
					} else {
						return internal::fold_from_vector_to_vector_generic<
								descr, left, sparse, masked, monoid,
								true, false, false
							>(
								lower_bound, upper_bound, local_fold_into, local_m_ptr,
								local_to_fold, fold_into, m, to_fold, op, phase
							);
					}
				}
			} else {
				if( already_dense_input_to_fold ) {
					if( already_dense_mask ) {
						return internal::fold_from_vector_to_vector_generic<
								descr, left, sparse, masked, monoid,
								false, true, true
							>(
								lower_bound, upper_bound, local_fold_into, local_m_ptr,
								local_to_fold, fold_into, m, to_fold, op, phase
							);
					} else {
						return internal::fold_from_vector_to_vector_generic<
								descr, left, sparse, masked, monoid,
								false, true, false
							>(
								lower_bound, upper_bound, local_fold_into, local_m_ptr,
								local_to_fold, fold_into, m, to_fold, op, phase
							);
					}
				} else {
					if( already_dense_mask ) {
						return internal::fold_from_vector_to_vector_generic<
								descr, left, sparse, masked, monoid,
								false, false, true
							>(
								lower_bound, upper_bound, local_fold_into, local_m_ptr,
								local_to_fold, fold_into, m, to_fold, op, phase
							);
					} else {
						return internal::fold_from_vector_to_vector_generic<
								descr, left, sparse, masked, monoid,
								false, false, false
							>(
								lower_bound, upper_bound, local_fold_into, local_m_ptr,
								local_to_fold, fold_into, m, to_fold, op, phase
							);
					}
				}
			}
		}

		template<
			bool left_scalar,
			bool right_scalar,
			bool left_sparse,
			bool right_sparse,
			Descriptor descr,
			class OP,
			bool already_dense_input_x,
			bool already_dense_input_y,
			typename OutputType,
			typename InputType1,
			typename InputType2,
			typename Coords
		>
		RC dense_apply_generic(
			const size_t lower_bound,
			const size_t upper_bound,
			const Coords &local_x,
			const Coords &local_y,
			Vector< OutputType, nonblocking, Coords > &z_vector,
			const internal::Wrapper< left_scalar, InputType1, Coords > x_wrapper,
			const internal::Wrapper< right_scalar, InputType2, Coords > y_wrapper,
			const OP &op
		);

		template<
			bool left_scalar,
			bool right_scalar,
			bool left_sparse,
			bool right_sparse,
			Descriptor descr,
			class OP,
			typename OutputType,
			typename InputType1,
			typename InputType2,
			typename Coords
		>
		RC boolean_dispatcher_dense_apply_generic(
			const bool already_dense_input_x,
			const bool already_dense_input_y,
			const size_t lower_bound,
			const size_t upper_bound,
			const Coords &local_x,
			const Coords &local_y,
			Vector< OutputType, nonblocking, Coords > &z_vector,
			const internal::Wrapper< left_scalar, InputType1, Coords > x_wrapper,
			const internal::Wrapper< right_scalar, InputType2, Coords > y_wrapper,
			const OP &op
		) {
			if( already_dense_input_x ) {
				if( already_dense_input_y ) {
					return internal::dense_apply_generic<
							left_scalar, right_scalar, left_sparse, right_sparse, descr, OP,
							true, true
						>(
							lower_bound, upper_bound,
							local_x, local_y, z_vector, x_wrapper, y_wrapper, op
						);
				} else {
					return internal::dense_apply_generic<
							left_scalar, right_scalar, left_sparse, right_sparse, descr, OP,
							true, true
						>(
							lower_bound, upper_bound,
							local_x, local_y, z_vector, x_wrapper, y_wrapper, op
						);
				}
			} else {
				if( already_dense_input_y ) {
					return internal::dense_apply_generic<
							left_scalar, right_scalar, left_sparse, right_sparse, descr, OP,
							true, true
						>(
							lower_bound, upper_bound,
							local_x, local_y, z_vector, x_wrapper, y_wrapper, op
						);
				} else {
					return internal::dense_apply_generic<
							left_scalar, right_scalar, left_sparse, right_sparse, descr, OP,
							true, true
						>(
							lower_bound, upper_bound,
							local_x, local_y, z_vector, x_wrapper, y_wrapper, op
						);
				}
			}
		}

		template<
			bool masked,
			bool monoid,
			bool x_scalar,
			bool y_scalar,
			Descriptor descr,
			class OP,
			bool already_dense_mask,
			bool already_dense_input_x,
			bool already_dense_input_y,
			typename OutputType,
			typename MaskType,
			typename InputType1,
			typename InputType2,
			typename Coords
		>
		RC sparse_apply_generic(
			const size_t lower_bound,
			const size_t upper_bound,
			Coords &local_z,
			const Coords * const local_mask_ptr,
			const Coords &local_x,
			const Coords &local_y,
			Vector< OutputType, nonblocking, Coords > &z_vector,
			const Vector< MaskType, nonblocking, Coords > * const mask_vector,
			const internal::Wrapper< x_scalar, InputType1, Coords > x_wrapper,
			const internal::Wrapper< y_scalar, InputType2, Coords > y_wrapper,
			const OP &op
		);

		template<
			bool masked,
			bool monoid,
			bool x_scalar,
			bool y_scalar,
			Descriptor descr,
			class OP,
			typename OutputType,
			typename MaskType,
			typename InputType1,
			typename InputType2,
			typename Coords
		>
		RC boolean_dispatcher_sparse_apply_generic(
			const bool already_dense_mask,
			const bool already_dense_input_x,
			const bool already_dense_input_y,
			const size_t lower_bound,
			const size_t upper_bound,
			Coords &local_z,
			const Coords * const local_mask_ptr,
			const Coords &local_x,
			const Coords &local_y,
			Vector< OutputType, nonblocking, Coords > &z_vector,
			const Vector< MaskType, nonblocking, Coords > * const mask_vector,
			const internal::Wrapper< x_scalar, InputType1, Coords > x_wrapper,
			const internal::Wrapper< y_scalar, InputType2, Coords > y_wrapper,
			const OP &op
		) {
			if( already_dense_mask ) {
				if( already_dense_input_x ) {
					if( already_dense_input_y ) {
						return internal::sparse_apply_generic<
								masked, monoid, x_scalar, y_scalar, descr, OP,
								true, true, true
							> (
								lower_bound, upper_bound, local_z, local_mask_ptr, local_x, local_y,
								z_vector, mask_vector, x_wrapper, y_wrapper, op
							);
					} else {
						return internal::sparse_apply_generic<
								masked, monoid, x_scalar, y_scalar, descr, OP,
								true, true, false
							> (
								lower_bound, upper_bound, local_z, local_mask_ptr, local_x, local_y,
								z_vector, mask_vector, x_wrapper, y_wrapper, op
							);
					}
				} else {
					if( already_dense_input_y ) {
						return internal::sparse_apply_generic<
								masked, monoid, x_scalar, y_scalar, descr, OP,
								true, false, true
							> (
								lower_bound, upper_bound,
								local_z, local_mask_ptr, local_x, local_y,
								z_vector, mask_vector, x_wrapper, y_wrapper, op
							);
					} else {
						return internal::sparse_apply_generic<
								masked, monoid, x_scalar, y_scalar, descr, OP,
								true, false, false
							> (
								lower_bound, upper_bound, local_z, local_mask_ptr, local_x, local_y,
								z_vector, mask_vector, x_wrapper, y_wrapper, op
							);
					}
				}
			} else {
				if( already_dense_input_x ) {
					if( already_dense_input_y ) {
						return internal::sparse_apply_generic<
								masked, monoid, x_scalar, y_scalar, descr, OP,
								false, true, true
							> (
								lower_bound, upper_bound,
								local_z, local_mask_ptr, local_x, local_y,
								z_vector, mask_vector, x_wrapper, y_wrapper, op
							);
					} else {
						return internal::sparse_apply_generic<
								masked, monoid, x_scalar, y_scalar, descr, OP,
								false, true, false
							> (
								lower_bound, upper_bound, local_z, local_mask_ptr, local_x, local_y,
								z_vector, mask_vector, x_wrapper, y_wrapper, op
							);
					}
				} else {
					if( already_dense_input_y ) {
						return internal::sparse_apply_generic<
								masked, monoid, x_scalar, y_scalar, descr, OP,
								false, false, true
							> (
								lower_bound, upper_bound, local_z, local_mask_ptr, local_x, local_y,
								z_vector, mask_vector, x_wrapper, y_wrapper, op
							);
					} else {
						return internal::sparse_apply_generic<
								masked, monoid, x_scalar, y_scalar, descr, OP,
								false, false, false
							> (
								lower_bound, upper_bound, local_z, local_mask_ptr, local_x, local_y,
								z_vector, mask_vector, x_wrapper, y_wrapper, op
							);
					}
				}
			}
		}

		template<
			bool left_scalar,
			bool right_scalar,
			bool left_sparse,
			bool right_sparse,
			Descriptor descr,
			class OP,
			bool already_dense_mask,
			bool already_dense_input_x,
			bool already_dense_input_y,
			typename OutputType,
			typename MaskType,
			typename InputType1,
			typename InputType2,
			typename Coords
		>
		RC masked_apply_generic(
			const size_t lower_bound,
			const size_t upper_bound,
			Coords &local_z,
			const Coords &local_mask,
			const Coords &local_x,
			const Coords &local_y,
			Vector< OutputType, nonblocking, Coords > &z_vector,
			const Vector< MaskType, nonblocking, Coords > &mask_vector,
			const internal::Wrapper< left_scalar, InputType1, Coords > x_wrapper,
			const internal::Wrapper< right_scalar, InputType2, Coords > y_wrapper,
			const OP &op,
#ifdef GRB_BOOLEAN_DISPATCHER
			const InputType1 * const left_identity,
			const InputType2 * const right_identity
#else
			const InputType1 * const left_identity = nullptr,
			const InputType2 * const right_identity = nullptr
#endif
		);

		template<
			bool left_scalar,
			bool right_scalar,
			bool left_sparse,
			bool right_sparse,
			Descriptor descr,
			class OP,
			typename OutputType,
			typename MaskType,
			typename InputType1,
			typename InputType2,
			typename Coords
		>
		RC boolean_dispatcher_masked_apply_generic(
			const bool already_dense_mask,
			const bool already_dense_input_x,
			const bool already_dense_input_y,
			const size_t lower_bound,
			const size_t upper_bound,
			Coords &local_z,
			const Coords &local_mask,
			const Coords &local_x,
			const Coords &local_y,
			Vector< OutputType, nonblocking, Coords > &z_vector,
			const Vector< MaskType, nonblocking, Coords > &mask_vector,
			const internal::Wrapper< left_scalar, InputType1, Coords > x_wrapper,
			const internal::Wrapper< right_scalar, InputType2, Coords > y_wrapper,
			const OP &op,
			const InputType1 * const left_identity = nullptr,
			const InputType2 * const right_identity = nullptr
		) {
			if( already_dense_mask ) {
				if( already_dense_input_x ) {
					if( already_dense_input_y ) {
						return internal::masked_apply_generic<
								left_scalar, right_scalar, left_sparse, right_sparse, descr, OP,
								true, true, true
							>(
								lower_bound, upper_bound, local_z, local_mask, local_x, local_y,
								z_vector, mask_vector, x_wrapper, y_wrapper, op, left_identity, right_identity
							);
					} else {
						return internal::masked_apply_generic<
								left_scalar, right_scalar, left_sparse, right_sparse, descr, OP,
								true, true, false
							>(
								lower_bound, upper_bound, local_z, local_mask, local_x, local_y,
								z_vector, mask_vector, x_wrapper, y_wrapper,
								op, left_identity, right_identity
							);
					}
				} else {
					if( already_dense_input_y ) {
						return internal::masked_apply_generic<
								left_scalar, right_scalar, left_sparse, right_sparse, descr, OP,
								true, false, true
							>(
								lower_bound, upper_bound, local_z, local_mask, local_x, local_y,
								z_vector, mask_vector, x_wrapper, y_wrapper,
								op, left_identity, right_identity
							);
					} else {
						return internal::masked_apply_generic<
								left_scalar, right_scalar, left_sparse, right_sparse, descr, OP,
								true, false, false
							>(
								lower_bound, upper_bound, local_z, local_mask, local_x, local_y,
								z_vector, mask_vector, x_wrapper, y_wrapper,
								op, left_identity, right_identity
							);
					}
				}
			} else {
				if( already_dense_input_x ) {
					if( already_dense_input_y ) {
						return internal::masked_apply_generic<
								left_scalar, right_scalar, left_sparse, right_sparse, descr, OP,
								false, true, true
							>(
								lower_bound, upper_bound, local_z, local_mask, local_x, local_y,
								z_vector, mask_vector, x_wrapper, y_wrapper,
								op, left_identity, right_identity
							);
					} else {
						return internal::masked_apply_generic<
								left_scalar, right_scalar, left_sparse, right_sparse, descr, OP,
								false, true, false
							>(
								lower_bound, upper_bound, local_z, local_mask, local_x, local_y,
								z_vector, mask_vector, x_wrapper, y_wrapper,
								op, left_identity, right_identity
							);
					}
				} else {
					if( already_dense_input_y ) {
						return internal::masked_apply_generic<
								left_scalar, right_scalar, left_sparse, right_sparse, descr, OP,
								false, false, true
							>(
								lower_bound, upper_bound, local_z, local_mask, local_x, local_y,
								z_vector, mask_vector, x_wrapper, y_wrapper,
								op, left_identity, right_identity
							);
					} else {
						return internal::masked_apply_generic<
								left_scalar, right_scalar, left_sparse, right_sparse, descr, OP,
								false, false, false
							>(
								lower_bound, upper_bound, local_z, local_mask, local_x, local_y,
								z_vector, mask_vector, x_wrapper, y_wrapper,
								op, left_identity, right_identity
							);
					}
				}
			}
		}

		template<
			Descriptor descr,
			bool a_scalar,
			bool x_scalar,
			bool y_scalar,
			bool y_zero,
			bool already_dense_output,
			bool already_dense_mask,
			bool already_dense_input_a,
			bool already_dense_input_x,
			bool already_dense_input_y,
			typename OutputType,
			typename MaskType,
			typename InputType1,
			typename InputType2,
			typename InputType3,
			typename Coords,
			class Ring
		>
		RC sparse_eWiseMulAdd_maskDriven(
			const size_t lower_bound,
			const size_t upper_bound,
			Coords &local_z,
			const Coords &local_m,
			const Coords &local_a,
			const Coords &local_x,
			const Coords &local_y,
			Vector< OutputType, nonblocking, Coords > &z_vector,
			const Vector< MaskType, nonblocking, Coords > &m_vector,
			const internal::Wrapper< a_scalar, InputType1, Coords > &a_wrapper,
			const internal::Wrapper< x_scalar, InputType2, Coords > &x_wrapper,
			const internal::Wrapper< y_scalar, InputType3, Coords > &y_wrapper,
			const Ring &ring
		);

		template<
			Descriptor descr,
			bool a_scalar,
			bool x_scalar,
			bool y_scalar,
			bool y_zero,
			typename OutputType,
			typename MaskType,
			typename InputType1,
			typename InputType2,
			typename InputType3,
			typename Coords,
			class Ring
		>
		RC boolean_dispatcher_sparse_eWiseMulAdd_maskDriven(
			const bool already_dense_output,
			const bool already_dense_mask,
			const bool already_dense_input_a,
			const bool already_dense_input_x,
			const bool already_dense_input_y,
			const size_t lower_bound,
			const size_t upper_bound,
			Coords &local_z,
			const Coords &local_m,
			const Coords &local_a,
			const Coords &local_x,
			const Coords &local_y,
			Vector< OutputType, nonblocking, Coords > &z_vector,
			const Vector< MaskType, nonblocking, Coords > &m_vector,
			const internal::Wrapper< a_scalar, InputType1, Coords > &a_wrapper,
			const internal::Wrapper< x_scalar, InputType2, Coords > &x_wrapper,
			const internal::Wrapper< y_scalar, InputType3, Coords > &y_wrapper,
			const Ring &ring
		) {
			if( already_dense_output ) {
				if( already_dense_mask ) {
					if( already_dense_input_a ) {
						if( already_dense_input_x ) {
							if( already_dense_input_y ) {
								return internal::sparse_eWiseMulAdd_maskDriven<
										descr, a_scalar, x_scalar, y_scalar, y_zero,
										true, true, true, true, true
									>(
										lower_bound, upper_bound, local_z, local_m, local_a, local_x, local_y,
										z_vector, m_vector, a_wrapper, x_wrapper, y_wrapper, ring
									);
							} else {
								return internal::sparse_eWiseMulAdd_maskDriven<
										descr, a_scalar, x_scalar, y_scalar, y_zero,
										true, true, true, true, false
									>(
										lower_bound, upper_bound, local_z, local_m, local_a, local_x, local_y,
										z_vector, m_vector, a_wrapper, x_wrapper, y_wrapper, ring
									);
							}
						} else {
							if( already_dense_input_y ) {
								return internal::sparse_eWiseMulAdd_maskDriven<
										descr, a_scalar, x_scalar, y_scalar, y_zero,
										true, true, true, false, true
									>(
										lower_bound, upper_bound, local_z, local_m, local_a, local_x, local_y,
										z_vector, m_vector, a_wrapper, x_wrapper, y_wrapper, ring
									);
							} else {
								return internal::sparse_eWiseMulAdd_maskDriven<
										descr, a_scalar, x_scalar, y_scalar, y_zero,
										true, true, true, false, false
									>(
										lower_bound, upper_bound, local_z, local_m, local_a, local_x, local_y,
										z_vector, m_vector, a_wrapper, x_wrapper, y_wrapper, ring
									);
							}
						}
					} else {
						if( already_dense_input_x ) {
							if( already_dense_input_y ) {
								return internal::sparse_eWiseMulAdd_maskDriven<
										descr, a_scalar, x_scalar, y_scalar, y_zero,
										true, true, false, true, true
									>(
										lower_bound, upper_bound, local_z, local_m, local_a, local_x, local_y,
										z_vector, m_vector, a_wrapper, x_wrapper, y_wrapper, ring
									);
							} else {
								return internal::sparse_eWiseMulAdd_maskDriven<
										descr, a_scalar, x_scalar, y_scalar, y_zero,
										true, true, false, true, false
									>(
										lower_bound, upper_bound, local_z, local_m, local_a, local_x, local_y,
										z_vector, m_vector, a_wrapper, x_wrapper, y_wrapper, ring
									);
							}
						} else {
							if( already_dense_input_y ) {
								return internal::sparse_eWiseMulAdd_maskDriven<
										descr, a_scalar, x_scalar, y_scalar, y_zero,
										true, true, false, false, true
									>(
										lower_bound, upper_bound, local_z, local_m, local_a, local_x, local_y,
										z_vector, m_vector, a_wrapper, x_wrapper, y_wrapper, ring
									);
							} else {
								return internal::sparse_eWiseMulAdd_maskDriven<
										descr, a_scalar, x_scalar, y_scalar, y_zero,
										true, true, false, false, false
									>(
										lower_bound, upper_bound, local_z, local_m, local_a, local_x, local_y,
										z_vector, m_vector, a_wrapper, x_wrapper, y_wrapper, ring
									);
							}
						}
					}
				} else {
					if( already_dense_input_a ) {
						if( already_dense_input_x ) {
							if( already_dense_input_y ) {
								return internal::sparse_eWiseMulAdd_maskDriven<
										descr, a_scalar, x_scalar, y_scalar, y_zero,
										true, false, true, true, true
									>(
										lower_bound, upper_bound, local_z, local_m, local_a, local_x, local_y,
										z_vector, m_vector, a_wrapper, x_wrapper, y_wrapper, ring
									);
							} else {
								return internal::sparse_eWiseMulAdd_maskDriven<
										descr, a_scalar, x_scalar, y_scalar, y_zero,
										true, false, true, true, false
									>(
										lower_bound, upper_bound, local_z, local_m, local_a, local_x, local_y,
										z_vector, m_vector, a_wrapper, x_wrapper, y_wrapper, ring
									);
							}
						} else {
							if( already_dense_input_y ) {
								return internal::sparse_eWiseMulAdd_maskDriven<
										descr, a_scalar, x_scalar, y_scalar, y_zero,
										true, false, true, false, true
									>(
										lower_bound, upper_bound, local_z, local_m, local_a, local_x, local_y,
										z_vector, m_vector, a_wrapper, x_wrapper, y_wrapper, ring
									);
							} else {
								return internal::sparse_eWiseMulAdd_maskDriven<
										descr, a_scalar, x_scalar, y_scalar, y_zero,
										true, false, true, false, false
									>(
										lower_bound, upper_bound, local_z, local_m, local_a, local_x, local_y,
										z_vector, m_vector, a_wrapper, x_wrapper, y_wrapper, ring
									);
							}
						}
					} else {
						if( already_dense_input_x ) {
							if( already_dense_input_y ) {
								return internal::sparse_eWiseMulAdd_maskDriven<
										descr, a_scalar, x_scalar, y_scalar, y_zero,
										true, false, false, true, true
									>(
										lower_bound, upper_bound, local_z, local_m, local_a, local_x, local_y,
										z_vector, m_vector, a_wrapper, x_wrapper, y_wrapper, ring
									);
							} else {
								return internal::sparse_eWiseMulAdd_maskDriven<
										descr, a_scalar, x_scalar, y_scalar, y_zero,
										true, false, false, true, false
									>(
										lower_bound, upper_bound, local_z, local_m, local_a, local_x, local_y,
										z_vector, m_vector, a_wrapper, x_wrapper, y_wrapper, ring
									);
							}
						} else {
							if( already_dense_input_y ) {
								return internal::sparse_eWiseMulAdd_maskDriven<
										descr, a_scalar, x_scalar, y_scalar, y_zero,
										true, false, false, false, true
									>(
										lower_bound, upper_bound, local_z, local_m, local_a, local_x, local_y,
										z_vector, m_vector, a_wrapper, x_wrapper, y_wrapper, ring
									);
							} else {
								return internal::sparse_eWiseMulAdd_maskDriven<
										descr, a_scalar, x_scalar, y_scalar, y_zero,
										true, false, false, false, false
									>(
										lower_bound, upper_bound, local_z, local_m, local_a, local_x, local_y,
										z_vector, m_vector, a_wrapper, x_wrapper, y_wrapper, ring
									);
							}
						}
					}
				}
			} else {
				if( already_dense_mask ) {
					if( already_dense_input_a ) {
						if( already_dense_input_x ) {
							if( already_dense_input_y ) {
								return internal::sparse_eWiseMulAdd_maskDriven<
										descr, a_scalar, x_scalar, y_scalar, y_zero,
										false, true, true, true, true
									>(
										lower_bound, upper_bound, local_z, local_m, local_a, local_x, local_y,
										z_vector, m_vector, a_wrapper, x_wrapper, y_wrapper, ring
									);
							} else {
								return internal::sparse_eWiseMulAdd_maskDriven<
										descr, a_scalar, x_scalar, y_scalar, y_zero,
										false, true, true, true, false
									>(
										lower_bound, upper_bound, local_z, local_m, local_a, local_x, local_y,
										z_vector, m_vector, a_wrapper, x_wrapper, y_wrapper, ring
									);
							}
						} else {
							if( already_dense_input_y ) {
								return internal::sparse_eWiseMulAdd_maskDriven<
										descr, a_scalar, x_scalar, y_scalar, y_zero,
										false, true, true, false, true
									>(
										lower_bound, upper_bound, local_z, local_m, local_a, local_x, local_y,
										z_vector, m_vector, a_wrapper, x_wrapper, y_wrapper, ring
									);
							} else {
								return internal::sparse_eWiseMulAdd_maskDriven<
										descr, a_scalar, x_scalar, y_scalar, y_zero,
										false, true, true, false, false
									>(
										lower_bound, upper_bound, local_z, local_m, local_a, local_x, local_y,
											z_vector, m_vector, a_wrapper, x_wrapper, y_wrapper, ring
									);
							}
						}
					} else {
						if( already_dense_input_x ) {
							if( already_dense_input_y ) {
								return internal::sparse_eWiseMulAdd_maskDriven<
										descr, a_scalar, x_scalar, y_scalar, y_zero,
										false, true, false, true, true
									>(
										lower_bound, upper_bound, local_z, local_m, local_a, local_x, local_y,
										z_vector, m_vector, a_wrapper, x_wrapper, y_wrapper, ring
									);
							} else {
								return internal::sparse_eWiseMulAdd_maskDriven<
										descr, a_scalar, x_scalar, y_scalar, y_zero,
										false, true, false, true, false
									>(
										lower_bound, upper_bound, local_z, local_m, local_a, local_x, local_y,
										z_vector, m_vector, a_wrapper, x_wrapper, y_wrapper, ring
									);
							}
						} else {
							if( already_dense_input_y ) {
								return internal::sparse_eWiseMulAdd_maskDriven<
										descr, a_scalar, x_scalar, y_scalar, y_zero,
										false, true, false, false, true
									>( lower_bound, upper_bound, local_z, local_m, local_a, local_x, local_y,
										z_vector, m_vector, a_wrapper, x_wrapper, y_wrapper, ring
									);
							} else {
								return internal::sparse_eWiseMulAdd_maskDriven<
										descr, a_scalar, x_scalar, y_scalar, y_zero,
										false, true, false, false, false
									>(
										lower_bound, upper_bound, local_z, local_m, local_a, local_x, local_y,
										z_vector, m_vector, a_wrapper, x_wrapper, y_wrapper, ring
									);
							}
						}
					}
				} else {
					if( already_dense_input_a ) {
						if( already_dense_input_x ) {
							if( already_dense_input_y ) {
								return internal::sparse_eWiseMulAdd_maskDriven<
										descr, a_scalar, x_scalar, y_scalar, y_zero,
										false, false, true, true, true
									>(
										lower_bound, upper_bound, local_z, local_m, local_a, local_x, local_y,
										z_vector, m_vector, a_wrapper, x_wrapper, y_wrapper, ring
									);
							} else {
								return internal::sparse_eWiseMulAdd_maskDriven<
										descr, a_scalar, x_scalar, y_scalar, y_zero,
										false, false, true, true, false
									>(
										lower_bound, upper_bound, local_z, local_m, local_a, local_x, local_y,
										z_vector, m_vector, a_wrapper, x_wrapper, y_wrapper, ring
									);
							}
						} else {
							if( already_dense_input_y ) {
								return internal::sparse_eWiseMulAdd_maskDriven<
										descr, a_scalar, x_scalar, y_scalar, y_zero,
										false, false, true, false, true
									>(
										lower_bound, upper_bound, local_z, local_m, local_a, local_x, local_y,
										z_vector, m_vector, a_wrapper, x_wrapper, y_wrapper, ring
									);
							} else {
								return internal::sparse_eWiseMulAdd_maskDriven<
										descr, a_scalar, x_scalar, y_scalar, y_zero,
										false, false, true, false, false
									>(
										lower_bound, upper_bound, local_z, local_m, local_a, local_x, local_y,
										z_vector, m_vector, a_wrapper, x_wrapper, y_wrapper, ring
									);
							}
						}
					} else {
						if( already_dense_input_x ) {
							if( already_dense_input_y ) {
								return internal::sparse_eWiseMulAdd_maskDriven<
										descr, a_scalar, x_scalar, y_scalar, y_zero,
										false, false, false, true, true
									>(
										lower_bound, upper_bound, local_z, local_m, local_a, local_x, local_y,
										z_vector, m_vector, a_wrapper, x_wrapper, y_wrapper, ring
									);
							} else {
								return internal::sparse_eWiseMulAdd_maskDriven<
										descr, a_scalar, x_scalar, y_scalar, y_zero,
										false, false, false, true, false
									>(
										lower_bound, upper_bound, local_z, local_m, local_a, local_x, local_y,
										z_vector, m_vector, a_wrapper, x_wrapper, y_wrapper, ring
									);
							}
						} else {
							if( already_dense_input_y ) {
								return internal::sparse_eWiseMulAdd_maskDriven<
										descr, a_scalar, x_scalar, y_scalar, y_zero,
										false, false, false, false, true
									>(
										lower_bound, upper_bound, local_z, local_m, local_a, local_x, local_y,
										z_vector, m_vector, a_wrapper, x_wrapper, y_wrapper, ring
									);
							} else {
								return internal::sparse_eWiseMulAdd_maskDriven<
										descr, a_scalar, x_scalar, y_scalar, y_zero,
										false, false, false, false, false
									>(
										lower_bound, upper_bound, local_z, local_m, local_a, local_x, local_y,
										z_vector, m_vector, a_wrapper, x_wrapper, y_wrapper, ring
									);
							}
						}
					}
				}
			}
		}

		template<
			Descriptor descr,
			bool masked,
			bool x_scalar,
			bool y_scalar,
			bool y_zero,
			bool mulSwitched,
			bool already_dense_output,
			bool already_dense_mask,
			bool already_dense_input_a,
			bool already_dense_input_x,
			bool already_dense_input_y,
			typename OutputType,
			typename MaskType,
			typename InputType1,
			typename InputType2,
			typename InputType3,
			typename Coords,
			class Ring
		>
		RC twoPhase_sparse_eWiseMulAdd_mulDriven(
			const size_t lower_bound,
			const size_t upper_bound,
			Coords &local_z,
			const Coords * const local_m,
			const Coords &local_a,
			const Coords &local_x,
			const Coords &local_y,
			Vector< OutputType, nonblocking, Coords > &z_vector,
			const Vector< MaskType, nonblocking, Coords > * const m_vector,
			const Vector< InputType1, nonblocking, Coords > &a_vector,
			const internal::Wrapper< x_scalar, InputType2, Coords > &x_wrapper,
			const internal::Wrapper< y_scalar, InputType3, Coords > &y_wrapper,
			const Ring &ring
		);

		template<
			Descriptor descr,
			bool masked,
			bool x_scalar,
			bool y_scalar,
			bool y_zero,
			bool mulSwitched,
			typename OutputType,
			typename MaskType,
			typename InputType1,
			typename InputType2,
			typename InputType3,
			typename Coords,
			class Ring
		>
		RC boolean_dispatcher_twoPhase_sparse_eWiseMulAdd_mulDriven(
			const bool already_dense_output,
			const bool already_dense_mask,
			const bool already_dense_input_a,
			const bool already_dense_input_x,
			const bool already_dense_input_y,
			const size_t lower_bound,
			const size_t upper_bound,
			Coords &local_z,
			const Coords * const local_m,
			const Coords &local_a,
			const Coords &local_x,
			const Coords &local_y,
			Vector< OutputType, nonblocking, Coords > &z_vector,
			const Vector< MaskType, nonblocking, Coords > * const m_vector,
			const Vector< InputType1, nonblocking, Coords > &a_vector,
			const internal::Wrapper< x_scalar, InputType2, Coords > &x_wrapper,
			const internal::Wrapper< y_scalar, InputType3, Coords > &y_wrapper,
			const Ring &ring = Ring()
		) {
			if( already_dense_output ) {
				if( already_dense_mask ) {
					if( already_dense_input_a ) {
						if( already_dense_input_x ) {
							if( already_dense_input_y ) {
								return internal::twoPhase_sparse_eWiseMulAdd_mulDriven<
										descr, masked, x_scalar, y_scalar, y_zero, mulSwitched,
										true, true, true, true, true
									>(
										lower_bound, upper_bound, local_z, local_m, local_a, local_x, local_y,
										z_vector, m_vector, a_vector, x_wrapper, y_wrapper, ring
									);
							} else {
								return internal::twoPhase_sparse_eWiseMulAdd_mulDriven<
										descr, masked, x_scalar, y_scalar, y_zero, mulSwitched,
										true, true, true, true, false
									>(
										lower_bound, upper_bound, local_z, local_m, local_a, local_x, local_y,
										z_vector, m_vector, a_vector, x_wrapper, y_wrapper, ring
									);
							}
						} else {
							if( already_dense_input_y ) {
								return internal::twoPhase_sparse_eWiseMulAdd_mulDriven<
										descr, masked, x_scalar, y_scalar, y_zero, mulSwitched,
										true, true, true, false, true
									>(
										lower_bound, upper_bound, local_z, local_m, local_a, local_x, local_y,
										z_vector, m_vector, a_vector, x_wrapper, y_wrapper, ring
									);
							} else {
								return internal::twoPhase_sparse_eWiseMulAdd_mulDriven<
										descr, masked, x_scalar, y_scalar, y_zero, mulSwitched,
										true, true, true, false, false
									>(
										lower_bound, upper_bound, local_z, local_m, local_a, local_x, local_y,
										z_vector, m_vector, a_vector, x_wrapper, y_wrapper, ring
									);
							}
						}
					} else {
						if( already_dense_input_x ) {
							if( already_dense_input_y ) {
								return internal::twoPhase_sparse_eWiseMulAdd_mulDriven<
										descr, masked, x_scalar, y_scalar, y_zero, mulSwitched,
										true, true, false, true, true
									>(
										lower_bound, upper_bound, local_z, local_m, local_a, local_x, local_y,
										z_vector, m_vector, a_vector, x_wrapper, y_wrapper, ring
									);
							} else {
								return internal::twoPhase_sparse_eWiseMulAdd_mulDriven<
										descr, masked, x_scalar, y_scalar, y_zero, mulSwitched,
										true, true, false, true, false
									>(
										lower_bound, upper_bound, local_z, local_m, local_a, local_x, local_y,
										z_vector, m_vector, a_vector, x_wrapper, y_wrapper, ring
									);
							}
						} else {
							if( already_dense_input_y ) {
								return internal::twoPhase_sparse_eWiseMulAdd_mulDriven<
										descr, masked, x_scalar, y_scalar, y_zero, mulSwitched,
										true, true, false, false, true
									>(
										lower_bound, upper_bound, local_z, local_m, local_a, local_x, local_y,
										z_vector, m_vector, a_vector, x_wrapper, y_wrapper, ring
									);
							} else {
								return internal::twoPhase_sparse_eWiseMulAdd_mulDriven<
										descr, masked, x_scalar, y_scalar, y_zero, mulSwitched,
										true, true, false, false, false
									>(
										lower_bound, upper_bound, local_z, local_m, local_a, local_x, local_y,
										z_vector, m_vector, a_vector, x_wrapper, y_wrapper, ring
									);
							}
						}
					}
				} else {
					if( already_dense_input_a ) {
						if( already_dense_input_x ) {
							if( already_dense_input_y ) {
								return internal::twoPhase_sparse_eWiseMulAdd_mulDriven<
										descr, masked, x_scalar, y_scalar, y_zero, mulSwitched,
										true, false, true, true, true
									>(
										lower_bound, upper_bound, local_z, local_m, local_a, local_x, local_y,
										z_vector, m_vector, a_vector, x_wrapper, y_wrapper, ring
									);
							} else {
								return internal::twoPhase_sparse_eWiseMulAdd_mulDriven<
										descr, masked, x_scalar, y_scalar, y_zero, mulSwitched,
										true, false, true, true, false
									>(
										lower_bound, upper_bound, local_z, local_m, local_a, local_x, local_y,
										z_vector, m_vector, a_vector, x_wrapper, y_wrapper, ring
									);
							}
						} else {
							if( already_dense_input_y ) {
								return internal::twoPhase_sparse_eWiseMulAdd_mulDriven<
										descr, masked, x_scalar, y_scalar, y_zero, mulSwitched,
										true, false, true, false, true
									>(
										lower_bound, upper_bound, local_z, local_m, local_a, local_x, local_y,
										z_vector, m_vector, a_vector, x_wrapper, y_wrapper, ring
									);
							} else {
								return internal::twoPhase_sparse_eWiseMulAdd_mulDriven<
										descr, masked, x_scalar, y_scalar, y_zero, mulSwitched,
										true, false, true, false, false
									>(
										lower_bound, upper_bound, local_z, local_m, local_a, local_x, local_y,
										z_vector, m_vector, a_vector, x_wrapper, y_wrapper, ring
									);
							}
						}
					} else {
						if( already_dense_input_x ) {
							if( already_dense_input_y ) {
								return internal::twoPhase_sparse_eWiseMulAdd_mulDriven<
										descr, masked, x_scalar, y_scalar, y_zero, mulSwitched,
										true, false, false, true, true
									>(
										lower_bound, upper_bound, local_z, local_m, local_a, local_x, local_y,
										z_vector, m_vector, a_vector, x_wrapper, y_wrapper, ring
									);
							} else {
								return internal::twoPhase_sparse_eWiseMulAdd_mulDriven<
										descr, masked, x_scalar, y_scalar, y_zero, mulSwitched,
										true, false, false, true, false
									>(
										lower_bound, upper_bound, local_z, local_m, local_a, local_x, local_y,
										z_vector, m_vector, a_vector, x_wrapper, y_wrapper, ring
									);
							}
						} else {
							if( already_dense_input_y ) {
								return internal::twoPhase_sparse_eWiseMulAdd_mulDriven<
										descr, masked, x_scalar, y_scalar, y_zero, mulSwitched,
										true, false, false, false, true
									>(
										lower_bound, upper_bound, local_z, local_m, local_a, local_x, local_y,
										z_vector, m_vector, a_vector, x_wrapper, y_wrapper, ring
									);
							} else {
								return internal::twoPhase_sparse_eWiseMulAdd_mulDriven<
										descr, masked, x_scalar, y_scalar, y_zero, mulSwitched,
										true, false, false, false, false
									>(
										lower_bound, upper_bound, local_z, local_m, local_a, local_x, local_y,
										z_vector, m_vector, a_vector, x_wrapper, y_wrapper, ring
									);
							}
						}
					}
				}
			} else {
				if( already_dense_mask ) {
					if( already_dense_input_a ) {
						if( already_dense_input_x ) {
							if( already_dense_input_y ) {
								return internal::twoPhase_sparse_eWiseMulAdd_mulDriven<
										descr, masked, x_scalar, y_scalar, y_zero, mulSwitched,
										false, true, true, true, true
									>(
										lower_bound, upper_bound, local_z, local_m, local_a, local_x, local_y,
										z_vector, m_vector, a_vector, x_wrapper, y_wrapper, ring
									);
							} else {
								return internal::twoPhase_sparse_eWiseMulAdd_mulDriven<
										descr, masked, x_scalar, y_scalar, y_zero, mulSwitched,
										false, true, true, true, false
									>(
										lower_bound, upper_bound, local_z, local_m, local_a, local_x, local_y,
										z_vector, m_vector, a_vector, x_wrapper, y_wrapper, ring
									);
							}
						} else {
							if( already_dense_input_y ) {
								return internal::twoPhase_sparse_eWiseMulAdd_mulDriven<
										descr, masked, x_scalar, y_scalar, y_zero, mulSwitched,
										false, true, true, false, true
									>(
										lower_bound, upper_bound, local_z, local_m, local_a, local_x, local_y,
										z_vector, m_vector, a_vector, x_wrapper, y_wrapper, ring
									);
							} else {
								return internal::twoPhase_sparse_eWiseMulAdd_mulDriven<
										descr, masked, x_scalar, y_scalar, y_zero, mulSwitched,
										false, true, true, false, false
									>(
										lower_bound, upper_bound, local_z, local_m, local_a, local_x, local_y,
										z_vector, m_vector, a_vector, x_wrapper, y_wrapper, ring
									);
							}
						}
					} else {
						if( already_dense_input_x ) {
							if( already_dense_input_y ) {
								return internal::twoPhase_sparse_eWiseMulAdd_mulDriven<
										descr, masked, x_scalar, y_scalar, y_zero, mulSwitched,
										false, true, false, true, true
									>(
										lower_bound, upper_bound, local_z, local_m, local_a, local_x, local_y,
										z_vector, m_vector, a_vector, x_wrapper, y_wrapper, ring
									);
							} else {
								return internal::twoPhase_sparse_eWiseMulAdd_mulDriven<
										descr, masked, x_scalar, y_scalar, y_zero, mulSwitched,
										false, true, false, true, false
									>(
										lower_bound, upper_bound, local_z, local_m, local_a, local_x, local_y,
										z_vector, m_vector, a_vector, x_wrapper, y_wrapper, ring
									);
							}
						} else {
							if( already_dense_input_y ) {
								return internal::twoPhase_sparse_eWiseMulAdd_mulDriven<
										descr, masked, x_scalar, y_scalar, y_zero, mulSwitched,
										false, true, false, false, true
									>(
										lower_bound, upper_bound, local_z, local_m, local_a, local_x, local_y,
										z_vector, m_vector, a_vector, x_wrapper, y_wrapper, ring
									);
							} else {
								return internal::twoPhase_sparse_eWiseMulAdd_mulDriven<
										descr, masked, x_scalar, y_scalar, y_zero, mulSwitched,
										false, true, false, false, false
									>(
										lower_bound, upper_bound, local_z, local_m, local_a, local_x, local_y,
										z_vector, m_vector, a_vector, x_wrapper, y_wrapper, ring
									);
							}
						}
					}
				} else {
					if( already_dense_input_a ) {
						if( already_dense_input_x ) {
							if( already_dense_input_y ) {
								return internal::twoPhase_sparse_eWiseMulAdd_mulDriven<
										descr, masked, x_scalar, y_scalar, y_zero, mulSwitched,
										false, false, true, true, true
									>(
										lower_bound, upper_bound, local_z, local_m, local_a, local_x, local_y,
										z_vector, m_vector, a_vector, x_wrapper, y_wrapper, ring
									);
							} else {
								return internal::twoPhase_sparse_eWiseMulAdd_mulDriven<
										descr, masked, x_scalar, y_scalar, y_zero, mulSwitched,
										false, false, true, true, false
									>(
										lower_bound, upper_bound, local_z, local_m, local_a, local_x, local_y,
										z_vector, m_vector, a_vector, x_wrapper, y_wrapper, ring
									);
							}
						} else {
							if( already_dense_input_y ) {
								return internal::twoPhase_sparse_eWiseMulAdd_mulDriven<
										descr, masked, x_scalar, y_scalar, y_zero, mulSwitched,
										false, false, true, false, true
									>(
										lower_bound, upper_bound, local_z, local_m, local_a, local_x, local_y,
										z_vector, m_vector, a_vector, x_wrapper, y_wrapper, ring
									);
							} else {
								return internal::twoPhase_sparse_eWiseMulAdd_mulDriven<
										descr, masked, x_scalar, y_scalar, y_zero, mulSwitched,
										false, false, true, false, false
									>(
										lower_bound, upper_bound, local_z, local_m, local_a, local_x, local_y,
										z_vector, m_vector, a_vector, x_wrapper, y_wrapper, ring
									);
							}
						}
					} else {
						if( already_dense_input_x ) {
							if( already_dense_input_y ) {
								return internal::twoPhase_sparse_eWiseMulAdd_mulDriven<
										descr, masked, x_scalar, y_scalar, y_zero, mulSwitched,
										false, false, false, true, true
									>(
										lower_bound, upper_bound, local_z, local_m, local_a, local_x, local_y,
										z_vector, m_vector, a_vector, x_wrapper, y_wrapper, ring
									);
							} else {
								return internal::twoPhase_sparse_eWiseMulAdd_mulDriven<
										descr, masked, x_scalar, y_scalar, y_zero, mulSwitched,
										false, false, false, true, false
									>(
										lower_bound, upper_bound, local_z, local_m, local_a, local_x, local_y,
										z_vector, m_vector, a_vector, x_wrapper, y_wrapper, ring
									);
							}
						} else {
							if( already_dense_input_y ) {
								return internal::twoPhase_sparse_eWiseMulAdd_mulDriven<
										descr, masked, x_scalar, y_scalar, y_zero, mulSwitched,
										false, false, false, false, true
									>(
										lower_bound, upper_bound, local_z, local_m, local_a, local_x, local_y,
										z_vector, m_vector, a_vector, x_wrapper, y_wrapper, ring
									);
							} else {
								return internal::twoPhase_sparse_eWiseMulAdd_mulDriven<
										descr, masked, x_scalar, y_scalar, y_zero, mulSwitched,
										false, false, false, false, false
									>(
										lower_bound, upper_bound, local_z, local_m, local_a, local_x, local_y,
										z_vector, m_vector, a_vector, x_wrapper, y_wrapper, ring
									);
							}
						}
					}
				}
			}
		}

		template<
			Descriptor descr,
			bool already_dense_input_x,
			bool already_dense_input_y,
			class AddMonoid,
			class AnyOp,
			typename InputType1,
			typename InputType2,
			typename Coords
		>
		RC sparse_dot_generic(
			typename AddMonoid::D3 &thread_local_output,
			const size_t lower_bound,
			const size_t upper_bound,
			const Coords &local_x,
			const Coords &local_y,
			const Vector< InputType1, nonblocking, Coords > &x,
			const Vector< InputType2, nonblocking, Coords > &y,
			const size_t local_nz,
			const AddMonoid &addMonoid,
			const AnyOp &anyOp
		);

		template<
			Descriptor descr,
			class AddMonoid,
			class AnyOp,
			typename InputType1,
			typename InputType2,
			typename Coords
		>
		RC boolean_dispatcher_sparse_dot_generic(
			const bool already_dense_input_x,
			const bool already_dense_input_y,
			typename AddMonoid::D3 &thread_local_output,
			const size_t lower_bound,
			const size_t upper_bound,
			const Coords &local_x,
			const Coords &local_y,
			const Vector< InputType1, nonblocking, Coords > &x,
			const Vector< InputType2, nonblocking, Coords > &y,
			const size_t local_nz,
			const AddMonoid &addMonoid,
			const AnyOp &anyOp
		) {
			if( already_dense_input_x ) {
				if( already_dense_input_y ) {
					return internal::sparse_dot_generic<
							descr, true, true
						>(
							thread_local_output, lower_bound, upper_bound, local_x, local_y,
							x, y, local_nz, addMonoid, anyOp
						);
				} else {
					return internal::sparse_dot_generic<
							descr, true, false
						>(
							thread_local_output, lower_bound, upper_bound, local_x, local_y,
							x, y, local_nz, addMonoid, anyOp
						);
				}
			} else {
				if( already_dense_input_y ) {
					return internal::sparse_dot_generic<
							descr, false, true
						>(
							thread_local_output, lower_bound, upper_bound, local_x, local_y,
							x, y, local_nz, addMonoid, anyOp
						);
				} else {
					return internal::sparse_dot_generic<
							descr, false, false
						>(
							thread_local_output, lower_bound, upper_bound, local_x, local_y,
							x, y, local_nz, addMonoid, anyOp
						);
				}
			}
		}

	} // end namespace ``internal''

} // end namespace ``grb''

#endif

