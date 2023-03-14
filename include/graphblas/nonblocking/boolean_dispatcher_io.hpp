
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
 * Dispatchers for the nonblocking i/o primitives.
 *
 * @author Aristeidis Mastoras
 * @date 24th of October, 2022
 */

#ifndef _H_GRB_NONBLOCKING_BOOLEAN_DISPATCHER_IO
#define _H_GRB_NONBLOCKING_BOOLEAN_DISPATCHER_IO

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


namespace grb {

	namespace internal {

		template<
			Descriptor descr,
			bool loop_over_vector_length,
			bool already_dense_mask,
			bool mask_is_dense,
			typename DataType,
			typename MaskType,
			typename T,
			typename Coords
		>
		RC masked_set(
			const size_t lower_bound,
			const size_t upper_bound,
			Coords &local_x,
			const Coords &local_mask,
			Vector< DataType, nonblocking, Coords > &x,
			const Vector< MaskType, nonblocking, Coords > &m,
			const T val
		);

		template<
			Descriptor descr,
			typename DataType,
			typename MaskType,
			typename T,
			typename Coords
		>
		RC boolean_dispatcher_masked_set(
			const bool loop_over_vector_length,
			const bool already_dense_mask,
			const bool mask_is_dense,
			const size_t lower_bound,
			const size_t upper_bound,
			Coords &local_x,
			const Coords &local_mask,
			Vector< DataType, nonblocking, Coords > &x,
			const Vector< MaskType, nonblocking, Coords > &m,
			const T val
		) {
			if( loop_over_vector_length ) {
				if( already_dense_mask ) {
					if( mask_is_dense ) {
						return internal::masked_set<
								descr, true, true, true
							>( lower_bound, upper_bound, local_x, local_mask, x, m, val );
					} else {
						return internal::masked_set<
								descr, true, true, false
							>( lower_bound, upper_bound, local_x, local_mask, x, m, val );
					}
				} else {
					if( mask_is_dense ) {
						return internal::masked_set<
								descr, true, false, true
							>( lower_bound, upper_bound, local_x, local_mask, x, m, val );
					} else {
						return internal::masked_set<
								descr, true, false, false
							>( lower_bound, upper_bound, local_x, local_mask, x, m, val );
					}
				}
			} else {
				if( already_dense_mask ) {
					if( mask_is_dense ) {
						return internal::masked_set<
								descr, false, true, true
							>( lower_bound, upper_bound, local_x, local_mask, x, m, val );
					} else {
						return internal::masked_set<
								descr, false, true, false
							>( lower_bound, upper_bound, local_x, local_mask, x, m, val );
					}
				} else {
					if( mask_is_dense ) {
						return internal::masked_set<
								descr, false, false, true
							>( lower_bound, upper_bound, local_x, local_mask, x, m, val );
					} else {
						return internal::masked_set<
								descr, false, false, false
							>( lower_bound, upper_bound, local_x, local_mask, x, m, val );
					}
				}
			}
		}

		template<
			Descriptor descr,
			bool out_is_void,
			bool in_is_void,
			bool sparse,
			bool already_dense_vectors,
			bool already_dense_input,
			typename OutputType,
			typename InputType,
			typename Coords
		>
		RC set_generic(
			const size_t lower_bound,
			const size_t upper_bound,
			Coords &local_x,
			const Coords &local_y,
			Vector< OutputType, nonblocking, Coords > &x,
			const Vector< InputType, nonblocking, Coords > &y
		);

		template< Descriptor descr,
			bool out_is_void,
			bool in_is_void,
			bool sparse,
			typename OutputType,
			typename InputType,
			typename Coords
		>
		RC boolean_dispatcher_set_generic(
			const bool already_dense_vectors,
			const bool already_dense_input,
			const size_t lower_bound,
			const size_t upper_bound,
			Coords &local_x,
			const Coords &local_y,
			Vector< OutputType, nonblocking, Coords > &x,
			const Vector< InputType, nonblocking, Coords > &y
		) {
			if( already_dense_vectors ) {
				if( already_dense_input ) {
					return internal::set_generic<
							descr, out_is_void, in_is_void, sparse,
							true, true
						>( lower_bound, upper_bound, local_x, local_y, x, y );
				} else {
					return internal::set_generic<
							descr, out_is_void, in_is_void, sparse,
							true, false
						>( lower_bound, upper_bound, local_x, local_y, x, y );
				}
			} else {
				if( already_dense_input ) {
					return internal::set_generic<
							descr, out_is_void, in_is_void, sparse,
							false, true
						>( lower_bound, upper_bound, local_x, local_y, x, y );
				} else {
					return internal::set_generic<
							descr, out_is_void, in_is_void, sparse,
							false, false
						>( lower_bound, upper_bound, local_x, local_y, x, y );
				}
			}
		}

		template<
			Descriptor descr,
			bool out_is_void,
			bool in_is_void,
			bool loop_over_y,
			bool already_dense_input_y,
			bool already_dense_mask,
			bool mask_is_dense,
			typename OutputType,
			typename MaskType,
			typename InputType,
			typename Coords
		>
		RC masked_set(
			const size_t lower_bound,
			const size_t upper_bound,
			Coords &local_x,
			const Coords &local_mask,
			const Coords &local_y,
			Vector< OutputType, nonblocking, Coords > &x,
			const Vector< MaskType, nonblocking, Coords > &mask,
			const Vector< InputType, nonblocking, Coords > &y
		);

		template<
			Descriptor descr,
			bool out_is_void,
			bool in_is_void,
			typename OutputType,
			typename MaskType,
			typename InputType,
			typename Coords
		>
		RC boolean_dispatcher_masked_set(
			const bool loop_over_y,
			const bool already_dense_input_y,
			const bool already_dense_mask,
			const bool mask_is_dense,
			const size_t lower_bound,
			const size_t upper_bound,
			Coords &local_x,
			const Coords &local_mask,
			const Coords &local_y,
			Vector< OutputType, nonblocking, Coords > &x,
			const Vector< MaskType, nonblocking, Coords > &mask,
			const Vector< InputType, nonblocking, Coords > &y
		) {
			if( loop_over_y ) {
				if( already_dense_input_y ) {
					if( already_dense_mask ) {
						if( mask_is_dense ) {
							return internal::masked_set<
									descr, out_is_void, in_is_void,
									true, true, true, true
								>( lower_bound, upper_bound, local_x, local_mask, local_y, x, mask, y );
						} else {
							return internal::masked_set<
									descr, out_is_void, in_is_void,
									true, true, true, false
								>( lower_bound, upper_bound, local_x, local_mask, local_y, x, mask, y );
						}
					} else {
						if( mask_is_dense ) {
							return internal::masked_set<
									descr, out_is_void, in_is_void,
									true, true, false, true
								>( lower_bound, upper_bound, local_x, local_mask, local_y, x, mask, y );
						} else {
							return internal::masked_set<
									descr, out_is_void, in_is_void,
									true, true, false, false
								>( lower_bound, upper_bound, local_x, local_mask, local_y, x, mask, y );
						}
					}
				} else {
					if( already_dense_mask ) {
						if( mask_is_dense ) {
							return internal::masked_set<
									descr, out_is_void, in_is_void,
									true, false, true, true
								>( lower_bound, upper_bound, local_x, local_mask, local_y, x, mask, y );
						} else {
							return internal::masked_set<
									descr, out_is_void, in_is_void,
									true, false, true, false
								>( lower_bound, upper_bound, local_x, local_mask, local_y, x, mask, y );
						}
					} else {
						if( mask_is_dense ) {
							return internal::masked_set<
									descr, out_is_void, in_is_void,
									true, false, false, true
								>( lower_bound, upper_bound, local_x, local_mask, local_y, x, mask, y );
						} else {
							return internal::masked_set<
									descr, out_is_void, in_is_void,
									true, false, false, false
								>( lower_bound, upper_bound, local_x, local_mask, local_y, x, mask, y );
						}
					}
				}
			} else {
				if( already_dense_input_y ) {
					if( already_dense_mask ) {
						if( mask_is_dense ) {
							return internal::masked_set<
									descr, out_is_void, in_is_void,
									false, true, true, true
								>( lower_bound, upper_bound, local_x, local_mask, local_y, x, mask, y );
						} else {
							return internal::masked_set<
									descr, out_is_void, in_is_void,
									false, true, true, false
								>( lower_bound, upper_bound, local_x, local_mask, local_y, x, mask, y );
						}
					} else {
						if( mask_is_dense ) {
							return internal::masked_set<
									descr, out_is_void, in_is_void,
									false, true, false, true
								>( lower_bound, upper_bound, local_x, local_mask, local_y, x, mask, y );
						} else {
							return internal::masked_set<
									descr, out_is_void, in_is_void,
									false, true, false, false
								>( lower_bound, upper_bound, local_x, local_mask, local_y, x, mask, y );
						}
					}
				} else {
					if( already_dense_mask ) {
						if( mask_is_dense ) {
							return internal::masked_set<
									descr, out_is_void, in_is_void,
									false, false, true, true
								>( lower_bound, upper_bound, local_x, local_mask, local_y, x, mask, y );
						} else {
							return internal::masked_set<
									descr, out_is_void, in_is_void,
									false, false, true, false
								>( lower_bound, upper_bound, local_x, local_mask, local_y, x, mask, y );
						}
					} else {
						if( mask_is_dense ) {
							return internal::masked_set<
									descr, out_is_void, in_is_void,
									false, false, false, true
								>( lower_bound, upper_bound, local_x, local_mask, local_y, x, mask, y );
						} else {
							return internal::masked_set<
									descr, out_is_void, in_is_void,
									false, false, false, false
								>( lower_bound, upper_bound, local_x, local_mask, local_y, x, mask, y );
						}
					}
				}
			}
		}

	} // end namespace ``internal''

} // end namespace ``grb''

#endif

