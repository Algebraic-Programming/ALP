
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
 * Level-1 primitive implementation for nonblocking.
 *
 * \internal
 * \todo Relies significantly on a past reference level-1 implementation. Can we
 *       reuse?
 * \endinternal
 *
 * @author Aristeidis Mastoras
 * @date 16th of May, 2022
 */

#ifndef _H_GRB_NONBLOCKING_BLAS1
#define _H_GRB_NONBLOCKING_BLAS1

#include <iostream>    //for printing to stderr
#include <type_traits> //for std::enable_if

#include <omp.h>

#include <graphblas/utils/suppressions.h>
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
#include "boolean_dispatcher_blas1.hpp"

#define NO_CAST_ASSERT( x, y, z )                                              \
	static_assert( x,                                                          \
		"\n\n"                                                                 \
		"********************************************************************" \
		"********************************************************************" \
		"******************************\n"                                     \
		"*     ERROR      | " y " " z ".\n"                                    \
		"********************************************************************" \
		"********************************************************************" \
		"******************************\n"                                     \
		"* Possible fix 1 | Remove no_casting from the template parameters "   \
		"in this call to " y ".\n"                                             \
		"* Possible fix 2 | Provide a value that matches the expected type.\n" \
		"********************************************************************" \
		"********************************************************************" \
		"******************************\n" );

#define NO_CAST_OP_ASSERT( x, y, z )                                           \
	static_assert( x,                                                          \
		"\n\n"                                                                 \
		"********************************************************************" \
		"********************************************************************" \
		"******************************\n"                                     \
		"*     ERROR      | " y " " z ".\n"                                    \
		"********************************************************************" \
		"********************************************************************" \
		"******************************\n"                                     \
		"* Possible fix 1 | Remove no_casting from the template parameters "   \
		"in this call to " y ".\n"                                             \
		"* Possible fix 2 | For all mismatches in the domains of input "       \
		"parameters and the operator domains, as specified in the "            \
		"documentation of the function " y ", supply an input argument of "    \
		"the expected type instead.\n"                                         \
		"* Possible fix 3 | Provide a compatible operator where all domains "  \
		"match those of the input parameters, as specified in the "            \
		"documentation of the function " y ".\n"                               \
		"********************************************************************" \
		"********************************************************************" \
		"******************************\n" );


namespace grb {

	namespace internal {

		extern LazyEvaluation le;

	}

}

namespace grb {

	/**
	 * \defgroup BLAS1_NB The Level-1 ALP/GraphBLAS routines -- nonblocking backend
	 *
	 * @{
	 */

	namespace internal {

		template<
			bool left,
			class Monoid,
			typename InputType,
			class Coords
		>
		RC fold_from_vector_to_scalar_dense(
			typename Monoid::D3 &thread_local_output,
			const size_t lower_bound,
			const size_t upper_bound,
			const Vector< InputType, nonblocking, Coords > &to_fold,
			const Monoid &monoid
		) {
			const InputType *__restrict__ const raw = internal::getRaw( to_fold );

			const size_t start = lower_bound;
			const size_t end = upper_bound;

			if( start < end ) {
				if( left ) {
					monoid.getOperator().foldlArray(
						thread_local_output, raw + start, end - start );
				} else {
					monoid.getOperator().foldrArray(
						raw + start, thread_local_output, end - start );
				}
			}
			return SUCCESS;
		}

		template<
			Descriptor descr,
			bool masked,
			bool left,
#ifdef GRB_BOOLEAN_DISPATCHER
			bool already_dense_input_to_fold,
			bool already_dense_mask,
#endif
			class Monoid,
			typename InputType,
			typename MaskType,
			class Coords
		>
		RC fold_from_vector_to_scalar_vectorDriven(
#ifndef GRB_BOOLEAN_DISPATCHER
			bool already_dense_input_to_fold,
			bool already_dense_mask,
#endif
			typename Monoid::D3 &thread_local_output,
			const size_t lower_bound,
			const size_t upper_bound,
			const Coords &local_to_fold,
			const Coords &local_mask,
			const Vector< InputType, nonblocking, Coords > &to_fold,
			const Vector< MaskType, nonblocking, Coords > &mask,
			const Monoid &monoid
		) {
			const size_t n = internal::getCoordinates( to_fold ).size();
			const size_t local_n = upper_bound - lower_bound;
			const size_t local_to_fold_nz = ( already_dense_input_to_fold )
				? local_n
				: local_to_fold.nonzeroes();

			assert( n > 0 );
			assert( !masked || internal::getCoordinates( mask ).size() == n );

#ifdef NDEBUG
			(void) n;
			(void) local_n;
#endif

			RC ret = SUCCESS;

			const size_t start = 0;
			const size_t end = local_to_fold_nz;

			// compute thread-local partial reduction
			for( size_t k = start; k < end; ++k ) {
				const size_t i = ( (already_dense_input_to_fold)
					? k
					: local_to_fold.index( k ) ) + lower_bound;
				if( masked ) {
					if( already_dense_mask ) {
						if( !utils::interpretMask< descr >(
							internal::getCoordinates( mask ).assigned( i ),
							internal::getRaw( mask ), i )
						) {
							continue;
						}
					} else {
						if( !utils::interpretMask< descr >(
							local_mask.assigned( i - lower_bound ), internal::getRaw( mask ), i )
						) {
							continue;
						}
					}
				}
				RC local_rc;
				if( left ) {
					local_rc = foldl< descr >( thread_local_output,
						internal::getRaw( to_fold )[ i ], monoid.getOperator() );
				} else {
					local_rc = foldr< descr >( internal::getRaw( to_fold )[ i ],
						thread_local_output, monoid.getOperator() );
				}
				assert( local_rc == SUCCESS );
				if( local_rc != SUCCESS ) {
					ret = local_rc;
				}
			}

			return ret;
		}

		template<
			Descriptor descr,
			bool left,
#ifdef GRB_BOOLEAN_DISPATCHER
			bool already_dense_input_to_fold,
			bool already_dense_mask,
#endif
			class Monoid,
			typename InputType,
			typename MaskType,
			class Coords
		>
		RC fold_from_vector_to_scalar_maskDriven(
#ifndef GRB_BOOLEAN_DISPATCHER
			bool already_dense_input_to_fold,
			bool already_dense_mask,
#endif
			typename Monoid::D3 &thread_local_output,
			const size_t lower_bound,
			const size_t upper_bound,
			const Coords &local_to_fold,
			const Coords &local_mask,
			const Vector< InputType, nonblocking, Coords > &to_fold,
			const Vector< MaskType, nonblocking, Coords > &mask,
			const Monoid &monoid
		) {
			const size_t n = internal::getCoordinates( to_fold ).size();

			assert( internal::getCoordinates( mask ).size() == n );
			assert( n > 0 );
#ifdef NDEBUG
			(void) n;
#endif
			const size_t local_n = upper_bound - lower_bound;
			const size_t local_mask_nz = ( already_dense_mask )
				? local_n
				: local_mask.nonzeroes();

			RC ret = SUCCESS;

			const size_t start = 0;
			const size_t end = local_mask_nz;

			// compute thread-local partial reduction
			for( size_t k = start; k < end; ++k ) {
				const size_t i = ( (already_dense_mask)
						? k
						: local_mask.index( k )
					) + lower_bound;
				if( !( already_dense_input_to_fold ||
					local_to_fold.assigned( i - lower_bound ) )
				) {
					continue;
				}
				if( !utils::interpretMask< descr >( true, internal::getRaw( mask ), i ) ) {
					continue;
				}
				RC local_rc;
				if( left ) {
					local_rc = foldl< descr >( thread_local_output,
						internal::getRaw( to_fold )[ i ], monoid.getOperator() );
				} else {
					local_rc = foldr< descr >( internal::getRaw( to_fold )[ i ],
						thread_local_output, monoid.getOperator() );
				}
				assert( local_rc == SUCCESS );
				if( local_rc != SUCCESS ) {
					ret = local_rc;
				}
			}

			return ret;
		}

		template<
			Descriptor descr,
			bool masked,
			bool left,
#ifdef GRB_BOOLEAN_DISPATCHER
			bool already_dense_input_to_fold,
			bool already_dense_mask,
#endif
			class Monoid,
			typename InputType,
			typename MaskType,
			class Coords
		>
		RC fold_from_vector_to_scalar_fullLoopSparse(
#ifndef GRB_BOOLEAN_DISPATCHER
			bool already_dense_input_to_fold,
			bool already_dense_mask,
#endif
			typename Monoid::D3 &thread_local_output,
			const size_t lower_bound,
			const size_t upper_bound,
			const Coords &local_to_fold,
			const Coords &local_mask,
			const Vector< InputType, nonblocking, Coords > &to_fold,
			const Vector< MaskType, nonblocking, Coords > &mask,
			const Monoid &monoid
		) {
#ifdef _DEBUG
			std::cout << "Entered fold_from_vector_to_scalar_fullLoopSparse\n";
#endif

#ifndef NDEBUG
			const size_t n = internal::getCoordinates( to_fold ).size();
			const size_t local_n = already_dense_input_to_fold
				? upper_bound - lower_bound
				: local_to_fold.size();
			assert( local_n > 0 );

			(void) n;
#endif
			RC ret = SUCCESS;

			size_t i = lower_bound;
			const size_t end = upper_bound;

			// some sanity checks
			assert( i <= end );
			assert( end <= n );

			// assume current i needs to be processed, forward until we find an index
			// for which the mask evaluates true
			bool process_current_i = true;
			if( masked && i < end ) {
				process_current_i = utils::interpretMask< descr >(
					already_dense_mask
						? internal::getCoordinates( mask ).assigned( i )
						: local_mask.assigned( i - lower_bound ),
					internal::getRaw( mask ), i ) && (
						already_dense_input_to_fold || local_to_fold.assigned( i - lower_bound )
					);
				// if not
				while( !process_current_i ) {
					// forward to next element
					(void) ++i;
					// check that we are within bounds
					if( i == end ) {
						break;
					}
					// evaluate whether we should process this i-th element
					process_current_i = utils::interpretMask< descr >(
						already_dense_mask
							? internal::getCoordinates( mask ).assigned( i )
							: local_mask.assigned( i - lower_bound ),
						internal::getRaw( mask ), i ) && (
							already_dense_input_to_fold || local_to_fold.assigned( i - lower_bound )
						);
				}
			}

			if( !masked && i < end ) {
				process_current_i = local_to_fold.assigned( i - lower_bound );
				while( !process_current_i ) {
					(void) ++i;
					if( i == end ) {
						break;
					}
					process_current_i = already_dense_input_to_fold ||
						local_to_fold.assigned( i - lower_bound );
				}
			}

#ifndef NDEBUG
			if( i < end ) {
				assert( i < n );
			}
#endif

			// declare thread-local variable and set our variable to the first value in
			// our block
			typename Monoid::D3 local =
				monoid.template getIdentity< typename Monoid::D3 >();
			if( end > 0 ) {
				if( i < end ) {
#ifdef _DEBUG
					std::cout << "\t processing start index " << i << "\n";
#endif
					local = static_cast< typename Monoid::D3 >(
						internal::getRaw( to_fold )[ i ] );
				}
			}

			// if we have more values to fold
			if( i + 1 < end ) {

				// keep going until we run out of values to fold
				while( true ) {

					// forward to next variable
					(void) ++i;

					// forward more (possibly) if in the masked case
					if( masked && i < end ) {
						assert( i < n );
						process_current_i = utils::interpretMask< descr >(
								already_dense_mask
									? internal::getCoordinates( mask ).assigned( i )
									: local_mask.assigned( i - lower_bound ),
								internal::getRaw( mask ), i
							) && (
								already_dense_input_to_fold ||
								local_to_fold.assigned( i - lower_bound )
							);
						while( !process_current_i ) {
							(void) ++i;
							if( i == end ) {
								break;
							}
							assert( i < end );
							assert( i < n );
							process_current_i = utils::interpretMask< descr >(
									already_dense_mask
										? internal::getCoordinates( mask ).assigned( i )
										: local_mask.assigned( i - lower_bound ),
									internal::getRaw( mask ), i
								) && (
									already_dense_input_to_fold ||
									local_to_fold.assigned( i - lower_bound )
								);
						}
					}
					if( !masked && i < end ) {
						assert( i < n );
						process_current_i = already_dense_input_to_fold ||
							local_to_fold.assigned( i - lower_bound );
						while( !process_current_i ) {
							(void) ++i;
							if( i == end ) {
								break;
							}
							assert( i < end );
							assert( i < n );
							process_current_i = already_dense_input_to_fold ||
								local_to_fold.assigned( i - lower_bound );
						}
					}

					// stop if past end
					if( i >= end ) {
						break;
					}

#ifdef _DEBUG
					std::cout << "\t processing index " << i << "\n";
#endif

					// do fold
					assert( i < n );
					if( left ) {
						ret = ret ? ret : foldl< descr >( local, internal::getRaw( to_fold )[ i ],
							monoid.getOperator() );
					} else {
						ret = ret ? ret : foldr< descr >( internal::getRaw( to_fold )[ i ], local,
							monoid.getOperator() );
					}
					assert( ret == SUCCESS );

					if( ret != SUCCESS ) {
						break;
					}
				}
			}

			if( left ) {
				ret = ret ? ret : foldl< descr >( thread_local_output, local,
					monoid.getOperator() );
			} else {
				ret = ret ? ret : foldr< descr >( local, thread_local_output,
					monoid.getOperator() );
			}
			assert( ret == SUCCESS );

			return ret;
		}

		/**
		 * Dispatches to any of the four above variants depending on asymptotic cost
		 * analysis.
		 */
		template<
			Descriptor descr = descriptors::no_operation,
			bool masked,
			bool left, // if this is false, assumes right-looking fold
			class Monoid,
			typename IOType,
			typename InputType,
			typename MaskType,
			typename Coords
		>
		RC fold_from_vector_to_scalar_generic(
			IOType &fold_into,
			const Vector< InputType, nonblocking, Coords > &to_fold,
			const Vector< MaskType, nonblocking, Coords > &mask,
			const Monoid &monoid
		) {
			// static sanity checks
			static_assert( grb::is_monoid< Monoid >::value,
				"grb::foldl can only be called using monoids. This "
				"function should not have been called-- please submit a "
				"bugreport." );

			const size_t n = internal::getCoordinates( to_fold ).size();

			// mask must be of equal size as input vector
			if( masked && n != size( mask ) ) {
				return MISMATCH;
			}

			// handle trivial cases
			if( n == 0 ) {
				return SUCCESS;
			}

			// some globals used during the folding
			RC ret = SUCCESS;
			typename Monoid::D3 global =
				monoid.template getIdentity< typename Monoid::D3 >();

			size_t local_reduced_size = NONBLOCKING::numThreads() *
				config::CACHE_LINE_SIZE::value();
			IOType local_reduced[ local_reduced_size ];

			for(
				size_t i = 0;
				i < local_reduced_size;
				i += config::CACHE_LINE_SIZE::value()
			) {
				local_reduced[ i ] = monoid.template getIdentity< typename Monoid::D3 >();
			}

			constexpr const bool dense_descr = descr & descriptors::dense;

			internal::Pipeline::stage_type func =
				[&to_fold, &mask, &monoid, &local_reduced] (
					internal::Pipeline &pipeline,
					const size_t lower_bound,
					const size_t upper_bound
				) {
#ifdef _NONBLOCKING_DEBUG
					#pragma omp critical
					std::cout << "\t\tExecution of stage fold_from_vector_to_scalar_generic "
						"in the range(" << lower_bound << ", " << upper_bound << ")" << std::endl;
#endif
					RC ret = SUCCESS;

					Coords local_to_fold, local_mask;
					size_t local_n = upper_bound - lower_bound;
					size_t local_to_fold_nz = local_n;
					size_t local_mask_nz = local_n;

#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
					const bool already_dense_vectors = dense_descr ||
						pipeline.allAlreadyDenseVectors();
#else
					(void) pipeline;
					constexpr const bool already_dense_vectors = dense_descr;
#endif

					bool already_dense_input_to_fold = true;
					bool already_dense_mask = true;

					if( !already_dense_vectors ) {
#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
						already_dense_input_to_fold = pipeline.containsAlreadyDenseVector(
							&internal::getCoordinates( to_fold ) );
						if( !already_dense_input_to_fold ) {
#else
							already_dense_input_to_fold = false;
#endif
							local_to_fold = internal::getCoordinates( to_fold ).asyncSubset(
								lower_bound, upper_bound );
							local_to_fold_nz = local_to_fold.nonzeroes();
#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
						}
#endif
						if( masked ) {
#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
							already_dense_mask = pipeline.containsAlreadyDenseVector(
								&internal::getCoordinates( mask ) );
							if( !already_dense_mask ) {
#else
								already_dense_mask = false;
#endif
								local_mask = internal::getCoordinates( mask ).asyncSubset( lower_bound,
									upper_bound );
								local_mask_nz = local_mask.nonzeroes();
#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
							}
#endif
						}
					}

					unsigned int thread_id = omp_get_thread_num() *
						config::CACHE_LINE_SIZE::value();

					// dispatch, dense variant
					if( ( (descr & descriptors::dense) || local_to_fold_nz == local_n ) && (
							!masked || (
								(descr & descriptors::structural) &&
								!(descr & descriptors::invert_mask) &&
								local_mask_nz == local_n
							)
						)
					) {
#ifdef _DEBUG
						std::cout << "\t dispatching to dense variant\n";
#endif
						ret = fold_from_vector_to_scalar_dense< left >(
							local_reduced[ thread_id ], lower_bound, upper_bound, to_fold, monoid );
					} else if( masked && (descr & descriptors::invert_mask ) ) {
						// in this case we are forced to dispatch to O(n)
#ifdef _DEBUG
						std::cout << "\t forced dispatch to O(n) sparse variant\n";
#endif

#ifdef GRB_BOOLEAN_DISPATCHER
						ret = boolean_dispatcher_fold_from_vector_to_scalar_fullLoopSparse<
#else
						ret = fold_from_vector_to_scalar_fullLoopSparse<
#endif
								descr, true, left
							>(
								already_dense_input_to_fold, already_dense_mask,
								local_reduced[ thread_id ], lower_bound, upper_bound,
								local_to_fold, local_mask, to_fold, mask, monoid
							);
					} else {
						constexpr const size_t threeWs =
							sizeof( typename Coords::StackType ) +
							sizeof( typename Coords::ArrayType ) +
							MaskWordSize< descr, MaskType >::value;
						const size_t fullLoop = masked
							? 2 * sizeof( typename Coords::ArrayType ) * local_n +
								sizeof( MaskType ) * local_mask_nz
							: sizeof( typename Coords::ArrayType ) * local_n;
						const size_t vectorLoop = masked
							? threeWs * local_to_fold_nz
							: sizeof( typename Coords::StackType ) * local_to_fold_nz;
						const size_t maskLoop = masked
							? threeWs * local_mask_nz
							: std::numeric_limits< size_t >::max();
						if( fullLoop >= vectorLoop && maskLoop >= vectorLoop ) {
#ifdef _DEBUG
							std::cout << "\t dispatching to vector-driven sparse variant\n";
#endif

#ifdef GRB_BOOLEAN_DISPATCHER
							ret = boolean_dispatcher_fold_from_vector_to_scalar_vectorDriven<
#else
							ret = fold_from_vector_to_scalar_vectorDriven<
#endif
									descr, masked, left
								>(
									already_dense_input_to_fold, already_dense_mask,
									local_reduced[ thread_id ], lower_bound, upper_bound,
									local_to_fold, local_mask, to_fold, mask, monoid
								);
						} else if( vectorLoop >= fullLoop && maskLoop >= fullLoop ) {
#ifdef _DEBUG
							std::cout << "\t dispatching to O(n) sparse variant\n";
#endif

#ifdef GRB_BOOLEAN_DISPATCHER
							ret = boolean_dispatcher_fold_from_vector_to_scalar_fullLoopSparse<
#else
							ret = fold_from_vector_to_scalar_fullLoopSparse<
#endif
									descr, masked, left
								>(
									already_dense_input_to_fold, already_dense_mask,
									local_reduced[ thread_id ], lower_bound, upper_bound,
									local_to_fold, local_mask, to_fold, mask, monoid
								);
						} else {
							assert( maskLoop < fullLoop && maskLoop < vectorLoop );
							assert( masked );
#ifdef _DEBUG
							std::cout << "\t dispatching to mask-driven sparse variant\n";
#endif

#ifdef GRB_BOOLEAN_DISPATCHER
							ret = boolean_dispatcher_fold_from_vector_to_scalar_maskDriven<
#else
							ret = fold_from_vector_to_scalar_maskDriven<
#endif
									descr, left
								>(
									already_dense_input_to_fold, already_dense_mask,
									local_reduced[ thread_id ], lower_bound, upper_bound,
									local_to_fold, local_mask, to_fold, mask, monoid
								);
						}
					}

					return ret;
				};

#ifdef _NONBLOCKING_DEBUG
			std::cout << "\t\tStage added to a pipeline: "
				<< "fold_from_vector_to_scalar_generic" << std::endl;
#endif

			ret = ret ? ret : internal::le.addStage(
					std::move( func ),
					internal::Opcode::BLAS1_FOLD_VECTOR_SCALAR_GENERIC,
					n,
					sizeof( IOType ),
					dense_descr,
					true,
					nullptr, nullptr, nullptr, nullptr,
					&to_fold,
					( masked ) ? &mask : nullptr,
					nullptr,
					nullptr,
					&internal::getCoordinates( to_fold ),
					(masked) ? &internal::getCoordinates( mask ) : nullptr,
					nullptr,
					nullptr,
					nullptr
				);

			if( ret == SUCCESS ) {
				for(
					size_t i = 0;
					i < local_reduced_size;
					i += config::CACHE_LINE_SIZE::value()
				) {
					RC rc;
					if( left ) {
						rc = foldl< descr >( global, local_reduced[ i ], monoid.getOperator() );
					} else {
						rc = foldr< descr >( local_reduced[ i ], global, monoid.getOperator() );
					}
					assert( rc == SUCCESS );
					if( rc != SUCCESS ) {
						ret = rc;
					}
				}
			}

			// accumulate
#ifdef _DEBUG
			std::cout << "\t accumulating " << global << " into " << fold_into << "\n";
#endif

			if( ret == SUCCESS ) {
				if( left ) {
					ret = foldl< descr >( fold_into, global, monoid.getOperator() );
				} else {
					ret = foldr< descr >( global, fold_into, monoid.getOperator() );
				}
			}

			return ret;
		}

		/**
		 * \internal
		 * @tparam left   If false, right-looking fold is assumed (and left-looking
		 *                otherwise)
		 * @tparam sparse Whether \a vector was sparse
		 * @tparam monoid Whether \a op is actually a monoid
		 * \endinternal
		 */
		template<
			Descriptor descr,
			bool left,
			bool sparse,
			bool masked,
			bool monoid,
#ifdef GRB_BOOLEAN_DISPATCHER
			bool already_dense_output,
			bool already_dense_mask,
#endif
			typename MaskType,
			typename IOType,
			typename InputType,
			typename Coords,
			class OP
		>
		RC fold_from_scalar_to_vector_generic(
#ifndef GRB_BOOLEAN_DISPATCHER
			bool already_dense_output,
			bool already_dense_mask,
#endif
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
			constexpr const bool dense_descr = descr & descriptors::dense;
			assert( !masked || mask != nullptr );
			assert( !masked || local_mask_ptr != nullptr );

			Coords local_mask;
			if( masked ) {
				local_mask = *local_mask_ptr;
			}

			const size_t local_n = upper_bound - lower_bound;
			const size_t local_vector_nz = (sparse || !already_dense_output)
				? local_vector.nonzeroes() : local_n;
			const size_t local_mask_nz = ( masked )
				? ( ( already_dense_mask )
						? local_n
						: local_mask.nonzeroes()
					)
				: 0;

			const size_t n = internal::getCoordinates( vector ).size();

			if( masked && internal::getCoordinates( *mask ).size() != n ) {
				return MISMATCH;
			}
			if( dense_descr && sparse ) {
				return ILLEGAL;
			}
			if( n == 0 ) {
				return SUCCESS;
			}
			if( phase == RESIZE ) {
				return SUCCESS;
			}

			assert( phase == EXECUTE );
			IOType * __restrict__ const x = internal::getRaw( vector );
			const MaskType * __restrict__ const m = ( masked )
				? internal::getRaw( *mask )
				: nullptr;

			if( sparse && monoid && !masked ) {
				for( size_t i = lower_bound; i < upper_bound; ++i ) {
					if( already_dense_output || local_vector.assigned( i - lower_bound ) ) {
						if( left ) {
							(void) foldl< descr >( x[ i ], scalar, op );
						} else {
							(void) foldr< descr >( scalar, x[ i ], op );
						}
					} else {
						x[ i ] = static_cast< IOType >( scalar );
					}
				}

				if( !already_dense_output ) {
					local_vector.local_assignAllNotAlreadyAssigned();
				}
			} else if( sparse && monoid && masked ) {
				for( size_t i = 0; i < local_mask_nz; ++i ) {
					const size_t index = ( ( already_dense_mask )
						? i
						: local_mask.index( i ) ) + lower_bound;
					if( already_dense_mask ) {
						if( !internal::getCoordinates( *mask ).template mask< descr >(
							index, m )
						) {
							continue;
						}
					} else {
						if( !local_mask.template mask< descr >( index - lower_bound,
							m + lower_bound )
						) {
							continue;
						}
					}
					if( already_dense_output || local_vector.assign( index - lower_bound ) ) {
						if( left ) {
							(void) foldl< descr >( x[ index ], scalar, op );
						} else {
							(void) foldr< descr >( scalar, x[ index ], op );
						}
					} else {
						x[ index ] = static_cast< IOType >( scalar );
					}
				}
			} else if( sparse && !monoid ) {
				const bool maskDriven = masked ? local_mask_nz < local_vector_nz : false;
				if( maskDriven ) {
					for( size_t i = 0; i < local_mask_nz; ++i ) {
						const size_t index = ( ( already_dense_mask )
							? i
							: local_mask.index( i ) ) + lower_bound;
						if( already_dense_mask ) {
							if( !internal::getCoordinates( *mask ).template mask< descr >(
								index, m )
							) {
								continue;
							}
						} else {
							if( !local_mask.template mask< descr >( index - lower_bound,
								m + lower_bound )
							) {
								continue;
							}
						}
						if( already_dense_output || local_vector.assign( index - lower_bound ) ) {
							if( left ) {
								(void) foldl< descr >( x[ index ], scalar, op );
							} else {
								(void) foldr< descr >( scalar, x[ index ], op );
							}
						}
					}
				} else {
					for( size_t i = 0; i < local_vector_nz; ++i ) {
						const size_t index = (already_dense_output
								? i
								: local_vector.index( i )
							) + lower_bound;
						if( masked ) {
							if( already_dense_mask ) {
								if( !( internal::getCoordinates( *mask ).template mask< descr >(
									index, m ) )
								) {
									continue;
								}
							} else {
								if( !local_mask.template mask< descr >( index - lower_bound, m +
									lower_bound )
								) {
									continue;
								}
							}
						}
						if( left ) {
							(void) foldl< descr >( x[ index ], scalar, op );
						} else {
							(void) foldr< descr >( scalar, x[ index ], op );
						}
					}
				}
			} else if( !sparse && masked ) {
				for( size_t i = 0; i < local_mask_nz; ++i ) {
					const size_t index = ( ( already_dense_mask )
						? i
						: local_mask.index( i ) ) + lower_bound;
					if( already_dense_mask ) {
						if( !( internal::getCoordinates( *mask ).template mask< descr >(
							index, m ) )
						) {
							continue;
						}
					} else {
						if( !local_mask.template mask< descr >( index - lower_bound, m +
							lower_bound )
						) {
							continue;
						}
					}

					if( left ) {
						(void) foldl< descr >( x[ index ], scalar, op );
					} else {
						(void) foldr< descr >( scalar, x[ index ], op );
					}
				}
			} else {
				// if target vector is dense and there is no mask, then
				// there is no difference between monoid or non-monoid behaviour.
				assert( !sparse );
				assert( !masked );
				assert( local_vector_nz == local_n );

				if( local_n > 0 ) {
					if( left ) {
						op.eWiseFoldlAS( x + lower_bound, scalar, local_n );
					} else {
						op.eWiseFoldrSA( scalar, x + lower_bound, local_n );
					}
				}
			}

			return SUCCESS;
		}

		template<
			Descriptor descr,
			bool left, // if this is false, the right-looking fold is assumed
			bool sparse,
			bool masked,
			bool monoid,
#ifdef GRB_BOOLEAN_DISPATCHER
			bool already_dense_output,
			bool already_dense_input_to_fold,
			bool already_dense_mask,
#endif
			typename MaskType,
			typename IOType,
			typename IType,
			typename Coords,
			class OP
		>
		RC fold_from_vector_to_vector_generic(
#ifndef GRB_BOOLEAN_DISPATCHER
			bool already_dense_output,
			bool already_dense_input_to_fold,
			bool already_dense_mask,
#endif
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
			constexpr const bool dense_descr = descr & descriptors::dense;
			assert( !masked || (m != nullptr) );

			Coords local_m;
			if( masked && !already_dense_mask ) {
				local_m = *local_m_ptr;
			}

			const size_t local_n = upper_bound - lower_bound;
			const size_t local_fold_into_nz = already_dense_output
				? local_n
				: local_fold_into.nonzeroes();
			const size_t local_to_fold_nz = already_dense_input_to_fold
				? local_n
				: local_to_fold.nonzeroes();
			const size_t local_m_nz = ( masked )
				? ( already_dense_mask
						? local_n
						: local_m.nonzeroes()
					)
				: 0;

			const size_t n = size( fold_into );
			if( n != size( to_fold ) ) {
				return MISMATCH;
			}
			if( masked && size( *m ) != n ) {
				return MISMATCH;
			}
			if( dense_descr && sparse ) {
				return ILLEGAL;
			}
			if( phase == RESIZE ) {
				return SUCCESS;
			}

			assert( phase == EXECUTE );

			if( !sparse && !masked ) {
#ifdef _DEBUG
				std::cout << "fold_from_vector_to_vector_generic: in dense variant\n";
#endif

#ifdef _DEBUG
				std::cout << "fold_from_vector_to_vector_generic: in sequential variant\n";
#endif
				if( left ) {
					op.eWiseFoldlAA( internal::getRaw( fold_into ) + lower_bound,
						internal::getRaw( to_fold ) + lower_bound, local_n );
				} else {
					op.eWiseFoldrAA( internal::getRaw( to_fold ) + lower_bound,
						internal::getRaw( fold_into ) + lower_bound, local_n );
				}
			} else {
#ifdef _DEBUG
				std::cout << "fold_from_vector_to_vector_generic: in sparse variant\n";
				std::cout << "\tfolding vector of " << local_to_fold_nz << " nonzeroes "
					<< "into a vector of " << local_fold_into_nz << " nonzeroes...\n";
#endif
				if(
					masked &&
					local_fold_into_nz == local_n &&
					local_to_fold_nz == local_n
				) {
					// use sparsity structure of mask for this eWiseFold
					if( left ) {
#ifdef _DEBUG
						std::cout << "fold_from_vector_to_vector_generic: foldl, using the "
							<< "mask's sparsity structure\n";
#endif
						for( size_t k = 0; k < local_m_nz; ++k ) {
							const size_t i = ( already_dense_mask
									? k
									: local_m.index( k )
								) + lower_bound;
#ifdef _DEBUG
							std::cout << "Left-folding " << to_fold[ i ] << " into "
								<< fold_into[ i ];
#endif
							(void) foldl< descr >( fold_into[ i ], to_fold[ i ], op );
#ifdef _DEBUG
							std::cout << " resulting into " << fold_into[ i ] << "\n";
#endif
						}
					} else {
#ifdef _DEBUG
						std::cout << "fold_from_vector_to_vector_generic: foldl, using the "
							<< "mask's sparsity structure\n";
#endif
						for( size_t k = 0; k < local_m_nz; ++k ) {
							const size_t i = ( already_dense_mask
									? k
									: local_m.index( k )
								) + lower_bound;
#ifdef _DEBUG
							std::cout << "Right-folding " << to_fold[ i ] << " into "
								<< fold_into[ i ];
#endif
							(void) foldr< descr >( to_fold[ i ], fold_into[ i ], op );
#ifdef _DEBUG
							std::cout << " resulting into " << fold_into[ i ] << "\n";
#endif
						}
					}
				} else if( !masked && local_fold_into_nz == local_n ) {
					// use sparsity structure of to_fold for this eWiseFold
					if( left ) {
#ifdef _DEBUG
						std::cout << "fold_from_vector_to_vector_generic: foldl, using "
							<< "to_fold's sparsity\n";
#endif
						for( size_t k = 0; k < local_to_fold_nz; ++k ) {
							const size_t i = ( already_dense_input_to_fold
									? k
									: local_to_fold.index( k )
								) + lower_bound;
#ifdef _DEBUG
								std::cout << "Left-folding " << to_fold[ i ] << " into "
									<< fold_into[ i ];
#endif
								(void) foldl< descr >( fold_into[ i ], to_fold[ i ], op );
#ifdef _DEBUG
								std::cout << " resulting into " << fold_into[ i ] << "\n";
#endif
						}
					} else {
#ifdef _DEBUG
						std::cout << "fold_from_vector_to_vector_generic: foldl, using "
							<< "to_fold's sparsity\n";
#endif
						for( size_t k = 0; k < local_to_fold_nz; ++k ) {
							const size_t i = ( already_dense_input_to_fold
									? k
									: local_to_fold.index( k )
								) + lower_bound;
#ifdef _DEBUG
							std::cout << "Right-folding " << to_fold[ i ] << " into "
								<< fold_into[ i ];
#endif
							(void) foldr< descr >( to_fold[ i ], fold_into[ i ], op );
#ifdef _DEBUG
							std::cout << " resulting into " << fold_into[ i ] << "\n";
#endif
						}
					}
				} else if( !masked && local_to_fold_nz == local_n ) {
					// use sparsity structure of fold_into for this eWiseFold
					if( left ) {
#ifdef _DEBUG
						std::cout << "fold_from_vector_to_vector_generic: foldl, using "
							<< "fold_into's sparsity\n";
#endif
						for( size_t k = 0; k < local_fold_into_nz; ++k ) {
							const size_t i = ( already_dense_output
									? k
									: local_fold_into.index( k )
								) + lower_bound;
#ifdef _DEBUG
							std::cout << "Left-folding " << to_fold[ i ] << " into "
								<< fold_into[ i ];
#endif
							(void) foldl< descr >( fold_into[ i ], to_fold[ i ], op );
#ifdef _DEBUG
							std::cout << " resulting into " << fold_into[ i ] << "\n";
#endif
						}
					} else {
#ifdef _DEBUG
						std::cout << "fold_from_vector_to_vector_generic: foldr, using "
							<< "fold_into's sparsity\n";
#endif
						for( size_t k = 0; k < local_fold_into_nz; ++k ) {
							const size_t i = ( already_dense_output ?
									k :
									local_fold_into.index( k )
								) + lower_bound;
#ifdef _DEBUG
							std::cout << "Right-folding " << to_fold[ i ] << " into " << fold_into[ i ];
#endif
							(void) foldr< descr >( to_fold[ i ], fold_into[ i ], op );
#ifdef _DEBUG
							std::cout << " resulting into " << fold_into[ i ] << "\n";
#endif
						}
					}
				} else {
#ifdef _DEBUG
					std::cout << "fold_from_vector_to_vector_generic: using specialised "
						<< "code to merge two sparse vectors and, potentially, "
						<< "output masks\n";
#endif
					const IType * __restrict__ const tf_raw = internal::getRaw( to_fold );
					IOType * __restrict__ const fi_raw = internal::getRaw( fold_into );
#ifdef _DEBUG
					std::cout << "\tin sequential version...\n";
#endif
					for( size_t k = 0; k < local_to_fold_nz; ++k ) {
						const size_t i = ( already_dense_input_to_fold
								? k
								: local_to_fold.index( k )
							) + lower_bound;
						if( masked ) {
							if( already_dense_mask ) {
								if( !internal::getCoordinates( *m ).template mask< descr >( i,
									internal::getRaw( *m ) )
								) {
									continue;
								}
							} else {
								if( !local_m.template mask< descr >( i - lower_bound,
									internal::getRaw( *m ) + lower_bound )
								) {
									continue;
								}
							}
						}

						assert( i < n );
						if( already_dense_output ||
							local_fold_into.assigned( i - lower_bound )
						) {
							if( left ) {
#ifdef _DEBUG
								std::cout << "\tfoldl< descr >( fi_raw[ i ], tf_raw[ i ], op ), i = "
									<< i << ": " << tf_raw[ i ] << " goes into " << fi_raw[ i ];
#endif
								(void)foldl< descr >( fi_raw[ i ], tf_raw[ i ], op );
#ifdef _DEBUG
								std::cout << " which results in " << fi_raw[ i ] << "\n";
#endif
							} else {
#ifdef _DEBUG
								std::cout << "\tfoldr< descr >( tf_raw[ i ], fi_raw[ i ], op ), i = "
									<< i << ": " << tf_raw[ i ] << " goes into " << fi_raw[ i ];
#endif
								(void) foldr< descr >( tf_raw[ i ], fi_raw[ i ], op );
#ifdef _DEBUG
								std::cout << " which results in " << fi_raw[ i ] << "\n";
#endif
							}
						} else if( monoid ) {
#ifdef _DEBUG
							std::cout << "\tindex " << i << " is unset. Old value " << fi_raw[ i ]
								<< " will be overwritten with " << tf_raw[ i ] << "\n";
#endif
							fi_raw[ i ] = tf_raw[ i ];
							(void) local_fold_into.assign( i - lower_bound );
						}
					}
				}
			}

#ifdef _DEBUG
			std::cout << "\tCall to fold_from_vector_to_vector_generic done. "
				<< "Output now contains " << local_fold_into_nz << " / "
				<< local_n << " nonzeroes.\n";
#endif
			return SUCCESS;
		}

	} // namespace internal

	template<
		Descriptor descr = descriptors::no_operation,
		class Monoid,
		typename InputType,
		typename IOType,
		typename MaskType,
		typename Coords
	>
	RC foldr(
		const Vector< InputType, nonblocking, Coords > &x,
		const Vector< MaskType, nonblocking, Coords > &mask,
		IOType &beta,
		const Monoid &monoid = Monoid(),
		const typename std::enable_if<
				!grb::is_object< InputType >::value &&
				!grb::is_object< IOType >::value &&
				!grb::is_object< MaskType >::value &&
				grb::is_monoid< Monoid >::value,
			void
		>::type * const = nullptr
	) {
		// static sanity checks
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< IOType, InputType >::value ), "grb::foldr",
			"called with a scalar IO type that does not match the input vector type" );
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< InputType, typename Monoid::D1 >::value ), "grb::foldr",
			"called with an input vector value type that does not match the first "
			"domain of the given monoid" );
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< InputType, typename Monoid::D2 >::value ), "grb::foldr",
			"called with an input vector type that does not match the second domain of "
			"the given monoid" );
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< InputType, typename Monoid::D3 >::value ), "grb::foldr",
			"called with an input vector type that does not match the third domain of "
			"the given monoid" );
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< bool, MaskType >::value ), "grb::foldr",
			"called with a vector mask type that is not boolean" );

		if( size( mask ) > 0 ) {
			return internal::template fold_from_vector_to_scalar_generic<
					descr, true, false
				>( beta, x, mask, monoid );
		} else {
			return internal::template fold_from_vector_to_scalar_generic<
					descr, false, false
				>( beta, x, mask, monoid );
		}
	}

	template<
		Descriptor descr = descriptors::no_operation,
		class Monoid,
		typename InputType,
		typename IOType,
		typename Coords
	>
	RC foldr(
		const Vector< InputType, nonblocking, Coords > &x,
		IOType &beta,
		const Monoid &monoid = Monoid(),
		const typename std::enable_if<
				!grb::is_object< InputType >::value &&
				!grb::is_object< IOType >::value &&
				grb::is_monoid< Monoid >::value,
			void
		>::type * const = nullptr
	) {
		// static sanity checks
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< IOType, InputType >::value ), "grb::foldr",
			"called with a scalar IO type that does not match the input vector type" );
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< InputType, typename Monoid::D1 >::value ), "grb::foldr",
			"called with an input vector value type that does not match the first "
			"domain of the given monoid" );
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< InputType, typename Monoid::D2 >::value ), "grb::foldr",
			"called with an input vector type that does not match the second domain of "
			"the given monoid" );
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< InputType, typename Monoid::D3 >::value ), "grb::foldr",
			"called with an input vector type that does not match the third domain of "
			"the given monoid" );

		Vector< bool, nonblocking, Coords > empty_mask( 0 );
		return internal::template fold_from_vector_to_scalar_generic<
				descr, false, false
			>( beta, x, empty_mask, monoid );
	}

	template<
		Descriptor descr = descriptors::no_operation,
		class Monoid,
		typename IOType,
		typename InputType,
		typename Coords
	>
	RC foldr(
		const InputType &alpha,
		Vector< IOType, nonblocking, Coords > &y,
		const Monoid &monoid = Monoid(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if< !grb::is_object< InputType >::value &&
				!grb::is_object< IOType >::value &&
				grb::is_monoid< Monoid >::value,
			void
		>::type * const = nullptr
	) {
		// static sanity checks
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< typename Monoid::D1, IOType >::value ), "grb::foldl",
			"called with a vector x of a type that does not match the first domain "
			"of the given operator" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< typename Monoid::D2, InputType >::value ), "grb::foldl",
			"called on a vector y of a type that does not match the second domain "
			"of the given operator" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< typename Monoid::D3, IOType >::value ), "grb::foldl",
			"called on a vector x of a type that does not match the third domain "
			"of the given operator" );

		RC ret = SUCCESS;

		constexpr const bool dense_descr = descr & descriptors::dense;

		internal::Pipeline::stage_type func =
			[alpha, &y, &monoid, phase] (
				internal::Pipeline &pipeline,
				const size_t lower_bound, const size_t upper_bound
			) {
#ifdef _NONBLOCKING_DEBUG
				#pragma omp critical
				std::cout << "\t\tExecution of stage foldl(alpha, y, monoid) in the range(" << lower_bound << ", " << upper_bound << ")" << std::endl;
#endif
				RC rc = SUCCESS;

				const Vector< bool, nonblocking, Coords > * const null_mask = nullptr;
				const Coords * const local_null_mask = nullptr;

				Coords local_y;
				size_t local_y_nz;
				bool sparse = false;

#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
				const bool already_dense_vectors = dense_descr || pipeline.allAlreadyDenseVectors();
#else
				(void) pipeline;
				constexpr const bool already_dense_vectors = dense_descr;
#endif

				bool already_dense_output = true;

				if( !already_dense_vectors ) {
					const size_t local_n = upper_bound - lower_bound;
#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
					already_dense_output = pipeline.containsAlreadyDenseVector( &internal::getCoordinates( y ) );
					if( !already_dense_output ) {
#else
						already_dense_output = false;
#endif
						local_y = internal::getCoordinates( y ).asyncSubset( lower_bound, upper_bound );
						local_y_nz = local_y.nonzeroes();
						if( local_y_nz < local_n ) {
							sparse = true;
						}
#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
					}
#endif
				}

				if( sparse ) {
#ifdef GRB_BOOLEAN_DISPATCHER
					rc = internal::boolean_dispatcher_fold_from_scalar_to_vector_generic<
#else
					rc = internal::fold_from_scalar_to_vector_generic<
#endif
							descr, false, true, false, true
						>(
							already_dense_output, true,
							lower_bound, upper_bound, local_y, local_null_mask,
							y, null_mask, alpha, monoid.getOperator(), phase
						);
				} else {
#ifdef GRB_BOOLEAN_DISPATCHER
					rc = internal::boolean_dispatcher_fold_from_scalar_to_vector_generic<
#else
					rc = internal::fold_from_scalar_to_vector_generic<
#endif
							descr, false, false, false, true
						>(
							already_dense_output, true,
							lower_bound, upper_bound, local_y, local_null_mask,
							y, null_mask, alpha, monoid.getOperator(), phase
						);
				}

#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
				if( !already_dense_output ) {
#else
				if( !already_dense_vectors ) {
#endif
					internal::getCoordinates( y ).asyncJoinSubset( local_y, lower_bound,
						upper_bound );
			}

			return rc;
		};

		ret = ret ? ret : internal::le.addStage(
				std::move( func ),
				internal::Opcode::BLAS1_FOLD_SCALAR_VECTOR_GENERIC,
				internal::getCoordinates( y ).size(),
				sizeof( IOType ),
				dense_descr, true,
				&y, nullptr,
				&internal::getCoordinates( y ), nullptr,
				nullptr, nullptr, nullptr, nullptr,
				nullptr, nullptr, nullptr, nullptr,
				nullptr
			);

#ifdef _NONBLOCKING_DEBUG
		std::cout << "\t\tStage added to a pipeline: foldl(alpha, y, monoid)"
			<< std::endl;
#endif
		return ret;
	}

	template<
		Descriptor descr = descriptors::no_operation,
		class OP,
		typename IOType,
		typename InputType,
		typename Coords
	>
	RC foldr(
		const InputType &alpha,
		Vector< IOType, nonblocking, Coords > &y,
		const OP &op = OP(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
				!grb::is_object< InputType >::value &&
				!grb::is_object< IOType >::value &&
				grb::is_operator< OP >::value,
			void
		>::type * const = nullptr
	) {
		// static sanity checks
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< typename OP::D1, IOType >::value ), "grb::foldl",
			"called with a vector x of a type that does not match the first domain "
			"of the given operator" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< typename OP::D2, InputType >::value ), "grb::foldl",
			"called on a vector y of a type that does not match the second domain "
			"of the given operator" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< typename OP::D3, IOType >::value ), "grb::foldl",
			"called on a vector x of a type that does not match the third domain "
			"of the given operator" );

		RC ret = SUCCESS;

		constexpr const bool dense_descr = descr & descriptors::dense;

		internal::Pipeline::stage_type func =
			[alpha, &y, &op, phase] (
				internal::Pipeline &pipeline,
				const size_t lower_bound, const size_t upper_bound
			) {
#ifdef _NONBLOCKING_DEBUG
				#pragma omp critical
				{
					std::cout << "\t\tExecution of stage foldl(alpha, y, op) in the range("
						<< lower_bound << ", " << upper_bound << ")" << std::endl;
				}
#endif
				RC rc = SUCCESS;

				const Vector< bool, nonblocking, Coords > * const null_mask = nullptr;
				const Coords * const local_null_mask = nullptr;

				Coords local_y;
				const size_t local_n = upper_bound - lower_bound;
				size_t local_y_nz;
				bool sparse = false;

#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
				const bool already_dense_vectors = dense_descr ||
					pipeline.allAlreadyDenseVectors();
#else
				(void) pipeline;
				constexpr const bool already_dense_vectors = dense_descr;
#endif
				bool already_dense_output = true;

				if( !already_dense_vectors ) {
#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
					already_dense_output = pipeline.containsAlreadyDenseVector(
						&internal::getCoordinates( y ) );
					if( !already_dense_output ) {
#else
						already_dense_output = false;
#endif
						local_y = internal::getCoordinates( y ).asyncSubset( lower_bound,
							upper_bound );
						local_y_nz = local_y.nonzeroes();
						if( local_y_nz < local_n ) {
							sparse = true;
						}
#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
					}
#endif
				}

				if( sparse ) {
#ifdef GRB_BOOLEAN_DISPATCHER
					rc = internal::boolean_dispatcher_fold_from_scalar_to_vector_generic<
#else
					rc = internal::fold_from_scalar_to_vector_generic<
#endif
							descr, false, true, false, false
						>(
							already_dense_output, true,
							lower_bound, upper_bound,
							local_y, local_null_mask, y, null_mask,
							alpha, op, phase
						);
				} else {
#ifdef GRB_BOOLEAN_DISPATCHER
					rc = internal::boolean_dispatcher_fold_from_scalar_to_vector_generic<
#else
					rc = internal::fold_from_scalar_to_vector_generic<
#endif
							descr, false, false, false, false
						>(
							already_dense_output, true,
							lower_bound, upper_bound, local_y, local_null_mask,
							y, null_mask, alpha, op, phase
						);
				}

#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
				if( !already_dense_output ) {
#else
				if( !already_dense_vectors ) {
#endif
					internal::getCoordinates( y ).asyncJoinSubset( local_y, lower_bound,
						upper_bound );
				}

				return rc;
			};

		ret = ret ? ret : internal::le.addStage(
			std::move( func ),
			internal::Opcode::BLAS1_FOLD_SCALAR_VECTOR_GENERIC,
			internal::getCoordinates( y ).size(),
			sizeof( IOType ),
			dense_descr, true,
			&y, nullptr,
			&internal::getCoordinates( y ), nullptr,
			nullptr, nullptr, nullptr, nullptr,
			nullptr, nullptr, nullptr, nullptr,
			nullptr
		);

#ifdef _NONBLOCKING_DEBUG
		std::cout << "\t\tStage added to a pipeline: foldl(alpha, y, op)"
			<< std::endl;
#endif
		return ret;
	}

	template<
		Descriptor descr = descriptors::no_operation,
		class OP,
		typename IOType,
		typename InputType,
		typename Coords
	>
	RC foldr(
		const Vector< InputType, nonblocking, Coords > &x,
		Vector< IOType, nonblocking, Coords > &y,
		const OP &op = OP(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
				grb::is_operator< OP >::value &&
				!grb::is_object< InputType >::value &&
				!grb::is_object< IOType >::value,
			void
		>::type * = nullptr
	) {
		// static sanity checks
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< typename OP::D1, InputType >::value ), "grb::eWiseFoldr",
			"called with a vector x of a type that does not match the first domain "
			"of the given operator" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< typename OP::D2, IOType >::value ), "grb::eWiseFoldr",
			"called on a vector y of a type that does not match the second domain "
			"of the given operator" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< typename OP::D3, IOType >::value ), "grb::eWiseFoldr",
			"called on a vector y of a type that does not match the third domain "
			"of the given operator" );

		const size_t n = size( x );
		if( n != size( y ) ) {
			return MISMATCH;
		}

#ifdef _DEBUG
		std::cout << "In foldr ([T]<-[T])\n";
#endif

		RC ret = SUCCESS;

		constexpr const bool dense_descr = descr & descriptors::dense;

		internal::Pipeline::stage_type func =
			[&x, &y, &op, phase] (
				internal::Pipeline &pipeline,
				const size_t lower_bound, const size_t upper_bound
			) {
#ifdef _NONBLOCKING_DEBUG
				#pragma omp critical
				std::cout << "\t\tExecution of stage foldr(x, y, operator) in the range("
					<< lower_bound << ", " << upper_bound << ")" << std::endl;
#endif
				RC rc = SUCCESS;

				const Vector< bool, nonblocking, Coords > * const null_mask = nullptr;
				const Coords * const local_null_mask = nullptr;

				Coords local_x, local_y;
				const size_t local_n = upper_bound - lower_bound;
				size_t local_x_nz, local_y_nz;
				bool sparse = false;

#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
				const bool already_dense_vectors = dense_descr ||
					pipeline.allAlreadyDenseVectors();
#else
				(void) pipeline;
				constexpr const bool already_dense_vectors = dense_descr;
#endif
				bool already_dense_output = true;
				bool already_dense_input = true;

				if( !already_dense_vectors ) {
#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
					already_dense_output = pipeline.containsAlreadyDenseVector(
						&internal::getCoordinates( y ) );
					if( !already_dense_output ) {
#else
						already_dense_output = false;
#endif
						local_y = internal::getCoordinates( y ).asyncSubset( lower_bound,
							upper_bound );
						local_y_nz = local_y.nonzeroes();
						if( local_y_nz < local_n ) {
							sparse = true;
						}
#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
					}

					already_dense_input = pipeline.containsAlreadyDenseVector(
						&internal::getCoordinates( x ) );
					if( !already_dense_input ) {
#else
						already_dense_input = false;
#endif
						local_x = internal::getCoordinates( x ).asyncSubset( lower_bound,
							upper_bound );
						local_x_nz = local_x.nonzeroes();
						if( local_x_nz < local_n ) {
							sparse = true;
						}
#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
					}
#endif
				}

				if( sparse ) {
#ifdef GRB_BOOLEAN_DISPATCHER
					rc = internal::boolean_dispatcher_fold_from_vector_to_vector_generic<
#else
					rc = internal::fold_from_vector_to_vector_generic<
#endif
							descr, false, true, false, false
						>(
							already_dense_output, already_dense_input, true,
							lower_bound, upper_bound,
							local_y, local_null_mask,
							local_x, y,
							null_mask, x,
							op, phase
						);
				} else {
#ifdef GRB_BOOLEAN_DISPATCHER
					rc = internal::boolean_dispatcher_fold_from_vector_to_vector_generic<
#else
					rc = internal::fold_from_vector_to_vector_generic<
#endif
							descr, false, false, false, false
						>(
							already_dense_output, already_dense_input, true,
							lower_bound, upper_bound,
							local_y, local_null_mask,
							local_x,
							y, null_mask,
							x,
							op, phase
						);
				}

#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
				if( !already_dense_output ) {
#else
				if( !already_dense_vectors ) {
#endif
					internal::getCoordinates( y ).asyncJoinSubset( local_y, lower_bound,
						upper_bound );
				}

				return rc;
			};

		ret = ret ? ret : internal::le.addStage(
				std::move( func ),
				internal::Opcode::BLAS1_FOLD_VECTOR_VECTOR_GENERIC,
				n, sizeof( IOType ),
				dense_descr, true,
				&y, nullptr,
				&internal::getCoordinates( y ), nullptr,
				&x, nullptr, nullptr, nullptr,
				&internal::getCoordinates( x ), nullptr, nullptr, nullptr,
				nullptr
			);

#ifdef _NONBLOCKING_DEBUG
		std::cout << "\t\tStage added to a pipeline: foldr(x, y, operator)"
			<< std::endl;
#endif
		return ret;
	}

	template<
		Descriptor descr = descriptors::no_operation,
		class OP,
		typename IOType,
		typename MaskType,
		typename InputType,
		typename Coords
	>
	RC foldr(
		const Vector< InputType, nonblocking, Coords > &x,
		const Vector< MaskType, nonblocking, Coords > &m,
		Vector< IOType, nonblocking, Coords > &y,
		const OP &op = OP(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if< grb::is_operator< OP >::value &&
			!grb::is_object< InputType >::value &&
			!grb::is_object< MaskType >::value &&
			!grb::is_object< IOType >::value, void
		>::type * = nullptr
	) {
		// static sanity checks
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< typename OP::D1, InputType >::value ), "grb::eWiseFoldr",
			"called with a vector x of a type that does not match the first domain "
			"of the given operator" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< typename OP::D2, IOType >::value ), "grb::eWiseFoldr",
			"called on a vector y of a type that does not match the second domain "
			"of the given operator" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< typename OP::D3, IOType >::value ), "grb::eWiseFoldr",
			"called on a vector y of a type that does not match the third domain "
			"of the given operator" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< bool, MaskType >::value ), "grb::eWiseFoldr",
			"called with a non-Boolean mask" );

		if( size( m ) == 0 ) {
			return foldr< descr >( x, y, op, phase );
		}

		const size_t n = size( x );
		if( n != size( y ) || n != size( m ) ) {
			return MISMATCH;
		}

		RC ret = SUCCESS;

		constexpr const bool dense_descr = descr & descriptors::dense;

		internal::Pipeline::stage_type func =
			[&x, &m, &y, &op, phase] (
				internal::Pipeline &pipeline,
				const size_t lower_bound, const size_t upper_bound
			) {
#ifdef _NONBLOCKING_DEBUG
				#pragma omp critical
				std::cout << "\t\tExecution of stage foldr(x, m, y, operator) in the range(" << lower_bound << ", " << upper_bound << ")" << std::endl;
#endif
				RC rc = SUCCESS;

				Coords local_x, local_m, local_y;
				const size_t local_n = upper_bound - lower_bound;
				size_t local_x_nz, local_y_nz;
				bool sparse = false;

#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
				const bool already_dense_vectors = dense_descr ||
					pipeline.allAlreadyDenseVectors();
#else
				constexpr const bool already_dense_vectors = dense_descr;
#endif
				bool already_dense_output = true;
				bool already_dense_mask = true;
				bool already_dense_input = true;

				if( !already_dense_vectors ) {
#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
					already_dense_output = pipeline.containsAlreadyDenseVector(
						&internal::getCoordinates( y ) );
					if( !already_dense_output ) {
#else
						already_dense_output = false;
#endif
						local_y = internal::getCoordinates( y ).asyncSubset( lower_bound,
							upper_bound );
						local_y_nz = local_y.nonzeroes();
						if( local_y_nz < local_n ) {
							sparse = true;
						}
#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
					}

					already_dense_mask = pipeline.containsAlreadyDenseVector(
						&internal::getCoordinates( m ) );
					if( !already_dense_mask ) {
#else
						already_dense_mask = false;
#endif
						local_m = internal::getCoordinates( m ).asyncSubset( lower_bound,
							upper_bound );
#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
					}

					already_dense_input = pipeline.containsAlreadyDenseVector(
						&internal::getCoordinates( x ) );
					if( !already_dense_input ) {
#else
						already_dense_input = false;
#endif
						local_x = internal::getCoordinates( x ).asyncSubset( lower_bound,
							upper_bound );
						local_x_nz = local_x.nonzeroes();
						if( local_x_nz < local_n ) {
							sparse = true;
						}
#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
					}
#endif
				}

				if( sparse ) {
#ifdef GRB_BOOLEAN_DISPATCHER
					rc = internal::boolean_dispatcher_fold_from_vector_to_vector_generic<
#else
					rc = internal::fold_from_vector_to_vector_generic<
#endif
							descr, false, true, true, false
						>(
							already_dense_output, already_dense_input, already_dense_mask,
							lower_bound, upper_bound,
							local_y, &local_m, local_x,
							y, &m, x,
							op, phase
						);
				} else {
#ifdef GRB_BOOLEAN_DISPATCHER
					rc = internal::boolean_dispatcher_fold_from_vector_to_vector_generic<
#else
					rc = internal::fold_from_vector_to_vector_generic<
#endif
							descr, false, false, true, false
						>(
							already_dense_output, already_dense_input, already_dense_mask,
							lower_bound, upper_bound,
							local_y, &local_m, local_x,
							y, &m, x,
							op, phase
						);
				}

#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
				if( !already_dense_output ) {
#else
				if( !already_dense_vectors ) {
#endif
					internal::getCoordinates( y ).asyncJoinSubset( local_y, lower_bound,
						upper_bound );
				}

				return rc;
			};

		ret = ret ? ret : internal::le.addStage(
				std::move( func ),
				internal::Opcode::BLAS1_FOLD_MASKED_VECTOR_VECTOR_GENERIC,
				n, sizeof( IOType ),
				dense_descr, true,
				&y, nullptr, &internal::getCoordinates( y ), nullptr,
				&x, &m, nullptr, nullptr,
				&internal::getCoordinates( x ), &internal::getCoordinates( m ),
				nullptr, nullptr,
				nullptr
			);

#ifdef _NONBLOCKING_DEBUG
		std::cout << "\t\tStage added to a pipeline: foldr(x, m, y, operator)"
			<< std::endl;
#endif
		return ret;
	}

	template<
		Descriptor descr = descriptors::no_operation,
		class Monoid,
		typename IOType,
		typename InputType,
		typename Coords
	>
	RC foldr(
		const Vector< InputType, nonblocking, Coords > &x,
		Vector< IOType, nonblocking, Coords > &y,
		const Monoid &monoid = Monoid(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if< grb::is_monoid< Monoid >::value &&
				!grb::is_object< InputType >::value &&
				!grb::is_object< IOType >::value,
			void
		>::type * = nullptr
	) {
		// static sanity checks
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< typename Monoid::D1, InputType >::value ), "grb::eWiseFoldr",
			"called with a vector x of a type that does not match the first domain "
			"of the given monoid" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< typename Monoid::D2, IOType >::value ), "grb::eWiseFoldr",
			"called on a vector y of a type that does not match the second domain "
			"of the given monoid" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< typename Monoid::D3, IOType >::value ), "grb::eWiseFoldr",
			"called on a vector y of a type that does not match the third domain "
			"of the given monoid" );

		// dynamic sanity checks
		const size_t n = size( x );
		if( n != size( y ) ) {
			return MISMATCH;
		}

		RC ret = SUCCESS;

		constexpr const bool dense_descr = descr & descriptors::dense;

		internal::Pipeline::stage_type func =
			[&x, &y, &monoid, phase] (
				internal::Pipeline &pipeline,
				const size_t lower_bound, const size_t upper_bound
			) {
#ifdef _NONBLOCKING_DEBUG
				#pragma omp critical
				std::cout << "\t\tExecution of stage foldr(x, y, monoid) in the range("
					<< lower_bound << ", " << upper_bound << ")" << std::endl;
#endif
				RC rc = SUCCESS;

				const Vector< bool, nonblocking, Coords > * const null_mask = nullptr;
				const Coords * const local_null_mask = nullptr;

				Coords local_x, local_y;
				const size_t local_n = upper_bound - lower_bound;
				size_t local_x_nz, local_y_nz;
				bool sparse = false;

#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
				const bool already_dense_vectors = dense_descr ||
					pipeline.allAlreadyDenseVectors();
#else
				constexpr const bool already_dense_vectors = dense_descr;
#endif
				bool already_dense_output = true;
				bool already_dense_input = true;

				if( !already_dense_vectors ) {
#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
					already_dense_output = pipeline.containsAlreadyDenseVector(
						&internal::getCoordinates( y ) );
					if( !already_dense_output ) {
#else
						already_dense_output = false;
#endif
						local_y = internal::getCoordinates( y ).asyncSubset( lower_bound,
							upper_bound );
						local_y_nz = local_y.nonzeroes();
						if( local_y_nz < local_n ) {
							sparse = true;
						}
#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
					}

					already_dense_input = pipeline.containsAlreadyDenseVector(
						&internal::getCoordinates( x ) );
					if( !already_dense_input ) {
#else
						already_dense_input = false;
#endif
						local_x = internal::getCoordinates( x ).asyncSubset( lower_bound,
							upper_bound );
						local_x_nz = local_x.nonzeroes();
						if( local_x_nz < local_n ) {
							sparse = true;
						}
					}
#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
				}
#endif
				if( sparse ) {
#ifdef GRB_BOOLEAN_DISPATCHER
					rc = internal::boolean_dispatcher_fold_from_vector_to_vector_generic<
#else
					rc = internal::fold_from_vector_to_vector_generic<
#endif
							descr, false, true, false, true
						>(
							already_dense_output, already_dense_input, true,
							lower_bound, upper_bound,
							local_y, local_null_mask, local_x,
							y, null_mask, x,
							monoid.getOperator(), phase
						);
				} else {
#ifdef GRB_BOOLEAN_DISPATCHER
					rc = internal::boolean_dispatcher_fold_from_vector_to_vector_generic<
#else
					rc = internal::fold_from_vector_to_vector_generic<
#endif
							descr, false, false, false, true
						>(
							already_dense_output, already_dense_input, true,
							lower_bound, upper_bound,
							local_y, local_null_mask, local_x,
							y, null_mask, x,
							monoid.getOperator(), phase
						);
				}

#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
			if( !already_dense_output ) {
#else
			if( !already_dense_vectors ) {
#endif
				internal::getCoordinates( y ).asyncJoinSubset( local_y, lower_bound,
					upper_bound );
			}

			return rc;
		};

		ret = ret ? ret : internal::le.addStage(
				std::move( func ),
				internal::Opcode::BLAS1_FOLD_VECTOR_VECTOR_GENERIC,
				n, sizeof( IOType ), dense_descr, true,
				&y, nullptr, &internal::getCoordinates( y ), nullptr,
				&x, nullptr, nullptr, nullptr,
				&internal::getCoordinates( x ), nullptr, nullptr, nullptr,
				nullptr
			);

#ifdef _NONBLOCKING_DEBUG
		std::cout << "\t\tStage added to a pipeline: foldr(x, y, monoid)"
			<< std::endl;
#endif
		return ret;
	}

	template<
		Descriptor descr = descriptors::no_operation,
		class Monoid,
		typename IOType,
		typename MaskType,
		typename InputType,
		typename Coords
	>
	RC foldr(
		const Vector< InputType, nonblocking, Coords > &x,
		const Vector< MaskType, nonblocking, Coords > &m,
		Vector< IOType, nonblocking, Coords > &y,
		const Monoid &monoid = Monoid(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if< grb::is_monoid< Monoid >::value &&
			!grb::is_object< MaskType >::value &&
			!grb::is_object< InputType >::value &&
			!grb::is_object< IOType >::value, void
		>::type * = nullptr
	) {
		// static sanity checks
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< typename Monoid::D1, InputType >::value ), "grb::eWiseFoldr",
			"called with a vector x of a type that does not match the first domain "
			"of the given monoid" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< typename Monoid::D2, IOType >::value ), "grb::eWiseFoldr",
			"called on a vector y of a type that does not match the second domain "
			"of the given monoid" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< typename Monoid::D3, IOType >::value ), "grb::eWiseFoldr",
			"called on a vector y of a type that does not match the third domain "
			"of the given monoid" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< bool, MaskType >::value ), "grb::eWiseFoldr",
			"called with a mask of non-Boolean type" );

		// check empty mask
		if( size( m ) == 0 ) {
			return foldr< descr >( x, y, monoid, phase );
		}

		// dynamic sanity checks
		const size_t n = size( x );
		if( n != size( y ) || n != size( m ) ) {
			return MISMATCH;
		}

		RC ret = SUCCESS;

		constexpr const bool dense_descr = descr & descriptors::dense;

		internal::Pipeline::stage_type func =
			[&x, &m, &y, &monoid, phase] (
				internal::Pipeline &pipeline,
				const size_t lower_bound, const size_t upper_bound
			) {
#ifdef _NONBLOCKING_DEBUG
				#pragma omp critical
				std::cout << "\t\tExecution of stage foldr(x, m, y, monoid) in the range("
					<< lower_bound << ", " << upper_bound << ")" << std::endl;
#endif
				RC rc = SUCCESS;

				Coords local_x, local_m, local_y;
				const size_t local_n = upper_bound - lower_bound;
				size_t local_x_nz, local_y_nz;
				bool sparse = false;

#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
				const bool already_dense_vectors = dense_descr ||
					pipeline.allAlreadyDenseVectors();
#else
				constexpr const bool already_dense_vectors = dense_descr;
#endif

				bool already_dense_output = true;
				bool already_dense_mask = true;
				bool already_dense_input = true;

				if( !already_dense_vectors ) {
#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
					already_dense_output = pipeline.containsAlreadyDenseVector(
						&internal::getCoordinates( y ) );
					if( !already_dense_output ) {
#else
						already_dense_output = false;
#endif
						local_y = internal::getCoordinates( y ).asyncSubset( lower_bound,
							upper_bound );
						local_y_nz = local_y.nonzeroes();
						if( local_y_nz < local_n ) {
							sparse = true;
						}
#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
					}

					already_dense_mask = pipeline.containsAlreadyDenseVector(
						&internal::getCoordinates( m ) );
					if( !already_dense_mask ) {
#else
						already_dense_mask = false;
#endif
						local_m = internal::getCoordinates( m ).asyncSubset( lower_bound,
							upper_bound );
#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
					}

					already_dense_input = pipeline.containsAlreadyDenseVector(
						&internal::getCoordinates( x ) );
					if( !already_dense_input ) {
#else
						already_dense_input = false;
#endif
						local_x = internal::getCoordinates( x ).asyncSubset( lower_bound,
							upper_bound );
						local_x_nz = local_x.nonzeroes();
						if( local_x_nz < local_n ) {
							sparse = true;
						}
#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
					}
#endif
				}

				if( sparse ) {
#ifdef GRB_BOOLEAN_DISPATCHER
					rc = internal::boolean_dispatcher_fold_from_vector_to_vector_generic<
#else
					rc = internal::fold_from_vector_to_vector_generic<
#endif
							descr, false, true, true, true
						>(
							already_dense_output, already_dense_input, already_dense_mask,
							lower_bound, upper_bound,
							local_y, &local_m, local_x,
							y, &m, x,
							monoid.getOperator(), phase
						);
				} else {
#ifdef GRB_BOOLEAN_DISPATCHER
					rc = internal::boolean_dispatcher_fold_from_vector_to_vector_generic<
#else
					rc = internal::fold_from_vector_to_vector_generic<
#endif
							descr, false, false, true, true
						>(
							already_dense_output, already_dense_input, already_dense_mask,
							lower_bound, upper_bound,
							local_y, &local_m, local_x,
							y, &m, x,
							monoid.getOperator(), phase
						);
				}

#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
				if( !already_dense_output ) {
#else
				if( !already_dense_vectors ) {
#endif
					internal::getCoordinates( y ).asyncJoinSubset( local_y, lower_bound,
						upper_bound );
				}

				return rc;
			};

		ret = ret ? ret : internal::le.addStage(
				std::move( func ),
				internal::Opcode::BLAS1_FOLD_MASKED_VECTOR_VECTOR_GENERIC,
				n, sizeof( IOType ), dense_descr, true,
				&y, nullptr, &internal::getCoordinates( y ), nullptr,
				&x, &m, nullptr, nullptr,
				&internal::getCoordinates( x ), &internal::getCoordinates( m ),
				nullptr, nullptr,
				nullptr
			);

#ifdef _NONBLOCKING_DEBUG
			std::cout << "\t\tStage added to a pipeline: foldr(x, m, y, monoid)"
				<< std::endl;
#endif
		return ret;
	}

	template<
		Descriptor descr = descriptors::no_operation,
		class Op,
		typename IOType,
		typename InputType,
		typename Coords
	>
	RC foldl(
		Vector< IOType, nonblocking, Coords > &x,
		const InputType beta,
		const Op &op = Op(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if< !grb::is_object< IOType >::value &&
			!grb::is_object< InputType >::value &&
			grb::is_operator< Op >::value, void
		>::type * = nullptr
	) {
		// static sanity checks
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Op::D1, IOType >::value ),
			"grb::foldl",
			"called with a vector x of a type that does not match the first domain "
			"of the given operator" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Op::D2, InputType >::value ),
			"grb::foldl",
			"called on a vector y of a type that does not match the second domain "
			"of the given operator" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Op::D3, IOType >::value ),
			"grb::foldl",
			"called on a vector x of a type that does not match the third domain "
			"of the given operator" );

		RC ret = SUCCESS;

		constexpr const bool dense_descr = descr & descriptors::dense;

		internal::Pipeline::stage_type func =
			[&x, beta, &op, phase] (
				internal::Pipeline &pipeline,
				const size_t lower_bound, const size_t upper_bound
			) {
#ifdef _NONBLOCKING_DEBUG
				#pragma omp critical
				std::cout << "\t\tExecution of stage foldl(x, beta, op) in the range("
					<< lower_bound << ", " << upper_bound << ")" << std::endl;
#endif
				RC rc = SUCCESS;

				const Vector< bool, nonblocking, Coords > * const null_mask = nullptr;
				const Coords * const local_null_mask = nullptr;

				Coords local_x;
				const size_t local_n = upper_bound - lower_bound;
				size_t local_x_nz = local_n;
				bool sparse = false;

#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
				const bool already_dense_vectors = dense_descr ||
					pipeline.allAlreadyDenseVectors();
#else
				constexpr const bool already_dense_vectors = dense_descr;
#endif

				bool already_dense_output = true;

				if( !already_dense_vectors ) {
#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
					already_dense_output = pipeline.containsAlreadyDenseVector(
						&internal::getCoordinates( x ) );
					if( !already_dense_output ) {
#else
						already_dense_output = false;
#endif
						local_x = internal::getCoordinates( x ).asyncSubset( lower_bound,
							upper_bound );
						local_x_nz = local_x.nonzeroes();
						if( local_x_nz < local_n ) {
							sparse = true;
						}
#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
					}
#endif
				}

				if( sparse ) {
#ifdef GRB_BOOLEAN_DISPATCHER
					rc = internal::boolean_dispatcher_fold_from_scalar_to_vector_generic<
#else
					rc = internal::fold_from_scalar_to_vector_generic<
#endif
							descr, true, true, false, false
						>(
							already_dense_output, true,
							lower_bound, upper_bound,
							local_x, local_null_mask,
							x, null_mask,
							beta,
							op, phase
						);
				} else {
#ifdef GRB_BOOLEAN_DISPATCHER
					rc = internal::boolean_dispatcher_fold_from_scalar_to_vector_generic<
#else
				rc = internal::fold_from_scalar_to_vector_generic<
#endif
						descr, true, false, false, false
					>(
						already_dense_output, true,
						lower_bound, upper_bound,
						local_x, local_null_mask,
						x, null_mask, beta,
						op, phase
					);
				}

#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
				if( !already_dense_output ) {
#else
				if( !already_dense_vectors ) {
#endif
					internal::getCoordinates( x ).asyncJoinSubset( local_x, lower_bound, upper_bound );
				}

				return rc;
			};

		ret = ret ? ret : internal::le.addStage(
				std::move( func ),
				internal::Opcode::BLAS1_FOLD_SCALAR_VECTOR_GENERIC,
				internal::getCoordinates( x ).size(), sizeof( IOType ),
				dense_descr, true,
				&x, nullptr,
				&internal::getCoordinates( x ), nullptr,
				nullptr, nullptr, nullptr, nullptr,
				nullptr, nullptr, nullptr, nullptr,
				nullptr
			);

#ifdef _NONBLOCKING_DEBUG
		std::cout << "\t\tStage added to a pipeline: foldl(x, beta, op)" << std::endl;
#endif
		return ret;
	}

	template<
		Descriptor descr = descriptors::no_operation,
		class Op,
		typename IOType,
		typename MaskType,
		typename InputType,
		typename Coords
	>
	RC foldl(
		Vector< IOType, nonblocking, Coords > &x,
		const Vector< MaskType, nonblocking, Coords > &m,
		const InputType beta,
		const Op &op = Op(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if< !grb::is_object< IOType >::value &&
			!grb::is_object< MaskType >::value &&
			!grb::is_object< InputType >::value &&
			grb::is_operator< Op >::value, void
		>::type * = nullptr
	) {
		// static sanity checks
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Op::D1, IOType >::value ),
			"grb::foldl",
			"called with a vector x of a type that does not match the first domain "
			"of the given operator" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Op::D2, InputType >::value ),
			"grb::foldl",
			"called on a vector y of a type that does not match the second domain "
			"of the given operator" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Op::D3, IOType >::value ),
			"grb::foldl",
			"called on a vector x of a type that does not match the third domain "
			"of the given operator" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting ) ||
				std::is_same< bool, MaskType >::value ),
			"grb::foldl (reference, vector <- scalar, masked)",
			"provided mask does not have boolean entries" );

		// check empty mask
		if( size( m ) == 0 ) {
			return foldl< descr >( x, beta, op, phase );
		}

		// dynamic checks
		const size_t n = size( x );
		if( size( m ) != n ) {
			return MISMATCH;
		}

		// catch trivial phase
		if( phase == RESIZE ) {
			return SUCCESS;
		}

		RC ret = SUCCESS;

		constexpr const bool dense_descr = descr & descriptors::dense;

		internal::Pipeline::stage_type func =
			[&x, &m, beta, &op, phase] (
				internal::Pipeline &pipeline,
				const size_t lower_bound, const size_t upper_bound
			) {
#ifdef _NONBLOCKING_DEBUG
				#pragma omp critical
				std::cout << "\t\tExecution of stage foldl(x, m, beta, op) in the range("
					<< lower_bound << ", " << upper_bound << ")" << std::endl;
#endif
				RC rc = SUCCESS;

				Coords local_x, local_mask;
				const size_t local_n = upper_bound - lower_bound;
				size_t local_x_nz = local_n;
				bool sparse = false;

#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
				const bool already_dense_vectors = dense_descr ||
					pipeline.allAlreadyDenseVectors();
#else
				constexpr const bool already_dense_vectors = dense_descr;
#endif

				bool already_dense_output = true;
				bool already_dense_mask = true;

				if( !already_dense_vectors ) {
#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
					already_dense_output = pipeline.containsAlreadyDenseVector(
						&internal::getCoordinates( x ) );
					if( !already_dense_output ) {
#else
						already_dense_output = false;
#endif
						local_x = internal::getCoordinates( x ).asyncSubset( lower_bound,
							upper_bound );
						local_x_nz = local_x.nonzeroes();
						if( local_x_nz < local_n ) {
							sparse = true;
						}
#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
					}

					already_dense_mask = pipeline.containsAlreadyDenseVector(
						&internal::getCoordinates( m ) );
					if( !already_dense_mask ) {
#else
						already_dense_mask = false;
#endif
						local_mask = internal::getCoordinates( m ).asyncSubset( lower_bound,
							upper_bound );
#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
					}
#endif
				}

				if( sparse ) {
#ifdef GRB_BOOLEAN_DISPATCHER
					rc = internal::boolean_dispatcher_fold_from_scalar_to_vector_generic<
#else
					rc = internal::fold_from_scalar_to_vector_generic<
#endif
							descr, true, true, true, false
						>(
							already_dense_output, already_dense_mask,
							lower_bound, upper_bound,
							local_x, &local_mask,
							x, &m,
							beta,
							op, phase
						);
				} else {
#ifdef GRB_BOOLEAN_DISPATCHER
					rc = internal::boolean_dispatcher_fold_from_scalar_to_vector_generic<
#else
					rc = internal::fold_from_scalar_to_vector_generic<
#endif
							descr, true, false, true, false
						>(
							already_dense_output, already_dense_mask,
							lower_bound, upper_bound,
							local_x, &local_mask,
							x, &m,
							beta,
							op, phase
						);
				}

#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
				if( !already_dense_output ) {
#else
				if( !already_dense_vectors ) {
#endif
					internal::getCoordinates( x ).asyncJoinSubset( local_x, lower_bound,
						upper_bound );
				}

				return rc;
			};

		ret = ret ? ret : internal::le.addStage(
				std::move( func ),
				internal::Opcode::BLAS1_FOLD_MASKED_SCALAR_VECTOR_GENERIC,
				n, sizeof( IOType ),
				dense_descr, true,
				&x, nullptr, &internal::getCoordinates( x ), nullptr,
				&m, nullptr, nullptr, nullptr,
				&internal::getCoordinates( m ), nullptr, nullptr, nullptr,
				nullptr
			);

#ifdef _NONBLOCKING_DEBUG
		std::cout << "\t\tStage added to a pipeline: foldl(x, m, beta, op)"
			<< std::endl;
#endif
		return ret;
	}

	template<
		Descriptor descr = descriptors::no_operation,
		class Monoid,
		typename IOType,
		typename InputType,
		typename Coords
	>
	RC foldl(
		Vector< IOType, nonblocking, Coords > &x,
		const InputType beta,
		const Monoid &monoid = Monoid(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if< !grb::is_object< IOType >::value &&
			!grb::is_object< InputType >::value &&
			grb::is_monoid< Monoid >::value, void
		>::type * = nullptr
	) {
		// static sanity checks
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< typename Monoid::D1, IOType >::value ), "grb::foldl",
			"called with a vector x of a type that does not match the first domain "
			"of the given monoid" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< typename Monoid::D2, InputType >::value ), "grb::foldl",
			"called on a vector y of a type that does not match the second domain "
			"of the given monoid" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< typename Monoid::D3, IOType >::value ), "grb::foldl",
			"called on a vector x of a type that does not match the third domain "
			"of the given monoid" );

		RC ret = SUCCESS;

		constexpr const bool dense_descr = descr & descriptors::dense;

		internal::Pipeline::stage_type func =
			[&x, beta, &monoid, phase] (
				internal::Pipeline &pipeline,
				const size_t lower_bound, const size_t upper_bound
			) {
#ifdef _NONBLOCKING_DEBUG
				#pragma omp critical
				std::cout << "\t\tExecution of stage foldl(x, beta, monoid) in the range("
					<< lower_bound << ", " << upper_bound << ")" << std::endl;
#endif
				RC rc = SUCCESS;

				const Vector< bool, nonblocking, Coords > * const null_mask = nullptr;
				const Coords * const local_null_mask = nullptr;

				Coords local_x;
				const size_t local_n = upper_bound - lower_bound;
				size_t local_x_nz = local_n;
				bool sparse = false;

#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
				const bool already_dense_vectors = dense_descr ||
					pipeline.allAlreadyDenseVectors();
#else
				(void) pipeline;
				constexpr const bool already_dense_vectors = dense_descr;
#endif

				bool already_dense_output = true;

				if( !already_dense_vectors ) {
#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
					already_dense_output = pipeline.containsAlreadyDenseVector(
						&internal::getCoordinates( x ) );
					if( !already_dense_output ) {
#else
						already_dense_output = false;
#endif
						local_x = internal::getCoordinates( x ).asyncSubset( lower_bound,
							upper_bound );
						local_x_nz = local_x.nonzeroes();
						if( local_x_nz < local_n ) {
							sparse = true;
						}
#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
					}
#endif
				}

				if( sparse ) {
#ifdef GRB_BOOLEAN_DISPATCHER
					rc = internal::boolean_dispatcher_fold_from_scalar_to_vector_generic<
#else
					rc = internal::fold_from_scalar_to_vector_generic<
#endif
							descr, true, true, false, true
						>(
							already_dense_output, true,
							lower_bound, upper_bound,
							local_x, local_null_mask,
							x, null_mask,
							beta,
							monoid.getOperator(), phase
						);
				} else {
#ifdef GRB_BOOLEAN_DISPATCHER
					rc = internal::boolean_dispatcher_fold_from_scalar_to_vector_generic<
#else
					rc = internal::fold_from_scalar_to_vector_generic<
#endif
							descr, true, false, false, true
						>(
							already_dense_output, true,
							lower_bound, upper_bound,
							local_x, local_null_mask,
							x, null_mask,
							beta,
							monoid.getOperator(), phase
						);
				}

#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
				if( !already_dense_output ) {
#else
				if( !already_dense_vectors ) {
#endif
					internal::getCoordinates( x ).asyncJoinSubset( local_x, lower_bound,
						upper_bound );
				}

				return rc;
			};

		ret = ret ? ret : internal::le.addStage(
				std::move( func ),
				internal::Opcode::BLAS1_FOLD_SCALAR_VECTOR_GENERIC,
				internal::getCoordinates( x ).size(), sizeof( IOType ),
				dense_descr, true,
				&x, nullptr, &internal::getCoordinates( x ), nullptr,
				nullptr, nullptr, nullptr, nullptr,
				nullptr, nullptr, nullptr, nullptr,
				nullptr
			);

#ifdef _NONBLOCKING_DEBUG
		std::cout << "\t\tStage added to a pipeline: foldl(x, beta, monoid)"
			<< std::endl;
#endif
		return ret;
	}

	template<
		Descriptor descr = descriptors::no_operation,
		class Monoid,
		typename IOType,
		typename MaskType,
		typename InputType,
		typename Coords
	>
	RC foldl(
		Vector< IOType, nonblocking, Coords > &x,
		const Vector< MaskType, nonblocking, Coords > &m,
		const InputType &beta,
		const Monoid &monoid = Monoid(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
				!grb::is_object< IOType >::value &&
				!grb::is_object< MaskType >::value &&
				!grb::is_object< InputType >::value &&
				grb::is_monoid< Monoid >::value,
			void
		>::type * = nullptr
	) {
		// static sanity checks
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< typename Monoid::D1, IOType >::value ), "grb::foldl",
			"called with a vector x of a type that does not match the first domain "
			"of the given monoid" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< typename Monoid::D2, InputType >::value ), "grb::foldl",
			"called on a vector y of a type that does not match the second domain "
			"of the given monoid" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< typename Monoid::D3, IOType >::value ), "grb::foldl",
			"called on a vector x of a type that does not match the third domain "
			"of the given monoid" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< bool, MaskType >::value ),
			"grb::foldl (nonblocking, vector <- scalar, masked, monoid)",
			"provided mask does not have boolean entries" );

		// check for empty mask
		if( size( m ) == 0 ) {
			return foldl< descr >( x, beta, monoid, phase );
		}

		// dynamic checks
		const size_t n = size( x );
		if( n != size( m ) ) { return MISMATCH; }

		RC ret = SUCCESS;

		constexpr const bool dense_descr = descr & descriptors::dense;

		internal::Pipeline::stage_type func =
			[&x, &m, beta, &monoid, phase] (
				internal::Pipeline &pipeline,
				const size_t lower_bound, const size_t upper_bound
			) {
#ifdef _NONBLOCKING_DEBUG
				#pragma omp critical
				std::cout << "\t\tExecution of stage foldl(x, m, beta, monoid) in the "
					<< "range(" << lower_bound << ", " << upper_bound << ")" << std::endl;
#endif
				RC rc = SUCCESS;

				Coords local_x, local_m;
				const size_t local_n = upper_bound - lower_bound;
				size_t local_x_nz = local_n;
				bool sparse = false;

#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
				const bool already_dense_vectors = dense_descr ||
					pipeline.allAlreadyDenseVectors();
#else
				(void) pipeline;
				constexpr const bool already_dense_vectors = dense_descr;
#endif

				bool already_dense_output = true;
				bool already_dense_mask = true;

				if( !already_dense_vectors ) {
#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
					already_dense_output = pipeline.containsAlreadyDenseVector(
						&internal::getCoordinates( x ) );
					if( !already_dense_output ) {
#else
						already_dense_output = false;
#endif
						local_x = internal::getCoordinates( x ).asyncSubset( lower_bound,
							upper_bound );
						local_x_nz = local_x.nonzeroes();
						if( local_x_nz < local_n ) {
							sparse = true;
						}
#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
					}

					already_dense_mask = pipeline.containsAlreadyDenseVector(
						&internal::getCoordinates( m ) );
					if( !already_dense_mask ) {
#else
						already_dense_mask = false;
#endif
						local_m = internal::getCoordinates( m ).asyncSubset( lower_bound,
							upper_bound );
#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
					}
#endif
				}

				if( sparse ) {
#ifdef GRB_BOOLEAN_DISPATCHER
					rc = internal::boolean_dispatcher_fold_from_scalar_to_vector_generic<
#else
					rc = internal::fold_from_scalar_to_vector_generic<
#endif
							descr, true, true, true, true
						>(
							already_dense_output, already_dense_mask,
							lower_bound, upper_bound,
							local_x, &local_m,
							x, &m,
							beta,
							monoid.getOperator(), phase
						);
				} else {
#ifdef GRB_BOOLEAN_DISPATCHER
					rc = internal::boolean_dispatcher_fold_from_scalar_to_vector_generic<
#else
					rc = internal::fold_from_scalar_to_vector_generic<
#endif
							descr, true, false, true, true
						>(
							already_dense_output, already_dense_mask,
							lower_bound, upper_bound,
							local_x, &local_m,
							x, &m,
							beta,
							monoid.getOperator(), phase
						);
				}

#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
				if( !already_dense_output ) {
#else
				if( !already_dense_vectors ) {
#endif
					internal::getCoordinates( x ).asyncJoinSubset( local_x, lower_bound,
						upper_bound );
				}

				return rc;
			};

		ret = ret ? ret : internal::le.addStage(
				std::move( func ),
				internal::Opcode::BLAS1_FOLD_MASKED_SCALAR_VECTOR_GENERIC,
				internal::getCoordinates( x ).size(), sizeof( IOType ),
				dense_descr, true,
				&x, nullptr, &internal::getCoordinates( x ), nullptr,
				&m, nullptr, nullptr, nullptr,
				&internal::getCoordinates( m ), nullptr, nullptr, nullptr,
				nullptr
			);

#ifdef _NONBLOCKING_DEBUG
		std::cout << "\t\tStage added to a pipeline: foldl(x, m, beta, monoid)"
			<< std::endl;
#endif
		return ret;
	}

	template<
		Descriptor descr = descriptors::no_operation,
		class OP,
		typename IOType,
		typename InputType,
		typename Coords
	>
	RC foldl(
		Vector< IOType, nonblocking, Coords > &x,
		const Vector< InputType, nonblocking, Coords > &y,
		const OP &op = OP(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if< grb::is_operator< OP >::value &&
			!grb::is_object< IOType >::value &&
			!grb::is_object< InputType >::value, void
		>::type * = nullptr
	) {
		// static sanity checks
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< typename OP::D1, IOType >::value ), "grb::foldl",
			"called with a vector x of a type that does not match the first domain "
			"of the given operator" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< typename OP::D2, InputType >::value ), "grb::foldl",
			"called on a vector y of a type that does not match the second domain "
			"of the given operator" );
		NO_CAST_OP_ASSERT( (!( descr & descriptors::no_casting) ||
			std::is_same< typename OP::D3, IOType >::value ), "grb::foldl",
			"called on a vector x of a type that does not match the third domain "
			"of the given operator" );

		// dynamic sanity checks
		const size_t n = size( x );
		if( n != size( y ) ) {
			return MISMATCH;
		}

		RC ret = SUCCESS;

		constexpr const bool dense_descr = descr & descriptors::dense;

		internal::Pipeline::stage_type func =
			[&x, &y, &op, phase] (
				internal::Pipeline &pipeline,
				const size_t lower_bound, const size_t upper_bound
			) {
#ifdef _NONBLOCKING_DEBUG
				#pragma omp critical
				std::cout << "\t\tExecution of stage foldl(x, y, operator) in the range("
					<< lower_bound << ", " << upper_bound << ")" << std::endl;
#endif
				RC rc = SUCCESS;

				const Vector< bool, nonblocking, Coords > * const null_mask = nullptr;
				const Coords * const local_null_mask = nullptr;

				Coords local_x, local_y;
				const size_t local_n = upper_bound - lower_bound;
				size_t local_x_nz, local_y_nz;
				bool sparse = false;

#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
				const bool already_dense_vectors = dense_descr ||
					pipeline.allAlreadyDenseVectors();
#else
				(void) pipeline;
				constexpr const bool already_dense_vectors = dense_descr;
#endif

				bool already_dense_output = true;
				bool already_dense_input = true;

				if( !already_dense_vectors ) {
#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
					already_dense_output = pipeline.containsAlreadyDenseVector(
						&internal::getCoordinates( x ) );
					if( !already_dense_output ) {
#else
						already_dense_output = false;
#endif
						local_x = internal::getCoordinates( x ).asyncSubset( lower_bound,
							upper_bound );
						local_x_nz = local_x.nonzeroes();
						if( local_x_nz < local_n ) {
							sparse = true;
						}
#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
					}

					already_dense_input = pipeline.containsAlreadyDenseVector(
						&internal::getCoordinates( y ) );
					if( !already_dense_input ) {
#else
						already_dense_input = false;
#endif
						local_y = internal::getCoordinates( y ).asyncSubset( lower_bound,
							upper_bound );
						local_y_nz = local_y.nonzeroes();
						if( local_y_nz < local_n ) {
							sparse = true;
						}
#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
					}
#endif
				}

				if( sparse ) {
#ifdef GRB_BOOLEAN_DISPATCHER
					rc = internal::boolean_dispatcher_fold_from_vector_to_vector_generic<
#else
					rc = internal::fold_from_vector_to_vector_generic<
#endif
							descr, true, true, false, false
						>(
							already_dense_output, already_dense_input, true,
							lower_bound, upper_bound,
							local_x, local_null_mask, local_y,
							x, null_mask, y,
							op, phase
						);
				} else {
#ifdef GRB_BOOLEAN_DISPATCHER
					rc = internal::boolean_dispatcher_fold_from_vector_to_vector_generic<
#else
					rc = internal::fold_from_vector_to_vector_generic<
#endif
							descr, true, false, false, false
						>(
							already_dense_output, already_dense_input, true,
							lower_bound, upper_bound,
							local_x, local_null_mask, local_y,
							x, null_mask, y,
							op, phase
						);
				}

#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
				if( !already_dense_output ) {
#else
				if( !already_dense_vectors ) {
#endif
					internal::getCoordinates( x ).asyncJoinSubset( local_x, lower_bound,
						upper_bound );
				}

				return rc;
			};

		ret = ret ? ret : internal::le.addStage(
				std::move( func ),
				internal::Opcode::BLAS1_FOLD_VECTOR_VECTOR_GENERIC,
				n, sizeof( IOType ), dense_descr, true,
				&x, nullptr, &internal::getCoordinates( x ), nullptr,
				&y, nullptr, nullptr, nullptr,
				&internal::getCoordinates( y ), nullptr, nullptr, nullptr,
				nullptr
			);

#ifdef _NONBLOCKING_DEBUG
		std::cout << "\t\tStage added to a pipeline: foldl(x, y, operator)"
			<< std::endl;
#endif
		return ret;
	}

	template<
		Descriptor descr = descriptors::no_operation,
		class Monoid,
		typename IOType,
		typename InputType,
		typename Coords
	>
	RC foldl(
		Vector< IOType, nonblocking, Coords > &x,
		const Vector< InputType, nonblocking, Coords > &y,
		const Monoid &monoid = Monoid(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if< grb::is_monoid< Monoid >::value &&
			!grb::is_object< IOType >::value &&
			!grb::is_object< InputType >::value, void
		>::type * = nullptr
	) {
		// static sanity checks
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< typename Monoid::D1, IOType >::value ), "grb::foldl",
			"called with a vector x of a type that does not match the first domain "
			"of the given operator" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< typename Monoid::D2, InputType >::value ), "grb::foldl",
			"called on a vector y of a type that does not match the second domain "
			"of the given operator" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< typename Monoid::D3, IOType >::value ), "grb::foldl",
			"called on a vector x of a type that does not match the third domain "
			"of the given operator" );

		// dynamic sanity checks
		const size_t n = size( x );
		if( n != size( y ) ) {
			return MISMATCH;
		}

		RC ret = SUCCESS;

		constexpr const bool dense_descr = descr & descriptors::dense;

		internal::Pipeline::stage_type func =
			[&x, &y, &monoid, phase] (
				internal::Pipeline &pipeline,
				const size_t lower_bound, const size_t upper_bound
			) {
#ifdef _NONBLOCKING_DEBUG
				#pragma omp critical
				std::cout << "\t\tExecution of stage foldl(x, y, monoid) in the range("
					<< lower_bound << ", " << upper_bound << ")" << std::endl;
#endif
				RC rc = SUCCESS;

				const Vector< bool, nonblocking, Coords > * const null_mask = nullptr;
				const Coords * const local_null_mask = nullptr;

				Coords local_x, local_y;
				const size_t local_n = upper_bound - lower_bound;
				size_t local_x_nz, local_y_nz;
				bool sparse = false;

#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
				const bool already_dense_vectors = dense_descr ||
					pipeline.allAlreadyDenseVectors();
#else
				(void) pipeline;
				constexpr const bool already_dense_vectors = dense_descr;
#endif

				bool already_dense_output = true;
				bool already_dense_input = true;

				if( !already_dense_vectors ) {
#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
					already_dense_output = pipeline.containsAlreadyDenseVector(
						&internal::getCoordinates( x ) );
					if( !already_dense_output ) {
#else
						already_dense_output = false;
#endif
						local_x = internal::getCoordinates( x ).asyncSubset( lower_bound,
							upper_bound );
						local_x_nz = local_x.nonzeroes();
						if( local_x_nz < local_n ) {
							sparse = true;
						}
#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
					}

					already_dense_input = pipeline.containsAlreadyDenseVector(
						&internal::getCoordinates( y ) );
					if( !already_dense_input ) {
#else
						already_dense_input = false;
#endif
						local_y = internal::getCoordinates( y ).asyncSubset( lower_bound,
							upper_bound );
						local_y_nz = local_y.nonzeroes();
						if( local_y_nz < local_n ) {
							sparse = true;
						}
#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
					}
#endif
				}

				if( sparse ) {
#ifdef GRB_BOOLEAN_DISPATCHER
					rc = internal::boolean_dispatcher_fold_from_vector_to_vector_generic<
#else
					rc = internal::fold_from_vector_to_vector_generic<
#endif
							descr, true, true, false, true
						>(
							already_dense_output, already_dense_input, true,
							lower_bound, upper_bound,
							local_x, local_null_mask, local_y,
							x, null_mask, y,
							monoid.getOperator(), phase
						);
				} else {
#ifdef GRB_BOOLEAN_DISPATCHER
					rc = internal::boolean_dispatcher_fold_from_vector_to_vector_generic<
#else
					rc = internal::fold_from_vector_to_vector_generic<
#endif
							descr, true, false, false, true
						>(
							already_dense_output, already_dense_input, true,
							lower_bound, upper_bound,
							local_x, local_null_mask, local_y,
							x, null_mask, y,
							monoid.getOperator(), phase
						);
				}

#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
				if( !already_dense_output ) {
#else
				if( !already_dense_vectors ) {
#endif
					internal::getCoordinates( x ).asyncJoinSubset( local_x, lower_bound,
						upper_bound );
				}

				return rc;
			};

		ret = ret ? ret : internal::le.addStage(
				std::move( func ),
				internal::Opcode::BLAS1_FOLD_VECTOR_VECTOR_GENERIC,
				n, sizeof( IOType ), dense_descr, true,
				&x, nullptr, &internal::getCoordinates( x ), nullptr,
				&y, nullptr, nullptr, nullptr,
				&internal::getCoordinates( y ), nullptr, nullptr, nullptr,
				nullptr
			);

#ifdef _NONBLOCKING_DEBUG
		std::cout << "\t\tStage added to a pipeline: foldl(x, y, monoid)"
			<< std::endl;
#endif
		return ret;
	}

	template<
		Descriptor descr = descriptors::no_operation,
		class OP,
		typename IOType,
		typename MaskType,
		typename InputType,
		typename Coords
	>
	RC foldl(
		Vector< IOType, nonblocking, Coords > &x,
		const Vector< MaskType, nonblocking, Coords > &m,
		const Vector< InputType, nonblocking, Coords > &y,
		const OP &op = OP(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if< grb::is_operator< OP >::value &&
			!grb::is_object< IOType >::value &&
			!grb::is_object< MaskType >::value &&
			!grb::is_object< InputType >::value, void
		>::type * = nullptr
	) {
		// static sanity checks
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< typename OP::D1, IOType >::value ), "grb::foldl",
			"called with a vector x of a type that does not match the first domain "
			"of the given operator" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< typename OP::D2, InputType >::value ), "grb::foldl",
			"called on a vector y of a type that does not match the second domain "
			"of the given operator" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< typename OP::D3, IOType >::value ), "grb::foldl",
			"called on a vector x of a type that does not match the third domain "
			"of the given operator" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< bool, MaskType >::value ), "grb::foldl",
			"called with a mask that does not have boolean entries " );

		// catch empty mask
		if( size( m ) == 0 ) {
			return foldl< descr >( x, y, op, phase );
		}

		// dynamic sanity checks
		const size_t n = size( x );
		if( n != size( y ) || n != size( m ) ) {
			return MISMATCH;
		}

		RC ret = SUCCESS;

		constexpr const bool dense_descr = descr & descriptors::dense;

		internal::Pipeline::stage_type func =
			[&x, &m, &y, &op, phase] (
				internal::Pipeline &pipeline,
				const size_t lower_bound, const size_t upper_bound
			) {
#ifdef _NONBLOCKING_DEBUG
				#pragma omp critical
				std::cout << "\t\tExecution of stage foldl(x, m, y, op) in the range("
					<< lower_bound << ", " << upper_bound << ")" << std::endl;
#endif
				RC rc = SUCCESS;

				Coords local_x, local_y, local_m;
				const size_t local_n = upper_bound - lower_bound;
				size_t local_x_nz = local_n;
				size_t local_y_nz = local_n;
				bool sparse = false;

#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
				const bool already_dense_vectors = dense_descr ||
					pipeline.allAlreadyDenseVectors();
#else
				(void) pipeline;
				constexpr const bool already_dense_vectors = dense_descr;
#endif
				bool already_dense_output = true;
				bool already_dense_input = true;
				bool already_dense_mask = true;

				if( !already_dense_vectors ) {
#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
					already_dense_output = pipeline.containsAlreadyDenseVector(
						&internal::getCoordinates( x ) );
					if( !already_dense_output ) {
#else
						already_dense_output = false;
#endif
						local_x = internal::getCoordinates( x ).asyncSubset( lower_bound,
							upper_bound );
						local_x_nz = local_x.nonzeroes();
						if( local_x_nz < local_n ) {
							sparse = true;
						}
#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
					}

					already_dense_mask = pipeline.containsAlreadyDenseVector(
						&internal::getCoordinates( m ) );
					if( !already_dense_mask ) {
#else
						already_dense_mask = false;
#endif
						local_m = internal::getCoordinates( m ).asyncSubset( lower_bound,
							upper_bound );
#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
					}

					already_dense_input = pipeline.containsAlreadyDenseVector(
						&internal::getCoordinates( y ) );
					if( !already_dense_input ) {
#else
						already_dense_input = false;
#endif
						local_y = internal::getCoordinates( y ).asyncSubset( lower_bound,
							upper_bound );
						local_y_nz = local_y.nonzeroes();
						if( local_y_nz < local_n ) {
							sparse = true;
						}
#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
					}
#endif
				}

				if( sparse ) {
#ifdef GRB_BOOLEAN_DISPATCHER
					rc = internal::boolean_dispatcher_fold_from_vector_to_vector_generic<
#else
					rc = internal::fold_from_vector_to_vector_generic<
#endif
							descr, true, true, true, false
						>(
							already_dense_output, already_dense_input, already_dense_mask,
							lower_bound, upper_bound,
							local_x, &local_m, local_y,
							x, &m, y,
							op, phase
						);
				} else {
					assert( local_x_nz == local_n );
					assert( local_y_nz == local_n );
#ifdef GRB_BOOLEAN_DISPATCHER
					rc = internal::boolean_dispatcher_fold_from_vector_to_vector_generic<
#else
					rc = internal::fold_from_vector_to_vector_generic<
#endif
							descr, true, false, true, false
						>(
							already_dense_output, already_dense_input, already_dense_mask,
							lower_bound, upper_bound,
							local_x, &local_m, local_y,
							x, &m, y,
							op, phase
						);
				}

#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
			if( !already_dense_output ) {
#else
			if( !already_dense_vectors ) {
#endif
				internal::getCoordinates( x ).asyncJoinSubset( local_x, lower_bound,
					upper_bound );
			}

			return rc;
		};

		ret = ret ? ret : internal::le.addStage(
				std::move( func ),
				internal::Opcode::BLAS1_FOLD_MASKED_VECTOR_VECTOR_GENERIC,
				n, sizeof( IOType ), dense_descr, true,
				&x, nullptr, &internal::getCoordinates( x ), nullptr,
				&y, &m, nullptr, nullptr,
				&internal::getCoordinates( y ), &internal::getCoordinates( m ), nullptr, nullptr,
				nullptr
			);

#ifdef _NONBLOCKING_DEBUG
		std::cout << "\t\tStage added to a pipeline: foldl(x, m, y, op)" << std::endl;
#endif
		return ret;
	}

	template<
		Descriptor descr = descriptors::no_operation,
		class Monoid,
		typename IOType,
		typename MaskType,
		typename InputType,
		typename Coords
	>
	RC foldl(
		Vector< IOType, nonblocking, Coords > &x,
		const Vector< MaskType, nonblocking, Coords > &m,
		const Vector< InputType, nonblocking, Coords > &y,
		const Monoid &monoid = Monoid(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if< grb::is_monoid< Monoid >::value &&
			!grb::is_object< IOType >::value &&
			!grb::is_object< MaskType >::value &&
			!grb::is_object< InputType >::value, void
		>::type * = nullptr
	) {
		// static sanity checks
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< typename Monoid::D1, IOType >::value ), "grb::foldl",
			"called with a vector x of a type that does not match the first domain "
			"of the given operator" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< typename Monoid::D2, InputType >::value ), "grb::foldl",
			"called on a vector y of a type that does not match the second domain "
			"of the given operator" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< typename Monoid::D3, IOType >::value ), "grb::foldl",
			"called on a vector x of a type that does not match the third domain "
			"of the given operator" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< bool, MaskType >::value ), "grb::foldl",
			"called with a mask that does not have boolean entries" );

		// catch empty mask
		if( size( m ) == 0 ) {
			return foldl< descr >( x, y, monoid, phase );
		}

		// dynamic sanity checks
		const size_t n = size( x );
		if( n != size( y ) || n != size( m ) ) {
			return MISMATCH;
		}

		RC ret = SUCCESS;

		constexpr const bool dense_descr = descr & descriptors::dense;

		internal::Pipeline::stage_type func =
			[&x, &m, &y, &monoid, phase] (
				internal::Pipeline &pipeline,
				const size_t lower_bound, const size_t upper_bound
			) {
#ifdef _NONBLOCKING_DEBUG
				#pragma omp critical
				std::cout << "\t\tExecution of stage foldl(x, m, y, monoid) in the range("
					<< lower_bound << ", " << upper_bound << ")" << std::endl;
#endif
				RC rc = SUCCESS;

				Coords local_x, local_y, local_m;
				const size_t local_n = upper_bound - lower_bound;
				size_t local_x_nz = local_n;
				size_t local_y_nz = local_n;
				bool sparse = false;

#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
				const bool already_dense_vectors = dense_descr ||
					pipeline.allAlreadyDenseVectors();
#else
				(void) pipeline;
				constexpr const bool already_dense_vectors = dense_descr;
#endif
				bool already_dense_output = true;
				bool already_dense_input = true;
				bool already_dense_mask = true;

				if( !already_dense_vectors ) {
#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
					already_dense_output = pipeline.containsAlreadyDenseVector(
						&internal::getCoordinates( x ) );
					if( !already_dense_output ) {
#else
						already_dense_output = false;
#endif
						local_x = internal::getCoordinates( x ).asyncSubset( lower_bound,
							upper_bound );
						local_x_nz = local_x.nonzeroes();
						if( local_x_nz < local_n ) {
							sparse = true;
						}
#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
					}

					already_dense_mask = pipeline.containsAlreadyDenseVector(
						&internal::getCoordinates( m ) );
					if( !already_dense_mask ) {
#else
						already_dense_mask = false;
#endif
						local_m = internal::getCoordinates( m ).asyncSubset( lower_bound,
							upper_bound );
#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
					}

					already_dense_input = pipeline.containsAlreadyDenseVector(
						&internal::getCoordinates( y ) );
					if( !already_dense_input ) {
#else
						already_dense_input = false;
#endif
						local_y = internal::getCoordinates( y ).asyncSubset( lower_bound,
							upper_bound );
						local_y_nz = local_y.nonzeroes();
						if( local_y_nz < local_n ) {
							sparse = true;
						}
#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
					}
#endif
				}

				if( sparse ) {
#ifdef GRB_BOOLEAN_DISPATCHER
					rc = internal::boolean_dispatcher_fold_from_vector_to_vector_generic<
#else
					rc = internal::fold_from_vector_to_vector_generic<
#endif
							descr, true, true, true, true
						>(
							already_dense_output, already_dense_input, already_dense_mask,
							lower_bound, upper_bound,
							local_x, &local_m, local_y,
							x, &m, y,
							monoid.getOperator(), phase
						);
				} else {
					assert( local_x_nz == local_n );
					assert( local_y_nz == local_n );

#ifdef GRB_BOOLEAN_DISPATCHER
					rc = internal::boolean_dispatcher_fold_from_vector_to_vector_generic<
#else
					rc = internal::fold_from_vector_to_vector_generic<
#endif
							descr, true, false, true, true
						>(
							already_dense_output, already_dense_input, already_dense_mask,
							lower_bound, upper_bound,
							local_x, &local_m, local_y,
							x, &m, y,
							monoid.getOperator(), phase
						);
				}

#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
				if( !already_dense_output ) {
#else
				if( !already_dense_vectors ) {
#endif
					internal::getCoordinates( x ).asyncJoinSubset( local_x, lower_bound,
						upper_bound );
				}

				return rc;
			};

		ret = ret ? ret : internal::le.addStage(
				std::move( func ),
				internal::Opcode::BLAS1_FOLD_MASKED_VECTOR_VECTOR_GENERIC,
				n, sizeof( IOType ),
				dense_descr, true,
				&x, nullptr, &internal::getCoordinates( x ), nullptr,
				&y, &m, nullptr, nullptr,
				&internal::getCoordinates( y ), &internal::getCoordinates( m ),
				nullptr, nullptr,
				nullptr
			);

#ifdef _NONBLOCKING_DEBUG
		std::cout << "\t\tStage added to a pipeline: foldl(x, m, y, monoid)"
			<< std::endl;
#endif
		return ret;
	}

	namespace internal {

		template<
			bool left_scalar,
			bool right_scalar,
			bool left_sparse,
			bool right_sparse,
			Descriptor descr, class OP,
#ifdef GRB_BOOLEAN_DISPATCHER
			bool already_dense_input_x,
			bool already_dense_input_y,
#endif
			typename OutputType,
			typename InputType1,
			typename InputType2,
			typename Coords
		>
		RC dense_apply_generic(
#ifndef GRB_BOOLEAN_DISPATCHER
			bool already_dense_input_x,
			bool already_dense_input_y,
#endif
			const size_t lower_bound,
			const size_t upper_bound,
			const Coords &local_x,
			const Coords &local_y,
			Vector< OutputType, nonblocking, Coords > &z_vector,
			const internal::Wrapper< left_scalar, InputType1, Coords > x_wrapper,
			const internal::Wrapper< right_scalar, InputType2, Coords > y_wrapper,
			const OP &op
		) {
#ifdef _DEBUG
			std::cout << "\t internal::dense_apply_generic called\n";
#endif
			static_assert( !(left_scalar && left_sparse),
				"The left-hand side must be scalar OR sparse, but cannot be both!" );
			static_assert( !(right_scalar && right_sparse),
				"The right-hand side must be scalar OR sparse, but cannot be both!" );
			static_assert( !(left_sparse && right_sparse),
				"If both left- and right-hand sides are sparse, use sparse_apply_generic "
				"instead." );

			// create local copies of the input const pointers
			OutputType * __restrict__ const z_p = internal::getRaw( z_vector );
			const InputType1 * __restrict__ x_p = x_wrapper.getRaw();
			const InputType2 * __restrict__ y_p = y_wrapper.getRaw();

			const size_t local_n = upper_bound - lower_bound;

			constexpr const size_t block_size = OP::blocksize;
			const size_t num_blocks = local_n / block_size;

#ifndef NDEBUG
			const bool has_coda = local_n % block_size > 0;
#endif
			size_t i = 0 + lower_bound;
			const size_t start = 0;
			const size_t end = num_blocks;

			// declare and initialise local buffers for SIMD
			OutputType z_b[ block_size ];
			InputType1 x_b[ block_size ];
			InputType2 y_b[ block_size ];
			bool x_m[ block_size ];
			bool y_m[ block_size ];
			for( size_t k = 0; k < block_size; ++k ) {
				if( left_scalar ) {
					x_b[ k ] = x_wrapper.getValue();
				}
				if( right_scalar ) {
					y_b[ k ] = y_wrapper.getValue();
				}
			}

			for( size_t block = start; block < end; ++block ) {
				size_t local_i = i;
				for( size_t k = 0; k < block_size; ++k ) {
					if( !left_scalar ) {
						x_b[ k ] = x_p[ local_i ];
					}
					if( !right_scalar ) {
						y_b[ k ] = y_p[ local_i ];
					}
					if( left_sparse ) {
						x_m[ k ] = already_dense_input_x || local_x.assigned( local_i -
							lower_bound );
					}
					if( right_sparse ) {
						y_m[ k ] = already_dense_input_y || local_y.assigned( local_i -
							lower_bound );
					}
					(void) ++local_i;
				}
				for( size_t k = 0; k < block_size; ++k ) {
					RC rc = SUCCESS;
					if( left_sparse && !x_m[ k ] ) {
						z_b[ k ] = y_b[ k ]; // WARNING: assumes monoid semantics!
					} else if( right_sparse && !y_m[ k ] ) {
						z_b[ k ] = x_b[ k ]; // WARNING: assumes monoid semantics!
					} else {
						rc = apply( z_b[ k ], x_b[ k ], y_b[ k ], op );
					}
					assert( rc == SUCCESS );
#ifdef NDEBUG
					(void) rc;
#endif
				}
				for( size_t k = 0; k < block_size; ++k, ++i ) {
					z_p[ i ] = z_b[ k ];
				}
			}

#ifndef NDEBUG
			if( has_coda ) {
				assert( i < local_n + lower_bound );
			} else {
				assert( i == local_n + lower_bound );
			}
#endif

			i = end * block_size + lower_bound;
			for( ; i < local_n + lower_bound; ++i ) {
				RC rc = SUCCESS;
				if( left_scalar && right_scalar ) {
					rc = apply( z_p[ i ], x_wrapper.getValue(), y_wrapper.getValue(), op );
				} else if( left_scalar && !right_scalar ) {
					if( right_sparse && !( already_dense_input_y || local_y.assigned( i -
						lower_bound ) )
					) {
						z_p[ i ] = x_wrapper.getValue();
					} else {
						rc = apply( z_p[ i ], x_wrapper.getValue(), y_p[ i ], op );
					}
				} else if( !left_scalar && right_scalar ) {
					if( left_sparse && !( already_dense_input_x || local_x.assigned( i -
						lower_bound ) )
					) {
						z_p[ i ] = y_wrapper.getValue();
					} else {
						rc = apply( z_p[ i ], x_p[ i ], y_wrapper.getValue(), op );
					}
				} else {
					assert( !left_scalar && !right_scalar );
					if( left_sparse && !(already_dense_input_x || local_x.assigned( i -
						lower_bound ) )
					) {
						z_p[ i ] = y_p[ i ];
					} else if( right_sparse && !(already_dense_input_y || local_y.assigned( i -
						lower_bound ) )
					) {
						z_p[ i ] = x_p[ i ];
					} else {
						assert( !left_sparse && !right_sparse );
						rc = apply( z_p[ i ], x_p[ i ], y_p[ i ], op );
					}
				}
				assert( rc == SUCCESS );
#ifdef NDEBUG
				(void) rc;
#endif
			}

			return SUCCESS;
		}

		template<
			bool masked,
			bool monoid,
			bool x_scalar,
			bool y_scalar,
			Descriptor descr,
			class OP,
#ifdef GRB_BOOLEAN_DISPATCHER
			bool already_dense_mask,
			bool already_dense_input_x,
			bool already_dense_input_y,
#endif
			typename OutputType,
			typename MaskType,
			typename InputType1,
			typename InputType2,
			typename Coords
		>
		RC sparse_apply_generic(
#ifndef GRB_BOOLEAN_DISPATCHER
			bool already_dense_mask,
			bool already_dense_input_x,
			bool already_dense_input_y,
#endif
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
#ifndef GRB_NO_NOOP_CHECKS
			static_assert( !internal::maybe_noop< OP >::value, "Warning: you may be "
				"generating an output vector with uninitialised values. Define "
				"the GRB_NO_NOOP_CHECKS macro to disable this check.\n" );
#endif
			// create local copies of the input const pointers
			OutputType * __restrict__ const z_p = internal::getRaw( z_vector );
			const MaskType * __restrict__ const mask_p = ( masked )
				? internal::getRaw( *mask_vector )
				: nullptr;
			const InputType1 * __restrict__ x_p = x_wrapper.getRaw();
			const InputType2 * __restrict__ y_p = y_wrapper.getRaw();

			Coords local_mask;
			if( masked ) {
				local_mask = *local_mask_ptr;
			}

			const size_t local_n = upper_bound - lower_bound;
			const size_t local_x_nz = already_dense_input_x
				? local_n
				: local_x.nonzeroes();
			const size_t local_y_nz = already_dense_input_y
				? local_n
				: local_y.nonzeroes();

			// assertions
			assert( !masked || local_mask_ptr != nullptr );
			assert( !masked || local_mask_ptr->size() == local_n );
			assert( x_scalar || local_x_nz <= local_n );
			assert( y_scalar || local_y_nz <= local_n );

#ifdef _DEBUG
			std::cout << "\tinternal::sparse_apply_generic called\n";
#endif
			constexpr const size_t block_size = OP::blocksize;

			// swap so that we do the expensive pass over the container with the fewest
			// nonzeroes first
			assert( !x_scalar || !y_scalar );
			const bool swap = ( ( x_scalar || already_dense_input_x )
					? local_n
					: local_x_nz
				) > ( ( y_scalar || already_dense_input_y )
					? local_n
					: local_y_nz
				);
			const Coordinates< nonblocking > &loop_coors = swap ? local_y : local_x;
			const Coordinates< nonblocking > &chk_coors = swap ? local_x : local_y;
			const bool already_dense_loop = swap
				? already_dense_input_y
				: already_dense_input_x;
			const bool already_dense_chk = swap
				? already_dense_input_x
				: already_dense_input_y;

			const size_t loop_coors_nz = swap ? local_y_nz : local_x_nz;
			const size_t chk_coors_nz = swap ? local_x_nz : local_y_nz;
#ifdef _DEBUG
			std::cout << "\t\tfirst-phase loop of size " << loop_coors.size() << "\n";
			if( x_scalar || y_scalar ) {
				std::cout << "\t\tthere will be no second phase because one of the inputs "
					<< "is scalar\n";
			} else {
				std::cout << "\t\tsecond-phase loop of size " << chk_coors.size() << "\n";
			}
#endif
			// declare buffers for vectorisation
			size_t offsets[ block_size ];
			OutputType z_b[ block_size ];
			InputType1 x_b[ block_size ];
			InputType2 y_b[ block_size ];
			bool mask[ block_size ];
			bool x_m[ block_size ];
			bool y_m[ block_size ];

			if( x_scalar ) {
				for( size_t k = 0; k < block_size; ++k ) {
					x_b[ k ] = x_wrapper.getValue();
				}
			}
			if( y_scalar ) {
				for( size_t k = 0; k < block_size; ++k ) {
					y_b[ k ] = y_wrapper.getValue();
				}
			}

			// expensive pass #1
			size_t start = 0;
			size_t end = loop_coors_nz / block_size;
			size_t k = 0;
			for( size_t b = start; b < end; ++b ) {
				// perform gathers
				for( size_t i = 0; i < block_size; ++i ) {
					const size_t index = ( already_dense_loop )
						? ( ( k++ ) + lower_bound )
						: ( loop_coors.index( k++ ) + lower_bound );
					offsets[ i ] = index;
					assert( index < local_n + lower_bound );
					if( masked ) {
						if( already_dense_mask ) {
							mask[ i ] = internal::getCoordinates( *mask_vector ).template
								mask< descr >( index, mask_p );
						} else {
							mask[ i ] = local_mask.template mask< descr >( index - lower_bound,
								mask_p + lower_bound );
						}
					}
				}
				// perform gathers
				for( size_t i = 0; i < block_size; ++i ) {
					if( !masked || mask[ i ] ) {
						if( !x_scalar ) {
							x_b[ i ] = x_p[ offsets[ i ] ];
						}
						if( !x_scalar && !y_scalar ) {
							y_m[ i ] = already_dense_chk || chk_coors.assigned( offsets[ i ] -
								lower_bound );
						} else {
							y_m[ i ] = true;
						}
						if( !y_scalar ) {
							y_b[ i ] = y_p[ offsets[ i ] ];
						}
					} else {
						y_m[ i ] = false;
					}
				}
				// perform compute
				for( size_t i = 0; i < block_size; ++i ) {
					RC rc = SUCCESS;
					if( y_m[ i ] ) {
						rc = apply( z_b[ i ], x_b[ i ], y_b[ i ], op );
					} else if( monoid ) {
						if( swap ) {
							z_b[ i ] = static_cast< typename OP::D3 >( x_b[ i ] );
						} else {
							z_b[ i ] = static_cast< typename OP::D3 >( y_b[ i ] );
						}
					}
					assert( rc == SUCCESS );
#ifdef NDEBUG
					(void) rc;
#endif
				}
				// part that may or may not be vectorised (can we do something about this??)
				for( size_t i = 0; i < block_size; ++i ) {
					if( !masked || mask[ i ] ) {
						if( y_m[ i ] || monoid ) {
							(void) local_z.assign( offsets[ i ] - lower_bound );
						}
					}
				}
				// perform scatter
				for( size_t i = 0; i < block_size; ++i ) {
					if( !masked || mask[ i ] ) {
						if( monoid || y_m[ i ] ) {
							GRB_UTIL_IGNORE_MAYBE_UNINITIALIZED // the only way the below could write
							                                    // an uninitialised value is if the
											    // static_assert at the top of this
							z_p[ offsets[ i ] ] = z_b[ i ];     // function had triggered. See also
							GRB_UTIL_RESTORE_WARNINGS           // internal issue #321.
						}
					}
				}
			}

			for( ; k < loop_coors_nz; ++k ) {
				const size_t index = ( already_dense_loop )
					? k + lower_bound
					: loop_coors.index( k ) + lower_bound;
				if( masked ) {
					if( already_dense_mask ) {
						if( !internal::getCoordinates( *mask_vector ).template mask< descr >(
							index, mask_p )
						) {
							continue;
						}
					} else {
						if( !local_mask.template mask< descr >( index - lower_bound, mask_p +
							lower_bound )
						) {
							continue;
						}
					}
				}
				RC rc = SUCCESS;
				(void) local_z.assign( index - lower_bound );
				if( x_scalar || y_scalar || already_dense_chk || chk_coors.assigned(
					index - lower_bound )
				) {
					rc = apply(
						z_p[ index ],
						( x_scalar )
							? x_wrapper.getValue()
							: x_p[ index ],
						( y_scalar )
							? y_wrapper.getValue()
							: y_p[ index ],
						op
					);
				} else if( monoid ) {
					if( swap ) {
						z_p[ index ] = x_scalar ?
							static_cast< typename OP::D3 >( x_wrapper.getValue() ) :
							static_cast< typename OP::D3 >( x_p[ index ] );
					} else {
						z_p[ index ] = y_scalar ?
							static_cast< typename OP::D3 >( y_wrapper.getValue() ) :
							static_cast< typename OP::D3 >( y_p[ index ] );
					}
				}
				assert( rc == SUCCESS );
#ifdef NDEBUG
				(void) rc;
#endif
			}

			// cheaper pass #2, only required if we are using monoid semantics
			// AND if both inputs are vectors
			if( monoid && !x_scalar && !y_scalar ) {
				start = 0;
				end = chk_coors_nz / block_size;
				k = 0;
				for( size_t b = start; b < end; ++b ) {
					// streaming load
					for( size_t i = 0; i < block_size; i++ ) {
						offsets[ i ] = ( already_dense_chk )
							? ( ( k++ ) + lower_bound )
							: ( chk_coors.index( k++ ) + lower_bound );
						assert( offsets[ i ] < local_n + lower_bound );
					}
					// pure gather
					for( size_t i = 0; i < block_size; i++ ) {
						x_m[ i ] = already_dense_loop || loop_coors.assigned( offsets[ i ] -
							lower_bound );
					}
					// gather-like
					for( size_t i = 0; i < block_size; i++ ) {
						if( masked ) {
							if( already_dense_mask ) {
								mask[ i ] = utils::interpretMask< descr >(
										internal::getCoordinates( *mask_vector ).assigned( offsets[ i ] ),
										mask_p, offsets[ i ]
									);
							} else {
								mask[ i ] = utils::interpretMask< descr >(
										local_mask.assigned( offsets[ i ] - lower_bound ),
										mask_p, offsets[ i ]
									);
							}
						}
					}
					// SIMD
					for( size_t i = 0; i < block_size; i++ ) {
						x_m[ i ] = ! x_m[ i ];
					}
					// SIMD
					for( size_t i = 0; i < block_size; i++ ) {
						if( masked ) {
							mask[ i ] = mask[ i ] && x_m[ i ];
						}
					}
					if( !swap ) {
						// gather
						for( size_t i = 0; i < block_size; ++i ) {
							if( masked ) {
								if( mask[ i ] ) {
									y_b[ i ] = y_p[ offsets[ i ] ];
								}
							} else {
								if( x_m[ i ] ) {
									y_b[ i ] = y_p[ offsets[ i ] ];
								}
							}
						}
						// SIMD
						for( size_t i = 0; i < block_size; i++ ) {
							if( masked ) {
								if( mask[ i ] ) {
									z_b[ i ] = y_b[ i ];
								}
							} else {
								if( x_m[ i ] ) {
									z_b[ i ] = y_b[ i ];
								}
							}
						}
					} else {
						// gather
						for( size_t i = 0; i < block_size; ++i ) {
							if( masked ) {
								if( mask[ i ] ) {
									x_b[ i ] = x_p[ offsets[ i ] ];
								}
							} else {
								if( x_m[ i ] ) {
									x_b[ i ] = x_p[ offsets[ i ] ];
								}
							}
						}
						// SIMD
						for( size_t i = 0; i < block_size; i++ ) {
							if( masked ) {
								if( mask[ i ] ) {
									z_b[ i ] = static_cast< typename OP::D3 >( x_b[ i ] );
								}
							} else {
								if( x_m[ i ] ) {
									z_b[ i ] = static_cast< typename OP::D3 >( x_b[ i ] );
								}
							}
						}
					}
					// SIMD-like
					for( size_t i = 0; i < block_size; i++ ) {
						if( masked ) {
							if( mask[ i ] ) {
								(void)local_z.assign( offsets[ i ] - lower_bound );
							}
						} else {
							if( x_m[ i ] ) {
								(void)local_z.assign( offsets[ i ] - lower_bound );
							}
						}
					}
					// scatter
					for( size_t i = 0; i < block_size; i++ ) {
						if( masked ) {
							if( mask[ i ] ) {
								GRB_UTIL_IGNORE_MAYBE_UNINITIALIZED

								z_p[ offsets[ i ] ] = z_b[ i ];

								GRB_UTIL_RESTORE_WARNINGS
							}
						} else {
							if( x_m[ i ] ) {
#ifdef _DEBUG
								std::cout << "\t\t writing out " << z_b[ i ] << " to index "
									<< offsets[ i ] << "\n";
#endif
								GRB_UTIL_IGNORE_MAYBE_UNINITIALIZED // the only way the below could write
								                                    // an uninitialised value is if the
												    // static_assert at the top of this
								z_p[ offsets[ i ] ] = z_b[ i ];     // function had triggered. See also
								GRB_UTIL_RESTORE_WARNINGS           // internal issue #321.
							}
						}
					}
				}
				for( ; k < chk_coors_nz; ++k ) {
					const size_t index = ( ( already_dense_chk )
						? k
						: chk_coors.index( k ) ) + lower_bound;
					assert( index < local_n + lower_bound );
					if( already_dense_loop || loop_coors.assigned( index - lower_bound) ) {
						continue;
					}
					if( masked ) {
						if( already_dense_mask ) {
							if( !internal::getCoordinates( *mask_vector ).template mask< descr >(
								index, mask_p )
							) {
								continue;
							}
						} else {
							if( !local_mask.template mask< descr >( index - lower_bound , mask_p +
								lower_bound )
							) {
								continue;
							}
						}
					}
					(void) local_z.assign( index - lower_bound );
					z_p[ index ] = swap ? x_p[ index ] : y_p[ index ];
				}
			}

			return SUCCESS;
		}

		template<
			bool left_scalar,
			bool right_scalar,
			bool left_sparse,
			bool right_sparse,
			Descriptor descr,
			class OP,
#ifdef GRB_BOOLEAN_DISPATCHER
			bool already_dense_mask,
			bool already_dense_input_x,
			bool already_dense_input_y,
#endif
			typename OutputType, typename MaskType,
			typename InputType1, typename InputType2,
			typename Coords
		>
		RC masked_apply_generic(
#ifndef GRB_BOOLEAN_DISPATCHER
			bool already_dense_mask,
			bool already_dense_input_x,
			bool already_dense_input_y,
#endif
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
		) {
#ifdef _DEBUG
			std::cout << "In masked_apply_generic< " << left_scalar << ", "
				<< right_scalar << ", " << left_sparse << ", " << right_sparse << ", "
				<< descr << " > with lower_bound = " << lower_bound << " and upper_bound = "
				<< upper_bound << "\n";
#endif
			// assertions
			static_assert( !(left_scalar && left_sparse),
				"left_scalar and left_sparse cannot both be set!"
			);
			static_assert( !(right_scalar && right_sparse),
				"right_scalar and right_sparse cannot both be set!"
			);
			assert( !left_sparse || left_identity != nullptr );
			assert( !right_sparse || right_identity != nullptr );

			// create local copies of the input const pointers
			OutputType * __restrict__ const z_p = internal::getRaw( z_vector );
			const MaskType * __restrict__ const mask_p = internal::getRaw( mask_vector );
			const InputType1 * __restrict__ x_p = x_wrapper.getRaw();
			const InputType2 * __restrict__ y_p = y_wrapper.getRaw();

			const size_t local_n = upper_bound - lower_bound;
			const size_t local_mask_nz = ( already_dense_mask )
				? local_n
				: local_mask.nonzeroes();
#ifdef _DEBUG
			std::cout << "\tinternal::masked_apply_generic called with nnz(mask)="
				<< local_mask_nz << " and descriptor " << descr << "\n";
			if( local_mask_nz > 0 ) {
				std::cout << "\t\tNonzero mask indices: "
					<< ( already_dense_mask ? 0 : local_mask.index( 0 ) );
				assert( local_mask.assigned( local_mask.index( 0 ) ) );
				for( size_t k = 1; k < local_mask_nz; ++k ) {
					std::cout << ", "
						<< ( ( already_dense_mask ) ? k : local_mask.index( k ) );
					assert(
						already_dense_mask ||
						local_mask.assigned( local_mask.index( k ) )
					);
				}
				std::cout << "\n";
			}

			size_t unset = 0;
			for( size_t i = 0; i < local_n; ++i ) {
				if( !( already_dense_mask || local_mask.assigned( i ) ) ) {
					(void) ++unset;
				}
			}
			assert( unset == local_n - local_mask_nz );
#endif
			// whether to use a Theta(n) or a Theta(nnz(mask)) loop
			const bool bigLoop = local_mask_nz == local_n ||
				(descr & descriptors::invert_mask);

			// get block size
			constexpr size_t size_t_block_size = config::SIMD_SIZE::value() /
				sizeof( size_t );
			constexpr size_t op_block_size = OP::blocksize;
			constexpr size_t min_block_size = op_block_size > size_t_block_size
				? size_t_block_size
				: op_block_size;

			if( bigLoop ) {
#ifdef _DEBUG
				std::cerr << "\t in bigLoop variant\n";
#endif
				size_t i = 0 + lower_bound;

				constexpr const size_t block_size = op_block_size;
				const size_t num_blocks = local_n / block_size;
				const size_t start = 0;
				const size_t end = num_blocks;

				// declare buffers that fit in a single SIMD register and initialise if
				// needed
				bool mask_b[ block_size ];
				OutputType z_b[ block_size ];
				InputType1 x_b[ block_size ];
				InputType2 y_b[ block_size ];
				for( size_t k = 0; k < block_size; ++k ) {
					if( left_scalar ) {
						x_b[ k ] = x_wrapper.getValue();
					}
					if( right_scalar ) {
						y_b[ k ] = y_wrapper.getValue();
					}
				}
				for( size_t b = start; b < end; ++b ) {
					for( size_t k = 0; k < block_size; ++k ) {
						const size_t index = i + k;
						assert( index < local_n + lower_bound );
						if( already_dense_mask ) {
							mask_b[ k ] = internal::getCoordinates( mask_vector ).template
								mask< descr >( index, mask_p );
						} else {
							mask_b[ k ] = local_mask.template
								mask< descr >( index - lower_bound, mask_p + lower_bound );
						}
					}
					// check for no output
					if( left_sparse && right_sparse ) {
						for( size_t k = 0; k < block_size; ++k ) {
							const size_t index = i + k;
							assert( index < local_n + lower_bound );
							if( mask_b[ k ] ) {
								if( !( already_dense_input_x ||
										local_x.assigned( index - lower_bound )
									) && !(
										already_dense_input_y ||
										local_y.assigned( index - lower_bound )
									)
								) {
									mask_b[ k ] = false;
								}
							}
						}
					}
					for( size_t k = 0; k < block_size; ++k ) {
						const size_t index = i + k;
						assert( index < local_n + lower_bound );
						if( mask_b[ k ] ) {
							if( !left_scalar ) {
								if( left_sparse && !(
									already_dense_input_x || local_x.assigned( index - lower_bound )
								) ) {
									x_b[ k ] = *left_identity;
								} else {
									x_b[ k ] = *( x_p + index );
								}
							}
							if( !right_scalar ) {
								if( right_sparse && !(
									already_dense_input_y || local_y.assigned( index - lower_bound )
								) ) {
									y_b[ k ] = *right_identity;
								} else {
									y_b[ k ] = *( y_p + index );
								}
							}
						}
					}
					for( size_t k = 0; k < block_size; ++k ) {
						if( mask_b[ k ] ) {
							apply( z_b[ k ], x_b[ k ], y_b[ k ], op );
						}
					}
					for( size_t k = 0; k < block_size; ++k ) {
						const size_t index = i + k;
						assert( index < local_n + lower_bound );
						if( mask_b[ k ] ) {
							(void) local_z.assign( index - lower_bound );
							GRB_UTIL_IGNORE_MAYBE_UNINITIALIZED // This is only triggered with
							*( z_p + index ) = z_b[ k ];        // mask_b[ k ], which in the above
							GRB_UTIL_RESTORE_WARNINGS           // loop also triggeres initialising
							                                    // z_b[ k ]
						}
					}

					i += block_size;
				}
				// scalar coda
				for(
					size_t i = end * block_size + lower_bound;
					i < local_n + lower_bound;
					++i
				) {
					if( already_dense_mask ) {
						if( !internal::getCoordinates( mask_vector ).template mask< descr >( i,
							mask_p )
						) {
							continue;
						}
					} else {
						if( !local_mask.template mask< descr >( i - lower_bound, mask_p +
							lower_bound )
						) {
							continue;
						}
					}

					if( left_sparse && right_sparse ) {
						if( !( already_dense_input_x || local_x.assigned( i  - lower_bound ) ) &&
							!( already_dense_input_y || local_y.assigned( i - lower_bound ) )
						) {
							continue;
						}
					}
					(void) local_z.assign( i - lower_bound );
					const InputType1 x_e = left_scalar
							? x_wrapper.getValue()
							: ( (!left_sparse || already_dense_input_x ||
								local_x.assigned( i - lower_bound ))
								? *(x_p + i)
								: *left_identity
							);
					const InputType2 y_e = right_scalar
							? y_wrapper.getValue()
							: ( (!right_sparse || already_dense_input_y ||
								local_y.assigned( i - lower_bound ))
								? *(y_p + i)
								: *right_identity
							);
					OutputType * const z_e = z_p + i;
					apply( *z_e, x_e, y_e, op );
				}
			} else {
#ifdef _DEBUG
				std::cerr << "\t in smallLoop variant\n";
#endif
				// declare buffers that fit in a single SIMD register and initialise if
				// needed
				constexpr const size_t block_size = size_t_block_size > 0
					? min_block_size
					: op_block_size;
				bool mask_b[ block_size ];
				OutputType z_b[ block_size ];
				InputType1 x_b[ block_size ];
				InputType2 y_b[ block_size ];
				size_t indices[ block_size ];
				for( size_t k = 0; k < block_size; ++k ) {
					if( left_scalar ) {
						x_b[ k ] = x_wrapper.getValue();
					}
					if( right_scalar ) {
						y_b[ k ] = y_wrapper.getValue();
					}
				}

				// loop over mask pattern
				const size_t mask_nnz = local_mask_nz;
				const size_t num_blocks = mask_nnz / block_size;
				const size_t start = 0;
				const size_t end = num_blocks;

				size_t k = 0;

				// vectorised code
				for( size_t b = start; b < end; ++b ) {
					for( size_t t = 0; t < block_size; ++t ) {
						indices[ t ] = (already_dense_mask ) ? k + t : local_mask.index( k + t );
					}
					for( size_t t = 0; t < block_size; ++t ) {
						if( already_dense_mask ) {
							mask_b[ t ] = internal::getCoordinates( mask_vector ).template
								mask< descr >( indices[ t ], mask_p );
						} else {
							mask_b[ t ] = local_mask.template
								mask< descr >( indices[ t ], mask_p + lower_bound );
						}
					}
					for( size_t t = 0; t < block_size; ++t ) {
						if( mask_b[ t ] ) {
							if( !left_scalar ) {
								if( left_sparse && !( already_dense_input_x ||
									local_x.assigned( indices[ t ] ) )
								) {
									x_b[ t ] = *left_identity;
								} else {
									x_b[ t ] = *( x_p + indices[ t ] + lower_bound );
								}
							}
							if( !right_scalar ) {
								if( right_sparse && !( already_dense_input_y ||
									local_y.assigned( indices[ t ] ) )
								) {
									y_b[ t ] = *right_identity;
								} else {
									y_b[ t ] = *( y_p + indices[ t ] + lower_bound );
								}
							}
						}
					}
					// check for no output
					if( left_sparse && right_sparse ) {
						for( size_t t = 0; t < block_size; ++t ) {
							const size_t index = indices[ t ];
							assert( index < local_n + lower_bound );
							if( mask_b[ t ] ) {
								if( !( already_dense_input_x || local_x.assigned( index ) ) &&
									!( already_dense_input_y || local_y.assigned( index ) )
								) {
									mask_b[ t ] = false;
								}
							}
						}
					}
					for( size_t t = 0; t < block_size; ++t ) {
						if( mask_b[ t ] ) {
							apply( z_b[ t ], x_b[ t ], y_b[ t ], op );
						}
					}
					for( size_t t = 0; t < block_size; ++t ) {
						if( mask_b[ t ] ) {
							(void) local_z.assign( indices[ t ] );
							GRB_UTIL_IGNORE_MAYBE_UNINITIALIZED               // z_b is computed from
							*( z_p + indices[ t ] + lower_bound ) = z_b[ t ]; // x_b and y_b, which
							GRB_UTIL_RESTORE_WARNINGS                         // are both initialised
							                                                  // if mask_b is true
						}
					}

					k += block_size;
				}

				// scalar coda
				for( size_t k = end * block_size; k < mask_nnz; ++k ) {
					const size_t i = already_dense_mask
						? k + lower_bound
						: local_mask.index( k ) + lower_bound;
					if( ( already_dense_mask &&
							internal::getCoordinates( mask_vector ).template mask< descr >(
								i, mask_p
							)
						) || local_mask.template mask< descr >(
							i - lower_bound, mask_p + lower_bound
						)
					) {
						if( left_sparse && right_sparse ) {
							if( !( already_dense_input_x || local_x.assigned( i  - lower_bound ) ) &&
								!( already_dense_input_y || local_y.assigned( i - lower_bound ) )
							) {
								continue;
							}
						}
						(void) local_z.assign( i - lower_bound );
						const InputType1 x_e = left_scalar
							? x_wrapper.getValue()
							: (
								(!left_sparse || already_dense_input_x ||
									local_x.assigned( i - lower_bound ) )
									? *(x_p + i)
									: *left_identity
							);
						const InputType2 y_e = right_scalar
							? y_wrapper.getValue()
							: (
								(!right_sparse || already_dense_input_y ||
									local_y.assigned( i - lower_bound ) )
									? *(y_p + i)
									: *right_identity
							);
						OutputType * const z_e = z_p + i;
						apply( *z_e, x_e, y_e, op );
					}
				}
			}
			return SUCCESS;
		}

	} // end namespace ``grb::internal''

	template<
		Descriptor descr = descriptors::no_operation,
		class OP,
		typename OutputType,
		typename InputType1,
		typename InputType2,
		typename Coords
	>
	RC eWiseApply(
		Vector< OutputType, nonblocking, Coords > &z,
		const Vector< InputType1, nonblocking, Coords > &x,
		const InputType2 beta,
		const OP &op = OP(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			!grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			grb::is_operator< OP >::value, void
		>::type * const = nullptr
	) {
		// static checks
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< typename OP::D1, InputType1 >::value ), "grb::eWiseApply",
			"called with a left-hand input element type that does not match the "
			"first domain of the given operator" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< typename OP::D2, InputType2 >::value ), "grb::eWiseApply",
			"called with a right-hand input element type that does not match the "
			"second domain of the given operator" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< typename OP::D3, OutputType >::value ), "grb::eWiseApply",
			"called with an output element type that does not match the "
			"third domain of the given operator" );
#ifdef _DEBUG
		std::cout << "In eWiseApply ([T1]<-[T2]<-T3), operator variant\n";
#endif
		// sanity check
		auto &z_coors = internal::getCoordinates( z );
		const size_t n = z_coors.size();
		if( internal::getCoordinates( x ).size() != n ) {
			return MISMATCH;
		}

		if( phase == RESIZE ) {
			return SUCCESS;
		}
		assert( phase == EXECUTE );

		RC ret = SUCCESS;

		constexpr const bool dense_descr = descr & descriptors::dense;

		internal::Pipeline::stage_type func =
			[&z, &x, beta, &op] (
				internal::Pipeline &pipeline,
				const size_t lower_bound, const size_t upper_bound
			) {
#ifdef _NONBLOCKING_DEBUG
				#pragma omp critical
				std::cout << "\t\tExecution of stage eWiseApply(z, x, beta, operator) in "
					<< "the range(" << lower_bound << ", " << upper_bound << ")" << std::endl;
#endif
				RC rc = SUCCESS;

				const Vector< bool, nonblocking, Coords > * const null_mask = nullptr;
				const Coords * const local_null_mask = nullptr;

				Coords local_mask, local_x, local_y, local_z;
				const size_t local_n = upper_bound - lower_bound;

#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
				const bool already_dense_vectors = dense_descr ||
					pipeline.allAlreadyDenseVectors();
#else
				(void) pipeline;
				constexpr const bool already_dense_vectors = dense_descr;
#endif

				bool already_dense_input_x = true;

				size_t local_x_nz = local_n;

				if( !already_dense_vectors ) {
					local_z = internal::getCoordinates( z ).asyncSubset( lower_bound,
						upper_bound );

#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
					already_dense_input_x = pipeline.containsAlreadyDenseVector(
						&internal::getCoordinates( x ) );
					if( !already_dense_input_x ) {
#else
						already_dense_input_x = false;
#endif
						local_x = internal::getCoordinates( x ).asyncSubset( lower_bound,
							upper_bound );
						local_x_nz = local_x.nonzeroes();
#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
					}
#endif
				}

				const internal::Wrapper< false, InputType1, Coords > x_wrapper( x );
				const internal::Wrapper< true, InputType2, Coords > y_wrapper( beta );

				// the global stack counter must be set to 0 unless it's guaranteed
				// that none of the local_clear and local_assignAll will be invoked
				// - local_clear is not invoked when the dense descriptor is given,
				//   since the output vector will eventually become dense
				// - local_assignAll is not invoked when the output vector is already dense
				// therefore, the following condition relies on global information,
				// i.e., the dense descriptor and the already_dense_output
				if( !already_dense_vectors ) {
					if( lower_bound == 0 ) {
						internal::getCoordinates( z ).reset_global_nnz_counter();
					}
				}

				if( local_x_nz == local_n ) {
					if( !already_dense_vectors ) {
						local_z.local_assignAll( );
					}

					// call dense apply
#ifdef GRB_BOOLEAN_DISPATCHER
					rc = internal::boolean_dispatcher_dense_apply_generic<
#else
					rc = internal::dense_apply_generic<
#endif
							false, true, false, false, descr | descriptors::dense, OP,
							OutputType, InputType1, InputType2, Coords
						>(
							already_dense_input_x, true,
							lower_bound, upper_bound,
							local_x, local_y,
							z, x_wrapper, y_wrapper,
							op
						);
				} else {
					if( !already_dense_vectors ) {
						local_z.local_clear();
#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
						pipeline.markMaybeSparseVector( &internal::getCoordinates( z ) );
#endif
					}

					// since z and x may not perfectly overlap, and since the intersection is
					// unknown a priori, we must iterate over the nonzeroes of x
#ifdef GRB_BOOLEAN_DISPATCHER
					rc = internal::boolean_dispatcher_sparse_apply_generic<
#else
					rc = internal::sparse_apply_generic<
#endif
							false, false, false, true, descr, OP,
							OutputType, bool, InputType1, InputType2, Coords
						>(
							true, already_dense_input_x, true,
							lower_bound, upper_bound,
							local_z, local_null_mask, local_x, local_y,
							z, null_mask, x_wrapper, y_wrapper, op
						);
				}

				if( !already_dense_vectors ) {
					internal::getCoordinates( z ).asyncJoinSubset( local_z, lower_bound,
						upper_bound );
				}

				return rc;
			};

		ret = ret ? ret : internal::le.addStage(
				std::move( func ),
				internal::Opcode::BLAS1_EWISEAPPLY,
				n, sizeof( OutputType ), dense_descr, true,
				&z, nullptr, &internal::getCoordinates( z ), nullptr,
				&x, nullptr, nullptr, nullptr,
				&internal::getCoordinates( x ), nullptr, nullptr, nullptr,
				nullptr
			);

#ifdef _NONBLOCKING_DEBUG
		std::cout << "\t\tStage added to a pipeline: eWiseApply(z, x, beta, operator)"
			<< std::endl;
#endif

		return ret;
	}

	template<
		Descriptor descr = descriptors::no_operation,
		class OP,
		typename OutputType,
		typename InputType1,
		typename InputType2,
		typename Coords
	>
	RC eWiseApply(
		Vector< OutputType, nonblocking, Coords > &z,
		const InputType1 alpha,
		const InputType2 beta,
		const OP &op = OP(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			!grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			grb::is_operator< OP >::value, void
		>::type * const = nullptr
	) {
		// static checks
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< typename OP::D1, InputType1 >::value ), "grb::eWiseApply",
			"called with a left-hand input element type that does not match the "
			"first domain of the given operator" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< typename OP::D2, InputType2 >::value ), "grb::eWiseApply",
			"called with a right-hand input element type that does not match the "
			"second domain of the given operator" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< typename OP::D3, OutputType >::value ), "grb::eWiseApply",
			"called with an output element type that does not match the "
			"third domain of the given operator" );
#ifdef _DEBUG
		std::cout << "In eWiseApply ([T1]<-T2<-T3), operator variant\n";
#endif
		if( (descr & descriptors::dense) && nnz( z ) < size( z ) ) {
			return ILLEGAL;
		}
		if( phase == RESIZE ) {
			return SUCCESS;
		}
		assert( phase == EXECUTE );

		typename OP::D3 val;
		RC ret = apply< descr >( val, alpha, beta, op );
		ret = ret ? ret : set< descr >( z, val );
		return ret;
	}

	template<
		Descriptor descr = descriptors::no_operation,
		class OP,
		typename OutputType, typename MaskType,
		typename InputType1, typename InputType2,
		typename Coords
	>
	RC eWiseApply(
		Vector< OutputType, nonblocking, Coords > &z,
		const Vector< MaskType, nonblocking, Coords > &mask,
		const InputType1 alpha,
		const InputType2 beta,
		const OP &op = OP(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			!grb::is_object< OutputType >::value &&
			!grb::is_object< MaskType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			grb::is_operator< OP >::value, void
		>::type * const = nullptr
	) {
		// static checks
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< typename OP::D1, InputType1 >::value ), "grb::eWiseApply",
			"called with a left-hand input element type that does not match the "
			"first domain of the given operator" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< typename OP::D2, InputType2 >::value ), "grb::eWiseApply",
			"called with a right-hand input element type that does not match the "
			"second domain of the given operator" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< typename OP::D3, OutputType >::value ), "grb::eWiseApply",
			"called with an output element type that does not match the "
			"third domain of the given operator" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< bool, MaskType >::value ), "grb::eWiseApply",
			"called with an output mask element type that is not Boolean " );
#ifdef _DEBUG
		std::cout << "In masked eWiseApply ([T1]<-T2<-T3), operator variant\n";
#endif
		// check trivial dispatch
		if( size( mask ) == 0 ) {
			return eWiseApply< descr >( z, alpha, beta, op, phase );
		}

		// dynamic checks
		if( size( mask ) != size( z ) ) {
			return MISMATCH;
		}
		if( (descr & descriptors::dense) &&
			( nnz( z ) < size( z ) || nnz( mask ) < size( mask ) )
		) {
			return ILLEGAL;
		}

		// check trivial dispatch
		if( phase == RESIZE ) {
			return SUCCESS;
		}
		assert( phase == EXECUTE );

		typename OP::D3 val;
		RC ret = apply< descr >( val, alpha, beta, op );
		ret = ret ? ret : set< descr >( z, mask, val );
		return ret;
	}

	template<
		Descriptor descr = descriptors::no_operation,
		class Monoid,
		typename OutputType,
		typename InputType1,
		typename InputType2,
		typename Coords
	>
	RC eWiseApply(
		Vector< OutputType, nonblocking, Coords > &z,
		const InputType1 alpha,
		const InputType2 beta,
		const Monoid &monoid = Monoid(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
				!grb::is_object< OutputType >::value &&
				!grb::is_object< InputType1 >::value &&
				!grb::is_object< InputType2 >::value &&
				grb::is_monoid< Monoid >::value,
			void
		>::type * const = nullptr
	) {
		// static checks
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< typename Monoid::D1, InputType1 >::value ), "grb::eWiseApply",
			"called with a left-hand input element type that does not match the "
			"first domain of the given monoid" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< typename Monoid::D2, InputType2 >::value ), "grb::eWiseApply",
			"called with a right-hand input element type that does not match the "
			"second domain of the given monoid" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< typename Monoid::D3, OutputType >::value ), "grb::eWiseApply",
			"called with an output element type that does not match the "
			"third domain of the given monoid" );
#ifdef _DEBUG
		std::cout << "In eWiseApply ([T1]<-T2<-T3), monoid variant\n";
#endif
		// simply delegate to operator variant
		return eWiseApply< descr >( z, alpha, beta, monoid.getOperator(), phase );
	}

	template<
		Descriptor descr = descriptors::no_operation,
		class Monoid,
		typename OutputType, typename MaskType,
		typename InputType1, typename InputType2,
		typename Coords
	>
	RC eWiseApply(
		Vector< OutputType, nonblocking, Coords > &z,
		const Vector< MaskType, nonblocking, Coords > &mask,
		const InputType1 alpha,
		const InputType2 beta,
		const Monoid &monoid = Monoid(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			!grb::is_object< OutputType >::value &&
			!grb::is_object< MaskType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			grb::is_monoid< Monoid >::value, void
		>::type * const = nullptr
	) {
		// static checks
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< typename Monoid::D1, InputType1 >::value ), "grb::eWiseApply",
			"called with a left-hand input element type that does not match the "
			"first domain of the given monoid" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< typename Monoid::D2, InputType2 >::value ), "grb::eWiseApply",
			"called with a right-hand input element type that does not match the "
			"second domain of the given monoid" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< typename Monoid::D3, OutputType >::value ), "grb::eWiseApply",
			"called with an output element type that does not match the "
			"third domain of the given monoid" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< bool, MaskType >::value ), "grb::eWiseApply",
			"called with an output mask element type that is not Boolean " );
#ifdef _DEBUG
		std::cout << "In masked eWiseApply ([T1]<-T2<-T3), monoid variant\n";
#endif
		// simply delegate to operator variant
		return eWiseApply< descr >( z, mask, alpha, beta, monoid.getOperator(),
			phase );
	}

	template<
		Descriptor descr = descriptors::no_operation,
		class OP,
		typename OutputType,
		typename MaskType,
		typename InputType1,
		typename InputType2,
		typename Coords
	>
	RC eWiseApply(
		Vector< OutputType, nonblocking, Coords > &z,
		const Vector< MaskType, nonblocking, Coords > &mask,
		const Vector< InputType1, nonblocking, Coords > &x,
		const InputType2 beta,
		const OP &op = OP(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if< !grb::is_object< OutputType >::value &&
				!grb::is_object< MaskType >::value &&
				!grb::is_object< InputType1 >::value &&
				!grb::is_object< InputType2 >::value &&
				grb::is_operator< OP >::value,
			void
		>::type * const = nullptr
	) {
		// static checks
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< typename OP::D1, InputType1 >::value ), "grb::eWiseApply",
			"called with a left-hand input element type that does not match the "
			"first domain of the given operator" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< typename OP::D2, InputType2 >::value ), "grb::eWiseApply",
			"called with a right-hand input element type that does not match the "
			"second domain of the given operator" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< typename OP::D3, OutputType >::value ), "grb::eWiseApply",
			"called with an output element type that does not match the "
			"third domain of the given operator" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< bool, MaskType >::value ), "grb::eWiseApply",
			"called with an output mask element type that is not Boolean " );
#ifdef _DEBUG
		std::cout << "In masked eWiseApply ([T1]<-[T2]<-T3, using operator)\n";
#endif
		// check for empty mask
		if( size( mask ) == 0 ) {
			return eWiseApply< descr >( z, x, beta, op );
		}

		// other run-time checks
		const size_t n = internal::getCoordinates( z ).size();
		if( internal::getCoordinates( x ).size() != n ) {
			return MISMATCH;
		}
		if( internal::getCoordinates( mask ).size() != n ) {
			return MISMATCH;
		}

		if( phase == RESIZE ) {
			return SUCCESS;
		}
		assert( phase == EXECUTE );

		RC ret = SUCCESS;

		constexpr const bool dense_descr = descr & descriptors::dense;
		constexpr const bool dense_mask = dense_descr &&
			(descr & descriptors::structural) && !(descr & descriptors::invert_mask);

		internal::Pipeline::stage_type func =
			[&z, &mask, &x, beta, &op] (
				internal::Pipeline &pipeline,
				const size_t lower_bound, const size_t upper_bound
			) {
#ifdef _NONBLOCKING_DEBUG
				#pragma omp critical
				std::cout << "\t\tExecution of stage eWiseApply(z, mask, x, beta, "
					<< "operator) in the range(" << lower_bound << ", " << upper_bound << ")"
					<< std::endl;
#endif
				RC rc = SUCCESS;

				Coords local_mask, local_x, local_y, local_z;
				const size_t local_n = upper_bound - lower_bound;

#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
				const bool already_dense_vectors = dense_descr ||
					pipeline.allAlreadyDenseVectors();
#else
				constexpr const bool already_dense_vectors = dense_descr;
#endif

				const bool mask_is_dense = (descr & descriptors::structural) &&
					!(descr & descriptors::invert_mask) && already_dense_vectors;

				bool already_dense_mask = true;
				bool already_dense_input_x = true;

				size_t local_mask_nz = local_n;
				size_t local_x_nz = local_n;

				if( !mask_is_dense ) {
					local_z = internal::getCoordinates( z ).asyncSubset( lower_bound,
						upper_bound );
					if( dense_descr && local_z.nonzeroes() < local_n ) {
						return ILLEGAL;
					}
				}

				if( !already_dense_vectors ) {
#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
					already_dense_mask = pipeline.containsAlreadyDenseVector(
						&internal::getCoordinates( mask ) );
					if( !already_dense_mask ) {
#else
						already_dense_mask = false;
#endif
						local_mask = internal::getCoordinates( mask ).asyncSubset( lower_bound,
							upper_bound );
						local_mask_nz = local_mask.nonzeroes();
#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
					}

					already_dense_input_x = pipeline.containsAlreadyDenseVector(
						&internal::getCoordinates( x ) );
					if( !already_dense_input_x ) {
#else
						already_dense_mask = false;
#endif
						local_x = internal::getCoordinates( x ).asyncSubset( lower_bound,
							upper_bound );
						local_x_nz = local_x.nonzeroes();
#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
					}
#endif
				}

				const internal::Wrapper< false, InputType1, Coords > x_wrapper( x );
				const internal::Wrapper< true, InputType2, Coords > y_wrapper( beta );

				if( !mask_is_dense ) {
					// the output sparsity structure is implied by mask and descr
					local_z.local_clear();
					if( lower_bound == 0 ) {
						internal::getCoordinates( z ).reset_global_nnz_counter();
#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
						pipeline.markMaybeSparseVector( &internal::getCoordinates( z ) );
#endif
						if( dense_descr ) {
							pipeline.markMaybeSparseDenseDescriptorVerification(
								&internal::getCoordinates( z ) );
						}
					}
				}

				if(
					(descr & descriptors::dense) ||
					(local_x_nz == local_n) ||
					(local_mask_nz <= local_x_nz)
				) {
#ifdef GRB_BOOLEAN_DISPATCHER
					rc = internal::boolean_dispatcher_masked_apply_generic<
#else
					rc = internal::masked_apply_generic<
#endif
							false, true, false, false, descr, OP,
							OutputType, MaskType, InputType1, InputType2, Coords
						>(
							already_dense_mask, already_dense_input_x, true,
							lower_bound, upper_bound,
							local_z, local_mask, local_x, local_y,
							z, mask, x_wrapper, y_wrapper,
							op
						);
				} else {
#ifdef GRB_BOOLEAN_DISPATCHER
					rc = internal::boolean_dispatcher_sparse_apply_generic<
#else
					rc = internal::sparse_apply_generic<
#endif
							true, false, false, true, descr, OP,
							OutputType, bool, InputType1, InputType2, Coords
						>(
							already_dense_mask, already_dense_input_x, true,
							lower_bound, upper_bound,
							local_z, &local_mask, local_x, local_y,
							z, &mask, x_wrapper, y_wrapper,
							op
						);
				}

				if( !mask_is_dense ) {
					internal::getCoordinates( z ).asyncJoinSubset( local_z, lower_bound,
						upper_bound );
				}

				return rc;
			};

		ret = ret ? ret : internal::le.addStage(
				std::move( func ),
				internal::Opcode::BLAS1_MASKED_EWISEAPPLY,
				n, sizeof( OutputType ), dense_descr, dense_mask,
				&z, nullptr, &internal::getCoordinates( z ), nullptr,
				&x, &mask, nullptr, nullptr,
				&internal::getCoordinates( x ), &internal::getCoordinates( mask ),
				nullptr, nullptr,
				nullptr
			);

#ifdef _NONBLOCKING_DEBUG
		std::cout << "\t\tStage added to a pipeline: eWiseApply(z, mask, x, beta, "
			<< "operator)" << std::endl;
#endif
		return ret;
	}

	template<
		Descriptor descr = descriptors::no_operation,
		class Monoid,
		typename OutputType,
		typename InputType1,
		typename InputType2,
		typename Coords
	>
	RC eWiseApply(
		Vector< OutputType, nonblocking, Coords > &z,
		const Vector< InputType1, nonblocking, Coords > &x,
		const Vector< InputType2, nonblocking, Coords > &y,
		const Monoid &monoid = Monoid(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if< !grb::is_object< OutputType >::value &&
				!grb::is_object< InputType1 >::value &&
				!grb::is_object< InputType2 >::value &&
				grb::is_monoid< Monoid >::value,
			void
		>::type * const = nullptr
	) {
		// static checks
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< typename Monoid::D1, InputType1 >::value ), "grb::eWiseApply",
			"called with a left-hand input element type that does not match the "
			"first domain of the given monoid" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< typename Monoid::D2, InputType2 >::value ), "grb::eWiseApply",
			"called with a right-hand input element type that does not match the "
			"second domain of the given monoid" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< typename Monoid::D3, OutputType >::value ), "grb::eWiseApply",
			"called with an output element type that does not match the "
			"third domain of the given monoid" );
#ifdef _DEBUG
		std::cout << "In unmasked eWiseApply ([T1]<-[T2]<-[T3], using monoid)\n";
#endif
		// other run-time checks
		const size_t n = internal::getCoordinates( z ).size();
		if( internal::getCoordinates( x ).size() != n ) {
			return MISMATCH;
		}
		if( internal::getCoordinates( y ).size() != n ) {
			return MISMATCH;
		}

		if( phase == RESIZE ) {
			return SUCCESS;
		}
		assert( phase == EXECUTE );

		// check if we can dispatch to dense variant
		if( (descr & descriptors::dense) ) {
			return eWiseApply< descr >( z, x, y, monoid.getOperator() );
		}

		RC ret = SUCCESS;

		constexpr const bool dense_descr = descr & descriptors::dense;

		internal::Pipeline::stage_type func = [&z, &x, &y, &monoid, phase] (
			internal::Pipeline &pipeline,
			const size_t lower_bound, const size_t upper_bound
		) {
#ifdef _NONBLOCKING_DEBUG
			#pragma omp critical
			std::cout << "\t\tExecution of stage eWiseApply(z, x, y, monoid) in the "
				<< "range(" << lower_bound << ", " << upper_bound << ")" << std::endl;
#endif
			RC rc = SUCCESS;

			const Vector< bool, nonblocking, Coords > * const null_mask = nullptr;
			const Coords * const local_null_mask = nullptr;

			Coords local_x, local_y, local_z;

#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
			const bool already_dense_vectors = dense_descr ||
				pipeline.allAlreadyDenseVectors();
#else
			( void )pipeline;
			constexpr const bool already_dense_vectors = dense_descr;
#endif

			bool already_dense_input_x = true;
			bool already_dense_input_y = true;

			if( !already_dense_vectors ) {
				local_z = internal::getCoordinates( z ).asyncSubset( lower_bound,
					upper_bound );

#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
				already_dense_input_x = pipeline.containsAlreadyDenseVector(
					&internal::getCoordinates( x ) );
				if( !already_dense_input_x ) {
#else
					already_dense_input_x = false;
#endif
					local_x = internal::getCoordinates( x ).asyncSubset( lower_bound,
						upper_bound );
#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
				}

				already_dense_input_y = pipeline.containsAlreadyDenseVector(
					&internal::getCoordinates( y ) );
				if( !already_dense_input_y ) {
#else
					already_dense_input_y = false;
#endif
					local_y = internal::getCoordinates( y ).asyncSubset( lower_bound,
						upper_bound );
#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
				}
#endif
			}

			const internal::Wrapper< false, InputType1, Coords > x_wrapper( x );
			const internal::Wrapper< false, InputType2, Coords > y_wrapper( y );

			// we are in the unmasked sparse variant
			const auto op = monoid.getOperator();

			if( !already_dense_vectors ) {
				// z will have an a-priori unknown sparsity structure
				local_z.local_clear();
				if( lower_bound == 0 ) {
					internal::getCoordinates( z ).reset_global_nnz_counter();
#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
					pipeline.markMaybeSparseVector( &internal::getCoordinates( z ) );
#endif
				}
			}

#ifdef GRB_BOOLEAN_DISPATCHER
			rc = internal::boolean_dispatcher_sparse_apply_generic<
#else
			rc = internal::sparse_apply_generic<
#endif
					false, true, false, false, descr, typename Monoid::Operator,
					OutputType, bool, InputType1, InputType2, Coords
				>(
					true, already_dense_input_x, already_dense_input_y,
					lower_bound, upper_bound,
					local_z, local_null_mask, local_x, local_y,
					z, null_mask, x_wrapper, y_wrapper,
					op
				);

			if( !already_dense_vectors ) {
				internal::getCoordinates( z ).asyncJoinSubset( local_z, lower_bound,
					upper_bound );
			}

			return rc;
		};

		ret = ret ? ret : internal::le.addStage(
				std::move( func ),
				internal::Opcode::BLAS1_EWISEAPPLY,
				n, sizeof( OutputType ), dense_descr, true,
				&z, nullptr, &internal::getCoordinates( z ), nullptr,
				&x, &y, nullptr, nullptr,
				&internal::getCoordinates( x ), &internal::getCoordinates( y ),
				nullptr, nullptr,
				nullptr
			);

#ifdef _NONBLOCKING_DEBUG
		std::cout << "\t\tStage added to a pipeline: eWiseApply(z, x, y, monoid)"
			<< std::endl;
#endif
		return ret;
	}

	template<
		Descriptor descr = descriptors::no_operation,
		class Monoid,
		typename OutputType,
		typename InputType1,
		typename InputType2,
		typename Coords
	>
	RC eWiseApply(
		Vector< OutputType, nonblocking, Coords > &z,
		const InputType1 alpha,
		const Vector< InputType2, nonblocking, Coords > &y,
		const Monoid &monoid = Monoid(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if< !grb::is_object< OutputType >::value &&
				!grb::is_object< InputType1 >::value &&
				!grb::is_object< InputType2 >::value &&
				grb::is_monoid< Monoid >::value,
			void
		>::type * const = nullptr
	) {
		// static checks
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< typename Monoid::D1, InputType1 >::value ), "grb::eWiseApply",
			"called with a left-hand input element type that does not match the "
			"first domain of the given monoid" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< typename Monoid::D2, InputType2 >::value ), "grb::eWiseApply",
			"called with a right-hand input element type that does not match the "
			"second domain of the given monoid" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< typename Monoid::D3, OutputType >::value ), "grb::eWiseApply",
			"called with an output element type that does not match the "
			"third domain of the given monoid" );
#ifdef _DEBUG
		std::cout << "In unmasked eWiseApply ([T1]<-T2<-[T3], using monoid)\n";
#endif
		// other run-time checks
		const size_t n = internal::getCoordinates( z ).size();
		if( internal::getCoordinates( y ).size() != n ) {
			return MISMATCH;
		}

		if( phase == RESIZE ) {
			return SUCCESS;
		}
		assert( phase == EXECUTE );

		RC ret = SUCCESS;

		// check if we can dispatch to dense variant
		if( (descr & descriptors::dense) ) {
			return eWiseApply< descr >( z, alpha, y, monoid.getOperator() );
		}

		constexpr const bool dense_descr = descr & descriptors::dense;

		internal::Pipeline::stage_type func = [&z, alpha, &y, &monoid] (
			internal::Pipeline &pipeline,
			const size_t lower_bound, const size_t upper_bound
		) {
#ifdef _NONBLOCKING_DEBUG
			#pragma omp critical
			std::cout << "\t\tExecution of stage eWiseApply(z, alpha, y, monoid) in the "
				<< "range(" << lower_bound << ", " << upper_bound << ")" << std::endl;
#endif
			RC rc = SUCCESS;

			Coords local_x, local_y, local_z;

#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
			const bool already_dense_vectors = dense_descr ||
				pipeline.allAlreadyDenseVectors();
#else
			(void) pipeline;
			constexpr const bool already_dense_vectors = dense_descr;
#endif

#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
			bool already_dense_output = true;
#endif
			bool already_dense_input_y = true;

			// when it's guaranteed that the output will become dense
			// the only criterion to avoid reading the local coordinates is if it the
			// output is already dense
#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
			already_dense_output = pipeline.containsAlreadyDenseVector(
				&internal::getCoordinates( z ) );
			if( !already_dense_output ) {
#endif
				local_z = internal::getCoordinates( z ).asyncSubset( lower_bound,
					upper_bound );
#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
			}
#endif
			if( !already_dense_vectors ) {
#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
				already_dense_input_y = pipeline.containsAlreadyDenseVector(
					&internal::getCoordinates( y ) );
				if( !already_dense_input_y ) {
#else
					already_dense_input_y = false;
#endif
					local_y = internal::getCoordinates( y ).asyncSubset( lower_bound,
						upper_bound );
#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
				}
#endif
			}

			const internal::Wrapper< true, InputType1, Coords > x_wrapper( alpha );
			const internal::Wrapper< false, InputType2, Coords > y_wrapper( y );

			// we are in the unmasked sparse variant
			const auto &op = monoid.getOperator();

#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
			if( !already_dense_output ) {
#endif
				local_z.local_assignAllNotAlreadyAssigned();
#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
			}
#endif

			// dispatch to generic function
#ifdef GRB_BOOLEAN_DISPATCHER
			rc = internal::boolean_dispatcher_dense_apply_generic<
#else
			rc = internal::dense_apply_generic<
#endif
					true, false, false, true, descr, typename Monoid::Operator,
					OutputType, InputType1, InputType2, Coords
				>(
					true, already_dense_input_y,
					lower_bound, upper_bound,
					local_x, local_y,
					z, x_wrapper, y_wrapper, op
				);

#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
			if( !already_dense_output ) {
#else
			if( !already_dense_vectors ) {
#endif
				internal::getCoordinates( z ).asyncJoinSubset( local_z, lower_bound,
					upper_bound );
			}

			return rc;
		};

		ret = ret ? ret : internal::le.addStage(
				std::move( func ),
				internal::Opcode::BLAS1_EWISEAPPLY,
				n, sizeof( OutputType ), dense_descr, true,
				&z, nullptr, &internal::getCoordinates( z ), nullptr,
				&y, nullptr, nullptr, nullptr,
				&internal::getCoordinates( y ), nullptr, nullptr, nullptr,
				nullptr
			);

#ifdef _NONBLOCKING_DEBUG
		std::cout << "\t\tStage added to a pipeline: eWiseApply(z, alpha, y, monoid)"
			<< std::endl;
#endif
		return ret;
	}

	template<
		Descriptor descr = descriptors::no_operation,
		class Monoid,
		typename OutputType,
		typename InputType1,
		typename InputType2,
		typename Coords
	>
	RC eWiseApply(
		Vector< OutputType, nonblocking, Coords > &z,
		const Vector< InputType1, nonblocking, Coords > &x,
		const InputType2 beta,
		const Monoid &monoid = Monoid(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if< !grb::is_object< OutputType >::value &&
				!grb::is_object< InputType1 >::value &&
				!grb::is_object< InputType2 >::value &&
				grb::is_monoid< Monoid >::value,
			void >::type * const = nullptr
	) {
		// static checks
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< typename Monoid::D1, InputType1 >::value ), "grb::eWiseApply",
			"called with a left-hand input element type that does not match the "
			"first domain of the given monoid" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< typename Monoid::D2, InputType2 >::value ), "grb::eWiseApply",
			"called with a right-hand input element type that does not match the "
			"second domain of the given monoid" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< typename Monoid::D3, OutputType >::value ), "grb::eWiseApply",
			"called with an output element type that does not match the "
			"third domain of the given monoid" );
#ifdef _DEBUG
		std::cout << "In unmasked eWiseApply ([T1]<-[T2]<-T3, using monoid)\n";
#endif
		// other run-time checks
		const size_t n = internal::getCoordinates( z ).size();
		if( internal::getCoordinates( x ).size() != n ) {
			return MISMATCH;
		}

		if( phase == RESIZE ) {
			return SUCCESS;
		}
		assert( phase == EXECUTE );

		// check if we can dispatch to dense variant
		if( (descr & descriptors::dense) ) {
			return eWiseApply< descr >( z, x, beta, monoid.getOperator() );
		}

		RC ret = SUCCESS;

		constexpr const bool dense_descr = descr & descriptors::dense;

		internal::Pipeline::stage_type func = [&z, &x, beta, &monoid] (
			internal::Pipeline &pipeline,
			const size_t lower_bound, const size_t upper_bound
		) {
#ifdef _NONBLOCKING_DEBUG
			#pragma omp critical
			std::cout << "\t\tExecution of stage eWiseApply(z, x, beta, monoid) in the "
				<< "range(" << lower_bound << ", " << upper_bound << ")" << std::endl;
#endif
			RC rc = SUCCESS;

			Coords local_x, local_y, local_z;

#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
			const bool already_dense_vectors = dense_descr ||
				pipeline.allAlreadyDenseVectors();
#else
			(void) pipeline;
			constexpr const bool already_dense_vectors = dense_descr;
#endif

#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
			bool already_dense_output = true;
#endif
			bool already_dense_input_x = true;

#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
			already_dense_output = pipeline.containsAlreadyDenseVector(
				&internal::getCoordinates( z ) );
			if( !already_dense_output ) {
#endif
				local_z = internal::getCoordinates( z ).asyncSubset( lower_bound,
					upper_bound );
#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
			}
#endif

			if( !already_dense_vectors ) {
#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
				already_dense_input_x = pipeline.containsAlreadyDenseVector(
					&internal::getCoordinates( x ) );
				if( !already_dense_input_x ) {
#else
					already_dense_input_x = false;
#endif
					local_x = internal::getCoordinates( x ).asyncSubset( lower_bound,
						upper_bound );
#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
				}
#endif
			}

			const internal::Wrapper< false, InputType1, Coords > x_wrapper( x );
			const internal::Wrapper< true, InputType2, Coords > y_wrapper( beta );

			// we are in the unmasked sparse variant
			const auto &op = monoid.getOperator();

#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
			if( !already_dense_output ) {
#endif
				// the result will always be dense
				local_z.local_assignAllNotAlreadyAssigned();
#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
			}
#endif

			// dispatch
#ifdef GRB_BOOLEAN_DISPATCHER
			rc = internal::boolean_dispatcher_dense_apply_generic<
#else
			rc = internal::dense_apply_generic<
#endif
					false, true, true, false, descr, typename Monoid::Operator,
					OutputType, InputType1, InputType2, Coords
				>(
					already_dense_input_x, true,
					lower_bound, upper_bound,
					local_x, local_y,
					z, x_wrapper, y_wrapper,
					op
				);

#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
			if( !already_dense_output ) {
#else

			if( !already_dense_vectors ) {
#endif
				internal::getCoordinates( z ).asyncJoinSubset( local_z, lower_bound,
					upper_bound );
			}

			return rc;
		};

		ret = ret ? ret : internal::le.addStage(
				std::move( func ),
				internal::Opcode::BLAS1_EWISEAPPLY,
				n, sizeof( OutputType ), dense_descr, true,
				&z, nullptr, &internal::getCoordinates( z ), nullptr,
				&x, nullptr, nullptr, nullptr,
				&internal::getCoordinates( x ), nullptr, nullptr, nullptr,
				nullptr
			);

#ifdef _NONBLOCKING_DEBUG
		std::cout << "\t\tStage added to a pipeline: eWiseApply(z, x, beta, monoid)"
			<< std::endl;
#endif
		return ret;
	}

	template<
		Descriptor descr = descriptors::no_operation,
		class Monoid,
		typename OutputType,
		typename MaskType,
		typename InputType1,
		typename InputType2,
		typename Coords
	>
	RC eWiseApply(
		Vector< OutputType, nonblocking, Coords > &z,
		const Vector< MaskType, nonblocking, Coords > &mask,
		const Vector< InputType1, nonblocking, Coords > &x,
		const Vector< InputType2, nonblocking, Coords > &y,
		const Monoid &monoid = Monoid(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
				!grb::is_object< OutputType >::value &&
				!grb::is_object< MaskType >::value &&
				!grb::is_object< InputType1 >::value &&
				!grb::is_object< InputType2 >::value &&
				grb::is_monoid< Monoid >::value,
			void
		>::type * const = nullptr
	) {
		// static checks
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< typename Monoid::D1, InputType1 >::value ), "grb::eWiseApply",
			"called with a left-hand input element type that does not match the "
			"first domain of the given monoid" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< typename Monoid::D2, InputType2 >::value ), "grb::eWiseApply",
			"called with a right-hand input element type that does not match the "
			"second domain of the given monoid" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< typename Monoid::D3, OutputType >::value ), "grb::eWiseApply",
			"called with an output element type that does not match the "
			"third domain of the given monoid" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< bool, MaskType >::value ), "grb::eWiseApply",
			"called with an output mask element type that is not Boolean " );
#ifdef _DEBUG
		std::cout << "In masked eWiseApply ([T1]<-[T2]<-[T3], using monoid)\n";
#endif
		if( size( mask ) == 0 ) {
			return eWiseApply< descr >( z, x, y, monoid, phase );
		}

		// other run-time checks
		const size_t n = internal::getCoordinates( z ).size();
		if( internal::getCoordinates( x ).size() != n ) {
			return MISMATCH;
		}
		if( internal::getCoordinates( y ).size() != n ) {
			return MISMATCH;
		}
		if( internal::getCoordinates( mask ).size() != n ) {
			return MISMATCH;
		}

		if( phase == RESIZE ) {
			return SUCCESS;
		}
		assert( phase == EXECUTE );

		// check if we can dispatch to dense variant
		if( (descr & descriptors::dense) ) {
			return eWiseApply< descr >( z, mask, x, y, monoid.getOperator() );
		}

		RC ret = SUCCESS;

		constexpr const bool dense_descr = descr & descriptors::dense;
		constexpr const bool dense_mask = dense_descr &&
			(descr & descriptors::structural) && !(descr & descriptors::invert_mask);

		internal::Pipeline::stage_type func = [&z, &mask, &x, &y, &monoid] (
			internal::Pipeline &pipeline,
			const size_t lower_bound, const size_t upper_bound
		) {
#ifdef _NONBLOCKING_DEBUG
			#pragma omp critical
			std::cout << "\t\tExecution of stage eWiseApply(z, mask, x, y, monoid) in "
				<< "the range(" << lower_bound << ", " << upper_bound << ")" << std::endl;
#endif
			RC rc = SUCCESS;

			Coords local_mask, local_x, local_y, local_z;
			const size_t local_n = upper_bound - lower_bound;
			size_t local_mask_nz = local_n;
			size_t local_x_nz = local_n;
			size_t local_y_nz = local_n;

#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
			const bool already_dense_vectors = dense_descr ||
				pipeline.allAlreadyDenseVectors();
#else
			constexpr const bool already_dense_vectors = dense_descr;
#endif

			const bool mask_is_dense = (descr & descriptors::structural) &&
				!(descr & descriptors::invert_mask) && already_dense_vectors;

			bool already_dense_mask = true;
			bool already_dense_input_x = true;
			bool already_dense_input_y = true;

			if( !mask_is_dense ) {
				local_z = internal::getCoordinates( z ).asyncSubset( lower_bound,
					upper_bound );
				if( dense_descr && local_z.nonzeroes() < local_n ) {
					return ILLEGAL;
				}
			}

			if( !already_dense_vectors ) {
#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
				already_dense_mask = pipeline.containsAlreadyDenseVector(
					&internal::getCoordinates( mask ) );
				if( !already_dense_mask ) {
#else
					already_dense_mask = false;
#endif
					local_mask = internal::getCoordinates( mask ).asyncSubset( lower_bound,
						upper_bound );
					local_mask_nz = local_mask.nonzeroes();
#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
				}

				already_dense_input_x = pipeline.containsAlreadyDenseVector(
					&internal::getCoordinates( x ) );
				if( !already_dense_input_x ) {
#else
					already_dense_input_x = false;
#endif
					local_x = internal::getCoordinates( x ).asyncSubset( lower_bound,
						upper_bound );
					local_x_nz = local_x.nonzeroes();
#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
				}

				already_dense_input_y = pipeline.containsAlreadyDenseVector(
					&internal::getCoordinates( y ) );
				if( !already_dense_input_y ) {
#else
					already_dense_input_y = false;
#endif
					local_y = internal::getCoordinates( y ).asyncSubset( lower_bound,
						upper_bound );
					local_y_nz = local_y.nonzeroes();
#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
				}
#endif
			}

			const internal::Wrapper< false, InputType1, Coords > x_wrapper( x );
			const internal::Wrapper< false, InputType2, Coords > y_wrapper( y );

			// we are in the masked sparse variant
			const InputType1 left_identity = monoid.template getIdentity< InputType1 >();
			const InputType2 right_identity =
				monoid.template getIdentity< InputType2 >();
			const auto &op = monoid.getOperator();

			if( !mask_is_dense ) {
				// z will have an a priori unknown sparsity structure
				local_z.local_clear();
				if( lower_bound == 0 ) {
					internal::getCoordinates( z ).reset_global_nnz_counter();
#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
					pipeline.markMaybeSparseVector( &internal::getCoordinates( z ) );
#endif
					if( dense_descr ) {
						pipeline.markMaybeSparseDenseDescriptorVerification(
							&internal::getCoordinates( z ) );
					}
				}
			}

			if( local_x_nz < local_n &&
				local_y_nz < local_n &&
				local_x_nz + local_y_nz < local_mask_nz
			) {
#ifdef GRB_BOOLEAN_DISPATCHER
				rc = internal::boolean_dispatcher_sparse_apply_generic<
#else
				rc = internal::sparse_apply_generic<
#endif
						true, true, false, false, descr, typename Monoid::Operator,
						OutputType, bool, InputType1, InputType2, Coords
					>(
						already_dense_mask, already_dense_input_x, already_dense_input_y,
						lower_bound, upper_bound,
						local_z, &local_mask, local_x, local_y,
						z, &mask, x_wrapper, y_wrapper,
						op
					);
			} else if( local_x_nz < local_n && local_y_nz == local_n ) {
#ifdef GRB_BOOLEAN_DISPATCHER
				rc = internal::boolean_dispatcher_masked_apply_generic<
#else
				rc = internal::masked_apply_generic<
#endif
						false, false, true, false, descr, typename Monoid::Operator,
						OutputType, MaskType, InputType1, InputType2, Coords
					>(
						already_dense_mask, already_dense_input_x, already_dense_input_y,
						lower_bound, upper_bound,
						local_z, local_mask, local_x, local_y,
						z, mask, x_wrapper, y_wrapper,
						op,
						&left_identity, nullptr
					);
			} else if( local_y_nz < local_n && local_x_nz == local_n ) {
#ifdef GRB_BOOLEAN_DISPATCHER
				rc = internal::boolean_dispatcher_masked_apply_generic<
#else
				rc = internal::masked_apply_generic<
#endif
						false, false, false, true, descr, typename Monoid::Operator,
						OutputType, MaskType, InputType1, InputType2, Coords
					>(
						already_dense_mask, already_dense_input_x, already_dense_input_y,
						lower_bound, upper_bound,
						local_z, local_mask, local_x, local_y,
						z, mask, x_wrapper, y_wrapper,
						op,
						nullptr, &right_identity
					);
			} else {
#ifdef GRB_BOOLEAN_DISPATCHER
				rc = internal::boolean_dispatcher_masked_apply_generic<
#else
				rc = internal::masked_apply_generic<
#endif
						false, false, true, true, descr, typename Monoid::Operator,
						OutputType, MaskType, InputType1, InputType2, Coords
					>(
						already_dense_mask, already_dense_input_x, already_dense_input_y,
						lower_bound, upper_bound,
						local_z, local_mask, local_x, local_y,
						z, mask, x_wrapper, y_wrapper,
						op,
						&left_identity, &right_identity
					);
			}

			if( !mask_is_dense ) {
				internal::getCoordinates( z ).asyncJoinSubset( local_z, lower_bound,
					upper_bound );
			}

			return rc;
		};

		ret = ret ? ret : internal::le.addStage(
				std::move( func ),
				internal::Opcode::BLAS1_MASKED_EWISEAPPLY,
				n, sizeof( OutputType ), dense_descr, dense_mask,
				&z, nullptr, &internal::getCoordinates( z ), nullptr,
				&x, &y, &mask, nullptr,
				&internal::getCoordinates( x ), &internal::getCoordinates( y ),
				&internal::getCoordinates( mask ), nullptr,
				nullptr
			);

#ifdef _NONBLOCKING_DEBUG
		std::cout << "\t\tStage added to a pipeline: eWiseApply(z, mask, x, y, "
			<< "monoid)" << std::endl;
#endif
		return ret;
	}

	template<
		Descriptor descr = descriptors::no_operation,
		class Monoid,
		typename OutputType,
		typename MaskType,
		typename InputType1,
		typename InputType2,
		typename Coords
	>
	RC eWiseApply(
		Vector< OutputType, nonblocking, Coords > &z,
		const Vector< MaskType, nonblocking, Coords > &mask,
		const InputType1 alpha,
		const Vector< InputType2, nonblocking, Coords > &y,
		const Monoid &monoid = Monoid(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if< !grb::is_object< OutputType >::value &&
				!grb::is_object< MaskType >::value &&
				!grb::is_object< InputType1 >::value &&
				!grb::is_object< InputType2 >::value &&
				grb::is_monoid< Monoid >::value,
			void >::type * const = nullptr
	) {
		// static checks
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< typename Monoid::D1, InputType1 >::value ), "grb::eWiseApply",
			"called with a left-hand input element type that does not match the "
			"first domain of the given monoid" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< typename Monoid::D2, InputType2 >::value ), "grb::eWiseApply",
			"called with a right-hand input element type that does not match the "
			"second domain of the given monoid" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< typename Monoid::D3, OutputType >::value ), "grb::eWiseApply",
			"called with an output element type that does not match the "
			"third domain of the given monoid" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< bool, MaskType >::value ), "grb::eWiseApply",
			"called with an output mask element type that is not Boolean " );
#ifdef _DEBUG
		std::cout << "In masked eWiseApply ([T1]<-T2<-[T3], using monoid)\n";
#endif
		if( size( mask ) == 0 ) {
			return eWiseApply< descr >( z, alpha, y, monoid );
		}

		// other run-time checks
		const size_t n = internal::getCoordinates( z ).size();
		if( internal::getCoordinates( y ).size() != n ) {
			return MISMATCH;
		}
		if( internal::getCoordinates( mask ).size() != n ) {
			return MISMATCH;
		}

		if( phase == RESIZE ) {
			return SUCCESS;
		}
		assert( phase == EXECUTE );

		// check if we can dispatch to dense variant
		if( descr & descriptors::dense ) {
			return eWiseApply< descr >( z, mask, alpha, y, monoid.getOperator() );
		}

		RC ret = SUCCESS;

		constexpr const bool dense_descr = descr & descriptors::dense;
		constexpr const bool dense_mask = dense_descr &&
			(descr & descriptors::structural) && !(descr & descriptors::invert_mask);

		internal::Pipeline::stage_type func = [&z, &mask, alpha, &y, &monoid] (
			internal::Pipeline &pipeline,
			const size_t lower_bound, const size_t upper_bound
		) {
#ifdef _NONBLOCKING_DEBUG
			#pragma omp critical
			std::cout << "\t\tExecution of stage eWiseApply(z, mask, alpha, y, monoid) "
				<< "in the range(" << lower_bound << ", " << upper_bound << ")"
				<< std::endl;
#endif
			RC rc = SUCCESS;

			Coords local_mask, local_x, local_y, local_z;
			const size_t local_n = upper_bound - lower_bound;

#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
			const bool already_dense_vectors = dense_descr ||
				pipeline.allAlreadyDenseVectors();
#else
			constexpr const bool already_dense_vectors = dense_descr;
#endif

			const bool mask_is_dense = (descr & descriptors::structural) &&
				!(descr & descriptors::invert_mask) && already_dense_vectors;

			bool already_dense_mask = true;
			bool already_dense_input_y = true;

			if( !mask_is_dense ) {
				local_z = internal::getCoordinates( z ).asyncSubset( lower_bound,
					upper_bound );
				if( dense_descr && local_z.nonzeroes() < local_n ) {
					return ILLEGAL;
				}
			}

			if( !already_dense_vectors ) {
#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
				already_dense_mask = pipeline.containsAlreadyDenseVector(
					&internal::getCoordinates( mask ) );
				if( !already_dense_mask ) {
#else
					already_dense_mask = false;
#endif
					local_mask = internal::getCoordinates( mask ).asyncSubset( lower_bound,
						upper_bound );
#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
				}

				already_dense_input_y = pipeline.containsAlreadyDenseVector(
					&internal::getCoordinates( y ) );
				if( !already_dense_input_y ) {
#else
					already_dense_input_y = false;
#endif
					local_y = internal::getCoordinates( y ).asyncSubset( lower_bound,
						upper_bound );
#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
				}
#endif
			}

			const internal::Wrapper< true, InputType1, Coords > x_wrapper( alpha );
			const internal::Wrapper< false, InputType2, Coords > y_wrapper( y );

			// we are in the masked sparse variant
			const InputType2 right_identity =
				monoid.template getIdentity< InputType2 >();
			const auto &op = monoid.getOperator();

			if( !mask_is_dense ) {
				// the sparsity structure of z will be a result of the given mask and descr
				local_z.local_clear();
				if( lower_bound == 0 ) {
					internal::getCoordinates( z ).reset_global_nnz_counter();
#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
					pipeline.markMaybeSparseVector( &internal::getCoordinates( z ) );
#endif
					if( dense_descr ) {
						pipeline.markMaybeSparseDenseDescriptorVerification(
							&internal::getCoordinates( z ) );
					}
				}
			}

#ifdef GRB_BOOLEAN_DISPATCHER
			rc = internal::boolean_dispatcher_masked_apply_generic<
#else
			rc = internal::masked_apply_generic<
#endif
					true, false, false, true, descr, typename Monoid::Operator,
					OutputType, MaskType, InputType1, InputType2, Coords
				>(
					already_dense_mask, true, already_dense_input_y,
					lower_bound, upper_bound,
					local_z, local_mask, local_x, local_y,
					z, mask, x_wrapper, y_wrapper,
					op,
					nullptr, &right_identity
				);

			if( !mask_is_dense ) {
				internal::getCoordinates( z ).asyncJoinSubset( local_z, lower_bound,
					upper_bound );
			}

			return rc;
		};

		ret = ret ? ret : internal::le.addStage(
				std::move( func ),
				internal::Opcode::BLAS1_MASKED_EWISEAPPLY,
				n, sizeof( OutputType ), dense_descr, dense_mask,
				&z, nullptr, &internal::getCoordinates( z ), nullptr,
				&y, &mask, nullptr, nullptr,
				&internal::getCoordinates( y ), &internal::getCoordinates( mask ),
				nullptr, nullptr,
				nullptr
			);

#ifdef _NONBLOCKING_DEBUG
		std::cout << "\t\tStage added to a pipeline: eWiseApply(z, mask, alpha, y, "
			<< "monoid)" << std::endl;
#endif
		return ret;
	}

	template<
		Descriptor descr = descriptors::no_operation,
		class Monoid,
		typename OutputType,
		typename MaskType,
		typename InputType1,
		typename InputType2,
		typename Coords
	>
	RC eWiseApply(
		Vector< OutputType, nonblocking, Coords > &z,
		const Vector< MaskType, nonblocking, Coords > &mask,
		const Vector< InputType1, nonblocking, Coords > &x,
		const InputType2 beta,
		const Monoid &monoid = Monoid(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if< !grb::is_object< OutputType >::value &&
				!grb::is_object< MaskType >::value &&
				!grb::is_object< InputType1 >::value &&
				!grb::is_object< InputType2 >::value &&
				grb::is_monoid< Monoid >::value,
			void
		>::type * const = nullptr
	) {
		// static checks
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< typename Monoid::D1, InputType1 >::value ), "grb::eWiseApply",
			"called with a left-hand input element type that does not match the "
			"first domain of the given monoid" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< typename Monoid::D2, InputType2 >::value ), "grb::eWiseApply",
			"called with a right-hand input element type that does not match the "
			"second domain of the given monoid" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< typename Monoid::D3, OutputType >::value ), "grb::eWiseApply",
			"called with an output element type that does not match the "
			"third domain of the given monoid" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< bool, MaskType >::value ), "grb::eWiseApply",
			"called with an output mask element type that is not Boolean " );
#ifdef _DEBUG
		std::cout << "In masked eWiseApply ([T1]<-[T2]<-T3, using monoid)\n";
#endif
		if( size( mask ) == 0 ) {
			return eWiseApply< descr >( z, x, beta, monoid );
		}

		// other run-time checks
		const size_t n = internal::getCoordinates( z ).size();
		if( internal::getCoordinates( x ).size() != n ) {
			return MISMATCH;
		}
		if( internal::getCoordinates( mask ).size() != n ) {
			return MISMATCH;
		}

		if( phase == RESIZE ) {
			return SUCCESS;
		}
		assert( phase == EXECUTE );

		// check if we can dispatch to dense variant
		if( (descr & descriptors::dense) ) {
			return eWiseApply< descr >( z, mask, x, beta, monoid.getOperator() );
		}

		RC ret = SUCCESS;

		constexpr const bool dense_descr = descr & descriptors::dense;
		constexpr const bool dense_mask = dense_descr &&
			(descr & descriptors::structural) && !(descr & descriptors::invert_mask);

		internal::Pipeline::stage_type func = [&z, &mask, &x, beta, &monoid] (
			internal::Pipeline &pipeline,
			const size_t lower_bound, const size_t upper_bound
		) {
#ifdef _NONBLOCKING_DEBUG
			#pragma omp critical
			std::cout << "\t\tExecution of stage eWiseApply(z, mask, x, beta, monoid) "
				<< "in the range(" << lower_bound << ", " << upper_bound << ")"
				<< std::endl;
#endif
			RC rc = SUCCESS;

			Coords local_mask, local_x, local_y, local_z;
			const size_t local_n = upper_bound - lower_bound;

#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
			const bool already_dense_vectors = dense_descr ||
				pipeline.allAlreadyDenseVectors();
#else
			constexpr const bool already_dense_vectors = dense_descr;
#endif

			const bool mask_is_dense = (descr & descriptors::structural) &&
				!(descr & descriptors::invert_mask) && already_dense_vectors;

			bool already_dense_mask = true;
			bool already_dense_input_x = true;

			if( !mask_is_dense ) {
				local_z = internal::getCoordinates( z ).asyncSubset( lower_bound,
					upper_bound );
				if( dense_descr && local_z.nonzeroes() < local_n ) {
					return ILLEGAL;
				}
			}

			if( !already_dense_vectors ) {
#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
				already_dense_mask = pipeline.containsAlreadyDenseVector(
					&internal::getCoordinates( mask ) );
				if( !already_dense_mask ) {
#else
				already_dense_mask = false;
#endif
					local_mask = internal::getCoordinates( mask ).asyncSubset( lower_bound,
						upper_bound );
#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
				}

				already_dense_input_x = pipeline.containsAlreadyDenseVector(
					&internal::getCoordinates( x ) );
				if( !already_dense_input_x ) {
#else
					already_dense_input_x = false;
#endif
					local_x = internal::getCoordinates( x ).asyncSubset( lower_bound,
						upper_bound );
#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
				}
#endif
			}

			const internal::Wrapper< false, InputType1, Coords > x_wrapper( x );
			const internal::Wrapper< true, InputType2, Coords > y_wrapper( beta );

			// we are in the masked sparse variant
			const InputType1 left_identity = monoid.template getIdentity< InputType1 >();
			const auto &op = monoid.getOperator();

			if( !mask_is_dense ) {
				local_z.local_clear();
				if( lower_bound == 0 ) {
					internal::getCoordinates( z ).reset_global_nnz_counter();
#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
					pipeline.markMaybeSparseVector( &internal::getCoordinates( z ) );
#endif
					if( dense_descr ) {
						pipeline.markMaybeSparseDenseDescriptorVerification(
							&internal::getCoordinates( z ) );
					}
				}
			}

#ifdef GRB_BOOLEAN_DISPATCHER
			rc = internal::boolean_dispatcher_masked_apply_generic<
#else
			rc = internal::masked_apply_generic<
#endif
					false, true, true, false, descr, typename Monoid::Operator,
					OutputType, MaskType, InputType1, InputType2, Coords
				>(
					already_dense_mask, already_dense_input_x, true,
					lower_bound, upper_bound,
					local_z, local_mask, local_x, local_y,
					z, mask, x_wrapper, y_wrapper,
					op,
					&left_identity
				);

			if( !mask_is_dense ) {
				internal::getCoordinates( z ).asyncJoinSubset( local_z, lower_bound,
					upper_bound );
			}

			return rc;
		};

		ret = ret ? ret : internal::le.addStage(
				std::move( func ),
				internal::Opcode::BLAS1_MASKED_EWISEAPPLY,
				n, sizeof( OutputType ), dense_descr, dense_mask,
				&z, nullptr, &internal::getCoordinates( z ), nullptr,
				&x, &mask, nullptr, nullptr,
				&internal::getCoordinates( x ), &internal::getCoordinates( mask ),
				nullptr, nullptr,
				nullptr
			);

#ifdef _NONBLOCKING_DEBUG
		std::cout << "\t\tStage added to a pipeline: eWiseApply(z, mask, x, beta, "
			<< "monoid)" << std::endl;
#endif
		return ret;
	}

	template<
		Descriptor descr = descriptors::no_operation,
		class OP,
		typename OutputType,
		typename InputType1,
		typename InputType2,
		typename Coords
	>
	RC eWiseApply(
		Vector< OutputType, nonblocking, Coords > &z,
		const InputType1 alpha,
		const Vector< InputType2, nonblocking, Coords > &y,
		const OP &op = OP(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
				!grb::is_object< OutputType >::value &&
				!grb::is_object< InputType1 >::value &&
				!grb::is_object< InputType2 >::value &&
				grb::is_operator< OP >::value,
			void
		>::type * const = nullptr
	) {
		// static checks
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< typename OP::D1, InputType1 >::value ), "grb::eWiseApply",
			"called with a left-hand input element type that does not match the "
			"first domain of the given operator" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< typename OP::D2, InputType2 >::value ), "grb::eWiseApply",
			"called with a right-hand input element type that does not match the "
			"second domain of the given operator" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< typename OP::D3, OutputType >::value ), "grb::eWiseApply",
			"called with an output element type that does not match the "
			"third domain of the given operator" );
#ifdef _DEBUG
		std::cout << "In eWiseApply ([T1]<-T2<-[T3]), operator variant\n";
#endif
		// sanity check
		const size_t n = internal::getCoordinates( z ).size();
		if( internal::getCoordinates( y ).size() != n ) {
			return MISMATCH;
		}

		if( phase == RESIZE ) {
			return SUCCESS;
		}
		assert( phase == EXECUTE );

		// check if we can dispatch
		if( static_cast< const void * >( &z ) ==
			static_cast< const void * >( &y )
		) {
			return foldr< descr >( alpha, z, op );
		}

		RC ret = SUCCESS;

		constexpr const bool dense_descr = descr & descriptors::dense;

		internal::Pipeline::stage_type func = [&z, alpha, &y, &op] (
			internal::Pipeline &pipeline,
			const size_t lower_bound, const size_t upper_bound
		) {
#ifdef _NONBLOCKING_DEBUG
			#pragma omp critical
			std::cout << "\t\tExecution of stage eWiseApply(z, alpha, y, operator) in "
				<< "the range(" << lower_bound << ", " << upper_bound << ")" << std::endl;
#endif
			RC rc = SUCCESS;

			const Vector< bool, nonblocking, Coords > * const null_mask = nullptr;
			const Coords * const local_null_mask = nullptr;

			Coords local_mask, local_x, local_y, local_z;
			const size_t local_n = upper_bound - lower_bound;
			size_t local_y_nz = local_n;

#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
			const bool already_dense_vectors = dense_descr ||
				pipeline.allAlreadyDenseVectors();
#else
			(void) pipeline;
			constexpr const bool already_dense_vectors = dense_descr;
#endif

			bool already_dense_input_y = true;

			if( !already_dense_vectors ) {
				local_z = internal::getCoordinates( z ).asyncSubset( lower_bound,
					upper_bound );

#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
				already_dense_input_y = pipeline.containsAlreadyDenseVector(
					&internal::getCoordinates( y ) );
				if( !already_dense_input_y ) {
#else
					already_dense_input_y = false;
#endif
					local_y = internal::getCoordinates( y ).asyncSubset( lower_bound,
						upper_bound );
					local_y_nz = local_y.nonzeroes();
#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
				}
#endif
			}

			const internal::Wrapper< true, InputType1, Coords > x_wrapper( alpha );
			const internal::Wrapper< false, InputType2, Coords > y_wrapper( y );

			if( !already_dense_vectors ) {
				if( lower_bound == 0 ) {
					internal::getCoordinates( z ).reset_global_nnz_counter();
				}
			}

			// check for dense variant
			if( (descr & descriptors::dense) || local_y_nz == local_n ) {
				if( !already_dense_vectors ) {
					local_z.local_assignAll( );
				}

#ifdef GRB_BOOLEAN_DISPATCHER
				rc = internal::boolean_dispatcher_dense_apply_generic<
#else
				rc = internal::dense_apply_generic<
#endif
						true, false, false, false, descr, OP,
						OutputType, InputType1, InputType2, Coords
					>(
						true, already_dense_input_y,
						lower_bound, upper_bound,
						local_x, local_y, z,
						x_wrapper, y_wrapper,
						op
					);
			} else {
				if( !already_dense_vectors ) {
					local_z.local_clear();
#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
					pipeline.markMaybeSparseVector( &internal::getCoordinates( z ) );
#endif
				}

				// we are in the sparse variant
#ifdef GRB_BOOLEAN_DISPATCHER
				rc = internal::boolean_dispatcher_sparse_apply_generic<
						false, false, true, false, descr, OP,
#else
				rc = internal::sparse_apply_generic<
						false, false, true, false, descr, OP,
#endif
						OutputType, bool, InputType1, InputType2, Coords
					>(
						true, true, already_dense_input_y,
						lower_bound, upper_bound,
						local_z, local_null_mask, local_x, local_y,
						z, null_mask, x_wrapper, y_wrapper,
						op
					);
			}

			if( !already_dense_vectors ) {
				internal::getCoordinates( z ).asyncJoinSubset( local_z, lower_bound,
					upper_bound );
			}

			return rc;
		};

		ret = ret ? ret : internal::le.addStage(
				std::move( func ),
				internal::Opcode::BLAS1_EWISEAPPLY,
				n, sizeof( OutputType ), dense_descr, true,
				&z, nullptr, &internal::getCoordinates( z ), nullptr,
				&y, nullptr, nullptr, nullptr,
				&internal::getCoordinates( y ), nullptr, nullptr, nullptr,
				nullptr
			);

#ifdef _NONBLOCKING_DEBUG
		std::cout << "\t\tStage added to a pipeline: eWiseApply(z, alpha, y, "
			<< "operator)" << std::endl;
#endif
		return ret;
	}

	template<
		Descriptor descr = descriptors::no_operation,
		class OP,
		typename OutputType,
		typename MaskType,
		typename InputType1,
		typename InputType2,
		typename Coords
	>
	RC eWiseApply(
		Vector< OutputType, nonblocking, Coords > &z,
		const Vector< MaskType, nonblocking, Coords > &mask,
		const InputType1 alpha,
		const Vector< InputType2, nonblocking, Coords > &y,
		const OP &op = OP(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
				!grb::is_object< OutputType >::value &&
				!grb::is_object< MaskType >::value &&
				!grb::is_object< InputType1 >::value &&
				!grb::is_object< InputType2 >::value &&
				grb::is_operator< OP >::value,
			void
		>::type * const = nullptr
	) {
		// static checks
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< typename OP::D1, InputType1 >::value ), "grb::eWiseApply",
			"called with a left-hand input element type that does not match the "
			"first domain of the given operator" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< typename OP::D2, InputType2 >::value ), "grb::eWiseApply",
			"called with a right-hand input element type that does not match the "
			"second domain of the given operator" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< typename OP::D3, OutputType >::value ), "grb::eWiseApply",
			"called with an output element type that does not match the "
			"third domain of the given operator" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< bool, MaskType >::value ), "grb::eWiseApply",
			"called with an output mask element type that is not Boolean " );
#ifdef _DEBUG
		std::cout << "In masked eWiseApply ([T1]<-T2<-[T3], operator variant)\n";
#endif
		// check for empty mask
		if( size( mask ) == 0 ) {
			return eWiseApply< descr >( z, alpha, y, op );
		}

		// sanity check
		const size_t n = internal::getCoordinates( z ).size();
		if( internal::getCoordinates( y ).size() != n ) {
			return MISMATCH;
		}
		if( internal::getCoordinates( mask ).size() != n ) {
			return MISMATCH;
		}

		if( phase == RESIZE ) {
			return SUCCESS;
		}
		assert( phase == EXECUTE );

		RC ret = SUCCESS;

		constexpr const bool dense_descr = descr & descriptors::dense;
		constexpr const bool dense_mask = dense_descr &&
			(descr & descriptors::structural) && !(descr & descriptors::invert_mask);

		internal::Pipeline::stage_type func = [&z, &mask, alpha, &y, &op] (
			internal::Pipeline &pipeline,
			const size_t lower_bound, const size_t upper_bound
		) {
#ifdef _NONBLOCKING_DEBUG
			#pragma omp critical
			std::cout << "\t\tExecution of stage eWiseApply(z, mask, alpha, y, "
				<< "operator) in the range(" << lower_bound << ", " << upper_bound << ")"
				<< std::endl;
#endif
			RC rc = SUCCESS;

			Coords local_mask, local_x, local_y, local_z;
			const size_t local_n = upper_bound - lower_bound;
			size_t local_mask_nz = local_n;
			size_t local_y_nz = local_n;

#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
			const bool already_dense_vectors = dense_descr ||
				pipeline.allAlreadyDenseVectors();
#else
			constexpr const bool already_dense_vectors = dense_descr;
#endif

			const bool mask_is_dense = (descr & descriptors::structural) &&
				!(descr & descriptors::invert_mask) && already_dense_vectors;

			bool already_dense_mask = true;
			bool already_dense_input_y = true;

			if( !mask_is_dense ) {
				local_z = internal::getCoordinates( z ).asyncSubset( lower_bound,
					upper_bound );
				if( dense_descr && local_z.nonzeroes() < local_n ) {
					return ILLEGAL;
				}
			}

			if( !already_dense_vectors ) {
#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
				already_dense_mask = pipeline.containsAlreadyDenseVector(
					&internal::getCoordinates( mask ) );
				if( !already_dense_mask ) {
#else
					already_dense_mask = false;
#endif
					local_mask = internal::getCoordinates( mask ).asyncSubset( lower_bound,
						upper_bound );
					local_mask_nz = local_mask.nonzeroes();
#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
				}

				already_dense_input_y = pipeline.containsAlreadyDenseVector(
					&internal::getCoordinates( y ) );
				if( !already_dense_input_y ) {
#else
					already_dense_input_y = false;
#endif
					local_y = internal::getCoordinates( y ).asyncSubset( lower_bound,
						upper_bound );
					local_y_nz = local_y.nonzeroes();
#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
				}
#endif
			}

			const internal::Wrapper< true, InputType1, Coords > x_wrapper( alpha );
			const internal::Wrapper< false, InputType2, Coords > y_wrapper( y );

			if( !mask_is_dense ) {
				local_z.local_clear();
				if( lower_bound == 0 ) {
					internal::getCoordinates( z ).reset_global_nnz_counter();
#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
					pipeline.markMaybeSparseVector( &internal::getCoordinates( z ) );
#endif
					if( dense_descr ) {
						pipeline.markMaybeSparseDenseDescriptorVerification(
							&internal::getCoordinates( z ) );
					}
				}
			}

			if( (descr & descriptors::dense) ||
				(local_y_nz == local_n) ||
				local_mask_nz <= local_y_nz
			) {
#ifdef GRB_BOOLEAN_DISPATCHER
				rc = internal::boolean_dispatcher_masked_apply_generic<
#else
				rc = internal::masked_apply_generic<
#endif
						true, false, false, false, descr, OP,
						OutputType, bool, InputType1, InputType2, Coords
					>(
						already_dense_mask, true, already_dense_input_y,
						lower_bound, upper_bound,
						local_z, local_mask, local_x, local_y,
						z, mask, x_wrapper, y_wrapper,
						op
					);
			} else {
#ifdef GRB_BOOLEAN_DISPATCHER
				rc = internal::boolean_dispatcher_sparse_apply_generic<
#else
				rc = internal::sparse_apply_generic<
#endif
						true, false, true, false, descr, OP,
						OutputType, bool, InputType1, InputType2, Coords
					>(
						already_dense_mask, true, already_dense_input_y,
						lower_bound, upper_bound,
						local_z, &local_mask, local_x, local_y,
						z, &mask, x_wrapper, y_wrapper,
						op
					);
			}

			if( !mask_is_dense ) {
				internal::getCoordinates( z ).asyncJoinSubset( local_z, lower_bound,
					upper_bound );
			}

			return rc;
		};

		ret = ret ? ret : internal::le.addStage(
				std::move( func ),
				internal::Opcode::BLAS1_MASKED_EWISEAPPLY,
				n, sizeof( OutputType ), dense_descr, dense_mask,
				&z, nullptr, &internal::getCoordinates( z ), nullptr,
				&y, &mask, nullptr, nullptr,
				&internal::getCoordinates( y ), &internal::getCoordinates( mask ),
				nullptr, nullptr,
				nullptr
			);

#ifdef _NONBLOCKING_DEBUG
		std::cout << "\t\tStage added to a pipeline: eWiseApply(z, mask, alpha, y, "
			<< "operator)" << std::endl;
#endif
		return ret;
	}

	template<
		Descriptor descr = descriptors::no_operation,
		class OP,
		typename OutputType,
		typename InputType1,
		typename InputType2,
		typename Coords
	>
	RC eWiseApply(
		Vector< OutputType, nonblocking, Coords > &z,
		const Vector< InputType1, nonblocking, Coords > &x,
		const Vector< InputType2, nonblocking, Coords > &y,
		const OP &op = OP(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if< !grb::is_object< OutputType >::value &&
				!grb::is_object< InputType1 >::value &&
				!grb::is_object< InputType2 >::value &&
				grb::is_operator< OP >::value,
			void
		>::type * const = nullptr
	) {
		// static checks
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< typename OP::D1, InputType1 >::value ), "grb::eWiseApply",
			"called with a left-hand input element type that does not match the "
			"first domain of the given operator" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< typename OP::D2, InputType2 >::value ), "grb::eWiseApply",
			"called with a right-hand input element type that does not match the "
			"second domain of the given operator" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< typename OP::D3, OutputType >::value ), "grb::eWiseApply",
			"called with an output element type that does not match the "
			"third domain of the given operator" );
#ifdef _DEBUG
		std::cout << "In eWiseApply ([T1]<-[T2]<-[T3]), operator variant\n";
#endif
		// sanity check
		const size_t n = internal::getCoordinates( z ).size();
		if( internal::getCoordinates( x ).size() != n ||
			internal::getCoordinates( y ).size() != n
		) {
#ifdef _DEBUG
			std::cerr << "\tinput vectors mismatch in dimensions!\n";
#endif
			return MISMATCH;
		}

		// check for possible shortcuts
		// trivial dispatch
		if( n == 0 ) {
			return SUCCESS;
		}

		// check for possible shortcuts, after dynamic checks
		if( getID( x ) == getID( y ) && is_idempotent< OP >::value ) {
			return set< descr >( z, x, phase );
		}
		if( getID( x ) == getID( z ) ) {
			return foldl< descr >( z, y, op, phase );
		}
		if( getID( y ) == getID( z ) ) {
			return foldr< descr >( x, z, op, phase );
		}

		if( phase == RESIZE ) {
			return SUCCESS;
		}
		assert( phase == EXECUTE );

		RC ret = SUCCESS;

		constexpr const bool dense_descr = descr & descriptors::dense;

		internal::Pipeline::stage_type func = [&z, &x, &y, &op] (
			internal::Pipeline &pipeline,
			const size_t lower_bound, const size_t upper_bound
		) {
#ifdef _NONBLOCKING_DEBUG
			#pragma omp critical
			std::cout << "\t\tExecution of stage eWiseApply(z, x, y, operator) in the "
				<< "range(" << lower_bound << ", " << upper_bound << ")" << std::endl;
#endif
			RC rc = SUCCESS;

			const Vector< bool, nonblocking, Coords > * const null_mask = nullptr;
			const Coords * const local_null_mask = nullptr;

			Coords local_x, local_y, local_z;
			const size_t local_n = upper_bound - lower_bound;
			size_t local_x_nz = local_n;
			size_t local_y_nz = local_n;
			bool sparse = false;

#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
			const bool already_dense_vectors = dense_descr ||
				pipeline.allAlreadyDenseVectors();
#else
			(void) pipeline;
			constexpr const bool already_dense_vectors = dense_descr;
#endif

			bool already_dense_input_x = true;
			bool already_dense_input_y = true;

			if( !already_dense_vectors ) {
				local_z = internal::getCoordinates( z ).asyncSubset( lower_bound,
					upper_bound );

#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
				already_dense_input_x = pipeline.containsAlreadyDenseVector(
					&internal::getCoordinates( x ) );
				if( !already_dense_input_x ) {
#else
					already_dense_input_x = false;
#endif
					local_x = internal::getCoordinates( x ).asyncSubset( lower_bound,
						upper_bound );
					local_x_nz = local_x.nonzeroes();
					if( local_x_nz < local_n ) {
						sparse = true;
					}
#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
				}

				already_dense_input_y = pipeline.containsAlreadyDenseVector(
					&internal::getCoordinates( y ) );
				if( !already_dense_input_y ) {
#else
					already_dense_input_y = false;
#endif
					local_y = internal::getCoordinates( y ).asyncSubset( lower_bound,
						upper_bound );
					local_y_nz = local_y.nonzeroes();
					if( local_y_nz < local_n ) {
						sparse = true;
					}
#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
				}
#endif
			}

			if( !already_dense_vectors ) {
				if( lower_bound == 0 ) {
					internal::getCoordinates( z ).reset_global_nnz_counter();
				}
			}

			if( sparse ) {
				if( !already_dense_vectors ) {
					local_z.local_clear();
#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
					pipeline.markMaybeSparseVector( &internal::getCoordinates( z ) );
#endif
				}

				const internal::Wrapper< false, InputType1, Coords > x_wrapper( x );
				const internal::Wrapper< false, InputType2, Coords > y_wrapper( y );

#ifdef GRB_BOOLEAN_DISPATCHER
				rc = internal::boolean_dispatcher_sparse_apply_generic<
#else
				rc = internal::sparse_apply_generic<
#endif
						false, false, false, false, descr | descriptors::dense, OP,
						OutputType, bool, InputType1, InputType2, Coords
					>(
						true, already_dense_input_x, already_dense_input_y,
						lower_bound, upper_bound,
						local_z, local_null_mask, local_x, local_y,
						z, null_mask, x_wrapper, y_wrapper,
						op
					);
			} else {
				if( !already_dense_vectors ) {
					local_z.local_assignAll( );
				}

				if( upper_bound > lower_bound ) {
					const InputType1 * __restrict__ a = internal::getRaw( x );
					const InputType2 * __restrict__ b = internal::getRaw( y );
					OutputType * __restrict__ c = internal::getRaw( z );

					// this function is vectorised
					op.eWiseApply( a + lower_bound, b + lower_bound, c + lower_bound, local_n);
				}
			}

			if( !already_dense_vectors ) {
				internal::getCoordinates( z ).asyncJoinSubset( local_z, lower_bound,
					upper_bound );
			}

			return rc;
		};

		ret = ret ? ret : internal::le.addStage(
				std::move( func ),
				internal::Opcode::BLAS1_EWISEAPPLY,
				n, sizeof( OutputType ), dense_descr, true,
				&z, nullptr, &internal::getCoordinates( z ), nullptr,
				&x, &y, nullptr, nullptr,
				&internal::getCoordinates( x ), &internal::getCoordinates( y ),
				nullptr, nullptr,
				nullptr
			);

#ifdef _NONBLOCKING_DEBUG
		std::cout << "\t\tStage added to a pipeline: eWiseApply(z, x, y, operator)"
			<< std::endl;
#endif
		return ret;
	}

	template<
		Descriptor descr = descriptors::no_operation,
		class OP,
		typename OutputType, typename MaskType,
		typename InputType1, typename InputType2,
		typename Coords
	>
	RC eWiseApply(
		Vector< OutputType, nonblocking, Coords > &z,
		const Vector< MaskType, nonblocking, Coords > &mask,
		const Vector< InputType1, nonblocking, Coords > &x,
		const Vector< InputType2, nonblocking, Coords > &y,
		const OP &op = OP(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
				!grb::is_object< OutputType >::value &&
				!grb::is_object< MaskType >::value &&
				!grb::is_object< InputType1 >::value &&
				!grb::is_object< InputType2 >::value &&
				grb::is_operator< OP >::value,
			void
		>::type * const = nullptr
	) {
		// static checks
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< typename OP::D1, InputType1 >::value ), "grb::eWiseApply",
			"called with a left-hand input element type that does not match the "
			"first domain of the given operator" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< typename OP::D2, InputType2 >::value ), "grb::eWiseApply",
			"called with a right-hand input element type that does not match the "
			"second domain of the given operator" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< typename OP::D3, OutputType >::value ), "grb::eWiseApply",
			"called with an output element type that does not match the "
			"third domain of the given operator" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< bool, MaskType >::value ), "grb::eWiseApply",
			"called with an output mask element type that is not Boolean " );
#ifdef _DEBUG
		std::cout << "In masked eWiseApply ([T1]<-[T2]<-[T3], using operator)\n";
#endif
		// check for empty mask
		if( size( mask ) == 0 ) {
			return eWiseApply< descr >( z, x, y, op, phase );
		}

		// other run-time checks
		const size_t n = internal::getCoordinates( z ).size();
		if( internal::getCoordinates( x ).size() != n ) {
			return MISMATCH;
		}
		if( internal::getCoordinates( y ).size() != n ) {
			return MISMATCH;
		}
		if( internal::getCoordinates( mask ).size() != n ) {
			return MISMATCH;
		}

		if( phase == RESIZE ) {
			return SUCCESS;
		}
		assert( phase == EXECUTE );

		RC ret = SUCCESS;

		constexpr const bool dense_descr = descr & descriptors::dense;
		constexpr const bool dense_mask = dense_descr &&
			(descr & descriptors::structural) && !(descr & descriptors::invert_mask);

		internal::Pipeline::stage_type func = [&z, &mask, &x, &y, &op] (
			internal::Pipeline &pipeline,
			const size_t lower_bound, const size_t upper_bound
		) {
#ifdef _NONBLOCKING_DEBUG
			#pragma omp critical
			std::cout << "\t\tExecution of stage eWiseApply(z, mask, x, y, operator) in "
				<< "the range(" << lower_bound << ", " << upper_bound << ")" << std::endl;
#endif
			RC rc = SUCCESS;

			Coords local_mask, local_x, local_y, local_z;
			const size_t local_n = upper_bound - lower_bound;
			size_t local_mask_nz = local_n;
			size_t local_x_nz = local_n;
			size_t local_y_nz = local_n;

#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
			const bool already_dense_vectors = dense_descr ||
				pipeline.allAlreadyDenseVectors();
#else
			constexpr const bool already_dense_vectors = dense_descr;
#endif

			const bool mask_is_dense = (descr & descriptors::structural) &&
				!(descr & descriptors::invert_mask) && already_dense_vectors;

			bool already_dense_mask = true;
			bool already_dense_input_x = true;
			bool already_dense_input_y = true;

			if( !mask_is_dense ) {
				local_z = internal::getCoordinates( z ).asyncSubset( lower_bound,
					upper_bound );
				if( dense_descr && local_z.nonzeroes() < local_n ) {
					return ILLEGAL;
				}
			}

			if( !already_dense_vectors ) {
#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
				already_dense_mask = pipeline.containsAlreadyDenseVector(
					&internal::getCoordinates( mask ) );
				if( !already_dense_mask ) {
#else
					already_dense_mask = false;
#endif
					local_mask = internal::getCoordinates( mask ).asyncSubset( lower_bound,
						upper_bound );
					local_mask_nz = local_mask.nonzeroes();
#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
				}

				already_dense_input_x = pipeline.containsAlreadyDenseVector(
					&internal::getCoordinates( x ) );
				if( !already_dense_input_x ) {
#else
					already_dense_input_x = false;
#endif
					local_x = internal::getCoordinates( x ).asyncSubset( lower_bound,
						upper_bound );
					local_x_nz = local_x.nonzeroes();
#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
				}

				already_dense_input_y = pipeline.containsAlreadyDenseVector(
					&internal::getCoordinates( y ) );
				if( !already_dense_input_y ) {
#else
					already_dense_input_y = false;
#endif
					local_y = internal::getCoordinates( y ).asyncSubset( lower_bound,
						upper_bound );
					local_y_nz = local_y.nonzeroes();
#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
				}
#endif
			}

			const internal::Wrapper< false, InputType1, Coords > x_wrapper( x );
			const internal::Wrapper< false, InputType2, Coords > y_wrapper( y );

			const size_t sparse_loop = std::min( local_x_nz, local_y_nz );

			if( !mask_is_dense ) {
				local_z.local_clear();
				if( lower_bound == 0 ) {
					internal::getCoordinates( z ).reset_global_nnz_counter();
#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
					pipeline.markMaybeSparseVector( &internal::getCoordinates( z ) );
#endif
					if( dense_descr ) {
						pipeline.markMaybeSparseDenseDescriptorVerification(
							&internal::getCoordinates( z ) );
					}
				}
			}

			if( (descr & descriptors::dense) ||
				(local_x_nz == local_n && local_y_nz == local_n) ||
				( !(descr & descriptors::invert_mask) && sparse_loop >= local_mask_nz )
			) {
				// use loop over mask
#ifdef GRB_BOOLEAN_DISPATCHER
				rc = internal::boolean_dispatcher_masked_apply_generic<
#else
				rc = internal::masked_apply_generic<
#endif
						false, false, false, false, descr, OP,
						OutputType, bool, InputType1, InputType2, Coords
					>(
						already_dense_mask, already_dense_input_x, already_dense_input_y,
						lower_bound, upper_bound,
						local_z, local_mask, local_x, local_y,
						z, mask, x_wrapper, y_wrapper,
						op
					);

			} else {
				// use loop over sparse inputs
#ifdef GRB_BOOLEAN_DISPATCHER
				rc = internal::boolean_dispatcher_sparse_apply_generic<
#else
				rc = internal::sparse_apply_generic<
#endif
						true, false, false, false, descr, OP,
						OutputType, bool, InputType1, InputType2, Coords
					>(
						already_dense_mask, already_dense_input_x, already_dense_input_y,
						lower_bound, upper_bound,
						local_z, &local_mask, local_x, local_y,
						z, &mask, x_wrapper, y_wrapper,
						op
					);
			}

			if( !mask_is_dense ) {
				internal::getCoordinates( z ).asyncJoinSubset( local_z, lower_bound,
					upper_bound );
			}

			return rc;
		};

		ret = ret ? ret : internal::le.addStage(
				std::move( func ),
				internal::Opcode::BLAS1_MASKED_EWISEAPPLY,
				n, sizeof( OutputType ), dense_descr, dense_mask,
				&z, nullptr, &internal::getCoordinates( z ), nullptr,
				&x, &y, &mask, nullptr,
				&internal::getCoordinates( x ), &internal::getCoordinates( y ),
				&internal::getCoordinates( mask ), nullptr,
				nullptr
			);

#ifdef _NONBLOCKING_DEBUG
		std::cout << "\t\tStage added to a pipeline: eWiseApply(z, mask, x, y, "
			<< "operator)" << std::endl;
#endif
		return ret;
	}

	template<
		Descriptor descr = descriptors::no_operation,
		class Ring,
		typename OutputType,
		typename InputType1,
		typename InputType2,
		typename Coords
	>
	RC eWiseAdd(
		Vector< OutputType, nonblocking, Coords > &z,
		const Vector< InputType1, nonblocking, Coords > &x,
		const Vector< InputType2, nonblocking, Coords > &y,
		const Ring &ring = Ring(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
				!grb::is_object< OutputType >::value &&
				!grb::is_object< InputType1 >::value &&
				!grb::is_object< InputType2 >::value &&
				grb::is_semiring< Ring >::value,
			void
		>::type * const = nullptr
	) {
		// static sanity checks
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< typename Ring::D4, OutputType >::value ), "grb::eWiseAdd",
			"called with an output vector with element type that does not match the "
			"fourth domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< typename Ring::D3, InputType1 >::value ), "grb::eWiseAdd",
			"called with a left-hand side input vector with element type that does not "
			"match the third domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< typename Ring::D4, OutputType >::value ), "grb::eWiseAdd",
			"called with a right-hand side input vector with element type that does "
			"not match the fourth domain of the given semiring" );
#ifdef _DEBUG
		std::cout << "eWiseAdd (nonblocking, vector <- vector + vector) dispatches to "
			<< "two folds using the additive monoid\n";
#endif
		RC ret = foldl< descr >( z, x, ring.getAdditiveMonoid(), phase );
		ret = ret ? ret : foldl< descr >( z, y, ring.getAdditiveMonoid(), phase );
		return ret;
	}

	template<
		Descriptor descr = descriptors::no_operation,
		class Ring,
		typename InputType1,
		typename InputType2,
		typename OutputType,
		typename Coords
	>
	RC eWiseAdd(
		Vector< OutputType, nonblocking, Coords > &z,
		const InputType1 alpha,
		const Vector< InputType2, nonblocking, Coords > &y,
		const Ring &ring = Ring(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
				!grb::is_object< OutputType >::value &&
				!grb::is_object< InputType1 >::value &&
				!grb::is_object< InputType2 >::value &&
				grb::is_semiring< Ring >::value,
			void
		>::type * const = nullptr
	) {
		// static sanity checks
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< typename Ring::D1, InputType1 >::value ), "grb::eWiseAdd",
			"called with a left-hand side input vector with element type that does not "
			"match the first domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< typename Ring::D2, InputType2 >::value ), "grb::eWiseAdd",
			"called with a right-hand side input vector with element type that does "
			"not match the second domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< typename Ring::D3, OutputType >::value ), "grb::eWiseAdd",
			"called with an output vector with element type that does not match the "
			"third domain of the given semiring" );
#ifdef _DEBUG
		std::cout << "eWiseAdd (nonblocking, vector <- scalar + vector) dispatches to "
			<< "two folds with the additive monoid\n";
#endif
		RC ret = foldl< descr >( z, alpha, ring.getAdditiveMonoid(), phase );
		ret = ret ? ret : foldl< descr >( z, y, ring.getAdditiveMonoid(), phase );
		return ret;
	}

	template<
		Descriptor descr = descriptors::no_operation,
		class Ring,
		typename InputType1,
		typename InputType2,
		typename OutputType,
		typename Coords
	>
	RC eWiseAdd(
		Vector< OutputType, nonblocking, Coords > &z,
		const Vector< InputType1, nonblocking, Coords > &x,
		const InputType2 beta,
		const Ring &ring = Ring(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
				!grb::is_object< OutputType >::value &&
				!grb::is_object< InputType1 >::value &&
				!grb::is_object< InputType2 >::value &&
				grb::is_semiring< Ring >::value,
			void
		>::type * const = nullptr
	) {
		// static sanity checks
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< typename Ring::D1, InputType1 >::value ), "grb::eWiseAdd",
			"called with a left-hand side input vector with element type that does not "
			"match the first domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< typename Ring::D2, InputType2 >::value ), "grb::eWiseAdd",
			"called with a right-hand side input vector with element type that does "
			"not match the second domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< typename Ring::D3, OutputType >::value ), "grb::eWiseAdd",
			"called with an output vector with element type that does not match the "
			"third domain of the given semiring" );
#ifdef _DEBUG
		std::cout << "eWiseAdd (nonblocking, vector <- vector + scalar) dispatches to "
			<< "two folds with the additive monoid\n";
#endif
		RC ret = foldl< descr >( z, x, ring.getAdditiveMonoid(), phase );
		ret = ret ? ret : foldl< descr >( z, beta, ring.getAdditiveMonoid(), phase );
		return ret;
	}

	template<
		Descriptor descr = descriptors::no_operation,
		class Ring,
		typename InputType1,
		typename InputType2,
		typename OutputType,
		typename Coords
	>
	RC eWiseAdd(
		Vector< OutputType, nonblocking, Coords > &z,
		const InputType1 alpha,
		const InputType2 beta,
		const Ring &ring = Ring(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
				!grb::is_object< OutputType >::value &&
				!grb::is_object< InputType1 >::value &&
				!grb::is_object< InputType2 >::value &&
				grb::is_semiring< Ring >::value,
			void
		>::type * const = nullptr
	) {
		// static sanity checks
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< typename Ring::D1, InputType1 >::value ), "grb::eWiseAdd",
			"called with a left-hand side input vector with element type that does not "
			"match the first domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< typename Ring::D2, InputType2 >::value ), "grb::eWiseAdd",
			"called with a right-hand side input vector with element type that does "
			"not match the second domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< typename Ring::D3, OutputType >::value ), "grb::eWiseAdd",
			"called with an output vector with element type that does not match the "
			"third domain of the given semiring" );
#ifdef _DEBUG
		std::cout << "eWiseAdd (nonblocking, vector <- scalar + scalar) dispatches to "
			<< "foldl with precomputed scalar and additive monoid\n";
#endif
		const typename Ring::D4 add;
		(void) apply( add, alpha, beta, ring.getAdditiveOperator() );
		return foldl< descr >( z, add, ring.getAdditiveMonoid(), phase );
	}

	template<
		Descriptor descr = descriptors::no_operation,
		class Ring,
		typename OutputType,
		typename MaskType,
		typename InputType1,
		typename InputType2,
		typename Coords
	>
	RC eWiseAdd(
		Vector< OutputType, nonblocking, Coords > &z,
		const Vector< MaskType, nonblocking, Coords > &m,
		const Vector< InputType1, nonblocking, Coords > &x,
		const Vector< InputType2, nonblocking, Coords > &y,
		const Ring &ring = Ring(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
				!grb::is_object< OutputType >::value &&
				!grb::is_object< MaskType >::value &&
				!grb::is_object< InputType1 >::value &&
				!grb::is_object< InputType2 >::value &&
				grb::is_semiring< Ring >::value,
			void
		>::type * const = nullptr
	) {
		// static sanity checks
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< typename Ring::D4, OutputType >::value ), "grb::eWiseAdd",
			"called with an output vector with element type that does not match the "
			"fourth domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< typename Ring::D3, InputType1 >::value ), "grb::eWiseAdd",
			"called with a left-hand side input vector with element type that does not "
			"match the third domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< typename Ring::D4, OutputType >::value ), "grb::eWiseAdd",
			"called with a right-hand side input vector with element type that does "
			"not match the fourth domain of the given semiring" );
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< MaskType, bool >::value ),
			"grb::eWiseAdd (vector <- vector + vector, masked)",
			"called with non-bool mask element types" );
#ifdef _DEBUG
		std::cout << "eWiseAdd (nonblocking, vector <- vector + vector, masked) "
			<< "dispatches to two folds using the additive monoid\n";
#endif
		RC ret = foldl< descr >( z, m, x, ring.getAdditiveMonoid(), phase );
		ret = ret ? ret : foldl< descr >( z, m, y, ring.getAdditiveMonoid(), phase );
		return ret;
	}

	template<
		Descriptor descr = descriptors::no_operation,
		class Ring,
		typename InputType1,
		typename InputType2,
		typename OutputType,
		typename MaskType,
		typename Coords
	>
	RC eWiseAdd(
		Vector< OutputType, nonblocking, Coords > &z,
		const Vector< MaskType, nonblocking, Coords > &m,
		const InputType1 alpha,
		const Vector< InputType2, nonblocking, Coords > &y,
		const Ring &ring = Ring(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
				!grb::is_object< OutputType >::value &&
				!grb::is_object< MaskType >::value &&
				!grb::is_object< InputType1 >::value &&
				!grb::is_object< InputType2 >::value &&
				grb::is_semiring< Ring >::value,
			void
		>::type * const = nullptr
	) {
		// static sanity checks
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< typename Ring::D1, InputType1 >::value ), "grb::eWiseAdd",
			"called with a left-hand side input vector with element type that does not "
			"match the first domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< typename Ring::D2, InputType2 >::value ), "grb::eWiseAdd",
			"called with a right-hand side input vector with element type that does "
			"not match the second domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< typename Ring::D3, OutputType >::value ), "grb::eWiseAdd",
			"called with an output vector with element type that does not match the "
			"third domain of the given semiring" );
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< MaskType, bool >::value ),
			"grb::eWiseAdd (vector <- scalar + vector, masked)",
			"called with non-bool mask element types" );
#ifdef _DEBUG
		std::cout << "eWiseAdd (nonblocking, vector <- scalar + vector, masked) "
			<< "dispatches to two folds using the additive monoid\n";
#endif
		RC ret = foldl< descr >( z, m, alpha, ring.getAdditiveMonoid(), phase );
		ret = ret ? ret : foldl< descr >( z, m, y, ring.getAdditiveMonoid(), phase );
		return ret;
	}

	template<
		Descriptor descr = descriptors::no_operation,
		class Ring,
		typename InputType1,
		typename InputType2,
		typename OutputType,
		typename MaskType,
		typename Coords
	>
	RC eWiseAdd(
		Vector< OutputType, nonblocking, Coords > &z,
		const Vector< MaskType, nonblocking, Coords > &m,
		const Vector< InputType1, nonblocking, Coords > &x,
		const InputType2 beta,
		const Ring &ring = Ring(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
				!grb::is_object< OutputType >::value &&
				!grb::is_object< MaskType >::value &&
				!grb::is_object< InputType1 >::value &&
				!grb::is_object< InputType2 >::value &&
				grb::is_semiring< Ring >::value,
			void
		>::type * const = nullptr
	) {
		// static sanity checks
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< typename Ring::D1, InputType1 >::value ), "grb::eWiseAdd",
			"called with a left-hand side input vector with element type that does not "
			"match the first domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< typename Ring::D2, InputType2 >::value ), "grb::eWiseAdd",
			"called with a right-hand side input vector with element type that does "
			"not match the second domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< typename Ring::D3, OutputType >::value ), "grb::eWiseAdd",
			"called with an output vector with element type that does not match the "
			"third domain of the given semiring" );
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< MaskType, bool >::value ),
			"grb::eWiseAdd (vector <- vector + scalar, masked)",
			"called with non-bool mask element types" );
#ifdef _DEBUG
		std::cout << "eWiseAdd (nonblocking, vector <- vector + scalar, masked) "
			<< "dispatches to eWiseApply using the additive monoid\n";
#endif
		RC ret = foldl< descr >( z, m, x, ring.getAdditiveMonoid(), phase );
		ret = ret ? ret : foldl< descr >( z, m, beta, ring.getAdditiveMonoid(),
			phase );
		return ret;
	}

	template<
		Descriptor descr = descriptors::no_operation,
		class Ring,
		typename InputType1,
		typename InputType2,
		typename OutputType,
		typename MaskType,
		typename Coords
	>
	RC eWiseAdd(
		Vector< OutputType, nonblocking, Coords > &z,
		const Vector< OutputType, nonblocking, Coords > &m,
		const InputType1 alpha,
		const InputType2 beta,
		const Ring &ring = Ring(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
				!grb::is_object< OutputType >::value &&
				!grb::is_object< MaskType >::value &&
				!grb::is_object< InputType1 >::value &&
				!grb::is_object< InputType2 >::value &&
				grb::is_semiring< Ring >::value,
			void
		>::type * const = nullptr
	) {
		// static sanity checks
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< typename Ring::D1, InputType1 >::value ), "grb::eWiseAdd",
			"called with a left-hand side input vector with element type that does not "
			"match the first domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< typename Ring::D2, InputType2 >::value ), "grb::eWiseAdd",
			"called with a right-hand side input vector with element type that does "
			"not match the second domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< typename Ring::D3, OutputType >::value ), "grb::eWiseAdd",
			"called with an output vector with element type that does not match the "
			"third domain of the given semiring" );
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< MaskType, bool >::value ),
			"grb::eWiseAdd (vector <- scalar + scalar, masked)",
			"called with non-bool mask element types" );
#ifdef _DEBUG
		std::cout << "eWiseAdd (nonblocking, vector <- scalar + scalar, masked) "
			<< "dispatches to foldl with precomputed scalar and additive monoid\n";
#endif
		const typename Ring::D4 add;
		(void) apply( add, alpha, beta, ring.getAdditiveOperator() );
		return foldl< descr >( z, m, add, ring.getAdditiveMonoid(), phase );
	}

	// declare an internal version of eWiseMulAdd containing the full sparse &
	// dense implementations
	namespace internal {

		template<
			Descriptor descr,
			bool a_scalar,
			bool x_scalar,
			bool y_scalar,
			bool y_zero,
#ifdef GRB_BOOLEAN_DISPATCHER
			bool already_dense_output,
			bool already_dense_mask,
			bool already_dense_input_a,
			bool already_dense_input_x,
			bool already_dense_input_y,
#endif
			typename OutputType,
			typename MaskType,
			typename InputType1,
			typename InputType2,
			typename InputType3,
			typename Coords,
			class Ring
		>
		RC sparse_eWiseMulAdd_maskDriven(
#ifndef GRB_BOOLEAN_DISPATCHER
			bool already_dense_output,
			bool already_dense_mask,
			bool already_dense_input_a,
			bool already_dense_input_x,
			bool already_dense_input_y,
#endif
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
			static_assert( !(descr & descriptors::invert_mask),
				"Cannot loop over mask nonzeroes if invert_mask is given. "
				"Please submit a bug report" );
			static_assert( !a_scalar || !x_scalar,
				"If both a and x are scalars, this is operation is a simple eWiseApply "
				"with the additive operator if the semiring." );
			static_assert( !y_zero || y_scalar,
				"If y_zero is given, then y_scalar must be given also." );

			OutputType * __restrict__ z = internal::getRaw( z_vector );
			const MaskType * __restrict__ const m = internal::getRaw( m_vector );

			// create local copies of the input const pointers
			const InputType1 * __restrict__ const a = a_wrapper.getRaw();
			const InputType2 * __restrict__ const x = x_wrapper.getRaw();
			const InputType3 * __restrict__ const y = y_wrapper.getRaw();

			const size_t local_n = upper_bound - lower_bound;
			const size_t local_m_nz = already_dense_mask ? local_n : local_m.nonzeroes();

			const size_t local_start = 0;
			const size_t local_end = local_m_nz;

			size_t k = local_start;

			// scalar coda and parallel main body
			for( ; k < local_end; ++k ) {
				const size_t index = ( already_dense_mask ? k : local_m.index( k ) ) +
					lower_bound;
				assert( index - lower_bound < local_n );
				if( already_dense_mask ) {
					if( !internal::getCoordinates( m_vector ).template mask< descr >(
						index, m )
					) {
						continue;
					}
				} else {
					if( !local_m.template mask< descr >( index - lower_bound, m +
						lower_bound )
					) {
						continue;
					}
				}
				typename Ring::D3 t = ring.template getZero< typename Ring::D3 >();
				if(
					(
						a_scalar || already_dense_input_a ||
						local_a.assigned( index - lower_bound )
					) && (
						x_scalar || already_dense_input_x ||
						local_x.assigned( index - lower_bound)
					)
				) {
					const InputType1 a_p = ( a_scalar )
						? a_wrapper.getValue()
						: *( a + index );
					const InputType2 x_p = ( x_scalar )
						? x_wrapper.getValue()
						: *( x + index );
					(void) apply( t, a_p, x_p, ring.getMultiplicativeOperator() );
					if( !y_zero && (
						y_scalar || already_dense_input_y ||
						local_y.assigned( index - lower_bound ) )
					) {
						const InputType3 y_p = ( y_scalar )
							? y_wrapper.getValue()
							: *( y + index );
						typename Ring::D4 b;
						(void) apply( b, t, y_p, ring.getAdditiveOperator() );
						if( already_dense_output || local_z.assigned( index - lower_bound ) ) {
							typename Ring::D4 out = static_cast< typename Ring::D4 >( z[ index ] );
							(void) foldr( b, out, ring.getAdditiveOperator() );
							z[ index ] = static_cast< OutputType >( out );
						} else {
							(void) local_z.assign( index - lower_bound );
							z[ index ] = static_cast< OutputType >( b );
						}
					} else if( already_dense_output ||
						local_z.assigned( index - lower_bound )
					) {
						typename Ring::D4 out = static_cast< typename Ring::D4 >( z[ index ] );
						(void) foldr( t, out, ring.getAdditiveOperator() );
						z[ index ] = static_cast< OutputType >( out );
					} else {
						(void) local_z.assign( index - lower_bound );
						z[ index ] = static_cast< OutputType >( t );
					}
				} else if( !y_zero && (
					already_dense_input_y || y_scalar ||
					local_y.assigned( index - lower_bound ) )
				) {
					if( already_dense_output || local_z.assigned( index - lower_bound ) ) {
						typename Ring::D4 out = static_cast< typename Ring::D4 >( z[ index ] );
						(void) foldr( y[ index ], out, ring.getAdditiveOperator() );
						z[ index ] = static_cast< OutputType >( out );
					} else {
						(void)local_z.assign( index - lower_bound );
						z[ index ] = static_cast< OutputType >( t );
					}
				}
			}

			return SUCCESS;
		}

		template<
			Descriptor descr,
			bool masked,
			bool x_scalar,
			bool y_scalar,
			bool y_zero,
			bool mulSwitched,
#ifdef GRB_BOOLEAN_DISPATCHER
			bool already_dense_output,
			bool already_dense_mask,
			bool already_dense_input_a,
			bool already_dense_input_x,
			bool already_dense_input_y,
#endif
			typename OutputType,
			typename MaskType,
			typename InputType1,
			typename InputType2,
			typename InputType3,
			typename Coords,
			class Ring
		>
		RC twoPhase_sparse_eWiseMulAdd_mulDriven(
#ifndef GRB_BOOLEAN_DISPATCHER
			bool already_dense_output,
			bool already_dense_mask,
			bool already_dense_input_a,
			bool already_dense_input_x,
			bool already_dense_input_y,
#endif
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
		) {
			OutputType * __restrict__ z = internal::getRaw( z_vector );
			const MaskType * __restrict__ const m = masked
				? internal::getRaw( *m_vector )
				: nullptr;
			const InputType1 * __restrict__ const a = internal::getRaw( a_vector );

			// create local copies of the input const pointers
			const InputType2 * __restrict__ const x = x_wrapper.getRaw();

			const size_t local_n = upper_bound - lower_bound;
			const size_t local_a_nz = already_dense_input_a
				? local_n
				: local_a.nonzeroes();

			for( size_t i = 0; i < local_a_nz; ++i ) {
				const size_t index = ( already_dense_input_a ? i : local_a.index( i ) ) +
					lower_bound;
				if( masked ) {
					if( already_dense_mask ) {
						if( !internal::getCoordinates( *m_vector ).template mask< descr >(
							index, m )
						) {
							continue;
						}
					} else {
						if( !local_m->template mask< descr >( index - lower_bound,
							m + lower_bound )
						) {
							continue;
						}
					}
				}

				if( x_scalar || already_dense_input_x ||
					local_x.assigned( index - lower_bound )
				) {
					typename Ring::D3 t;
					const InputType1 a_p = *( a + index );
					const InputType2 x_p = ( x_scalar )
						? x_wrapper.getValue()
						: *( x + index );

					if( mulSwitched ) {
						(void) apply( t, x_p, a_p, ring.getMultiplicativeOperator() );
					} else {
						(void) apply( t, a_p, x_p, ring.getMultiplicativeOperator() );
					}

					if( already_dense_output || local_z.assign( index - lower_bound ) ) {
						typename Ring::D4 b = static_cast< typename Ring::D4 >( z[ index ] );
						(void) foldr( t, b, ring.getAdditiveOperator() );
						z[ index ] = static_cast< OutputType >( b );
					} else {
						z[ index ] = static_cast< OutputType >(
							static_cast< typename Ring::D4 >( t )
						);
					}
				}
			}

			RC rc = SUCCESS;

			// now handle addition
			if( !y_zero ) {
				// now handle addition
				if( masked ) {
					if( y_scalar ) {
						rc = internal::fold_from_scalar_to_vector_generic<
#ifdef GRB_BOOLEAN_DISPATCHER
								descr, true, true, true, true,
								already_dense_output, already_dense_mask
#else
								descr, true, true, true, true
#endif
							>(
#ifndef GRB_BOOLEAN_DISPATCHER
								already_dense_output, already_dense_mask,
#endif
								lower_bound, upper_bound, local_z, local_m,
								z_vector, m_vector, y_wrapper.getValue(),
								ring.getAdditiveMonoid().getOperator(), EXECUTE
							);
					} else {
						rc = fold_from_vector_to_vector_generic<
#ifdef GRB_BOOLEAN_DISPATCHER
								descr, true, true, true, true,
								already_dense_output, already_dense_input_y, already_dense_mask
#else
								descr, true, true, true, true
#endif
							>(
#ifndef GRB_BOOLEAN_DISPATCHER
								already_dense_output, already_dense_input_y, already_dense_mask,
#endif
								lower_bound, upper_bound,
								local_z, local_m, local_y,
								z_vector, m_vector, *( y_wrapper.getPointer() ),
								ring.getAdditiveMonoid().getOperator(), EXECUTE
							);
					}
				} else {
					if( y_scalar ) {
						rc = fold_from_scalar_to_vector_generic<
#ifdef GRB_BOOLEAN_DISPATCHER
								descr, true, true, false, true,
								already_dense_output, already_dense_mask
#else
								descr, true, true, false, true
#endif
							>(
#ifndef GRB_BOOLEAN_DISPATCHER
								already_dense_output, already_dense_mask,
#endif
								lower_bound, upper_bound,
								local_z, local_m,
								z_vector, m_vector, y_wrapper.getValue(),
								ring.getAdditiveMonoid().getOperator(), EXECUTE
							);
					} else {
						rc = fold_from_vector_to_vector_generic<
#ifdef GRB_BOOLEAN_DISPATCHER
								descr, true, true, false, true,
								already_dense_output, already_dense_input_y, already_dense_mask
#else
								descr, true, true, false, true
#endif
							>(
#ifndef GRB_BOOLEAN_DISPATCHER
								already_dense_output, already_dense_input_y, already_dense_mask,
#endif
								lower_bound, upper_bound,
								local_z, local_m, local_y,
								z_vector, m_vector, *( y_wrapper.getPointer() ),
								ring.getAdditiveMonoid().getOperator(), EXECUTE
							);
					}
				}
			}

			// done
			return rc;
		}

		template<
			Descriptor descr,
			bool a_scalar,
			bool x_scalar,
			bool y_scalar,
			bool y_zero,
			bool assign_z,
			typename OutputType,
			typename InputType1,
			typename InputType2,
			typename InputType3,
			typename Coords,
			class Ring
		>
		RC dense_eWiseMulAdd(
			const size_t lower_bound,
			const size_t upper_bound,
			Vector< OutputType, nonblocking, Coords > &z_vector,
			const internal::Wrapper< a_scalar, InputType1, Coords > &a_wrapper,
			const internal::Wrapper< x_scalar, InputType2, Coords > &x_wrapper,
			const internal::Wrapper< y_scalar, InputType3, Coords > &y_wrapper,
			const Ring &ring = Ring()
		) {
#ifdef _DEBUG
			std::cout << "\tdense_eWiseMulAdd: loop size will be "
				<< (upper_bound - lower_bound) << " in the range(" << lower_bound << ", "
				<< upper_bound << ")\n";
#endif
			const size_t start = lower_bound;
			const size_t end = upper_bound;

			OutputType * __restrict__ z = internal::getRaw( z_vector );

			// create local copies of the input const pointers
			const InputType1 * __restrict__ a = a_wrapper.getRaw();
			const InputType2 * __restrict__ x = x_wrapper.getRaw();
			const InputType3 * __restrict__ y = y_wrapper.getRaw();

			assert( z != a );
			assert( z != x );
			assert( z != y );
			assert( a != x || a == nullptr );
			assert( a != y || a == nullptr );
			assert( x != y || x == nullptr );

			// vector registers
			typename Ring::D1 aa[ Ring::blocksize ];
			typename Ring::D2 xx[ Ring::blocksize ];
			typename Ring::D3 tt[ Ring::blocksize ];
			typename Ring::D4 bb[ Ring::blocksize ];
			typename Ring::D4 yy[ Ring::blocksize ];
			typename Ring::D4 zz[ Ring::blocksize ];

			if( a_scalar ) {
				for( size_t b = 0; b < Ring::blocksize; ++b ) {
					aa[ b ] = a_wrapper.getValue();
				}
			}
			if( x_scalar ) {
				for( size_t b = 0; b < Ring::blocksize; ++b ) {
					xx[ b ] = x_wrapper.getValue();
				}
			}
			if( y_scalar ) {
				if( y_zero ) {
					for( size_t b = 0; b < Ring::blocksize; ++b ) {
						yy[ b ] = ring.template getZero< typename Ring::D4 >();
					}
				} else {
					for( size_t b = 0; b < Ring::blocksize; ++b ) {
						yy[ b ] = y_wrapper.getValue();
					}
				}
			}

			// do vectorised out-of-place operations. Allows for aligned overlap.
			// Non-aligned ovelap is not possible due to GraphBLAS semantics.
			size_t i = start;
			// note: read the tail code (under this while loop) comments first for
			// greater understanding
			while( i + Ring::blocksize <= end ) {
#ifdef _DEBUG
				std::cout << "\tdense_eWiseMulAdd: handling block of size "
					<< Ring::blocksize << " starting at index " << i << "\n";
#endif
				// read-in
				if( !a_scalar ) {
					for( size_t b = 0; b < Ring::blocksize; ++b ) {
						aa[ b ] = static_cast< typename Ring::D2 >( a[ i + b ] );
					}
				}
				if( !x_scalar ) {
					for( size_t b = 0; b < Ring::blocksize; ++b ) {
						xx[ b ] = static_cast< typename Ring::D2 >( x[ i + b ] );
					}
				}
				if( !y_scalar ) {
					for( size_t b = 0; b < Ring::blocksize; ++b ) {
						yy[ b ] = static_cast< typename Ring::D4 >( y[ i + b ] );
					}
				}
				if( !assign_z ) {
					for( size_t b = 0; b < Ring::blocksize; ++b ) {
						zz[ b ] = static_cast< typename Ring::D4 >( z[ i + b ] );
					}
				}

				// operate
				if( !y_zero ) {
					for( size_t b = 0; b < Ring::blocksize; ++b ) {
						apply( tt[ b ], aa[ b ], xx[ b ], ring.getMultiplicativeOperator() );
						apply( bb[ b ], tt[ b ], yy[ b ], ring.getAdditiveOperator() );
					}
				} else {
					assert( y_scalar );
					for( size_t b = 0; b < Ring::blocksize; ++b ) {
						apply( bb[ b ], aa[ b ], xx[ b ], ring.getMultiplicativeOperator() );
					}
				}
				if( !assign_z ) {
					for( size_t b = 0; b < Ring::blocksize; ++b ) {
						foldr( bb[ b ], zz[ b ], ring.getAdditiveOperator() );
					}
				}

				// write-out
				if( assign_z ) {
					for( size_t b = 0; b < Ring::blocksize; ++b, ++i ) {
						z[ i ] = static_cast< OutputType >( bb[ b ] );
					}
				} else {
					for( size_t b = 0; b < Ring::blocksize; ++b, ++i ) {
						z[ i ] = static_cast< OutputType >( zz[ b ] );
					}
				}
			}

			// perform tail
			if( !a_scalar ) {
				a += i;
			}
			if( !x_scalar ) {
				x += i;
			}
			if( !y_scalar ) {
				y += i;
			}
			z += i;
			for( ; i < end; ++i ) {
				// do multiply
				const typename Ring::D1 &as = ( a_scalar )
					? static_cast< typename Ring::D1 >( a_wrapper.getValue() )
					: static_cast< typename Ring::D1 >( *a );
				const typename Ring::D2 &xs = ( x_scalar )
					? static_cast< typename Ring::D2 >( x_wrapper.getValue() )
					: static_cast< typename Ring::D2 >( *x );
				typename Ring::D4 ys = ( y_scalar )
					? static_cast< typename Ring::D4 >( y_wrapper.getValue() )
					: static_cast< typename Ring::D4 >( *y );
				typename Ring::D3 ts;

				if( !y_zero ) {
					RC always_succeeds = apply( ts, as, xs, ring.getMultiplicativeOperator() );
					assert( always_succeeds == SUCCESS );
					always_succeeds = foldr( ts, ys, ring.getAdditiveOperator() );
					assert( always_succeeds == SUCCESS );
#ifdef NDEBUG
					(void) always_succeeds;
#endif
				} else {
					RC always_succeeds = apply( ys, as, xs, ring.getMultiplicativeOperator() );
					assert( always_succeeds == SUCCESS );
#ifdef NDEBUG
					(void) always_succeeds;
#endif
				}

				// write out
				if( assign_z ) {
					*z = static_cast< OutputType >( ys );
				} else {
					RC always_succeeds = foldr( ys, *z, ring.getAdditiveOperator() );
					assert( always_succeeds == SUCCESS );
#ifdef NDEBUG
					(void) always_succeeds;
#endif
				}

				// move pointers
				if( !a_scalar ) {
					(void)a++;
				}
				if( !x_scalar ) {
					(void)x++;
				}
				if( !y_scalar ) {
					(void)y++;
				}
				(void)z++;
			}

			// done
			return SUCCESS;
		}

		template<
			Descriptor descr,
			bool masked,
			bool a_scalar,
			bool x_scalar,
			bool y_scalar,
			bool y_zero,
			typename MaskType,
			class Ring,
			typename InputType1,
			typename InputType2,
			typename InputType3,
			typename OutputType,
			typename Coords
		>
		RC eWiseMulAdd_dispatch(
			Vector< OutputType, nonblocking, Coords > &z_vector,
			const Vector< MaskType, nonblocking, Coords > * const m_vector,
			const internal::Wrapper< a_scalar, InputType1, Coords > &a_wrapper,
			const internal::Wrapper< x_scalar, InputType2, Coords > &x_wrapper,
			const internal::Wrapper< y_scalar, InputType3, Coords > &y_wrapper,
			const size_t n,
			const Ring &ring
		) {
			static_assert( !y_zero || y_scalar, "If y is zero, y_scalar must be true. "
				"Triggering this assertion indicates an incorrect call to this "
				"function; please submit a bug report" );
#ifdef _DEBUG
			std::cout << "\t in eWiseMulAdd_dispatch\n";
#endif
			RC ret = SUCCESS;

			constexpr const bool dense_descr = descr & descriptors::dense;

			internal::Pipeline::stage_type func =
				[&z_vector, m_vector, a_wrapper, x_wrapper, y_wrapper, &ring] (
					internal::Pipeline &pipeline,
					const size_t lower_bound, const size_t upper_bound
				) {
#ifdef _NONBLOCKING_DEBUG
					#pragma omp critical
					std::cout << "\t\tExecution of stage eWiseMulAdd_dispatch in the range("
						<< lower_bound << ", " << upper_bound << ")" << std::endl;
#endif
					RC rc = SUCCESS;

					Coords local_z, local_m, local_a, local_x, local_y;
					const size_t local_n = upper_bound - lower_bound;
					size_t local_z_nz = local_n;
					size_t local_m_nz = local_n;
					size_t local_a_nz = local_n;
					size_t local_x_nz = local_n;
					size_t local_y_nz = local_n;

#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
					const bool already_dense_vectors = dense_descr ||
						pipeline.allAlreadyDenseVectors();
#else
					(void) pipeline;
					constexpr const bool already_dense_vectors = dense_descr;
#endif
					bool already_dense_output = true;
					bool already_dense_mask = true;
					bool already_dense_input_a = true;
					bool already_dense_input_x = true;
					bool already_dense_input_y = true;

					if( !already_dense_vectors ) {
#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
						already_dense_output = pipeline.containsAlreadyDenseVector(
							&internal::getCoordinates( z_vector ) );
						if( !already_dense_output ) {
#else
							already_dense_output = false;
#endif
							local_z = internal::getCoordinates( z_vector ).asyncSubset( lower_bound,
								upper_bound );
							local_z_nz = local_z.nonzeroes();
#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
						}
#endif
						if( masked ) {
#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
							already_dense_mask = pipeline.containsAlreadyDenseVector(
								&internal::getCoordinates( *m_vector ) );
							if( !already_dense_mask ) {
#else
								already_dense_mask = false;
#endif
								local_m = internal::getCoordinates( *m_vector ).asyncSubset(
									lower_bound, upper_bound );
								local_m_nz = local_m.nonzeroes();
#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
							}
#endif
						}

						if( !a_scalar ) {
#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
							already_dense_input_a = pipeline.containsAlreadyDenseVector(
								a_wrapper.getCoordinates() );
							if ( !already_dense_input_a ) {
#else
								already_dense_input_a = false;
#endif
								local_a = a_wrapper.getCoordinates()->asyncSubset( lower_bound,
									upper_bound );
								local_a_nz = local_a.nonzeroes();
#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
							}
#endif
						}

						if( !x_scalar ) {
#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
							already_dense_input_x = pipeline.containsAlreadyDenseVector(
								x_wrapper.getCoordinates() );
							if( !already_dense_input_x ) {
#else
								already_dense_input_x = false;
#endif
								local_x = x_wrapper.getCoordinates()->asyncSubset( lower_bound,
									upper_bound );
								local_x_nz = local_x.nonzeroes();
#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
							}
#endif
						}

						if( !y_scalar ) {
#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
							already_dense_input_y = pipeline.containsAlreadyDenseVector(
								y_wrapper.getCoordinates() );
							if( !already_dense_input_y ) {
#else
								already_dense_input_y = false;
#endif
								local_y = y_wrapper.getCoordinates()->asyncSubset( lower_bound,
									upper_bound );
								local_y_nz = local_y.nonzeroes();
#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
							}
#endif
						}
					}

					// check whether we are in the sparse or dense case
					const bool mask_is_dense = !masked || (
							(descr & descriptors::structural) &&
							!(descr & descriptors::invert_mask) &&
							local_m_nz == local_n
						);
					const size_t z_nns = local_z_nz;

					// the below Boolean shall be true only if the inputs a, x, and y generate
					// a dense output vector. It furthermore shall be set to false only if the
					// output vector was either empty or fully dense. This is done to determine
					// the exact case the dense variant of the eWiseMulAdd implementations can
					// be used.
					const bool sparse = ( a_scalar ? false : ( local_a_nz < local_n ) ) ||
						( x_scalar ? false : ( local_x_nz < local_n ) ) ||
						( y_scalar ? false : ( local_y_nz < local_n ) ) ||
						( z_nns > 0 && z_nns < local_n ) ||
						( masked && !mask_is_dense );
					assert( !(sparse && dense_descr) );
#ifdef _DEBUG
					std::cout << "\t\t (sparse, dense)=(" << sparse << ", " << dense_descr
						<< ")\n";
#endif
					// pre-assign coors if output is dense but was previously totally empty
					const bool assign_z = z_nns == 0 && !sparse;

					if( assign_z ) {
#ifdef _DEBUG
						std::cout << "\t\t detected output will be dense while "
							<< "the output vector presently is completely empty. We therefore "
							<< "pre-assign all output coordinates\n";
#endif
#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
						if( !already_dense_output ) {
#endif
							// the result will always be dense
							local_z.local_assignAllNotAlreadyAssigned();
							local_z_nz = local_z.nonzeroes();
#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
						}
#endif
					}

					if( !dense_descr && sparse ) {
						// the below computes loop sizes multiplied with the number of vectors that
						// each loop needs to touch. Possible vectors are: z, m, a, x, and y.
						const size_t mask_factor = masked ? 1 : 0;
						const size_t mul_loop_size = ( 3 + mask_factor ) * std::min(
								( a_scalar ? local_n : local_a_nz ),
								( x_scalar ? local_n : local_x_nz )
							) + ( y_zero ? 0 :
								(2 + mask_factor) * ( y_scalar ? local_n : local_y_nz )
							);
#ifdef _DEBUG
						std::cout << "\t\t mul_loop_size = " << mul_loop_size << "\n";
#endif

						const size_t mask_loop_size = ( y_zero ? 4 : 5 ) * local_m_nz;

						if( masked && mask_loop_size < mul_loop_size ) {
#ifdef _DEBUG
							std::cout << "\t\t mask_loop_size= " << mask_loop_size << "\n";
							std::cout << "\t\t will be driven by output mask\n";
#endif

#ifdef GRB_BOOLEAN_DISPATCHER
							rc = boolean_dispatcher_sparse_eWiseMulAdd_maskDriven<
#else
							rc = sparse_eWiseMulAdd_maskDriven<
#endif
									descr, a_scalar, x_scalar, y_scalar, y_zero
								>(
									already_dense_output, already_dense_mask, already_dense_input_a,
									already_dense_input_x, already_dense_input_y,
									lower_bound, upper_bound,
									local_z, local_m, local_a, local_x, local_y,
									z_vector, *m_vector, a_wrapper, x_wrapper, y_wrapper,
									ring
								);
						} else {
#ifdef _DEBUG
							std::cout << "\t\t will be driven by the multiplication a*x\n";
#endif
							static_assert( !(a_scalar && x_scalar),
								"The case of the multiplication being between two scalars should have"
								"been caught earlier. Please submit a bug report." );

							if( a_scalar ) {
#ifdef GRB_BOOLEAN_DISPATCHER
								rc = boolean_dispatcher_twoPhase_sparse_eWiseMulAdd_mulDriven<
#else
								rc = twoPhase_sparse_eWiseMulAdd_mulDriven<
#endif
										descr, masked, a_scalar, y_scalar, y_zero, true
									>(
										already_dense_output, already_dense_mask, already_dense_input_x,
										already_dense_input_a, already_dense_input_y,
										lower_bound, upper_bound,
										local_z, &local_m, local_x, local_a, local_y,
										z_vector, m_vector, *(x_wrapper.getPointer()), a_wrapper, y_wrapper,
										ring
									);
							} else if( x_scalar ) {
#ifdef GRB_BOOLEAN_DISPATCHER
								rc = boolean_dispatcher_twoPhase_sparse_eWiseMulAdd_mulDriven<
#else
								rc = twoPhase_sparse_eWiseMulAdd_mulDriven<
#endif
										descr, masked, x_scalar, y_scalar, y_zero, false
									>(
										already_dense_output, already_dense_mask, already_dense_input_a,
										already_dense_input_x, already_dense_input_y,
										lower_bound, upper_bound,
										local_z, &local_m, local_a, local_x, local_y,
										z_vector, m_vector, *(a_wrapper.getPointer()), x_wrapper, y_wrapper,
										ring
									);
							} else if( local_a_nz <= local_x_nz ) {
#ifdef GRB_BOOLEAN_DISPATCHER
								rc = boolean_dispatcher_twoPhase_sparse_eWiseMulAdd_mulDriven<
#else
								rc = twoPhase_sparse_eWiseMulAdd_mulDriven<
#endif
										descr, masked, x_scalar, y_scalar, y_zero, false
									>(
										already_dense_output, already_dense_mask, already_dense_input_a,
										already_dense_input_x, already_dense_input_y,
										lower_bound, upper_bound,
										local_z, &local_m, local_a, local_x, local_y,
										z_vector, m_vector, *(a_wrapper.getPointer()), x_wrapper, y_wrapper,
										ring
									);
							} else {
								assert( local_x_nz < local_a_nz );
#ifdef GRB_BOOLEAN_DISPATCHER
								rc = boolean_dispatcher_twoPhase_sparse_eWiseMulAdd_mulDriven<
#else
								rc = twoPhase_sparse_eWiseMulAdd_mulDriven<
#endif
										descr, masked, a_scalar, y_scalar, y_zero, true
									>(
										already_dense_output, already_dense_mask, already_dense_input_x,
										already_dense_input_a, already_dense_input_y,
										lower_bound, upper_bound,
										local_z, &local_m, local_x, local_a, local_y,
										z_vector, m_vector, *(x_wrapper.getPointer()), a_wrapper, y_wrapper,
										ring
									);
							}
						}
					} else {
						// all that remains is the dense case
						assert( a_scalar || local_a_nz == local_n );
						assert( x_scalar || local_x_nz == local_n );
						assert( y_scalar || local_y_nz == local_n );
						assert( ! masked || mask_is_dense );
						assert( local_z_nz == local_n );
#ifdef _DEBUG
						std::cout << "\t\t will perform a dense eWiseMulAdd\n";
#endif
						if( assign_z ) {
							rc = dense_eWiseMulAdd<
									descr, a_scalar, x_scalar, y_scalar, y_zero, true
								>(
									lower_bound, upper_bound,
									z_vector, a_wrapper, x_wrapper, y_wrapper,
									ring
								);
						} else {
							rc = dense_eWiseMulAdd<
									descr, a_scalar, x_scalar, y_scalar, y_zero, false
								>(
									lower_bound, upper_bound,
									z_vector, a_wrapper, x_wrapper, y_wrapper,
									ring
								);
						}
					}

#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
					if( !already_dense_output ) {
#else
					if( !already_dense_vectors ) {
#endif
						internal::getCoordinates( z_vector ).asyncJoinSubset( local_z,
							lower_bound, upper_bound );
					}

					return rc;
				};

			ret = ret ? ret : internal::le.addStage(
					std::move( func ),
					internal::Opcode::BLAS1_EWISEMULADD_DISPATCH,
					n, sizeof( OutputType ), dense_descr, true,
					&z_vector, nullptr, &internal::getCoordinates( z_vector ), nullptr,
					masked ? m_vector : nullptr, a_wrapper.getPointer(),
					x_wrapper.getPointer(), y_wrapper.getPointer(),
					masked ? &internal::getCoordinates( *m_vector ) : nullptr,
					a_wrapper.getCoordinates(), x_wrapper.getCoordinates(),
					y_wrapper.getCoordinates(),
					nullptr
				);

#ifdef _NONBLOCKING_DEBUG
			std::cout << "\t\tStage added to a pipeline: eWiseMulAdd_dispatch"
				<< std::endl;
#endif
			return ret;
		}

	} // namespace internal

	template<
		Descriptor descr = descriptors::no_operation,
		class Ring,
		typename InputType1,
		typename InputType2,
		typename InputType3,
		typename OutputType,
		typename Coords
	>
	RC eWiseMulAdd(
		Vector< OutputType, nonblocking, Coords > &z,
		const InputType1 alpha,
		const Vector< InputType2, nonblocking, Coords > &x,
		const Vector< InputType3, nonblocking, Coords > &y,
		const Ring &ring = Ring(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
				!grb::is_object< OutputType >::value &&
				!grb::is_object< InputType1 >::value &&
				!grb::is_object< InputType2 >::value &&
				!grb::is_object< InputType3 >::value &&
				grb::is_semiring< Ring >::value,
			void
		>::type * const = nullptr
	) {
		// static sanity checks
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Ring::D1, InputType1 >::value ),
			"grb::eWiseMulAdd",
			"called with a left-hand scalar alpha of an element type that does not "
			"match the first domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Ring::D2, InputType2 >::value ),
			"grb::eWiseMulAdd",
			"called with a right-hand vector x with an element type that does not "
			"match the second domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Ring::D4, InputType3 >::value ),
			"grb::eWiseMulAdd",
			"called with an additive vector y with an element type that does not "
			"match the fourth domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Ring::D4, OutputType >::value ),
			"grb::eWiseMulAdd",
			"called with a result vector z with an element type that does not match "
			"the fourth domain of the given semiring" );

		// dynamic sanity checks
		const size_t n = size( z );
		if( size( x ) != n || size( y ) != n ) {
			return MISMATCH;
		}
		if( phase == RESIZE ) {
			return SUCCESS;
		}
		assert( phase == EXECUTE );

		// catch trivial cases
		const InputType1 zeroIT1 = ring.template getZero< InputType1 >();
		if( alpha == zeroIT1 ) {
			return foldl< descr >( z, y, ring.getAdditiveMonoid() );
		}

		const Vector< bool, nonblocking, Coords > * const null_mask = nullptr;

		const internal::Wrapper< true, InputType1, Coords > a_wrapper( alpha );
		const internal::Wrapper< false, InputType2, Coords > x_wrapper( x );
		const internal::Wrapper< false, InputType3, Coords > y_wrapper( y );

		// sparse or dense case
		return internal::eWiseMulAdd_dispatch<
				descr, false, true, false, false, false,
				bool, Ring, InputType1, InputType2, InputType3, OutputType, Coords
			>( z,  null_mask, a_wrapper, x_wrapper, y_wrapper, n, ring );
	}

	template<
		Descriptor descr = descriptors::no_operation,
		class Ring,
		typename InputType1,
		typename InputType2,
		typename InputType3,
		typename OutputType,
		typename Coords
	>
	RC eWiseMulAdd(
		Vector< OutputType, nonblocking, Coords > &z,
		const Vector< InputType1, nonblocking, Coords > &a,
		const InputType2 chi,
		const Vector< InputType3, nonblocking, Coords > &y,
		const Ring &ring = Ring(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			!grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			!grb::is_object< InputType3 >::value &&
			grb::is_semiring< Ring >::value,
		void >::type * const = nullptr
	) {
		// static sanity checks
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Ring::D1, InputType1 >::value ),
			"grb::eWiseMulAdd",
			"called with a left-hand scalar alpha of an element type that does not "
			"match the first domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Ring::D2, InputType2 >::value ),
			"grb::eWiseMulAdd",
			"called with a right-hand vector x with an element type that does not "
			"match the second domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Ring::D4, InputType3 >::value ),
			"grb::eWiseMulAdd",
			"called with an additive vector y with an element type that does not "
			"match the fourth domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Ring::D4, OutputType >::value ),
			"grb::eWiseMulAdd",
			"called with a result vector z with an element type that does not match "
			"the fourth domain of the given semiring" );

		// dynamic sanity checks
		const size_t n = size( z );
		if( size( a ) != n || size( y ) != n ) {
			return MISMATCH;
		}

		if( phase == RESIZE ) {
			return SUCCESS;
		}
		assert( phase == EXECUTE );

		// catch trivial cases
		const InputType1 zeroIT2 = ring.template getZero< InputType2 >();
		if( chi == zeroIT2 ) {
			return foldl< descr >( z, y, ring.getAdditiveMonoid() );
		}

		const Vector< bool, nonblocking, Coords > * const null_mask = nullptr;

		const internal::Wrapper< false, InputType1, Coords > a_wrapper( a );
		const internal::Wrapper< true, InputType2, Coords > x_wrapper( chi );
		const internal::Wrapper< false, InputType3, Coords > y_wrapper( y );

		// sparse or dense case
		return internal::eWiseMulAdd_dispatch<
				descr, false, false, true, false, false,
				bool, Ring, InputType1, InputType2, InputType3, OutputType, Coords
			>( z,  null_mask, a_wrapper, x_wrapper, y_wrapper, n, ring );
	}

	template<
		Descriptor descr = descriptors::no_operation,
		bool y_zero = false,
		class Ring,
		typename InputType1,
		typename InputType2,
		typename InputType3,
		typename OutputType,
		typename Coords
	>
	RC eWiseMulAdd(
		Vector< OutputType, nonblocking, Coords > &z,
		const Vector< InputType1, nonblocking, Coords > &a,
		const Vector< InputType2, nonblocking, Coords > &x,
		const InputType3 gamma,
		const Ring &ring = Ring(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			!grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			!grb::is_object< InputType3 >::value &&
			grb::is_semiring< Ring >::value,
		void >::type * const = nullptr
	) {
		// static sanity checks
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Ring::D1, InputType1 >::value ),
			"grb::eWiseMulAdd",
			"called with a left-hand scalar alpha of an element type that does not "
			"match the first domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Ring::D2, InputType2 >::value ),
			"grb::eWiseMulAdd",
			"called with a right-hand vector x with an element type that does not "
			"match the second domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Ring::D4, InputType3 >::value ),
			"grb::eWiseMulAdd",
			"called with an additive vector y with an element type that does not "
			"match the fourth domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Ring::D4, OutputType >::value ),
			"grb::eWiseMulAdd",
			"called with a result vector z with an element type that does not match "
			"the fourth domain of the given semiring" );

		// dynamic sanity checks
		const size_t n = size( z );
		if( size( a ) != n || size( x ) != n ) {
			return MISMATCH;
		}

		if( phase == RESIZE ) {
			return SUCCESS;
		}
		assert( phase == EXECUTE );

		const Vector< bool, nonblocking, Coords > * const null_mask = nullptr;

		const internal::Wrapper< false, InputType1, Coords > a_wrapper( a );
		const internal::Wrapper< false, InputType2, Coords > x_wrapper( x );
		const internal::Wrapper< true, InputType3, Coords > y_wrapper( gamma );

		// sparse or dense case
		return internal::eWiseMulAdd_dispatch<
				descr, false, false, false, true, y_zero,
				bool, Ring, InputType1, InputType2, InputType3, OutputType, Coords
			>( z, null_mask, a_wrapper, x_wrapper, y_wrapper, n, ring );
	}

	template<
		Descriptor descr = descriptors::no_operation,
		bool y_zero = false,
		class Ring,
		typename InputType1,
		typename InputType2,
		typename InputType3,
		typename OutputType,
		typename Coords
	>
	RC eWiseMulAdd(
		Vector< OutputType, nonblocking, Coords > &z,
		const Vector< InputType1, nonblocking, Coords > &a,
		const InputType2 beta,
		const InputType3 gamma,
		const Ring &ring = Ring(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			!grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			!grb::is_object< InputType3 >::value &&
			grb::is_semiring< Ring >::value,
		void >::type * const = nullptr
	) {
		// static sanity checks
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Ring::D1, InputType1 >::value ),
			"grb::eWiseMulAdd",
			"called with a left-hand scalar alpha of an element type that does not "
			"match the first domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Ring::D2, InputType2 >::value ),
			"grb::eWiseMulAdd",
			"called with a right-hand vector x with an element type that does not "
			"match the second domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Ring::D4, InputType3 >::value ),
			"grb::eWiseMulAdd",
			"called with an additive vector y with an element type that does not "
			"match the fourth domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Ring::D4, OutputType >::value ),
			"grb::eWiseMulAdd",
			"called with a result vector z with an element type that does not match "
			"the fourth domain of the given semiring" );

		// dynamic sanity checks
		const size_t n = size( z );
		if( size( a ) != n ) {
			return MISMATCH;
		}

		if( phase == RESIZE ) {
			return SUCCESS;
		}
		assert( phase == EXECUTE );

		// catch trivial dispatches
		const InputType2 zeroIT2 = ring.template getZero< InputType2 >();
		if( beta == zeroIT2 ) {
			return foldl< descr >( z, gamma, ring.getAdditiveMonoid() );
		}

		const Vector< bool, nonblocking, Coords > * const null_mask = nullptr;

		const internal::Wrapper< false, InputType1, Coords > a_wrapper( a );
		const internal::Wrapper< true, InputType2, Coords > x_wrapper( beta );
		const internal::Wrapper< true, InputType3, Coords > y_wrapper( gamma );

		// sparse or dense case
		return internal::eWiseMulAdd_dispatch<
				descr, false, false, true, true, y_zero,
				bool, Ring, InputType1, InputType2, InputType3, OutputType, Coords
			>( z,  null_mask, a_wrapper, x_wrapper, y_wrapper, n, ring );
	}

	template<
		Descriptor descr = descriptors::no_operation,
		bool y_zero = false,
		class Ring,
		typename InputType1,
		typename InputType2,
		typename InputType3,
		typename OutputType,
		typename Coords
	>
	RC eWiseMulAdd(
		Vector< OutputType, nonblocking, Coords > &z,
		const InputType1 alpha,
		const Vector< InputType2, nonblocking, Coords > &x,
		const InputType3 gamma,
		const Ring &ring = Ring(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if< !grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			!grb::is_object< InputType3 >::value &&
			grb::is_semiring< Ring >::value, void
		>::type * const = nullptr
	) {
		// static sanity checks
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Ring::D1, InputType1 >::value ),
			"grb::eWiseMulAdd",
			"called with a left-hand scalar alpha of an element type that does not "
			"match the first domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Ring::D2, InputType2 >::value ),
			"grb::eWiseMulAdd",
			"called with a right-hand vector x with an element type that does not "
			"match the second domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Ring::D4, InputType3 >::value ),
			"grb::eWiseMulAdd",
			"called with an additive vector y with an element type that does not "
			"match the fourth domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Ring::D4, OutputType >::value ),
			"grb::eWiseMulAdd",
			"called with a result vector z with an element type that does not match "
			"the fourth domain of the given semiring" );

		// dynamic sanity checks
		const size_t n = size( z );
		if( size( x ) != n ) {
			return MISMATCH;
		}

		if( phase == RESIZE ) {
			return SUCCESS;
		}
		assert( phase == EXECUTE );

		// catch trivial cases
		const InputType1 zeroIT1 = ring.template getZero< InputType1 >();
		if( alpha == zeroIT1 ) {
			return foldl< descr >( z, gamma, ring.getAdditiveMonoid() );
		}

		const Vector< bool, nonblocking, Coords > * null_mask = nullptr;

		const internal::Wrapper< true, InputType1, Coords > a_wrapper( alpha );
		const internal::Wrapper< false, InputType2, Coords > x_wrapper( x );
		const internal::Wrapper< true, InputType3, Coords > y_wrapper( gamma );

		// sparse or dense case
		return internal::eWiseMulAdd_dispatch<
				descr, false, true, false, true, y_zero,
				bool, Ring, InputType1, InputType2, InputType3, OutputType, Coords
			>( z,  null_mask, a_wrapper, x_wrapper, y_wrapper, n, ring );
	}

	template<
		Descriptor descr = descriptors::no_operation,
		class Ring,
		typename OutputType,
		typename InputType1,
		typename InputType2,
		typename InputType3,
		typename Coords
	>
	RC eWiseMulAdd(
		Vector< OutputType, nonblocking, Coords > &z,
		const InputType1 alpha,
		const InputType2 beta,
		const Vector< InputType3, nonblocking, Coords > &y,
		const Ring &ring = Ring(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			!grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			!grb::is_object< InputType3 >::value &&
			grb::is_semiring< Ring >::value,
		void >::type * const = nullptr
	) {
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Ring::D1, InputType1 >::value ),
			"grb::eWiseMulAdd(vector,scalar,scalar,scalar)",
			"First domain of semiring does not match first input type" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Ring::D2, InputType2 >::value ),
			"grb::eWiseMulAdd(vector,scalar,scalar,scalar)",
			"Second domain of semiring does not match second input type" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Ring::D4, InputType3 >::value ),
			"grb::eWiseMulAdd(vector,scalar,scalar,scalar)",
			"Fourth domain of semiring does not match third input type" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Ring::D4, OutputType >::value ),
			"grb::eWiseMulAdd(vector,scalar,scalar,scalar)",
			"Fourth domain of semiring does not match output type" );
#ifdef _DEBUG
		std::cout << "eWiseMulAdd (nonblocking, vector <- scalar x scalar + vector) "
			<< "precomputes scalar multiply and dispatches to eWiseAdd (nonblocking, "
			<< "vector <- scalar + vector)\n";
#endif
		// dynamic checks
		const size_t n = size( z );
		if( size( y ) != n ) { return MISMATCH; }

		typename Ring::D3 mul_result;
		RC rc = grb::apply( mul_result, alpha, beta,
			ring.getMultiplicativeOperator() );
#ifdef NDEBUG
		(void) rc;
#else
		assert( rc == SUCCESS );
#endif
		return eWiseAdd< descr >( z, mul_result, y, ring, phase );
	}

	template<
		Descriptor descr = descriptors::no_operation,
		class Ring,
		typename OutputType,
		typename InputType1,
		typename InputType2,
		typename InputType3,
		typename Coords
	>
	RC eWiseMulAdd(
		Vector< OutputType, nonblocking, Coords > &z,
		const InputType1 alpha,
		const InputType2 beta,
		const InputType3 gamma,
		const Ring &ring = Ring(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if< !grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			!grb::is_object< InputType3 >::value &&
			grb::is_semiring< Ring >::value, void
		>::type * const = nullptr
	) {
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< typename Ring::D1, InputType1 >::value ),
			"grb::eWiseMulAdd(vector,scalar,scalar,scalar)",
			"First domain of semiring does not match first input type" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< typename Ring::D2, InputType2 >::value ),
			"grb::eWiseMulAdd(vector,scalar,scalar,scalar)",
			"Second domain of semiring does not match second input type" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< typename Ring::D4, InputType3 >::value ),
			"grb::eWiseMulAdd(vector,scalar,scalar,scalar)",
			"Fourth domain of semiring does not match third input type" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< typename Ring::D4, OutputType >::value ),
			"grb::eWiseMulAdd(vector,scalar,scalar,scalar)",
			"Fourth domain of semiring does not match output type" );
#ifdef _DEBUG
		std::cout << "eWiseMulAdd (nonblocking, vector <- scalar x scalar + scalar) "
			<< "precomputes scalar operations and dispatches to set (nonblocking)\n";
#endif
		typename Ring::D3 mul_result;
		RC rc = grb::apply( mul_result, alpha, beta,
			ring.getMultiplicativeOperator() );
#ifdef NDEBUG
		(void) rc;
#endif
		assert( rc == SUCCESS );
		typename Ring::D4 add_result;
		rc = grb::apply( add_result, mul_result, gamma, ring.getAdditiveOperator() );
#ifdef NDEBUG
		(void) rc;
#endif
		assert( rc == SUCCESS );
		return grb::foldl< descr >( z, add_result, ring.getAdditiveMonoid(), phase );
	}

	template<
		Descriptor descr = descriptors::no_operation,
		class Ring,
		typename InputType1,
		typename InputType2,
		typename InputType3,
		typename OutputType,
		typename Coords
	>
	RC eWiseMulAdd(
		Vector< OutputType, nonblocking, Coords > &z,
		const Vector< InputType1, nonblocking, Coords > &a,
		const Vector< InputType2, nonblocking, Coords > &x,
		const Vector< InputType3, nonblocking, Coords > &y,
		const Ring &ring = Ring(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			!grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			!grb::is_object< InputType3 >::value &&
			grb::is_semiring< Ring >::value,
		void >::type * const = nullptr
	) {
		(void) ring;
		// static sanity checks
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Ring::D1, InputType1 >::value ),
			"grb::eWiseMulAdd",
			"called with a left-hand vector a with an element type that does not "
			"match the first domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Ring::D2, InputType2 >::value ),
			"grb::eWiseMulAdd",
			"called with a right-hand vector x with an element type that does not "
			"match the second domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Ring::D4, InputType3 >::value ),
			"grb::eWiseMulAdd",
			"called with an additive vector y with an element type that does not "
			"match the fourth domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Ring::D4, OutputType >::value ),
			"grb::eWiseMulAdd",
			"called with a result vector z with an element type that does not match "
			"the fourth domain of the given semiring" );

		// dynamic sanity checks
		const size_t n = size( z );
		if( size( x ) != n || size( y ) != n || size( a ) != n ) {
			return MISMATCH;
		}

		if( phase == RESIZE ) {
			return SUCCESS;
		}
		assert( phase == EXECUTE );

		const Vector< bool, nonblocking, Coords > * const null_mask = nullptr;

		const internal::Wrapper< false, InputType1, Coords > a_wrapper( a );
		const internal::Wrapper< false, InputType2, Coords > x_wrapper( x );
		const internal::Wrapper< false, InputType3, Coords > y_wrapper( y );

		// sparse or dense case
		return internal::eWiseMulAdd_dispatch<
				descr, false, false, false, false, false,
				bool, Ring, InputType1, InputType2, InputType3, OutputType, Coords
			>( z, null_mask, a_wrapper, x_wrapper, y_wrapper, n, ring );
	}

	template<
		Descriptor descr = descriptors::no_operation,
		class Ring,
		typename InputType1,
		typename InputType2,
		typename InputType3,
		typename OutputType,
		typename MaskType,
		typename Coords
	>
	RC eWiseMulAdd(
		Vector< OutputType, nonblocking, Coords > &z,
		const Vector< MaskType, nonblocking, Coords > &m,
		const InputType1 alpha,
		const Vector< InputType2, nonblocking, Coords > &x,
		const Vector< InputType3, nonblocking, Coords > &y,
		const Ring &ring = Ring(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			!grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			!grb::is_object< InputType3 >::value &&
			grb::is_semiring< Ring >::value &&
			!grb::is_object< MaskType >::value,
		void >::type * const = nullptr
	) {
		// static sanity checks
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Ring::D1, InputType1 >::value ),
			"grb::eWiseMulAdd",
			"called with a left-hand scalar alpha of an element type that does not "
			"match the first domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Ring::D2, InputType2 >::value ),
			"grb::eWiseMulAdd",
			"called with a right-hand vector x with an element type that does not "
			"match the second domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Ring::D4, InputType3 >::value ),
			"grb::eWiseMulAdd",
			"called with an additive vector y with an element type that does not "
			"match the fourth domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Ring::D4, OutputType >::value ),
			"grb::eWiseMulAdd",
			"called with a result vector z with an element type that does not match "
			"the fourth domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< bool, MaskType >::value ),
			"grb::eWiseMulAdd",
			"called with a mask vector m with a non-bool element type" );

		// catch empty mask
		if( size( m ) == 0 ) {
			return eWiseMulAdd< descr >( z, alpha, x, y, ring, phase );
		}

		// dynamic sanity checks
		const size_t n = size( z );
		if( size( x ) != n || size( y ) != n || size( m ) != n ) {
			return MISMATCH;
		}

		if( phase == RESIZE ) {
			return SUCCESS;
		}
		assert( phase == EXECUTE );

		// catch trivial cases
		const InputType1 zeroIT1 = ring.template getZero< InputType1 >();
		if( alpha == zeroIT1 ) {
			return foldl< descr >( z, m, y, ring.getAdditiveMonoid() );
		}

		const internal::Wrapper< true, InputType1, Coords > a_wrapper( alpha );
		const internal::Wrapper< false, InputType2, Coords > x_wrapper( x );
		const internal::Wrapper< false, InputType3, Coords > y_wrapper( y );

		// sparse or dense case
		return internal::eWiseMulAdd_dispatch<
				descr, true, true, false, false, false,
				bool, Ring, InputType1, InputType2, InputType3, OutputType, Coords
			>( z, &m, a_wrapper, x_wrapper, y_wrapper, n, ring );
	}

	template<
		Descriptor descr = descriptors::no_operation,
		class Ring,
		typename InputType1,
		typename InputType2,
		typename InputType3,
		typename OutputType,
		typename MaskType,
		typename Coords
	>
	RC eWiseMulAdd(
		Vector< OutputType, nonblocking, Coords > &z,
		const Vector< MaskType, nonblocking, Coords > &m,
		const Vector< InputType1, nonblocking, Coords > &a,
		const InputType2 chi,
		const Vector< InputType3, nonblocking, Coords > &y,
		const Ring &ring = Ring(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			!grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			!grb::is_object< InputType3 >::value &&
			grb::is_semiring< Ring >::value &&
			!grb::is_object< MaskType >::value,
		void >::type * const = nullptr
	) {
		// static sanity checks
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Ring::D1, InputType1 >::value ),
			"grb::eWiseMulAdd",
			"called with a left-hand scalar alpha of an element type that does not "
			"match the first domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Ring::D2, InputType2 >::value ),
			"grb::eWiseMulAdd",
			"called with a right-hand vector x with an element type that does not "
			"match the second domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Ring::D4, InputType3 >::value ),
			"grb::eWiseMulAdd",
			"called with an additive vector y with an element type that does not "
			"match the fourth domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Ring::D4, OutputType >::value ),
			"grb::eWiseMulAdd",
			"called with a result vector z with an element type that does not match "
			"the fourth domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< bool, MaskType >::value ),
			"grb::eWiseMulAdd",
			"called with a mask vector m with a non-bool element type" );

		// catch empty mask
		if( size( m ) == 0 ) {
			return eWiseMulAdd< descr >( z, a, chi, y, ring, phase );
		}

		// dynamic sanity checks
		const size_t n = size( z );
		if( size( a ) != n || size( y ) != n || size( m ) != n ) {
			return MISMATCH;
		}

		if( phase == RESIZE ) {
			return SUCCESS;
		}
		assert( phase == EXECUTE );

		// catch trivial cases
		const InputType1 zeroIT2 = ring.template getZero< InputType2 >();
		if( chi == zeroIT2 ) {
			return foldl< descr >( z, m, y, ring.getAdditiveMonoid() );
		}

		const internal::Wrapper< false, InputType1, Coords > a_wrapper( a );
		const internal::Wrapper< true, InputType2, Coords > x_wrapper( chi );
		const internal::Wrapper< false, InputType3, Coords > y_wrapper( y );

		// sparse or dense case
		return internal::eWiseMulAdd_dispatch<
				descr, true, false, true, false, false,
				bool, Ring, InputType1, InputType2, InputType3, OutputType, Coords
			>( z, &m, a_wrapper, x_wrapper, y_wrapper, n, ring );
	}

	template<
		Descriptor descr = descriptors::no_operation,
		bool y_zero = false,
		class Ring,
		typename InputType1,
		typename InputType2,
		typename InputType3,
		typename OutputType,
		typename MaskType,
		typename Coords
	>
	RC eWiseMulAdd(
		Vector< OutputType, nonblocking, Coords > &z,
		const Vector< MaskType, nonblocking, Coords > &m,
		const Vector< InputType1, nonblocking, Coords > &a,
		const Vector< InputType2, nonblocking, Coords > &x,
		const InputType3 gamma,
		const Ring &ring = Ring(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if< !grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			!grb::is_object< InputType3 >::value &&
			grb::is_semiring< Ring >::value &&
			!grb::is_object< MaskType >::value, void
		>::type * const = nullptr
	) {
		// static sanity checks
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Ring::D1, InputType1 >::value ),
			"grb::eWiseMulAdd",
			"called with a left-hand scalar alpha of an element type that does not "
			"match the first domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Ring::D2, InputType2 >::value ),
			"grb::eWiseMulAdd",
			"called with a right-hand vector x with an element type that does not "
			"match the second domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Ring::D4, InputType3 >::value ),
			"grb::eWiseMulAdd",
			"called with an additive vector y with an element type that does not "
			"match the fourth domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Ring::D4, OutputType >::value ),
			"grb::eWiseMulAdd",
			"called with a result vector z with an element type that does not match "
			"the fourth domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< bool, MaskType >::value ),
			"grb::eWiseMulAdd",
			"called with a mask vector m with a non-bool element type" );

		// catch empty mask
		if( size( m ) == 0 ) {
			return eWiseMulAdd< descr, y_zero >( z, a, x, gamma, ring, phase );
		}

		// dynamic sanity checks
		const size_t n = size( z );
		if( size( a ) != n || size( x ) != n || size( m ) != n ) {
			return MISMATCH;
		}

		if( phase == RESIZE ) {
			return SUCCESS;
		}
		assert( phase == EXECUTE );

		const internal::Wrapper< false, InputType1, Coords > a_wrapper( a );
		const internal::Wrapper< false, InputType2, Coords > x_wrapper( x );
		const internal::Wrapper< true, InputType3, Coords > y_wrapper( gamma );

		// sparse or dense case
		return internal::eWiseMulAdd_dispatch<
				descr, true, false, false, true, y_zero,
				bool, Ring, InputType1, InputType2, InputType3, OutputType, Coords
			>( z, &m, a_wrapper, x_wrapper, y_wrapper, n, ring );
	}

	template<
		Descriptor descr = descriptors::no_operation,
		bool y_zero = false,
		class Ring,
		typename InputType1,
		typename InputType2,
		typename InputType3,
		typename OutputType,
		typename MaskType,
		typename Coords
	>
	RC eWiseMulAdd(
		Vector< OutputType, nonblocking, Coords > &z,
		const Vector< MaskType, nonblocking, Coords > &m,
		const Vector< InputType1, nonblocking, Coords > &a,
		const InputType2 beta,
		const InputType3 gamma,
		const Ring &ring = Ring(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			!grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			!grb::is_object< InputType3 >::value &&
			grb::is_semiring< Ring >::value &&
			!grb::is_object< MaskType >::value,
		void >::type * const = nullptr
	) {
		// static sanity checks
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Ring::D1, InputType1 >::value ),
			"grb::eWiseMulAdd",
			"called with a left-hand scalar alpha of an element type that does not "
			"match the first domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Ring::D2, InputType2 >::value ),
			"grb::eWiseMulAdd",
			"called with a right-hand vector x with an element type that does not "
			"match the second domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Ring::D4, InputType3 >::value ),
			"grb::eWiseMulAdd",
			"called with an additive vector y with an element type that does not "
			"match the fourth domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Ring::D4, OutputType >::value ),
			"grb::eWiseMulAdd",
			"called with a result vector z with an element type that does not match "
			"the fourth domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< bool, MaskType >::value ),
			"grb::eWiseMulAdd",
			"called with a mask vector m with a non-bool element type" );

		// catch empty mask
		if( size( m ) == 0 ) {
			return eWiseMulAdd< descr, y_zero >( z, a, beta, gamma, ring, phase );
		}

		// dynamic sanity checks
		const size_t n = size( z );
		if( size( a ) != n || size( m ) != n ) {
			return MISMATCH;
		}

		if( phase == RESIZE ) {
			return SUCCESS;
		}
		assert( phase == EXECUTE );

		// catch trivial dispatch
		const InputType2 zeroIT2 = ring.template getZero< InputType2 >();
		if( zeroIT2 == beta ) {
#ifdef _DEBUG
			std::cout << "eWiseMulAdd (nonblocking, masked, vector<-vector<-scalar<-"
				<< "scalar) dispatches to foldl\n";
#endif
			return foldl< descr >( z, m, gamma, ring.getAdditiveMonoid() );
		}

		const internal::Wrapper< false, InputType1, Coords > a_wrapper( a );
		const internal::Wrapper< true, InputType2, Coords > x_wrapper( beta );
		const internal::Wrapper< true, InputType3, Coords > y_wrapper( gamma );

		// sparse or dense case
		return internal::eWiseMulAdd_dispatch<
				descr, true, false, true, true, y_zero,
				bool, Ring, InputType1, InputType2, InputType3, OutputType, Coords
			>( z, &m, a_wrapper, x_wrapper, y_wrapper, n, ring );
	}

	template<
		Descriptor descr = descriptors::no_operation,
		bool y_zero = false,
		class Ring,
		typename InputType1,
		typename InputType2,
		typename InputType3,
		typename OutputType,
		typename MaskType,
		typename Coords
	>
	RC eWiseMulAdd(
		Vector< OutputType, nonblocking, Coords > &z,
		const Vector< MaskType, nonblocking, Coords > &m,
		const InputType1 alpha,
		const Vector< InputType2, nonblocking, Coords > &x,
		const InputType3 gamma,
		const Ring &ring = Ring(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			!grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			!grb::is_object< InputType3 >::value &&
			grb::is_semiring< Ring >::value &&
			!grb::is_object< MaskType >::value,
		void >::type * const = nullptr
	) {
		// static sanity checks
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Ring::D1, InputType1 >::value ),
			"grb::eWiseMulAdd",
			"called with a left-hand scalar alpha of an element type that does not "
			"match the first domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Ring::D2, InputType2 >::value ),
			"grb::eWiseMulAdd",
			"called with a right-hand vector x with an element type that does not "
			"match the second domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Ring::D4, InputType3 >::value ),
			"grb::eWiseMulAdd",
			"called with an additive vector y with an element type that does not "
			"match the fourth domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Ring::D4, OutputType >::value ),
			"grb::eWiseMulAdd",
			"called with a result vector z with an element type that does not match "
			"the fourth domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< bool, MaskType >::value ),
			"grb::eWiseMulAdd",
			"called with a mask vector m with a non-bool element type" );

		// catch empty mask
		if( size( m ) == 0 ) {
			return eWiseMulAdd< descr, y_zero >( z, alpha, x, gamma, ring, phase );
		}

		// dynamic sanity checks
		const size_t n = size( z );
		if( size( x ) != n || size( m ) != n ) {
			return MISMATCH;
		}

		if( phase == RESIZE ) {
			return SUCCESS;
		}
		assert( phase == EXECUTE );

		// catch trivial dispatch
		const InputType1 zeroIT1 = ring.template getZero< InputType1 >();
		if( alpha == zeroIT1 ) {
#ifdef _DEBUG
			std::cout << "eWiseMulAdd (nonblocking, masked, vector<-scalar<-scalar<-"
				<< "scalar) dispatches to foldl\n";
#endif
			return foldl< descr >( z, m, gamma, ring.getAdditiveMonoid() );
		}

		const internal::Wrapper< true, InputType1, Coords > a_wrapper( alpha );
		const internal::Wrapper< false, InputType2, Coords > x_wrapper( x );
		const internal::Wrapper< true, InputType3, Coords > y_wrapper( gamma );

		// sparse or dense case
		return internal::eWiseMulAdd_dispatch<
				descr, true, true, false, true, y_zero,
				bool, Ring, InputType1, InputType2, InputType3, OutputType, Coords
			>( z, &m, a_wrapper, x_wrapper, y_wrapper, n, ring );
	}

	template<
		Descriptor descr = descriptors::no_operation,
		class Ring,
		typename InputType1,
		typename InputType2,
		typename InputType3,
		typename OutputType,
		typename MaskType,
		typename Coords
	>
	RC eWiseMulAdd(
		Vector< OutputType, nonblocking, Coords > &z,
		const Vector< MaskType, nonblocking, Coords > &m,
		const Vector< InputType1, nonblocking, Coords > &a,
		const Vector< InputType2, nonblocking, Coords > &x,
		const Vector< InputType3, nonblocking, Coords > &y,
		const Ring &ring = Ring(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			!grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			!grb::is_object< InputType3 >::value &&
			grb::is_semiring< Ring >::value &&
			!grb::is_object< MaskType >::value,
		void >::type * const = nullptr
	) {
		// static sanity checks
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Ring::D1, InputType1 >::value ),
			"grb::eWiseMulAdd",
			"called with a left-hand vector a with an element type that does not "
			"match the first domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Ring::D2, InputType2 >::value ),
			"grb::eWiseMulAdd",
			"called with a right-hand vector x with an element type that does not "
			"match the second domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Ring::D4, InputType3 >::value ),
			"grb::eWiseMulAdd",
			"called with an additive vector y with an element type that does not "
			"match the fourth domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Ring::D4, OutputType >::value ),
			"grb::eWiseMulAdd",
			"called with a result vector z with an element type that does not match "
			"the fourth domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< bool, MaskType >::value ),
			"grb::eWiseMulAdd",
			"called with a mask vector m with a non-bool element type" );

		// catch empty mask
		if( size( m ) == 0 ) {
			return eWiseMulAdd< descr >( z, a, x, y, ring, phase );
		}

		// dynamic sanity checks
		const size_t n = size( z );
		if( size( x ) != n || size( y ) != n || size( a ) != n || size( m ) != n ) {
			return MISMATCH;
		}

		if( phase == RESIZE ) {
			return SUCCESS;
		}
		assert( phase == EXECUTE );

		const internal::Wrapper< false, InputType1, Coords > a_wrapper( a );
		const internal::Wrapper< false, InputType2, Coords > x_wrapper( x );
		const internal::Wrapper< false, InputType3, Coords > y_wrapper( y );

		// sparse or dense case
		return internal::eWiseMulAdd_dispatch<
				descr, true, false, false, false, false,
				bool, Ring, InputType1, InputType2, InputType3, OutputType, Coords
			>( z, &m, a_wrapper, x_wrapper, y_wrapper, n, ring );
	}

	template<
		Descriptor descr = descriptors::no_operation,
		class Ring,
		typename OutputType,
		typename MaskType,
		typename InputType1,
		typename InputType2,
		typename InputType3,
		typename Coords
	>
	RC eWiseMulAdd(
		Vector< OutputType, nonblocking, Coords > &z,
		const Vector< MaskType, nonblocking, Coords > &m,
		const InputType1 alpha,
		const InputType2 beta,
		const Vector< InputType3, nonblocking, Coords > &y,
		const Ring &ring = Ring(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if< !grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			!grb::is_object< InputType3 >::value &&
			grb::is_semiring< Ring >::value &&
			!grb::is_object< MaskType >::value, void
		>::type * const = nullptr
	) {
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Ring::D1, InputType1 >::value ),
			"grb::eWiseMulAdd(vector,scalar,scalar,scalar)",
			"First domain of semiring does not match first input type" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Ring::D2, InputType2 >::value ),
			"grb::eWiseMulAdd(vector,scalar,scalar,scalar)",
			"Second domain of semiring does not match second input type" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Ring::D4, InputType3 >::value ),
			"grb::eWiseMulAdd(vector,scalar,scalar,scalar)",
			"Fourth domain of semiring does not match third input type" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Ring::D4, OutputType >::value ),
			"grb::eWiseMulAdd(vector,scalar,scalar,scalar)",
			"Fourth domain of semiring does not match output type" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< bool, MaskType >::value ),
			"grb::eWiseMulAdd",
			"called with a mask vector with a non-bool element type" );
#ifdef _DEBUG
		std::cout << "eWiseMulAdd (nonblocking, vector <- scalar x scalar + vector, "
			<< "masked) precomputes scalar multiply and dispatches to eWiseAdd "
			<< "(nonblocking, vector <- scalar + vector, masked)\n";
#endif
		// dynamic checks
		const size_t n = size( z );
		if( size( m ) != n || size( y ) != n ) {
			return MISMATCH;
		}

		typename Ring::D3 mul_result;
		RC rc = grb::apply( mul_result, alpha, beta,
			ring.getMultiplicativeOperator() );
#ifdef NDEBUG
		(void) rc;
#else
		assert( rc == SUCCESS );
#endif
		return grb::eWiseAdd< descr >( z, m, mul_result, y, ring, phase );
	}

	template<
		Descriptor descr = descriptors::no_operation,
		class Ring,
		typename OutputType,
		typename MaskType,
		typename InputType1,
		typename InputType2,
		typename InputType3,
		typename Coords
	>
	RC eWiseMulAdd(
		Vector< OutputType, nonblocking, Coords > &z,
		const Vector< MaskType, nonblocking, Coords > &m,
		const InputType1 alpha,
		const InputType2 beta,
		const InputType3 gamma,
		const Ring &ring = Ring(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			!grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			!grb::is_object< InputType3 >::value &&
			grb::is_semiring< Ring >::value,
		void >::type * const = nullptr
	) {
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Ring::D1, InputType1 >::value ),
			"grb::eWiseMulAdd(vector,scalar,scalar,scalar)",
			"First domain of semiring does not match first input type" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Ring::D2, InputType2 >::value ),
			"grb::eWiseMulAdd(vector,scalar,scalar,scalar)",
			"Second domain of semiring does not match second input type" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Ring::D4, InputType3 >::value ),
			"grb::eWiseMulAdd(vector,scalar,scalar,scalar)",
			"Fourth domain of semiring does not match third input type" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Ring::D4, OutputType >::value ),
			"grb::eWiseMulAdd(vector,scalar,scalar,scalar)",
			"Fourth domain of semiring does not match output type" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< bool, MaskType >::value ),
			"grb::eWiseMulAdd",
			"called with a mask vector with a non-bool element type" );
#ifdef _DEBUG
		std::cout << "eWiseMulAdd (nonblocking, vector <- scalar x scalar + scalar, "
			<< "masked) precomputes scalar operations and dispatches to foldl "
			<< "(nonblocking, masked)\n";
#endif
		// dynamic checks
		const size_t n = size( z );
		if( size( m ) != n ) {
			return MISMATCH;
		}

		typename Ring::D3 mul_result;
		RC rc = grb::apply( mul_result, alpha, beta,
			ring.getMultiplicativeOperator() );
		assert( rc == SUCCESS );
#ifdef NDEBUG
		(void) rc;
#endif
		typename Ring::D4 add_result;
		rc = grb::apply( add_result, mul_result, gamma, ring.getAdditiveOperator() );
		assert( rc == SUCCESS );
#ifdef NDEBUG
		(void) rc;
#endif
		return grb::foldl( z, m, add_result, ring.getAdditiveMonoid(), phase );
	}

	template<
		Descriptor descr = descriptors::no_operation,
		class Ring,
		typename InputType1,
		typename InputType2,
		typename OutputType,
		typename Coords
	>
	RC eWiseMul(
		Vector< OutputType, nonblocking, Coords > &z,
		const Vector< InputType1, nonblocking, Coords > &x,
		const Vector< InputType2, nonblocking, Coords > &y,
		const Ring & ring = Ring(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			!grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			grb::is_semiring< Ring >::value,
		void >::type * const = nullptr
	) {
		// static sanity checks
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Ring::D1, InputType1 >::value ),
			"grb::eWiseMul",
			"called with a left-hand side input vector with element type that does not "
			"match the first domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Ring::D2, InputType2 >::value ),
			"grb::eWiseMul",
			"called with a right-hand side input vector with element type that does "
			"not match the second domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Ring::D3, OutputType >::value ),
			"grb::eWiseMul",
			"called with an output vector with element type that does not match the "
			"third domain of the given semiring" );

		// dynamic checks
		const size_t n = size( z );
		if( size( x ) != n || size( y ) != n ) {
			return MISMATCH;
		}

		// check trivial phase
		if( phase == RESIZE ) {
			return SUCCESS;
		}

#ifdef _DEBUG
		std::cout << "eWiseMul (nonblocking, vector <- vector x vector) dispatches "
			<< "to eWiseMulAdd (vector <- vector x vector + 0)\n";
#endif
		return eWiseMulAdd< descr, true >(
			z, x, y, ring.template getZero< typename Ring::D4 >(), ring, phase );
	}

	template<
		Descriptor descr = descriptors::no_operation,
		class Ring,
		typename InputType1,
		typename InputType2,
		typename OutputType,
		typename Coords
	>
	RC eWiseMul(
		Vector< OutputType, nonblocking, Coords > &z,
		const InputType1 alpha,
		const Vector< InputType2, nonblocking, Coords > &y,
		const Ring &ring = Ring(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			!grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			grb::is_semiring< Ring >::value,
		void >::type * const = nullptr
	) {
		// static sanity checks
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Ring::D1, InputType1 >::value ),
			"grb::eWiseMul",
			"called with a left-hand side input vector with element type that does not "
			"match the first domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Ring::D2, InputType2 >::value ),
			"grb::eWiseMul",
			"called with a right-hand side input vector with element type that does "
			"not match the second domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Ring::D3, OutputType >::value ),
			"grb::eWiseMul",
			"called with an output vector with element type that does not match the "
			"third domain of the given semiring" );

		// dynamic checks
		const size_t n = size( z );
		if( size( y ) != n ) { return MISMATCH; }

		// check for trivial phase
		if( phase == RESIZE ) {
			return SUCCESS;
		}

		// check trivial
		if( alpha == ring.template getZero< typename Ring::D1 >() ) {
			return SUCCESS;
		}

#ifdef _DEBUG
		std::cout << "eWiseMul (nonblocking, vector <- scalar x vector) dispatches "
			<< "to eWiseMulAdd (vector <- scalar x vector + 0)\n";
#endif
		return eWiseMulAdd< descr, true >(
			z, alpha, y, ring.template getZero< typename Ring::D4 >(), ring, phase );
	}

	template<
		Descriptor descr = descriptors::no_operation,
		class Ring,
		typename InputType1,
		typename InputType2,
		typename OutputType,
		typename Coords
	>
	RC eWiseMul(
		Vector< OutputType, nonblocking, Coords > &z,
		const Vector< InputType1, nonblocking, Coords > &x,
		const InputType2 beta,
		const Ring &ring = Ring(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			!grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			grb::is_semiring< Ring >::value,
		void >::type * const = nullptr
	) {
		// static sanity checks
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Ring::D1, InputType1 >::value ),
			"grb::eWiseMul",
			"called with a left-hand side input vector with element type that does not "
			"match the first domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Ring::D2, InputType2 >::value ),
			"grb::eWiseMul",
			"called with a right-hand side input vector with element type that does "
			"not match the second domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Ring::D3, OutputType >::value ),
			"grb::eWiseMul",
			"called with an output vector with element type that does not match the "
			"third domain of the given semiring" );

		// dynamic checks
		const size_t n = size( z );
		if( size( x ) != n ) {
			return MISMATCH;
		}

		// catch trivial phase
		if( phase == RESIZE ) {
			return SUCCESS;
		}

		// check trivial
		if( beta == ring.template getZero< typename Ring::D2 >() ) {
			return SUCCESS;
		}

#ifdef _DEBUG
		std::cout << "eWiseMul (nonblocking) dispatches to eWiseMulAdd with 0.0 as "
			<< "additive scalar\n";
#endif

		return eWiseMulAdd< descr, true >(
			z, x, beta, ring.template getZero< typename Ring::D4 >(), ring, phase );
	}

	template<
		Descriptor descr = descriptors::no_operation,
		class Ring,
		typename InputType1,
		typename InputType2,
		typename OutputType,
		typename Coords
	>
	RC eWiseMul(
		Vector< OutputType, nonblocking, Coords > &z,
		const InputType1 alpha,
		const InputType2 beta,
		const Ring &ring = Ring(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			!grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			grb::is_semiring< Ring >::value,
		void >::type * const = nullptr
	) {
		// static sanity checks
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Ring::D1, InputType1 >::value ),
			"grb::eWiseMul",
			"called with a left-hand side input vector with element type that does not "
			"match the first domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Ring::D2, InputType2 >::value ),
			"grb::eWiseMul",
			"called with a right-hand side input vector with element type that does "
			"not match the second domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Ring::D3, OutputType >::value ),
			"grb::eWiseMul",
			"called with an output vector with element type that does not match the "
			"third domain of the given semiring" );

		// check for trivial phase
		if( phase == RESIZE ) {
			return SUCCESS;
		}

		// check trivial
		if( alpha == ring.template getZero< typename Ring::D1 >() ) {
			return SUCCESS;
		}
		if( beta == ring.template getZero< typename Ring::D2 >() ) {
			return SUCCESS;
		}

#ifdef _DEBUG
		std::cout << "eWiseMul (nonblocking) dispatches to scalar apply and foldl\n";
#endif
		typename Ring::D3 temp;
		RC always_success = apply( temp, alpha, beta,
			ring.getMultiplicativeOperator() );
		assert( always_success == SUCCESS );
#ifdef NDEBUG
		(void) always_success;
#endif
		return foldl< descr >( z, temp, ring.getAdditiveMonoid(), phase );
	}

	template<
		Descriptor descr = descriptors::no_operation,
		class Ring,
		typename InputType1,
		typename InputType2,
		typename OutputType,
		typename MaskType,
		typename Coords
	>
	RC eWiseMul(
		Vector< OutputType, nonblocking, Coords > &z,
		const Vector< MaskType, nonblocking, Coords > &m,
		const Vector< InputType1, nonblocking, Coords > &x,
		const Vector< InputType2, nonblocking, Coords > &y,
		const Ring &ring = Ring(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			!grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			!grb::is_object< MaskType >::value &&
			grb::is_semiring< Ring >::value,
		void >::type * const = nullptr
	) {
		// static sanity checks
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Ring::D1, InputType1 >::value ),
			"grb::eWiseMul",
			"called with a left-hand side input vector with element type that does not "
			"match the first domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Ring::D2, InputType2 >::value ),
			"grb::eWiseMul",
			"called with a right-hand side input vector with element type that does "
			"not match the second domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Ring::D3, OutputType >::value ),
			"grb::eWiseMul",
			"called with an output vector with element type that does not match the "
			"third domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< bool, MaskType >::value ),
			"grb::eWiseMulAdd",
			"called with a mask vector with a non-bool element type" );

		// check for empty mask
		if( size( m ) == 0 ) {
			return eWiseMul< descr >( z, x, y, ring, phase );
		}

		// dynamic checks
		const size_t n = size( z );
		if( size( m ) != n || size( x ) != n || size( y ) != n ) {
			return MISMATCH;
		}

		// check trivial phase
		if( phase == RESIZE ) {
			return SUCCESS;
		}

#ifdef _DEBUG
		std::cout << "eWiseMul (nonblocking, vector <- vector x vector, masked) "
			<< "dispatches to eWiseMulAdd (vector <- vector x vector + 0, masked)\n";
#endif
		return eWiseMulAdd< descr, true >(
			z, m, x, y, ring.template getZero< typename Ring::D4 >(), ring, phase );
	}

	template<
		Descriptor descr = descriptors::no_operation,
		class Ring,
		typename InputType1,
		typename InputType2,
		typename OutputType,
		typename MaskType,
		typename Coords
	>
	RC eWiseMul(
		Vector< OutputType, nonblocking, Coords > &z,
		const Vector< MaskType, nonblocking, Coords > &m,
		const InputType1 alpha,
		const Vector< InputType2, nonblocking, Coords > &y,
		const Ring &ring = Ring(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			!grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			!grb::is_object< MaskType >::value &&
			grb::is_semiring< Ring >::value,
		void >::type * const = nullptr
	) {
		// static sanity checks
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Ring::D1, InputType1 >::value ),
			"grb::eWiseMul",
			"called with a left-hand side input vector with element type that does not "
			"match the first domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Ring::D2, InputType2 >::value ),
			"grb::eWiseMul",
			"called with a right-hand side input vector with element type that does "
			"not match the second domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Ring::D3, OutputType >::value ),
			"grb::eWiseMul",
			"called with an output vector with element type that does not match the "
			"third domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< bool, MaskType >::value ),
			"grb::eWiseMulAdd",
			"called with a mask vector _m with a non-bool element type" );

		// check for empty mask
		if( size( m ) == 0 ) {
			return eWiseMul< descr >( z, alpha, y, ring, phase );
		}

		// dynamic checks
		const size_t n = size( z );
		if( size( m ) != n || size( y ) != n ) { return MISMATCH; }

		// check for trivial phase
		if( phase == RESIZE ) {
			return SUCCESS;
		}

		// check trivial
		if( alpha == ring.template getZero< typename Ring::D1 >() ) {
			return SUCCESS;
		}

#ifdef _DEBUG
		std::cout << "eWiseMul (nonblocking, vector <- scalar x vector, masked) "
			<< "dispatches to eWiseMulAdd (vector <- scalar x vector + 0, masked)\n";
#endif
		return eWiseMulAdd< descr, true >(
			z, m, alpha, y, ring.template getZero< typename Ring::D4 >(), ring, phase );
	}

	template<
		Descriptor descr = descriptors::no_operation,
		class Ring,
		typename InputType1,
		typename InputType2,
		typename OutputType,
		typename MaskType,
		typename Coords
	>
	RC eWiseMul(
		Vector< OutputType, nonblocking, Coords > &z,
		const Vector< MaskType, nonblocking, Coords > &m,
		const Vector< InputType1, nonblocking, Coords > &x,
		const InputType2 beta,
		const Ring &ring = Ring(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			!grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			!grb::is_object< MaskType >::value &&
			grb::is_semiring< Ring >::value,
		void >::type * const = nullptr
	) {
		// static sanity checks
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Ring::D1, InputType1 >::value ),
			"grb::eWiseMul",
			"called with a left-hand side input vector with element type that does not "
			"match the first domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Ring::D2, InputType2 >::value ),
			"grb::eWiseMul",
			"called with a right-hand side input vector with element type that does "
			"not match the second domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Ring::D3, OutputType >::value ),
			"grb::eWiseMul",
			"called with an output vector with element type that does not match the "
			"third domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< bool, MaskType >::value ),
			"grb::eWiseMulAdd",
			"called with a mask vector _m with a non-bool element type" );

		// check for empty mask
		if( size( m ) == 0 ) {
			return eWiseMul< descr >( z, x, beta, ring, phase );
		}

		// dynamic checks
		const size_t n = size( z );
		if( size( m ) != n || size( x ) != n ) { return MISMATCH; }

		// check for trivial phase
		if( phase == RESIZE ) {
			return SUCCESS;
		}

		// check trivial
		if( beta == ring.template getZero< typename Ring::D2 >() ) {
			return SUCCESS;
		}

#ifdef _DEBUG
		std::cout << "eWiseMul (nonblocking, masked) dispatches to masked "
			<< "eWiseMulAdd with 0.0 as additive scalar\n";
#endif
		return eWiseMulAdd< descr, true >(
			z, m, x, beta, ring.template getZero< typename Ring::D4 >(), ring, phase );
	}

	template<
		Descriptor descr = descriptors::no_operation,
		class Ring,
		typename InputType1,
		typename InputType2,
		typename OutputType,
		typename MaskType,
		typename Coords
	>
	RC eWiseMul(
		Vector< OutputType, nonblocking, Coords > &z,
		const Vector< MaskType, nonblocking, Coords > &m,
		const InputType1 alpha,
		const InputType2 beta,
		const Ring &ring = Ring(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			!grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			!grb::is_object< MaskType >::value &&
			grb::is_semiring< Ring >::value,
		void >::type * const = nullptr
	) {
		// static sanity checks
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Ring::D1, InputType1 >::value ),
			"grb::eWiseMul",
			"called with a left-hand side input vector with element type that does not "
			"match the first domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Ring::D2, InputType2 >::value ),
			"grb::eWiseMul",
			"called with a right-hand side input vector with element type that does "
			"not match the second domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Ring::D3, OutputType >::value ),
			"grb::eWiseMul",
			"called with an output vector with element type that does not match the "
			"third domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< bool, MaskType >::value ),
			"grb::eWiseMulAdd",
			"called with a mask vector _m with a non-bool element type" );

		// check for empty mask
		if( size( m ) == 0 ) {
			return eWiseMul< descr >( z, alpha, beta, ring, phase );
		}

		// dynamic checks
		const size_t n = size( z );
		if( size( m ) != n ) { return MISMATCH; }

		// check for trivial phase
		if( phase == RESIZE ) {
			return SUCCESS;
		}

		// check trivial
		if( alpha == ring.template getZero< typename Ring::D1 >() ) {
			return SUCCESS;
		}
		if( beta == ring.template getZero< typename Ring::D2 >() ) {
			return SUCCESS;
		}

#ifdef _DEBUG
		std::cout << "eWiseMul (nonblocking, masked) dispatches to masked foldl\n";
#endif
		typename Ring::D3 temp;
		const RC always_success = apply( temp, alpha, beta,
			ring.getMultiplicativeOperator() );
		assert( always_success == SUCCESS );
#ifdef NDEBUG
		(void) always_success;
#endif
		return foldl< descr >( z, m, temp, ring.getAdditiveMonoid(), EXECUTE );
	}

	// internal namespace for implementation of grb::dot
	namespace internal {

		template<
			Descriptor descr,
#ifdef GRB_BOOLEAN_DISPATCHER
			bool already_dense_input_x,
			bool already_dense_input_y,
#endif
			class AddMonoid,
			class AnyOp,
			typename InputType1,
			typename InputType2,
			typename Coords
		>
		RC sparse_dot_generic(
#ifndef GRB_BOOLEAN_DISPATCHER
			bool already_dense_input_x,
			bool already_dense_input_y,
#endif
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
#ifdef _DEBUG
			std::cout << "\t\t in sparse variant, nonzero range " << lower_bound << "--"
				<< upper_bound << ", blocksize " << AnyOp::blocksize << "\n";
#else
			(void) upper_bound;
#endif

			// get raw alias
			const InputType1 * __restrict__ a = internal::getRaw( x );
			const InputType2 * __restrict__ b = internal::getRaw( y );

			size_t i = 0;
			if( local_nz > 0 ) {
				while( i + AnyOp::blocksize < local_nz ) {
					// declare buffers
					static_assert( AnyOp::blocksize > 0,
						"Configuration error: vectorisation blocksize set to 0!" );
					typename AnyOp::D1 xx[ AnyOp::blocksize ];
					typename AnyOp::D2 yy[ AnyOp::blocksize ];
					typename AnyOp::D3 zz[ AnyOp::blocksize ];
					bool mask[ AnyOp::blocksize ];

					// prepare registers
					for( size_t k = 0; k < AnyOp::blocksize; ++k, ++i ) {
						mask[ k ] = already_dense_input_x ||
							local_x.assigned( already_dense_input_y ? i : local_y.index( i ) );
					}

					// rewind
					i -= AnyOp::blocksize;

					// do masked load
					for( size_t k = 0; k < AnyOp::blocksize; ++k, ++i ) {
						if( mask[ k ] ) {
							xx[ k ] = static_cast< typename AnyOp::D1 >(
								a[ ( already_dense_input_y ? i : local_y.index( i ) ) + lower_bound ] );
							yy[ k ] = static_cast< typename AnyOp::D2 >(
								b[ ( already_dense_input_y ? i : local_y.index( i ) ) + lower_bound ] );
						}
					}

					// perform element-wise multiplication
					if( internal::maybe_noop< AnyOp >::value ) {
						// we are forced to first initialise zz before doing masked apply
						for( size_t k = 0; k < AnyOp::blocksize; ++k ) {
							zz[ k ] = addMonoid.template getIdentity< typename AnyOp::D3 >();
						}
						for( size_t k = 0; k < AnyOp::blocksize; ++k ) {
							if( mask[ k ] ) {
								GRB_UTIL_IGNORE_MAYBE_UNINITIALIZED        // yy and xx cannot be used
										                           // uninitialised or mask
								apply( zz[ k ], xx[ k ], yy[ k ], anyOp ); // would be false while zz
								GRB_UTIL_RESTORE_WARNINGS                  // init is just above
							}
						}
					} else {
						// if apply surely initialises zz, we could use a blend-like op
						for( size_t k = 0; k < AnyOp::blocksize; ++k ) {
							if( mask[ k ] ) {
								apply( zz[ k ], xx[ k ], yy[ k ], anyOp );
							} else {
								zz[ k ] = addMonoid.template getIdentity< typename AnyOp::D3 >();
							}
						}
					}

					// perform reduction into output element
					addMonoid.getOperator().foldlArray( thread_local_output, zz,
						AnyOp::blocksize );
					//^--> note that this foldl operates on raw arrays,
					//     and thus should not be mistaken with a foldl
					//     on a grb::Vector.
				}

				// perform element-by-element updates for remainder (if any)
				for( ; i < local_nz; ++i ) {
					typename AddMonoid::D3 temp =
						addMonoid.template getIdentity< typename AddMonoid::D3 >();
					const size_t index = ( already_dense_input_y ? i : local_y.index( i ) ) +
						lower_bound;
					if( already_dense_input_x || local_x.assigned( index - lower_bound ) ) {
						apply( temp, a[ index ], b[ index ], anyOp );
						foldr( temp, thread_local_output, addMonoid.getOperator() );
					}
				}
			}

			return SUCCESS;
		}

		template<
			Descriptor descr = descriptors::no_operation,
			class AddMonoid,
			class AnyOp,
			typename OutputType,
			typename InputType1,
			typename InputType2,
			typename Coords
		>
		RC dot_generic(
			OutputType &z,
			const Vector< InputType1, nonblocking, Coords > &x,
			const Vector< InputType2, nonblocking, Coords > &y,
			const AddMonoid &addMonoid,
			const AnyOp &anyOp,
			const Phase &phase
		) {
			const size_t n = internal::getCoordinates( x ).size();

			if( phase == RESIZE ) {
				return SUCCESS;
			}
			assert( phase == EXECUTE );

			RC ret = SUCCESS;

			const size_t start = 0;
			const size_t end = n;

			if( end > start ) {

				typename AddMonoid::D3 reduced =
					addMonoid.template getIdentity< typename AddMonoid::D3 >();

				size_t reduced_size = NONBLOCKING::numThreads() *
					config::CACHE_LINE_SIZE::value();
				typename AddMonoid::D3 array_reduced[ reduced_size ];

				for(
					size_t i = 0;
					i < reduced_size;
					i += config::CACHE_LINE_SIZE::value()
				) {
					array_reduced[ i ] =
						addMonoid.template getIdentity< typename AddMonoid::D3 >();
				}

				constexpr const bool dense_descr = descr & descriptors::dense;

				internal::Pipeline::stage_type func =
					[&x, &y, &addMonoid, &anyOp, &array_reduced] (
						internal::Pipeline &pipeline,
						const size_t lower_bound, const size_t upper_bound
					) {
#ifdef _NONBLOCKING_DEBUG
						#pragma omp critical
						std::cout << "\t\tExecution of stage dot-generic in the range("
							<< lower_bound << ", " << upper_bound << ")" << std::endl;
#endif
						RC rc = SUCCESS;

						Coords local_x, local_y;
						const size_t local_n = upper_bound - lower_bound;
						size_t local_x_nz = local_n;
						size_t local_y_nz = local_n;
						bool sparse = false;

#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
						const bool already_dense_vectors = dense_descr ||
							pipeline.allAlreadyDenseVectors();
#else
						(void) pipeline;
						constexpr const bool already_dense_vectors = dense_descr;
#endif
						bool already_dense_input_x = true;
						bool already_dense_input_y = true;

						if( !already_dense_vectors ) {
#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
							already_dense_input_x = pipeline.containsAlreadyDenseVector(
								&internal::getCoordinates( x ) );
							if( !already_dense_input_x ) {
#else
								already_dense_input_x = false;
#endif
								local_x = internal::getCoordinates( x ).asyncSubset(
									lower_bound, upper_bound );
								local_x_nz = local_x.nonzeroes();
								if( local_x_nz < local_n ) {
									sparse = true;
								}
#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
							}

							already_dense_input_y = pipeline.containsAlreadyDenseVector(
								&internal::getCoordinates( y ) );
							if( !already_dense_input_y ) {
#else
								already_dense_input_y = false;
#endif
								local_y = internal::getCoordinates( y ).asyncSubset( lower_bound,
									upper_bound );
								local_y_nz = local_y.nonzeroes();
								if( local_y_nz < local_n ) {
									sparse = true;
								}
#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
							}
#endif
						}

						unsigned int thread_id =
							omp_get_thread_num() * config::CACHE_LINE_SIZE::value();

						if( sparse ) {
							if( local_x_nz < local_y_nz ) {
#ifdef GRB_BOOLEAN_DISPATCHER
								rc = internal::boolean_dispatcher_sparse_dot_generic<
#else
								rc = internal::sparse_dot_generic<
#endif
										descr, AddMonoid, AnyOp, InputType1, InputType2, Coords
									>(
										already_dense_input_x, already_dense_input_y,
										array_reduced[ thread_id ],
										lower_bound, upper_bound,
										local_x, local_y,
										x, y,
										local_x_nz,
										addMonoid, anyOp
									 );
							} else {
#ifdef GRB_BOOLEAN_DISPATCHER
								rc = internal::boolean_dispatcher_sparse_dot_generic<
#else
								rc = internal::sparse_dot_generic<
#endif
										descr, AddMonoid, AnyOp, InputType1, InputType2, Coords
									>(
										already_dense_input_y, already_dense_input_x,
										array_reduced[ thread_id ],
										lower_bound, upper_bound,
										local_y, local_x, x, y, local_y_nz,
										addMonoid, anyOp
									);
							}
						} else {
							// get raw alias
							const InputType1 * __restrict__ a = internal::getRaw( x );
							const InputType2 * __restrict__ b = internal::getRaw( y );

							size_t i = lower_bound;
							if( upper_bound > lower_bound ) {
								while( i + AnyOp::blocksize < upper_bound ) {
									// declare buffers
									static_assert( AnyOp::blocksize > 0,
										"Configuration error: vectorisation blocksize set to 0!" );

									typename AnyOp::D1 xx[ AnyOp::blocksize ];
									typename AnyOp::D2 yy[ AnyOp::blocksize ];
									typename AnyOp::D3 zz[ AnyOp::blocksize ];

									// prepare registers
									for( size_t k = 0; k < AnyOp::blocksize; ++k ) {
										xx[ k ] = static_cast< typename AnyOp::D1 >( a[ i ] );
										yy[ k ] = static_cast< typename AnyOp::D2 >( b[ i++ ] );
									}

									// perform element-wise multiplication
									if( internal::maybe_noop< AnyOp >::value ) {
										for( size_t k = 0; k < AnyOp::blocksize; ++k ) {
											zz[ k ] = addMonoid.template getIdentity< typename AnyOp::D3 >();
										}
									}
									for( size_t k = 0; k < AnyOp::blocksize; ++k ) {
										apply( zz[ k ], xx[ k ], yy[ k ], anyOp );
									}

									// perform reduction into output element
									addMonoid.getOperator().foldlArray( array_reduced[ thread_id ], zz,
										AnyOp::blocksize );
									//^--> note that this foldl operates on raw arrays,
									//     and thus should not be mistaken with a foldl
									//     on a grb::Vector.
#ifdef _DEBUG
									std::cout << "\t\t " << ( i - AnyOp::blocksize ) << "--" << i << ": "
										<< "running reduction = " << array_reduced[ thread_id ] << "\n";
#endif
								}

								// perform element-by-element updates for remainder (if any)
								for( ; i < upper_bound; ++i ) {
									OutputType temp = addMonoid.template getIdentity< OutputType >();
									apply( temp, a[ i ], b[ i ], anyOp );
									foldr( temp, array_reduced[ thread_id ], addMonoid.getOperator() );
								}
							}
						}

						// the local coordinates for the input vectors have not been updated as
						// they are read-only therefore, we don't need to invoke asyncJoinSubset;
						// the output is a scalar
						return rc;
					};

#ifdef _NONBLOCKING_DEBUG
				std::cout << "\t\tStage added to a pipeline: dot-generic" << std::endl;
#endif

				ret = ret ? ret : internal::le.addStage(
						std::move( func ),
						internal::Opcode::BLAS1_DOT_GENERIC,
						end, sizeof( OutputType ), dense_descr, true,
						nullptr, nullptr, nullptr, nullptr,
						&x, &y, nullptr, nullptr,
						&internal::getCoordinates( x ), &internal::getCoordinates( y ),
						nullptr, nullptr,
						nullptr
					);

				for(
					size_t i = 0;
					i < reduced_size;
					i += config::CACHE_LINE_SIZE::value()
				) {
					foldl( reduced, array_reduced[ i ], addMonoid.getOperator() );
				}

				// write back result
				z = static_cast< OutputType >( reduced );
			} else {
				// this has been tested by the unittest
			}

#ifdef _DEBUG
			std::cout << "\t returning " << z << "\n";
#endif
			// done!
			return ret;
		}

	} // namespace internal

	template<
		Descriptor descr = descriptors::no_operation,
		class AddMonoid,
		class AnyOp,
		typename OutputType,
		typename InputType1,
		typename InputType2,
		typename Coords
	>
	RC dot(
		OutputType &z,
		const Vector< InputType1, nonblocking, Coords > &x,
		const Vector< InputType2, nonblocking, Coords > &y,
		const AddMonoid &addMonoid = AddMonoid(),
		const AnyOp &anyOp = AnyOp(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			!grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			grb::is_monoid< AddMonoid >::value &&
			grb::is_operator< AnyOp >::value,
		void >::type * const = nullptr
	) {
		// static sanity checks
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< InputType1, typename AnyOp::D1 >::value ), "grb::dot",
			"called with a left-hand vector value type that does not match the first "
			"domain of the given multiplicative operator" );
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< InputType2, typename AnyOp::D2 >::value ), "grb::dot",
			"called with a right-hand vector value type that does not match the second "
			"domain of the given multiplicative operator" );
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< typename AddMonoid::D3, typename AnyOp::D1 >::value ),
			"grb::dot",
			"called with a multiplicative operator output domain that does not match "
			"the first domain of the given additive operator" );
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< OutputType, typename AddMonoid::D2 >::value ), "grb::dot",
			"called with an output vector value type that does not match the second "
			"domain of the given additive operator" );
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< typename AddMonoid::D3, typename AddMonoid::D2 >::value ),
			"grb::dot",
			"called with an additive operator whose output domain does not match its "
			"second input domain" );
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< OutputType, typename AddMonoid::D3 >::value ), "grb::dot",
			"called with an output vector value type that does not match the third "
			"domain of the given additive operator" );

#ifdef _DEBUG
		std::cout << "In grb::dot (nonblocking). "
			<< "I/O scalar on input reads " << z << "\n";
#endif

		// dynamic sanity check
		const size_t n = internal::getCoordinates( y ).size();
		if( internal::getCoordinates( x ).size() != n ) {
			return MISMATCH;
		}

#ifdef _DEBUG
		std::cout << "\t dynamic checks pass\n";
#endif

		// dot will be computed out-of-place here. A separate field is needed because
		// of possible multi-threaded computation of the dot.
		OutputType oop = addMonoid.template getIdentity< OutputType >();

		RC ret = SUCCESS;

		ret = internal::dot_generic< descr >( oop, x, y, addMonoid, anyOp, phase );

		// fold out-of-place dot product into existing input, and exit
#ifdef _DEBUG
		std::cout << "\t dot_generic returned " << oop << ", "
			<< "which will be folded into " << z << " "
			<< "using the additive monoid\n";
#endif
		ret = ret ? ret : foldl( z, oop, addMonoid.getOperator() );
#ifdef _DEBUG
		std::cout << "\t returning " << z << "\n";
#endif
		return ret;
	}

	template<
		Descriptor descr = descriptors::no_operation,
		class Ring,
		typename IOType,
		typename InputType1,
		typename InputType2,
		typename Coords
	>
	RC dot(
		IOType &x,
		const Vector< InputType1, nonblocking, Coords > &left,
		const Vector< InputType2, nonblocking, Coords > &right,
		const Ring &ring = Ring(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			!grb::is_object< IOType >::value &&
			grb::is_semiring< Ring >::value,
		void >::type * const = nullptr
	) {
#ifdef _DEBUG
		std::cout << "In grb::dot (nonblocking, semiring version)\n"
			<< "\t dispatches to monoid-operator version\n";
#endif
		return grb::dot< descr >( x, left, right, ring.getAdditiveMonoid(),
			ring.getMultiplicativeOperator(), phase );
	}

	template<
		Descriptor descr = descriptors::no_operation,
		typename Func,
		typename DataType,
		typename Coords
	>
	RC eWiseMap( const Func f, Vector< DataType, nonblocking, Coords > &x ) {

		RC ret = SUCCESS;

		const size_t n = internal::getCoordinates( x ).size();

		constexpr const bool dense_descr = descr & descriptors::dense;

		internal::Pipeline::stage_type func = [f, &x] (
			internal::Pipeline &pipeline, const size_t lower_bound, const size_t upper_bound
		) {
#ifdef _NONBLOCKING_DEBUG
			#pragma omp critical
			std::cout << "\t\tExecution of stage eWiseMap(f, x) in the range("
				<< lower_bound << ", " << upper_bound << ")" << std::endl;
#endif
			RC rc = SUCCESS;

			Coords local_x;
			const size_t local_n = upper_bound - lower_bound;
			size_t local_x_nz = local_n;
			bool sparse = false;

			bool already_dense_input_x = true;

			if( !dense_descr ) {
#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
				already_dense_input_x = pipeline.containsAlreadyDenseVector(
					&internal::getCoordinates( x ) );
				if( !already_dense_input_x ) {
#else
					already_dense_input_x = false;
#endif
					local_x = internal::getCoordinates( x ).asyncSubset( lower_bound,
						upper_bound );
					local_x_nz = local_x.nonzeroes();
					if( local_x_nz < local_n ) {
						sparse = true;
					}
#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
				}
#endif
			}

			if( sparse ) {
				// the sparse case is possible only when the local coordinates are already
				// initialized
				assert( already_dense_input_x == false );
				for( size_t k = 0; k < local_x_nz; ++k ) {
					DataType &xval = internal::getRaw( x )[ local_x.index( k ) + lower_bound ];
					xval = f( xval );
				}
			} else {
				for( size_t i = lower_bound; i < upper_bound; ++i ) {
					DataType &xval = internal::getRaw( x )[ i ];
					xval = f( xval );
				}
			}

			return rc;
		};

		ret = ret ? ret : internal::le.addStage(
				std::move( func ),
				internal::Opcode::BLAS1_EWISEMAP,
				n, sizeof( DataType ), dense_descr, true,
				&x, nullptr, &internal::getCoordinates( x ), nullptr,
				nullptr, nullptr, nullptr, nullptr,
				nullptr, nullptr, nullptr, nullptr,
				nullptr
			);

#ifdef _NONBLOCKING_DEBUG
		std::cout << "\t\tStage added to a pipeline: eWiseMap(f, x)" << std::endl;
#endif
		return ret;
	}

	namespace internal {

		template<
			Descriptor descr = descriptors::no_operation,
			typename Func,
			typename DataType1,
			typename DataType2,
			typename Coords,
			typename... Args
		>
		RC eWiseLambda_helper(
			std::vector< const void * > all_vectors_ptr,
			size_t maximum_data_type_size,
			const Func f,
			const Vector< DataType1, nonblocking, Coords > &x,
			const Vector< DataType2, nonblocking, Coords > &y,
			Args const &... args
		) {
			// catch mismatch
			if( size( x ) != size( y ) ) {
				return MISMATCH;
			}

			all_vectors_ptr.push_back( &y );
			maximum_data_type_size = std::max( maximum_data_type_size, sizeof( DataType2 ) );

			// continue
			return eWiseLambda_helper( all_vectors_ptr, maximum_data_type_size, f, x,
				args... );
		}

		template<
			Descriptor descr = descriptors::no_operation,
			typename Func,
			typename DataType,
			typename Coords
		>
		RC eWiseLambda_helper(
			std::vector< const void * > all_vectors_ptr,
			size_t maximum_data_type_size,
			const Func f,
			const Vector< DataType, nonblocking, Coords > &x
		) {
			// all pointers, except one, have been stored, and the last one will be
			// stored by the normal eWiseLambda
			return eWiseLambda< descr, Func, DataType, Coords >( f, x, all_vectors_ptr,
				maximum_data_type_size );
		}
	};

	template<
		Descriptor descr = descriptors::no_operation,
		typename Func,
		typename DataType1,
		typename DataType2,
		typename Coords,
		typename... Args
	>
	RC eWiseLambda(
		const Func f,
		const Vector< DataType1, nonblocking, Coords > &x,
		const Vector< DataType2, nonblocking, Coords > &y,
		Args const &... args
	) {

		// create an empty vector to store pointers for all vectors passed to
		// eWiseLambda
		std::vector< const void * > all_vectors_ptr;

		// invoke the helper function to store the pointers
		return internal::eWiseLambda_helper( all_vectors_ptr, 0, f, x, y, args...);
	}

	template<
		Descriptor descr = descriptors::no_operation,
		typename Func,
		typename DataType,
		typename Coords
	>
	RC eWiseLambda(
		const Func f,
		const Vector< DataType, nonblocking, Coords > &x,
		std::vector< const void * > all_vectors_ptr = std::vector< const void *>(),
		size_t maximum_data_type_size = 0
	) {
#ifdef _DEBUG
		std::cout << "Info: entering eWiseLambda function on vectors.\n";
#endif

		all_vectors_ptr.push_back( &x );
		maximum_data_type_size =
			std::max( maximum_data_type_size, sizeof( DataType ) );

		RC ret = SUCCESS;

		constexpr const bool dense_descr = descr & descriptors::dense;

		internal::Pipeline::stage_type func = [f, &x] (
			internal::Pipeline &pipeline,
			const size_t lower_bound, const size_t upper_bound
		) {
#ifdef _NONBLOCKING_DEBUG
			#pragma omp critical
			std::cout << "\t\tExecution of stage eWiseLambda in the range("
				<< lower_bound << ", " << upper_bound << ")" << std::endl;
#endif
			Coords local_x;
			const size_t local_n = upper_bound - lower_bound;
			size_t local_x_nz;
			bool sparse = false;

#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
			const bool already_dense_vectors = dense_descr ||
				pipeline.allAlreadyDenseVectors();
#else
			(void) pipeline;
			constexpr const bool already_dense_vectors = dense_descr;
#endif

			bool already_dense_output = true;

			if( !already_dense_vectors ) {
#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
				already_dense_output = pipeline.containsAlreadyDenseVector(
					&internal::getCoordinates( x ) );
				if( !already_dense_output ) {
#else
					already_dense_output = false;
#endif
					local_x = internal::getCoordinates( x ).asyncSubset( lower_bound,
						upper_bound );
					local_x_nz = local_x.nonzeroes();
					if( local_x_nz < local_n ) {
						sparse = true;
					}
#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
				}
#endif
			}

			if( sparse ) {
				if ( already_dense_output ) {
					for( size_t k = 0; k < local_x_nz; ++k ) {
						f( k + lower_bound );
					}
				} else {
					for( size_t k = 0; k < local_x_nz; ++k ) {
						const size_t i = local_x.index( k ) + lower_bound;
						f( i );
					}
				}
			} else {
				for (size_t i = lower_bound; i < upper_bound; i++) {
					f( i );
				}
			}

			// the local coordinates for all vectors of eWiseLambda cannot change
			// therefore, we don't need to invoke asyncJoinSubset for any of them

			return SUCCESS;
		};

		// eWiseLambda is a special case as we don't know which of the accessed
		// vectors are read-only therefore, we assume that all vectors may be written,
		// but the sparsity structure cannot change i.e., the coordinates of each
		// vector cannot be updated, but we pass the coordinates of x for the loop
		// size
		ret = ret ? ret : internal::le.addeWiseLambdaStage(
				std::move( func ),
				internal::Opcode::BLAS1_EWISELAMBDA,
				internal::getCoordinates( x ).size(), maximum_data_type_size, dense_descr,
				all_vectors_ptr, &internal::getCoordinates( x )
			);

#ifdef _NONBLOCKING_DEBUG
		std::cout << "\t\tStage added to a pipeline: eWiseLambda" << std::endl;
#endif
		return ret;
	}

	template<
		Descriptor descr = descriptors::no_operation,
		class Monoid,
		typename InputType,
		typename IOType,
		typename MaskType,
		typename Coords
	>
	RC foldl(
		IOType &x,
		const Vector< InputType, nonblocking, Coords > &y,
		const Vector< MaskType, nonblocking, Coords > &mask,
		const Monoid &monoid = Monoid(),
		const typename std::enable_if< !grb::is_object< IOType >::value &&
			!grb::is_object< InputType >::value &&
			!grb::is_object< MaskType >::value &&
			grb::is_monoid< Monoid >::value, void
		>::type * const = nullptr
	) {
#ifdef _DEBUG
		std::cout << "foldl: IOType <- [InputType] with a monoid called. "
			<< "Array has size " << size( y ) << " with " << nnz( y ) << " nonzeroes. "
			<< "It has a mask of size " << size( mask ) << " with " << nnz( mask )
			<< " nonzeroes.\n";
#endif

		// static sanity checks
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< IOType, InputType >::value ), "grb::foldl",
			"called with a scalar IO type that does not match the input vector type" );
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< InputType, typename Monoid::D1 >::value ), "grb::foldl",
			"called with an input vector value type that does not match the first "
			"domain of the given monoid" );
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< InputType, typename Monoid::D2 >::value ), "grb::foldl",
			"called with an input vector type that does not match the second domain of "
			"the given monoid" );
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< InputType, typename Monoid::D3 >::value ), "grb::foldl",
			"called with an input vector type that does not match the third domain of "
			"the given monoid" );
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< bool, MaskType >::value ), "grb::foldl",
			"called with a vector mask type that is not boolean" );

		if( size( mask ) > 0 ) {
			return internal::template fold_from_vector_to_scalar_generic<
					descr, true, true
				>( x, y, mask, monoid );
		} else {
			return internal::template fold_from_vector_to_scalar_generic<
					descr, false, true
				>( x, y, mask, monoid );
		}
	}

	template<
		Descriptor descr = descriptors::no_operation,
		class Monoid,
		typename IOType,
		typename InputType,
		typename Coords
	>
	RC foldl(
		IOType &x,
		const Vector< InputType, nonblocking, Coords > &y,
		const Monoid &monoid = Monoid(),
		const typename std::enable_if<
			!grb::is_object< IOType >::value &&
			!grb::is_object< InputType >::value &&
			grb::is_monoid< Monoid >::value, void
		>::type * const = nullptr
	) {
#ifdef _DEBUG
		std::cout << "foldl: IOType <- [InputType] with a monoid called. "
			<< "Array has size " << size( y ) << " with " << nnz( y ) << " nonzeroes. "
			<< "It has no mask.\n";
#endif

		// static sanity checks
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< IOType, InputType >::value ), "grb::reduce",
			"called with a scalar IO type that does not match the input vector type" );
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< InputType, typename Monoid::D1 >::value ), "grb::reduce",
			"called with an input vector value type that does not match the first "
			"domain of the given monoid" );
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< InputType, typename Monoid::D2 >::value ), "grb::reduce",
			"called with an input vector type that does not match the second domain of "
			"the given monoid" );
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< InputType, typename Monoid::D3 >::value ), "grb::reduce",
			"called with an input vector type that does not match the third domain of "
			"the given monoid" );

		// do reduction
		Vector< bool, nonblocking, Coords > empty_mask( 0 );
		return internal::template fold_from_vector_to_scalar_generic<
				descr, false, true
			>( x, y, empty_mask, monoid );
	}

	template<
		Descriptor descr = descriptors::no_operation,
		typename T,
		typename U,
		typename Coords
	>
	RC zip(
		Vector< std::pair< T, U >, nonblocking, Coords > &z,
		const Vector< T, nonblocking, Coords > &x,
		const Vector< U, nonblocking, Coords > &y,
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			!grb::is_object< T >::value &&
			!grb::is_object< U >::value,
		void >::type * const = nullptr
	) {
		const size_t n = size( z );
		if( n != size( x ) ) {
			return MISMATCH;
		}
		if( n != size( y ) ) {
			return MISMATCH;
		}
		if( phase == RESIZE ) {
			return SUCCESS;
		}
		assert( phase == EXECUTE );

		const T * const x_raw = internal::getRaw( x );
		const U * const y_raw = internal::getRaw( y );
		std::pair< T, U > * z_raw = internal::getRaw( z );

		RC ret = SUCCESS;

		constexpr const bool dense_descr = descr & descriptors::dense;

		internal::Pipeline::stage_type func = [&z, x_raw, y_raw, z_raw] (
			internal::Pipeline &pipeline,
			const size_t lower_bound, const size_t upper_bound
		) {
#ifdef _NONBLOCKING_DEBUG
			std::cout << "\t\tExecution of stage zip(z, x, y) in the range("
				<< lower_bound << ", " << upper_bound << ")" << std::endl;
#endif
			RC rc = SUCCESS;

			Coords local_z;

#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
			bool already_dense_output = true;
#else
			(void) pipeline;
#endif

#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
			already_dense_output = pipeline.containsAlreadyDenseVector(
				&internal::getCoordinates( z ) );
			if( !dense_descr && !already_dense_output ) {
#else
			if( !dense_descr ) {
#endif
				local_z = internal::getCoordinates( z ).asyncSubset( lower_bound,
					upper_bound );
			}

#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
			if( !dense_descr && !already_dense_output ) {
#else
			if( !dense_descr ) {
#endif
				// the result will always be dense
				local_z.local_assignAllNotAlreadyAssigned();
			}

			for( size_t i = lower_bound; i < upper_bound; ++i ) {
				z_raw[ i ].first = x_raw[ i ];
				z_raw[ i ].second = y_raw[ i ];
			}

#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
			if( !dense_descr && !already_dense_output ) {
#else
			if( !dense_descr ) {
#endif
				internal::getCoordinates( z ).asyncJoinSubset( local_z, lower_bound,
					upper_bound );
			}

			return rc;
		};

		ret = ret ? ret : internal::le.addStage(
				std::move( func ),
				internal::Opcode::BLAS1_ZIP,
				internal::getCoordinates( x ).size(), sizeof( T ) + sizeof( U ),
				dense_descr, true,
				&z, nullptr, &internal::getCoordinates( z ), nullptr,
				&x, &y, nullptr, nullptr,
				&internal::getCoordinates( x ), &internal::getCoordinates( y ),
				nullptr, nullptr,
				nullptr
			);

#ifdef _NONBLOCKING_DEBUG
		std::cout << "\t\tStage added to a pipeline: zip(z, x, y)" << std::endl;
#endif
		return SUCCESS;
	}

	template<
		Descriptor descr = descriptors::no_operation,
		typename T,
		typename U,
		typename Coords
	>
	RC unzip(
		Vector< T, nonblocking, Coords > &x,
		Vector< U, nonblocking, Coords > &y,
		const Vector< std::pair< T, U >, nonblocking, Coords > &in,
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			!grb::is_object< T >::value &&
			!grb::is_object< U >::value,
		void >::type * const = nullptr
	) {
		const size_t n = size( in );
		if( n != size( x ) ) {
			return MISMATCH;
		}
		if( n != size( y ) ) {
			return MISMATCH;
		}

		if( phase == RESIZE ) {
			return SUCCESS;
		}
		assert( phase == EXECUTE );

		T * const x_raw = internal::getRaw( x );
		U * const y_raw = internal::getRaw( y );
		const std::pair< T, U > * in_raw = internal::getRaw( in );

		RC ret = SUCCESS;

		constexpr const bool dense_descr = descr & descriptors::dense;

		internal::Pipeline::stage_type func = [&x, &y, x_raw, y_raw, in_raw] (
			internal::Pipeline &pipeline,
			const size_t lower_bound, const size_t upper_bound
		) {
#ifdef _NONBLOCKING_DEBUG
			#pragma omp critical
			std::cout << "\t\tExecution of stage unzip(x, y, in) in the range("
				<< lower_bound << ", " << upper_bound << ")" << std::endl;
#endif
			RC rc = SUCCESS;

			Coords local_x, local_y;

			bool already_dense_output_x = true;
			bool already_dense_output_y = true;

#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
			already_dense_output_x = pipeline.containsAlreadyDenseVector(
				&internal::getCoordinates( x ) );
			if( !dense_descr && !already_dense_output_x ) {
#else
			if( !dense_descr ) {
				already_dense_output_x = false;
#endif
				local_x = internal::getCoordinates( x ).asyncSubset( lower_bound,
					upper_bound );
				local_x.local_assignAllNotAlreadyAssigned();
			}

#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
			already_dense_output_y = pipeline.containsAlreadyDenseVector(
				&internal::getCoordinates( y ) );
			if( !dense_descr && !already_dense_output_y ) {
#else
			if( !dense_descr ) {
				already_dense_output_y = false;
#endif
				local_y = internal::getCoordinates( y ).asyncSubset( lower_bound,
					upper_bound );
				local_y.local_assignAllNotAlreadyAssigned();
			}

			for( size_t i = lower_bound; i < upper_bound; ++i ) {
				x_raw[ i ] = in_raw[ i ].first;
				y_raw[ i ] = in_raw[ i ].second;
			}

#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
			if( !dense_descr && !already_dense_output_x ) {
#else
			if( !dense_descr ) {
#endif
				internal::getCoordinates( x ).asyncJoinSubset( local_x, lower_bound,
					upper_bound );
			}

#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
			if( !dense_descr && !already_dense_output_y ) {
#else
			if( !dense_descr ) {
#endif
				internal::getCoordinates( y ).asyncJoinSubset( local_y, lower_bound,
					upper_bound );
			}

			return rc;
		};

		ret = ret ? ret : internal::le.addStage(
				std::move( func ),
				internal::Opcode::BLAS1_UNZIP,
				internal::getCoordinates( x ).size(), std::max( sizeof( T ), sizeof( U ) ),
				dense_descr, true,
				&x, &y,
				&internal::getCoordinates( x ), &internal::getCoordinates( y ),
				&in, nullptr, nullptr, nullptr,
				&internal::getCoordinates( in ), nullptr, nullptr, nullptr,
				nullptr
			);

#ifdef _NONBLOCKING_DEBUG
		std::cout << "\t\tStage added to a pipeline: unzip(x, y, in)" << std::endl;
#endif
		return SUCCESS;
	}

/** @} */
//   ^-- ends BLAS-1 NB module

} // end namespace ``grb''

#undef NO_CAST_ASSERT
#undef NO_CAST_OP_ASSERT

#endif // end `_H_GRB_NONBLOCKING_BLAS1'

