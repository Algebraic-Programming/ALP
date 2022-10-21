
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
 * @date 5th of December 2016
 */

#if !defined _H_GRB_REFERENCE_BLAS1 || defined _H_GRB_REFERENCE_OMP_BLAS1
#define _H_GRB_REFERENCE_BLAS1

#include <graphblas/utils/suppressions.h>

#include <iostream>    //for printing to stderr
#include <type_traits> //for std::enable_if

#include <graphblas/backends.hpp>
#include <graphblas/blas0.hpp>
#include <graphblas/descriptors.hpp>
#include <graphblas/internalops.hpp>
#include <graphblas/ops.hpp>
#include <graphblas/rc.hpp>
#include <graphblas/semiring.hpp>

#include "coordinates.hpp"
#include "vector.hpp"

#ifdef _H_GRB_REFERENCE_OMP_BLAS1
 #include <omp.h>
#endif

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

	/**
	 * \defgroup BLAS1_REF The Level-1 ALP/GraphBLAS routines -- reference backend
	 *
	 * @{
	 */

	namespace internal {

		/**
		 * Folds a vector into a scalar assuming the vector is dense.
		 */
		template< bool left, class Monoid, typename InputType, class Coords >
		RC fold_from_vector_to_scalar_dense(
			typename Monoid::D3 &global,
			const Vector< InputType, reference, Coords > &to_fold,
			const Monoid &monoid
		) {
			const InputType *__restrict__ const raw = internal::getRaw( to_fold );
			const size_t n = internal::getCoordinates( to_fold ).nonzeroes();
			assert( n == internal::getCoordinates( to_fold ).size() );
			assert( n > 0 );
			RC ret = SUCCESS;
			size_t global_start, global_end;
			if( left ) {
				global = raw[ 0 ];
				global_start = 1;
				global_end = n;
			} else {
				global = raw[ n - 1 ];
				global_start = 0;
				global_end = n - 1;
			}

			// catch trivial case
			if( global_start >= global_end ) {
				return SUCCESS;
			}

#ifdef _H_GRB_REFERENCE_OMP_BLAS1
			#pragma omp parallel
			{
				size_t start, end;
				config::OMP::localRange( start, end, global_start, global_end );
#else
				const size_t start = global_start;
				const size_t end = global_end;
#endif
				if( start < end ) {
					typename Monoid::D3 local =
						monoid.template getIdentity< typename Monoid::D3 >();
					if( left ) {
						monoid.getOperator().foldlArray( local, raw + start, end - start );
					} else {
						monoid.getOperator().foldrArray( raw + start, local, end - start );
					}
					RC local_rc = SUCCESS;
#ifdef _H_GRB_REFERENCE_OMP_BLAS1
					#pragma omp critical
					{
#endif
#ifdef _DEBUG
						std::cout << "\t\t folding " << local << " into " << global << "\n";
#endif
						if( left ) {
							local_rc = foldl( global, local, monoid.getOperator() );
						} else {
							local_rc = foldr( local, global, monoid.getOperator() );
						}
#ifdef _H_GRB_REFERENCE_OMP_BLAS1
					}
#endif
					if( local_rc != SUCCESS ) {
						ret = local_rc;
					}
				}
#ifdef _H_GRB_REFERENCE_OMP_BLAS1
			}
#endif
			return ret;
		}

		/**
		 * Folds a vector into a scalar.
		 *
		 * May be masked, and the vector is assumed sparse.
		 *
		 * This variant is driven by the sparsity pattern of the vector.
		 */
		template<
			Descriptor descr,
			bool masked, bool left, class Monoid,
			typename InputType, typename MaskType,
			class Coords
		>
		RC fold_from_vector_to_scalar_vectorDriven(
			typename Monoid::D3 &global,
			const Vector< InputType, reference, Coords > &to_fold,
			const Vector< MaskType, reference, Coords > &mask,
			const Monoid &monoid
		) {
			const size_t n = internal::getCoordinates( to_fold ).size();
			const size_t nz = internal::getCoordinates( to_fold ).nonzeroes();
#ifdef NDEBUG
			(void) n;
#endif

			assert( n > 0 );
			assert( nz > 0 );
			assert( !masked || internal::getCoordinates( mask ).size() == n );

			RC ret = SUCCESS;

			// compute in parallel
#ifdef _H_GRB_REFERENCE_OMP_BLAS1
			#pragma omp parallel
			{
				size_t start, end;
				config::OMP::localRange( start, end, 0, nz );
#else
				const size_t start = 0;
				const size_t end = nz;
#endif
				// compute thread-local partial reduction
				typename Monoid::D3 local =
					monoid.template getIdentity< typename Monoid::D3 >();
				for( size_t k = start; k < end; ++k ) {
					const size_t i = internal::getCoordinates( to_fold ).index( k );
					if( masked && !utils::interpretMask< descr >(
							internal::getCoordinates( mask ).assigned( i ),
							internal::getRaw( mask ),
							i
						)
					) {
						continue;
					}
					RC local_rc;
					if( left ) {
						local_rc = foldl< descr >(
								local, internal::getRaw( to_fold )[ i ],
								monoid.getOperator()
							);
					} else {
						local_rc = foldr< descr >(
								internal::getRaw( to_fold )[ i ], local,
								monoid.getOperator()
							);
					}
					assert( local_rc == SUCCESS );
					if( local_rc != SUCCESS ) {
						ret = local_rc;
					}
				}

				// fold into global
#ifdef _H_GRB_REFERENCE_OMP_BLAS1
				#pragma omp critical
				{
#endif
					if( ret == SUCCESS && start < end ) {
						if( left ) {
							ret = foldl< descr >( global, local, monoid.getOperator() );
						} else {
							ret = foldr< descr >( local, global, monoid.getOperator() );
						}
						assert( ret == SUCCESS );
					}
#ifdef _H_GRB_REFERENCE_OMP_BLAS1
				}

			} // end pragma omp parallel
#endif

			// done
			return ret;
		}

		/**
		 * Folds a vector into a scalar.
		 *
		 * Must be masked, and both mask and vector are assumed sparse.
		 *
		 * This variant is driven by the sparsity pattern of the mask.
		 */
		template<
			Descriptor descr,
			bool left, class Monoid,
			typename InputType, typename MaskType,
			class Coords
		>
		RC fold_from_vector_to_scalar_maskDriven(
			typename Monoid::D3 &global,
			const Vector< InputType, reference, Coords > &to_fold,
			const Vector< MaskType, reference, Coords > &mask,
			const Monoid &monoid
		) {
			const size_t n = internal::getCoordinates( to_fold ).size();
			const size_t nz = internal::getCoordinates( mask ).nonzeroes();

			assert( internal::getCoordinates( mask ).size() == n );
			assert( n > 0 );
			assert( nz > 0 );
#ifdef NDEBUG
			(void) n;
#endif

			RC ret = SUCCESS;

			// compute in parallel
#ifdef _H_GRB_REFERENCE_OMP_BLAS1
			#pragma omp parallel
			{
				size_t start, end;
				config::OMP::localRange( start, end, 0, nz );
#else
				const size_t start = 0;
				const size_t end = nz;
#endif
				// compute thread-local partial reduction
				typename Monoid::D3 local =
					monoid.template getIdentity< typename Monoid::D3 >();
				for( size_t k = start; k < end; ++k ) {
					const size_t i = internal::getCoordinates( mask ).index( k );
					if( !internal::getCoordinates( to_fold ).assigned( i ) ) {
						continue;
					}
					if( !utils::interpretMask< descr >( true, internal::getRaw( mask ), i ) ) {
						continue;
					}
					RC local_rc;
					if( left ) {
						local_rc = foldl< descr >(
								local, internal::getRaw( to_fold )[ i ],
								monoid.getOperator()
							);
					} else {
						local_rc = foldr< descr >(
								internal::getRaw( to_fold )[ i ], local,
								monoid.getOperator()
							);
					}
					assert( local_rc == SUCCESS );
					if( local_rc != SUCCESS ) {
						ret = local_rc;
					}
				}

				// fold into global
#ifdef _H_GRB_REFERENCE_OMP_BLAS1
				#pragma omp critical
				{
#endif
					if( ret == SUCCESS && start < end ) {
						if( left ) {
							ret = foldl< descr >( global, local, monoid.getOperator() );
						} else {
							ret = foldr< descr >( local, global, monoid.getOperator() );
						}
						assert( ret == SUCCESS );
					}
#ifdef _H_GRB_REFERENCE_OMP_BLAS1
				}

			} // end pragma omp parallel
#endif

			// done
			return ret;
		}

		/**
		 * Folds a vector into a scalar.
		 *
		 * May be masked, and the vector may be sparse.
		 *
		 * This variant uses an O(n) loop, where n is the size of the vector.
		 */
		template<
			Descriptor descr,
			bool masked, bool left, class Monoid,
			typename InputType, typename MaskType,
			class Coords
		>
		RC fold_from_vector_to_scalar_fullLoopSparse(
			typename Monoid::D3 &global,
			const Vector< InputType, reference, Coords > &to_fold,
			const Vector< MaskType, reference, Coords > &mask,
			const Monoid &monoid
		) {
#ifdef _DEBUG
			std::cout << "Entered fold_from_vector_to_scalar_fullLoopSparse\n";
#endif
			const auto &to_fold_coors = internal::getCoordinates( to_fold );
			const size_t n = to_fold_coors.size();
			assert( n > 0 );
			RC ret = SUCCESS;
#ifdef _H_GRB_REFERENCE_OMP_BLAS1
			#pragma omp parallel
			{
				// parallel case (masked & unmasked)
				size_t i, end;
				config::OMP::localRange( i, end, 0, n );
#else
				// masked sequential case
				size_t i = 0;
				const size_t end = n;
#endif
				// some sanity checks
				assert( i <= end );
				assert( end <= n );

				// assume current i needs to be processed, forward until we find an index
				// for which the mask evaluates true
				bool process_current_i = true;
				if( masked && i < end ) {
					process_current_i = utils::interpretMask< descr >(
						internal::getCoordinates( mask ).assigned( i ),
						internal::getRaw( mask ),
						i
					) && to_fold_coors.assigned( i );
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
							internal::getCoordinates( mask ).assigned( i ),
							internal::getRaw( mask ),
							i
						) && to_fold_coors.assigned( i );
					}
				}
				if( !masked && i < end ) {
					process_current_i = to_fold_coors.assigned( i );
					while( !process_current_i ) {
						(void) ++i;
						if( i == end ) {
							break;
						}
						process_current_i = to_fold_coors.assigned( i );
					}
				}

				// whether we have any nonzeroes assigned at all
				const bool empty = i >= end;
#ifndef _H_GRB_REFERENCE_OMP_BLAS1
				(void) empty;
#endif

#ifndef NDEBUG
				if( i < end ) {
					assert( i < n );
				}
#endif

				// declare thread-local variable and set our variable to the first value in our block
				typename Monoid::D3 local =
					monoid.template getIdentity< typename Monoid::D3 >();
				if( end > 0 ) {
					if( i < end ) {
#ifdef _DEBUG
						std::cout << "\t processing start index " << i << "\n";
#endif

						local = static_cast< typename Monoid::D3 >(
								internal::getRaw( to_fold )[ i ]
							);
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
								internal::getCoordinates( mask ).assigned( i ),
								internal::getRaw( mask ),
								i
							) && to_fold_coors.assigned( i );
							while( !process_current_i ) {
								(void) ++i;
								if( i == end ) {
									break;
								}
								assert( i < end );
								assert( i < n );
								process_current_i = utils::interpretMask< descr >(
									internal::getCoordinates( mask ).assigned( i ),
									internal::getRaw( mask ),
									i
								) && to_fold_coors.assigned( i );
							}
						}
						if( !masked && i < end ) {
							assert( i < n );
							process_current_i = to_fold_coors.assigned( i );
							while( !process_current_i ) {
								(void) ++i;
								if( i == end ) {
									break;
								}
								assert( i < end );
								assert( i < n );
								process_current_i = to_fold_coors.assigned( i );
							}
						}

						// stop if past end
						if( i >= end ) {
							break;
						}

#ifdef _DEBUG
						std::cout << "\t processing index " << i << "\n";
#endif

						// store result of fold in local variable
						RC local_rc;

						// do fold
						assert( i < n );
						if( left ) {
							local_rc = foldl< descr >(
									local, internal::getRaw( to_fold )[ i ],
									monoid.getOperator()
								);
						} else {
							local_rc = foldr< descr >(
									internal::getRaw( to_fold )[ i ], local,
									monoid.getOperator()
								);
						}
						assert( local_rc == SUCCESS );

						// error propagation
						if( local_rc != SUCCESS ) {
							ret = local_rc;
							break;
						}
					}
				}

#ifdef _H_GRB_REFERENCE_OMP_BLAS1
				// reduce all local folds into the global one
				#pragma omp critical
				{
					// if non-empty, fold local variable into global one
					if( !empty ) {
						// local return type to avoid racing writes
						RC local_rc;
						if( left ) {
							local_rc = foldl< descr >( global, local, monoid.getOperator() );
						} else {
							local_rc = foldr< descr >( local, global, monoid.getOperator() );
						}
						assert( local_rc == SUCCESS );
						if( local_rc != SUCCESS ) {
							ret = local_rc;
						}
					}
				}
			} // end pragma omp parallel
#else
			// in the sequential case, simply copy the locally computed reduced scalar
			// into the output field
			global = local;
#endif

			// done
			return ret;
		}

#ifndef _H_GRB_REFERENCE_OMP_BLAS1
		/**
		 * A helper template class for selecting the right variant for
		 * fold-from-vector-to-scalar.
		 *
		 * When the mask is structural, the returned value shall be zero, otherwise
		 * it will be the byte size of \a MaskType.
		 */
		template< Descriptor descr, typename MaskType >
		struct MaskWordSize {
			static constexpr const size_t value = (descr & descriptors::structural)
				? 0
				: sizeof( MaskType );
		};

		/**
		 * Specialisation for <tt>void</tt> mask types.
		 *
		 * Always returns zero.
		 */
		template< Descriptor descr >
		struct MaskWordSize< descr, void > {
			static constexpr const size_t value = 0;
		};
#endif

		/**
		 * Dispatches to any of the four above variants depending on asymptotic cost
		 * analysis.
		 */
		template<
			Descriptor descr = descriptors::no_operation,
			bool masked, bool left, // if this is false, assumes right-looking fold
			class Monoid,
			typename IOType, typename InputType, typename MaskType,
			typename Coords
		>
		RC fold_from_vector_to_scalar_generic(
			IOType &fold_into,
			const Vector< InputType, reference, Coords > &to_fold,
			const Vector< MaskType, reference, Coords > &mask,
			const Monoid &monoid
		) {
			// static sanity checks
			static_assert( grb::is_monoid< Monoid >::value,
				"grb::foldl can only be called using monoids. This "
				"function should not have been called-- please submit a "
				"bugreport." );

			const size_t n  = internal::getCoordinates( to_fold ).size();
			const size_t nz = internal::getCoordinates( to_fold ).nonzeroes();

			// mask must be of equal size as input vector
			if( masked && n != size( mask ) ) {
				return MISMATCH;
			}

			// density checks, if needed
			if( (descr & descriptors::dense) ) {
				if( nnz( to_fold ) < n ) {
					return ILLEGAL;
				}
				if( masked && nnz( mask ) < size( mask ) ) {
					return ILLEGAL;
				}
			}

			// handle trivial cases
			if( n == 0 ) {
				return SUCCESS;
			}
			if( nz == 0 ) {
				return SUCCESS;
			}
			if( masked && !(descr & descriptors::invert_mask) &&
				nnz( mask ) == 0
			) {
				return SUCCESS;
			}
			if( masked && (descr & descriptors::invert_mask) &&
				(descr & descriptors::structural) &&
				nnz( mask ) == n
			) {
				return SUCCESS;
			}

			// some globals used during the folding
			RC ret = SUCCESS;
			typename Monoid::D3 global =
				monoid.template getIdentity< typename Monoid::D3 >();

			// dispatch, dense variant
			if( ((descr & descriptors::dense) || nnz( to_fold ) == n) && (
					!masked || (
						(descr & descriptors::structural) &&
						!(descr & descriptors::invert_mask) &&
						nnz( mask ) == n
					)
				)
			) {
#ifdef _DEBUG
				std::cout << "\t dispatching to dense variant\n";
#endif
				ret = fold_from_vector_to_scalar_dense< left >(
					global, to_fold, monoid );
			} else if( masked && (descr & descriptors::invert_mask ) ) {
				// in this case we are forced to dispatch to O(n)
#ifdef _DEBUG
				std::cout << "\t forced dispatch to O(n) sparse variant\n";
#endif
				ret = fold_from_vector_to_scalar_fullLoopSparse< descr, true, left >(
					global, to_fold, mask, monoid );
			} else {
				constexpr const size_t threeWs =
					sizeof( typename Coords::StackType ) +
					sizeof( typename Coords::ArrayType ) +
					MaskWordSize< descr, MaskType >::value;
				const size_t fullLoop = masked
					? 2 * sizeof( typename Coords::ArrayType ) * n +
						sizeof( MaskType ) * nnz( mask )
					: sizeof( typename Coords::ArrayType ) * n;
				const size_t vectorLoop = masked
					? threeWs * nnz( to_fold )
					: sizeof( typename Coords::StackType ) * nnz( to_fold );
				const size_t maskLoop = masked
					? threeWs * nnz( mask )
					: std::numeric_limits< size_t >::max();
				if( fullLoop >= vectorLoop && maskLoop >= vectorLoop ) {
#ifdef _DEBUG
					std::cout << "\t dispatching to vector-driven sparse variant\n";
#endif
					ret = fold_from_vector_to_scalar_vectorDriven< descr, masked, left >(
						global, to_fold, mask, monoid );
				} else if( vectorLoop >= fullLoop && maskLoop >= fullLoop ) {
#ifdef _DEBUG
					std::cout << "\t dispatching to O(n) sparse variant\n";
#endif
					ret = fold_from_vector_to_scalar_fullLoopSparse< descr, masked, left >(
						global, to_fold, mask, monoid );
				} else {
					assert( maskLoop < fullLoop && maskLoop < vectorLoop );
					assert( masked );
#ifdef _DEBUG
					std::cout << "\t dispatching to mask-driven sparse variant\n";
#endif
					ret = fold_from_vector_to_scalar_maskDriven< descr, left >(
						global, to_fold, mask, monoid );
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

			// done
			return ret;
		}

		/**
		 * \internal Only applies to sparse vectors and non-monoid folding.
		 */
		template<
			Descriptor descr,
			bool left, bool masked,
			typename IOType, typename MaskType, typename InputType,
			class OP, typename Coords
		>
		RC fold_from_scalar_to_vector_generic_vectorDriven(
			Vector< IOType, reference, Coords > &vector,
			const MaskType * __restrict__ m,
			const Coords * const m_coors,
			const InputType &scalar,
			const OP &op,
			const Phase &phase
		) {
#ifndef NDEBUG
			constexpr const bool dense_descr = descr & descriptors::dense;
#endif
			const size_t n = size( vector );

			// input checking is done by fold_from_scalar_to_vector_generic
			// we hence here only assert
			assert( !masked || m_coors->size() == n );
			assert( !dense_descr || nnz( vector ) == n );
			assert( !dense_descr || !masked || m_coors->nonzeroes() == n );

			if( n == 0 ) {
				return SUCCESS;
			}
			if( phase == RESIZE ) {
				return SUCCESS;
			}

			assert( phase == EXECUTE );
			assert( !masked || (m_coors != nullptr) );
			IOType * __restrict__ x = internal::getRaw( vector );
			Coords &coors = internal::getCoordinates( vector );
			assert( coors.nonzeroes() < coors.size() );

			if( masked ) {
#ifdef _H_GRB_REFERENCE_OMP_BLAS1
				// choose dynamic schedule since the mask otherwise likely leads to
				// significant imbalance
				#pragma omp parallel for schedule( dynamic, config::CACHE_LINE_SIZE::value() )
#endif
				for( size_t i = 0; i < n; ++i ) {
					const size_t index = coors.index( i );
					if( !( m_coors->template mask< descr >( index, m ) ) ) {
						continue;
					}
					if( left ) {
						(void) foldl< descr >( x[ index ], scalar, op );
					} else {
						(void) foldr< descr >( scalar, x[ index ], op );
					}
				}
			} else {
#ifdef _H_GRB_REFERENCE_OMP_BLAS1
				#pragma omp parallel
				{
					size_t start, end;
					config::OMP::localRange( start, end, 0, n );
#else
					const size_t start = 0;
					const size_t end = n;
#endif
					for( size_t i = start; i < end; ++i ) {
						const size_t index = coors.index( i );
						if( left ) {
							(void) foldl< descr >( x[ index ], scalar, op );
						} else {
							(void) foldr< descr >( scalar, x[ index ], op );
						}
					}
#ifdef _H_GRB_REFERENCE_OMP_BLAS1
				}
#endif
			}
			return SUCCESS;
		}

		/**
		 * \internal Only applies to masked folding.
		 */
		template<
			Descriptor descr,
			bool left, bool sparse, bool monoid,
			typename IOType, typename MaskType, typename InputType,
			class OP, typename Coords
		>
		RC fold_from_scalar_to_vector_generic_maskDriven(
			Vector< IOType, reference, Coords > &vector,
			const MaskType * __restrict__ m,
			const Coords &m_coors,
			const InputType &scalar,
			const OP &op,
			const Phase &phase
		) {
#ifndef NDEBUG
			constexpr const bool dense_descr = descr & descriptors::dense;
#endif
			const size_t n = size( vector );

			// input checking is done by fold_from_scalar_to_vector_generic
			// we hence here only assert
			assert( m_coors.size() == n );
			assert( !dense_descr || nnz( vector ) == n );
			assert( !dense_descr || m_coors.nonzeroes() == n );

			if( n == 0 ) {
				return SUCCESS;
			}
			if( phase == RESIZE ) {
				return SUCCESS;
			}

			assert( phase == EXECUTE );
			IOType * __restrict__ x = internal::getRaw( vector );
			Coords &coors = internal::getCoordinates( vector );
#ifdef _H_GRB_REFERENCE_OMP_BLAS1
			#pragma omp parallel
			{
				auto localUpdate = coors.EMPTY_UPDATE();
				const size_t maxAsyncAssigns = coors.maxAsyncAssigns();
				size_t asyncAssigns = 0;
				// choose dynamic schedule since the mask otherwise likely leads to
				// significant imbalance
				#pragma omp for schedule( dynamic, config::CACHE_LINE_SIZE::value() ) nowait
				for( size_t i = 0; i < m_coors.nonzeroes(); ++i ) {
					const size_t index = m_coors.index( i );
					if( !m_coors.template mask< descr >( index, m ) ) {
						continue;
					}
					if( !sparse || coors.asyncAssign( index, localUpdate ) ) {
						if( left ) {
							(void) foldl< descr >( x[ index ], scalar, op );
						} else {
							(void) foldr< descr >( scalar, x[ index ], op );
						}
					} else if( sparse && monoid ) {
						x[ index ] = static_cast< IOType >( scalar );
						(void) asyncAssigns++;
						if( asyncAssigns == maxAsyncAssigns ) {
							(void) coors.joinUpdate( localUpdate );
							asyncAssigns = 0;
						}
					}
				}
				while( sparse && monoid && !coors.joinUpdate( localUpdate ) ) {}
			} // end pragma omp parallel
#else
			for( size_t i = 0; i < m_coors.nonzeroes(); ++i ) {
				const size_t index = m_coors.index( i );
				if( !m_coors.template mask< descr >( index, m ) ) {
					continue;
				}
				if( !sparse || coors.assign( index ) ) {
					if( left ) {
						(void) foldl< descr >( x[ index ], scalar, op );
					} else {
						(void) foldr< descr >( scalar, x[ index ], op );
					}
				} else if( sparse && monoid ) {
					x[ index ] = static_cast< IOType >( scalar );
				}
			}
#endif
			return SUCCESS;
		}

		template< Descriptor descr,
			bool left,   // if this is false, the right-looking fold is assumed
			bool sparse, // whether \a vector was sparse
			bool masked,
			bool monoid, // whether \a op was passed as a monoid
			typename MaskType, typename IOType, typename InputType,
			typename Coords, class OP
		>
		RC fold_from_scalar_to_vector_generic(
			Vector< IOType, reference, Coords > &vector,
			const MaskType * __restrict__ m,
			const Coords * const m_coors,
			const InputType &scalar,
			const OP &op,
			const Phase &phase
		) {
			constexpr const bool dense_descr = descr & descriptors::dense;
			assert( !masked || m != nullptr );
			assert( !masked || m_coors != nullptr );
			auto &coor = internal::getCoordinates( vector );
			const size_t n = coor.size();

			if( masked && m_coors->size() != n ) {
				return MISMATCH;
			}
			if( dense_descr && sparse ) {
				return ILLEGAL;
			}
			if( dense_descr && nnz( vector ) < n ) {
				return ILLEGAL;
			}
			if( dense_descr && masked && m_coors->nonzeroes() < n ) {
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

			if( sparse && monoid && !masked ) {
				// output will become dense, use Theta(n) loop
#ifdef _H_GRB_REFERENCE_OMP_BLAS1
				#pragma omp parallel
				{
					size_t start, end;
					config::OMP::localRange( start, end, 0, n );
#else
					const size_t start = 0;
					const size_t end = n;
#endif
					for( size_t i = start; i < end; ++i ) {
						if( coor.assigned( i ) ) {
							if( left ) {
								(void) foldl< descr >( x[ i ], scalar, op );
							} else {
								(void) foldr< descr >( scalar, x[ i ], op );
							}
						} else {
							x[ i ] = static_cast< IOType >( scalar );
						}
					}
#ifdef _H_GRB_REFERENCE_OMP_BLAS1
				}
#endif
				coor.assignAll();
				return SUCCESS;
			} else if( sparse && monoid && masked ) {
				return fold_from_scalar_to_vector_generic_maskDriven<
						descr, left, true, true
					>( vector, m, *m_coors, scalar, op, phase );
			} else if( sparse && !monoid ) {
				const bool maskDriven = masked ? m_coors->nonzeroes() < coor.nonzeroes() : false;
				if( maskDriven ) {
					return fold_from_scalar_to_vector_generic_maskDriven<
						descr, left, true, false
					>( vector, m, *m_coors, scalar, op, phase );
				} else {
					return fold_from_scalar_to_vector_generic_vectorDriven<
							descr, left, masked
						>( vector, m, m_coors, scalar, op, phase );
				}
			} else if( !sparse && masked ) {
				return fold_from_scalar_to_vector_generic_maskDriven<
						descr, left, false, monoid
					>( vector, m, *m_coors, scalar, op, phase );
			} else {
				// if target vector is dense and there is no mask, then
				// there is no difference between monoid or non-monoid behaviour.
				assert( !sparse );
				assert( !masked );
				assert( coor.nonzeroes() == coor.size() );
#ifdef _H_GRB_REFERENCE_OMP_BLAS1
				#pragma omp parallel
				{
					size_t start, end;
					config::OMP::localRange( start, end, 0, coor.size() );
#else
					const size_t start = 0;
					const size_t end = coor.size();
#endif
					const size_t local_n = end - start;
					if( local_n > 0 ) {
						if( left ) {
							op.eWiseFoldlAS( internal::getRaw( vector ) + start, scalar, local_n );
						} else {
							op.eWiseFoldrSA( scalar, internal::getRaw( vector ) + start, local_n );
						}
					}
#ifdef _H_GRB_REFERENCE_OMP_BLAS1
				} // end pragma omp parallel
#endif
			}
			return SUCCESS;
		}

		/**
		 * Generic fold implementation on two vectors.
		 *
		 * @tparam descr  The descriptor under which the operation takes place.
		 * @tparam left   Whether we are folding left (or right, otherwise).
		 * @tparam sparse Whether one of \a fold_into or \a to_fold is sparse.
		 * @tparam OP     The operator to use while folding.
		 * @tparam IType  The input data type (of \a to_fold).
		 * @tparam IOType The input/output data type (of \a fold_into).
		 *
		 * \note Sparseness is passed explicitly since it is illegal when not
		 *       called using a monoid. This function, however, has no way to
		 *       check for this user input.
		 *
		 * @param[in,out] fold_into The vector whose elements to fold into.
		 * @param[in]     to_fold   The vector whose elements to fold.
		 * @param[in]     op        The operator to use while folding.
		 *
		 * The sizes of \a fold_into and \a to_fold must match; this is an elementwise
		 * fold.
		 *
		 * @returns #ILLEGAL  If \a sparse is <tt>false</tt> while one of \a fold_into
		 *                    or \a to_fold is sparse.
		 * @returns #MISMATCH If the sizes of \a fold_into and \a to_fold do not
		 *                    match.
		 * @returns #SUCCESS  On successful completion of this function call.
		 */
		template<
			Descriptor descr,
			bool left, // if this is false, the right-looking fold is assumed
			bool sparse, bool masked, bool monoid,
			typename MaskType, typename IOType, typename IType,
			class OP,
			typename Coords
		>
		RC fold_from_vector_to_vector_generic(
			Vector< IOType, reference, Coords > &fold_into,
			const Vector< MaskType, reference, Coords > * const m,
			const Vector< IType, reference, Coords > &to_fold,
			const OP &op,
			const Phase phase
		) {
			constexpr const bool dense_descr = descr & descriptors::dense;
			assert( !masked || (m != nullptr) );
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
			if( !sparse && nnz( fold_into ) < n ) {
				return ILLEGAL;
			}
			if( !sparse && nnz( to_fold ) < n ) {
				return ILLEGAL;
			}
			if( phase == RESIZE ) {
				return SUCCESS;
			}

			assert( phase == EXECUTE );

			// take at least a number of elements so that no two threads operate on the
			// same cache line
#ifdef _H_GRB_REFERENCE_OMP_BLAS1
			const constexpr size_t blocksize =
				config::SIMD_BLOCKSIZE< IOType >::value() >
					config::SIMD_BLOCKSIZE< IType >::value() ?
				config::SIMD_BLOCKSIZE< IOType >::value() :
				config::SIMD_BLOCKSIZE< IType >::value();
			static_assert( blocksize > 0, "Config error: zero blocksize in call to"
				"fold_from_vector_to_vector_generic!" );
#endif
			if( !sparse && !masked ) {
#ifdef _DEBUG
				std::cout << "fold_from_vector_to_vector_generic: in dense variant\n";
#endif
#ifdef _H_GRB_REFERENCE_OMP_BLAS1
 #ifdef _DEBUG
				std::cout << "fold_from_vector_to_vector_generic: in OpenMP variant\n";
 #endif
				#pragma omp parallel
				{
					size_t start, end;
					config::OMP::localRange( start, end, 0, n, blocksize );
					const size_t range = end - start;
					assert( end <= n );
					assert( range + start <= n );
					if( left ) {
						op.eWiseFoldlAA( internal::getRaw( fold_into ) + start,
							internal::getRaw( to_fold ) + start, range );
					} else {
						op.eWiseFoldrAA( internal::getRaw( to_fold ) + start,
							internal::getRaw( fold_into ) + start, range );
					}
				}
#else
#ifdef _DEBUG
				std::cout << "fold_from_vector_to_vector_generic: in sequential variant\n";
#endif
				if( left ) {
					op.eWiseFoldlAA( internal::getRaw( fold_into ),
						internal::getRaw( to_fold ), n );
				} else {
					op.eWiseFoldrAA( internal::getRaw( to_fold ),
						internal::getRaw( fold_into ), n );
				}
#endif
			} else {
#ifdef _DEBUG
				std::cout << "fold_from_vector_to_vector_generic: in sparse variant\n";
				std::cout << "\tfolding vector of " << nnz( to_fold ) << " nonzeroes "
					<< "into a vector of " << nnz( fold_into ) << " nonzeroes...\n";
#endif
				if( masked && nnz( fold_into ) == n && nnz( to_fold ) == n ) {
					// use sparsity structure of mask for this eWiseFold
					if( left ) {
#ifdef _DEBUG
						std::cout << "fold_from_vector_to_vector_generic: using eWiseLambda, "
							<< "foldl, using to_fold's sparsity\n";
#endif
						return eWiseLambda(
							[ &fold_into, &to_fold, &op ]( const size_t i ) {
#ifdef _DEBUG
								std::cout << "Left-folding " << to_fold[ i ] << " into " << fold_into[ i ];
#endif
								(void) foldl< descr >( fold_into[ i ], to_fold[ i ], op );
#ifdef _DEBUG
								std::cout << " resulting into " << fold_into[ i ] << "\n";
#endif
							}, *m, to_fold, fold_into );
					} else {
#ifdef _DEBUG
						std::cout << "fold_from_vector_to_vector_generic: using eWiseLambda, "
							<< "foldl, using to_fold's sparsity\n";
#endif
						return eWiseLambda(
							[ &fold_into, &to_fold, &op ]( const size_t i ) {
#ifdef _DEBUG
								std::cout << "Right-folding " << to_fold[ i ] << " into "
									<< fold_into[ i ];
#endif
								(void) foldr< descr >( to_fold[ i ], fold_into[ i ], op );
#ifdef _DEBUG
								std::cout << " resulting into " << fold_into[ i ] << "\n";
#endif
							}, *m, to_fold, fold_into );
					}
				} else if( !masked && nnz( fold_into ) == n ) {
					// use sparsity structure of to_fold for this eWiseFold
					if( left ) {
#ifdef _DEBUG
						std::cout << "fold_from_vector_to_vector_generic: using eWiseLambda, "
							<< "foldl, using to_fold's sparsity\n";
#endif
						return eWiseLambda(
							[ &fold_into, &to_fold, &op ]( const size_t i ) {
#ifdef _DEBUG
								std::cout << "Left-folding " << to_fold[ i ] << " into "
									<< fold_into[ i ];
#endif
								(void) foldl< descr >( fold_into[ i ], to_fold[ i ], op );
#ifdef _DEBUG
								std::cout << " resulting into " << fold_into[ i ] << "\n";
#endif
							}, to_fold, fold_into );
					} else {
#ifdef _DEBUG
						std::cout << "fold_from_vector_to_vector_generic: using eWiseLambda, "
							<< "foldl, using to_fold's sparsity\n";
#endif
						return eWiseLambda(
							[ &fold_into, &to_fold, &op ]( const size_t i ) {
#ifdef _DEBUG
								std::cout << "Right-folding " << to_fold[ i ] << " into "
									<< fold_into[ i ];
#endif
								(void) foldr< descr >( to_fold[ i ], fold_into[ i ], op );
#ifdef _DEBUG
								std::cout << " resulting into " << fold_into[ i ] << "\n";
#endif
							}, to_fold, fold_into );
					}
				} else if( !masked && nnz( to_fold ) == n ) {
					// use sparsity structure of fold_into for this eWiseFold
					if( left ) {
#ifdef _DEBUG
						std::cout << "fold_from_vector_to_vector_generic: using eWiseLambda, "
							"foldl, using fold_into's sparsity\n";
#endif
						return eWiseLambda(
							[ &fold_into, &to_fold, &op ]( const size_t i ) {
#ifdef _DEBUG
								std::cout << "Left-folding " << to_fold[ i ] << " into "
									<< fold_into[ i ];
#endif
								(void) foldl< descr >( fold_into[ i ], to_fold[ i ], op );
#ifdef _DEBUG
								std::cout << " resulting into " << fold_into[ i ] << "\n";
#endif
							},
							fold_into, to_fold );
					} else {
#ifdef _DEBUG
						std::cout << "fold_from_vector_to_vector_generic: using eWiseLambda, "
							<< "foldr, using fold_into's sparsity\n";
#endif
						return eWiseLambda(
							[ &fold_into, &to_fold, &op ]( const size_t i ) {
#ifdef _DEBUG
								std::cout << "Right-folding " << to_fold[ i ] << " into "
									<< fold_into[ i ];
#endif
								(void) foldr< descr >( to_fold[ i ], fold_into[ i ], op );
#ifdef _DEBUG
								std::cout << " resulting into " << fold_into[ i ] << "\n";
#endif
							}, fold_into, to_fold );
					}
					/* TODO: internal issue #66. Also replaces the above eWiseLambda
					} else if( !monoid ) {
#ifdef _DEBUG
					    std::cout << "fold_from_vector_to_vector_generic (non-monoid): using "
						<< "specialised code to merge two sparse vectors and/or to handle "
						<< "output masks\n";
#endif
					    //both sparse, cannot rely on #eWiseLambda
					    const bool intoDriven = nnz( fold_into ) < nnz( to_fold );
					    if( masked ) {
					        if( intoDriven && (nnz( *m ) < nnz( fold_into )) ) {
					            // maskDriven
					        } else {
					            // dstDriven
					        }
					    } else if( intoDriven ) {
					        //dstDriven
					    } else {
					        //srcDriven
					    }
					} else {
					    const bool vectorDriven = nnz( fold_into ) + nnz( to_fold ) < size( fold_into );
					    const bool maskDriven = masked ?
					        nnz( *m ) < (nnz( fold_into ) + nnz( to_fold )) :
					        false;
					    if( maskDriven ) {
					        // maskDriven
					    } else if( vectorDriven ) {
					        // vectorDriven (monoid)
					    } else {
					        // Theta(n) loop
					    }
					}*/
				} else {
#ifdef _DEBUG
					std::cout << "fold_from_vector_to_vector_generic (non-monoid): using "
						<< "specialised code to merge two sparse vectors and/or to "
						<< "handle output masks\n";
#endif
					assert( !monoid );
					const IType * __restrict__ const tf_raw = internal::getRaw( to_fold );
					IOType * __restrict__ const fi_raw = internal::getRaw( fold_into );
					auto &fi = internal::getCoordinates( fold_into );
					const auto &tf = internal::getCoordinates( to_fold );
#ifdef _H_GRB_REFERENCE_OMP_BLAS1
 #ifdef _DEBUG
					std::cout << "\tfold_from_vector_to_vector_generic, "
						<< "in OpenMP parallel code\n";
 #endif
					#pragma omp parallel
					{
						size_t start, end;
						config::OMP::localRange( start, end, 0, tf.nonzeroes() );
						internal::Coordinates< reference >::Update local_update =
							fi.EMPTY_UPDATE();
						const size_t maxAsyncAssigns = fi.maxAsyncAssigns();
						size_t asyncAssigns = 0;
						for( size_t k = start; k < end; ++k ) {
							const size_t i = tf.index( k );
							if( masked && !internal::getCoordinates( *m ).template mask< descr >(
									i,
									internal::getRaw( *m )
								)
							) {
								continue;
							}
							if( fi.assigned( i ) ) {
								if( left ) {
#ifdef _DEBUG
									#pragma omp critical
									{
										std::cout << "\tfoldl< descr >( fi_raw[ i ], tf_raw[ i ], op ), i = "
											<< i << ": " << tf_raw[ i ] << " goes into " << fi_raw[ i ];
									}
#endif
									(void) foldl< descr >( fi_raw[ i ], tf_raw[ i ], op );
#ifdef _DEBUG
									#pragma omp critical
									std::cout << " which results in " << fi_raw[ i ] << "\n";
#endif
								} else {
#ifdef _DEBUG
									#pragma omp critical
									{
										std::cout << "\tfoldr< descr >( tf_raw[ i ], fi_raw[ i ], op ), i = "
											<< i << ": " << tf_raw[ i ] << " goes into " << fi_raw[ i ];
									}
#endif
									(void) foldr< descr >( tf_raw[ i ], fi_raw[ i ], op );
#ifdef _DEBUG
									#pragma omp critical
									std::cout << " which results in " << fi_raw[ i ] << "\n";
#endif
								}
							} else if( monoid ) {
#ifdef _DEBUG
								#pragma omp critical
								{
									std::cout << " index " << i << " is unset. Old value " << fi_raw[ i ]
										<< " will be overwritten with " << tf_raw[ i ] << "\n";
								}
#endif
								fi_raw[ i ] = tf_raw[ i ];
								if( !fi.asyncAssign( i, local_update ) ) {
									(void) ++asyncAssigns;
								}
							}
							if( asyncAssigns == maxAsyncAssigns ) {
								const bool was_empty = fi.joinUpdate( local_update );
#ifdef NDEBUG
								(void) was_empty;
#else
								assert( !was_empty );
#endif
								asyncAssigns = 0;
							}
						}
						while( !fi.joinUpdate( local_update ) ) {}
					}
#else
#ifdef _DEBUG
					std::cout << "\tin sequential version...\n";
#endif
					for( size_t k = 0; k < tf.nonzeroes(); ++k ) {
						const size_t i = tf.index( k );
						if( masked && !internal::getCoordinates( *m ).template mask< descr >(
								i,
								internal::getRaw( *m )
							)
						) {
							continue;
						}
						assert( i < n );
						if( fi.assigned( i ) ) {
							if( left ) {
#ifdef _DEBUG
								std::cout << "\tfoldl< descr >( fi_raw[ i ], tf_raw[ i ], op ), i = "
									<< i << ": " << tf_raw[ i ] << " goes into " << fi_raw[ i ];
#endif
								(void) foldl< descr >( fi_raw[ i ], tf_raw[ i ], op );
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
							(void) fi.assign( i );
						}
					}
#endif
				}
			}

#ifdef _DEBUG
			std::cout << "\tCall to fold_from_vector_to_vector_generic done. "
				<< "Output now contains " << nnz( fold_into ) << " / "
				<< size( fold_into ) << " nonzeroes.\n";
#endif

			// done
			return SUCCESS;
		}

	} // namespace internal

	/**
	 * Folds all elements in an ALP/GraphBLAS vector \a x into a single value
	 * \a beta.
	 *
	 * The original value of \a beta is used as the right-hand side input of the
	 * operator \a op. A left-hand side input for \a op is retrieved from the
	 * input vector \a x. The result of the operation is stored in \a beta. This
	 * process is repeated for every element in \a x.
	 *
	 * At function exit, \a beta will equal
	 * \f$ \beta \odot x_0 \odot x_1 \odot \ldots x_{n-1} \f$.
	 *
	 * @tparam descr     The descriptor used for evaluating this function. By
	 *                   default, this is grb::descriptors::no_operation.
	 * @tparam OP        The type of the operator to be applied.
	 *                   The operator must be associative.
	 * @tparam InputType The type of the elements of \a x.
	 * @tparam IOType    The type of the value \a y.
	 *
	 * @param[in]     x    The input vector \a x that will not be modified. This
	 *                     input vector must be dense.
	 * @param[in,out] beta On function entry: the initial value to be applied to
	 *                     \a op from the right-hand side.
	 *                     On function exit: the result of repeated applications
	 *                     from the left-hand side of elements of \a x.
	 * @param[in]    op    The monoid under which to perform this right-folding.
	 *
	 * \note We only define fold under monoids, not under plain operators.
	 *
	 * @returns grb::SUCCESS This function always succeeds.
	 * @returns grb::ILLEGAL When a sparse vector is passed. In this case, the call
	 *                       to this function will have no other effects.
	 *
	 * \warning Since this function folds from left-to-right using binary
	 *          operators, this function \em cannot take sparse vectors as input--
	 *          a monoid is required to give meaning to missing vector entries.
	 *          See grb::reducer for use with sparse vectors instead.
	 *
	 * \parblock
	 * \par Valid descriptors
	 * grb::descriptors::no_operation, grb::descriptors::no_casting.
	 *
	 * \note Invalid descriptors will be ignored.
	 *
	 * If grb::descriptors::no_casting is specified, then 1) the first domain of
	 * \a op must match \a IOType, 2) the second domain of \a op must match
	 * \a InputType, and 3) the third domain must match \a IOType. If one of these
	 * is not true, the code shall not compile.
	 *
	 * \endparblock
	 *
	 * \parblock
	 * \par Valid operator types
	 * The given operator \a op is required to be:
	 *   -# associative.
	 * \endparblock
	 *
	 * \parblock
	 * \par Performance semantics
	 *      -# This call comprises \f$ \Theta(n) \f$ work, where \f$ n \f$ equals
	 *         the size of the vector \a x. The constant factor depends on the
	 *         cost of evaluating the underlying binary operator. A good
	 *         implementation uses vectorised instructions whenever the input
	 *         domains, the output domain, and the operator used allow for this.
	 *
	 *      -# This call will not result in additional dynamic memory allocations.
	 *
	 *      -# This call takes \f$ \mathcal{O}(1) \f$ memory beyond the memory
	 *         used by the application at the point of a call to this function.
	 *
	 *      -# This call incurs at most
	 *         \f$ n \cdot\mathit{sizeof}(\mathit{InputType}) + \mathcal{O}(1) \f$
	 *         bytes of data movement. A good implementation will rely on in-place
	 *         operators.
	 * \endparblock
	 *
	 * @see grb::operators::internal::Operator for a discussion on when in-place
	 *      and/or vectorised operations are used.
	 *
	 * \todo Update and move functional specification to base, and revise
	 *       performance semantics.
	 */
	template<
		Descriptor descr = descriptors::no_operation, class Monoid,
		typename InputType, typename IOType, typename MaskType,
		typename Coords
	>
	RC foldr(
		const Vector< InputType, reference, Coords > &x,
		const Vector< MaskType, reference, Coords > &mask,
		IOType &beta,
		const Monoid &monoid = Monoid(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			!grb::is_object< InputType >::value &&
			!grb::is_object< IOType >::value &&
			!grb::is_object< MaskType >::value &&
			grb::is_monoid< Monoid >::value, void
		>::type * const = nullptr
	) {
		(void) phase;
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
		Descriptor descr = descriptors::no_operation, class Monoid,
		typename InputType, typename IOType,
		typename Coords
	>
	RC foldr(
		const Vector< InputType, reference, Coords > &x,
		IOType &beta,
		const Monoid &monoid = Monoid(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			!grb::is_object< InputType >::value &&
			!grb::is_object< IOType >::value &&
			grb::is_monoid< Monoid >::value, void
		>::type * const = nullptr
	) {
		(void) phase;

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

		grb::Vector< bool, reference, Coords > empty_mask( 0 );
		return internal::template fold_from_vector_to_scalar_generic<
			descr, false, false
		>( beta, x, empty_mask, monoid );
	}

	/**
	 * For all elements in an ALP/GraphBLAS vector \a y, fold the value
	 * \f$ \alpha \f$ into each element.
	 *
	 * The original value of \f$ \alpha \f$ is used as the left-hand side input
	 * of the operator \a op. The right-hand side inputs for \a op are retrieved
	 * from the input vector \a y. The result of the operation is stored in \a y,
	 * thus overwriting its previous values.
	 *
	 * The value of \f$ y_i \f$ after a call to thus function thus equals
	 * \f$ \alpha \odot y_i \f$, for all \f$ i \in \{ 0, 1, \dots, n - 1 \} \f$.
	 *
	 * @tparam descr     The descriptor used for evaluating this function. By
	 *                   default, this is grb::descriptors::no_operation.
	 * @tparam OP        The type of the operator to be applied.
	 * @tparam InputType The type of \a alpha.
	 * @tparam IOType    The type of the elements in \a y.
	 *
	 * @param[in]     alpha The input value to apply as the left-hand side input
	 *                      to \a op.
	 * @param[in,out] y     On function entry: the initial values to be applied as
	 *                      the right-hand side input to \a op.
	 *                      On function exit: the output data.
	 * @param[in]     op    The monoid under which to perform this left-folding.
	 *
	 * @returns grb::SUCCESS This function always succeeds.
	 *
	 * \note We only define fold under monoids, not under plain operators.
	 *
	 * \parblock
	 * \par Valid descriptors
	 * grb::descriptors::no_operation, grb::descriptors::no_casting.
	 *
	 * \note Invalid descriptors will be ignored.
	 *
	 * If grb::descriptors::no_casting is specified, then 1) the first domain of
	 * \a op must match \a IOType, 2) the second domain of \a op must match
	 * \a InputType, and 3) the third domain must match \a IOType. If one of these
	 * is not true, the code shall not compile.
	 *
	 * \endparblock
	 *
	 * \parblock
	 * \par Valid operator types
	 * The given operator \a op is required to be:
	 *   -# (no requirements).
	 * \endparblock
	 *
	 * \parblock
	 * \par Performance semantics
	 *      -# This call comprises \f$ \Theta(n) \f$ work, where \f$ n \f$ equals
	 *         the size of the vector \a x. The constant factor depends on the
	 *         cost of evaluating the underlying binary operator. A good
	 *         implementation uses vectorised instructions whenever the input
	 *         domains, the output domain, and the operator used allow for this.
	 *
	 *      -# This call will not result in additional dynamic memory allocations.
	 *
	 *      -# This call takes \f$ \mathcal{O}(1) \f$ memory beyond the memory
	 *         used by the application at the point of a call to this function.
	 *
	 *      -# This call incurs at most
	 *         \f$ 2n \cdot \mathit{sizeof}(\mathit{IOType}) + \mathcal{O}(1) \f$
	 *         bytes of data movement.
	 * \endparblock
	 *
	 * @see grb::operators::internal::Operator for a discussion on when in-place
	 *      and/or vectorised operations are used.
	 *
	 * \todo Update and move functional specification to base, and revise
	 *       performance semantics.
	 */
	template<
		Descriptor descr = descriptors::no_operation,
		class Monoid,
		typename IOType, typename InputType,
		typename Coords
	>
	RC foldr(
		const InputType &alpha,
		Vector< IOType, reference, Coords > &y,
		const Monoid &monoid = Monoid(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if< !grb::is_object< InputType >::value &&
			!grb::is_object< IOType >::value &&
			grb::is_monoid< Monoid >::value, void
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

		// if no monoid was given, then we can only handle dense vectors
		auto null_coor = &( internal::getCoordinates( y ) );
		null_coor = nullptr;
		if( nnz( y ) < size( y ) ) {
			return internal::fold_from_scalar_to_vector_generic<
				descr, false, true, false, true, void
			>(
				y, nullptr, null_coor, alpha, monoid.getOperator(), phase
			);
		} else {
			return internal::fold_from_scalar_to_vector_generic<
				descr, false, false, false, true, void
			>(
				y, nullptr, null_coor, alpha, monoid.getOperator(), phase
			);
		}
	}

	/**
	 * Folds a vector into a scalar using an operator.
	 *
	 * \todo Update and move functional specification to base, and revise
	 *       performance semantics.
	 */
	template<
		Descriptor descr = descriptors::no_operation,
		class OP,
		typename IOType, typename InputType,
		typename Coords
	>
	RC foldr(
		const InputType &alpha,
		Vector< IOType, reference, Coords > &y,
		const OP &op = OP(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			!grb::is_object< InputType >::value &&
			!grb::is_object< IOType >::value &&
			grb::is_operator< OP >::value, void
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

		// if no monoid was given, then we can only handle dense vectors
		auto null_coor = &( internal::getCoordinates( y ) );
		null_coor = nullptr;
		if( nnz( y ) < size( y ) ) {
			return internal::fold_from_scalar_to_vector_generic<
				descr, false, true, false, false, void
			>( y, nullptr, null_coor, alpha, op, phase );
		} else {
			return internal::fold_from_scalar_to_vector_generic<
				descr, false, false, false, false, void
			>( y, nullptr, null_coor, alpha, op, phase );
		}
	}

	/**
	 * Folds all elements in a GraphBLAS vector \a x into the corresponding
	 * elements from an input/output vector \a y. The vectors must be of equal
	 * size \f$ n \f$. For all \f$ i \in \{0,1,\ldots,n-1\} \f$, the new value
	 * of at the i-th index of \a y after a call to this function thus equals
	 * \f$ x_i \odot y_i \f$.
	 *
	 * @tparam descr     The descriptor used for evaluating this function. By
	 *                   default, this is grb::descriptors::no_operation.
	 * @tparam OP        The type of the operator to be applied.
	 * @tparam IOType    The type of the elements of \a x.
	 * @tparam InputType The type of the elements of \a y.
	 *
	 * @param[in]     x  The input vector \a y that will not be modified.
	 * @param[in,out] y  On function entry: the initial value to be applied to
	 *                   \a op as the right-hand side input.
	 *                   On function exit: the result of repeated applications
	 *                   from the right-hand side using elements from \a y.
	 * @param[in]     op The operator under which to perform this right-folding.
	 *
	 * @returns grb::SUCCESS This function always succeeds.
	 *
	 * \note The element-wise fold is also defined for monoids.
	 *
	 * \parblock
	 * \par Valid descriptors
	 * grb::descriptors::no_operation, grb::descriptors::no_casting.
	 *
	 * \note Invalid descriptors will be ignored.
	 *
	 * If grb::descriptors::no_casting is specified, then 1) the first domain of
	 * \a op must match \a InputType, 2) the second domain of \a op must match
	 * \a IOType, and 3) the third domain must match \a IOType. If one of these
	 * is not true, the code shall not compile.
	 *
	 * \endparblock
	 *
	 * \parblock
	 * \par Valid operator types
	 * The given operator \a op is required to be:
	 *   -# (no requirements).
	 * \endparblock
	 *
	 * \parblock
	 * \par Performance semantics
	 *      -# This call comprises \f$ \Theta(n) \f$ work, where \f$ n \f$ equals
	 *         the size of the vector \a x. The constant factor depends on the
	 *         cost of evaluating the underlying binary operator. A good
	 *         implementation uses vectorised instructions whenever the input
	 *         domains, the output domain, and the operator used allow for this.
	 *
	 *      -# This call will not result in additional dynamic memory allocations.
	 *
	 *      -# This call takes \f$ \mathcal{O}(1) \f$ memory beyond the memory
	 *         used by the application at the point of a call to this function.
	 *
	 *      -# This call incurs at most
	 *         \f$ n \cdot (
	 *                       \mathit{sizeof}(InputType) + 2\mathit{sizeof}(IOType)
	 *                     ) + \mathcal{O}(1)
	 *         \f$
	 *         bytes of data movement. A good implementation will rely on in-place
	 *         operators whenever allowed.
	 * \endparblock
	 *
	 * @see grb::operators::internal::Operator for a discussion on when in-place
	 *      and/or vectorised operations are used.
	 *
	 * \todo Update and move functional specification to base, and revise
	 *       performance semantics.
	 */
	template<
		Descriptor descr = descriptors::no_operation,
		class OP,
		typename IOType, typename InputType,
		typename Coords
	>
	RC foldr(
		const Vector< InputType, reference, Coords > &x,
		Vector< IOType, reference, Coords > &y,
		const OP &op = OP(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			grb::is_operator< OP >::value &&
			!grb::is_object< InputType >::value &&
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

		if( size( x ) != size( y ) ) {
			return MISMATCH;
		}

		const size_t n = size( x );

		if( descr & descriptors::dense ) {
			if( nnz( x ) != n || nnz( y ) != n ) { return ILLEGAL; }
		}

#ifdef _DEBUG
		std::cout << "In foldr ([T]<-[T])\n";
#endif

		const Vector< bool, reference, Coords > * const null_mask = nullptr;
		if( nnz( x ) < size( x ) || nnz( y ) < size( y ) ) {
			return internal::fold_from_vector_to_vector_generic<
				descr, false, true, false, false
			>( y, null_mask, x, op, phase );
		} else {
			return internal::fold_from_vector_to_vector_generic<
				descr, false, false, false, false
			>( y, null_mask, x, op, phase );
		}
	}

	/**
	 * Fold-right from a vector into another vector.
	 *
	 * Masked operator variant.
	 *
	 * \todo Update and move functional specification to base, and revise
	 *       performance semantics.
	 */
	template<
		Descriptor descr = descriptors::no_operation,
		class OP,
		typename IOType, typename MaskType, typename InputType,
		typename Coords
	>
	RC foldr(
		const Vector< InputType, reference, Coords > &x,
		const Vector< MaskType, reference, Coords > &m,
		Vector< IOType, reference, Coords > &y,
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
		if( descr & descriptors::dense ) {
			if( size( m ) > 0 && nnz( m ) != n ) { return ILLEGAL; }
			if( nnz( x ) != n || nnz( y ) != n ) { return ILLEGAL; }
		}

		if( nnz( x ) < n || nnz( y ) < n ) {
			return internal::fold_from_vector_to_vector_generic<
				descr, false, true, true, false
			>( y, &m, x, op, phase );
		} else {
			return internal::fold_from_vector_to_vector_generic<
				descr, false, false, true, false
			>( y, &m, x, op, phase );
		}
	}

	/**
	 * Folds all elements in an ALP/ GraphBLAS vector \a x into the corresponding
	 * elements from an input/output vector \a y. The vectors must be of equal
	 * size \f$ n \f$. For all \f$ i \in \{0,1,\ldots,n-1\} \f$, the new value
	 * of at the i-th index of \a y after a call to this function thus equals
	 * \f$ x_i \odot y_i \f$.
	 *
	 * @tparam descr     The descriptor used for evaluating this function. By
	 *                   default, this is grb::descriptors::no_operation.
	 * @tparam Monoid    The type of the monoid to be applied.
	 * @tparam IOType    The type of the elements of \a x.
	 * @tparam InputType The type of the elements of \a y.
	 *
	 * @param[in]       x    The input vector \a y that will not be modified.
	 * @param[in,out]   y    On function entry: the initial value to be applied
	 *                       to \a op as the right-hand side input.
	 *                       On function exit: the result of repeated applications
	 *                       from the right-hand side using elements from \a y.
	 * @param[in]     monoid The monoid under which to perform this right-folding.
	 *
	 * @returns grb::SUCCESS This function always succeeds.
	 *
	 * \note The element-wise fold is also defined for operators.
	 *
	 * \parblock
	 * \par Valid descriptors
	 * grb::descriptors::no_operation, grb::descriptors::no_casting.
	 *
	 * \note Invalid descriptors will be ignored.
	 *
	 * If grb::descriptors::no_casting is specified, then 1) the first domain of
	 * \a op must match \a InputType, 2) the second domain of \a op must match
	 * \a IOType, and 3) the third domain must match \a IOType. If one of these
	 * is not true, the code shall not compile.
	 *
	 * \endparblock
	 *
	 * \parblock
	 * \par Valid monoid types
	 * The given operator \a op is required to be:
	 *   -# (no requirements).
	 * \endparblock
	 *
	 * \parblock
	 * \par Performance semantics
	 *      -# This call comprises \f$ \Theta(n) \f$ work, where \f$ n \f$ equals
	 *         the size of the vector \a x. The constant factor depends on the
	 *         cost of evaluating the underlying binary operator. A good
	 *         implementation uses vectorised instructions whenever the input
	 *         domains, the output domain, and the operator used allow for this.
	 *
	 *      -# This call will not result in additional dynamic memory allocations.
	 *
	 *      -# This call takes \f$ \mathcal{O}(1) \f$ memory beyond the memory
	 *         used by the application at the point of a call to this function.
	 *
	 *      -# This call incurs at most
	 *         \f$ n \cdot (
	 *                       \mathit{sizeof}(InputType) + 2\mathit{sizeof}(IOType)
	 *                     ) + \mathcal{O}(1)
	 *         \f$
	 *         bytes of data movement. A good implementation will rely on in-place
	 *         operators whenever allowed.
	 * \endparblock
	 *
	 * @see grb::operators::internal::Operator for a discussion on when in-place
	 *      and/or vectorised operations are used.
	 *
	 * \todo Update and move functional specification to base, and revise
	 *       performance semantics.
	 */
	template<
		Descriptor descr = descriptors::no_operation, class Monoid,
		typename IOType, typename InputType,
		typename Coords
	>
	RC foldr(
		const Vector< InputType, reference, Coords > &x,
		Vector< IOType, reference, Coords > &y,
		const Monoid &monoid = Monoid(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if< grb::is_monoid< Monoid >::value &&
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

		// dynamic sanity checks
		const size_t n = size( x );
		if( n != size( y ) ) {
			return MISMATCH;
		}
		if( descr & descriptors::dense ) {
			if( nnz( x ) != n || nnz( y ) != n ) { return ILLEGAL; }
		}

		const Vector< bool, reference, Coords > * const null_mask = nullptr;
		if( nnz( x ) < n || nnz( y ) < n ) {
			return internal::fold_from_vector_to_vector_generic<
				descr, false, true, false, true
			>( y, null_mask, x, monoid.getOperator(), phase );
		} else {
			return internal::fold_from_vector_to_vector_generic<
				descr, false, false, false, true
			>( y, null_mask, x, monoid.getOperator(), phase );
		}
	}

	/**
	 * Fold-rights a vector into another vector.
	 *
	 * Masked monoid variant.
	 *
	 * \todo Update and move functional specification to base, and revise
	 *       performance semantics.
	 */
	template<
		Descriptor descr = descriptors::no_operation, class Monoid,
		typename IOType, typename MaskType, typename InputType,
		typename Coords
	>
	RC foldr(
		const Vector< InputType, reference, Coords > &x,
		const Vector< MaskType, reference, Coords > &m,
		Vector< IOType, reference, Coords > &y,
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
		if( descr & descriptors::dense ) {
			if( size( m ) > 0 && nnz( m ) != n ) { return ILLEGAL; }
			if( nnz( x ) != n || nnz( y ) != n ) { return ILLEGAL; }
		}

		if( nnz( x ) < n || nnz( y ) < n ) {
			return internal::fold_from_vector_to_vector_generic<
				descr, false, true, true, true
			>( y, &m, x, monoid.getOperator(), phase );
		} else {
			return internal::fold_from_vector_to_vector_generic<
				descr, false, false, true, true
			>( y, &m, x, monoid.getOperator(), phase );
		}
	}

	/**
	 * For all elements in a GraphBLAS vector \a x, fold the value \f$ \beta \f$
	 * into each element.
	 *
	 * The original value of \f$ \beta \f$ is used as the right-hand side input
	 * of the operator \a op. The left-hand side inputs for \a op are retrieved
	 * from the input vector \a x. The result of the operation is stored in
	 * \f$ \beta \f$, thus overwriting its previous value. This process is
	 * repeated for every element in \a y.
	 *
	 * The value of \f$ x_i \f$ after a call to thus function thus equals
	 * \f$ x_i \odot \beta \f$, for all \f$ i \in \{ 0, 1, \dots, n - 1 \} \f$.
	 *
	 * @tparam descr     The descriptor used for evaluating this function. By
	 *                   default, this is grb::descriptors::no_operation.
	 * @tparam OP        The type of the operator to be applied.
	 * @tparam IOType    The type of the value \a beta.
	 * @tparam InputType The type of the elements of \a x.
	 *
	 * @param[in,out] x    On function entry: the initial values to be applied as
	 *                     the left-hand side input to \a op. The input vector must
	 *                     be dense.
	 *                     On function exit: the output data.
	 * @param[in]     beta The input value to apply as the right-hand side input
	 *                     to \a op.
	 * @param[in]     op   The operator under which to perform this left-folding.
	 *
	 * @returns grb::SUCCESS This function always succeeds.
	 *
	 * \note This function is also defined for monoids.
	 *
	 * \warning If \a x is sparse and this operation is requested, a monoid instead
	 *          of an operator is required!
	 *
	 * \parblock
	 * \par Valid descriptors
	 * grb::descriptors::no_operation, grb::descriptors::no_casting.
	 *
	 * \note Invalid descriptors will be ignored.
	 *
	 * If grb::descriptors::no_casting is specified, then 1) the first domain of
	 * \a op must match \a IOType, 2) the second domain of \a op must match
	 * \a InputType, and 3) the third domain must match \a IOType. If one of these
	 * is not true, the code shall not compile.
	 *
	 * \endparblock
	 *
	 * \parblock
	 * \par Valid operator types
	 * The given operator \a op is required to be:
	 *   -# (no requirement).
	 * \endparblock
	 *
	 * \parblock
	 * \par Performance semantics
	 *      -# This call comprises \f$ \Theta(n) \f$ work, where \f$ n \f$ equals
	 *         the size of the vector \a x. The constant factor depends on the
	 *         cost of evaluating the underlying binary operator. A good
	 *         implementation uses vectorised instructions whenever the input
	 *         domains, the output domain, and the operator used allow for this.
	 *
	 *      -# This call will not result in additional dynamic memory allocations.
	 *
	 *      -# This call takes \f$ \mathcal{O}(1) \f$ memory beyond the memory
	 *         used by the application at the point of a call to this function.
	 *
	 *      -# This call incurs at most
	 *         \f$ 2n \cdot \mathit{sizeof}(\mathit{IOType}) + \mathcal{O}(1) \f$
	 *         bytes of data movement.
	 * \endparblock
	 *
	 * @see grb::operators::internal::Operator for a discussion on when in-place
	 *      and/or vectorised operations are used.
	 *
	 * \todo Update and move functional specification to base, and revise
	 *       performance semantics.
	 */
	template<
		Descriptor descr = descriptors::no_operation, class Op,
		typename IOType, typename InputType,
		typename Coords
	>
	RC foldl(
		Vector< IOType, reference, Coords > &x,
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
			std::is_same< typename Op::D1, IOType >::value ), "grb::foldl",
			"called with a vector x of a type that does not match the first domain "
			"of the given operator" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< typename Op::D2, InputType >::value ), "grb::foldl",
			"called on a vector y of a type that does not match the second domain "
			"of the given operator" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< typename Op::D3, IOType >::value ), "grb::foldl",
			"called on a vector x of a type that does not match the third domain "
			"of the given operator" );

		// if no monoid was given, then we can only handle dense vectors
		auto null_coor = &( internal::getCoordinates( x ) );
		null_coor = nullptr;
		if( nnz( x ) < size( x ) ) {
			return internal::fold_from_scalar_to_vector_generic<
				descr, true, true, false, false, void
			>( x, nullptr, null_coor, beta, op, phase );
		} else {
			return internal::fold_from_scalar_to_vector_generic<
				descr, true, false, false, false, void
			>( x, nullptr, null_coor, beta, op, phase );
		}
}

	/**
	 * For all elements in an ALP/GraphBLAS vector \a x, fold the value
	 * \f$ \beta \f$ into each element.
	 *
	 * Masked operator variant.
	 *
	 * \todo Update and move functional specification to base, and revise
	 *       performance semantics.
	 */
	template<
		Descriptor descr = descriptors::no_operation, class Op,
		typename IOType, typename MaskType, typename InputType,
		typename Coords
	>
	RC foldl(
		Vector< IOType, reference, Coords > &x,
		const Vector< MaskType, reference, Coords > &m,
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
			std::is_same< typename Op::D1, IOType >::value ), "grb::foldl",
			"called with a vector x of a type that does not match the first domain "
			"of the given operator" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< typename Op::D2, InputType >::value ), "grb::foldl",
			"called on a vector y of a type that does not match the second domain "
			"of the given operator" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< typename Op::D3, IOType >::value ), "grb::foldl",
			"called on a vector x of a type that does not match the third domain "
			"of the given operator" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting ) ||
			std::is_same< bool, MaskType >::value ),
			"grb::foldl (reference, vector <- scalar, masked)",
			"provided mask does not have boolean entries" );
		if( size( m ) == 0 ) {
			return foldl< descr >( x, beta, op, phase );
		}
		const auto m_coor = &( internal::getCoordinates( m ) );
		const auto m_p = internal::getRaw( m );
		if( nnz( x ) < size( x ) ) {
			return internal::fold_from_scalar_to_vector_generic<
				descr, true, true, true, false
			>( x, m_p, m_coor, beta, op, phase );
		} else {
			return internal::fold_from_scalar_to_vector_generic<
				descr, true, false, true, false
			>( x, m_p, m_coor, beta, op, phase );
		}
	}

	/**
	 * For all elements in an ALP/GraphBLAS vector \a x, fold the value
	 * \f$ \beta \f$ into each element.
	 *
	 * The original value of \f$ \beta \f$ is used as the right-hand side input
	 * of the operator \a op. The left-hand side inputs for \a op are retrieved
	 * from the input vector \a x. The result of the operation is stored in
	 * \f$ \beta \f$, thus overwriting its previous value. This process is
	 * repeated for every element in \a y.
	 *
	 * The value of \f$ x_i \f$ after a call to thus function thus equals
	 * \f$ x_i \odot \beta \f$, for all \f$ i \in \{ 0, 1, \dots, n - 1 \} \f$.
	 *
	 * @tparam descr     The descriptor used for evaluating this function. By
	 *                   default, this is grb::descriptors::no_operation.
	 * @tparam Monoid    The type of the monoid to be applied.
	 * @tparam IOType    The type of the value \a beta.
	 * @tparam InputType The type of the elements of \a x.
	 *
	 * @param[in,out] x    On function entry: the initial values to be applied as
	 *                     the left-hand side input to \a op.
	 *                     On function exit: the output data.
	 * @param[in]     beta The input value to apply as the right-hand side input
	 *                     to \a op.
	 * @param[in]   monoid The monoid under which to perform this left-folding.
	 *
	 * @returns grb::SUCCESS This function always succeeds.
	 *
	 * \note This function is also defined for operators.
	 *
	 * \parblock
	 * \par Valid descriptors
	 * grb::descriptors::no_operation, grb::descriptors::no_casting.
	 *
	 * \note Invalid descriptors will be ignored.
	 *
	 * If grb::descriptors::no_casting is specified, then 1) the first domain of
	 * \a op must match \a IOType, 2) the second domain of \a op must match
	 * \a InputType, and 3) the third domain must match \a IOType. If one of these
	 * is not true, the code shall not compile.
	 *
	 * \endparblock
	 *
	 * \parblock
	 * \par Valid operator types
	 * The given operator \a op is required to be:
	 *   -# (no requirement).
	 * \endparblock
	 *
	 * \parblock
	 * \par Performance semantics
	 *      -# This call comprises \f$ \Theta(n) \f$ work, where \f$ n \f$ equals
	 *         the size of the vector \a x. The constant factor depends on the
	 *         cost of evaluating the underlying binary operator. A good
	 *         implementation uses vectorised instructions whenever the input
	 *         domains, the output domain, and the operator used allow for this.
	 *
	 *      -# This call will not result in additional dynamic memory allocations.
	 *
	 *      -# This call takes \f$ \mathcal{O}(1) \f$ memory beyond the memory
	 *         used by the application at the point of a call to this function.
	 *
	 *      -# This call incurs at most
	 *         \f$ 2n \cdot \mathit{sizeof}(\mathit{IOType}) + \mathcal{O}(1) \f$
	 *         bytes of data movement.
	 * \endparblock
	 *
	 * @see grb::operators::internal::Operator for a discussion on when in-place
	 *      and/or vectorised operations are used.
	 *
	 * \todo Update and move functional specification to base, and revise
	 *       performance semantics.
	 */
	template<
		Descriptor descr = descriptors::no_operation, class Monoid,
		typename IOType, typename InputType,
		typename Coords
	>
	RC foldl(
		Vector< IOType, reference, Coords > &x,
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

		// delegate to generic case
		auto null_coor = &( internal::getCoordinates( x ) );
		null_coor = nullptr;
		if( (descr & descriptors::dense) ||
			internal::getCoordinates( x ).isDense()
		) {
			return internal::fold_from_scalar_to_vector_generic<
				descr, true, false, false, true, void
			>( x, nullptr, null_coor, beta, monoid.getOperator(), phase );
		} else {
			return internal::fold_from_scalar_to_vector_generic<
				descr, true, true, false, true, void
			>( x, nullptr, null_coor, beta, monoid.getOperator(), phase );
		}
	}

	/**
	 * For all elements in an ALP/GraphBLAS vector \a x, fold the value
	 * \f$ \beta \f$ into each element.
	 *
	 * Masked monoid variant.
	 *
	 * \todo Update and move functional specification to base, and revise
	 *       performance semantics.
	 */
	template<
		Descriptor descr = descriptors::no_operation, class Monoid,
		typename IOType, typename MaskType, typename InputType,
		typename Coords
	>
	RC foldl(
		Vector< IOType, reference, Coords > &x,
		const Vector< MaskType, reference, Coords > &m,
		const InputType &beta,
		const Monoid &monoid = Monoid(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			!grb::is_object< IOType >::value &&
			!grb::is_object< MaskType >::value &&
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
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< bool, MaskType >::value ),
			"grb::foldl (reference, vector <- scalar, masked, monoid)",
			"provided mask does not have boolean entries" );
		if( size( m ) == 0 ) {
			return foldl< descr >( x, beta, monoid, phase );
		}

		// delegate to generic case
		auto m_coor = &( internal::getCoordinates( m ) );
		auto m_p = internal::getRaw( m );
		if( (descr & descriptors::dense) ||
			internal::getCoordinates( x ).isDense()
		) {
			return internal::fold_from_scalar_to_vector_generic<
				descr, true, false, true, true
			>( x, m_p, m_coor, beta, monoid.getOperator(), phase );
		} else {
			return internal::fold_from_scalar_to_vector_generic<
				descr, true, true, true, true
			>( x, m_p, m_coor, beta, monoid.getOperator(), phase );
		}
	}

	/**
	 * Folds all elements in a GraphBLAS vector \a y into the corresponding
	 * elements from an input/output vector \a x. The vectors must be of equal
	 * size \f$ n \f$. For all \f$ i \in \{0,1,\ldots,n-1\} \f$, the new value
	 * of at the i-th index of \a x after a call to this function thus equals
	 * \f$ x_i \odot y_i \f$.
	 *
	 * @tparam descr     The descriptor used for evaluating this function. By
	 *                   default, this is grb::descriptors::no_operation.
	 * @tparam OP        The type of the operator to be applied.
	 * @tparam IOType    The type of the value \a x.
	 * @tparam InputType The type of the elements of \a y.
	 *
	 * @param[in,out] x On function entry: the vector whose elements are to be
	 *                  applied to \a op as the left-hand side input.
	 *                  On function exit: the vector containing the result of
	 *                  the requested computation.
	 * @param[in]    y  The input vector \a y whose elements are to be applied
	 *                  to \a op as right-hand side input.
	 * @param[in]    op The operator under which to perform this left-folding.
	 *
	 * @returns grb::SUCCESS This function always succeeds.
	 *
	 * \note This function is also defined for monoids.
	 *
	 * \parblock
	 * \par Valid descriptors
	 * grb::descriptors::no_operation, grb::descriptors::no_casting.
	 *
	 * \note Invalid descriptors will be ignored.
	 *
	 * If grb::descriptors::no_casting is specified, then 1) the first domain of
	 * \a op must match \a IOType, 2) the second domain of \a op must match
	 * \a InputType, and 3) the third domain must match \a IOType. If one of these
	 * is not true, the code shall not compile.
	 *
	 * \endparblock
	 *
	 * \parblock
	 * \par Valid operator types
	 * The given operator \a op is required to be:
	 *   -# (no requirements).
	 * \endparblock
	 *
	 * \parblock
	 * \par Performance semantics
	 *      -# This call comprises \f$ \Theta(n) \f$ work, where \f$ n \f$ equals
	 *         the size of the vector \a x. The constant factor depends on the
	 *         cost of evaluating the underlying binary operator. A good
	 *         implementation uses vectorised instructions whenever the input
	 *         domains, the output domain, and the operator used allow for this.
	 *
	 *      -# This call will not result in additional dynamic memory allocations.
	 *
	 *      -# This call takes \f$ \mathcal{O}(1) \f$ memory beyond the memory
	 *         used by the application at the point of a call to this function.
	 *
	 *      -# This call incurs at most
	 *         \f$ n \cdot (
	 *                \mathit{sizeof}(\mathit{IOType}) +
	 *                \mathit{sizeof}(\mathit{InputType})
	 *             ) + \mathcal{O}(1) \f$
	 *         bytes of data movement. A good implementation will apply in-place
	 *         vectorised instructions whenever the input domains, the output
	 *         domain, and the operator used allow for this.
	 * \endparblock
	 *
	 * @see grb::operators::internal::Operator for a discussion on when in-place
	 *      and/or vectorised operations are used.
	 *
	 * \todo Update and move functional specification to base, and revise
	 *       performance semantics.
	 */
	template<
		Descriptor descr = descriptors::no_operation, class OP,
		typename IOType, typename InputType, typename Coords
	>
	RC foldl(
		Vector< IOType, reference, Coords > &x,
		const Vector< InputType, reference, Coords > &y,
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
		if( descr & descriptors::dense ) {
			if( nnz( x ) != n || nnz( y ) != n ) { return ILLEGAL; }
		}

		// all OK, execute
		const Vector< bool, reference, Coords > * const null_mask = nullptr;
		if( nnz( x ) < n || nnz( y ) < n ) {
			return internal::fold_from_vector_to_vector_generic<
				descr, true, true, false, false
			>( x, null_mask, y, op, phase );
		} else {
			assert( nnz( x ) == n );
			assert( nnz( y ) == n );
			return internal::fold_from_vector_to_vector_generic<
				descr, true, false, false, false
			>( x, null_mask, y, op, phase );
		}
	}

	/**
	 * Folds all elements in an ALP/GraphBLAS vector \a y into the corresponding
	 * elements from an input/output vector \a x. The vectors must be of equal
	 * size \f$ n \f$. For all \f$ i \in \{0,1,\ldots,n-1\} \f$, the new value
	 * of at the i-th index of \a x after a call to this function thus equals
	 * \f$ x_i \odot y_i \f$.
	 *
	 * @tparam descr     The descriptor used for evaluating this function. By
	 *                   default, this is grb::descriptors::no_operation.
	 * @tparam Monoid    The type of the monoid to be applied.
	 * @tparam IOType    The type of the value \a x.
	 * @tparam InputType The type of the elements of \a y.
	 *
	 * @param[in,out]  x    On function entry: the vector whose elements are to be
	 *                      applied to \a op as the left-hand side input.
	 *                      On function exit: the vector containing the result of
	 *                      the requested computation.
	 * @param[in]      y    The input vector \a y whose elements are to be applied
	 *                      to \a op as right-hand side input.
	 * @param[in]    monoid The operator under which to perform this left-folding.
	 *
	 * @returns grb::SUCCESS This function always succeeds.
	 *
	 * \note This function is also defined for operators.
	 *
	 * \parblock
	 * \par Valid descriptors
	 * grb::descriptors::no_operation, grb::descriptors::no_casting.
	 *
	 * \note Invalid descriptors will be ignored.
	 *
	 * If grb::descriptors::no_casting is specified, then 1) the first domain of
	 * \a op must match \a IOType, 2) the second domain of \a op must match
	 * \a InputType, and 3) the third domain must match \a IOType. If one of these
	 * is not true, the code shall not compile.
	 *
	 * \endparblock
	 *
	 * \parblock
	 * \par Valid operator types
	 * The given operator \a op is required to be:
	 *   -# (no requirements).
	 * \endparblock
	 *
	 * \parblock
	 * \par Performance semantics
	 *      -# This call comprises \f$ \Theta(n) \f$ work, where \f$ n \f$ equals
	 *         the size of the vector \a x. The constant factor depends on the
	 *         cost of evaluating the underlying binary operator. A good
	 *         implementation uses vectorised instructions whenever the input
	 *         domains, the output domain, and the operator used allow for this.
	 *
	 *      -# This call will not result in additional dynamic memory allocations.
	 *
	 *      -# This call takes \f$ \mathcal{O}(1) \f$ memory beyond the memory
	 *         used by the application at the point of a call to this function.
	 *
	 *      -# This call incurs at most
	 *         \f$ n \cdot (
	 *                \mathit{sizeof}(\mathit{IOType}) +
	 *                \mathit{sizeof}(\mathit{InputType})
	 *             ) + \mathcal{O}(1) \f$
	 *         bytes of data movement. A good implementation will apply in-place
	 *         vectorised instructions whenever the input domains, the output
	 *         domain, and the operator used allow for this.
	 * \endparblock
	 *
	 * @see grb::operators::internal::Operator for a discussion on when in-place
	 *      and/or vectorised operations are used.
	 *
	 * \todo Update and move functional specification to base, and revise
	 *       performance semantics.
	 */
	template<
		Descriptor descr = descriptors::no_operation, class Monoid,
		typename IOType, typename InputType,
		typename Coords
	>
	RC foldl(
		Vector< IOType, reference, Coords > &x,
		const Vector< InputType, reference, Coords > &y,
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
		if( descr & descriptors::dense ) {
			if( nnz( x ) != n || nnz( y ) != n ) { return ILLEGAL; }
		}

		// all OK, execute
		const Vector< bool, reference, Coords > * const null_mask = nullptr;
		if( nnz( x ) < n || nnz( y ) < n ) {
			return internal::fold_from_vector_to_vector_generic<
				descr, true, true, false, true
			>( x, null_mask, y, monoid.getOperator(), phase );
		} else {
			assert( nnz( x ) == n );
			assert( nnz( y ) == n );
			return internal::fold_from_vector_to_vector_generic<
				descr, true, false, false, true
			>( x, null_mask, y, monoid.getOperator(), phase );
		}
	}

	/**
	 * Fold-left a vector into another vector.
	 *
	 * Masked operator variant.
	 *
	 * \todo Update and move functional specification to base, and revise
	 *       performance semantics.
	 */
	template<
		Descriptor descr = descriptors::no_operation, class OP,
		typename IOType, typename MaskType, typename InputType,
		typename Coords
	>
	RC foldl(
		Vector< IOType, reference, Coords > &x,
		const Vector< MaskType, reference, Coords > &m,
		const Vector< InputType, reference, Coords > &y,
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
		if( descr & descriptors::dense ) {
			if( size( m ) > 0 && nnz( m ) != n ) { return ILLEGAL; }
			if( nnz( x ) != n || nnz( y ) != n ) { return ILLEGAL; }
		}

		// all OK, execute
		if( nnz( x ) < n || nnz( y ) < n ) {
			return internal::fold_from_vector_to_vector_generic<
				descr, true, true, true, false
			>( x, &m, y, op, phase );
		} else {
			assert( nnz( x ) == n );
			assert( nnz( y ) == n );
			return internal::fold_from_vector_to_vector_generic<
				descr, true, false, true, false
			>( x, &m, y, op, phase );
		}
	}

	/**
	 * Fold-lefts a vector into another vector.
	 *
	 * Masked monoid variant.
	 *
	 * \todo Update and move functional specification to base, and revise
	 *       performance semantics.
	 */
	template<
		Descriptor descr = descriptors::no_operation, class Monoid,
		typename IOType, typename MaskType, typename InputType,
		typename Coords
	>
	RC foldl(
		Vector< IOType, reference, Coords > &x,
		const Vector< MaskType, reference, Coords > &m,
		const Vector< InputType, reference, Coords > &y,
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
		if( descr & descriptors::dense ) {
			if( size( m ) > 0 && nnz( m ) != n ) { return ILLEGAL; }
			if( nnz( x ) != n || nnz( y ) != n ) { return ILLEGAL; }
		}

		// all OK, execute
		if( nnz( x ) < n || nnz( y ) < n ) {
			return internal::fold_from_vector_to_vector_generic<
				descr, true, true, true, true
			>( x, &m, y, monoid.getOperator(), phase );
		} else {
			assert( nnz( x ) == n );
			assert( nnz( y ) == n );
			return internal::fold_from_vector_to_vector_generic<
				descr, true, false, true, true
			>( x, &m, y, monoid.getOperator(), phase );
		}
	}

	namespace internal {

		/**
		 * \internal eWiseApply of guaranteed complexity Theta(n) that generates dense
		 *           outputs.
		 */
		template<
			bool left_scalar, bool right_scalar, bool left_sparse, bool right_sparse,
			Descriptor descr, class OP,
			typename OutputType, typename InputType1, typename InputType2
		>
		RC dense_apply_generic(
			OutputType * const z_p,
			const InputType1 * const x_p,
			const Coordinates< reference > * const x_coors,
			const InputType2 * const y_p,
			const Coordinates< reference > * const y_coors,
			const OP &op,
			const size_t n
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
			assert( !left_sparse || x_coors != nullptr );
			assert( !right_sparse || y_coors != nullptr );

			constexpr const size_t block_size = OP::blocksize;
			const size_t num_blocks = n / block_size;

#ifndef _H_GRB_REFERENCE_OMP_BLAS1
 #ifndef NDEBUG
			const bool has_coda = n % block_size > 0;
 #endif
			size_t i = 0;
			const size_t start = 0;
			const size_t end = num_blocks;
#else
			#pragma omp parallel
			{
				size_t start, end;
				config::OMP::localRange( start, end, 0, num_blocks,
					config::CACHE_LINE_SIZE::value() / block_size );
#endif

				// declare and initialise local buffers for SIMD
				OutputType z_b[ block_size ];
				InputType1 x_b[ block_size ];
				InputType2 y_b[ block_size ];
				bool x_m[ block_size ];
				bool y_m[ block_size ];
				for( size_t k = 0; k < block_size; ++k ) {
					if( left_scalar ) {
						x_b[ k ] = *x_p;
					}
					if( right_scalar ) {
						y_b[ k ] = *y_p;
					}
				}

				for( size_t block = start; block < end; ++block ) {
#ifdef _H_GRB_REFERENCE_OMP_BLAS1
					size_t i = block * block_size;
#endif
					size_t local_i = i;
					for( size_t k = 0; k < block_size; ++k ) {
						if( !left_scalar ) {
							x_b[ k ] = x_p[ local_i ];
						}
						if( !right_scalar ) {
							y_b[ k ] = y_p[ local_i ];
						}
						if( left_sparse ) {
							x_m[ k ] = x_coors->assigned( local_i );
						}
						if( right_sparse ) {
							y_m[ k ] = y_coors->assigned( local_i );
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
#ifndef _H_GRB_REFERENCE_OMP_BLAS1
 #ifndef NDEBUG
				if( has_coda ) {
					assert( i < n );
				} else {
					assert( i == n );
				}
 #endif
#else
				size_t i = end * block_size;
				#pragma omp single
#endif
				for( ; i < n; ++i ) {
					RC rc = SUCCESS;
					if( left_scalar && right_scalar ) {
						rc = apply( z_p[ i ], *x_p, *y_p, op );
					} else if( left_scalar && !right_scalar ) {
						if( right_sparse && !( y_coors->assigned( i ) ) ) {
							z_p[ i ] = *x_p;
						} else {
							rc = apply( z_p[ i ], *x_p, y_p[ i ], op );
						}
					} else if( !left_scalar && right_scalar ) {
						if( left_sparse && !( x_coors->assigned( i ) ) ) {
							z_p[ i ] = *y_p;
						} else {
							rc = apply( z_p[ i ], x_p[ i ], *y_p, op );
						}
					} else {
						assert( !left_scalar && !right_scalar );
						if( left_sparse && !(x_coors->assigned( i )) ) {
							z_p[ i ] = y_p[ i ];
						} else if( right_sparse && !(y_coors->assigned( i )) ) {
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
#ifdef _H_GRB_REFERENCE_OMP_BLAS1
			} // end pragma omp parallel
#endif
			return SUCCESS;
		}

		/**
		 * \internal Implements generic eWiseApply that loops over input vector(s) to
		 *           generate a (likely) sparse output.
		 */
		template<
			bool masked, bool monoid, bool x_scalar, bool y_scalar,
			Descriptor descr, class OP,
			typename OutputType, typename MaskType,
			typename InputType1, typename InputType2
		>
		RC sparse_apply_generic(
			OutputType * const z_p,
			Coordinates< reference > &z_coors,
			const MaskType * const mask_p,
			const Coordinates< reference > * const mask_coors,
			const InputType1 * x_p,
			const Coordinates< reference > * const x_coors,
			const InputType2 * y_p,
			const Coordinates< reference > * const y_coors,
			const OP &op,
			const size_t n
		) {
#ifdef NDEBUG
			(void) n;
#endif
#ifndef GRB_NO_NOOP_CHECKS
			static_assert( !internal::maybe_noop< OP >::value, "Warning: you may be "
				"generating an output vector with uninitialised values. Define "
				"the GRB_NO_NOOP_CHECKS macro to disable this check.\n" );
#endif
			// assertions
			assert( !masked || mask_coors != nullptr );
			assert( !masked || mask_coors->size() == n );
			assert( y_scalar || ( y_coors != nullptr ) );
			assert( x_scalar || ( x_coors != nullptr ) );
			assert( x_scalar || x_coors->nonzeroes() <= n );
			assert( y_scalar || y_coors->nonzeroes() <= n );

#ifdef _DEBUG
			std::cout << "\tinternal::sparse_apply_generic called\n";
#endif
			constexpr const size_t block_size = OP::blocksize;

			// swap so that we do the expensive pass over the container with the fewest
			// nonzeroes first
			assert( !x_scalar || !y_scalar );
			const bool swap = ( x_scalar ? n : x_coors->nonzeroes() ) >
				( y_scalar ? n : y_coors->nonzeroes() );
			const Coordinates< reference > &loop_coors = swap ? *y_coors : *x_coors;
			const Coordinates< reference > &chk_coors = swap ? *x_coors : *y_coors;

#ifdef _DEBUG
			std::cout << "\t\tfirst-phase loop of size " << loop_coors.size() << "\n";
			if( x_scalar || y_scalar ) {
				std::cout << "\t\tthere will be no second phase because one of the inputs "
					<< "is scalar\n";
			} else {
				std::cout << "\t\tsecond-phase loop of size " << chk_coors.size() << "\n";
			}
#endif

#ifdef _H_GRB_REFERENCE_OMP_BLAS1
			#pragma omp parallel
			{
				const size_t maxAsyncAssigns = z_coors.maxAsyncAssigns();
				auto update = z_coors.EMPTY_UPDATE();
				size_t asyncAssigns = 0;
				assert( maxAsyncAssigns >= block_size );
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
						x_b[ k ] = *x_p;
					}
				}
				if( y_scalar ) {
					for( size_t k = 0; k < block_size; ++k ) {
						y_b[ k ] = *y_p;
					}
				}

				// expensive pass #1
				size_t start = 0;
				size_t end = loop_coors.nonzeroes() / block_size;
#ifndef _H_GRB_REFERENCE_OMP_BLAS1
				size_t k = 0;
#else
				// use dynamic schedule as the timings of gathers and scatters may vary
				// significantly
				#pragma omp for schedule( dynamic, config::CACHE_LINE_SIZE::value() ) nowait
#endif
				for( size_t b = start; b < end; ++b ) {
#ifdef _H_GRB_REFERENCE_OMP_BLAS1
					size_t k = b * block_size;
#endif
					// perform gathers
					for( size_t i = 0; i < block_size; ++i ) {
						const size_t index = loop_coors.index( k++ );
						offsets[ i ] = index;
						assert( index < n );
						if( masked ) {
							mask[ i ] = mask_coors->template mask< descr >( index, mask_p );
						}
					}
					// perform gathers
					for( size_t i = 0; i < block_size; ++i ) {
						if( !masked || mask[ i ] ) {
							if( !x_scalar ) {
								x_b[ i ] = x_p[ offsets[ i ] ];
							}
							if( !x_scalar && !y_scalar ) {
								y_m[ i ] = chk_coors.assigned( offsets[ i ] );
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
#ifndef _H_GRB_REFERENCE_OMP_BLAS1
								(void) z_coors.assign( offsets[ i ] );
#else
								if( !z_coors.asyncAssign( offsets[ i ], update ) ) {
									(void) ++asyncAssigns;
#ifdef _DEBUG
									std::cout << "\t\t now made " << asyncAssigns << " calls to "
										<< "asyncAssign; " << "added index " << offsets[ i ] << "\n";
#endif
								}
#endif
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
#ifdef _H_GRB_REFERENCE_OMP_BLAS1
					if( asyncAssigns > maxAsyncAssigns - block_size ) {
#ifdef _DEBUG
						std::cout << "\t\t " << omp_get_thread_num() << ": "
							<< "clearing local update at block " << b << ". "
							<< "It locally holds " << asyncAssigns << " entries. "
							<< "Update is at " << ( (void *)update ) << "\n";
#endif
#ifndef NDEBUG
						const bool was_empty =
#else
						(void)
#endif
							z_coors.joinUpdate( update );
						assert( !was_empty );
						asyncAssigns = 0;
					}
#endif
				}
#ifdef _H_GRB_REFERENCE_OMP_BLAS1
				// coda should be handled by a single thread
				#pragma omp single nowait
				{
					size_t k = end * block_size;
#endif
					for( ; k < loop_coors.nonzeroes(); ++k ) {
						const size_t index = loop_coors.index( k );
						if( masked && mask_coors->template mask< descr >( index, mask_p ) ) {
							continue;
						}
						RC rc = SUCCESS;
#ifndef _H_GRB_REFERENCE_OMP_BLAS1
						(void) z_coors.assign( index );
#else
						if( !z_coors.asyncAssign( index, update ) ) {
							(void) ++asyncAssigns;
						}
						if( asyncAssigns == maxAsyncAssigns ) {
#ifndef NDEBUG
							const bool was_empty =
#else
							(void)
#endif
								z_coors.joinUpdate( update );
							assert( !was_empty );
							asyncAssigns = 0;
						}
#endif
						if( x_scalar || y_scalar || chk_coors.assigned( index ) ) {
							rc = apply( z_p[ index ], x_p[ index ], y_p[ index ], op );
						} else if( monoid ) {
							if( swap ) {
								z_p[ index ] = x_scalar ?
									static_cast< typename OP::D3 >( *x_p ) :
									static_cast< typename OP::D3 >( x_p[ index ] );
							} else {
								z_p[ index ] = y_scalar ?
									static_cast< typename OP::D3 >( *y_p ) :
									static_cast< typename OP::D3 >( y_p[ index ] );
							}
						}
						assert( rc == SUCCESS );
#ifdef NDEBUG
						(void) rc;
#endif
					}
#ifdef _H_GRB_REFERENCE_OMP_BLAS1
				} // end omp single block
#endif

				// cheaper pass #2, only required if we are using monoid semantics
				// AND if both inputs are vectors
				if( monoid && !x_scalar && !y_scalar ) {
					start = 0;
					end = chk_coors.nonzeroes() / block_size;
#ifndef _H_GRB_REFERENCE_OMP_BLAS1
					k = 0;
#else
					// use dynamic schedule as the timings of gathers and scatters may vary
					// significantly
					#pragma omp for schedule( dynamic, config::CACHE_LINE_SIZE::value() ) nowait
#endif
					for( size_t b = start; b < end; ++b ) {
#ifdef _H_GRB_REFERENCE_OMP_BLAS1
						size_t k = b * block_size;
#endif
						// streaming load
						for( size_t i = 0; i < block_size; i++ ) {
							offsets[ i ] = chk_coors.index( k++ );
							assert( offsets[ i ] < n );
						}
						// pure gather
						for( size_t i = 0; i < block_size; i++ ) {
							x_m[ i ] = loop_coors.assigned( offsets[ i ] );
						}
						// gather-like
						for( size_t i = 0; i < block_size; i++ ) {
							if( masked ) {
								mask[ i ] = utils::interpretMask< descr >(
									mask_coors->assigned( offsets[ i ] ), mask_p, offsets[ i ]
								);
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
#ifndef _H_GRB_REFERENCE_OMP_BLAS1
									(void) z_coors.assign( offsets[ i ] );
#else
									if( !z_coors.asyncAssign( offsets[ i ], update ) ) {
										(void) ++asyncAssigns;
									}
#endif
								}
							} else {
								if( x_m[ i ] ) {
#ifndef _H_GRB_REFERENCE_OMP_BLAS1
									(void) z_coors.assign( offsets[ i ] );
#else
									if( !z_coors.asyncAssign( offsets[ i ], update ) ) {
										(void) ++asyncAssigns;
									}
#endif
								}
							}
						}
						// scatter
						for( size_t i = 0; i < block_size; i++ ) {
							if( masked ) {
								if( mask[ i ] ) {
									GRB_UTIL_IGNORE_MAYBE_UNINITIALIZED // if masked && mask[ i ], then
									z_p[ offsets[ i ] ] = z_b[ i ];     // z_b[ i ] was set from x or y in
									GRB_UTIL_RESTORE_WARNINGS           // the above
								}
							} else {
								if( x_m[ i ] ) {
#ifdef _DEBUG
									std::cout << "\t\t writing out " << z_b[ i ] << " to index " << offsets[ i ] << "\n";
#endif
									GRB_UTIL_IGNORE_MAYBE_UNINITIALIZED // the only way the below could write
									                                    // an uninitialised value is if the
													    // static_assert at the top of this
									z_p[ offsets[ i ] ] = z_b[ i ];     // function had triggered. See also
									GRB_UTIL_RESTORE_WARNINGS           // internal issue #321.
								}
							}
						}
#ifdef _H_GRB_REFERENCE_OMP_BLAS1
						if( asyncAssigns > maxAsyncAssigns - block_size ) {
#ifdef _DEBUG
							std::cout << "\t\t " << omp_get_thread_num() << ": "
								<< "clearing local update (2) at block " << b << ". It locally holds "
								<< asyncAssigns << " entries. Update is at " << ( (void *)update )
								<< "\n";
#endif
#ifndef NDEBUG
							const bool was_empty =
#else
							(void)
#endif
								z_coors.joinUpdate( update );
							assert( !was_empty );
							asyncAssigns = 0;
						}
#endif
					}
#ifdef _H_GRB_REFERENCE_OMP_BLAS1
					#pragma omp single nowait
					{
						size_t k = end * block_size;
#endif
						for( ; k < chk_coors.nonzeroes(); ++k ) {
							const size_t index = chk_coors.index( k );
							assert( index < n );
							if( loop_coors.assigned( index ) ) {
								continue;
							}
							if( masked && mask_coors->template mask< descr >( index, mask_p ) ) {
								continue;
							}
#ifndef _H_GRB_REFERENCE_OMP_BLAS1
							(void) z_coors.assign( index );
#else
							if( !z_coors.asyncAssign( index, update ) ) {
								(void) ++asyncAssigns;
							}
							if( asyncAssigns == maxAsyncAssigns ) {
#ifndef NDEBUG
								const bool was_empty =
#else
								(void)
#endif
									z_coors.joinUpdate( update );
								assert( !was_empty );
							}
#endif
							z_p[ index ] = swap ? x_p[ index ] : y_p[ index ];
						}
#ifdef _H_GRB_REFERENCE_OMP_BLAS1
					} // end pragma omp single
#endif
				}
#ifdef _H_GRB_REFERENCE_OMP_BLAS1
 #ifdef _DEBUG
				std::cout << "\t\t " << omp_get_thread_num()
					<< ": final local update clearing. It locally holds 0 "
					<< "entries. Update is at " << ( (void *)update ) << "\n";
 #endif
				while( !z_coors.joinUpdate( update ) ) {}
			} // end pragma omp parallel
#endif
			return SUCCESS;
		}

		/**
		 * \internal Whenever this function is called, the z_coors is assumed to be
		 *           cleared.
		 */
		template<
			bool left_scalar, bool right_scalar, bool left_sparse, bool right_sparse,
			Descriptor descr, class OP,
			typename OutputType, typename MaskType,
			typename InputType1, typename InputType2
		>
		RC masked_apply_generic(
			OutputType * const z_p,
			Coordinates< reference > &z_coors,
			const MaskType * const mask_p,
			const Coordinates< reference > &mask_coors,
			const InputType1 * const x_p,
			const InputType2 * const y_p,
			const OP &op,
			const size_t n,
			const Coordinates< reference > * const left_coors = nullptr,
			const InputType1 * const left_identity = nullptr,
			const Coordinates< reference > * const right_coors = nullptr,
			const InputType2 * const right_identity = nullptr
		) {
#ifdef _DEBUG
			std::cout << "In masked_apply_generic< " << left_scalar << ", "
				<< right_scalar << ", " << left_sparse << ", " << right_sparse << ", "
				<< descr << " > with vector size " << n << "\n";
#endif
			// assertions
			static_assert( !(left_scalar && left_sparse),
				"left_scalar and left_sparse cannot both be set!"
			);
			static_assert( !(right_scalar && right_sparse),
				"right_scalar and right_sparse cannot both be set!"
			);
			assert( !left_sparse || left_coors != nullptr );
			assert( !left_sparse || left_identity != nullptr );
			assert( !right_sparse || right_coors != nullptr );
			assert( !right_sparse || right_identity != nullptr );
			assert( z_coors.nonzeroes() == 0 );

#ifdef _DEBUG
			std::cout << "\tinternal::masked_apply_generic called with nnz(mask)="
				<< mask_coors.nonzeroes() << " and descriptor " << descr << "\n";
			if( mask_coors.nonzeroes() > 0 ) {
				std::cout << "\t\tNonzero mask indices: " << mask_coors.index( 0 );
				assert( mask_coors.assigned( mask_coors.index( 0 ) ) );
				for( size_t k = 1; k < mask_coors.nonzeroes(); ++k ) {
					std::cout << ", " << mask_coors.index( k );
					assert( mask_coors.assigned( mask_coors.index( k ) ) );
				}
				std::cout << "\n";
			}

			size_t unset = 0;
			for( size_t i = 0; i < mask_coors.size(); ++i ) {
				if( !mask_coors.assigned( i ) ) {
					(void) ++unset;
				}
			}
			assert( unset == mask_coors.size() - mask_coors.nonzeroes() );
#endif
			// whether to use a Theta(n) or a Theta(nnz(mask)) loop
			const bool bigLoop = mask_coors.nonzeroes() == n ||
				(descr & descriptors::invert_mask);

			// get block size
			constexpr size_t size_t_block_size =
				config::SIMD_SIZE::value() / sizeof( size_t );
			constexpr size_t op_block_size = OP::blocksize;
			constexpr size_t min_block_size = op_block_size > size_t_block_size ?
				size_t_block_size :
				op_block_size;

			if( bigLoop ) {
#ifdef _DEBUG
				std::cerr << "\t in bigLoop variant\n";
#endif
#ifndef _H_GRB_REFERENCE_OMP_BLAS1
				size_t i = 0;
#else
				#pragma omp parallel
				{
#endif
					constexpr const size_t block_size = op_block_size;
					const size_t num_blocks = n / block_size;
					const size_t start = 0;
					const size_t end = num_blocks;

					// declare buffers that fit in a single SIMD register and initialise if needed
					bool mask_b[ block_size ];
					OutputType z_b[ block_size ];
					InputType1 x_b[ block_size ];
					InputType2 y_b[ block_size ];
					for( size_t k = 0; k < block_size; ++k ) {
						if( left_scalar ) {
							x_b[ k ] = *x_p;
						}
						if( right_scalar ) {
							y_b[ k ] = *y_p;
						}
					}

#ifdef _H_GRB_REFERENCE_OMP_BLAS1
					auto update = z_coors.EMPTY_UPDATE();
					size_t asyncAssigns = 0;
					const size_t maxAsyncAssigns = z_coors.maxAsyncAssigns();
					assert( maxAsyncAssigns >= block_size );
					// choose dynamic schedule since the mask otherwise likely leads to
					// significant imbalance
					#pragma omp for schedule( dynamic, config::CACHE_LINE_SIZE::value() / block_size ) nowait
#endif
					// vectorised code
					for( size_t b = start; b < end; ++b ) {
#ifdef _H_GRB_REFERENCE_OMP_BLAS1
						size_t i = start * block_size;
#endif
						for( size_t k = 0; k < block_size; ++k ) {
							const size_t index = i + k;
							assert( index < n );
							mask_b[ k ] = mask_coors.template mask< descr >( index, mask_p );
						}
						// check for no output
						if( left_sparse && right_sparse ) {
							for( size_t k = 0; k < block_size; ++k ) {
								const size_t index = i + k;
								assert( index < n );
								if( mask_b[ k ] ) {
									if( !left_coors->assigned( index ) &&
										!right_coors->assigned( index )
									) {
										mask_b[ k ] = false;
									}
								}
							}
						}
						for( size_t k = 0; k < block_size; ++k ) {
							const size_t index = i + k;
							assert( index < n );
							if( mask_b[ k ] ) {
								if( !left_scalar ) {
									if( left_sparse && !left_coors->assigned( index ) ) {
										x_b[ k ] = *left_identity;
									} else {
										x_b[ k ] = *( x_p + index );
									}
								}
								if( !right_scalar ) {
									if( right_sparse && !right_coors->assigned( i + k ) ) {
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
							assert( index < n );
							if( mask_b[ k ] ) {
#ifdef _H_GRB_REFERENCE_OMP_BLAS1
								if( !z_coors.asyncAssign( index, update ) ) {
									(void) ++asyncAssigns;
								}
#else
								(void) z_coors.assign( index );
#endif
								*( z_p + index ) = z_b[ k ];
							}
						}
#ifdef _H_GRB_REFERENCE_OMP_BLAS1
						if( asyncAssigns > maxAsyncAssigns - block_size ) {
							(void) z_coors.joinUpdate( update );
							asyncAssigns = 0;
						}
#endif
						i += block_size;
					}
#ifdef _H_GRB_REFERENCE_OMP_BLAS1
					#pragma omp single nowait
#endif
					// scalar coda
					for( size_t i = end * block_size; i < n; ++i ) {
						if( mask_coors.template mask< descr >( i, mask_p ) ) {
							if( left_sparse && right_sparse ) {
								if( !left_coors->assigned( i ) && !right_coors->assigned( i ) ) {
									continue;
								}
							}
#ifdef _H_GRB_REFERENCE_OMP_BLAS1
							if( !z_coors.asyncAssign( i, update ) ) {
								(void) ++asyncAssigns;
							}
							if( asyncAssigns == maxAsyncAssigns ) {
								(void) z_coors.joinUpdate( update );
								asyncAssigns = 0;
							}
#else
							(void) z_coors.assign( i );
#endif
							const InputType1 * const x_e = left_scalar ?
								x_p :
								(
									(!left_sparse || left_coors->assigned( i )) ?
									x_p + i :
									left_identity
								);
							const InputType2 * const y_e = right_scalar ? y_p :
								(
									(!right_sparse || right_coors->assigned( i )) ?
									y_p + i :
									right_identity
								);
							OutputType * const z_e = z_p + i;
							apply( *z_e, *x_e, *y_e, op );
						}
					}
#ifdef _H_GRB_REFERENCE_OMP_BLAS1
					while( !z_coors.joinUpdate( update ) ) {}
				} // end pragma omp parallel
#endif
			} else {
#ifdef _DEBUG
				std::cerr << "\t in smallLoop variant\n";
#endif
#ifdef _H_GRB_REFERENCE_OMP_BLAS1
				#pragma omp parallel
				{
#endif
					// declare buffers that fit in a single SIMD register and initialise if
					// needed
					constexpr const size_t block_size = size_t_block_size > 0 ?
						min_block_size :
						op_block_size;
					bool mask_b[ block_size ];
					OutputType z_b[ block_size ];
					InputType1 x_b[ block_size ];
					InputType2 y_b[ block_size ];
					size_t indices[ block_size ];
					for( size_t k = 0; k < block_size; ++k ) {
						if( left_scalar ) {
							x_b[ k ] = *x_p;
						}
						if( right_scalar ) {
							y_b[ k ] = *y_p;
						}
					}

					// loop over mask pattern
					const size_t mask_nnz = mask_coors.nonzeroes();
					const size_t num_blocks = mask_nnz / block_size;
					const size_t start = 0;
					const size_t end = num_blocks;
#ifndef _H_GRB_REFERENCE_OMP_BLAS1
					size_t k = 0;
#else
					auto update = z_coors.EMPTY_UPDATE();
					size_t asyncAssigns = 0;
					const size_t maxAsyncAssigns = z_coors.maxAsyncAssigns();
					assert( maxAsyncAssigns >= block_size );
					// choose dynamic schedule since the mask otherwise likely leads to
					// significant imbalance
					#pragma omp for schedule( dynamic, config::CACHE_LINE_SIZE::value() / block_size ) nowait
#endif
					// vectorised code
					for( size_t b = start; b < end; ++b ) {
#ifdef _H_GRB_REFERENCE_OMP_BLAS1
						size_t k = b * block_size;
 #ifdef _DEBUG
						std::cout << "\t\t processing range " << k << "--" << ( k + block_size ) << "\n";
 #endif
#endif
						for( size_t t = 0; t < block_size; ++t ) {
							indices[ t ] = mask_coors.index( k + t );
						}
						for( size_t t = 0; t < block_size; ++t ) {
							mask_b[ t ] = mask_coors.template mask< descr >( indices[ t ], mask_p );
						}
						for( size_t t = 0; t < block_size; ++t ) {
							if( mask_b[ t ] ) {
								if( !left_scalar ) {
									if( left_sparse && !left_coors->assigned( indices[ t ] ) ) {
										x_b[ t ] = *left_identity;
									} else {
										x_b[ t ] = *( x_p + indices[ t ] );
									}
								}
								if( !right_scalar ) {
									if( right_sparse && !right_coors->assigned( indices[ t ] ) ) {
										y_b[ t ] = *right_identity;
									} else {
										y_b[ t ] = *( y_p + indices[ t ] );
									}
								}
							}
						}
						// check for no output
						if( left_sparse && right_sparse ) {
							for( size_t t = 0; t < block_size; ++t ) {
								const size_t index = indices[ t ];
								assert( index < n );
								if( mask_b[ t ] ) {
									if( !left_coors->assigned( index ) &&
										!right_coors->assigned( index )
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
#ifndef _H_GRB_REFERENCE_OMP_BLAS1
								(void) z_coors.assign( indices[ t ] );
#else
								if( !z_coors.asyncAssign( indices[ t ], update ) ) {
									(void) ++asyncAssigns;
#ifdef _DEBUG
									std::cout << "\t\t now made " << asyncAssigns << " calls to asyncAssign; "
										<< "added index " << indices[ t ] << "\n";
#endif
								}
#endif
								GRB_UTIL_IGNORE_MAYBE_UNINITIALIZED  // z_b is computed from x_b and
								*( z_p + indices[ t ] ) = z_b[ t ];  // y_b, which are both initialised
								GRB_UTIL_RESTORE_WARNINGS            // if mask_b is true
							}
						}
#ifndef _H_GRB_REFERENCE_OMP_BLAS1
						k += block_size;
#else
						if( asyncAssigns > maxAsyncAssigns - block_size ) {
							(void) z_coors.joinUpdate( update );
							asyncAssigns = 0;
						}
#endif
					}
#ifdef _H_GRB_REFERENCE_OMP_BLAS1
					#pragma omp single nowait
#endif
					// scalar coda
					for( size_t k = end * block_size; k < mask_nnz; ++k ) {
						const size_t i = mask_coors.index( k );
						if( mask_coors.template mask< descr >( i, mask_p ) ) {
							if( left_sparse && right_sparse ) {
								if( !left_coors->assigned( i ) && !right_coors->assigned( i ) ) {
									continue;
								}
							}
#ifndef _H_GRB_REFERENCE_OMP_BLAS1
							(void) z_coors.assign( i );
#else
							if( !z_coors.asyncAssign( i, update ) ) {
								(void) ++asyncAssigns;
							}
							if( asyncAssigns == maxAsyncAssigns ) {
								(void) z_coors.joinUpdate( update );
								asyncAssigns = 0;
							}
#endif
							const InputType1 * const x_e = left_scalar ?
								x_p : (
									(!left_sparse || left_coors->assigned( i )) ?
										x_p + i :
										left_identity
								);
							const InputType2 * const y_e = right_scalar ?
								y_p : (
									(!right_sparse || right_coors->assigned( i )) ?
									y_p + i :
									right_identity
								);
							OutputType * const z_e = z_p + i;
							apply( *z_e, *x_e, *y_e, op );
						}
					}
#ifdef _H_GRB_REFERENCE_OMP_BLAS1
					while( !z_coors.joinUpdate( update ) ) {}
				} // end pragma omp parallel
#endif
			}
			return SUCCESS;
		}

	} // end namespace ``grb::internal''

	/**
	 * Calculates the element-wise operation on one scalar to elements of one
	 * vector, \f$ z = x .* \beta \f$, using the given operator. The input and
	 * output vectors must be of equal length.
	 *
	 * The vectors \a x or \a y may not be sparse.
	 *
	 * For all valid indices \a i of \a z, its element \f$ z_i \f$ after
	 * the call to this function completes equals \f$ x_i \odot \beta \f$.
	 *
	 * \warning Use of sparse vectors is only supported in full generality
	 *          when applied via a monoid or semiring; otherwise, there is
	 *          no concept for correctly interpreting any missing vector
	 *          elements during the requested computation.
	 * \note    When applying element-wise operators on sparse vectors
	 *          using semirings, there is a difference between interpreting missing
	 *          values as an annihilating identity or as a neutral identity--
	 *          intuitively, such identities are known as `zero' or `one',
	 *          respectively. As a consequence, there are three different variants
	 *          for element-wise operations whose names correspond to their
	 *          intuitive meanings w.r.t. those identities:
	 *            -# eWiseAdd (neutral),
	 *            -# eWiseMul (annihilating), and
	 *            -# eWiseApply using monoids (neutral).
	 *          An eWiseAdd with some semiring and an eWiseApply using its additive
	 *          monoid are totally equivalent.
	 *
	 * @tparam descr      The descriptor to be used. Equal to
	 *                    descriptors::no_operation if left unspecified.
	 * @tparam OP         The operator to use.
	 * @tparam InputType1 The value type of the left-hand vector.
	 * @tparam InputType2 The value type of the right-hand scalar.
	 * @tparam OutputType The value type of the ouput vector.
	 *
	 * @param[in]   x   The left-hand input vector.
	 * @param[in]  beta The right-hand input scalar.
	 * @param[out]  z   The pre-allocated output vector.
	 * @param[in]   op  The operator to use.
	 *
	 * @return grb::MISMATCH Whenever the dimensions of \a x and \a z do not
	 *                       match. All input data containers are left untouched
	 *                       if this exit code is returned; it will be as though
	 *                       this call was never made.
	 * @return grb::SUCCESS  On successful completion of this call.
	 *
	 * \parblock
	 * \par Performance semantics
	 *      -# This call comprises \f$ \Theta(n) \f$ work, where \f$ n \f$ equals
	 *         the size of the vectors \a x and \a z. The constant factor depends
	 *         on the cost of evaluating the operator. A good implementation uses
	 *         vectorised instructions whenever the input domains, the output
	 *         domain, and the operator used allow for this.
	 *
	 *      -# This call takes \f$ \mathcal{O}(1) \f$ memory beyond the memory
	 *         used by the application at the point of a call to this function.
	 *
	 *      -# This call incurs at most
	 *         \f$ n(
	 *               \mathit{sizeof}(\mathit{D1}) + \mathit{sizeof}(\mathit{D3})
	 *             ) +
	 *         \mathcal{O}(1) \f$
	 *         bytes of data movement. A good implementation will stream \a y
	 *         into \a z to apply the multiplication operator in-place, whenever
	 *         the input domains, the output domain, and the operator allow for
	 *         this.
	 * \endparblock
	 */
	template<
		Descriptor descr = descriptors::no_operation, class OP,
		typename OutputType, typename InputType1, typename InputType2,
		typename Coords
	>
	RC eWiseApply(
		Vector< OutputType, reference, Coords > &z,
		const Vector< InputType1, reference, Coords > &x,
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
#ifdef _DEBUG
		std::cout << "In eWiseApply ([T1]<-[T2]<-T3), operator variant\n";
#endif

		// sanity check
		auto &z_coors = internal::getCoordinates( z );
		const size_t n = z_coors.size();
		if( internal::getCoordinates( x ).size() != n ) {
			return MISMATCH;
		}
		if( descr & descriptors::dense ) {
			if( nnz( z ) < size( z ) ) { return ILLEGAL; }
			if( nnz( x ) < size( x ) ) { return ILLEGAL; }
		}

		if( phase == RESIZE ) {
			return SUCCESS;
		}
		assert( phase == EXECUTE );

		const Coords * const no_coordinates = nullptr;
		if( nnz( x ) == n ) {
			// call dense apply
			z_coors.assignAll();
			return internal::dense_apply_generic<
				false, true, false, false, descr | descriptors::dense
			>(
				internal::getRaw( z ), internal::getRaw( x ),
				no_coordinates, &beta, no_coordinates,
				op, n
			);
		} else {
			// since z and x may not perfectly overlap, and since the intersection is
			// unknown a priori, we must iterate over the nonzeroes of x
			z_coors.clear();
			const bool * const null_mask = nullptr;
			return internal::sparse_apply_generic<
				false, false, false, true, descr
			> (
				internal::getRaw( z ), internal::getCoordinates( z ),
				null_mask, no_coordinates,
				internal::getRaw( x ), &( internal::getCoordinates( x ) ),
				&beta, no_coordinates, op, n
			);
		}
	}

	/**
	 * Computes \f$ z = x \odot y \f$, out of place.
	 *
	 * Specialisation for \a x and \a y scalar, operator version.
	 */
	template<
		Descriptor descr = descriptors::no_operation, class OP,
		typename OutputType, typename InputType1, typename InputType2,
		typename Coords
	>
	RC eWiseApply(
		Vector< OutputType, reference, Coords > &z,
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

	/**
	 * Computes \f$ z = x \odot y \f$, out of place.
	 *
	 * Specialisation for \a x and \a y scalar, monoid version.
	 */
	template<
		Descriptor descr = descriptors::no_operation, class Monoid,
		typename OutputType, typename InputType1, typename InputType2,
		typename Coords
	>
	RC eWiseApply(
		Vector< OutputType, reference, Coords > &z,
		const InputType1 alpha,
		const InputType2 beta,
		const Monoid &monoid = Monoid(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			!grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			grb::is_monoid< Monoid >::value, void
		>::type * const = nullptr
	) {
#ifdef _DEBUG
		std::cout << "In eWiseApply ([T1]<-T2<-T3), monoid variant\n";
#endif
		// simply delegate to operator variant
		return eWiseApply< descr >( z, alpha, beta, monoid.getOperator(), phase );
	}

	/**
	 * Computes \f$ z = x \odot y \f$, out of place.
	 *
	 * Specialisation for scalar \a y, masked operator version.
	 */
	template<
		Descriptor descr = descriptors::no_operation, class OP,
		typename OutputType, typename MaskType,
		typename InputType1, typename InputType2,
		typename Coords
	>
	RC eWiseApply(
		Vector< OutputType, reference, Coords > &z,
		const Vector< MaskType, reference, Coords > &mask,
		const Vector< InputType1, reference, Coords > &x,
		const InputType2 beta,
		const OP &op = OP(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if< !grb::is_object< OutputType >::value &&
			!grb::is_object< MaskType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			grb::is_operator< OP >::value, void
		>::type * const = nullptr
	) {
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
		if( descr & descriptors::dense ) {
			if( nnz( z ) < size( z ) ) { return ILLEGAL; }
			if( nnz( x ) < size( x ) ) { return ILLEGAL; }
			if( nnz( mask ) < size( mask ) ) { return ILLEGAL; }
		}

		if( phase == RESIZE ) {
			return SUCCESS;
		}
		assert( phase == EXECUTE );

		// check if can delegate to unmasked
		const auto &mask_coors = internal::getCoordinates( mask );
		if( (descr & descriptors::structural) &&
			!(descr & descriptors::invert_mask) &&
			mask_coors.nonzeroes() == n
		) {
			return eWiseApply< descr >( z, x, beta, op, phase );
		}

		auto &z_coors = internal::getCoordinates( z );
		OutputType * const z_p = internal::getRaw( z );
		const MaskType * const mask_p = internal::getRaw( mask );
		const InputType1 * const x_p = internal::getRaw( x );
		const auto &x_coors = internal::getCoordinates( x );

		// the output sparsity structure is implied by mask and descr
		z_coors.clear();

		if( (descr & descriptors::dense) ||
			(x_coors.nonzeroes() == n) ||
			(mask_coors.nonzeroes() <= x_coors.nonzeroes())
		) {
			return internal::masked_apply_generic<
				false, true, false, false, descr
			>( z_p, z_coors, mask_p, mask_coors, x_p, &beta, op, n );
		} else {
			const Coords * const null_coors = nullptr;
			return internal::sparse_apply_generic<
				true, false, false, true, descr
			>(
				z_p, z_coors,
				mask_p, &mask_coors,
				x_p, &x_coors,
				&beta, null_coors,
				op, n
			);
		}
	}

	/**
	 * Computes \f$ z = x \odot y \f$, out of place.
	 *
	 * Monoid version.
	 */
	template<
		Descriptor descr = descriptors::no_operation, class Monoid,
		typename OutputType, typename InputType1, typename InputType2,
		typename Coords
	>
	RC eWiseApply(
		Vector< OutputType, reference, Coords > &z,
		const Vector< InputType1, reference, Coords > &x,
		const Vector< InputType2, reference, Coords > &y,
		const Monoid &monoid = Monoid(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if< !grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			grb::is_monoid< Monoid >::value, void
		>::type * const = nullptr
	) {
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
		if( descr & descriptors::dense ) {
			if( nnz( z ) < size( z ) ) { return ILLEGAL; }
			if( nnz( x ) < size( x ) ) { return ILLEGAL; }
			if( nnz( y ) < size( y ) ) { return ILLEGAL; }
		}

		if( phase == RESIZE ) {
			return SUCCESS;
		}
		assert( phase == EXECUTE );

		// check if we can dispatch to dense variant
		if( (descr & descriptors::dense) ||
			(grb::nnz( x ) == n && grb::nnz( y ) == n)
		) {
			return eWiseApply< descr >( z, x, y, monoid.getOperator() );
		}

		// we are in the unmasked sparse variant
		auto &z_coors = internal::getCoordinates( z );
		OutputType * const z_p = internal::getRaw( z );
		const InputType1 * const x_p = internal::getRaw( x );
		const InputType2 * const y_p = internal::getRaw( y );
		const auto &x_coors = internal::getCoordinates( x );
		const auto &y_coors = internal::getCoordinates( y );
		const auto op = monoid.getOperator();

		// z will have an a-priori unknown sparsity structure
		z_coors.clear();

		const bool * const null_mask = nullptr;
		const Coords * const null_coors = nullptr;
		return internal::sparse_apply_generic< false, true, false, false, descr >(
			z_p, z_coors,
			null_mask, null_coors,
			x_p, &( x_coors ),
			y_p, &( y_coors ),
			op, n
		);
	}

	/**
	 * Computes \f$ z = x \odot y \f$, out of place.
	 *
	 * Specialisation for scalar \a x. Monoid version.
	 */
	template<
		Descriptor descr = descriptors::no_operation, class Monoid,
		typename OutputType, typename InputType1, typename InputType2,
		typename Coords
	>
	RC eWiseApply(
		Vector< OutputType, reference, Coords > &z,
		const InputType1 alpha,
		const Vector< InputType2, reference, Coords > &y,
		const Monoid &monoid = Monoid(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if< !grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			grb::is_monoid< Monoid >::value, void
		>::type * const = nullptr
	) {
#ifdef _DEBUG
		std::cout << "In unmasked eWiseApply ([T1]<-T2<-[T3], using monoid)\n";
#endif

		// other run-time checks
		const size_t n = internal::getCoordinates( z ).size();
		if( internal::getCoordinates( y ).size() != n ) {
			return MISMATCH;
		}
		if( descr & descriptors::dense ) {
			if( nnz( y ) < size( y ) ) { return ILLEGAL; }
			if( nnz( z ) < size( z ) ) { return ILLEGAL; }
		}

		if( phase == RESIZE ) {
			return SUCCESS;
		}
		assert( phase == EXECUTE );

		// check if we can dispatch to dense variant
		if( (descr & descriptors::dense) || grb::nnz( y ) == n ) {
			return eWiseApply< descr >( z, alpha, y, monoid.getOperator() );
		}

		// we are in the unmasked sparse variant
		auto &z_coors = internal::getCoordinates( z );
		OutputType * const z_p = internal::getRaw( z );
		const InputType2 * const y_p = internal::getRaw( y );
		const auto &y_coors = internal::getCoordinates( y );
		const auto &op = monoid.getOperator();

		// the result will always be dense
		if( z_coors.nonzeroes() < n ) {
			z_coors.assignAll();
		}

		// dispatch to generic function
		return internal::dense_apply_generic< true, false, false, true, descr >(
			z_p, &alpha, nullptr, y_p, &y_coors, op, n
		);
	}

	/**
	 * Computes \f$ z = x \odot y \f$, out of place.
	 *
	 * Specialisation for scalar \a y. Monoid version.
	 */
	template<
		Descriptor descr = descriptors::no_operation,
		class Monoid,
		typename OutputType, typename InputType1, typename InputType2,
		typename Coords
	>
	RC eWiseApply(
		Vector< OutputType, reference, Coords > &z,
		const Vector< InputType1, reference, Coords > &x,
		const InputType2 beta,
		const Monoid &monoid = Monoid(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if< !grb::is_object< OutputType >::value &&
				!grb::is_object< InputType1 >::value &&
				!grb::is_object< InputType2 >::value &&
				grb::is_monoid< Monoid >::value,
			void >::type * const = nullptr
	) {
#ifdef _DEBUG
		std::cout << "In unmasked eWiseApply ([T1]<-[T2]<-T3, using monoid)\n";
#endif
		// other run-time checks
		const size_t n = internal::getCoordinates( z ).size();
		if( internal::getCoordinates( x ).size() != n ) {
			return MISMATCH;
		}
		if( (descr & descriptors::dense) ) {
			if( nnz( z ) < size( z ) ) { return ILLEGAL; }
			if( nnz( x ) < size( x ) ) { return ILLEGAL; }
		}

		if( phase == RESIZE ) {
			return SUCCESS;
		}
		assert( phase == EXECUTE );

		// check if we can dispatch to dense variant
		if( (descr & descriptors::dense) || grb::nnz( x ) == n ) {
			return eWiseApply< descr >( z, x, beta, monoid.getOperator() );
		}

		// we are in the unmasked sparse variant
		auto &z_coors = internal::getCoordinates( z );
		OutputType * const z_p = internal::getRaw( z );
		const InputType1 * const x_p = internal::getRaw( x );
		const auto &x_coors = internal::getCoordinates( x );
		const auto &op = monoid.getOperator();

		// the result will always be dense
		if( z_coors.nonzeroes() < n ) {
			z_coors.assignAll();
		}

		// dispatch
		return internal::dense_apply_generic< false, true, true, false, descr >(
			z_p, x_p, &x_coors, &beta, nullptr, op, n
		);
	}

	/**
	 * Computes \f$ z = x \odot y \f$, out of place.
	 *
	 * Masked monoid version.
	 */
	template<
		Descriptor descr = descriptors::no_operation, class Monoid,
		typename OutputType, typename MaskType,
		typename InputType1, typename InputType2,
		typename Coords
	>
	RC eWiseApply(
		Vector< OutputType, reference, Coords > &z,
		const Vector< MaskType, reference, Coords > &mask,
		const Vector< InputType1, reference, Coords > &x,
		const Vector< InputType2, reference, Coords > &y,
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
		if( (descr & descriptors::dense) ) {
			if( nnz( z ) < size( z ) ) { return ILLEGAL; }
			if( nnz( x ) < size( x ) ) { return ILLEGAL; }
			if( nnz( y ) < size( y ) ) { return ILLEGAL; }
			if( nnz( mask ) < size( mask ) ) { return ILLEGAL; }
		}

		if( phase == RESIZE ) {
			return SUCCESS;
		}
		assert( phase == EXECUTE );

		// check if we can dispatch to dense variant
		if( (descr & descriptors::dense) ||
			(grb::nnz( x ) == n && grb::nnz( y ) == n)
		) {
			return eWiseApply< descr >( z, mask, x, y, monoid.getOperator() );
		}

		// we are in the masked sparse variant
		auto &z_coors = internal::getCoordinates( z );
		const auto &mask_coors = internal::getCoordinates( mask );
		OutputType * const z_p = internal::getRaw( z );
		const MaskType * const mask_p = internal::getRaw( mask );
		const InputType1 * const x_p = internal::getRaw( x );
		const InputType2 * const y_p = internal::getRaw( y );
		const auto &x_coors = internal::getCoordinates( x );
		const auto &y_coors = internal::getCoordinates( y );
		const InputType1 left_identity = monoid.template getIdentity< InputType1 >();
		const InputType2 right_identity = monoid.template getIdentity< InputType2 >();
		const auto &op = monoid.getOperator();

		// z will have an a priori unknown sparsity structure
		z_coors.clear();

		if( grb::nnz( x ) < n &&
			grb::nnz( y ) < n &&
			grb::nnz( x ) + grb::nnz( y ) < grb::nnz( mask )
		) {
			return internal::sparse_apply_generic< true, true, false, false, descr >(
				z_p, z_coors, mask_p, &mask_coors,
				x_p, &( x_coors ), y_p, &( y_coors ),
				op, n
			);
		} else if( grb::nnz( x ) < n && grb::nnz( y ) == n ) {
			return internal::masked_apply_generic<
				false, false, true, false, descr, typename Monoid::Operator,
				OutputType, MaskType, InputType1, InputType2
			>(
				z_p, z_coors, mask_p, mask_coors,
				x_p, y_p, op, n, &x_coors,
				&left_identity
			);
		} else if( grb::nnz( y ) < n && grb::nnz( x ) == n ) {
			return internal::masked_apply_generic<
				false, false, false, true, descr, typename Monoid::Operator,
				OutputType, MaskType, InputType1, InputType2
			>(
				z_p, z_coors, mask_p, mask_coors,
				x_p, y_p, op, n, nullptr, nullptr, &y_coors,
				&right_identity
			);
		} else {
			return internal::masked_apply_generic<
				false, false, true, true, descr, typename Monoid::Operator,
				OutputType, MaskType, InputType1, InputType2
			>(
				z_p, z_coors, mask_p, mask_coors,
				x_p, y_p, op, n, &x_coors, &left_identity,
				&y_coors, &right_identity
			);
		}
	}

	/**
	 * Computes \f$ z = x \odot y \f$, out of place.
	 *
	 * Specialisation for scalar \a x. Masked monoid version.
	 */
	template<
		Descriptor descr = descriptors::no_operation, class Monoid,
		typename OutputType, typename MaskType,
		typename InputType1, typename InputType2,
		typename Coords
	>
	RC eWiseApply(
		Vector< OutputType, reference, Coords > &z,
		const Vector< MaskType, reference, Coords > &mask,
		const InputType1 alpha,
		const Vector< InputType2, reference, Coords > &y,
		const Monoid &monoid = Monoid(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if< !grb::is_object< OutputType >::value &&
			!grb::is_object< MaskType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			grb::is_monoid< Monoid >::value,
		void >::type * const = nullptr
	) {
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
		if( descr & descriptors::dense ) {
			if( nnz( z ) < size( z ) ) { return ILLEGAL; }
			if( nnz( y ) < size( y ) ) { return ILLEGAL; }
			if( nnz( mask ) < size( mask ) ) { return ILLEGAL; }
		}

		if( phase == RESIZE ) {
			return SUCCESS;
		}
		assert( phase == EXECUTE );

		// check if we can dispatch to dense variant
		if( (descr & descriptors::dense) || grb::nnz( y ) == n ) {
			return eWiseApply< descr >( z, mask, alpha, y, monoid.getOperator() );
		}

		// we are in the masked sparse variant
		auto &z_coors = internal::getCoordinates( z );
		const auto &mask_coors = internal::getCoordinates( mask );
		OutputType * const z_p = internal::getRaw( z );
		const MaskType * const mask_p = internal::getRaw( mask );
		const InputType2 * const y_p = internal::getRaw( y );
		const auto &y_coors = internal::getCoordinates( y );
		const InputType2 right_identity = monoid.template getIdentity< InputType2 >();
		const auto &op = monoid.getOperator();

		// the sparsity structure of z will be a result of the given mask and descr
		z_coors.clear();

		return internal::masked_apply_generic<
			true, false, false, true, descr, typename Monoid::Operator,
			OutputType, MaskType, InputType1, InputType2
		>(
			z_p, z_coors, mask_p, mask_coors, &alpha, y_p,
			op, n,
			nullptr, nullptr, &y_coors, &right_identity
		);
	}

	/**
	 * Computes \f$ z = x \odot y \f$, out of place.
	 *
	 * Specialisation for scalar \a y. Masked monoid version.
	 */
	template<
		Descriptor descr = descriptors::no_operation, class Monoid,
		typename OutputType, typename MaskType,
		typename InputType1, typename InputType2,
		typename Coords
	>
	RC eWiseApply(
		Vector< OutputType, reference, Coords > &z,
		const Vector< MaskType, reference, Coords > &mask,
		const Vector< InputType1, reference, Coords > &x,
		const InputType2 beta,
		const Monoid &monoid = Monoid(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if< !grb::is_object< OutputType >::value &&
			!grb::is_object< MaskType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			grb::is_monoid< Monoid >::value, void
		>::type * const = nullptr
	) {
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
		if( descr & descriptors::dense ) {
			if( nnz( z ) < size( z ) ) { return ILLEGAL; }
			if( nnz( x ) < size( x ) ) { return ILLEGAL; }
			if( nnz( mask ) < size( mask ) ) { return ILLEGAL; }
		}

		if( phase == RESIZE ) {
			return SUCCESS;
		}
		assert( phase == EXECUTE );

		// check if we can dispatch to dense variant
		if( (descr & descriptors::dense) || grb::nnz( x ) == n ) {
			return eWiseApply< descr >( z, mask, x, beta, monoid.getOperator() );
		}

		// we are in the masked sparse variant
		auto &z_coors = internal::getCoordinates( z );
		const auto &mask_coors = internal::getCoordinates( mask );
		OutputType * const z_p = internal::getRaw( z );
		const MaskType * const mask_p = internal::getRaw( mask );
		const InputType1 * const x_p = internal::getRaw( x );
		const auto &x_coors = internal::getCoordinates( x );
		const InputType1 left_identity = monoid.template getIdentity< InputType1 >();
		const auto &op = monoid.getOperator();

		// the sparsity structure of z will be the result of the given mask and descr
		z_coors.clear();

		return internal::masked_apply_generic< false, true, true, false, descr >(
			z_p, z_coors, mask_p, mask_coors, x_p, &beta,
			op, n, &x_coors, &left_identity
		);
	}

	/**
	 * Calculates the element-wise operation on one scalar to elements of one
	 * vector, \f$ z = \alpha .* y \f$, using the given operator. The input and
	 * output vectors must be of equal length.
	 *
	 * The vectors \a x or \a y may not be sparse.
	 *
	 * For all valid indices \a i of \a z, its element \f$ z_i \f$ after
	 * the call to this function completes equals \f$ \alpha \odot y_i \f$.
	 *
	 * \warning Use of sparse vectors is only supported in full generality
	 *          when applied via a monoid or semiring; otherwise, there is
	 *          no concept for correctly interpreting any missing vector
	 *          elements during the requested computation.
	 * \note    When applying element-wise operators on sparse vectors
	 *          using semirings, there is a difference between interpreting missing
	 *          values as an annihilating identity or as a neutral identity--
	 *          intuitively, identities are known as `zero' or `one',
	 *          respectively. As a consequence, there are three different variants
	 *          for element-wise operations whose names correspond to their
	 *          intuitive meanings w.r.t. those identities:
	 *            -# eWiseAdd,
	 *            -# eWiseMul, and
	 *            -# eWiseMulAdd.
	 *
	 * @tparam descr The descriptor to be used. Equal to descriptors::no_operation
	 *               if left unspecified.
	 * @tparam OP    The operator to use.
	 * @tparam InputType1 The value type of the left-hand scalar.
	 * @tparam InputType2 The value type of the right-hand side vector.
	 * @tparam OutputType The value type of the ouput vector.
	 *
	 * @param[in]  alpha The left-hand scalar.
	 * @param[in]   y    The right-hand input vector.
	 * @param[out]  z    The pre-allocated output vector.
	 * @param[in]   op   The operator to use.
	 *
	 * @return grb::MISMATCH Whenever the dimensions of \a y and \a z do not
	 *                       match. All input data containers are left untouched
	 *                       if this exit code is returned; it will be as though
	 *                       this call was never made.
	 * @return grb::SUCCESS  On successful completion of this call.
	 *
	 * \parblock
	 * \par Performance semantics
	 *      -# This call comprises \f$ \Theta(n) \f$ work, where \f$ n \f$ equals
	 *         the size of the vectors \a y and \a z. The constant factor depends
	 *         on the cost of evaluating the operator. A good implementation uses
	 *         vectorised instructions whenever the input domains, the output
	 *         domain, and the operator used allow for this.
	 *
	 *      -# This call takes \f$ \mathcal{O}(1) \f$ memory beyond the memory
	 *         used by the application at the point of a call to this function.
	 *
	 *      -# This call incurs at most
	 *         \f$ n(
	 *               \mathit{sizeof}(\mathit{D2}) + \mathit{sizeof}(\mathit{D3})
	 *             ) +
	 *         \mathcal{O}(1) \f$
	 *         bytes of data movement. A good implementation will stream \a y
	 *         into \a z to apply the multiplication operator in-place, whenever
	 *         the input domains, the output domain, and the operator allow for
	 *         this.
	 * \endparblock
	 */
	template<
		Descriptor descr = descriptors::no_operation, class OP,
		typename OutputType, typename InputType1, typename InputType2,
		typename Coords
	>
	RC eWiseApply(
		Vector< OutputType, reference, Coords > &z,
		const InputType1 alpha,
		const Vector< InputType2, reference, Coords > &y,
		const OP &op = OP(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			!grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			grb::is_operator< OP >::value, void
		>::type * const = nullptr
	) {
#ifdef _DEBUG
		std::cout << "In eWiseApply ([T1]<-T2<-[T3]), operator variant\n";
#endif
		// dynamic sanity checks
		const size_t n = internal::getCoordinates( z ).size();
		if( internal::getCoordinates( y ).size() != n ) {
			return MISMATCH;
		}
		if( descr & descriptors::dense ) {
			if( nnz( z ) < size( z ) ) { return ILLEGAL; }
			if( nnz( y ) < size( y ) ) { return ILLEGAL; }
		}

		// check for trivial op
		if( n == 0 ) {
			return SUCCESS;
		}

		// check if we can dispatch
		if( getID( z ) == getID( y ) ) {
			return foldr< descr >( alpha, z, op );
		}

		if( phase == RESIZE ) {
			return SUCCESS;
		}
		assert( phase == EXECUTE );

		// check for dense variant
		if( (descr & descriptors::dense) ||
			internal::getCoordinates( y ).nonzeroes() == n
		) {
			internal::getCoordinates( z ).assignAll();
			const internal::Coordinates< reference > * const no_coordinates = nullptr;
			return internal::dense_apply_generic< true, false, false, false, descr >(
				 internal::getRaw( z ), &alpha, no_coordinates,
				 internal::getRaw( y ), no_coordinates, op, n
			);
		}

		// we are in the sparse variant
		internal::getCoordinates( z ).clear();
		const bool * const null_mask = nullptr;
		const Coords * const null_coors = nullptr;
		return internal::sparse_apply_generic< false, false, true, false, descr >(
			internal::getRaw( z ), internal::getCoordinates( z ), null_mask, null_coors,
			&alpha, null_coors,
			internal::getRaw( y ), &( internal::getCoordinates( y ) ),
			op, n
		);
	}

	/**
	 * Computes \f$ z = x \odot y \f$, out of place.
	 *
	 * Specialisation for scalar \a x. Masked operator version.
	 */
	template<
		Descriptor descr = descriptors::no_operation, class OP,
		typename OutputType, typename MaskType,
		typename InputType1, typename InputType2,
		typename Coords
	>
	RC eWiseApply(
		Vector< OutputType, reference, Coords > &z,
		const Vector< MaskType, reference, Coords > &mask,
		const InputType1 alpha,
		const Vector< InputType2, reference, Coords > &y,
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
#ifdef _DEBUG
		std::cout << "In masked eWiseApply ([T1]<-T2<-[T3], operator variant)\n";
#endif
		// check for empty mask
		if( size( mask ) == 0 ) {
			return eWiseApply< descr >( z, alpha, y, op );
		}

		// check delegate to unmasked
		const size_t n = internal::getCoordinates( mask ).size();
		const auto &mask_coors = internal::getCoordinates( mask );
		if( (descr & descriptors::structural) &&
			!(descr & descriptors::invert_mask) &&
			mask_coors.nonzeroes() == n
		) {
			return eWiseApply< descr >( z, alpha, y, op );
		}

		// sanity checks
		if( internal::getCoordinates( y ).size() != n ) {
			return MISMATCH;
		}
		if( internal::getCoordinates( z ).size() != n ) {
			return MISMATCH;
		}
		if( descr & descriptors::dense ) {
			if( nnz( z ) < size( z ) ) { return ILLEGAL; }
			if( nnz( y ) < size( y ) ) { return ILLEGAL; }
			if( nnz( mask ) < size( mask ) ) { return ILLEGAL; }
		}

		if( phase == RESIZE ) {
			return SUCCESS;
		}
		assert( phase == EXECUTE );

		auto &z_coors = internal::getCoordinates( z );
		OutputType * const z_p = internal::getRaw( z );
		const MaskType * const mask_p = internal::getRaw( mask );
		const InputType2 * const y_p = internal::getRaw( y );
		const auto &y_coors = internal::getCoordinates( y );

		// the output sparsity structure is implied by mask and descr
		z_coors.clear();

		if( (descr & descriptors::dense) ||
			(y_coors.nonzeroes() == n) ||
			mask_coors.nonzeroes() <= y_coors.nonzeroes()
		) {
			return internal::masked_apply_generic< true, false, false, false, descr >(
				z_p, z_coors, mask_p, mask_coors, &alpha, y_p, op, n
			);
		} else {
			const Coords * const null_coors = nullptr;
			return internal::sparse_apply_generic< true, false, true, false, descr >(
				z_p, z_coors, mask_p, &mask_coors, &alpha, null_coors, y_p, &y_coors, op, n
			);
		}
	}

	/**
	 * Calculates the element-wise operation on elements of two vectors,
	 * \f$ z = x .* y \f$, using the given operator. The vectors must be
	 * of equal length.
	 *
	 * The vectors \a x or \a y may not be sparse.
	 *
	 * For all valid indices \a i of \a z, its element \f$ z_i \f$ after
	 * the call to this function completes equals \f$ x_i \odot y_i \f$.
	 *
	 * \warning Use of sparse vectors is only supported in full generality
	 *          when applied via a monoid or semiring; otherwise, there is
	 *          no concept for correctly interpreting any missing vector
	 *          elements during the requested computation.
	 * \note    When applying element-wise operators on sparse vectors
	 *          using semirings, there is a difference between interpreting missing
	 *          values as an annihilating identity or as a neutral identity--
	 *          intuitively, identities are known as `zero' or `one',
	 *          respectively. As a consequence, there are three different variants
	 *          for element-wise operations whose names correspond to their
	 *          intuitive meanings w.r.t. those identities:
	 *            -# eWiseAdd,
	 *            -# eWiseMul, and
	 *            -# eWiseMulAdd.
	 *
	 * @tparam descr The descriptor to be used (descriptors::no_operation if left
	 *               unspecified).
	 * @tparam OP    The operator to use.
	 * @tparam InputType1 The value type of the left-hand side vector.
	 * @tparam InputType2 The value type of the right-hand side vector.
	 * @tparam OutputType The value type of the ouput vector.
	 *
	 * @param[in]  x  The left-hand input vector. May not equal \a y.
	 * @param[in]  y  The right-hand input vector. May not equal \a x.
	 * @param[out] z  The pre-allocated output vector.
	 * @param[in]  op The operator to use.
	 *
	 * @return grb::ILLEGAL  When \a x equals \a y.
	 * @return grb::MISMATCH Whenever the dimensions of \a x, \a y, and \a z
	 *                       do not match. All input data containers are left
	 *                       untouched if this exit code is returned; it will
	 *                       be as though this call was never made.
	 * @return grb::SUCCESS  On successful completion of this call.
	 *
	 * \parblock
	 * \par Performance semantics
	 *      -# This call comprises \f$ \Theta(n) \f$ work, where \f$ n \f$ equals
	 *         the size of the vectors \a x, \a y, and \a z. The constant factor
	 *         depends on the cost of evaluating the operator. A good
	 *         implementation uses vectorised instructions whenever the input
	 *         domains, the output domain, and the operator used allow for this.
	 *
	 *      -# This call takes \f$ \mathcal{O}(1) \f$ memory beyond the memory
	 *         used by the application at the point of a call to this function.
	 *
	 *      -# This call incurs at most
	 *         \f$ n(
	 *               \mathit{sizeof}(\mathit{OutputType}) +
	 *               \mathit{sizeof}(\mathit{InputType1}) +
	 *               \mathit{sizeof}(\mathit{InputType2})
	 *             ) +
	 *         \mathcal{O}(1) \f$
	 *         bytes of data movement. A good implementation will stream \a x or
	 *         \a y into \a z to apply the multiplication operator in-place,
	 *         whenever the input domains, the output domain, and the operator
	 *         used allow for this.
	 * \endparblock
	 */
	template<
		Descriptor descr = descriptors::no_operation, class OP,
		typename OutputType, typename InputType1, typename InputType2,
		typename Coords
	>
	RC eWiseApply(
		Vector< OutputType, reference, Coords > &z,
		const Vector< InputType1, reference, Coords > &x,
		const Vector< InputType2, reference, Coords > &y,
		const OP &op = OP(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if< !grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			grb::is_operator< OP >::value, void
		>::type * const = nullptr
	) {
#ifdef _DEBUG
		std::cout << "In eWiseApply ([T1]<-[T2]<-[T3]), operator variant\n";
#endif
		// dynamic sanity checks
		auto &z_coors = internal::getCoordinates( z );
		const size_t n = z_coors.size();
		if( internal::getCoordinates( x ).size() != n ||
			internal::getCoordinates( y ).size() != n
		) {
#ifdef _DEBUG
			std::cerr << "\tinput vectors mismatch in dimensions!\n";
#endif
			return MISMATCH;
		}
		if( descr & descriptors::dense ) {
			if( nnz( z ) < size( z ) ) { return ILLEGAL; }
			if( nnz( x ) < size( x ) ) { return ILLEGAL; }
			if( nnz( y ) < size( y ) ) { return ILLEGAL; }
		}

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

		// check for sparsity
		if( !(descr & descriptors::dense) && (
				internal::getCoordinates( x ).nonzeroes() < n ||
				internal::getCoordinates( y ).nonzeroes() < n
			)
		) {
			// sparse case
			z_coors.clear();
			const bool * const null_mask = nullptr;
			const Coords * const null_coors = nullptr;
			return internal::sparse_apply_generic<
				false, false, false, false, descr | descriptors::dense
			>(
				internal::getRaw( z ), z_coors, null_mask, null_coors,
				internal::getRaw( x ), &( internal::getCoordinates( x ) ),
				internal::getRaw( y ), &( internal::getCoordinates( y ) ),
				op, n
			);
		}

		// dense case
		if( internal::getCoordinates( z ).nonzeroes() < n ) {
			internal::getCoordinates( z ).assignAll();
		}

		const InputType1 * __restrict__ a = internal::getRaw( x );
		const InputType2 * __restrict__ b = internal::getRaw( y );
		OutputType * __restrict__ c = internal::getRaw( z );

		// no, so use eWiseApply
#ifdef _H_GRB_REFERENCE_OMP_BLAS1
		#pragma omp parallel
#endif
		{
#ifdef _H_GRB_REFERENCE_OMP_BLAS1
			size_t start, end;
			config::OMP::localRange( start, end, 0, n, OP::blocksize );
#else
			const size_t start = 0;
			const size_t end = n;
#endif
			if( end > start ) {
				// this function is vectorised
				op.eWiseApply( a + start, b + start, c + start, end - start );
			}
		}

		// done
		return SUCCESS;
	}

	/**
	 * Computes \f$ z = x \odot y \f$, out of place.
	 *
	 * Masked operator version.
	 */
	template<
		Descriptor descr = descriptors::no_operation, class OP,
		typename OutputType, typename MaskType,
		typename InputType1, typename InputType2,
		typename Coords
	>
	RC eWiseApply(
		Vector< OutputType, reference, Coords > &z,
		const Vector< MaskType, reference, Coords > &mask,
		const Vector< InputType1, reference, Coords > &x,
		const Vector< InputType2, reference, Coords > &y,
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
#ifdef _DEBUG
		std::cout << "In masked eWiseApply ([T1]<-[T2]<-[T3], using operator)\n";
#endif
		// check for empty mask
		if( size( mask ) == 0 ) {
			return eWiseApply< descr >( z, x, y, op, phase );
		}

		// check if can delegate to unmasked variant
		const auto &m_coors = internal::getCoordinates( mask );
		const size_t n = m_coors.size();
		if( m_coors.nonzeroes() == n &&
			(descr & descriptors::structural) &&
			!(descr & descriptors::invert_mask)
		) {
			return eWiseApply< descr >( z, x, y, op );
		}

		// other run-time checks
		auto &z_coors = internal::getCoordinates( z );
		const auto &mask_coors = internal::getCoordinates( mask );
		if( internal::getCoordinates( x ).size() != n ) {
			return MISMATCH;
		}
		if( internal::getCoordinates( y ).size() != n ) {
			return MISMATCH;
		}
		if( z_coors.size() != n ) {
			return MISMATCH;
		}
		if( descr & descriptors::dense ) {
			if( nnz( z ) < size( z ) ) { return ILLEGAL; }
			if( nnz( x ) < size( x ) ) { return ILLEGAL; }
			if( nnz( y ) < size( y ) ) { return ILLEGAL; }
			if( nnz( mask ) < size( mask ) ) { return ILLEGAL; }
		}

		if( phase == RESIZE ) {
			return SUCCESS;
		}
		assert( phase == EXECUTE );

		OutputType * const z_p = internal::getRaw( z );
		const MaskType * const mask_p = internal::getRaw( mask );
		const InputType1 * const x_p = internal::getRaw( x );
		const InputType2 * const y_p = internal::getRaw( y );
		const auto &x_coors = internal::getCoordinates( x );
		const auto &y_coors = internal::getCoordinates( y );
		const size_t sparse_loop =
			std::min( x_coors.nonzeroes(), y_coors.nonzeroes() );

		// the output sparsity structure is unknown a priori
		z_coors.clear();

		if( (descr & descriptors::dense) ||
			(x_coors.nonzeroes() == n && y_coors.nonzeroes() == n) ||
			( !(descr & descriptors::invert_mask) && sparse_loop >= m_coors.nonzeroes() )
		) {
			// use loop over mask
			return internal::masked_apply_generic< false, false, false, false, descr >(
				z_p, z_coors, mask_p, mask_coors, x_p, y_p, op, n
			);
		} else {
			// use loop over sparse inputs
			return internal::sparse_apply_generic< true, false, false, false, descr >(
				z_p, z_coors, mask_p, &mask_coors, x_p, &x_coors, y_p, &y_coors, op, n
			);
		}
	}

	/**
	 * Calculates the element-wise addition of two vectors, \f$ z = x .+ y \f$,
	 * under this semiring.
	 *
	 * @tparam descr      The descriptor to be used (descriptors::no_operation
	 *                    if left unspecified).
	 * @tparam Ring       The semiring type to perform the element-wise addition
	 *                    on.
	 * @tparam InputType1 The left-hand side input type to the additive operator
	 *                    of the \a ring.
	 * @tparam InputType2 The right-hand side input type to the additive operator
	 *                    of the \a ring.
	 * @tparam OutputType The the result type of the additive operator of the
	 *                    \a ring.
	 *
	 * @param[out]  z  The output vector of type \a OutputType. This may be a
	 *                 sparse vector.
	 * @param[in]   x  The left-hand input vector of type \a InputType1. This may
	 *                 be a sparse vector.
	 * @param[in]   y  The right-hand input vector of type \a InputType2. This may
	 *                 be a sparse vector.
	 * @param[in] ring The generalized semiring under which to perform this
	 *                 element-wise multiplication.
	 *
	 * @return grb::MISMATCH Whenever the dimensions of \a x, \a y, and \a z do
	 *                       not match. All input data containers are left
	 *                       untouched; it will be as though this call was never
	 *                       made.
	 * @return grb::SUCCESS  On successful completion of this call.
	 *
	 * \parblock
	 * \par Valid descriptors
	 * grb::descriptors::no_operation, grb::descriptors::no_casting,
	 * grb::descriptors::dense.
	 *
	 * \note Invalid descriptors will be ignored.
	 *
	 * If grb::descriptors::no_casting is specified, then 1) the third domain of
	 * \a ring must match \a InputType1, 2) the fourth domain of \a ring must match
	 * \a InputType2, 3) the fourth domain of \a ring must match \a OutputType. If
	 * one of these is not true, the code shall not compile.
	 * \endparblock
	 *
	 * \parblock
	 * \par Performance semantics
	 *      -# This call takes \f$ \Theta(n) \f$ work, where \f$ n \f$ equals the
	 *         size of the vectors \a x, \a y, and \a z. The constant factor
	 *         depends on the cost of evaluating the addition operator. A good
	 *         implementation uses vectorised instructions whenever the input
	 *         domains, the output domain, and the the additive operator used
	 *         allow for this.
	 *
	 *      -# This call will not result in additional dynamic memory allocations.
	 *         No system calls will be made.
	 *
	 *      -# This call takes \f$ \mathcal{O}(1) \f$ memory beyond the memory
	 *         used by the application at the point of a call to this function.
	 *
	 *      -# This call incurs at most
	 *         \f$ n( \mathit{sizeof}(
	 *             \mathit{InputType1} +
	 *             \mathit{InputType2} +
	 *             \mathit{OutputType}
	 *           ) + \mathcal{O}(1) \f$
	 *         bytes of data movement. A good implementation will stream \a x or
	 *         \a y into \a z to apply the additive operator in-place, whenever
	 *         the input domains, the output domain, and the operator used allow
	 *         for this.
	 * \endparblock
	 *
	 * @see This is a specialised form of eWiseMulAdd.
	 *
	 * \warning This primitive has been deprecated since version 0.5. Please update
	 *          any use of this operation to an equivalent one using a sequence of
	 *          folds using the additive monoid if \a z is used in-place, or in the
	 *          case of out-of-place use of \a z by a call to grb::eWiseApply using
	 *          the additive monoid.
	 */
	template<
		Descriptor descr = descriptors::no_operation, class Ring,
		typename OutputType, typename InputType1, typename InputType2,
		typename Coords
	>
	RC eWiseAdd(
		Vector< OutputType, reference, Coords > &z,
		const Vector< InputType1, reference, Coords > &x,
		const Vector< InputType2, reference, Coords > &y,
		const Ring &ring = Ring(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if< !grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			grb::is_semiring< Ring >::value, void
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
		std::cout << "eWiseAdd (reference, vector <- vector + vector) dispatches to "
			<< "eWiseApply( reference, vector <- vector . vector ) using an "
			<< "additive monoid\n";
#endif
		return eWiseApply< descr >( z, x, y, ring.getAdditiveMonoid(), phase );
	}

	/**
	 * Calculates the element-wise addition of two vectors, \f$ z = x .+ y \f$,
	 * under the given semiring.
	 *
	 * Specialisation for scalar \a x.
	 *
	 * \warning This primitive has been deprecated since version 0.5. Please update
	 *          any use of this operation to an equivalent one using a sequence of
	 *          folds using the additive monoid if \a z is used in-place, or in the
	 *          case of out-of-place use of \a z by a call to grb::eWiseApply using
	 *          the additive monoid.
	 */
	template<
		Descriptor descr = descriptors::no_operation, class Ring,
		typename InputType1, typename InputType2, typename OutputType,
		typename Coords
	>
	RC eWiseAdd(
		Vector< OutputType, reference, Coords > &z,
		const InputType1 alpha,
		const Vector< InputType2, reference, Coords > &y,
		const Ring &ring = Ring(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if< !grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			grb::is_semiring< Ring >::value, void
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
		std::cout << "eWiseAdd (reference, vector <- scalar + vector) dispatches to "
			<< "eWiseApply with additive monoid\n";
#endif
		return eWiseApply< descr >( z, alpha, y, ring.getAdditiveMonoid(), phase );
	}

	/**
	 * Calculates the element-wise addition of two vectors, \f$ z = x .+ y \f$,
	 * under the given semiring.
	 *
	 * Specialisation for scalar \a y.
	 *
	 * \warning This primitive has been deprecated since version 0.5. Please update
	 *          any use of this operation to an equivalent one using a sequence of
	 *          folds using the additive monoid if \a z is used in-place, or in the
	 *          case of out-of-place use of \a z by a call to grb::eWiseApply using
	 *          the additive monoid.
	 */
	template<
		Descriptor descr = descriptors::no_operation, class Ring,
		typename InputType1, typename InputType2, typename OutputType,
		typename Coords
	>
	RC eWiseAdd(
		Vector< OutputType, reference, Coords > &z,
		const Vector< InputType1, reference, Coords > &x,
		const InputType2 beta,
		const Ring &ring = Ring(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if< !grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			grb::is_semiring< Ring >::value, void
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
		std::cout << "eWiseAdd (reference, vector <- vector + scalar) dispatches to eWiseApply with additive monoid\n";
#endif
		return eWiseApply< descr >( z, x, beta, ring.getAdditiveMonoid(), phase );
	}

	/**
	 * Calculates the element-wise addition of two vectors, \f$ z = x .+ y \f$,
	 * under the given semiring.
	 *
	 * Specialisation for scalar \a x and \a y.
	 *
	 * \warning This primitive has been deprecated since version 0.5. Please update
	 *          any use of this operation to an equivalent one using a sequence of
	 *          folds using the additive monoid if \a z is used in-place, or in the
	 *          case of out-of-place use of \a z by a call to grb::eWiseApply using
	 *          the additive monoid.
	 */
	template<
		Descriptor descr = descriptors::no_operation, class Ring,
		typename InputType1, typename InputType2, typename OutputType,
		typename Coords
	>
	RC eWiseAdd(
		Vector< OutputType, reference, Coords > &z,
		const InputType1 alpha,
		const InputType2 beta,
		const Ring &ring = Ring(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if< !grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			grb::is_semiring< Ring >::value, void
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
		std::cout << "eWiseAdd (reference, vector <- scalar + scalar) dispatches to "
			<< "foldl with precomputed scalar and additive monoid\n";
#endif
		const typename Ring::D4 add;
		(void) apply( add, alpha, beta, ring.getAdditiveOperator() );
		return foldl< descr >( z, add, ring.getAdditiveMonoid(), phase );
	}

	/**
	 * Calculates the element-wise addition of two vectors, \f$ z = x .+ y \f$,
	 * under the given semiring.
	 *
	 * Masked version.
	 *
	 * \warning This primitive has been deprecated since version 0.5. Please update
	 *          any use of this operation to an equivalent one using a sequence of
	 *          folds using the additive monoid if \a z is used in-place, or in the
	 *          case of out-of-place use of \a z by a call to grb::eWiseApply using
	 *          the additive monoid.
	 */
	template<
		Descriptor descr = descriptors::no_operation, class Ring,
		typename OutputType, typename MaskType,
		typename InputType1, typename InputType2,
		typename Coords
	>
	RC eWiseAdd(
		Vector< OutputType, reference, Coords > &z,
		const Vector< MaskType, reference, Coords > &m,
		const Vector< InputType1, reference, Coords > &x,
		const Vector< InputType2, reference, Coords > &y,
		const Ring &ring = Ring(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if< !grb::is_object< OutputType >::value &&
			!grb::is_object< MaskType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			grb::is_semiring< Ring >::value, void
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
		std::cout << "eWiseAdd (reference, vector <- vector + vector, masked) "
			<< "dispatches to eWiseApply( reference, vector <- vector . vector ) using "
			<< "an additive monoid\n";
#endif
		return eWiseApply< descr >( z, m, x, y, ring.getAdditiveMonoid(), phase );
	}

	/**
	 * Calculates the element-wise addition of two vectors, \f$ z = x .+ y \f$,
	 * under the given semiring.
	 *
	 * Specialisation for scalar \a x, masked version
	 *
	 * \warning This primitive has been deprecated since version 0.5. Please update
	 *          any use of this operation to an equivalent one using a sequence of
	 *          folds using the additive monoid if \a z is used in-place, or in the
	 *          case of out-of-place use of \a z by a call to grb::eWiseApply using
	 *          the additive monoid.
	 */
	template<
		Descriptor descr = descriptors::no_operation, class Ring,
		typename InputType1, typename InputType2,
		typename OutputType, typename MaskType,
		typename Coords
	>
	RC eWiseAdd(
		Vector< OutputType, reference, Coords > &z,
		const Vector< MaskType, reference, Coords > &m,
		const InputType1 alpha,
		const Vector< InputType2, reference, Coords > &y,
		const Ring &ring = Ring(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if< !grb::is_object< OutputType >::value &&
			!grb::is_object< MaskType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			grb::is_semiring< Ring >::value, void
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
		std::cout << "eWiseAdd (reference, vector <- scalar + vector, masked) dispatches to eWiseApply with additive monoid\n";
#endif
		return eWiseApply< descr >( z, m, alpha, y, ring.getAdditiveMonoid(), phase );
	}

	/**
	 * Calculates the element-wise addition of two vectors, \f$ z = x .+ y \f$,
	 * under the given semiring.
	 *
	 * Specialisation for scalar \a y, masked version.
	 *
	 * \warning This primitive has been deprecated since version 0.5. Please update
	 *          any use of this operation to an equivalent one using a sequence of
	 *          folds using the additive monoid if \a z is used in-place, or in the
	 *          case of out-of-place use of \a z by a call to grb::eWiseApply using
	 *          the additive monoid.
	 */
	template<
		Descriptor descr = descriptors::no_operation, class Ring,
		typename InputType1, typename InputType2,
		typename OutputType, typename MaskType,
		typename Coords
	>
	RC eWiseAdd(
		Vector< OutputType, reference, Coords > &z,
		const Vector< MaskType, reference, Coords > &m,
		const Vector< InputType1, reference, Coords > &x,
		const InputType2 beta,
		const Ring &ring = Ring(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if< !grb::is_object< OutputType >::value &&
			!grb::is_object< MaskType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			grb::is_semiring< Ring >::value, void
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
		std::cout << "eWiseAdd (reference, vector <- vector + scalar, masked) dispatches to eWiseApply with additive monoid\n";
#endif
		return eWiseApply< descr >( z, m, x, beta, ring.getAdditiveMonoid(), phase );
	}

	/**
	 * Calculates the element-wise addition of two vectors, \f$ z = x .+ y \f$,
	 * under the given semiring.
	 *
	 * Specialisation for scalar \a x and \a y, masked version.
	 *
	 * \warning This primitive has been deprecated since version 0.5. Please update
	 *          any use of this operation to an equivalent one using a sequence of
	 *          folds using the additive monoid if \a z is used in-place, or in the
	 *          case of out-of-place use of \a z by a call to grb::eWiseApply using
	 *          the additive monoid.
	 */
	template<
		Descriptor descr = descriptors::no_operation, class Ring,
		typename InputType1, typename InputType2,
		typename OutputType, typename MaskType,
		typename Coords
	>
	RC eWiseAdd(
		Vector< OutputType, reference, Coords > &z,
		const Vector< OutputType, reference, Coords > &m,
		const InputType1 alpha,
		const InputType2 beta,
		const Ring &ring = Ring(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if< !grb::is_object< OutputType >::value &&
			!grb::is_object< MaskType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			grb::is_semiring< Ring >::value, void
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
		std::cout << "eWiseAdd (reference, vector <- scalar + scalar, masked) "
			<< "dispatches to foldl with precomputed scalar and additive monoid\n";
#endif
		const typename Ring::D4 add;
		(void) apply( add, alpha, beta, ring.getAdditiveOperator() );
		return foldl< descr >( z, m, add, ring.getAdditiveMonoid(), phase );
	}

	// declare an internal version of eWiseMulAdd containing the full
	// sparse & dense implementations
	namespace internal {

		/**
		 * \internal
		 * This variant fuses the multiplication, addition, and folding into the
		 * output vector while looping over the elements where the mask evaluates
		 * true. Since the reference implementation does not support Theta(n-nz)
		 * loops in cases where the mask is structural and inverted, it (statically)
		 * asserts that the mask is not inverted.
		 */
		template<
			Descriptor descr,
			bool a_scalar, bool x_scalar, bool y_scalar, bool y_zero,
			typename OutputType, typename MaskType,
			typename InputType1, typename InputType2, typename InputType3,
			typename CoorsType, class Ring
		>
		RC sparse_eWiseMulAdd_maskDriven(
			Vector< OutputType, reference, CoorsType > &z_vector,
			const MaskType * __restrict__ m,
			const CoorsType * const m_coors,
			const InputType1 * __restrict__ a,
			const CoorsType * const a_coors,
			const InputType2 * __restrict__ x,
			const CoorsType * const x_coors,
			const InputType3 * __restrict__ y,
			const CoorsType * const y_coors,
			const size_t n,
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
			auto &z_coors = internal::getCoordinates( z_vector );
			assert( z != a );
			assert( z != x );
			assert( z != y );
			assert( a != x );
			assert( a != y );
			assert( x != y );
			assert( m_coors != nullptr );
#ifdef NDEBUG
			(void) n;
#endif

#ifdef _H_GRB_REFERENCE_OMP_BLAS1
			#pragma omp parallel
			{
				size_t start, end;
				config::OMP::localRange( start, end, 0, m_coors->nonzeroes() );
#else
				const size_t start = 0;
				const size_t end = m_coors->nonzeroes();
#endif
				size_t k = start;

// vectorised code only for sequential backend
#ifndef _H_GRB_REFERENCE_OMP_BLAS1

				/*
				 * CODE DISABLED pending benchmarks to see whether it is worthwhile--
				 * see internal issue #163

				constexpr size_t size_t_bs =
					config::CACHE_LINE_SIZE::value() / sizeof(size_t) == 0 ?
					1 :
					config::CACHE_LINE_SIZE::value() / sizeof(size_t);
				constexpr size_t blocksize = Ring::blocksize < size_t_bs ?
					Ring::blocksize :
					size_t_bs;

				// vector registers
				bool am[ blocksize ];
				bool xm[ blocksize ];
				bool ym[ blocksize ];
				bool mm[ blocksize ];
				bool zm[ blocksize ];
				size_t indices[ blocksize ];
				typename Ring::D1 aa[ blocksize ];
				typename Ring::D2 xx[ blocksize ];
				typename Ring::D3 tt[ blocksize ];
				typename Ring::D4 bb[ blocksize ];
				typename Ring::D4 yy[ blocksize ];
				typename Ring::D4 zz[ blocksize ];

				if( a_scalar ) {
				    for( size_t b = 0; b < Ring::blocksize; ++b ) {
				        aa[ b ] = static_cast< typename Ring::D1 >(*a);
				    }
				}
				if( x_scalar ) {
				    for( size_t b = 0; b < Ring::blocksize; ++b ) {
				        xx[ b ] = static_cast< typename Ring::D2 >(*x);
				    }
				}
				if( y_scalar ) {
				    for( size_t b = 0; b < Ring::blocksize; ++b ) {
				        yy[ b ] = static_cast< typename Ring::D4 >(*y);
				    }
				}

				// vectorised loop
				while( k + Ring::blocksize < end ) {
				    // set masks
				    for( size_t b = 0; b < blocksize; ++b ) {
				        am[ b ] = xm[ b ] = ym[ b ] = mm[ b ] = zm[ b ] = false;
				    }
				    // gathers
				    for( size_t b = 0; b < blocksize; ++b ) {
				        indices[ b ] = m_coors->index( k + b );
				    }
				    for( size_t b = 0; b < blocksize; ++b ) {
				        mm[ b ] = utils::interpretMask< descr >( true, m, indices[ b ] );
				    }
				    // masked gathers
				    for( size_t b = 0; b < blocksize; ++b ) {
				        if( mm[ b ] ) {
				            zm[ b ] = z_coors.assigned( indices[ b ] );
				        }
				    }
				    if( !a_scalar ) {
				        for( size_t b = 0; b < blocksize; ++b ) {
				            if( mm[ b ] ) {
				                am[ b ] = a_coors->assigned( indices[ b ] );
				            }
				        }
				    }
				    if( !y_scalar ) {
				        for( size_t b = 0; b < blocksize; ++b ) {
				            if( mm[ b ] ) {
				                ym[ b ] = y_coors->assigned( indices[ b ] );
				            }
				        }
				    }
				    if( !a_scalar ) {
				        for( size_t b = 0; b < blocksize; ++b ) {
				            if( am[ b ] ) {
				                aa[ b ] =
				                   static_cast< typename Ring::D1 >( a[ indices[ b ] ] );
				            }
				        }
				    }
				    if( !x_scalar ) {
				        for( size_t b = 0; b < blocksize; ++b ) {
				            if( xm[ b ] ) {
				                xx[ b ] =
				                   static_cast< typename Ring::D2 >( y[ indices[ b ] ] );
				            }
				        }
				    }
				    if( !y_scalar ) {
				        for( size_t b = 0; b < blocksize; ++b ) {
				            if( ym[ b ] ) {
				                yy[ b ] =
				                   static_cast< typename Ring::D4 >( y[ indices[ b ] ] );
				            }
				        }
				    }

				    // do multiplication
				    if( !a_scalar && !x_scalar ) {
				        for( size_t b = 0; b < blocksize; ++b ) {
				            mm[ b ] = am[ b ] && xm[ b ];
				        }
				    } else if( a_scalar ) {
				        for( size_t b = 0; b < blocksize; ++b ) {
				            mm[ b ] = xm[ b ];
				        }
				    } else {
				        for( size_t b = 0; b < blocksize; ++b ) {
				            mm[ b ] = am[ b ];
				        }
				    }
				    for( size_t b = 0; b < blocksize; ++b ) {
				        if( mm[ b ] ) {
				            (void) apply( tt[ b ], aa[ b ], xx[ b ],
				               ring.getMultiplicativeOperator() );
				        }
				    }

				    // at this point am and xm are free to re-use

				    // do addition
				    for( size_t b = 0; b < blocksize; ++b ) {
				        xm[ b ] = mm[ b ] && !ym[ b ];
				    }
				    for( size_t b = 0; b < blocksize; ++b ) {
				        am[ b ] = !mm[ b ] && ym[ b ];
				    }
				    for( size_t b = 0; b < blocksize; ++b ) {
				        mm[ b ] = mm[ b ] && ym[ b ];
				    }
				    for( size_t b = 0; b < blocksize; ++b ) {
				        if( mm[ b ] ) {
				            (void) apply( bb[ b ], tt[ b ], yy[ b ],
				                ring.getAdditiveOperator() );
				        }
				    }
				    for( size_t b = 0; b < blocksize; ++b ) {
				        if( xm[ b ] ) {
				            bb[ b ] = tt[ b ];
				        }
				    }
				    for( size_t b = 0; b < blocksize; ++b ) {
				        if( am[ b ] ) {
				            bb[ b ] = yy[ b ];
				        }
				    }

				    // accumulate into output
				    for( size_t b = 0; b < blocksize; ++b ) {
				        xm[ b ] = xm[ b ] || am[ b ];
				    }
				    for( size_t b = 0; b < blocksize; ++b ) {
				        mm[ b ] = mm[ b ] || xm[ b ];
				    }
				    for( size_t b = 0; b < blocksize; ++b ) {
				        am[ b ] = mm[ b ] && zm[ b ];
				    }
				    for( size_t b = 0; b < blocksize; ++b ) {
				        if( am[ b ] ) {
				            (void) foldr( bb[ b ], zz[ b ], ring.getAdditiveOperator() );
				        }
				    }
				    for( size_t b = 0; b < blocksize; ++b ) {
				        xm[ b ] = mm[ b ] && !zm[ b ];
				    }
				    for( size_t b = 0; b < blocksize; ++b ) {
				        if( xm[ b ] ) {
				            zz[ b ] = bb[ b ];
				        }
				    }

				    // scatter-like
				    for( size_t b = 0; b < blocksize; ++b ) {
				        if( xm[ b ] ) {
				            (void) z_coors.assign( indices[ b ] );
				        }
				    }

				    // scatter
				    for( size_t b = 0; b < blocksize; ++b ) {
				        zm[ b ] = zm[ b ] || mm[ b ];
				    }
				    for( size_t b = 0; b < blocksize; ++b ) {
				        if( zm[ b ] ) {
				            z[ indices[ b ] ] = static_cast< OutputType >( zz[ b ] );
				        }
				    }

				    // move to next block
				    k += blocksize;
				}*/
#endif
				// scalar coda and parallel main body
#ifdef _H_GRB_REFERENCE_OMP_BLAS1
				internal::Coordinates< reference >::Update localUpdate =
					z_coors.EMPTY_UPDATE();
				const size_t maxAsyncAssigns = z_coors.maxAsyncAssigns();
				size_t asyncAssigns = 0;
#endif
				for( ; k < end; ++k ) {
					const size_t index = m_coors->index( k );
					assert( index < n );
					if( !( m_coors->template mask< descr >( index, m )) ) {
						continue;
					}
					typename Ring::D3 t = ring.template getZero< typename Ring::D3 >();
					if( ( a_scalar || a_coors->assigned( index ) ) &&
						( x_scalar || x_coors->assigned( index ) )
					) {
						const InputType1 * const a_p = a + ( a_scalar ? 0 : index );
						const InputType2 * const x_p = x + ( x_scalar ? 0 : index );
						(void) apply( t, *a_p, *x_p, ring.getMultiplicativeOperator() );
						if( !y_zero && (y_scalar || y_coors->assigned( index )) ) {
							const InputType3 * const y_p = y + ( y_scalar ? 0 : index );
							typename Ring::D4 b;
							(void) apply( b, t, *y_p, ring.getAdditiveOperator() );
							if( z_coors.assigned( index ) ) {
								typename Ring::D4 out = static_cast< typename Ring::D4 >( z[ index ] );
								(void) foldr( b, out, ring.getAdditiveOperator() );
								z[ index ] = static_cast< OutputType >( out );
							} else {
#ifdef _H_GRB_REFERENCE_OMP_BLAS1
								(void) z_coors.asyncAssign( index, localUpdate );
								(void) ++asyncAssigns;
#else
								(void) z_coors.assign( index );
#endif
								z[ index ] = static_cast< OutputType >( b );
							}
						} else if( z_coors.assigned( index ) ) {
							typename Ring::D4 out = static_cast< typename Ring::D4 >( z[ index ] );
							(void) foldr( t, out, ring.getAdditiveOperator() );
							z[ index ] = static_cast< OutputType >( out );
						} else {
#ifdef _H_GRB_REFERENCE_OMP_BLAS1
							(void) z_coors.asyncAssign( index, localUpdate );
							(void) ++asyncAssigns;
#else
							(void) z_coors.assign( index );
#endif
							z[ index ] = static_cast< OutputType >( t );
						}
					} else if( !y_zero && (y_scalar || y_coors->assigned( index )) ) {
						if( z_coors.assigned( index ) ) {
							typename Ring::D4 out = static_cast< typename Ring::D4 >( z[ index ] );
							(void) foldr( y[ index ], out, ring.getAdditiveOperator() );
							z[ index ] = static_cast< OutputType >( out );
						} else {
#ifdef _H_GRB_REFERENCE_OMP_BLAS1
							(void) z_coors.asyncAssign( index, localUpdate );
							(void) ++asyncAssigns;
#else
							(void) z_coors.assign( index );
#endif
							z[ index ] = static_cast< OutputType >( t );
						}
					}
#ifdef _H_GRB_REFERENCE_OMP_BLAS1
					if( asyncAssigns == maxAsyncAssigns ) {
						const bool was_empty = z_coors.joinUpdate( localUpdate );
#ifdef NDEBUG
						(void) was_empty;
#else
						assert( !was_empty );
#endif
						asyncAssigns = 0;
					}
#endif
				}
#ifdef _H_GRB_REFERENCE_OMP_BLAS1
				while( !z_coors.joinUpdate( localUpdate ) ) {}
			} // end pragma omp parallel
#endif
			return SUCCESS;
		}

		/**
		 * \internal
		 * Call this version if consuming the multiplication first and performing the
		 * addition separately is cheaper than fusing the computations as done in the
		 * mask-driven variant.
		 */
		template<
			Descriptor descr,
			bool masked, bool x_scalar, bool y_scalar, bool y_zero, bool mulSwitched,
			typename OutputType, typename MaskType,
			typename InputType1, typename InputType2, typename InputType3,
			typename CoorsType, class Ring
		>
		RC twoPhase_sparse_eWiseMulAdd_mulDriven(
			Vector< OutputType, reference, CoorsType > &z_vector,
			const Vector< MaskType, reference, CoorsType > * const m_vector,
			const InputType1 * __restrict__ a,
			const CoorsType &it_coors,
			const InputType2 * __restrict__ x,
			const CoorsType * const ck_coors,
			const Vector< InputType3, reference, CoorsType > * const y_vector,
			const InputType3 * __restrict__ y,
			const size_t n,
			const Ring &ring = Ring()
		) {
			InputType3 * __restrict__ z = internal::getRaw( z_vector );
			auto &z_coors = internal::getCoordinates( z_vector );
			assert( z != a );
			assert( z != x );
			assert( a != x );

#ifdef _H_GRB_REFERENCE_OMP_BLAS1
			#pragma omp parallel
			{
				auto localUpdate = z_coors.EMPTY_UPDATE();
				const size_t maxAsyncAssigns = z_coors.maxAsyncAssigns();
				size_t asyncAssigns = 0;
				// choose dynamic schedule since the mask otherwise likely leads to
				// significant imbalance
				#pragma omp for schedule( dynamic, config::CACHE_LINE_SIZE::value() ) nowait
#endif
				for( size_t i = 0; i < it_coors.nonzeroes(); ++i ) {
					const size_t index = it_coors.index( i );
					if( masked ) {
						const MaskType * __restrict__ const m = internal::getRaw( *m_vector );
						const CoorsType * const m_coors =
							&(internal::getCoordinates( *m_vector ));
						if( !m_coors->template mask< descr >( index, m ) ) {
							continue;
						}
					}
					if( x_scalar || ck_coors->assigned( index ) ) {
						typename Ring::D3 t;
						const InputType1 * const a_p = a + index;
						const InputType2 * const x_p = x_scalar ? x : x + index;
						if( mulSwitched ) {
							(void) apply( t, *x_p, *a_p, ring.getMultiplicativeOperator() );
						} else {
							(void) apply( t, *a_p, *x_p, ring.getMultiplicativeOperator() );
						}
#ifdef _H_GRB_REFERENCE_OMP_BLAS1
						if( z_coors.asyncAssign( index, localUpdate ) ) {
#else
							if( z_coors.assign( index ) ) {
#endif
								typename Ring::D4 b = static_cast< typename Ring::D4 >( z[ index ] );
								(void) foldr( t, b, ring.getAdditiveOperator() );
								z[ index ] = static_cast< OutputType >( b );
							} else {
								z[ index ] = static_cast< OutputType >(
									static_cast< typename Ring::D4 >( t )
								);
#ifdef _H_GRB_REFERENCE_OMP_BLAS1
								(void) ++asyncAssigns;
								if( asyncAssigns == maxAsyncAssigns ) {
									(void) z_coors.joinUpdate( localUpdate );
									asyncAssigns = 0;
								}
#endif
							}
						}
					}
#ifdef _H_GRB_REFERENCE_OMP_BLAS1
				while( !z_coors.joinUpdate( localUpdate ) ) {}
			}
#endif

			// now handle addition
			if( !y_zero ) {
				if( masked ) {
					if( y_scalar ) {
						return foldl< descr >( z_vector, *m_vector, *y, ring.getAdditiveMonoid() );
					} else {
						return foldl< descr >( z_vector, *m_vector, *y_vector,
							ring.getAdditiveMonoid() );
					}
				} else {
					if( y_scalar ) {
						return foldl< descr >( z_vector, *y, ring.getAdditiveMonoid() );
					} else {
						return foldl< descr >( z_vector, *y_vector, ring.getAdditiveMonoid() );
					}
				}
			}

			// done
			return SUCCESS;
		}

		/**
		 * \internal
		 * In this variant, all vector input vectors, except potentially the
		 * input/output vector \a z, are dense.
		 *
		 * If \a z was not dense, it is assumed to be empty.
		 *
		 * @tparam assign_z True if \a z was empty, false otherwise.
		 *
		 * This implement the eWiseMulAdd using a direct Theta(n) loop. This variant
		 * is cheaper than any of the sparse variants when the output is dense.
		 * \endinternal
		 */
		template<
			Descriptor descr,
			bool a_scalar, bool x_scalar, bool y_scalar, bool y_zero, bool assign_z,
			typename OutputType, typename InputType1,
			typename InputType2, typename InputType3,
			typename CoorsType, class Ring
		>
		RC dense_eWiseMulAdd(
			Vector< OutputType, reference, CoorsType > &z_vector,
			const InputType1 * __restrict__ const a_in,
			const InputType2 * __restrict__ const x_in,
			const InputType3 * __restrict__ const y_in,
			const size_t n,
			const Ring &ring = Ring()
		) {
#ifdef _DEBUG
			std::cout << "\tdense_eWiseMulAdd: loop size will be " << n << "\n";
#endif
			OutputType * __restrict__ const z_in = internal::getRaw( z_vector );
			assert( z_in != a_in );
			assert( z_in != x_in );
			assert( z_in != y_in );
			assert( a_in != x_in );
			assert( a_in != y_in );
			assert( x_in != y_in );

#ifdef _H_GRB_REFERENCE_OMP_BLAS1
			#pragma omp parallel
			{
				size_t start, end;
				config::OMP::localRange( start, end, 0, n );
#else
				const size_t start = 0;
				const size_t end = n;
#endif
				// create local copies of the input const pointers
				const InputType1 * __restrict__ a = a_in;
				const InputType2 * __restrict__ x = x_in;
				const InputType3 * __restrict__ y = y_in;
				OutputType * __restrict__ z = z_in;

				// vector registers
				typename Ring::D1 aa[ Ring::blocksize ];
				typename Ring::D2 xx[ Ring::blocksize ];
				typename Ring::D3 tt[ Ring::blocksize ];
				typename Ring::D4 bb[ Ring::blocksize ];
				typename Ring::D4 yy[ Ring::blocksize ];
				typename Ring::D4 zz[ Ring::blocksize ];

				if( a_scalar ) {
					for( size_t b = 0; b < Ring::blocksize; ++b ) {
						aa[ b ] = *a;
					}
				}
				if( x_scalar ) {
					for( size_t b = 0; b < Ring::blocksize; ++b ) {
						xx[ b ] = *x;
					}
				}
				if( y_scalar ) {
					if( y_zero ) {
						for( size_t b = 0; b < Ring::blocksize; ++b ) {
							yy[ b ] = ring.template getZero< typename Ring::D4 >();
						}
					} else {
						for( size_t b = 0; b < Ring::blocksize; ++b ) {
							yy[ b ] = *y;
						}
					}
				}

				// do vectorised out-of-place operations. Allows for aligned overlap.
				// Non-aligned ovelap is not possible due to GraphBLAS semantics.
				size_t i = start;
				// note: read the tail code (under this while loop) comments first for
				//       greater understanding
				while( i + Ring::blocksize <= end ) {
#ifdef _DEBUG
					std::cout << "\tdense_eWiseMulAdd: handling block of size " <<
						Ring::blocksize << " starting at index " << i << "\n";
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
					const typename Ring::D1 &as = static_cast< typename Ring::D1 >( *a );
					const typename Ring::D2 &xs = static_cast< typename Ring::D2 >( *x );
					typename Ring::D4 ys = static_cast< typename Ring::D4 >( *y );
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
						(void) a++;
					}
					if( !x_scalar ) {
						(void) x++;
					}
					if( !y_scalar ) {
						(void) y++;
					}
					(void) z++;
				}
#ifdef _H_GRB_REFERENCE_OMP_BLAS1
			} // end OpenMP parallel section
#endif

			// done
			return SUCCESS;
		}

		/**
		 * \internal
		 * Depending on sparsity patterns, there are many variants of eWiseMulAdd.
		 * This function identifies and calls the most opportune variant. Also used
		 * to implement eWiseMul, though in that case the implementation could still
		 * be optimised by e.g. a bool that indicates whether y is zero (internal
		 * issue #488).
		 *
		 * @tparam a_scalar Whether \a a is a scalar or a vector.
		 *
		 * (and so on for \a x and \a y).
		 *
		 * @tparam assign_y Whether to simply assign to \a y or whether to
		 *                  (potentially) fold into \a y (in case there are
		 *                  pre-existing elements)
		 *
		 * The other arguments pertain to the output, the mask, and the input vectors
		 * as well as their sizes-- and finally the semiring under which to perform
		 * the requested computation.
		 * \endinternal
		 */
		template<
			Descriptor descr,
			bool masked, bool a_scalar, bool x_scalar, bool y_scalar, bool y_zero,
			typename MaskType, class Ring,
			typename InputType1, typename InputType2,
			typename InputType3, typename OutputType,
			typename CoorsType
		>
		RC eWiseMulAdd_dispatch(
			Vector< OutputType, reference, CoorsType > &z_vector,
			const Vector< MaskType, reference, CoorsType > * const m_vector,
			const InputType1 * __restrict__ a,
			const CoorsType * const a_coors,
			const InputType2 * __restrict__ x,
			const CoorsType * const x_coors,
			const Vector< InputType3, reference, CoorsType > * const y_vector,
			const InputType3 * __restrict__ y,
			const CoorsType * const y_coors,
			const size_t n,
			const Ring &ring
		) {
			static_assert( !y_zero || y_scalar, "If y is zero, y_scalar must be true. "
				"Triggering this assertion indicates an incorrect call to this "
				"function; please submit a bug report" );
			const MaskType * __restrict__ m = nullptr;
			const CoorsType * m_coors = nullptr;
			assert( !masked || ( m_vector != nullptr ) );
			if( masked ) {
				m = internal::getRaw( *m_vector );
				m_coors = &( internal::getCoordinates( *m_vector ) );
			}
			assert( !masked || ( m_coors != nullptr ) );
			assert( !a_scalar || ( a_coors == nullptr ) );
			assert( !x_scalar || ( x_coors == nullptr ) );
			assert( !y_scalar || ( y_coors == nullptr ) );

			// check whether we are in the sparse or dense case
			constexpr bool dense = (descr & descriptors::dense);
			const bool mask_is_dense = !masked || (
				(descr & descriptors::structural) &&
				!(descr & descriptors::invert_mask) &&
				m_coors->nonzeroes() == n
			);
			const size_t z_nns = nnz( z_vector );

			// the below Boolean shall be true only if the inputs a, x, and y generate
			// a dense output vector. It furthermore shall be set to false only if the
			// output vector was either empty or fully dense. This is done to determine
			// the exact case the dense variant of the eWiseMulAdd implementations can
			// be used.
			const bool sparse = ( a_scalar ? false : ( a_coors->nonzeroes() < n ) ) ||
				( x_scalar ? false : ( x_coors->nonzeroes() < n ) ) ||
				( y_scalar ? false : ( y_coors->nonzeroes() < n ) ) ||
				( z_nns > 0 && z_nns < n ) ||
				( masked && !mask_is_dense );
			assert( !(sparse && dense) );

			// pre-assign coors if output is dense but was previously totally empty
			const bool assign_z = z_nns == 0 && !sparse;
			if( assign_z ) {
#ifdef _DEBUG
				std::cout << "\teWiseMulAdd_dispatch: detected output will be dense while "
					<< "the output vector presently is completely empty. We therefore "
					<< "pre-assign all output coordinates\n";
#endif
				internal::getCoordinates( z_vector ).assignAll();
			}

			if( !dense && sparse ) {
				// the below computes loop sizes multiplied with the number of vectors that
				// each loop needs to touch. Possible vectors are: z, m, a, x, and y.
				const size_t mask_factor = masked ? 1 : 0;
				const size_t mul_loop_size = ( 3 + mask_factor ) * std::min(
						( a_scalar ? n : a_coors->nonzeroes() ),
						( x_scalar ? n : x_coors->nonzeroes() )
					) + ( 2 + mask_factor ) * ( y_scalar ? n : y_coors->nonzeroes() );
				/** See internal issue #42 (closed): this variant, in a worst-case analysis
				 * is never preferred:
				const size_t add_loop_size = (4 + mask_factor) *
				        (y_scalar ? n : y_coors->nonzeroes()) +
				    (4 + mask_factor) * std::min(
				        (a_scalar ? n : a_coors->nonzeroes()),
				        (x_scalar ? n : x_coors->nonzeroes())
				    ) // min is worst-case, best case is 0, realistic is some a priori unknown
				      // problem-dependent overlap
				std::cout << "\t\teWiseMulAdd_dispatch: add_loop_size = " << add_loop_size
					<< "\n";
				;*/
#ifdef _DEBUG
				std::cout << "\t\teWiseMulAdd_dispatch: mul_loop_size = " << mul_loop_size
					<< "\n";
#endif
				if( masked ) {
					const size_t mask_loop_size = 5 * m_coors->nonzeroes();
#ifdef _DEBUG
					std::cout << "\t\teWiseMulAdd_dispatch: mask_loop_size= "
						<< mask_loop_size << "\n";
#endif
					// internal issue #42 (closed):
					// if( mask_loop_size < mul_loop_size && mask_loop_size < add_loop_size ) {
					if( mask_loop_size < mul_loop_size ) {
#ifdef _DEBUG
						std::cout << "\teWiseMulAdd_dispatch: will be driven by output mask\n";
#endif
						return sparse_eWiseMulAdd_maskDriven<
							descr, a_scalar, x_scalar, y_scalar, y_zero
						>( z_vector, m, m_coors, a, a_coors, x, x_coors, y, y_coors, n, ring );
					}
				}
				// internal issue #42 (closed), see also above:
				// if( mul_loop_size < add_loop_size ) {
#ifdef _DEBUG
				std::cout << "\teWiseMulAdd_dispatch: will be driven by the multiplication a*x\n";
#endif
				static_assert( !(a_scalar && x_scalar),
					"The case of the multiplication being between two scalars should have"
					"been caught earlier. Please submit a bug report." );
				if( a_scalar ) {
					return twoPhase_sparse_eWiseMulAdd_mulDriven<
						descr, masked, a_scalar, y_scalar, y_zero, true
					>( z_vector, m_vector, x, *x_coors, a, a_coors, y_vector, y, n, ring );
				} else if( x_scalar ) {
					return twoPhase_sparse_eWiseMulAdd_mulDriven<
						descr, masked, x_scalar, y_scalar, y_zero, false
					>( z_vector, m_vector, a, *a_coors, x, x_coors, y_vector, y, n, ring );
				} else if( a_coors->nonzeroes() <= x_coors->nonzeroes() ) {
					return twoPhase_sparse_eWiseMulAdd_mulDriven<
						descr, masked, x_scalar, y_scalar, y_zero, false
					>( z_vector, m_vector, a, *a_coors, x, x_coors, y_vector, y, n, ring );
				} else {
					assert( x_coors->nonzeroes() < a_coors->nonzeroes() );
					return twoPhase_sparse_eWiseMulAdd_mulDriven<
						descr, masked, a_scalar, y_scalar, y_zero, true
					>( z_vector, m_vector, x, *x_coors, a, a_coors, y_vector, y, n, ring );
				}
				/* internal issue #42 (closed), see also above:
				} else {
#ifdef _DEBUG
				    std::cout << "\teWiseMulAdd_dispatch: will be driven by the addition with y\n";
#endif
				    if( assign_z ) {
				        return twoPhase_sparse_eWiseMulAdd_addPhase1<
						descr, masked, a_scalar, x_scalar, y_scalar, true
					>(
				            z_vector,
				            m, m_coors,
				            a, a_coors,
				            x, x_coors,
				            y, y_coors,
				            n, ring
				        );
				    } else {
				        return twoPhase_sparse_eWiseMulAdd_addPhase1<
						descr, masked, a_scalar, x_scalar, y_scalar, false
					>(
				            z_vector,
				            m, m_coors,
				            a, a_coors,
				            x, x_coors,
				            y, y_coors,
				            n, ring
				        );
				    }
				}*/
			}

			// all that remains is the dense case
			assert( a_scalar || a_coors->nonzeroes() == n );
			assert( x_scalar || x_coors->nonzeroes() == n );
			assert( y_scalar || y_coors->nonzeroes() == n );
			assert( ! masked || mask_is_dense );
			assert( internal::getCoordinates( z_vector ).nonzeroes() == n );
#ifdef _DEBUG
			std::cout << "\teWiseMulAdd_dispatch: will perform a dense eWiseMulAdd\n";
#endif
			if( assign_z ) {
				return dense_eWiseMulAdd<
					descr, a_scalar, x_scalar, y_scalar, y_zero, true
				>( z_vector, a, x, y, n, ring );
			} else {
				return dense_eWiseMulAdd<
					descr, a_scalar, x_scalar, y_scalar, y_zero, false
				>( z_vector, a, x, y, n, ring );
			}
		}

	} // namespace internal

	/**
	 * Calculates the axpy, \f$ z = a * x .+ y \f$, under this semiring.
	 *
	 * Specialisation for when \a a is a scalar.
	 *
	 * \warning This function has been deprecated since version 0.5. If required,
	 *          consider instead a sequence of grb::foldl using the additive
	 *          monoid, followed by a call to grb::eWiseMul.
	 */
	template<
		Descriptor descr = descriptors::no_operation,
		class Ring,
		typename InputType1, typename InputType2,
		typename InputType3, typename OutputType,
		typename Coords
	>
	RC eWiseMulAdd(
		Vector< OutputType, reference, Coords > &z,
		const InputType1 alpha,
		const Vector< InputType2, reference, Coords > &x,
		const Vector< InputType3, reference, Coords > &y,
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
			"called with a right-hand vector _x with an element type that does not "
			"match the second domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Ring::D4, InputType3 >::value ),
			"grb::eWiseMulAdd",
			"called with an additive vector _y with an element type that does not "
			"match the fourth domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Ring::D4, OutputType >::value ),
			"grb::eWiseMulAdd",
			"called with a result vector _z with an element type that does not match "
			"the fourth domain of the given semiring" );

		// dynamic sanity checks
		const size_t n = size( z );
		if( size( x ) != n || size( y ) != n ) {
			return MISMATCH;
		}
		if( descr & descriptors::dense ) {
			if( nnz( x ) < n ) { return ILLEGAL; }
			if( nnz( y ) < n ) { return ILLEGAL; }
			if( nnz( z ) < n ) { return ILLEGAL; }
		}

		// catch trivial phase
		if( phase == RESIZE ) {
			return SUCCESS;
		}

		// catch trivial dispatches
		assert( phase == EXECUTE );
		const InputType1 zeroIT1 = ring.template getZero< InputType1 >();
		if( alpha == zeroIT1 || nnz( x ) == 0 ) {
			return foldl< descr >( z, y, ring.getAdditiveMonoid() );
		}
		if( nnz( y ) == 0 ) {
			return eWiseMulAdd< descr >(
				z, alpha, x,
				ring.template getZero< typename Ring::D4 >(),
				ring
			);
		}

		// check for density
		const Vector< bool, reference, Coords > * const null_mask = nullptr;
		auto null_coors = &( internal::getCoordinates( x ) );
		null_coors = nullptr;
		constexpr bool maybe_sparse = !(descr & descriptors::dense);
		if( maybe_sparse ) {
			// check whether all inputs are actually dense
			if( nnz( z ) == n && nnz( x ) == n && nnz( y ) == n ) {
				// yes, dispatch to version with dense descriptor set
				return internal::eWiseMulAdd_dispatch<
					descr | descriptors::dense, false, true, false, false, false
				>(
					z, null_mask, &alpha, null_coors,
					internal::getRaw( x ), &( internal::getCoordinates( x ) ),
					&y, internal::getRaw( y ), &( internal::getCoordinates( y ) ),
					n, ring
				);
			}
		}

		// sparse or dense case
		return internal::eWiseMulAdd_dispatch<
			descr, false, true, false, false, false
		>(
			z, null_mask, &alpha, null_coors,
			internal::getRaw( x ), &( internal::getCoordinates( x ) ),
			&y, internal::getRaw( y ), &( internal::getCoordinates( y ) ),
			n, ring
		);
	}

	/**
	 * Calculates the elementwise multiply-add, \f$ z = a .* x .+ y \f$, under
	 * this semiring.
	 *
	 * Specialisation for when \a x is a scalar.
	 *
	 * \warning This function has been deprecated since version 0.5. If required,
	 *          consider instead a sequence of grb::foldl using the additive
	 *          monoid, followed by a call to grb::eWiseMul.
	 */
	template<
		Descriptor descr = descriptors::no_operation,
		class Ring,
		typename InputType1, typename InputType2,
		typename InputType3, typename OutputType,
		typename Coords
	>
	RC eWiseMulAdd(
		Vector< OutputType, reference, Coords > &z,
		const Vector< InputType1, reference, Coords > &a,
		const InputType2 chi,
		const Vector< InputType3, reference, Coords > &y,
		const Ring &ring = Ring(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if< !grb::is_object< OutputType >::value &&
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
			"called with a right-hand vector _x with an element type that does not "
			"match the second domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Ring::D4, InputType3 >::value ),
			"grb::eWiseMulAdd",
			"called with an additive vector _y with an element type that does not "
			"match the fourth domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Ring::D4, OutputType >::value ),
			"grb::eWiseMulAdd",
			"called with a result vector _z with an element type that does not match "
			"the fourth domain of the given semiring" );

		// dynamic sanity checks
		const size_t n = size( z );
		if( size( a ) != n || size( y ) != n ) {
			return MISMATCH;
		}
		if( descr & descriptors::dense ) {
			if( nnz( a ) < n ) { return ILLEGAL; }
			if( nnz( y ) < n ) { return ILLEGAL; }
			if( nnz( z ) < n ) { return ILLEGAL; }
		}

		// catch trivial phase
		if( phase == RESIZE ) {
			return SUCCESS;
		}
		assert( phase == EXECUTE );

		// catch trivial dispatches
		const InputType1 zeroIT2 = ring.template getZero< InputType2 >();
		if( chi == zeroIT2 || nnz( a ) == 0 ) {
			return foldl< descr >( z, y, ring.getAdditiveMonoid() );
		}
		if( nnz( y ) == 0 ) {
			return eWiseMulAdd< descr >(
				z, a, chi,
				ring.template getZero< typename Ring::D4 >(),
				ring
			);
		}

		// check for density
		const Vector< bool, reference, Coords > * const null_mask = nullptr;
		auto null_coors = &( internal::getCoordinates( a ) );
		null_coors = nullptr;
		constexpr bool maybe_sparse = !(descr & descriptors::dense);
		if( maybe_sparse ) {
			// check whether all inputs are actually dense
			if( nnz( z ) == n && nnz( a ) == n && nnz( y ) == n ) {
				// yes, dispatch to version with dense descriptor set
				return internal::eWiseMulAdd_dispatch<
					descr | descriptors::dense, false, false, true, false, false
				>(
					z, null_mask,
					internal::getRaw( a ), &( internal::getCoordinates( a ) ),
					&chi, null_coors,
					&y, internal::getRaw( y ), &( internal::getCoordinates( y ) ),
					n, ring
				);
			}
		}

		// sparse or dense case
		return internal::eWiseMulAdd_dispatch<
			descr, false, false, true, false, false
		>(
			z, null_mask,
			internal::getRaw( a ), &( internal::getCoordinates( a ) ),
			&chi, null_coors,
			&y, internal::getRaw( y ), &( internal::getCoordinates( y ) ),
			n, ring
		);
	}

	/**
	 * Calculates the axpy, \f$ z = a * x .+ y \f$, under this semiring.
	 *
	 * Specialisation for when \a y is a scalar.
	 *
	 * \warning This function has been deprecated since version 0.5. If required,
	 *          consider instead a sequence of grb::foldl using the additive
	 *          monoid, followed by a call to grb::eWiseMul.
	 */
	template<
		Descriptor descr = descriptors::no_operation,
		bool y_zero,
		class Ring,
		typename InputType1, typename InputType2,
		typename InputType3, typename OutputType,
		typename Coords
	>
	RC eWiseMulAdd(
		Vector< OutputType, reference, Coords > &z,
		const Vector< InputType1, reference, Coords > &a,
		const Vector< InputType2, reference, Coords > &x,
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
			"called with a right-hand vector _x with an element type that does not "
			"match the second domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Ring::D4, InputType3 >::value ),
			"grb::eWiseMulAdd",
			"called with an additive vector _y with an element type that does not "
			"match the fourth domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Ring::D4, OutputType >::value ),
			"grb::eWiseMulAdd",
			"called with a result vector _z with an element type that does not match "
			"the fourth domain of the given semiring" );

		// dynamic sanity checks
		const size_t n = size( z );
		if( size( a ) != n || size( x ) != n ) {
			return MISMATCH;
		}
		if( descr & descriptors::dense ) {
			if( nnz( a ) < n ) { return ILLEGAL; }
			if( nnz( x ) < n ) { return ILLEGAL; }
			if( nnz( z ) < n ) { return ILLEGAL; }
		}

		// catch trivial phase
		if( phase == RESIZE ) {
			return SUCCESS;
		}
		assert( phase == EXECUTE );

		// catch trivial dispatches
		const InputType3 zeroIT3 = ring.template getZero< InputType3 >();
		if( nnz( a ) == 0 || nnz( x ) == 0 ) {
			if( gamma == zeroIT3 ) {
				return SUCCESS;
			} else {
				return foldl< descr >( z, gamma, ring.getAdditiveMonoid() );
			}
		}

		// check for density
		const Vector< bool, reference, Coords > * const null_mask = nullptr;
		const Vector< InputType3, reference, Coords > * const null_y = nullptr;
		auto null_coors = &( internal::getCoordinates( a ) );
		null_coors = nullptr;
		constexpr bool maybe_sparse = !(descr & descriptors::dense);
		if( maybe_sparse ) {
			// check whether all inputs are actually dense
			if( nnz( z ) == n && nnz( a ) == n && nnz( x ) == n ) {
				// yes, dispatch to version with dense descriptor set
				return internal::eWiseMulAdd_dispatch<
					descr | descriptors::dense, false, false, false, true, y_zero
				>(
					z, null_mask,
					internal::getRaw( a ), &( internal::getCoordinates( a ) ),
					internal::getRaw( x ), &( internal::getCoordinates( x ) ),
					null_y, &gamma, null_coors, n, ring
				);
			}
		}

		// sparse or dense case
		return internal::eWiseMulAdd_dispatch<
			descr, false, false, false, true, y_zero
		>(
			z, null_mask,
			internal::getRaw( a ), &( internal::getCoordinates( a ) ),
			internal::getRaw( x ), &( internal::getCoordinates( x ) ),
			null_y, &gamma, null_coors, n, ring
		);
	}

	/**
	 * Calculates the axpy, \f$ z = a * x .+ y \f$, under this semiring.
	 *
	 * Specialisation for when \a x and \a y are scalar.
	 *
	 * \warning This function has been deprecated since version 0.5. If required,
	 *          consider instead a sequence of grb::foldl using the additive
	 *          monoid, followed by a call to grb::eWiseMul.
	 */
	template<
		Descriptor descr = descriptors::no_operation,
		bool y_zero = false,
		class Ring,
		typename InputType1, typename InputType2,
		typename InputType3, typename OutputType,
		typename Coords
	>
	RC eWiseMulAdd(
		Vector< OutputType, reference, Coords > &z,
		const Vector< InputType1, reference, Coords > &a,
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
		// static sanity checks
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Ring::D1, InputType1 >::value ),
			"grb::eWiseMulAdd",
			"called with a left-hand scalar alpha of an element type that does not "
			"match the first domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Ring::D2, InputType2 >::value ),
			"grb::eWiseMulAdd",
			"called with a right-hand vector _x with an element type that does not "
			"match the second domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Ring::D4, InputType3 >::value ),
			"grb::eWiseMulAdd",
			"called with an additive vector _y with an element type that does not "
			"match the fourth domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Ring::D4, OutputType >::value ),
			"grb::eWiseMulAdd",
			"called with a result vector _z with an element type that does not match "
			"the fourth domain of the given semiring" );

		// dynamic sanity checks
		const size_t n = size( z );
		if( size( a ) != n ) {
			return MISMATCH;
		}
		if( descr & descriptors::dense ) {
			if( nnz( z ) < n ) { return ILLEGAL; }
			if( nnz( a ) < n ) { return ILLEGAL; }
		}

		if( phase == RESIZE ) {
			return SUCCESS;
		}
		assert( phase == EXECUTE );

		// catch trivial dispatches
		const InputType2 zeroIT2 = ring.template getZero< InputType2 >();
		const InputType3 zeroIT3 = ring.template getZero< InputType3 >();
		if( nnz( a ) == 0 || beta == zeroIT2 ) {
			return foldl< descr >( z, gamma, ring.getAdditiveMonoid() );
		}

		// check for density
		Vector< bool, reference, Coords > * const null_mask = nullptr;
		Vector< InputType3, reference, Coords > * const null_y = nullptr;
		auto null_coors = &( internal::getCoordinates( a ) );
		null_coors = nullptr;
		constexpr bool maybe_sparse = !(descr & descriptors::dense);
		if( maybe_sparse ) {
			// check whether all inputs are actually dense
			if( nnz( z ) == n && nnz( a ) == n ) {
				// yes, dispatch to version with dense descriptor set
				return internal::eWiseMulAdd_dispatch<
					descr | descriptors::dense, false, false, true, true, y_zero
				>(
					z, null_mask,
					internal::getRaw( a ), &( internal::getCoordinates( a ) ),
					&beta, null_coors, null_y, &gamma, null_coors, n, ring
				);
			}
		}

		// sparse or dense case
		return internal::eWiseMulAdd_dispatch<
			descr, false, false, true, true, y_zero
		>(
			z, null_mask,
			internal::getRaw( a ), &( internal::getCoordinates( a ) ),
			&beta, null_coors, null_y, &gamma, null_coors, n, ring
		);
	}

	/**
	 * Calculates the axpy, \f$ z = a * x .+ y \f$, under this semiring.
	 *
	 * Specialisation for when \a a and \a y are scalar.
	 *
	 * \warning This function has been deprecated since version 0.5. If required,
	 *          consider instead a sequence of grb::foldl using the additive
	 *          monoid, followed by a call to grb::eWiseMul.
	 */
	template<
		Descriptor descr = descriptors::no_operation,
		bool y_zero,
		class Ring,
		typename InputType1, typename InputType2,
		typename InputType3, typename OutputType,
		typename Coords
	>
	RC eWiseMulAdd(
		Vector< OutputType, reference, Coords > &z,
		const InputType1 alpha,
		const Vector< InputType2, reference, Coords > &x,
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
			"called with a right-hand vector _x with an element type that does not "
			"match the second domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Ring::D4, InputType3 >::value ),
			"grb::eWiseMulAdd",
			"called with an additive vector _y with an element type that does not "
			"match the fourth domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Ring::D4, OutputType >::value ),
			"grb::eWiseMulAdd",
			"called with a result vector _z with an element type that does not match "
			"the fourth domain of the given semiring" );

		// dynamic sanity checks
		const size_t n = size( z );
		if( size( x ) != n ) {
			return MISMATCH;
		}
		if( descr & descriptors::dense ) {
			if( nnz( z ) < n ) { return ILLEGAL; }
			if( nnz( x ) < n ) { return ILLEGAL; }
		}

		// catch trivial phase
		if( phase == RESIZE ) {
			return SUCCESS;
		}
		assert( phase == EXECUTE );

		// catch trivial dispatches
		const InputType1 zeroIT1 = ring.template getZero< InputType1 >();
		if( nnz( x ) == 0 || alpha == zeroIT1 ) {
			return foldl< descr >( z, gamma, ring.getAdditiveMonoid() );
		}

		// check for density
		const Vector< bool, reference, Coords > * null_mask = nullptr;
		const Vector< InputType3, reference, Coords > * null_y = nullptr;
		auto null_coors = &( internal::getCoordinates( x ) );
		null_coors = nullptr;
		constexpr bool maybe_sparse = !(descr & descriptors::dense);
		if( maybe_sparse ) {
			// check whether all inputs are actually dense
			if( nnz( z ) == n && nnz( x ) == n ) {
				// yes, dispatch to version with dense descriptor set
				return internal::eWiseMulAdd_dispatch<
					descr | descriptors::dense, false, true, false, true, y_zero
				>(
					z, null_mask, &alpha, null_coors,
					internal::getRaw( x ), &( internal::getCoordinates( x ) ),
					null_y, &gamma, null_coors, n, ring
				);
			}
		}

		// sparse or dense case
		return internal::eWiseMulAdd_dispatch<
			descr, false, true, false, true, y_zero
		>(
			z, null_mask, &alpha, null_coors,
			internal::getRaw( x ), &( internal::getCoordinates( x ) ),
			null_y, &gamma, null_coors, n, ring
		);
	}

	/**
	 * Calculates the axpy, \f$ z = a * x .+ y \f$, under this semiring.
	 *
	 * Specialisation for when \a a and \a x are scalar.
	 *
	 * \internal Dispatches to eWiseAdd.
	 *
	 * \warning This function has been deprecated since version 0.5. If required,
	 *          consider instead a sequence of grb::foldl using the additive
	 *          monoid, followed by a call to grb::eWiseMul.
	 */
	template<
		Descriptor descr = descriptors::no_operation, class Ring,
		typename OutputType, typename InputType1,
		typename InputType2, typename InputType3,
		typename Coords
	>
	RC eWiseMulAdd(
		Vector< OutputType, reference, Coords > &z,
		const InputType1 alpha,
		const InputType2 beta,
		const Vector< InputType3, reference, Coords > &y,
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
		std::cout << "eWiseMulAdd (reference, vector <- scalar x scalar + vector) "
			<< "precomputes scalar multiply and dispatches to eWiseAdd (reference, "
			<< "vector <- scalar + vector)\n";
#endif
		// dynamic checks
		const size_t n = size( z );
		if( size( y ) != n ) { return MISMATCH; }
		if( descr & descriptors::dense ) {
			if( nnz( z ) < n ) { return ILLEGAL; }
			if( nnz( y ) < n ) { return ILLEGAL; }
		}
		typename Ring::D3 mul_result;
		RC rc = grb::apply(
			mul_result,
			alpha, beta,
			ring.getMultiplicativeOperator()
		);
#ifdef NDEBUG
		(void) rc;
#else
		assert( rc == SUCCESS );
#endif
		return eWiseAdd< descr >( z, mul_result, y, ring, phase );
	}

	/**
	 * Calculates the axpy, \f$ z = a * x .+ y \f$, under this semiring.
	 *
	 * Specialisation for when \a a, \a x, and \a y are scalar.
	 *
	 * \internal Dispatches to set.
	 *
	 * \warning This function has been deprecated since version 0.5. If required,
	 *          consider instead a sequence of grb::foldl using the additive
	 *          monoid, followed by a call to grb::eWiseMul.
	 */
	template<
		Descriptor descr = descriptors::no_operation, class Ring,
		typename OutputType, typename InputType1,
		typename InputType2, typename InputType3,
		typename Coords
	>
	RC eWiseMulAdd(
		Vector< OutputType, reference, Coords > &z,
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
		std::cout << "eWiseMulAdd (reference, vector <- scalar x scalar + scalar) "
			<< "precomputes scalar operations and dispatches to set (reference)\n";
#endif
		// dynamic sanity checks
		if( descr & descriptors::dense ) {
			if( nnz( z ) < size( z ) ) { return ILLEGAL; }
		}

		typename Ring::D3 mul_result;
		RC rc = grb::apply( mul_result, alpha, beta,
			ring.getMultiplicativeOperator() );
#ifdef NDEBUG
		(void) rc;
#endif
		assert( rc == SUCCESS );
		typename Ring::D4 add_result;
		rc = grb::apply( add_result, mul_result, gamma,
			ring.getAdditiveOperator() );
#ifdef NDEBUG
		(void) rc;
#endif
		assert( rc == SUCCESS );
		return grb::foldl< descr >( z, add_result, phase );
	}

	/**
	 * Calculates the elementwise multiply-add, \f$ z = a .* x .+ y \f$, under
	 * this semiring.
	 *
	 * Any combination of \a a, \a x, and \a y may be a scalar. Any scalars equal
	 * to the given semiring's zero will be detected and automatically be
	 * transformed into calls to eWiseMul, eWiseAdd, and so on.
	 *
	 * @tparam descr      The descriptor to be used (descriptors::no_operation
	 *                    if left unspecified).
	 * @tparam Ring       The semiring type to perform the element-wise
	 *                    multiply-add on.
	 * @tparam InputType1 The left-hand side input type to the multiplicative
	 *                    operator of the \a ring.
	 * @tparam InputType2 The right-hand side input type to the multiplicative
	 *                    operator of the \a ring.
	 * @tparam InputType3 The output type to the multiplicative operator of the
	 *                    \a ring \em and the left-hand side input type to the
	 *                    additive operator of the \a ring.
	 * @tparam OutputType The right-hand side input type to the additive operator
	 *                    of the \a ring \em and the result type of the same
	 *                    operator.
	 *
	 * @param[out] _z  The pre-allocated output vector.
	 * @param[in]  _a  The elements for left-hand side multiplication.
	 * @param[in]  _x  The elements for right-hand side multiplication.
	 * @param[in]  _y  The elements for right-hand size addition.
	 * @param[in] ring The ring to perform the eWiseMulAdd under.
	 *
	 * @return grb::MISMATCH Whenever the dimensions of \a _a, \a _x, \a _y, and
	 *                       \a z do not match. In this case, all input data
	 *                       containers are left untouched and it will simply be
	 *                       as though this call was never made.
	 * @return grb::SUCCESS  On successful completion of this call.
	 *
	 * \warning An implementation is not obligated to detect overlap whenever
	 *          it occurs. If part of \a z overlaps with \a x, \a y, or \a a,
	 *          undefined behaviour will occur \em unless this function returns
	 *          grb::OVERLAP. In other words: an implementation which returns
	 *          erroneous results when vectors overlap and still returns
	 *          grb::SUCCESS thus is also a valid GraphBLAS implementation!
	 *
	 * \parblock
	 * \par Valid descriptors
	 * grb::descriptors::no_operation, grb::descriptors::no_casting.
	 *
	 * \note Invalid descriptors will be ignored.
	 *
	 * If grb::descriptors::no_casting is specified, then 1) the first domain of
	 * \a ring must match \a InputType1, 2) the second domain of \a ring must match
	 * \a InputType2, 3) the third domain of \a ring must match \a InputType3,
	 * 4) the fourth domain of \a ring must match \a OutputType. If one of these is
	 * not true, the code shall not compile.
	 * \endparblock
	 *
	 * \parblock
	 * \par Performance semantics
	 *      -# This call takes \f$ \Theta(n) \f$ work, where \f$ n \f$ equals the
	 *         size of the vectors \a _a, \a _x, \a _y, and \a _z. The constant
	 *         factor depends on the cost of evaluating the addition and
	 *         multiplication operators. A good implementation uses vectorised
	 *         instructions whenever the input domains, the output domain, and
	 *         the operators used allow for this.
	 *
	 *      -# This call takes \f$ \mathcal{O}(1) \f$ memory beyond the memory
	 *         already used by the application when this function is called.
	 *
	 *      -# This call incurs at most \f$ n( \mathit{sizeof}(
	 *           \mathit{InputType1} + \mathit{bool}
	 *           \mathit{InputType2} + \mathit{bool}
	 *           \mathit{InputType3} + \mathit{bool}
	 *           \mathit{OutputType} + \mathit{bool}
	 *         ) + \mathcal{O}(1) \f$
	 *         bytes of data movement. A good implementation will stream \a _a,
	 *         \a _x or \a _y into \a _z to apply the additive and multiplicative
	 *         operators in-place, whenever the input domains, the output domain,
	 *         and the operators used allow for this.
	 * \endparblock
	 *
	 * \warning This function has been deprecated since version 0.5. If required,
	 *          consider instead a sequence of grb::foldl using the additive
	 *          monoid, followed by a call to grb::eWiseMul.
	 */
	template<
		Descriptor descr = descriptors::no_operation,
		class Ring,
		typename InputType1, typename InputType2,
		typename InputType3, typename OutputType,
		typename Coords
	>
	RC eWiseMulAdd(
		Vector< OutputType, reference, Coords > &z,
		const Vector< InputType1, reference, Coords > &a,
		const Vector< InputType2, reference, Coords > &x,
		const Vector< InputType3, reference, Coords > &y,
		const Ring &ring = Ring(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if< !grb::is_object< OutputType >::value &&
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
			"called with a left-hand vector _a with an element type that does not "
			"match the first domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Ring::D2, InputType2 >::value ),
			"grb::eWiseMulAdd",
			"called with a right-hand vector _x with an element type that does not "
			"match the second domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Ring::D4, InputType3 >::value ),
			"grb::eWiseMulAdd",
			"called with an additive vector _y with an element type that does not "
			"match the fourth domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Ring::D4, OutputType >::value ),
			"grb::eWiseMulAdd",
			"called with a result vector _z with an element type that does not match "
			"the fourth domain of the given semiring" );

		// dynamic sanity checks
		const size_t n = size( z );
		if( size( x ) != n || size( y ) != n || size( a ) != n ) {
			return MISMATCH;
		}
		if( descr & descriptors::dense ) {
			if( nnz( z ) < n ) { return ILLEGAL; }
			if( nnz( x ) < n ) { return ILLEGAL; }
			if( nnz( y ) < n ) { return ILLEGAL; }
			if( nnz( a ) < n ) { return ILLEGAL; }
		}

		// catch trivial phase
		if( phase == RESIZE ) {
			return SUCCESS;
		}
		assert( phase == EXECUTE );


		// catch trivial dispatches
		if( nnz( a ) == 0 || nnz( x ) == 0 ) {
			return foldr< descr >( y, z, ring.getAdditiveMonoid(), phase );
		}
		if( nnz( y ) == 0 ) {
			return eWiseMulAdd< descr >(
				z, a, x,
				ring.template getZero< typename Ring::D4 >(),
				ring, phase
			);
		}

		const Vector< bool, reference, Coords > * const null_mask = nullptr;
		constexpr bool maybe_sparse = !(descr & descriptors::dense);
		if( maybe_sparse ) {
			// check for dense variant
			if( nnz( z ) == n && nnz( x ) == n && nnz( y ) == n && nnz( a ) == n ) {
				// yes, dispatch to version with dense descriptor set
				return internal::eWiseMulAdd_dispatch<
					descr | descriptors::dense, false, false, false, false, false
				>(
					z, null_mask,
					internal::getRaw( a ), &( internal::getCoordinates( a ) ),
					internal::getRaw( x ), &( internal::getCoordinates( x ) ),
					&y, internal::getRaw( y ), &( internal::getCoordinates( y ) ),
					n, ring
				);
			}
		}
		return internal::eWiseMulAdd_dispatch<
			descr, false, false, false, false, false
		>(
			z, null_mask,
			internal::getRaw( a ), &( internal::getCoordinates( a ) ),
			internal::getRaw( x ), &( internal::getCoordinates( x ) ),
			&y, internal::getRaw( y ), &( internal::getCoordinates( y ) ),
			n, ring
		);
	}

	/**
	 * Calculates the axpy, \f$ z = a * x .+ y \f$, under this semiring.
	 *
	 * Specialisation for when \a a is a scalar, masked version.
	 *
	 * \warning This function has been deprecated since version 0.5. If required,
	 *          consider instead a sequence of grb::foldl using the additive
	 *          monoid, followed by a call to grb::eWiseMul.
	 */
	template<
		Descriptor descr = descriptors::no_operation,
		class Ring,
		typename InputType1, typename InputType2, typename InputType3,
		typename OutputType, typename MaskType,
		typename Coords
	>
	RC eWiseMulAdd(
		Vector< OutputType, reference, Coords > &z,
		const Vector< MaskType, reference, Coords > &m,
		const InputType1 alpha,
		const Vector< InputType2, reference, Coords > &x,
		const Vector< InputType3, reference, Coords > &y,
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
			"called with a right-hand vector _x with an element type that does not "
			"match the second domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Ring::D4, InputType3 >::value ),
			"grb::eWiseMulAdd",
			"called with an additive vector _y with an element type that does not "
			"match the fourth domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Ring::D4, OutputType >::value ),
			"grb::eWiseMulAdd",
			"called with a result vector _z with an element type that does not match "
			"the fourth domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< bool, MaskType >::value ),
			"grb::eWiseMulAdd",
			"called with a mask vector _m with a non-bool element type" );

		// catch empty mask
		if( size( m ) == 0 ) {
			return eWiseMulAdd< descr >( z, alpha, x, y, ring, phase );
		}

		// dynamic sanity checks
		const size_t n = size( z );
		if( size( x ) != n || size( y ) != n || size( m ) != n ) {
			return MISMATCH;
		}
		if( descr & descriptors::dense ) {
			if( nnz( z ) < n ) { return ILLEGAL; }
			if( nnz( x ) < n ) { return ILLEGAL; }
			if( nnz( y ) < n ) { return ILLEGAL; }
			if( nnz( m ) < n ) { return ILLEGAL; }
		}

		// catch trivial phase
		if( phase == RESIZE ) {
			return SUCCESS;
		}
		assert( phase == EXECUTE );

		// catch trivial dispatches
		const InputType1 zeroIT1 = ring.template getZero< InputType1 >();
		if( alpha == zeroIT1 || nnz( x ) == 0 ) {
			return foldl< descr >( z, m, y, ring.getAdditiveMonoid() );
		}
		if( nnz( y ) == 0 ) {
			return eWiseMulAdd< descr >(
				z, m, alpha, x,
				ring.template getZero< typename Ring::D4 >(),
				ring
			);
		}

		// check for density
		const Vector< bool, reference, Coords > * const null_mask = nullptr;
		auto null_coors = &( internal::getCoordinates( x ) );
		null_coors = nullptr;
		constexpr bool maybe_sparse = !(descr & descriptors::dense);
		if( maybe_sparse ) {
			// check whether all inputs are actually dense
			if( nnz( z ) == n && nnz( x ) == n && nnz( y ) == n && (
				nnz( m ) == n &&
				(descr & descriptors::structural) &&
				!(descr & descriptors::invert_mask)
			) ) {
				// yes, dispatch to version with dense descriptor set
				return internal::eWiseMulAdd_dispatch<
					descr | descriptors::dense, false, true, false, false, false
				>(
					z, null_mask, &alpha, null_coors,
					internal::getRaw( x ), &( internal::getCoordinates( x ) ),
					&y, internal::getRaw( y ), &( internal::getCoordinates( y ) ),
					n, ring
				);
			}
		}

		// sparse case
		return internal::eWiseMulAdd_dispatch<
			descr, true, true, false, false, false
		>(
			z, &m, &alpha, null_coors,
			internal::getRaw( x ), &( internal::getCoordinates( x ) ),
			&y, internal::getRaw( y ), &( internal::getCoordinates( y ) ),
			n, ring
		);
	}

	/**
	 * Calculates the elementwise multiply-add, \f$ z = a .* x .+ y \f$, under
	 * this semiring.
	 *
	 * Specialisation for when \a x is a scalar, masked version.
	 *
	 * \warning This function has been deprecated since version 0.5. If required,
	 *          consider instead a sequence of grb::foldl using the additive
	 *          monoid, followed by a call to grb::eWiseMul.
	 */
	template<
		Descriptor descr = descriptors::no_operation,
		class Ring,
		typename InputType1, typename InputType2, typename InputType3,
		typename OutputType, typename MaskType,
		typename Coords
	>
	RC eWiseMulAdd(
		Vector< OutputType, reference, Coords > &z,
		const Vector< MaskType, reference, Coords > &m,
		const Vector< InputType1, reference, Coords > &a,
		const InputType2 chi,
		const Vector< InputType3, reference, Coords > &y,
		const Ring &ring = Ring(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			!grb::is_object< OutputType >::value &&
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
			"called with a right-hand vector _x with an element type that does not "
			"match the second domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Ring::D4, InputType3 >::value ),
			"grb::eWiseMulAdd",
			"called with an additive vector _y with an element type that does not "
			"match the fourth domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Ring::D4, OutputType >::value ),
			"grb::eWiseMulAdd",
			"called with a result vector _z with an element type that does not match "
			"the fourth domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< bool, MaskType >::value ),
			"grb::eWiseMulAdd",
			"called with a mask vector _m with a non-bool element type" );

		// catch empty mask
		if( size( m ) == 0 ) {
			return eWiseMulAdd< descr >( z, a, chi, y, ring, phase );
		}

		// dynamic sanity checks
		const size_t n = size( z );
		if( size( a ) != n || size( y ) != n || size( m ) != n ) {
			return MISMATCH;
		}
		if( descr & descriptors::dense ) {
			if( nnz( z ) < n ) { return ILLEGAL; }
			if( nnz( a ) < n ) { return ILLEGAL; }
			if( nnz( y ) < n ) { return ILLEGAL; }
			if( nnz( m ) < n ) { return ILLEGAL; }
		}

		// catch trivial case
		if( phase == RESIZE ) {
			return SUCCESS;
		}
		assert( phase == EXECUTE );

		// catch trivial dispatches
		const InputType1 zeroIT2 = ring.template getZero< InputType2 >();
		if( chi == zeroIT2 || nnz( a ) == 0 ) {
			return foldl< descr >( z, m, y, ring.getAdditiveMonoid() );
		}
		if( nnz( y ) == 0 ) {
			return eWiseMulAdd< descr >(
				z, m, a, chi,
				ring.template getZero< typename Ring::D4 >(),
				ring
			);
		}

		// check for density
		const Vector< bool, reference, Coords > * const null_mask = nullptr;
		auto null_coors = &( internal::getCoordinates( a ) );
		null_coors = nullptr;
		constexpr bool maybe_sparse = !(descr & descriptors::dense);
		if( maybe_sparse ) {
			// check whether all inputs are actually dense
			if( nnz( z ) == n && nnz( a ) == n && nnz( y ) == n && (
				nnz( m ) == n &&
				(descr & descriptors::structural) &&
				!(descr & descriptors::invert_mask)
			) ) {
				// yes, dispatch to version with dense descriptor set
				return internal::eWiseMulAdd_dispatch<
					descr | descriptors::dense, false, false, true, false, false
				>(
					z, null_mask,
					internal::getRaw( a ), &( internal::getCoordinates( a ) ),
					&chi, null_coors,
					&y, internal::getRaw( y ), &( internal::getCoordinates( y ) ),
					n, ring
				);
			}
		}

		// sparse case
		return internal::eWiseMulAdd_dispatch<
			descr, true, false, true, false, false
		>(
			z, &m,
			internal::getRaw( a ), &( internal::getCoordinates( a ) ),
			&chi, null_coors,
			&y, internal::getRaw( y ), &( internal::getCoordinates( y ) ),
			n, ring
		);
	}

	/**
	 * Calculates the axpy, \f$ z = a * x .+ y \f$, under this semiring.
	 *
	 * Specialisation for when \a y is a scalar, masked version.
	 *
	 * \warning This function has been deprecated since version 0.5. If required,
	 *          consider instead a sequence of grb::foldl using the additive
	 *          monoid, followed by a call to grb::eWiseMul.
	 */
	template<
		Descriptor descr = descriptors::no_operation,
		bool y_zero = false,
		class Ring,
		typename InputType1, typename InputType2, typename InputType3,
		typename OutputType, typename MaskType,
		typename Coords
	>
	RC eWiseMulAdd(
		Vector< OutputType, reference, Coords > &z,
		const Vector< MaskType, reference, Coords > &m,
		const Vector< InputType1, reference, Coords > &a,
		const Vector< InputType2, reference, Coords > &x,
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
			"called with a right-hand vector _x with an element type that does not "
			"match the second domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Ring::D4, InputType3 >::value ),
			"grb::eWiseMulAdd",
			"called with an additive vector _y with an element type that does not "
			"match the fourth domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Ring::D4, OutputType >::value ),
			"grb::eWiseMulAdd",
			"called with a result vector _z with an element type that does not match "
			"the fourth domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< bool, MaskType >::value ),
			"grb::eWiseMulAdd",
			"called with a mask vector _m with a non-bool element type" );

		// catch empty mask
		if( size( m ) == 0 ) {
			return eWiseMulAdd< descr, y_zero >( z, a, x, gamma, ring, phase );
		}

		// dynamic sanity checks
		const size_t n = size( z );
		if( size( a ) != n || size( x ) != n || size( m ) != n ) {
			return MISMATCH;
		}
		if( descr & descriptors::dense ) {
			if( nnz( z ) < n ) { return ILLEGAL; }
			if( nnz( a ) < n ) { return ILLEGAL; }
			if( nnz( x ) < n ) { return ILLEGAL; }
			if( nnz( m ) < n ) { return ILLEGAL; }
		}

		// catch trivial phase
		if( phase == RESIZE ) {
			return SUCCESS;
		}
		assert( phase == EXECUTE );

		// catch trivial dispatches
		const InputType3 zeroIT3 = ring.template getZero< InputType3 >();
		if( nnz( a ) == 0 || nnz( x ) == 0 ) {
			return foldl< descr >( z, m, gamma, ring.getAdditiveMonoid() );
		}

		// check for density
		const Vector< bool, reference, Coords > * const null_mask = nullptr;
		const Vector< InputType3, reference, Coords > * const null_y = nullptr;
		auto null_coors = &( internal::getCoordinates( a ) );
		null_coors = nullptr;
		constexpr bool maybe_sparse = !(descr & descriptors::dense);
		if( maybe_sparse ) {
			// check whether all inputs are actually dense
			if( nnz( z ) == n && nnz( a ) == n && nnz( x ) == n && (
				nnz( m ) == n &&
				(descr & descriptors::structural) &&
				!(descr & descriptors::invert_mask)
			) ) {
				// yes, dispatch to version with dense descriptor set
				return internal::eWiseMulAdd_dispatch<
					descr | descriptors::dense, false, false, false, true, y_zero
				>(
					z, null_mask,
					internal::getRaw( a ), &( internal::getCoordinates( a ) ),
					internal::getRaw( x ), &( internal::getCoordinates( x ) ),
					null_y, &gamma, null_coors,
					n, ring
				);
			}
		}

		// sparse case
		return internal::eWiseMulAdd_dispatch<
			descr, true, false, false, true, y_zero
		>(
			z, &m,
			internal::getRaw( a ), &( internal::getCoordinates( a ) ),
			internal::getRaw( x ), &( internal::getCoordinates( x ) ),
			null_y, &gamma, null_coors,
			n, ring
		);
	}

	/**
	 * Calculates the axpy, \f$ z = a * x .+ y \f$, under this semiring.
	 *
	 * Specialisation for when \a x and \a y are scalar, masked version.
	 *
	 * \warning This function has been deprecated since version 0.5. If required,
	 *          consider instead a sequence of grb::foldl using the additive
	 *          monoid, followed by a call to grb::eWiseMul.
	 */
	template<
		Descriptor descr = descriptors::no_operation,
		bool y_zero = false,
		class Ring,
		typename InputType1, typename InputType2, typename InputType3,
		typename OutputType, typename MaskType,
		typename Coords
	>
	RC eWiseMulAdd(
		Vector< OutputType, reference, Coords > &z,
		const Vector< MaskType, reference, Coords > &m,
		const Vector< InputType1, reference, Coords > &a,
		const InputType2 beta,
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
			"called with a right-hand vector _x with an element type that does not "
			"match the second domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Ring::D4, InputType3 >::value ),
			"grb::eWiseMulAdd",
			"called with an additive vector _y with an element type that does not "
			"match the fourth domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Ring::D4, OutputType >::value ),
			"grb::eWiseMulAdd",
			"called with a result vector _z with an element type that does not match "
			"the fourth domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< bool, MaskType >::value ),
			"grb::eWiseMulAdd",
			"called with a mask vector _m with a non-bool element type" );

		// catch empty mask
		if( size( m ) == 0 ) {
			return eWiseMulAdd< descr, y_zero >( z, a, beta, gamma, ring, phase );
		}

		// dynamic sanity checks
		const size_t n = size( z );
		if( size( a ) != n || size( m ) != n ) {
			return MISMATCH;
		}
		if( descr & descriptors::dense ) {
			if( nnz( z ) < n ) { return ILLEGAL; }
			if( nnz( a ) < n ) { return ILLEGAL; }
			if( nnz( m ) < n ) { return ILLEGAL; }
		}

		// catch trivial phase
		if( phase == RESIZE ) {
			return SUCCESS;
		}
		assert( phase == EXECUTE );

		// catch trivial dispatch
		const InputType2 zeroIT2 = ring.template getZero< InputType2 >();
		if( nnz( a ) || zeroIT2 == beta ) {
			return foldl< descr >( z, m, gamma, ring.getAdditiveMonoid() );
		}

		// check for density
		const Vector< bool, reference, Coords > * null_mask = nullptr;
		const Vector< InputType3, reference, Coords > * null_y = nullptr;
		auto null_coors = &( internal::getCoordinates( a ) );
		null_coors = nullptr;
		constexpr bool maybe_sparse = !(descr & descriptors::dense);
		if( maybe_sparse ) {
			// check whether all inputs are actually dense
			if( nnz( z ) == n && nnz( a ) == n && (
				nnz( m ) == n &&
				(descr & descriptors::structural) &&
				!(descr & descriptors::invert_mask)
			) ) {
				// yes, dispatch to version with dense descriptor set
				return internal::eWiseMulAdd_dispatch<
					descr | descriptors::dense, false, false, true, true, y_zero
				>(
					z, null_mask,
					internal::getRaw( a ), &( internal::getCoordinates( a ) ),
					&beta, null_coors, null_y, &gamma, null_coors,
					n, ring
				);
			}
		}

		// sparse case
		return internal::eWiseMulAdd_dispatch<
			descr, true, false, true, true, y_zero
		>(
			z, &m,
			internal::getRaw( a ), &( internal::getCoordinates( a ) ),
			&beta, null_coors, null_y, &gamma, null_coors,
			n, ring
		);
	}

	/**
	 * Calculates the axpy, \f$ z = a * x .+ y \f$, under this semiring.
	 *
	 * Specialisation for when \a a and \a y are scalar, masked version.
	 *
	 * \warning This function has been deprecated since version 0.5. If required,
	 *          consider instead a sequence of grb::foldl using the additive
	 *          monoid, followed by a call to grb::eWiseMul.
	 */
	template<
		Descriptor descr = descriptors::no_operation,
		bool y_zero = false,
		class Ring,
		typename InputType1, typename InputType2, typename InputType3,
		typename OutputType, typename MaskType,
		typename Coords
	>
	RC eWiseMulAdd(
		Vector< OutputType, reference, Coords > &z,
		const Vector< MaskType, reference, Coords > &m,
		const InputType1 alpha,
		const Vector< InputType2, reference, Coords > &x,
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
			"called with a right-hand vector _x with an element type that does not "
			"match the second domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Ring::D4, InputType3 >::value ),
			"grb::eWiseMulAdd",
			"called with an additive vector _y with an element type that does not "
			"match the fourth domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Ring::D4, OutputType >::value ),
			"grb::eWiseMulAdd",
			"called with a result vector _z with an element type that does not match "
			"the fourth domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< bool, MaskType >::value ),
			"grb::eWiseMulAdd",
			"called with a mask vector _m with a non-bool element type" );

		// catch empty mask
		if( size( m ) == 0 ) {
			return eWiseMulAdd< descr, y_zero >( z, alpha, x, gamma, ring, phase );
		}

		// dynamic sanity checks
		const size_t n = size( z );
		if( size( x ) != n || size( m ) != n ) {
			return MISMATCH;
		}
		if( descr & descriptors::dense ) {
			if( nnz( z ) < n ) { return ILLEGAL; }
			if( nnz( x ) < n ) { return ILLEGAL; }
			if( nnz( m ) < n ) { return ILLEGAL; }
		}

		// catch trivial phase
		if( phase == RESIZE ) {
			return SUCCESS;
		}
		assert( phase == EXECUTE );

		// catch trivial dispatch
		const InputType1 zeroIT1 = ring.template getZero< InputType1 >();
		if( nnz( x ) == 0 || alpha == zeroIT1 ) {
			return foldl< descr >( z, m, gamma, ring.getAdditiveMonoid() );
		}

		// check for density
		const Vector< bool, reference, Coords > * const null_mask = nullptr;
		const Vector< InputType3, reference, Coords > * const null_y = nullptr;
		auto null_coors = &( internal::getCoordinates( x ) );
		null_coors = nullptr;
		constexpr bool maybe_sparse = !(descr & descriptors::dense);
		if( maybe_sparse ) {
			// check whether all inputs are actually dense
			if( nnz( z ) == n && nnz( x ) == n && (
				nnz( m ) == n &&
				(descr & descriptors::structural) &&
				!(descr & descriptors::invert_mask)
			) ) {
				// yes, dispatch to version with dense descriptor set
				return internal::eWiseMulAdd_dispatch<
					descr | descriptors::dense, false, true, false, true, y_zero
				>(
					z, null_mask, &alpha, null_coors,
					internal::getRaw( x ), &( internal::getCoordinates( x ) ),
					null_y, &gamma, null_coors,
					n, ring
				);
			}
		}

		// sparse case
		return internal::eWiseMulAdd_dispatch<
			descr, true, true, false, true, y_zero
		>(
			z, &m, &alpha, null_coors,
			internal::getRaw( x ), &( internal::getCoordinates( x ) ),
			null_y, &gamma, null_coors,
			n, ring
		);
	}

	/**
	 * Calculates the axpy, \f$ z = a * x .+ y \f$, under this semiring.
	 *
	 * Masked version.
	 *
	 * \warning This function has been deprecated since version 0.5. If required,
	 *          consider instead a sequence of grb::foldl using the additive
	 *          monoid, followed by a call to grb::eWiseMul.
	 */
	template<
		Descriptor descr = descriptors::no_operation,
		class Ring,
		typename InputType1, typename InputType2, typename InputType3,
		typename OutputType, typename MaskType,
		typename Coords
	>
	RC eWiseMulAdd(
		Vector< OutputType, reference, Coords > &z,
		const Vector< MaskType, reference, Coords > &m,
		const Vector< InputType1, reference, Coords > &a,
		const Vector< InputType2, reference, Coords > &x,
		const Vector< InputType3, reference, Coords > &y,
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
			"called with a left-hand vector _a with an element type that does not "
			"match the first domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Ring::D2, InputType2 >::value ),
			"grb::eWiseMulAdd",
			"called with a right-hand vector _x with an element type that does not "
			"match the second domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Ring::D4, InputType3 >::value ),
			"grb::eWiseMulAdd",
			"called with an additive vector _y with an element type that does not "
			"match the fourth domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Ring::D4, OutputType >::value ),
			"grb::eWiseMulAdd",
			"called with a result vector _z with an element type that does not match "
			"the fourth domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< bool, MaskType >::value ),
			"grb::eWiseMulAdd",
			"called with a mask vector _m with a non-bool element type" );

		// catch empty mask
		if( size( m ) == 0 ) {
			return eWiseMulAdd< descr >( z, a, x, y, ring, phase );
		}

		// dynamic sanity checks
		const size_t n = size( z );
		if( size( x ) != n || size( y ) != n || size( a ) != n || size( m ) != n ) {
			return MISMATCH;
		}
		if( descr & descriptors::dense ) {
			if( nnz( z ) < n ) { return ILLEGAL; }
			if( nnz( x ) < n ) { return ILLEGAL; }
			if( nnz( y ) < n ) { return ILLEGAL; }
			if( nnz( a ) < n ) { return ILLEGAL; }
			if( nnz( m ) < n ) { return ILLEGAL; }
		}

		// catch trivial phase
		if( phase == RESIZE ) {
			return SUCCESS;
		}
		assert( phase == EXECUTE );

		// catch trivial dispatches
		if( nnz( a ) == 0 || nnz( x ) == 0 ) {
			return foldr< descr >( y, m, z, ring.getAdditiveMonoid(), phase );
		}
		if( nnz( y ) == 0 ) {
			return eWiseMulAdd< descr >(
				z, m, a, x,
				ring.template getZero< typename Ring::D4 >(),
				ring, phase
			);
		}

		const Vector< bool, reference, Coords > * const null_mask = nullptr;
		constexpr bool maybe_sparse = !(descr & descriptors::dense);
		if( maybe_sparse ) {
			// check for dense variant
			if( nnz( z ) == n && nnz( x ) == n && nnz( y ) == n && nnz( a ) == n && (
				nnz( m ) == n &&
				(descr & descriptors::structural) &&
				!(descr & descriptors::invert_mask)
			) ) {
				// yes, dispatch to version with dense descriptor set
				return internal::eWiseMulAdd_dispatch<
					descr | descriptors::dense, false, false, false, false, false
				>(
					z, null_mask,
					internal::getRaw( a ), &( internal::getCoordinates( a ) ),
					internal::getRaw( x ), &( internal::getCoordinates( x ) ),
					&y, internal::getRaw( y ), &( internal::getCoordinates( y ) ),
					n, ring
				);
			}
		}

		// sparse or dense variant
		return internal::eWiseMulAdd_dispatch<
			descr, true, false, false, false, false
		>(
			z, &m,
			internal::getRaw( a ), &( internal::getCoordinates( a ) ),
			internal::getRaw( x ), &( internal::getCoordinates( x ) ),
			&y, internal::getRaw( y ), &( internal::getCoordinates( y ) ),
			n, ring
		);
	}

	/**
	 * Computes \f$ z = z + a * x + y \f$.
	 *
	 * Specialisation for scalar \a a and \a x, masked version.
	 *
	 * \internal Dispatches to masked eWiseAdd.
	 *
	 * \warning This function has been deprecated since version 0.5. If required,
	 *          consider instead a sequence of grb::foldl using the additive
	 *          monoid, followed by a call to grb::eWiseMul.
	 */
	template<
		Descriptor descr = descriptors::no_operation, class Ring,
		typename OutputType, typename MaskType,
		typename InputType1, typename InputType2, typename InputType3,
		typename Coords
	>
	RC eWiseMulAdd(
		Vector< OutputType, reference, Coords > &z,
		const Vector< MaskType, reference, Coords > &m,
		const InputType1 alpha,
		const InputType2 beta,
		const Vector< InputType3, reference, Coords > &y,
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
		std::cout << "eWiseMulAdd (reference, vector <- scalar x scalar + vector, "
			<< "masked) precomputes scalar multiply and dispatches to eWiseAdd "
			<< "(reference, vector <- scalar + vector, masked)\n";
#endif
		// dynamic checks
		const size_t n = size( z );
		if( size( m ) != n || size( y ) != n ) {
			return MISMATCH;
		}
		if( descr & descriptors::dense ) {
			if( nnz( z ) < n ) { return ILLEGAL; }
			if( nnz( m ) < n ) { return ILLEGAL; }
			if( nnz( y ) < n ) { return ILLEGAL; }
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

	/**
	 * Computes \f$ z = z + a * x + y \f$.
	 *
	 * Specialisation for scalar \a a, \a x, and \a y, masked version.
	 *
	 * \internal Dispatches to masked set.
	 *
	 * \warning This function has been deprecated since version 0.5. If required,
	 *          consider instead a sequence of grb::foldl using the additive
	 *          monoid, followed by a call to grb::eWiseMul.
	 */
	template<
		Descriptor descr = descriptors::no_operation, class Ring,
		typename OutputType, typename MaskType,
		typename InputType1, typename InputType2, typename InputType3,
		typename Coords
	>
	RC eWiseMulAdd(
		Vector< OutputType, reference, Coords > &z,
		const Vector< MaskType, reference, Coords > &m,
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
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< bool, MaskType >::value ),
			"grb::eWiseMulAdd",
			"called with a mask vector with a non-bool element type" );
#ifdef _DEBUG
		std::cout << "eWiseMulAdd (reference, vector <- scalar x scalar + scalar, "
			<< "masked) precomputes scalar operations and dispatches to set "
			<< "(reference, masked)\n";
#endif
		// dynamic checks
		const size_t n = size( z );
		if( size( m ) != n ) { return MISMATCH; }
		if( descr & descriptors::dense ) {
			if( nnz( z ) < n ) { return ILLEGAL; }
			if( nnz( m ) < n ) { return ILLEGAL; }
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
		return grb::foldl( z, m, add_result, ring.getAdditiveOperator(), phase );
	}

	/**
	 * Calculates the element-wise multiplication of two vectors,
	 *     \f$ z = z + x .* y \f$,
	 * under a given semiring.
	 *
	 * @tparam descr      The descriptor to be used (descriptors::no_operation
	 *                    if left unspecified).
	 * @tparam Ring       The semiring type to perform the element-wise multiply
	 *                    on.
	 * @tparam InputType1 The left-hand side input type to the multiplicative
	 *                    operator of the \a ring.
	 * @tparam InputType2 The right-hand side input type to the multiplicative
	 *                    operator of the \a ring.
	 * @tparam OutputType The the result type of the multiplicative operator of
	 *                    the \a ring.
	 *
	 * @param[out]  z  The output vector of type \a OutputType.
	 * @param[in]   x  The left-hand input vector of type \a InputType1.
	 * @param[in]   y  The right-hand input vector of type \a InputType2.
	 * @param[in] ring The generalized semiring under which to perform this
	 *                 element-wise multiplication.
	 *
	 * @return grb::MISMATCH Whenever the dimensions of \a x, \a y, and \a z do
	 *                       not match. All input data containers are left
	 *                       untouched if this exit code is returned; it will be
	 *                       as though this call was never made.
	 * @return grb::SUCCESS  On successful completion of this call.
	 *
	 * \parblock
	 * \par Valid descriptors
	 * grb::descriptors::no_operation, grb::descriptors::no_casting.
	 *
	 * \note Invalid descriptors will be ignored.
	 *
	 * If grb::descriptors::no_casting is specified, then 1) the first domain of
	 * \a ring must match \a InputType1, 2) the second domain of \a ring must match
	 * \a InputType2, 3) the third domain of \a ring must match \a OutputType. If
	 * one of these is not true, the code shall not compile.
	 *
	 * \endparblock
	 *
	 * \parblock
	 * \par Performance semantics
	 *      -# This call takes \f$ \Theta(n) \f$ work, where \f$ n \f$ equals the
	 *         size of the vectors \a x, \a y, and \a z. The constant factor
	 *         depends on the cost of evaluating the multiplication operator. A
	 *         good implementation uses vectorised instructions whenever the input
	 *         domains, the output domain, and the multiplicative operator used
	 *         allow for this.
	 *
	 *      -# This call will not result in additional dynamic memory allocations.
	 *
	 *      -# This call takes \f$ \mathcal{O}(1) \f$ memory beyond the memory
	 *         used by the application at the point of a call to this function.
	 *
	 *      -# This call incurs at most \f$ n( \mathit{sizeof}(\mathit{D1}) +
	 *         \mathit{sizeof}(\mathit{D2}) + \mathit{sizeof}(\mathit{D3})) +
	 *         \mathcal{O}(1) \f$ bytes of data movement. A good implementation
	 *         will stream \a x or \a y into \a z to apply the multiplication
	 *         operator in-place, whenever the input domains, the output domain,
	 *         and the operator used allow for this.
	 * \endparblock
	 *
	 * \warning When given sparse vectors, the zero now annihilates instead of
	 *       acting as an identity. Thus the eWiseMul cannot simply map to an
	 *       eWiseApply of the multiplicative operator.
	 *
	 * @see This is a specialised form of eWiseMulAdd.
	 */
	template<
		Descriptor descr = descriptors::no_operation, class Ring,
		typename InputType1, typename InputType2, typename OutputType,
		typename Coords
	>
	RC eWiseMul(
		Vector< OutputType, reference, Coords > &z,
		const Vector< InputType1, reference, Coords > &x,
		const Vector< InputType2, reference, Coords > &y,
		const Ring & ring = Ring(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if< !grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			grb::is_semiring< Ring >::value, void
		>::type * const = nullptr
	) {
		// static sanity checks
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< typename Ring::D1, InputType1 >::value ), "grb::eWiseMul",
			"called with a left-hand side input vector with element type that does not "
			"match the first domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< typename Ring::D2, InputType2 >::value ), "grb::eWiseMul",
			"called with a right-hand side input vector with element type that does "
			"not match the second domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< typename Ring::D3, OutputType >::value ), "grb::eWiseMul",
			"called with an output vector with element type that does not match the "
			"third domain of the given semiring" );

		// dynamic checks
		const size_t n = size( z );
		if( size( x ) != n || size( y ) != n ) { return MISMATCH; }
		if( descr & descriptors::dense ) {
			if( nnz( z ) < n ) { return ILLEGAL; }
			if( nnz( x ) < n ) { return ILLEGAL; }
			if( nnz( y ) < n ) { return ILLEGAL; }
		}

		// check trivial phase
		if( phase == RESIZE ) { return SUCCESS; }

#ifdef _DEBUG
		std::cout << "eWiseMul (reference, vector <- vector x vector) dispatches "
			<< "to eWiseMulAdd (vector <- vector x vector + 0)\n";
#endif
		return eWiseMulAdd< descr, true >(
			z, x, y,
			ring.template getZero< typename Ring::D4 >(),
			ring, phase
		);
	}

	/**
	 * Computes \f$ z = z + x * y \f$.
	 *
	 * Specialisation for scalar \a x.
	 */
	template<
		Descriptor descr = descriptors::no_operation, class Ring,
		typename InputType1, typename InputType2, typename OutputType,
		typename Coords
	>
	RC eWiseMul(
		Vector< OutputType, reference, Coords > &z,
		const InputType1 alpha,
		const Vector< InputType2, reference, Coords > &y,
		const Ring &ring = Ring(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if< !grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			grb::is_semiring< Ring >::value, void
		>::type * const = nullptr
	) {
		// static sanity checks
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< typename Ring::D1, InputType1 >::value ), "grb::eWiseMul",
			"called with a left-hand side input vector with element type that does not "
			"match the first domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< typename Ring::D2, InputType2 >::value ), "grb::eWiseMul",
			"called with a right-hand side input vector with element type that does "
			"not match the second domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< typename Ring::D3, OutputType >::value ), "grb::eWiseMul",
			"called with an output vector with element type that does not match the "
			"third domain of the given semiring" );

		// dynamic checks
		const size_t n = size( z );
		if( size( y ) != n ) { return MISMATCH; }
		if( descr & descriptors::dense ) {
			if( nnz( z ) < n ) { return ILLEGAL; }
			if( nnz( y ) < n ) { return ILLEGAL; }
		}

#ifdef _DEBUG
		std::cout << "eWiseMul (reference, vector <- scalar x vector) dispatches to "
			<< "eWiseMulAdd (vector <- scalar x vector + 0)\n";
#endif
		return eWiseMulAdd< descr, true >( z, alpha, y,
			ring.template getZero< typename Ring::D4 >(), ring, phase );
	}

	/**
	 * Computes \f$ z = z + x * y \f$.
	 *
	 * Specialisation for scalar \a y.
	 */
	template<
		Descriptor descr = descriptors::no_operation, class Ring,
		typename InputType1, typename InputType2, typename OutputType,
		typename Coords
	>
	RC eWiseMul(
		Vector< OutputType, reference, Coords > &z,
		const Vector< InputType1, reference, Coords > &x,
		const InputType2 beta,
		const Ring &ring = Ring(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if< !grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			grb::is_semiring< Ring >::value, void
		>::type * const = nullptr
	) {
		// static sanity checks
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< typename Ring::D1, InputType1 >::value ), "grb::eWiseMul",
			"called with a left-hand side input vector with element type that does not "
			"match the first domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< typename Ring::D2, InputType2 >::value ), "grb::eWiseMul",
			"called with a right-hand side input vector with element type that does "
			"not match the second domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< typename Ring::D3, OutputType >::value ), "grb::eWiseMul",
			"called with an output vector with element type that does not match the "
			"third domain of the given semiring" );

		// dynamic checks
		const size_t n = size( z );
		if( size( x ) != n ) { return MISMATCH; }
		if( descr & descriptors::dense ) {
			if( nnz( z ) < n ) { return ILLEGAL; }
			if( nnz( x ) < n ) { return ILLEGAL; }
		}

		// catch trivial phase
		if( phase == RESIZE ) {
			return SUCCESS;
		}

#ifdef _DEBUG
		std::cout << "eWiseMul (reference) dispatches to eWiseMulAdd with 0.0 as "
			<< "additive scalar\n";
#endif

		return eWiseMulAdd< descr, true >(
			z, x, beta,
			ring.template getZero< typename Ring::D4 >(),
			ring, phase
		);
	}

	/**
	 * Computes \f$ z = z + x * y \f$.
	 *
	 * Specialisation for scalar \a y and scalar \a x.
	 */
	template<
		Descriptor descr = descriptors::no_operation, class Ring,
		typename InputType1, typename InputType2, typename OutputType,
		typename Coords
	>
	RC eWiseMul(
		Vector< OutputType, reference, Coords > &z,
		const InputType1 alpha,
		const InputType2 beta,
		const Ring &ring = Ring(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if< !grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			grb::is_semiring< Ring >::value, void
		>::type * const = nullptr
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
		if( descr & descriptors::dense ) {
			if( nnz( z ) < size( z ) ) { return ILLEGAL; }
		}

		// check for trivial phase
		if( phase == RESIZE ) {
			return SUCCESS;
		}

#ifdef _DEBUG
		std::cout << "eWiseMul (reference) dispatches to eWiseMulAdd with 0.0 as additive scalar\n";
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

	/**
	 * Calculates the element-wise multiplication of two vectors,
	 *     \f$ z = z + x .* y \f$,
	 * under a given semiring.
	 *
	 * Masked verison.
	 *
	 * \internal Dispatches to eWiseMulAdd with zero additive scalar.
	 */
	template<
		Descriptor descr = descriptors::no_operation, class Ring,
		typename InputType1, typename InputType2,
		typename OutputType, typename MaskType,
		typename Coords
	>
	RC eWiseMul(
		Vector< OutputType, reference, Coords > &z,
		const Vector< MaskType, reference, Coords > &m,
		const Vector< InputType1, reference, Coords > &x,
		const Vector< InputType2, reference, Coords > &y,
		const Ring &ring = Ring(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if< !grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			!grb::is_object< MaskType >::value &&
			grb::is_semiring< Ring >::value, void
		>::type * const = nullptr
	) {
		// static sanity checks
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< typename Ring::D1, InputType1 >::value ), "grb::eWiseMul",
			"called with a left-hand side input vector with element type that does not "
			"match the first domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< typename Ring::D2, InputType2 >::value ), "grb::eWiseMul",
			"called with a right-hand side input vector with element type that does "
			"not match the second domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< typename Ring::D3, OutputType >::value ), "grb::eWiseMul",
			"called with an output vector with element type that does not match the "
			"third domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< bool, MaskType >::value ), "grb::eWiseMulAdd",
			"called with a mask vector with a non-bool element type" );
#ifdef _DEBUG
		std::cout << "eWiseMul (reference, vector <- vector x vector, masked) "
			<< "dispatches to eWiseMulAdd (vector <- vector x vector + 0, masked)\n";
#endif
		return eWiseMulAdd< descr, true >(
			z, m, x, y,
			ring.template getZero< typename Ring::D4 >(),
			ring, phase
		);
	}

	/**
	 * Computes \f$ z = z + x * y \f$.
	 *
	 * Specialisation for scalar \a x, masked version.
	 *
	 * \internal Dispatches to eWiseMulAdd with zero additive scalar.
	 */
	template<
		Descriptor descr = descriptors::no_operation, class Ring,
		typename InputType1, typename InputType2,
		typename OutputType, typename MaskType,
		typename Coords
	>
	RC eWiseMul(
		Vector< OutputType, reference, Coords > &z,
		const Vector< MaskType, reference, Coords > &m,
		const InputType1 alpha,
		const Vector< InputType2, reference, Coords > &y,
		const Ring &ring = Ring(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if< !grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			!grb::is_object< MaskType >::value &&
			grb::is_semiring< Ring >::value, void
		>::type * const = nullptr
	) {
		// static sanity checks
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< typename Ring::D1, InputType1 >::value ), "grb::eWiseMul",
			"called with a left-hand side input vector with element type that does not "
			"match the first domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< typename Ring::D2, InputType2 >::value ), "grb::eWiseMul",
			"called with a right-hand side input vector with element type that does "
			"not match the second domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< typename Ring::D3, OutputType >::value ), "grb::eWiseMul",
			"called with an output vector with element type that does not match the "
			"third domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< bool, MaskType >::value ), "grb::eWiseMulAdd",
			"called with a mask vector _m with a non-bool element type" );

		// check for empty mask
		if( size( m ) == 0 ) {
			return eWiseMul< descr >( z, alpha, y, ring, phase );
		}

		// dynamic checks
		const size_t n = size( z );
		if( size( m ) != n || size( y ) != n ) { return MISMATCH; }
		if( descr & descriptors::dense ) {
			if( nnz( z ) < n ) { return ILLEGAL; }
			if( nnz( y ) < n ) { return ILLEGAL; }
			if( nnz( m ) < n ) { return ILLEGAL; }
		}

#ifdef _DEBUG
		std::cout << "eWiseMul (reference, vector <- scalar x vector, masked) "
			<< "dispatches to eWiseMulAdd (vector <- scalar x vector + 0, masked)\n";
#endif
		return eWiseMulAdd< descr, true >( z, m, alpha, y,
			ring.template getZero< typename Ring::D4 >(), ring, phase );
	}

	/**
	 * Computes \f$ z = z + x * y \f$.
	 *
	 * Specialisation for scalar \a y, masked version.
	 *
	 * \internal Dispatches to eWiseMulAdd with zero additive scalar.
	 */
	template<
		Descriptor descr = descriptors::no_operation, class Ring,
		typename InputType1, typename InputType2,
		typename OutputType, typename MaskType,
		typename Coords
	>
	RC eWiseMul(
		Vector< OutputType, reference, Coords > &z,
		const Vector< MaskType, reference, Coords > &m,
		const Vector< InputType1, reference, Coords > &x,
		const InputType2 beta,
		const Ring &ring = Ring(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if< !grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			!grb::is_object< MaskType >::value &&
			grb::is_semiring< Ring >::value, void
		>::type * const = nullptr
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
		if( descr & descriptors::dense ) {
			if( nnz( z ) < n ) { return ILLEGAL; }
			if( nnz( x ) < n ) { return ILLEGAL; }
			if( nnz( m ) < n ) { return ILLEGAL; }
		}

		// check for trivial phase
		if( phase == RESIZE ) {
			return SUCCESS;
		}

#ifdef _DEBUG
		std::cout << "eWiseMul (reference, masked) dispatches to masked eWiseMulAdd "
			<< "with 0.0 as additive scalar\n";
#endif
		return eWiseMulAdd< descr, true >(
			z, m, x, beta,
			ring.template getZero< typename Ring::D4 >(),
			ring.getMultiplicativeOperator(),
			phase
		);
	}

	/**
	 * Computes \f$ z = z + x * y \f$.
	 *
	 * Specialisation for scalar \a y, scalar \a x, masked version.
	 */
	template<
		Descriptor descr = descriptors::no_operation,
		class Ring,
		typename InputType1, typename InputType2,
		typename OutputType, typename MaskType,
		typename Coords
	>
	RC eWiseMul(
		Vector< OutputType, reference, Coords > &z,
		const Vector< MaskType, reference, Coords > &m,
		const InputType1 alpha,
		const InputType2 beta,
		const Ring &ring = Ring(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if< !grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			!grb::is_object< MaskType >::value &&
			grb::is_semiring< Ring >::value, void
		>::type * const = nullptr
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
		if( descr & descriptors::dense ) {
			if( nnz( z ) < n ) { return ILLEGAL; }
			if( nnz( m ) < n ) { return ILLEGAL; }
		}

		// check for trivial phase
		if( phase == RESIZE ) {
			return SUCCESS;
		}

#ifdef _DEBUG
		std::cout << "eWiseMul (reference, masked) dispatches to masked foldl\n";
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

		/** @see grb::dot */
		template<
			Descriptor descr = descriptors::no_operation,
			class AddMonoid, class AnyOp,
			typename OutputType, typename InputType1, typename InputType2,
			typename Coords
		>
		RC dot_generic(
			OutputType &z,
			const Vector< InputType1, reference, Coords > &x,
			const Vector< InputType2, reference, Coords > &y,
			const AddMonoid &addMonoid,
			const AnyOp &anyOp,
			const Phase &phase
		) {
			const size_t n = internal::getCoordinates( x ).size();
			if( n != internal::getCoordinates( y ).size() ) {
				return MISMATCH;
			}

			if( phase == RESIZE ) {
				return SUCCESS;
			}
			assert( phase == EXECUTE );

			// check whether dense flag is set correctly
			constexpr bool dense = descr & descriptors::dense;
			const size_t nzx = internal::getCoordinates( x ).nonzeroes();
			const size_t nzy = internal::getCoordinates( y ).nonzeroes();
			if( dense ) {
				if( n != nzx || n != nzy ) {
					return PANIC;
				}
			} else {
				if( n == nzx && n == nzy ) {
					return PANIC;
				}
			}

			size_t loopsize = n;
			auto * coors_r_p = &( internal::getCoordinates( x ) );
			auto * coors_q_p = &( internal::getCoordinates( y ) );
			if( !dense ) {
				if( nzx < nzy ) {
					loopsize = nzx;
				} else {
					loopsize = nzy;
					std::swap( coors_r_p, coors_q_p );
				}
			}
			auto &coors_r = *coors_r_p;
			auto &coors_q = *coors_q_p;

#ifdef _DEBUG
			std::cout << "\t In dot_generic with loopsize " << loopsize << "\n";
#endif

#ifdef _H_GRB_REFERENCE_OMP_BLAS1
			z = addMonoid.template getIdentity< OutputType >();
			#pragma omp parallel
			{
				size_t start, end;
				config::OMP::localRange( start, end, 0, loopsize, AnyOp::blocksize );
#else
				const size_t start = 0;
				const size_t end = loopsize;
#endif
#ifdef _DEBUG
				std::cout << "\t\t local thread has range " << start << "--" << end << "\n";
#endif
				if( end > start ) {
					// get raw alias
					const InputType1 * __restrict__ a = internal::getRaw( x );
					const InputType2 * __restrict__ b = internal::getRaw( y );

					// overwrite z with first multiplicant, if available-- otherwise, initialise
					// to zero:
					typename AddMonoid::D3 reduced =
						addMonoid.template getIdentity< typename AddMonoid::D3 >();
					if( dense ) {
						apply( reduced, a[ end - 1 ], b[ end - 1 ], anyOp );
					} else {
						const size_t index = coors_r.index( end - 1 );
						if( coors_q.assigned( index ) ) {
							apply( reduced, a[ index ], b[ index ], anyOp );
						}
					}

					// enter vectorised loop
					size_t i = start;
					if( dense ) {
						while( i + AnyOp::blocksize < end - 1 ) {
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
							addMonoid.getOperator().foldlArray( reduced, zz, AnyOp::blocksize );
							//^--> note that this foldl operates on raw arrays,
							//     and thus should not be mistaken with a foldl
							//     on a grb::Vector.
#ifdef _DEBUG
							std::cout << "\t\t " << (i-AnyOp::blocksize) << "--" << i << ": "
								<< "running reduction = " << reduced << "\n";
#endif
						}
					} else {
#ifdef _DEBUG
						std::cout << "\t\t in sparse variant, nonzero range " << start << "--"
							<< end << ", blocksize " << AnyOp::blocksize << "\n";
#endif
						while( i + AnyOp::blocksize < end - 1 ) {
							// declare buffers
							static_assert( AnyOp::blocksize > 0,
								"Configuration error: vectorisation blocksize set to 0!" );
							typename AnyOp::D1 xx[ AnyOp::blocksize ];
							typename AnyOp::D2 yy[ AnyOp::blocksize ];
							typename AnyOp::D3 zz[ AnyOp::blocksize ];
							bool mask[ AnyOp::blocksize ];

							// prepare registers
							for( size_t k = 0; k < AnyOp::blocksize; ++k, ++i ) {
								mask[ k ] = coors_q.assigned( coors_r.index( i ) );
							}

							// rewind
							i -= AnyOp::blocksize;

							// do masked load
							for( size_t k = 0; k < AnyOp::blocksize; ++k, ++i ) {
								if( mask[ k ] ) {
									xx[ k ] = static_cast< typename AnyOp::D1 >( a[ coors_r.index( i ) ] );
									yy[ k ] = static_cast< typename AnyOp::D2 >( b[ coors_r.index( i ) ] );
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
							addMonoid.getOperator().foldlArray( reduced, zz, AnyOp::blocksize );
							//^--> note that this foldl operates on raw arrays,
							//     and thus should not be mistaken with a foldl
							//     on a grb::Vector.
						}
					}

					// perform element-by-element updates for remainder (if any)
					for( ; i < end - 1; ++i ) {
						OutputType temp = addMonoid.template getIdentity< OutputType >();
						const size_t index = coors_r.index( i );
						if( dense || coors_q.assigned( index ) ) {
							apply( temp, a[ index ], b[ index ], anyOp );
							foldr( temp, reduced, addMonoid.getOperator() );
						}
					}

					// write back result
#ifdef _H_GRB_REFERENCE_OMP_BLAS1
					#pragma omp critical
					{ foldr( reduced, z, addMonoid.getOperator() ); }
#else
					z = static_cast< OutputType >( reduced );
#endif
				}
#ifdef _H_GRB_REFERENCE_OMP_BLAS1
			} // end parallel section
#endif
#ifdef _DEBUG
			std::cout << "\t returning " << z << "\n";
#endif

			// done!
			return SUCCESS;
		}

	} // namespace internal

	/**
	 * Calculates the dot product, \f$ \alpha = (x,y) \f$, under a given additive
	 * monoid and multiplicative operator.
	 *
	 * @tparam descr      The descriptor to be used (descriptors::no_operation
	 *                    if left unspecified).
	 * @tparam Ring       The semiring type to use.
	 * @tparam OutputType The output type.
	 * @tparam InputType1 The input element type of the left-hand input vector.
	 * @tparam InputType2 The input element type of the right-hand input vector.
	 *
	 * @param[in,out]  z    The output element \f$ z + \alpha \f$.
	 * @param[in]      x    The left-hand input vector.
	 * @param[in]      y    The right-hand input vector.
	 * @param[in] addMonoid The additive monoid under which the reduction of the
	 *                      results of element-wise multiplications of \a x and
	 *                      \a y are performed.
	 * @param[in]   anyop   The multiplicative operator under which element-wise
	 *                      multiplications of \a x and \a y are performed. This can
	 *                      be any binary operator.
	 *
	 * By the definition that a dot-product operates under any additive monoid and
	 * any binary operator, it follows that a dot-product under any semiring can be
	 * trivially reduced to a call to this version instead.
	 *
	 * @return grb::MISMATCH When the dimensions of \a x and \a y do not match. All
	 *                       input data containers are left untouched if this exit
	 *                       code is returned; it will be as though this call was
	 *                       never made.
	 * @return grb::SUCCESS  On successful completion of this call.
	 *
	 * \parblock
	 * \par Performance semantics
	 *      -# This call takes \f$ \Theta(n/p) \f$ work at each user process, where
	 *         \f$ n \f$ equals the size of the vectors \a x and \a y, and
	 *         \f$ p \f$ is the number of user processes. The constant factor
	 *         depends on the cost of evaluating the addition and multiplication
	 *         operators. A good implementation uses vectorised instructions
	 *         whenever the input domains, output domain, and the operators used
	 *         allow for this.
	 *
	 *      -# This call takes \f$ \mathcal{O}(1) \f$ memory beyond the memory used
	 *         by the application at the point of a call to this function.
	 *
	 *      -# This call incurs at most
	 *         \f$ n( \mathit{sizeof}(\mathit{D1}) + \mathit{sizeof}(\mathit{D2}) ) + \mathcal{O}(p) \f$
	 *         bytes of data movement.
	 *
	 *      -# This call incurs at most \f$ \Theta(\log p) \f$ synchronisations
	 *         between two or more user processes.
	 *
	 *      -# A call to this function does result in any system calls.
	 * \endparblock
	 *
	 * \note This requires an implementation to pre-allocate \f$ \Theta(p) \f$
	 *       memory for inter-process reduction, if the underlying communication
	 *       layer indeed requires such a buffer. This buffer may not be allocated
	 *       (nor freed) during a call to this function.
	 *
	 * \parblock
	 * \par Valid descriptors
	 *   -# grb::descriptors::no_operation
	 *   -# grb::descriptors::no_casting
	 *   -# grb::descriptors::dense
	 * \endparblock
	 *
	 * If the dense descriptor is set, this implementation returns grb::ILLEGAL if
	 * it was detected that either \a x or \a y was sparse. In this case, it shall
	 * otherwise be as though the call to this function had not occurred (no side
	 * effects).
	 *
	 * \note The standard, in contrast, only specifies undefined behaviour would
	 *       occur. This implementation goes beyond the standard by actually
	 *       specifying what will happen.
	 */
	template<
		Descriptor descr = descriptors::no_operation,
		class AddMonoid, class AnyOp,
		typename OutputType, typename InputType1, typename InputType2,
		typename Coords
	>
	RC dot(
		OutputType &z,
		const Vector< InputType1, reference, Coords > &x,
		const Vector< InputType2, reference, Coords > &y,
		const AddMonoid &addMonoid = AddMonoid(),
		const AnyOp &anyOp = AnyOp(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if< !grb::is_object< OutputType >::value &&
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
		std::cout << "In grb::dot (reference). "
			<< "I/O scalar on input reads " << z << "\n";
#endif

		// dynamic sanity check
		const size_t n = internal::getCoordinates( y ).size();
		if( internal::getCoordinates( x ).size() != n ) {
			return MISMATCH;
		}

		// cache nnzs
		const size_t nnzx = internal::getCoordinates( x ).nonzeroes();
		const size_t nnzy = internal::getCoordinates( y ).nonzeroes();

#ifdef _DEBUG
		std::cout << "\t dynamic checks pass\n";
#endif

		// catch trivial case
		if( nnzx == 0 && nnzy == 0 ) {
#ifdef _DEBUG
			std::cout << "\t at least one input vector is empty-- exiting\n";
#endif
			return SUCCESS;
		}

		// dot will be computed out-of-place here. A separate field is needed because
		// of possible multi-threaded computation of the dot.
		OutputType oop = addMonoid.template getIdentity< OutputType >();

		// if descriptor says nothing about being dense...
		RC ret = SUCCESS;
		if( !(descr & descriptors::dense) ) {
			// check if inputs are actually dense...
			if( nnzx == n && nnzy == n ) {
				// call dense implementation
#ifdef _DEBUG
				std::cout << "\t dispatching to dense dot_generic (I)\n";
#endif
				ret = internal::dot_generic< descr | descriptors::dense >( oop, x, y,
					addMonoid, anyOp, phase );
			} else {
				// pass to sparse implementation
#ifdef _DEBUG
				std::cout << "\t dispatching to sparse dot_generic\n";
#endif
				ret = internal::dot_generic< descr >( oop, x, y, addMonoid, anyOp, phase );
			}
		} else {
			// descriptor says dense, but if any of the vectors are actually sparse...
			if( nnzx < n || nnzy < n ) {
#ifdef _DEBUG
				std::cout << "\t dense descriptor given, but at least one input vector was "
					"sparse\n";
#endif
				return ILLEGAL;
			} else {
				// all OK, pass to dense implementation
#ifdef _DEBUG
				std::cout << "\t dispatching to dense dot_generic (II)\n";
#endif
				ret = internal::dot_generic< descr >( oop, x, y, addMonoid, anyOp, phase );
			}
		}

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

	/**
	 * \internal
	 * Provides a generic implementation of the dot computation on semirings by
	 * translating it into a dot computation on an additive commutative monoid
	 * with any multiplicative operator.
	 * \endinternal
	 */
	template<
		Descriptor descr = descriptors::no_operation, class Ring,
		typename IOType, typename InputType1, typename InputType2,
		typename Coords
	>
	RC dot(
		IOType &x,
		const Vector< InputType1, reference, Coords > &left,
		const Vector< InputType2, reference, Coords > &right,
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
		std::cout << "In grb::dot (reference, semiring version)\n"
			<< "\t dispatches to monoid-operator version\n";
#endif
		return grb::dot< descr >( x,
			left, right,
			ring.getAdditiveMonoid(),
			ring.getMultiplicativeOperator(),
			phase
		);
	}

	/** \internal No implementation notes. */
	template< typename Func, typename DataType, typename Coords >
	RC eWiseMap( const Func f, Vector< DataType, reference, Coords > &x ) {
		const auto &coors = internal::getCoordinates( x );
		if( coors.isDense() ) {
			// vector is distributed sequentially, so just loop over it
#ifdef _H_GRB_REFERENCE_OMP_BLAS1
			#pragma omp parallel
			{
				size_t start, end;
				config::OMP::localRange( start, end, 0, coors.size() );
#else
				const size_t start = 0;
				const size_t end = coors.size();
#endif
				for( size_t i = start; i < end; ++i ) {
					// apply the lambda
					DataType &xval = internal::getRaw( x )[ i ];
					xval = f( xval );
				}
#ifdef _H_GRB_REFERENCE_OMP_BLAS1
			}
#endif
		} else {
#ifdef _H_GRB_REFERENCE_OMP_BLAS1
			#pragma omp parallel
			{
				size_t start, end;
				config::OMP::localRange( start, end, 0, coors.nonzeroes() );
#else
				const size_t start = 0;
				const size_t end = coors.nonzeroes();
#endif
				for( size_t k = start; k < end; ++k ) {
					DataType &xval = internal::getRaw( x )[ coors.index( k ) ];
					xval = f( xval );
				}
#ifdef _H_GRB_REFERENCE_OMP_BLAS1
			}
#endif
		}
		// and done!
		return SUCCESS;
	}

	/**
	 * This is the eWiseLambda that performs length checking by recursion.
	 *
	 * in the reference implementation all vectors are distributed equally, so no
	 * need to synchronise any data structures. We do need to do error checking
	 * though, to see when to return grb::MISMATCH. That's this function.
	 *
	 * @see Vector::operator[]()
	 * @see Vector::lambda_reference
	 */
	template<
		typename Func,
		typename DataType1, typename DataType2,
		typename Coords, typename... Args
	>
	RC eWiseLambda(
		const Func f, const Vector< DataType1, reference, Coords > &x,
		const Vector< DataType2, reference, Coords > &y, Args const &... args
	) {
		// catch mismatch
		if( size( x ) != size( y ) ) {
			return MISMATCH;
		}
		// continue
		return eWiseLambda( f, x, args... );
	}

	/**
	 * No implementation notes. This is the `real' implementation on reference
	 * vectors.
	 *
	 * @see Vector::operator[]()
	 * @see Vector::lambda_reference
	 */
	template< typename Func, typename DataType, typename Coords >
	RC eWiseLambda(
		const Func f,
		const Vector< DataType, reference, Coords > &x
	) {
#ifdef _DEBUG
		std::cout << "Info: entering eWiseLambda function on vectors.\n";
#endif
		const auto &coors = internal::getCoordinates( x );
		if( coors.isDense() ) {
			// vector is distributed sequentially, so just loop over it
#ifdef _H_GRB_REFERENCE_OMP_BLAS1
			#pragma omp parallel
			{
				size_t start, end;
				config::OMP::localRange( start, end, 0, coors.size() );
#else
				const size_t start = 0;
				const size_t end = coors.size();
#endif
				for( size_t i = start; i < end; ++i ) {
					// apply the lambda
					f( i );
				}
#ifdef _H_GRB_REFERENCE_OMP_BLAS1
			}
#endif
		} else {
#ifdef _H_GRB_REFERENCE_OMP_BLAS1
			#pragma omp parallel
			{
				size_t start, end;
				config::OMP::localRange( start, end, 0, coors.nonzeroes() );
#else
				const size_t start = 0;
				const size_t end = coors.nonzeroes();
#endif
				for( size_t k = start; k < end; ++k ) {
					const size_t i = coors.index( k );
#ifdef _DEBUG
					std::cout << "\tprocessing coordinate " << k << " "
						<< "which has index " << i << "\n";
#endif
					f( i );
				}
#ifdef _H_GRB_REFERENCE_OMP_BLAS1
			}
#endif
		}
		// and done!
		return SUCCESS;
	}

	/**
	 * Reduces a vector into a scalar.
	 *
	 * See the base documentation for the full specification.
	 *
	 * \parblock
	 * \par Performance semantics
	 *      -# This call comprises \f$ \Theta(n) \f$ work. The quantity \f$ n \f$
	 *         is specified below.
	 *
	 *      -# This call comprises \f$ \Theta(n) + \mathcal{O}(p) \f$ operator
	 *         applications. The quantity \f$ n \f$ is specified below. The value
	 *         \f$ p \f$ is is the number of user processes.
	 *
	 *      -# This call will not result in additional dynamic memory allocations.
	 *         No system calls will be made.
	 *
	 *      -# This call takes \f$ \mathcal{O}(1) \f$ memory beyond the memory
	 *         used by the application at the point of a call to this function.
	 *
	 *      -# This call incurs at most
	 *         \f$ n \mathit{sizeof}(\mathit{InputType}) + \mathcal{O}(1) \f$
	 *         bytes of intra-process data movement. The quantity \f$ n \f$ is
	 *         specified below.
	 *
	 *      -# This call incurs at most \f$ \mathcal{O}(p) \f$ inter-process data
	 *         movement, where \f$ p \f$ is the number of user processes. It incurs
	 *         at least \f$ \Omega(\log p) \f$ inter-process data movement.
	 *
	 *      -# This call incurs at most \f$ \mathcal{O}(\log p) \f$
	 *         synchronisations, and at least one.
	 *
	 * If \a y is dense, then \f$ n \f$ is the size of \a y. If \a y is sparse,
	 * then \f$ n \f$ is the number of nonzeroes in \a y. If \a mask is non-empty
	 * and #grb::descriptors::invert_mask is given, then \f$ n \f$ equals the size
	 * of \a y. If \a mask is non-empty (and the mask is not inverted), then
	 * \f$ n \f$ is the minimum of the number of nonzeroes in \a y and \a mask.
	 * \endparblock
	 */
	template<
		Descriptor descr = descriptors::no_operation,
		class Monoid,
		typename InputType, typename IOType, typename MaskType,
		typename Coords
	>
	RC foldl(
		IOType &x,
		const Vector< InputType, reference, Coords > &y,
		const Vector< MaskType, reference, Coords > &mask,
		const Monoid &monoid = Monoid(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if< !grb::is_object< IOType >::value &&
			!grb::is_object< InputType >::value &&
			!grb::is_object< MaskType >::value &&
			grb::is_monoid< Monoid >::value, void
		>::type * const = nullptr
	) {
		(void) phase;
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

	/**
	 * Folds a vector into a scalar.
	 *
	 * Unmasked variant.
	 *
	 * For performance semantics, see the masked variant of this primitive.
	 *
	 * \internal Dispatches to the masked variant, using an empty mask.
	 */
	template<
		Descriptor descr = descriptors::no_operation,
		class Monoid,
		typename IOType, typename InputType,
		typename Coords
	>
	RC foldl(
		IOType &x,
		const Vector< InputType, reference, Coords > &y,
		const Monoid &monoid = Monoid(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			!grb::is_object< IOType >::value &&
			!grb::is_object< InputType >::value &&
			grb::is_monoid< Monoid >::value, void
		>::type * const = nullptr
	) {
		(void) phase;
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
		Vector< bool, reference, Coords > empty_mask( 0 );
		return internal::template fold_from_vector_to_scalar_generic<
			descr, false, true
		>( x, y, empty_mask, monoid );
	}

	/**
	 * TODO internal issue #195
	 */
	template<
		Descriptor descr = descriptors::no_operation,
		typename T, typename U, typename Coords
	>
	RC zip(
		Vector< std::pair< T, U >, reference, Coords > &z,
		const Vector< T, reference, Coords > &x,
		const Vector< U, reference, Coords > &y,
		const Phase &phase = EXECUTE,
		const typename std::enable_if< !grb::is_object< T >::value &&
			!grb::is_object< U >::value, void
		>::type * const = nullptr
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

		// internal issue #195
		if( nnz( x ) < n ) {
			return ILLEGAL;
		}
		if( nnz( y ) < n ) {
			return ILLEGAL;
		}

		auto &z_coors = internal::getCoordinates( z );
		const T * const x_raw = internal::getRaw( x );
		const U * const y_raw = internal::getRaw( y );
		std::pair< T, U > * z_raw = internal::getRaw( z );
		z_coors.assignAll();
		for( size_t i = 0; i < n; ++i ) {
			z_raw[ i ].first = x_raw[ i ];
			z_raw[ i ].second = y_raw[ i ];
		}
		return SUCCESS;
	}

	/**
	 * TODO internal issue #195
	 */
	template<
		Descriptor descr = descriptors::no_operation,
		typename T, typename U, typename Coords
	>
	RC unzip(
		Vector< T, reference, Coords > &x,
		Vector< U, reference, Coords > &y,
		const Vector< std::pair< T, U >, reference, Coords > &in,
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			!grb::is_object< T >::value &&
			!grb::is_object< U >::value, void
		>::type * const = nullptr
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

		// internal issue #195
		if( nnz( in ) < n ) {
			return ILLEGAL;
		}

		auto &x_coors = internal::getCoordinates( x );
		auto &y_coors = internal::getCoordinates( y );
		T * const x_raw = internal::getRaw( x );
		U * const y_raw = internal::getRaw( y );
		const std::pair< T, U > * in_raw = internal::getRaw( in );
		x_coors.assignAll();
		y_coors.assignAll();
		for( size_t i = 0; i < n; ++i ) {
			x_raw[ i ] = in_raw[ i ].first;
			y_raw[ i ] = in_raw[ i ].second;
		}
		return SUCCESS;
	}

/** @} */
//   ^-- ends BLAS-1 module

} // end namespace ``grb''

#undef NO_CAST_ASSERT
#undef NO_CAST_OP_ASSERT

// parse this unit again for OpenMP support
#ifdef _GRB_WITH_OMP
 #ifndef _H_GRB_REFERENCE_OMP_BLAS1
  #define _H_GRB_REFERENCE_OMP_BLAS1
  #define reference reference_omp
  #include "graphblas/reference/blas1.hpp"
  #undef reference
  #undef _H_GRB_REFERENCE_OMP_BLAS1
 #endif
#endif

#endif // end `_H_GRB_REFERENCE_BLAS1'

