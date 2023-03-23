
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
 * Defines the nonblocking level-2 parameters
 *
 * @author Aristeidis Mastoras
 * @date 16th of May, 2022
 */

#ifndef _H_GRB_NONBLOCKING_BLAS2
#define _H_GRB_NONBLOCKING_BLAS2

#include <limits>
#include <algorithm>
#include <type_traits>

#include <graphblas/base/blas2.hpp>
#include <graphblas/blas0.hpp>
#include <graphblas/descriptors.hpp>
#include <graphblas/internalops.hpp>
#include <graphblas/ops.hpp>

#include <graphblas/reference/compressed_storage.hpp>

#include "coordinates.hpp"
#include "forward.hpp"
#include "matrix.hpp"
#include "vector.hpp"
#include "lazy_evaluation.hpp"
#include "boolean_dispatcher_blas2.hpp"

#ifdef _DEBUG
#include "spmd.hpp"
#endif

#define NO_CAST_ASSERT( x, y, z )                                          \
	static_assert( x,                                                      \
		"\n\n"                                                             \
		"****************************************************************" \
		"****************************************************************" \
		"**************************************\n"                         \
		"*     ERROR      | " y " " z ".\n"                                \
		"****************************************************************" \
		"****************************************************************" \
		"**************************************\n"                         \
		"* Possible fix 1 | Remove no_casting from the template "          \
		"parameters in this call to " y ".\n"                              \
		"* Possible fix 2 | Provide objects with element types or "        \
		"domains that match the expected type.\n"                          \
		"****************************************************************" \
		"****************************************************************" \
		"**************************************\n" );


namespace grb {

	namespace internal {

		extern LazyEvaluation le;
	}
}

namespace grb {

	/**
	 * \addtogroup nonblocking
	 * @{
	 */

	// put the generic mxv implementation in an internal namespace
	namespace internal {

		template<
			bool output_dense,
			bool left_handed,
			class AdditiveMonoid,
			class Multiplication,
			template< typename > class One,
			typename IOType,
			typename InputType,
			typename SourceType,
			typename Coords
		>
		class addIdentityDuringMV<
			nonblocking, true, output_dense, left_handed,
			AdditiveMonoid, Multiplication, One,
			IOType, InputType, SourceType, Coords
		> {

			public:

				static void apply(
					Vector< IOType, nonblocking, Coords > &destination_vector,
					IOType * __restrict__ const &destination,
					const size_t &destination_range,
					const size_t &source_index,
					const AdditiveMonoid &add,
					const Multiplication &mul,
					const SourceType &input_element,
					const std::function< size_t( size_t ) > &src_local_to_global,
					const std::function< size_t( size_t ) > &dst_global_to_local
				) {

				}
		};

		template<
			Descriptor descr,
			bool masked,
			bool input_masked,
			bool left_handed,
			template< typename > class One,
#ifdef GRB_BOOLEAN_DISPATCHER
			bool already_dense_destination_vector,
			bool already_dense_mask_vector,
#endif
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
#ifndef GRB_BOOLEAN_DISPATCHER
			bool already_dense_destination_vector,
			bool already_dense_mask_vector,
#endif
			RC &rc,
			const size_t lower_bound,
			Coords &local_destination_vector,
			const Coords &local_mask_vector,
			Vector< IOType, nonblocking, Coords > &destination_vector,
			IOType &destination_element,
			const size_t &destination_index,
			const Vector< InputType1, nonblocking, Coords > &source_vector,
			const InputType1 * __restrict__ const &source,
			const size_t &source_range,
			const internal::Compressed_Storage<
				InputType2, RowColType, NonzeroType
			> &matrix,
			const Vector< InputType3, nonblocking, Coords > &mask_vector,
			const InputType3 * __restrict__ const &mask,
			const Vector< InputType4, nonblocking, Coords > &source_mask_vector,
			const InputType4 * __restrict__ const &source_mask,
			const AdditiveMonoid &add,
			const Multiplication &mul,
			const std::function< size_t( size_t ) > &src_local_to_global,
			const std::function< size_t( size_t ) > &src_global_to_local,
			const std::function< size_t( size_t ) > &dst_local_to_global
		) {
#ifndef _DEBUG
			(void) destination_vector;
#endif
			constexpr bool add_identity = descr & descriptors::add_identity;
			constexpr bool dense_hint = descr & descriptors::dense;
			constexpr bool explicit_zero = descr & descriptors::explicit_zero;
#ifdef _DEBUG
			constexpr bool use_index = descr & descriptors::use_index;
#endif
			assert( rc == SUCCESS );

			// check whether we should compute output here
			if( masked ) {
				if( already_dense_mask_vector ) {
					if( !internal::getCoordinates( mask_vector ).template
						mask< descr >( destination_index, mask )
					) {
#ifdef _DEBUG
						std::cout << "Masks says to skip processing destination index " <<
							destination_index << "\n";
#endif
						return;
					}
				} else {
					if( !local_mask_vector.template
						mask< descr >( destination_index - lower_bound, mask )
					) {
#ifdef _DEBUG
						std::cout << "Masks says to skip processing destination index " <<
							destination_index << "\n";
#endif
						return;
					}
				}
			}

			// take shortcut, if possible
			if( grb::has_immutable_nonzeroes< AdditiveMonoid >::value && (
					already_dense_destination_vector ||
					local_destination_vector.assigned( destination_index - lower_bound )
				) && destination_element != add.template getIdentity< IOType >()
			) {
				return;
			}

			// start output
			typename AdditiveMonoid::D3 output =
				add.template getIdentity< typename AdditiveMonoid::D3 >();
			bool set = false;

			// if we need to add identity, do so first:
			if( add_identity ) {
				const size_t id_location = src_global_to_local( dst_local_to_global(
					destination_index ) );
				// the SpMV primitive may access non-local elements, and thus referring to
				// the input vector by using local coordinates is incorrect
				// the input vector of an SpMV cannot be updated, i.e., written, by another
				// primitive executed in the same pipeline with the current SpMV
				// therefore, in the current design, it's safe to use global coordinates for
				// the input vector
				if( ( !input_masked ||
						internal::getCoordinates( source_mask_vector ).template
							mask< descr >( id_location, source_mask )
					) && id_location < source_range
				) {
					if( dense_hint || internal::getCoordinates( source_vector ).assigned( id_location ) ) {
						typename AdditiveMonoid::D1 temp;
						internal::CopyOrApplyWithIdentity<
								!left_handed, typename AdditiveMonoid::D1, InputType1, One
							>::set( temp, source_vector[ id_location ], mul );
						internal::CopyOrApplyWithIdentity<
								false, typename AdditiveMonoid::D3, typename AdditiveMonoid::D1,
								AdditiveMonoid::template Identity
							>::set( output, temp, add );
						set = true;
					}
				}
			}

			// handle row or column at destination_index
			// NOTE: This /em could be parallelised, but will probably only slow things
			//       down
#ifdef _DEBUG
			std::cout << "vxm_gather: processing destination index " << destination_index << " / "
				<< internal::getCoordinates( destination_vector ).size()
				<< ". Input matrix has " << ( matrix.col_start[ destination_index + 1 ] -
					matrix.col_start[ destination_index ] ) << " nonzeroes.\n";
#endif
			for(
				size_t k = matrix.col_start[ destination_index ];
				rc == SUCCESS &&
					k < static_cast< size_t >( matrix.col_start[ destination_index + 1 ] );
				++k
			) {
				// declare multiplication output field
				typename Multiplication::D3 result =
					add.template getIdentity< typename AdditiveMonoid::D3 >();
				// get source index
				const size_t source_index = matrix.row_index[ k ];
				// check mask
				if( input_masked &&
					!internal::getCoordinates( source_mask_vector ).template
						mask< descr >( source_index, source_mask )
				) {
#ifdef _DEBUG
					std::cout << "\t vxm_gather: skipping source index " << source_index
						<< " due to input mask\n";
#endif
					continue;
				}
				// check for sparsity at source
				if( !dense_hint ) {
					if( !internal::getCoordinates( source_vector ).assigned( source_index ) ) {
#ifdef _DEBUG
						std::cout << "\t vxm_gather: Skipping out of computation with source "
							<< "index " << source_index << " since it does not contain a nonzero\n";
#endif
						continue;
					}
				}
				// get nonzero
				typedef typename std::conditional<
					left_handed,
					typename Multiplication::D2,
					typename Multiplication::D1
				>::type RingNonzeroType;
				const RingNonzeroType nonzero =
					matrix.template getValue( k, One< RingNonzeroType >::value() );
#ifdef _DEBUG
				std::cout << "\t vxm_gather: interpreted nonzero is " << nonzero << ", "
					<< "which is the " << k << "-th nonzero and has source index "
					<< source_index << "\n";
#endif
				// check if we use source element or whether we use its index value instead
				typedef typename std::conditional<
					left_handed,
					typename Multiplication::D1,
					typename Multiplication::D2
				>::type SourceType;
				const SourceType apply_source = internal::ValueOrIndex<
					descr, SourceType, InputType1
				>::getFromArray( source, src_local_to_global, source_index );
#ifdef _DEBUG
				if( use_index ) {
					std::cout << "\t vxm_gather (use_index descriptor): apply( output, matrix "
						<< "nonzero, vector nonzero, * ) = apply( ";
				} else {
					std::cout << "\t vxm_gather: apply( output, matrix nonzero, vector "
						<< "nonzero, * ) = apply( ";
				}
				std::cout << " output, " << nonzero << ", "  << source << ", * )\n";
#endif
				//multiply
				internal::leftOrRightHandedMul<
					left_handed, typename Multiplication::D3,
					SourceType, RingNonzeroType, Multiplication
				>::mul( result, apply_source, nonzero, mul );
#ifdef _DEBUG
				std::cout << "\t vxm_gather: output (this nonzero) = " << result << "\n";
#endif

				// accumulate
#ifdef _DEBUG
				std::cout << "\t vxm_gather: foldr( " << result << ", " << output
					<< ", + );\n";
#endif
				rc = foldr( result, output, add.getOperator() );
#ifdef _DEBUG
				std::cout << "\t vxm_gather: output (sum at destination) = " << output
					<< "\n";
#endif
				set = true;

				// sanity check (but apply cannot fail)
				assert( rc == SUCCESS );
			}

#ifdef _DEBUG
			if( set ) {
				std::cout << "\t vxm_gather: local contribution to this output element at "
					<< "index " << destination_index << " will be " << output << " "
					<< "and this corresponds to an explicitly set nonzero.\n";
			} else {
				std::cout << "\t vxm_gather: local contribution to this output element at "
					<< "index " << destination_index << " will be " << output << " and this "
					<< "is an unset value.\n";
				if( already_dense_destination_vector ||
					local_destination_vector.assigned( destination_index - lower_bound )
				) {
					std::cout << "\t(old value " << destination_element << " will remain "
						<< "unmodified.)\n";
				} else {
					std::cout << "\t(no old value existed so the output vector will remain "
						<< "unset at this index.)\n";
				}
			}
#endif
			// finally, accumulate in output
			if( explicit_zero || set ) {
#ifdef _DEBUG
				std::cout << "\taccumulating " << output << " into output vector...\n";
#endif
				if( already_dense_destination_vector ||
					local_destination_vector.assign( destination_index - lower_bound )
				) {
#ifdef _DEBUG
					std::cout << "\tfoldl( " << destination_element << ", " << output << ", "
					       << "add.getOperator() );, destination_element = ";
#endif
					rc = foldl( destination_element, output, add.getOperator() );
#ifdef _DEBUG
					std::cout << destination_element << "\n";
#endif
				} else {
#ifdef _DEBUG
					std::cout << "\toutput vector element was previously not set. Old "
						<< "(possibly uninitialised value) " << destination_element << " will "
						<< "now be set to " << output << ", result (after, possibly, casting): ";
#endif
					destination_element = static_cast< IOType >( output );
#ifdef _DEBUG
					std::cout << destination_element << "\n";
#endif
				}
			}
		}

		template<
			Descriptor descr,
			bool masked,
			bool input_masked,
			bool left_handed,
			bool using_semiring,
			template< typename > class One,
			class AdditiveMonoid,
			class Multiplication,
			typename IOType,
			typename InputType1,
			typename InputType2,
			typename InputType3,
			typename InputType4,
			typename RIT,
			typename CIT,
			typename NIT,
			typename Coords
		>
		RC vxm_generic(
			Vector< IOType, nonblocking, Coords > &u,
			const Vector< InputType3, nonblocking, Coords > &mask,
			const Vector< InputType1, nonblocking, Coords > &v,
			const Vector< InputType4, nonblocking, Coords > &v_mask,
			const Matrix< InputType2, nonblocking, RIT, CIT, NIT > &A,
			const AdditiveMonoid &add,
			const Multiplication &mul,
			const Phase &phase,
			const std::function< size_t( size_t ) > row_l2g,
			const std::function< size_t( size_t ) > row_g2l,
			const std::function< size_t( size_t ) > col_l2g,
			const std::function< size_t( size_t ) > col_g2l
		) {
			// type sanity checking
			NO_CAST_ASSERT( ( descr > internal::MAX_DESCRIPTOR_VALUE ||
					!(descr & descriptors::no_casting) ||
					std::is_same< InputType3, bool >::value
				), "vxm (any variant)",
				"Mask type is not boolean" );
			NO_CAST_ASSERT( ( descr > internal::MAX_DESCRIPTOR_VALUE ||
					!(descr & descriptors::no_casting) ||
					!left_handed ||
					std::is_same< InputType1, typename Multiplication::D1 >::value
				), "vxm (any variant)",
				"Input vector type does not match multiplicative operator first "
				"input domain" );
			NO_CAST_ASSERT( ( descr > internal::MAX_DESCRIPTOR_VALUE ||
					!(descr & descriptors::no_casting) ||
					left_handed ||
					std::is_same< InputType2, typename Multiplication::D1 >::value
				), "vxm (any variant)",
				"Input vector type does not match multiplicative operator second "
				"input domain" );
			NO_CAST_ASSERT( ( descr > internal::MAX_DESCRIPTOR_VALUE ||
					!(descr & descriptors::no_casting) ||
					!left_handed ||
					std::is_same< InputType2, typename Multiplication::D2 >::value
				), "vxm (any variant)",
				"Input matrix type does not match multiplicative operator second "
				"input domain" );
			NO_CAST_ASSERT( ( descr > internal::MAX_DESCRIPTOR_VALUE ||
					!(descr & descriptors::no_casting) ||
					left_handed ||
					std::is_same< InputType1, typename Multiplication::D2 >::value
				), "vxm (any variant)",
				"Input matrix type does not match multiplicative operator first "
				"input domain" );

			RC ret = SUCCESS;

#ifdef _DEBUG
			const auto s = spmd< nonblocking >::pid();
			std::cout << s << ": nonblocking vxm called with a "
				<< descriptors::toString( descr ) << "\n";
#endif

			// get input and output vector sizes
			const size_t m = internal::getCoordinates( u ).size();
			const size_t n = internal::getCoordinates( v ).size();

			// get whether the matrix should be transposed prior to execution of this
			// vector-times-matrix operation
			constexpr bool transposed = descr & descriptors::transpose_matrix;

			// check for dimension mismatch
			if( ( transposed && ( n != ncols( A ) || m != nrows( A ) ) )
				|| ( !transposed && ( n != nrows( A ) || m != ncols( A ) ) ) ) {
#ifdef _DEBUG
				std::cout << "Mismatch of columns ( " << n << " vs. " << ncols( A )
					<< " ) or rows ( " << m << " vs. " << nrows( A ) << " ) with "
					<< "transposed value " << ((int)transposed) << "\n";
#endif
				return MISMATCH;
			}

			// check density
			if( descr & descriptors::dense ) {
				// it's safe to check the number of nonzeroes for the input vector and its
				// mask since both of them are read-only in the current design for
				// nonblocking execution
				if( nnz( v ) < size( v ) ) {
#ifdef _DEBUG
					std::cout << "\t Dense descriptor given but input vector was sparse\n";
#endif
					return ILLEGAL;
				}
				if( size( v_mask ) > 0 && nnz( v_mask ) < size( v_mask ) ) {
#ifdef _DEBUG
					std::cout << "\t Dense descriptor given but input mask has sparse "
						<< "structure\n";
#endif
					return ILLEGAL;
				}
			}

			// check mask
			if( masked ) {
				if( (transposed && internal::getCoordinates( mask ).size() != nrows( A ) ) ||
					( !transposed && internal::getCoordinates( mask ).size() != ncols( A ) )
				) {
#ifdef _DEBUG
					std::cout << "Mismatch of mask size ( "
						<< internal::getCoordinates( mask ).size() << " ) versus matrix rows "
						<< "or columns ( " << nrows( A ) << " or " << ncols( A ) << " with "
						<< "transposed value " << ((int)transposed) << "\n";
#endif
					return MISMATCH;
				}
			}

			// handle resize phase
			if( phase == RESIZE ) {
				return SUCCESS;
			}

			// get raw pointers
			assert( phase == EXECUTE );
			const InputType1 * __restrict__ const x = internal::getRaw( v );
			const InputType3 * __restrict__ const z = internal::getRaw( mask );
			const InputType4 * __restrict__ const vm = internal::getRaw( v_mask );
			IOType * __restrict__ const y = internal::getRaw( u );

			// check for illegal arguments
			if( !(descr & descriptors::safe_overlap) &&
				reinterpret_cast< const void * >( y ) ==
					reinterpret_cast< const void * >( x )
			) {
				std::cerr << "Warning: grb::internal::vxm_generic called with overlapping "
					<< "input and output vectors.\n";
				return OVERLAP;
			}
			if( masked && (reinterpret_cast<const void*>(y) ==
				reinterpret_cast<const void*>(z))
			) {
				std::cerr << "Warning: grb::internal::vxm_generic called with overlapping "
					<< "mask and output vectors.\n";
				return OVERLAP;
			}

#ifdef _DEBUG
			std::cout << s << ": performing SpMV / SpMSpV using an " << nrows( A )
				<< " by " << ncols( A ) << " matrix holding " << nnz( A )
				<< " nonzeroes.\n";
#endif

			// in the current design for nonblocking execution, the input vectors of
			// vxm_generic // cannot be overwritten by another stage of the same
			// pipeline, and therefore, it's safe to rely on the global coordinates of
			// the input vectors, as they are read-only this property is of special
			// importance when handling matrices of size "m" x "n" since the mismatch
			// between "m" and "n" requires special handling for the local coordinates of
			// the input vectors, the current design relies on the size of the output
			// vector which should match the sizes of all other vectors in the pipeline
			// the size of the input vector does not have to match the size of the other
			// vectors as long as the input vectors are read-only

			constexpr const bool dense_descr = descr & descriptors::dense;

			internal::Pipeline::stage_type func = [
				&u, &mask, &v, &v_mask, &A, &add, &mul,
				row_l2g, row_g2l, col_l2g, col_g2l,
				y, x, z, vm
#ifdef _DEBUG
				, s
#endif
			] (
				internal::Pipeline &pipeline,
				const size_t lower_bound, const size_t upper_bound
			) {
#ifdef _NONBLOCKING_DEBUG
				#pragma omp critical
				std::cout << "\t\tExecution of stage vxm_generic in the range("
					<< lower_bound << ", " << upper_bound << ")" << std::endl;
#endif
				(void) pipeline;

				RC rc = SUCCESS;

				Coords local_u, local_mask;
				const size_t local_n = upper_bound - lower_bound;
				size_t local_mask_nz = local_n;

#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
				const bool already_dense_vectors = dense_descr ||
					pipeline.allAlreadyDenseVectors();
#else
				constexpr const bool already_dense_vectors = dense_descr;
#endif

				bool already_dense_output = true;
				bool already_dense_output_mask = true;

				if( !already_dense_vectors ) {
#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
					already_dense_output = pipeline.containsAlreadyDenseVector(
						&internal::getCoordinates( u ) );
					if( !already_dense_output ) {
#else
						already_dense_output = false;
#endif
						local_u = internal::getCoordinates( u ).asyncSubset( lower_bound,
							upper_bound );
#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
					}
#endif
					if( masked ) {
#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
						already_dense_output_mask = pipeline.containsAlreadyDenseVector(
							&internal::getCoordinates( mask ) );
						if( !already_dense_output_mask ) {
#else
							already_dense_output_mask = false;
#endif
							local_mask = internal::getCoordinates( mask ).asyncSubset( lower_bound,
								upper_bound );
							local_mask_nz = local_mask.nonzeroes();
#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
						}
#endif
					}
				}

				// check if transpose is required
				if( descr & descriptors::transpose_matrix ) {
					// start compute u=vA^T
#ifdef _DEBUG
					std::cout << s << ": in u=vA^T=Av variant\n";
#endif

					// start u=vA^T using CRS
					// matrix = &(A.CRS);
					// TODO internal issue #193
					if( !masked || (descr & descriptors::invert_mask) ) {
						// loop over all columns of the input matrix (can be done in parallel):
#ifdef _DEBUG
						std::cout << s << ": in full CRS variant (gather)\n";
#endif

						for( size_t i = lower_bound; i < upper_bound; i++ ) {
#ifdef GRB_BOOLEAN_DISPATCHER
							boolean_dispatcher_vxm_inner_kernel_gather<
#else
							vxm_inner_kernel_gather<
#endif
									descr, masked, input_masked, left_handed, One
								>(
									already_dense_output, already_dense_output_mask,
									rc, lower_bound, local_u, local_mask,
									u, y[ i ], i, v, x, nrows( A ), internal::getCRS( A ),
									mask, z, v_mask, vm, add, mul,
									row_l2g, col_l2g, col_g2l
								);
						}

					} else {
#ifdef _DEBUG
						std::cout << s << ": in masked CRS variant (gather). Mask has "
							<< local_mask_nz << " nonzeroes and size " << local_n << ":\n";
						for( size_t k = 0; k < local_mask_nz; ++k ) {
							std::cout << " "
							<< ( ( already_dense_output_mask ? k : local_mask.index( k ) ) +
								lower_bound );
						}
						std::cout << "\n";
#endif
						assert( masked );

						for( size_t k = 0; k < local_mask_nz; ++k ) {
							const size_t i =
								( already_dense_output_mask ? k : local_mask.index( k ) ) +
								lower_bound;
							assert( i < nrows(A) );

#ifdef GRB_BOOLEAN_DISPATCHER
							boolean_dispatcher_vxm_inner_kernel_gather<
#else
							vxm_inner_kernel_gather<
#endif
									descr, false, input_masked, left_handed, One
								>(
									already_dense_output, already_dense_output_mask,
									rc, lower_bound, local_u, local_mask,
									u, y[ i ], i, v, x, nrows( A ), internal::getCRS( A ),
									mask, z, v_mask, vm, add, mul,
									row_l2g, col_l2g, col_g2l
								);
						}
					}
					// end compute u=vA^T
				} else {
#ifdef _DEBUG
					std::cout << s << ": in u=vA=A^Tv variant\n";
#endif
					// start u=vA using CCS
#ifdef _DEBUG
					std::cout << s << ": in column-major vector times matrix variant (u=vA)\n"
						<< "\t(this variant relies on the gathering inner kernel)\n";
#endif

					// if not transposed, then CCS is the data structure to go:
					// TODO internal issue #193
					if( !masked || (descr & descriptors::invert_mask) ) {
#ifdef _DEBUG
						std::cout << s << ": loop over all input matrix columns\n";
#endif

						for( size_t j = lower_bound; j < upper_bound; j++ ) {
#ifdef GRB_BOOLEAN_DISPATCHER
							boolean_dispatcher_vxm_inner_kernel_gather<
#else
							vxm_inner_kernel_gather<
#endif
									descr, masked, input_masked, left_handed, One
								>(
									already_dense_output, already_dense_output_mask,
									rc, lower_bound, local_u, local_mask,
									u, y[ j ], j, v, x, nrows( A ), internal::getCCS( A ),
									mask, z, v_mask, vm, add, mul,
									row_l2g, row_g2l, col_l2g
								);
						}
					} else {
						// loop only over the nonzero masks (can still be done in parallel!)
#ifdef _DEBUG
						std::cout << s << ": loop over mask indices\n";
#endif
						assert( masked );

						for( size_t k = 0; k < local_mask_nz; ++k ) {
							const size_t j =
								( already_dense_output_mask ? k : local_mask.index( k ) ) + lower_bound;
#ifdef GRB_BOOLEAN_DISPATCHER
							boolean_dispatcher_vxm_inner_kernel_gather<
#else
							vxm_inner_kernel_gather<
#endif
									descr, masked, input_masked, left_handed, One
								>(
									already_dense_output, already_dense_output_mask,
									rc, lower_bound, local_u, local_mask,
									u, y[ j ], j, v, x, nrows( A ), internal::getCCS( A ),
									mask, z, v_mask, vm, add, mul,
									row_l2g, row_g2l, col_l2g
								);
						}
					}
					// end computing u=vA
				}
#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
				if( !already_dense_output ) {
#else
				if( !already_dense_vectors ) {
#endif
					internal::getCoordinates( u ).asyncJoinSubset( local_u, lower_bound,
						upper_bound );
				}

				return rc;
			};

			// since the local coordinates are never used for the input vector and the
			// input mask they are added only for verification of legal usage of the
			// dense descriptor
			ret = ret ? ret : internal::le.addStage(
					std::move( func ),
					internal::Opcode::BLAS2_VXM_GENERIC,
					size( u ), sizeof( IOType ), dense_descr, true,
					&u, nullptr, &internal::getCoordinates( u ), nullptr,
					&v,
					masked ? &mask : nullptr,
					input_masked ? &v_mask : nullptr,
					nullptr,
					&internal::getCoordinates( v ),
					masked ? &internal::getCoordinates( mask ) : nullptr,
					input_masked ? &internal::getCoordinates( v_mask ) : nullptr,
					nullptr,
					&A
				);

#ifdef _NONBLOCKING_DEBUG
			std::cout << "\t\tStage added to a pipeline: vxm_generic" << std::endl;
#endif

#ifdef _DEBUG
			std::cout << s << ": exiting SpMV / SpMSpV.\n" << std::flush;
#endif
			return ret;
		}

	} // namespace internal

	template<
		Descriptor descr = descriptors::no_operation,
		class Ring,
		typename IOType,
		typename InputType1,
		typename InputType2,
		typename InputType3,
		typename RIT,
		typename CIT,
		typename NIT,
		typename Coords
	>
	RC vxm(
		Vector< IOType, nonblocking, Coords > &u,
		const Vector< InputType3, nonblocking, Coords > &mask,
		const Vector< InputType1, nonblocking, Coords > &v,
		const Matrix< InputType2, nonblocking, RIT, CIT, NIT > &A,
		const Ring &ring = Ring(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			grb::is_semiring< Ring >::value, void
		>::type * const = nullptr
	) {
		const Vector< bool, nonblocking, Coords > empty_mask( 0 );
		return vxm< descr, true, false >( u, mask, v, empty_mask, A, ring, phase );
	}

	template<
		Descriptor descr = descriptors::no_operation,
		class AdditiveMonoid,
		class MultiplicativeOperator,
		typename IOType,
		typename InputType1,
		typename InputType2,
		typename InputType3,
		typename RIT,
		typename CIT,
		typename NIT,
		typename Coords
	>
	RC vxm(
		Vector< IOType, nonblocking, Coords > &u,
		const Vector< InputType3, nonblocking, Coords > &mask,
		const Vector< InputType1, nonblocking, Coords > &v,
		const Matrix< InputType2, nonblocking, RIT, CIT, NIT > &A,
		const AdditiveMonoid &add = AdditiveMonoid(),
		const MultiplicativeOperator &mul = MultiplicativeOperator(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			grb::is_monoid< AdditiveMonoid >::value &&
			grb::is_operator< MultiplicativeOperator >::value &&
			!grb::is_object< IOType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			!grb::is_object< InputType3 >::value &&
			!std::is_same< InputType2, void >::value,
		void >::type * const = nullptr
	) {
		const grb::Vector< bool, nonblocking, Coords > empty_mask( 0 );
		return vxm< descr, true, false >( u, mask, v, empty_mask, A, add, mul,
			phase );
	}

	template<
		Descriptor descr = descriptors::no_operation,
		bool output_may_be_masked = true,
		bool input_may_be_masked = true,
		class Ring,
		typename IOType,
		typename InputType1,
		typename InputType2,
		typename InputType3,
		typename InputType4,
		typename RIT,
		typename CIT,
		typename NIT,
		typename Coords
	>
	RC vxm(
		Vector< IOType, nonblocking, Coords > &u,
		const Vector< InputType3, nonblocking, Coords > &mask,
		const Vector< InputType1, nonblocking, Coords > &v,
		const Vector< InputType4, nonblocking, Coords > &v_mask,
		const Matrix< InputType2, nonblocking, RIT, CIT, NIT > &A,
		const Ring &ring = Ring(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			grb::is_semiring< Ring >::value,
		void >::type * const = nullptr
	) {
		constexpr bool left_sided = true;
		if( output_may_be_masked && size( v_mask ) == 0 && size( mask ) > 0 ) {

			return internal::vxm_generic<
					descr, true, false, left_sided, true, Ring::template One
				>(
					u, mask, v, v_mask, A,
					ring.getAdditiveMonoid(), ring.getMultiplicativeOperator(),
					phase,
					[]( const size_t i ) {
						return i;
					},
					[]( const size_t i ) {
						return i;
					},
					[]( const size_t i ) {
						return i;
					},
					[]( const size_t i ) {
						return i;
					}
				);
		} else if( input_may_be_masked && size( mask ) == 0 && size( v_mask ) > 0 ) {
			return internal::vxm_generic<
					descr, false, true, left_sided, true, Ring::template One
				>(
					u, mask, v, v_mask, A,
					ring.getAdditiveMonoid(), ring.getMultiplicativeOperator(),
					phase,
					[]( const size_t i ) {
						return i;
					},
					[]( const size_t i ) {
						return i;
					},
					[]( const size_t i ) {
						return i;
					},
					[]( const size_t i ) {
						return i;
					}
				);
		} else if( output_may_be_masked && input_may_be_masked && size( mask ) > 0 && size( v_mask ) > 0 ) {
			return internal::vxm_generic<
					descr, true, true, left_sided, true, Ring::template One
				>(
					u, mask, v, v_mask, A,
					ring.getAdditiveMonoid(), ring.getMultiplicativeOperator(),
					phase,
					[]( const size_t i ) {
						return i;
					},
					[]( const size_t i ) {
						return i;
					},
					[]( const size_t i ) {
						return i;
					},
					[]( const size_t i ) {
						return i;
					}
				);
		} else {
			assert( size( mask ) == 0 );
			assert( size( v_mask ) == 0 );
			return internal::vxm_generic<
					descr, false, false, left_sided, true, Ring::template One
				>(
					u, mask, v, v_mask, A,
					ring.getAdditiveMonoid(), ring.getMultiplicativeOperator(),
					phase,
					[]( const size_t i ) {
						return i;
					},
					[]( const size_t i ) {
						return i;
					},
					[]( const size_t i ) {
						return i;
					},
					[]( const size_t i ) {
						return i;
					}
				);
		}
	}

	template<
		Descriptor descr = descriptors::no_operation,
		class Ring,
		typename Coords,
		typename RIT,
		typename CIT,
		typename NIT,
		typename IOType = typename Ring::D4,
		typename InputType1 = typename Ring::D1,
		typename InputType2 = typename Ring::D2
	>
	RC vxm(
		Vector< IOType, nonblocking, Coords > &u,
		const Vector< InputType1, nonblocking, Coords > &v,
		const Matrix< InputType2, nonblocking, RIT, CIT, NIT > &A,
		const Ring &ring = Ring(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			grb::is_semiring< Ring >::value, void
		>::type * const = nullptr
	) {
		const Vector< bool, nonblocking, Coords > empty_mask( 0 );
		return vxm< descr, false, false >( u, empty_mask, v, empty_mask, A, ring,
			phase );
	}

	template<
		Descriptor descr = descriptors::no_operation,
		class AdditiveMonoid,
		class MultiplicativeOperator,
		typename IOType,
		typename InputType1,
		typename InputType2,
		typename RIT,
		typename CIT,
		typename NIT,
		typename Coords
	>
	RC vxm(
		Vector< IOType, nonblocking, Coords > &u,
		const Vector< InputType1, nonblocking, Coords > &v,
		const Matrix< InputType2, nonblocking, RIT, CIT, NIT > &A,
		const AdditiveMonoid &add = AdditiveMonoid(),
		const MultiplicativeOperator &mul = MultiplicativeOperator(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			grb::is_monoid< AdditiveMonoid >::value &&
			grb::is_operator< MultiplicativeOperator >::value &&
			!grb::is_object< IOType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			!std::is_same< InputType2, void >::value,
		void >::type * const = nullptr
	) {
		const Vector< bool, nonblocking, Coords > empty_mask( 0 );
		return vxm< descr, false, false >( u, empty_mask, v, empty_mask, A, add, mul,
			phase );
	}

	template<
		Descriptor descr = descriptors::no_operation,
		class Ring,
		typename Coords,
		typename RIT,
		typename CIT,
		typename NIT,
		typename IOType = typename Ring::D4,
		typename InputType1 = typename Ring::D1,
		typename InputType2 = typename Ring::D2,
		typename InputType3 = bool
	>
	RC mxv(
		Vector< IOType, nonblocking, Coords > &u,
		const Vector< InputType3, nonblocking, Coords > &mask,
		const Matrix< InputType2, nonblocking, RIT, CIT, NIT > &A,
		const Vector< InputType1, nonblocking, Coords > &v,
		const Ring &ring,
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			grb::is_semiring< Ring >::value, void
		>::type * const = nullptr
	) {
		const Vector< bool, nonblocking, Coords > empty_mask( 0 );
		return mxv< descr, true, false >( u, mask, A, v, empty_mask, ring, phase );
	}

	template<
		Descriptor descr = descriptors::no_operation,
		bool output_may_be_masked = true,
		bool input_may_be_masked = true,
		class Ring,
		typename IOType,
		typename InputType1,
		typename InputType2,
		typename InputType3,
		typename InputType4,
		typename RIT,
		typename CIT,
		typename NIT,
		typename Coords
	>
	RC mxv(
		Vector< IOType, nonblocking, Coords > &u,
		const Vector< InputType3, nonblocking, Coords > &mask,
		const Matrix< InputType2, nonblocking, RIT, CIT, NIT > &A,
		const Vector< InputType1, nonblocking, Coords > &v,
		const Vector< InputType4, nonblocking, Coords > &v_mask,
		const Ring &ring,
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			grb::is_semiring< Ring >::value,
		void >::type * const = nullptr
	) {
		constexpr Descriptor new_descr = descr ^ descriptors::transpose_matrix;
		constexpr bool left_sided = false;
		if( output_may_be_masked && ( size( v_mask ) == 0 && size( mask ) > 0 ) ) {

			return internal::vxm_generic<
					new_descr, true, false, left_sided, true, Ring::template One
				>(
					u, mask, v, v_mask, A,
					ring.getAdditiveMonoid(), ring.getMultiplicativeOperator(),
					phase,
					[]( const size_t i ) {
						return i;
					},
					[]( const size_t i ) {
						return i;
					},
					[]( const size_t i ) {
						return i;
					},
					[]( const size_t i ) {
						return i;
					}
				);
		} else if( input_may_be_masked && ( size( mask ) == 0 &&
			size( v_mask ) > 0 )
		) {
			return internal::vxm_generic<
					new_descr, false, true, left_sided, true, Ring::template One
				>(
					u, mask, v, v_mask, A,
					ring.getAdditiveMonoid(), ring.getMultiplicativeOperator(),
					phase,
					[]( const size_t i ) {
						return i;
					},
					[]( const size_t i ) {
						return i;
					},
					[]( const size_t i ) {
						return i;
					},
					[]( const size_t i ) {
						return i;
					}
				);
		} else if( output_may_be_masked && input_may_be_masked && size( mask ) > 0 &&
			size( v_mask ) > 0
		) {
			return internal::vxm_generic<
					new_descr, true, true, left_sided, true, Ring::template One
				>(
					u, mask, v, v_mask, A,
					ring.getAdditiveMonoid(), ring.getMultiplicativeOperator(),
					phase,
					[]( const size_t i ) {
						return i;
					},
					[]( const size_t i ) {
						return i;
					},
					[]( const size_t i ) {
						return i;
					},
					[]( const size_t i ) {
						return i;
					}
				);
		} else {
			assert( size( mask ) == 0 );
			assert( size( v_mask ) == 0 );
			return internal::vxm_generic<
					new_descr, false, false, left_sided, true, Ring::template One
				>(
					u, mask, v, v_mask, A,
					ring.getAdditiveMonoid(), ring.getMultiplicativeOperator(),
					phase,
					[]( const size_t i ) {
						return i;
					},
					[]( const size_t i ) {
						return i;
					},
					[]( const size_t i ) {
						return i;
					},
					[]( const size_t i ) {
						return i;
					}
				);
		}
	}

	template<
		Descriptor descr = descriptors::no_operation,
		class Ring,
		typename Coords,
		typename RIT,
		typename CIT,
		typename NIT,
		typename IOType = typename Ring::D4,
		typename InputType1 = typename Ring::D1,
		typename InputType2 = typename Ring::D2
	>
	RC mxv(
		Vector< IOType, nonblocking, Coords > &u,
		const Matrix< InputType2, nonblocking, RIT, CIT, NIT > &A,
		const Vector< InputType1, nonblocking, Coords > &v,
		const Ring &ring,
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			grb::is_semiring< Ring >::value,
		void >::type * const = nullptr
	) {
		const Vector< bool, nonblocking, Coords > empty_mask( 0 );
		return mxv< descr, false, false >( u, empty_mask, A, v, empty_mask, ring,
			phase );
	}

	template<
		Descriptor descr = descriptors::no_operation,
		class AdditiveMonoid,
		class MultiplicativeOperator,
		typename IOType,
		typename InputType1,
		typename InputType2,
		typename RIT,
		typename CIT,
		typename NIT,
		typename Coords
	>
	RC mxv(
		Vector< IOType, nonblocking, Coords > &u,
		const Matrix< InputType2, nonblocking, RIT, CIT, NIT > &A,
		const Vector< InputType1, nonblocking, Coords > &v,
		const AdditiveMonoid &add = AdditiveMonoid(),
		const MultiplicativeOperator &mul = MultiplicativeOperator(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			grb::is_monoid< AdditiveMonoid >::value &&
			grb::is_operator< MultiplicativeOperator >::value &&
			!grb::is_object< IOType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			!std::is_same< InputType2, void >::value,
		void >::type * const = nullptr
	) {
		const Vector< bool, nonblocking, Coords > empty_mask( 0 );
		return mxv< descr, false, false >( u, empty_mask, A, v, empty_mask, add, mul,
			phase );
	}

	template<
		Descriptor descr = descriptors::no_operation,
		bool output_may_be_masked = true,
		bool input_may_be_masked = true,
		class AdditiveMonoid,
		class MultiplicativeOperator,
		typename IOType,
		typename InputType1,
		typename InputType2,
		typename InputType3,
		typename InputType4,
		typename RIT,
		typename CIT,
		typename NIT,
		typename Coords
	>
	RC vxm(
		Vector< IOType, nonblocking, Coords > &u,
		const Vector< InputType3, nonblocking, Coords > &mask,
		const Vector< InputType1, nonblocking, Coords > &v,
		const Vector< InputType4, nonblocking, Coords > &v_mask,
		const Matrix< InputType2, nonblocking, RIT, CIT, NIT > &A,
		const AdditiveMonoid &add = AdditiveMonoid(),
		const MultiplicativeOperator &mul = MultiplicativeOperator(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			grb::is_monoid< AdditiveMonoid >::value &&
			grb::is_operator< MultiplicativeOperator >::value &&
			!grb::is_object< IOType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			!grb::is_object< InputType3 >::value &&
			!grb::is_object< InputType4 >::value &&
			!std::is_same< InputType2, void >::value,
		void >::type * const = nullptr
	) {
		static_assert( !(descr & descriptors::add_identity), "Cannot add an "
			"identity if no concept of `one' is known. Suggested fix: use a semiring "
			"instead." );
		constexpr bool left_sided = true;
		if( output_may_be_masked && size( v_mask ) == 0 && size( mask ) > 0 ) {
			return internal::vxm_generic<
					descr, true, false, left_sided, false, AdditiveMonoid::template Identity
				>(
					u, mask, v, v_mask, A, add, mul, phase,
					[]( const size_t i ) {
						return i;
					},
					[]( const size_t i ) {
						return i;
					},
					[]( const size_t i ) {
						return i;
					},
					[]( const size_t i ) {
						return i;
					}
				);
		} else if( input_may_be_masked && size( v_mask ) > 0 && size( mask ) == 0 ) {
			return internal::vxm_generic<
					descr, false, true, left_sided, false, AdditiveMonoid::template Identity
				>(
					u, mask, v, v_mask, A, add, mul, phase,
					[]( const size_t i ) {
						return i;
					},
					[]( const size_t i ) {
						return i;
					},
					[]( const size_t i ) {
						return i;
					},
					[]( const size_t i ) {
						return i;
					}
				);
		} else if( output_may_be_masked && input_may_be_masked && size( mask ) > 0 &&
			size( v_mask ) > 0
		) {
			return internal::vxm_generic<
					descr, true, true, left_sided, false, AdditiveMonoid::template Identity
				>(
					u, mask, v, v_mask, A, add, mul, phase,
					[]( const size_t i ) {
						return i;
					},
					[]( const size_t i ) {
						return i;
					},
					[]( const size_t i ) {
						return i;
					},
					[]( const size_t i ) {
						return i;
					}
				);
		} else {
			assert( size( mask ) == 0 );
			assert( size( v_mask ) == 0 );
			return internal::vxm_generic<
					descr, false, false, left_sided, false, AdditiveMonoid::template Identity
				>(
					u, mask, v, v_mask, A, add, mul, phase,
					[]( const size_t i ) {
						return i;
					},
					[]( const size_t i ) {
						return i;
					},
					[]( const size_t i ) {
						return i;
					},
					[]( const size_t i ) {
						return i;
					}
				);
		}
	}

	template<
		Descriptor descr = descriptors::no_operation,
		bool output_may_be_masked = true,
		bool input_may_be_masked = true,
		class AdditiveMonoid,
		class MultiplicativeOperator,
		typename IOType,
		typename InputType1,
		typename InputType2,
		typename InputType3,
		typename InputType4,
		typename RIT,
		typename CIT,
		typename NIT,
		typename Coords
	>
	RC mxv(
		Vector< IOType, nonblocking, Coords > &u,
		const Vector< InputType3, nonblocking, Coords > &mask,
		const Matrix< InputType2, nonblocking, RIT, CIT, NIT > &A,
		const Vector< InputType1, nonblocking, Coords > &v,
		const Vector< InputType4, nonblocking, Coords > &v_mask,
		const AdditiveMonoid &add = AdditiveMonoid(),
		const MultiplicativeOperator &mul = MultiplicativeOperator(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			grb::is_monoid< AdditiveMonoid >::value &&
			grb::is_operator< MultiplicativeOperator >::value &&
			!grb::is_object< IOType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			!grb::is_object< InputType3 >::value &&
			!grb::is_object< InputType4 >::value &&
			!std::is_same< InputType2, void >::value,
		void >::type * const = nullptr
	) {
		static_assert( !(descr & descriptors::add_identity), "Cannot add an identity "
			"if no concept of `1' is known. Suggested fix: use a semiring "
			"instead." );
		constexpr Descriptor new_descr = descr ^ descriptors::transpose_matrix;
		constexpr bool left_sided = false;
		if( output_may_be_masked && size( v_mask ) == 0 && size( mask ) > 0 ) {
			return internal::vxm_generic<
					new_descr, true, false, left_sided, false, AdditiveMonoid::template Identity
				>(
					u, mask, v, v_mask, A, add, mul, phase,
					[]( const size_t i ) {
						return i;
					},
					[]( const size_t i ) {
						return i;
					},
					[]( const size_t i ) {
						return i;
					},
					[]( const size_t i ) {
						return i;
					}
				);
		} else if( input_may_be_masked && size( mask ) == 0 &&
			size( v_mask ) > 0
		) {
			return internal::vxm_generic<
					new_descr, false, true, left_sided, false, AdditiveMonoid::template Identity
				>(
					u, mask, v, v_mask, A, add, mul, phase,
					[]( const size_t i ) {
						return i;
					},
					[]( const size_t i ) {
						return i;
					},
					[]( const size_t i ) {
						return i;
					},
					[]( const size_t i ) {
						return i;
					}
				);
		} else if( output_may_be_masked && input_may_be_masked && size( mask ) > 0 &&
			size( v_mask ) > 0
		) {
			return internal::vxm_generic<
					new_descr, true, true, left_sided, false, AdditiveMonoid::template Identity
				>(
					u, mask, v, v_mask, A, add, mul, phase,
					[]( const size_t i ) {
						return i;
					},
					[]( const size_t i ) {
						return i;
					},
					[]( const size_t i ) {
						return i;
					},
					[]( const size_t i ) {
						return i;
					}
				);
		} else {
			assert( size( mask ) == 0 );
			assert( size( v_mask ) == 0 );
			return internal::vxm_generic<
					new_descr, false, false, left_sided, false, AdditiveMonoid::template Identity
				>(
					u, mask, v, v_mask, A, add, mul, phase,
					[]( const size_t i ) {
						return i;
					},
					[]( const size_t i ) {
						return i;
					},
					[]( const size_t i ) {
						return i;
					},
					[]( const size_t i ) {
						return i;
					}
				);
		}
	}

#if 0 //TODO remove?
	namespace internal {

		template<
			typename DataType,
			typename RIT,
			typename CIT,
			typename NIT,
			typename fwd_iterator
		>
		void addToCRS(
			const Matrix< DataType, nonblocking, RIT, CIT, NIT > &A,
			const fwd_iterator start,
			const fwd_iterator end
		) {

		}
	}
#endif

	template<
		class ActiveDistribution,
		typename Func,
		typename DataType,
		typename RIT,
		typename CIT,
		typename NIT
	>
	RC eWiseLambda(
		const Func f,
		const Matrix< DataType, nonblocking, RIT, CIT, NIT > &A,
		const size_t s,
		const size_t P
	) {
		if( internal::NONBLOCKING::warn_if_not_native &&
			config::PIPELINE::warn_if_not_native
		) {
			std::cerr << "Warning: eWiseLambda (nonblocking, matrix variant) currently "
				<< "delegates to a blocking implementation.\n"
				<< "         Further similar such warnings will be suppressed.\n";
			internal::NONBLOCKING::warn_if_not_native = false;
		}

		// nonblocking execution is not supported
		// first, execute any computation that is not completed
		internal::le.execution();

		// second, delegate to the reference backend
		return eWiseLambda< ActiveDistribution, Func, DataType, RIT, CIT, NIT >(
			f, internal::getRefMatrix( A ), s, P );
	}

	template<
		typename Func,
		typename DataType1,
		typename RIT,
		typename CIT,
		typename NIT,
		typename DataType2,
		typename Coords,
		typename... Args
	>
	RC eWiseLambda(
		const Func f,
		const Matrix< DataType1, nonblocking, RIT, CIT, NIT > &A,
		const Vector< DataType2, nonblocking, Coords > &x,
		Args... args
	) {
		// do size checking
		if( !( size( x ) == nrows( A ) || size( x ) == ncols( A ) ) ) {
			std::cerr << "Mismatching dimensions: given vector of size " << size( x )
				<< " has nothing to do with either matrix dimension (" << nrows( A )
				<< " nor " << ncols( A ) << ").\n";
			return MISMATCH;
		}

		return eWiseLambda( f, A, args... );
	}

	/** @} */

} // namespace grb

#undef NO_CAST_ASSERT

#endif // end _H_GRB_NONBLOCKING_BLAS2

