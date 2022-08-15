
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
 * Implements the BLAS-2 API for the reference and reference_omp backends.
 *
 * @author A. N. Yzelman
 * @date 5th of December 2016
 */

#if !defined _H_GRB_REFERENCE_BLAS2 || defined _H_GRB_REFERENCE_OMP_BLAS2
#define _H_GRB_REFERENCE_BLAS2

#include <limits>
#include <algorithm>
#include <type_traits>

#include <graphblas/base/blas2.hpp>
#include <graphblas/blas0.hpp>
#include <graphblas/descriptors.hpp>
#include <graphblas/internalops.hpp>
#include <graphblas/ops.hpp>

#include "compressed_storage.hpp"
#include "coordinates.hpp"
#include "forward.hpp"
#include "matrix.hpp"
#include "vector.hpp"

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

	/**
	 * \addtogroup reference
	 * @{
	 */

	// put the generic mxv implementation in an internal namespace
	namespace internal {

#ifndef _H_GRB_REFERENCE_OMP_BLAS2
		/**
		 * Class that during an SpMV is reponsible for resolving the add_identity
		 * descriptor.
		 *
		 * This base class assumes the SpMV was requested using an impure semiring--
		 * i.e., by using an additive monoid with any binary operator, and thus
		 * without the concept of one. In this case it is illegal to use the
		 * add_identity descriptor, and hence the below-defined apply is a no-op.
		 */
		template<
			grb::Backend backend, bool from_semiring,
			bool output_dense, bool left_handed,
			class AdditiveMonoid, class Multiplication, template< typename > class One,
			typename IOType, typename InputType, typename SourceType,
			typename Coords
		>
		class addIdentityDuringMV {

			public:

				inline static void apply(
					Vector< IOType, backend, Coords > &,
					IOType * __restrict__ const &,
					const size_t &,
					const size_t &,
					const AdditiveMonoid &,
					const Multiplication &,
					const SourceType &,
					const std::function< size_t( size_t ) > &,
					const std::function< size_t( size_t ) > &
				) {
					return;
				}

		};

		/**
		 * An abstraction for multiplying a matrix nonzero with a vector element.
		 *
		 * This multiplication can either happen with the vector element on the left
		 * side of the multiplication, or on the right side. This is what the template
		 * parameter \a left_handed encodes.
		 */
		template< bool left_handed, typename D3, typename D1, typename D2, class OP >
		class leftOrRightHandedMul;

		/** \internal Specialisation for left-sided multiplication. */
		template< typename D3, typename D1, typename D2, class OP >
		class leftOrRightHandedMul< true, D3, D1, D2, OP > {

			public:

				static void mul(
					D3 &out, const D1 &vector_element, const D2 &nonzero, const OP &mul
				) {
 #ifdef NDEBUG
					(void) apply( out, vector_element, nonzero, mul );
 #else
					assert( apply( out, vector_element, nonzero, mul ) == grb::SUCCESS );
 #endif
				}

		};

		/** \internal Specialisation for right-sided multiplication. */
		template< typename D3, typename D1, typename D2, class OP >
		class leftOrRightHandedMul< false, D3, D1, D2, OP > {

			public:

				static void mul(
					D3 &out, const D1 &vector_element, const D2 &nonzero, const OP &mul
				) {
 #ifdef NDEBUG
					(void) apply( out, nonzero, vector_element, mul );
 #else
					assert( apply( out, nonzero, vector_element, mul ) == grb::SUCCESS );
 #endif
				}

		};
#endif

		/**
		 * \internal This is the specialisation where the SpMV is guaranteed to have
		 * been called using a pure semiring. The below-defined apply hence does
		 * implement the addition of \f$ Ix \f$ to the output vector.
		 */
		template<
			bool output_dense, bool left_handed,
			class AdditiveMonoid, class Multiplication, template< typename > class One,
			typename IOType, typename InputType, typename SourceType,
			typename Coords
		>
		class addIdentityDuringMV<
			reference, true, output_dense, left_handed,
			AdditiveMonoid, Multiplication, One,
			IOType, InputType, SourceType, Coords
		> {

			public:

				static void apply(
					Vector< IOType, reference, Coords > &destination_vector,
					IOType * __restrict__ const &destination,
					const size_t &destination_range,
					const size_t &source_index,
					const AdditiveMonoid &add,
					const Multiplication &mul,
					const SourceType &input_element,
					const std::function< size_t( size_t ) > &src_local_to_global,
					const std::function< size_t( size_t ) > &dst_global_to_local
				) {
					const size_t global_location = src_local_to_global( source_index );
					const size_t id_location = dst_global_to_local( global_location );
#ifdef _DEBUG
					std::cout << "\t add_identity descriptor: input location == "
						<< source_index << " -> " << global_location << " -> " << id_location <<
						" == output location ?<? " << destination_range << "\n";
#endif
					if( id_location < destination_range ) {
						typename Multiplication::D3 temp;
						internal::CopyOrApplyWithIdentity<
							!left_handed, typename Multiplication::D3, InputType, One
						>::set(
							temp, input_element, mul
						);
						if( output_dense ||
							internal::getCoordinates( destination_vector ).assign( id_location ) ) {
#ifdef NDEBUG
							(void) foldl( destination[ id_location ], temp, add.getOperator() );
#else
							assert( foldl( destination[ id_location ], temp, add.getOperator() )
								== grb::SUCCESS );
#endif
						} else {
							internal::CopyOrApplyWithIdentity<
								false, IOType, typename Multiplication::D3,
								AdditiveMonoid::template Identity
							>::set( destination[ id_location ], temp, add );
						}
					}
				}

		};

		/**
		 * Once an entry of an output vector element is selected, this kernel
		 * computes the contribution to that element. This function is thread-safe.
		 *
		 * @tparam add_identity Whether descriptors::add_identity was set.
		 * @tparam use_index    Whether descriptors::use_index was set.
		 * @tparam dense_hint   Whether descriptors::dense was set.
		 * @tparam masked       Whether the computation has a mask on output.
		 * @tparam left_handed  Whether the vector nonzero is applied on the left-
		 *                      hand side of the matrix nonzero.
		 * @tparam mask_descriptor The descriptor containing mask interpration
		 *                         directives.
		 * @tparam Ring         Which generalised semiring is used to multiply over.
		 * @tparam IOType       The output vector element type.
		 * @tparam InputType1   The input vector element type.
		 * @tparam InputType2   The matrix nonzero element type.
		 * @tparam InputType3   The mask vector element type.
		 *
		 * @param[in,out] rc    The return code. Should be \a SUCCESS on entry. On
		 *                      successful function exit will remain \a SUCCESS, and
		 *                      will be set to a meaningful error message if the call
		 *                      failed.
		 * @param[in,out] local_update Keeps track of updates to the sparsity pattern
		 *                             of the output vector.
		 * @param[in,out] destination_vector View of the output vector.
		 * @param[in,out] destination_element The output vector element potentially
		 *                                    modified by multiplication.
		 * @param[in]     destination_index The index of the selected output vector
		 *                                  element.
		 * @param[in]     source_vector A view of the input vector.
		 * @param[in]     source        Pointer to the input vector elements.
		 * @param[in]     source_range  The number of elements in \a source.
		 * @param[in]     matrix        A view of the sparsity pattern and nonzeroes
		 *                              (if applicable) of the input matrix.
		 * @param[in]     nz            The number of nonzeroes in the matrix.
		 * @param[in]     mask_vector   A view of the mask vector. If \a masked is
		 *                              \a true, the dimensions must match that of
		 *                              \a destination_vector.
		 * @param[in]     mask          Pointer to the mask vector elements, if any.
		 * @param[in]     ring          The semiring to perform sparse matrix vector
		 *                              multiplication over.
		 * @param[in] src_local_to_global Function to map local source indices to
		 *                                global source indices.
		 * @param[in] src_global_to_local Function to map global source indices to
		 *                                local source indices.
		 * @param[in] dst_local_to_global Function to map local destination indices
		 *                                to global destination indices.
		 *
		 * For the latter three functions, an \a std::function must be given with
		 * exactly the signature <tt>size_t f(size_t index)</tt>.
		 *
		 * The value for \a rc shall only be modified if the call to this function
		 * did not succeed.
		 *
		 * This function is called by vxm_generic on both transposed input and on
		 * both CRS and CCS inputs. Type checking, dimension checking, etc. should
		 * all be done by the caller function.
		 */
		template<
			Descriptor descr,
			bool masked, // TODO issue #69
			bool input_masked, bool left_handed,
			template< typename > class One,
			class AdditiveMonoid, class Multiplication,
			typename IOType, typename InputType1, typename InputType2,
			typename InputType3, typename InputType4,
			typename Coords,
			typename RowColType, typename NonzeroType
		>
		inline void vxm_inner_kernel_gather(
			RC &rc,
#ifdef _H_GRB_REFERENCE_OMP_BLAS2
			internal::Coordinates< reference >::Update &local_update,
			size_t &asyncAssigns,
#endif
			Vector< IOType, reference, Coords > &destination_vector,
			IOType &destination_element,
			const size_t &destination_index,
			const Vector< InputType1, reference, Coords > &source_vector,
			const InputType1 * __restrict__ const &source,
			const size_t &source_range,
			const internal::Compressed_Storage< InputType2, RowColType, NonzeroType >
				&matrix,
			const size_t &nz,
			const Vector< InputType3, reference, Coords > &mask_vector,
			const InputType3 * __restrict__ const &mask,
			const Vector< InputType4, reference, Coords > &source_mask_vector,
			const InputType4 * __restrict__ const &source_mask,
			const AdditiveMonoid &add,
			const Multiplication &mul,
			const std::function< size_t( size_t ) > &src_local_to_global,
			const std::function< size_t( size_t ) > &src_global_to_local,
			const std::function< size_t( size_t ) > &dst_local_to_global
		) {
			constexpr bool add_identity = descr & descriptors::add_identity;
			constexpr bool dense_hint = descr & descriptors::dense;
			constexpr bool explicit_zero = descr & descriptors::explicit_zero;
#ifdef _DEBUG
			constexpr bool use_index = descr & descriptors::use_index;
#endif
			assert( rc == SUCCESS );
			assert( matrix.col_start[ destination_index ] <= nz );
			assert( matrix.col_start[ destination_index + 1 ] <= nz );

			// check whether we should compute output here
			if( masked ) {
				if( !internal::getCoordinates( mask_vector ).template
					mask< descr >( destination_index, mask ) ) {
#ifdef _DEBUG
					std::cout << "Masks says to skip processing destination index " <<
						destination_index << "\n";
#endif
					return;
				}
			}

			// take shortcut, if possible
			if( grb::has_immutable_nonzeroes< AdditiveMonoid >::value &&
				internal::getCoordinates( destination_vector ).
					assigned( destination_index ) &&
				destination_element != add.template getIdentity< IOType >()
			) {
				return;
			}

			// start output
			const auto &src_coordinates = internal::getCoordinates( source_vector );
			typename AdditiveMonoid::D3 output =
				add.template getIdentity< typename AdditiveMonoid::D3 >();
			bool set = false;

			// if we need to add identity, do so first:
			if( add_identity ) {
				const size_t id_location = src_global_to_local(
					dst_local_to_global( destination_index )
				);
				if( ( !input_masked ||
						internal::getCoordinates( source_mask_vector ).template
							mask< descr >( id_location, source_mask )
					) && id_location < source_range
				) {
					if( dense_hint || src_coordinates.assigned( id_location ) ) {
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
			std::cout << "vxm_gather: processing destination index "
				<< destination_index << " / "
				<< internal::getCoordinates( destination_vector ).size()
				<< ". Input matrix has " << ( matrix.col_start[ destination_index + 1 ] -
					matrix.col_start[ destination_index ] ) << " nonzeroes.\n";
#endif
			for(
				size_t k = matrix.col_start[ destination_index ];
				rc == SUCCESS && k < static_cast< size_t >(
					matrix.col_start[ destination_index + 1 ]
				);
				++k
			) {
				// declare multiplication output field
				typename Multiplication::D3 result = add.template
					getIdentity< typename AdditiveMonoid::D3 >();
				// get source index
				const size_t source_index = matrix.row_index[ k ];
				// check mask
				if( input_masked && !internal::getCoordinates( source_mask_vector ).template
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
					if( config::PREFETCHING< reference >::enabled() ) {
						size_t dist = k + 2 * config::PREFETCHING< reference >::distance();
						if( dist < nz ) {
							const size_t prefetch_target_assigned = matrix.row_index[ dist ];
							src_coordinates.prefetch_assigned( prefetch_target_assigned );
						}
						dist -= config::PREFETCHING< reference >::distance();
						if( dist < nz ) {
							const size_t prefetch_target_value = matrix.row_index[ dist ];
							if( src_coordinates.assigned( prefetch_target_value ) ) {
								src_coordinates.prefetch_value( prefetch_target_value, source );
							}
						}
					}
					if( !src_coordinates.assigned( source_index ) ) {
#ifdef _DEBUG
						std::cout << "\t vxm_gather: Skipping out of computation with source "
							<< "index " << source_index << " since it does not contain a nonzero\n";
#endif
						continue;
					}
				} else if( config::PREFETCHING< reference >::enabled() ) {
					// prefetch nonzero
					const size_t dist = k + config::PREFETCHING< reference >::distance();
					if( dist < nz ) {
						const size_t prefetch_target = matrix.row_index[ dist ];
						src_coordinates.prefetch_value( prefetch_target, source );
					}
				}
				// get nonzero
				typedef typename std::conditional<
					left_handed, typename Multiplication::D2, typename Multiplication::D1
				>::type RingNonzeroType;
				const RingNonzeroType nonzero = matrix.template
					getValue( k, One< RingNonzeroType >::value() );
#ifdef _DEBUG
				std::cout << "\t vxm_gather: interpreted nonzero is " << nonzero << ", "
					<< "which is the " << k << "-th nonzero and has source index "
					<< source_index << "\n";
#endif
				// check if we use source element or whether we use its index value instead
				typedef typename std::conditional<
					left_handed, typename Multiplication::D1, typename Multiplication::D2
				>::type SourceType;
				const SourceType apply_source = internal::ValueOrIndex<
					descr, SourceType, InputType1
				>::getFromArray( source, src_local_to_global, source_index );
#ifdef _DEBUG
				if( use_index ) {
					std::cout << "\t vxm_gather (use_index descriptor): "
						"apply( output, matrix nonzero, vector nonzero, * ) = "
						"apply( ";
				} else {
					std::cout << "\t vxm_gather: "
						"apply( output, matrix nonzero, vector nonzero, * ) = "
						"apply( ";
				}
				std::cout << " output, " << nonzero << ", "  << source << ", * )\n";
#endif
				//multiply
				internal::leftOrRightHandedMul<
					left_handed,
					typename Multiplication::D3, SourceType, RingNonzeroType,
					Multiplication
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
				if( internal::getCoordinates( destination_vector ).assigned(
					destination_index )
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
#ifdef _H_GRB_REFERENCE_OMP_BLAS2
				const bool was_already_assigned = internal::getCoordinates(
					destination_vector
				).asyncAssign( destination_index, local_update );
				if( !was_already_assigned ) {
					(void)asyncAssigns++;
				}
#else
				const bool was_already_assigned = internal::getCoordinates(
						destination_vector
					).assign( destination_index );
#endif
				if( dense_hint || was_already_assigned ) {
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

		/**
		 * Once an entry of an input vector element is selected, this kernel
		 * computes the contribution to the entire output vector. This function is
		 * thread-safe.
		 *
		 * @tparam add_identity Whether descriptors::add_identity was set.
		 * @tparam use_index    Whether descriptors::use_index was set.
		 * @tparam input_dense  Whether descriptors::dense was set and applies to the
		 *                      input vector.
		 * @tparam output_dense Whether descriptors::dense was set and applies to the
		 *                      output vector.
		 * @tparam masked       Whether the computation has a mask on output.
		 * @tparam left_handed  Whether the vector nonzero is applied on the left-hand
		 *                      size of the matrix nonzero.
		 * @tparam using_semiring Whether the original call made use of a pure
		 *                        semiring.
		 *
		 * @tparam mask_descriptor The descriptor containing mask interpration
		 *                         directives.
		 * @tparam Ring         Which generalised semiring is used to multiply over.
		 * @tparam IOType       The output vector element type.
		 * @tparam InputType1   The input vector element type.
		 * @tparam InputType2   The matrix nonzero element type.
		 * @tparam InputType3   The mask vector element type.
		 *
		 * \warning This function does not take into account the following descriptor:
		 *          #descriptors::explicit_zero. If set, the caller function must
		 *          account for this case; e.g., by zeroing out the buffer in the
		 *          preceding warning.
		 *
		 * @param[in,out] rc    The return code. Should be \a SUCCESS on entry. On
		 *                      successful function exit will remain \a SUCCESS, and
		 *                      will be set to a meaningful error message if the call
		 *                      failed.
		 * @param[in,out] local_update Keeps track of updates to the sparsity pattern
		 *                             of the output vector.
		 * @param[in,out] destination_vector View of the output vector.
		 * @param[in,out] destination  A pointer to the output vector elements.
		 * @param[in]     destination_range The number of output vector elements.
		 * @param[in]     source_vector A view of the input vector.
		 * @param[in]     source        Pointer to the input vector elements.
		 * @param[in]     source_index  The index of the selected input vector
		 *                              element.
		 * @param[in]     mask_vector   A view of the mask vector. If \a masked is
		 *                              \a true, the dimensions must match that of
		 *                              \a destination_vector.
		 * @param[in]     mask          Pointer to the mask vector elements, if any.
		 * @param[in]     ring          The semiring to perform sparse matrix vector
		 *                              multiplication over.
		 * @param[in] src_local_to_global Function to map local source indices to
		 *                                global source indices.
		 * @param[in] dst_global_to_local Function to map global destination indices
		 *                                to local destination indices.
		 *
		 * For the latter three functions, an \a std::function must be given with
		 * exactly the signature <tt>size_t f(size_t index)</tt>.
		 *
		 * The value for \a rc shall only be modified if the call to this function
		 * did not succeed.
		 *
		 * This function is called by vxm_generic on both transposed input and on
		 * both CRS and CCS inputs. Type checking, dimension checking, etc. should
		 * all be done by the caller function.
		 */
		template<
			Descriptor descr,
			bool input_dense, bool output_dense,
			bool masked, bool left_handed,
			bool using_semiring,
			template< typename > class One,
			typename IOType,
			class AdditiveMonoid, class Multiplication,
			typename InputType1, typename InputType2, typename InputType3,
			typename RowColType, typename NonzeroType,
			typename Coords
		>
		inline void vxm_inner_kernel_scatter( RC &rc,
			Vector< IOType, reference, Coords > &destination_vector,
			IOType * __restrict__ const &destination,
			const size_t &destination_range,
			const Vector< InputType1, reference, Coords > &source_vector,
			const InputType1 * __restrict__ const &source,
			const size_t &source_index,
			const internal::Compressed_Storage< InputType2, RowColType, NonzeroType >
				&matrix,
			const Vector< InputType3, reference, Coords > &mask_vector,
			const InputType3 * __restrict__ const &mask,
			const AdditiveMonoid &add,
			const Multiplication &mul,
			const std::function< size_t( size_t ) > &src_local_to_global,
			const std::function< size_t( size_t ) > &dst_global_to_local
		) {
			constexpr bool add_identity = descr & descriptors::add_identity;
			assert( rc == SUCCESS );
			assert( internal::getCoordinates( source_vector ).assigned( source_index ) );
#ifdef NDEBUG
			(void) source_vector;
#endif

			// mask did not fall through, so get current element
			typedef typename std::conditional<
				left_handed, typename Multiplication::D1, typename Multiplication::D2
			>::type SourceType;
			const SourceType input_element = internal::ValueOrIndex<
				descr, SourceType, InputType1
			>::getFromArray( source, src_local_to_global, source_index );

			// if we need to add identity, do so first:
			if( add_identity ) {
				internal::addIdentityDuringMV<
					reference, using_semiring,
					output_dense, left_handed,
					AdditiveMonoid, Multiplication, One,
					IOType, InputType1, SourceType, Coords
				>::apply(
					destination_vector, destination, destination_range,
					source_index,
					add, mul,
					input_element,
					src_local_to_global, dst_global_to_local
				);
			}

#ifdef _DEBUG
			std::cout << "vxm_scatter, source index " << source_index << " has "
				<< (matrix.col_start[ source_index + 1 ] - matrix.col_start[ source_index ])
				<< " nonzeroes.\n";
#endif
			// handle row or column at source_index
			for(
				size_t k = matrix.col_start[ source_index ];
				rc == SUCCESS && k < static_cast< size_t >(
					matrix.col_start[ source_index + 1 ]
				);
				++k
			) {
				// get output index
				const size_t destination_index = matrix.row_index[ k ];
				// check mask
				if( masked ) {
					if( !internal::getCoordinates( mask_vector ).template mask< descr >(
						destination_index, mask )
					) {
#ifdef _DEBUG
						std::cout << "\t output to index " << destination_index
							<< " ignored due to output masking\n";
#endif
						continue;
					}
				}
				// get nonzero
				typedef typename std::conditional<
					left_handed, typename Multiplication::D2, typename Multiplication::D1
				>::type RingNonzeroType;
				const RingNonzeroType nonzero = matrix.template
					getValue( k, One< RingNonzeroType >::value() );

				// do multiply
				typename Multiplication::D3 result = add.template
					getIdentity< typename Multiplication::D3 >();
#ifdef _DEBUG
				std::cout << "\t multiplying input vector element " << input_element
					<< " with matrix nonzero " << nonzero << "...\n";
#endif
				internal::leftOrRightHandedMul<
					left_handed,
					typename Multiplication::D3, SourceType, RingNonzeroType,
					Multiplication
				>::mul( result, input_element, nonzero, mul );

				// do add
#ifdef _DEBUG
				std::cout << "\t adding the result " << result
					<< " to the output vector at index " << destination_index << "\n";
#endif
				if( rc == SUCCESS && internal::getCoordinates(
						destination_vector
					).assign( destination_index )
				) {
#ifdef _DEBUG
					std::cout << "\t the result will be accumulated into the pre-existing "
						<< "value of " << destination[ destination_index ] << " which after "
						<< "accumulation now equals ";
#endif
					rc = foldl( destination[ destination_index ], result, add.getOperator() );
#ifdef _DEBUG
					std::cout << destination[ destination_index ] << " (at index "
						<< destination_index << ")\n";
#endif
				} else {
#ifdef _DEBUG
					std::cout << "\t since no entry existed at this position previously, "
						<< "destination[ " << destination_index << " ] now equals "
						<< result << "\n";
#endif
					destination[ destination_index ] =
						static_cast< typename AdditiveMonoid::D3 >( result );
				}
			}
		}

		/**
		 * Sparse matrix--vector multiplication \f$ u = vA \f$.
		 *
		 * @tparam descr        The descriptor used to perform this operation.
		 * @tparam masked       Whether the implementation is expecting a nontrivial
		 *                      mask.
		 * @tparam left_handed  Whether the vector nonzero is applied on the left-
		 *                      hand side of the matrix nonzero.
		 * @tparam using_semiring Whether the original call was made using a pure
		 *                        semiring.
		 * @tparam input_masked Whether the input vector is masked.
		 * @tparam Ring         The semiring used.
		 * @tparam Output_type  The output vector type.
		 * @tparam Input_type   The input vector type.
		 * @tparam Matrix_type  The input matrix type.
		 * @tparam Mask_type    The input mask type.
		 *
		 * @param[in,out] u   The output vector u. Contents shall be overwritten.
		 *                    The supplied vector must match the row dimension size of
		 *                    \a A.
		 * @param[in]  mask   A vector of arbitrary element types castable to booleans
		 *                    of size equal to \a u. Indicates which elements of \a u
		 *                    shall be written to. If omitted, a one-vector is
		 *                    assumed.
		 * @param[in]     v   The input vector v. The supplied vector must match the
		 *                    column dimension size of \a A.
		 * @param[in] v_mask  The mask to \a v. Only referred to if \a input_masked
		 *                    is set <tt>true</tt>.
		 * @param[in]     A   The input matrix A. The supplied matrix must match the
		 *                    dimensions of \a u and \a v.
		 * @param[in]  ring   The semiring to be used. This defines the additive and
		 *                    multiplicative monoids to be used. The additive monoid
		 *                    identity is used as the initial zero for performing
		 *                    this operation.
		 * @param[in] row_l2g An std::function that translates a local row
		 *                    coordinate into a global row coordinate (or a local
		 *                    column coordinate into a global column coordinate if
		 *                    grb::descriptors::transpose_matrix was given).
		 *                    This function is used to modify the behaviour of
		 *                    grb::descriptors::add_identity in case we only see a
		 *                    local part of a distributed matrix.
		 * @param[in] row_g2l An std::function that translates a global row coordinate
		 *                    into a local row coordinate (or column coordinates if
		 *                    grb::descriptors::transpose_matrix was given). If the
		 *                    global index is out of range, the function should
		 *                    return an invalid (too large) \a size_t. See above.
		 * @param[in] col_l2g An std::function that translates a local column
		 *                    coordinate into a global column coordinate. See above.
		 * @param[in] col_g2l An std::function that translates a global column
		 *                    coordinate into a local column coordinate. See above.
		 *
		 * \warning Bounds checking on output of \a row_l2g will only be performed in
		 *          debug mode.
		 *
		 * \parblock
		 * \par Performance semantics
		 *      -# This call takes \f$ \Theta(\mathit{nz}) + \mathcal{O}(m+n)\f$
		 *         work, where \f$ nz \f$ equals the number of nonzeroes in the
		 *         matrix, and \f$ m, n \f$ the dimensions of the matrix.
		 *
		 *      -# This call takes \f$ \mathcal{O}(1) \f$ memory beyond the memory
		 *         already used by the application when this function is called.
		 *
		 *      -# This call incurs at most
		 *         \f$ \mathit{nz}(
		 *                 \mathit{sizeof}(\mathit{D1} + \mathit{sizeof}(\mathit{D2}) +
		 *                 \mathit{sizeof}(\mathit{D3} + \mathit{sizeof}(\mathit{D4}) +
		 *                 \mathit{sizeof}(\mathit{RI} + \mathit{sizeof}(\mathit{CI}) +
		 *                 \mathit{sizeof}(T)
		 *         ) + \mathcal{O}(1) \f$
		 *         bytes of data movement, where RI is the row index data type
		 *         used by the input matrix \a A, CI is the column index data
		 *         type used by the input matrix \a A, and T is the data type of
		 *         elements of \a mask.
		 *         A good implementation will stream up to
		 *         \f$ (\mathit{nz}-m-1)\mathit{sizeof}(\mathit{RI}) \f$ or up to
		 *         \f$ (\mathit{nz}-n-1)\mathit{sizeof}(\mathit{CI}) \f$ less bytes
		 *         (assuming \f$ \mathit{nz} - 1 > m, n \f$. A standard CRS/CSR or
		 *         CCS/CSC scheme achieves this tighter bound.)
		 * \endparblock
		 *
		 * \warning This implementation forbids \a u to be equal to \a v.
		 *
		 * \warning This implementation forbids \a u to be equal to \a mask.
		 *
		 * \note This implementation has those restrictions since otherwise the
		 *       above performance semantics cannot be met.
		 */
		template<
			Descriptor descr,
			bool masked, bool input_masked,
			bool left_handed, bool using_semiring,
			template< typename > class One,
			class AdditiveMonoid, class Multiplication,
			typename IOType, typename InputType1, typename InputType2,
			typename InputType3, typename InputType4,
			typename RIT, typename CIT, typename NIT,
			typename Coords
		>
		RC vxm_generic(
			Vector< IOType, reference, Coords > &u,
			const Vector< InputType3, reference, Coords > &mask,
			const Vector< InputType1, reference, Coords > &v,
			const Vector< InputType4, reference, Coords > &v_mask,
			const Matrix< InputType2, reference, RIT, CIT, NIT > &A,
			const AdditiveMonoid &add,
			const Multiplication &mul,
			const Phase &phase,
			const std::function< size_t( size_t ) > &row_l2g,
			const std::function< size_t( size_t ) > &row_g2l,
			const std::function< size_t( size_t ) > &col_l2g,
			const std::function< size_t( size_t ) > &col_g2l
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
#ifdef _DEBUG
			const auto s = spmd< reference >::pid();
			std::cout << s << ": reference vxm called with a " << descriptors::toString( descr ) << "\n";
#endif

			// get input and output vector sizes
			const size_t m = internal::getCoordinates( u ).size();
			const size_t n = internal::getCoordinates( v ).size();

			// get whether the matrix should be transposed prior to execution of this
			// vector-times-matrix operation
			constexpr bool transposed = descr & descriptors::transpose_matrix;

			// get whether we may simply assume the vectors are dense
			constexpr bool dense_hint = descr & descriptors::dense;

			// get whether we are forced to use a row-major storage
			constexpr const bool crs_only = descr & descriptors::force_row_major;

			// check for dimension mismatch
			if( ( transposed && ( n != ncols( A ) || m != nrows( A ) ) ) ||
				( !transposed && ( n != nrows( A ) || m != ncols( A ) ) )
			) {
#ifdef _DEBUG
				std::cout << "Mismatch of columns ( " << n << " vs. " << ncols( A )
					<< " ) or rows ( " << m << " vs. " << nrows( A ) << " ) with "
					<< "transposed value " << ((int)transposed) << "\n";
#endif
				return MISMATCH;
			}

			// check density
			if( descr & descriptors::dense ) {
				if( nnz( v ) < size( v ) ) {
#ifdef _DEBUG
					std::cout << "\t Dense descriptor given but input vector was sparse\n";
#endif
					return ILLEGAL;
				}
				if( size( mask ) > 0 && nnz( mask ) < size( mask ) ) {
#ifdef _DEBUG
					std::cout << "\t Dense descriptor given but output mask has sparse "
						<< "structure\n";
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
				if( (transposed && internal::getCoordinates( mask ).size() != nrows( A )) ||
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

			// first handle trivial cases
			if( internal::getCoordinates( v ).nonzeroes() == 0 ||
				ncols( A ) == 0 || nrows( A ) == 0 || nnz( A ) == 0 || (
					masked && internal::getCoordinates( mask ).nonzeroes() == 0 &&
					!(descr & descriptors::invert_mask)
				) || (
					input_masked && internal::getCoordinates( v_mask ).nonzeroes() == 0 &&
					!(descr & descriptors::invert_mask)
				)
			) {
				// then the output must be empty
				for( size_t i = 0; i < m; ++i ) {
					if( internal::getCoordinates( u ).assigned( i ) ) {
						if(
							foldl( y[ i ],
							add.template getIdentity< IOType >(), add.getOperator()
						) != SUCCESS ) {
							return PANIC;
						}
					} else if( descr & descriptors::explicit_zero ) {
						if( setElement(
							u,
							add.template getIdentity< IOType >(),
							i
						) != SUCCESS ) {
							return PANIC;
						}
					}
				}
#ifdef _DEBUG
				std::cout << s << ": trivial operation requested; exiting without any ops. "
					<< "Input nonzeroes: " << internal::getCoordinates( v ).nonzeroes()
					<< ", matrix size " << nrows( A ) << " by " << ncols( A ) << " with "
					<< nnz( A ) << " nonzeroes.\n";
#endif
				// done
				return SUCCESS;
			}

			// check for illegal arguments
			if( !(descr & descriptors::safe_overlap) &&
				reinterpret_cast< const void * >( y ) ==
					reinterpret_cast< const void * >( x )
			) {
				std::cerr << "Warning: grb::internal::vxm_generic called with "
					"overlapping input and output vectors.\n";
				return OVERLAP;
			}
			if( masked && (reinterpret_cast<const void*>(y) == reinterpret_cast<const void*>(z)) ) {
				std::cerr << "Warning: grb::internal::vxm_generic called with "
					"overlapping mask and output vectors.\n";
				return OVERLAP;
			}

#ifdef _DEBUG
			std::cout << s << ": performing SpMV / SpMSpV using an " << nrows( A )
				<< " by " << ncols( A ) << " matrix holding " << nnz( A ) << " nonzeroes. "
				<< "The input vector holds " << internal::getCoordinates( v ).nonzeroes()
				<< " nonzeroes.\n";
#endif

			// whether the input mask should be the container used for
			// iterating over input nonzeroes, or whether the input
			// vector itself should be used. This depends on which
			// would lead to a smaller loop size.
			// Abbreviations:
			// - emiim: effective mask is input mask
			// -   eim: effective input mask
			const bool emiim = input_masked ? (
					( descr & descriptors::invert_mask ) || nnz( v ) < nnz( v_mask ) ?
					false :
					true
				) :
				false;
			const auto &eim = emiim ?
				internal::getCoordinates( v_mask ) :
				internal::getCoordinates( v );
#ifdef _DEBUG
			if( emiim ) {
				std::cout << s << ": effective mask is input mask\n";
			}
#endif

			// global return code. This will be updated by each thread from within a
			// critical section.
			RC global_rc = SUCCESS;

#ifdef _H_GRB_REFERENCE_OMP_BLAS2
			#pragma omp parallel
			{
				internal::Coordinates< reference >::Update local_update =
					internal::getCoordinates( u ).EMPTY_UPDATE();
				const size_t maxAsyncAssigns =
					internal::getCoordinates( u ).maxAsyncAssigns();
				size_t asyncAssigns = 0;
#endif
				// local return code
				RC rc = SUCCESS;
				// check if transpose is required
				if( descr & descriptors::transpose_matrix ) {
					// start compute u=vA^T
#ifdef _DEBUG
					std::cout << s << ": in u=vA^T=Av variant\n";
#endif
					// get loop sizes for each variant. Note that the CCS variant cannot be
					// parallelised without major pre-processing (or atomics), both of which
					// are significant overheads. We only choose it if we expect a sequential
					// execution to be faster compared to a parallel one.
					const size_t CRS_loop_size = masked ?
						std::min( nrows( A ), 2 * nnz( mask ) ) :
						nrows( A );
					const size_t CCS_seq_loop_size = !dense_hint ?
			                        std::min( ncols( A ), (
							input_masked && !( descr & descriptors::invert_mask ) ?
								2 * std::min( nnz( v_mask ), nnz( v ) ) :
								2 * nnz( v )
							)
						) :
			                        ncols( A );
#ifdef _H_GRB_REFERENCE_OMP_BLAS2
					// This variant plays it safe and always revert to a parallel mechanism,
					// even if we could mask on input
					// const size_t CCS_loop_size = CRS_loop_size + 1;
					// This variant modifies the sequential loop size to be P times more
					// expensive
					const size_t CCS_loop_size = crs_only ? CRS_loop_size + 1 :
						omp_get_num_threads() * CCS_seq_loop_size;
#else
					const size_t CCS_loop_size = crs_only ? CRS_loop_size + 1 :
						CCS_seq_loop_size;
#endif
					// choose best-performing variant.
					if( CCS_loop_size < CRS_loop_size ) {
						assert( !crs_only );
#ifdef _H_GRB_REFERENCE_OMP_BLAS2
						#pragma omp single
						{
#endif
							if( !input_masked && (dense_hint || nnz( v ) == ncols( A )) ) {
								// start u=vA^T using CCS
#ifdef _DEBUG
								std::cout << s << ": in full CCS variant (scatter)\n";
#endif
								// even though transposed, use CCS representation.
								// To avoid write conflicts, we keep things sequential.
								for( size_t j = 0; rc == SUCCESS && j < ncols( A ); ++j ) {
									if( input_masked &&
										!internal::getCoordinates( v_mask ).template mask< descr >( j, vm )
									) {
#ifdef _DEBUG
										std::cout << "\t mask at index " << j << " evaluates false; "
											<< "skipping\n";
#endif
										continue;
									}
									if( !internal::getCoordinates( v ).assigned( j ) ) {
#ifdef _DEBUG
										std::cout << "\t no input vector element at index " << j << "; "
											<< "skipping\n";
#endif
										continue;
									}
#ifdef _DEBUG
									std::cout << "\t processing index " << j << "\n";
#endif
									vxm_inner_kernel_scatter<
										descr, dense_hint, dense_hint, masked, left_handed, using_semiring,
										One
									>(
										rc,
										u, y, nrows( A ),
										v, x, j, internal::getCCS( A ),
										mask, z,
										add, mul,
										col_l2g, row_g2l
									);
								}
							} else {
#ifdef _DEBUG
								std::cout << s << ": in input-masked CCS variant (scatter)\n";
#endif
								// we know the exact sparsity pattern of the input vector
								// use it to call the inner kernel on those columns of A only
								for( size_t k = 0; k < eim.nonzeroes(); ++k ) {
									const size_t j = eim.index( k );
									if( input_masked && !internal::getCoordinates( v_mask ).template
										mask< descr >( j, vm )
									) {
#ifdef _DEBUG
										std::cout << s << "\t: input index " << j << " will not be processed "
											<< "due to being unmasked.\n";
#endif
										continue;
									}
									if( (!input_masked || emiim) &&
										!internal::getCoordinates( v ).assigned( j )
									) {
#ifdef _DEBUG
										std::cout << s << "\t: input index " << j << " will not be processed "
											<< "due to having no corresponding input vector element.\n";
#endif
										continue;
									}
#ifdef _DEBUG
									std::cout << s << ": processing input vector element " << j << "\n";
#endif
									vxm_inner_kernel_scatter<
										descr, false, dense_hint, masked, left_handed, using_semiring, One
									>(
										rc, u, y, nrows( A ), v, x, j, internal::getCCS( A ), mask, z,
										add, mul, col_l2g, row_g2l
									);
								}
							}
#ifdef _H_GRB_REFERENCE_OMP_BLAS2
						}
#endif
						// end u=vA^T using CCS
					} else {
						// start u=vA^T using CRS
						// matrix = &(A.CRS);
						// TODO internal issue #193
						if( !masked || (descr & descriptors::invert_mask) ) {
							// loop over all columns of the input matrix (can be done in parallel):
#ifdef _DEBUG
							std::cout << s << ": in full CRS variant (gather)\n";
#endif
#ifdef _H_GRB_REFERENCE_OMP_BLAS2
							size_t start, end;
							config::OMP::localRange( start, end, 0, nrows( A ) );
#else
							const size_t start = 0;
							const size_t end = nrows( A );
#endif
							for( size_t i = start; i < end; ++i ) {
								vxm_inner_kernel_gather<
									descr, masked, input_masked, left_handed, One
								>(
									rc,
#ifdef _H_GRB_REFERENCE_OMP_BLAS2
									local_update, asyncAssigns,
#endif
									u, y[ i ], i, v, x,
									nrows( A ), internal::getCRS( A ), nnz( A ),
									mask, z, v_mask, vm,
									add, mul, row_l2g, col_l2g, col_g2l
								);
#ifdef _H_GRB_REFERENCE_OMP_BLAS2
								if( asyncAssigns == maxAsyncAssigns ) {
									// warning: return code ignored for brevity;
									//         may not be the best thing to do
									(void) internal::getCoordinates( u ).joinUpdate( local_update );
									asyncAssigns = 0;
								}
#endif
							}
						} else {
#ifdef _DEBUG
							std::cout << s << ": in masked CRS variant (gather). Mask has "
								<< internal::getCoordinates( mask ).nonzeroes()
								<< " nonzeroes and size " << internal::getCoordinates( mask ).size()
								<< ":\n";
							for(
								size_t k = 0;
								k < internal::getCoordinates( mask ).nonzeroes();
								++k
							) {
								std::cout << " " << internal::getCoordinates( mask ).index( k );
							}
							std::cout << "\n";
#endif
							// loop only over the nonzero masks (can still be done in parallel!)
							// since mask structures are typically irregular, use dynamic schedule
#ifdef _H_GRB_REFERENCE_OMP_BLAS2
							#pragma omp for schedule( dynamic, config::CACHE_LINE_SIZE::value() ) nowait
#endif
							for(
								size_t k = 0;
								k < internal::getCoordinates( mask ).nonzeroes();
								++k
							) {
								const size_t i = internal::getCoordinates( mask ).index( k );
								assert( i < nrows( A ) );
								vxm_inner_kernel_gather< descr, false, input_masked, left_handed, One >(
									rc,
#ifdef _H_GRB_REFERENCE_OMP_BLAS2
									local_update, asyncAssigns,
#endif
									u, y[ i ], i, v, x,
									nrows( A ), internal::getCRS( A ), nnz( A ),
									mask, z, v_mask, vm,
									add, mul, row_l2g, col_l2g, col_g2l
								);
#ifdef _H_GRB_REFERENCE_OMP_BLAS2
								if( asyncAssigns == maxAsyncAssigns ) {
									// warning: return code ignored for brevity;
									//         may not be the best thing to do
									(void)internal::getCoordinates( u ).joinUpdate( local_update );
									asyncAssigns = 0;
								}
#endif
							}
						}
						// end u=vA^T using CRS
					}
					// end compute u=vA^T
				} else {
#ifdef _DEBUG
					std::cout << s << ": in u=vA=A^Tv variant\n";
#endif
					// start computing u=vA
					const size_t CCS_loop_size = masked ?
						std::min( ncols( A ), 2 * nnz( mask ) ) :
						ncols( A );
					const size_t CRS_seq_loop_size = !dense_hint ?
			                        std::min( nrows( A ), (
							input_masked &&
							!( descr & descriptors::invert_mask ) ?
								2 * std::min( nnz( v_mask ), nnz( v ) ) :
								2 * nnz( v )
						) ) :
						nrows( A );
#ifdef _H_GRB_REFERENCE_OMP_BLAS2
					// This variant ensures always choosing the parallel variant
					// const size_t CRS_loop_size = CCS_loop_size + 1;
					// This variant estimates this non-parallel variant's cost at a factor P
					// more
					const size_t CRS_loop_size = crs_only ? CRS_seq_loop_size + 1 :
						omp_get_num_threads() * CRS_seq_loop_size;
#else
					const size_t CRS_loop_size = crs_only ? CRS_seq_loop_size + 1:
						CRS_seq_loop_size;
#endif

					if( CRS_loop_size < CCS_loop_size ) {
#ifdef _DEBUG
						std::cout << s << ": in row-major vector times matrix variant (u=vA).\n"
							<< "\t (this variant relies on the scattering inner kernel)\n";
#endif
#ifdef _H_GRB_REFERENCE_OMP_BLAS2
						#pragma omp single
						{
#endif
							// start u=vA using CRS, sequential implementation only
							if( !dense_hint && nnz( v ) < nrows( A ) ) {
#ifdef _DEBUG
								std::cout << "\t looping over nonzeroes of the input vector or mask "
									<< "(whichever has fewer nonzeroes), calling scatter for each\n";
#endif
								// loop over nonzeroes of v only
								for( size_t k = 0; rc == SUCCESS && k < eim.nonzeroes(); ++k ) {
									const size_t i = eim.index( k );
									if( input_masked ) {
										if( !eim.template mask< descr >( i, vm ) ) {
#ifdef _DEBUG
											std::cout << "\t mask at position " << i
												<< " evaluates false; skipping\n";
#endif
											continue;
										}
										if( emiim && !internal::getCoordinates( v ).assigned( i ) ) {
#ifdef _DEBUG
											std::cout << "\t input vector has no element at position " << i
												<< "; skipping\n";
#endif
											continue;
										}
									}
#ifdef _DEBUG
									std::cout << "\t processing input vector element at position " << i
										<< "\n";
#endif
									vxm_inner_kernel_scatter<
										descr, false, dense_hint, masked, left_handed, using_semiring, One
									>(
										rc,
										u, y, ncols( A ), v, x, i,
										internal::getCRS( A ), mask, z,
										add, mul, row_l2g, col_g2l
									);
								}
							} else {
								// use straight for-loop over rows of A
#ifdef _DEBUG
								std::cout << "\t looping over rows of the input matrix, "
									<< "calling scatter for each\n";
#endif
								for( size_t i = 0; rc == SUCCESS && i < nrows( A ); ++i ) {
									if( input_masked && !internal::getCoordinates( v_mask ).template
										mask< descr >( i, vm )
									) {
#ifdef _DEBUG
										std::cout << "\t input mask evaluates false at position " << i
											<< "; skipping\n";
#endif
										continue;
									}
									if( !dense_hint && !internal::getCoordinates( v ).assigned( i ) ) {
#ifdef _DEBUG
										std::cout << "\t no input vector entry at position " << i
											<< "; skipping\n";
#endif
										continue;
									}
#ifdef _DEBUG
									std::cout << "\t processing entry " << i << "\n";
#endif
									vxm_inner_kernel_scatter<
										descr, dense_hint, dense_hint, masked, left_handed, using_semiring,
										One
									>(
										rc,
										u, y, ncols( A ), v, x, i,
										internal::getCRS( A ), mask, z,
										add, mul, row_l2g, col_g2l
									);
								}
							}
#ifdef _H_GRB_REFERENCE_OMP_BLAS2
						}
#endif
						// end u=vA using CRS
					} else {
						// start u=vA using CCS
						assert( !crs_only );
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
#ifdef _H_GRB_REFERENCE_OMP_BLAS2
							size_t start, end;
							config::OMP::localRange( start, end, 0, ncols( A ) );
#else
							const size_t start = 0;
							const size_t end = ncols( A );
#endif
							for( size_t j = start; j < end; ++j ) {
								vxm_inner_kernel_gather<
									descr, masked, input_masked, left_handed, One
								>(
									rc,
#ifdef _H_GRB_REFERENCE_OMP_BLAS2
									local_update, asyncAssigns,
#endif
									u, y[ j ], j,
									v, x,
									nrows( A ), internal::getCCS( A ), nnz( A ),
									mask, z, v_mask, vm,
									add, mul,
									row_l2g, row_g2l, col_l2g
								);
#ifdef _H_GRB_REFERENCE_OMP_BLAS2
								if( asyncAssigns == maxAsyncAssigns ) {
									// warning: return code ignored for brevity;
									//         may not be the best thing to do
									(void) internal::getCoordinates( u ).joinUpdate( local_update );
									asyncAssigns = 0;
								}
#endif
							}
						} else {
							// loop only over the nonzero masks (can still be done in parallel!)
#ifdef _DEBUG
							std::cout << s << ": loop over mask indices\n";
#endif
#ifdef _H_GRB_REFERENCE_OMP_BLAS2
							// since mask structures are typically irregular, use dynamic schedule
							#pragma omp for schedule( dynamic, config::CACHE_LINE_SIZE::value() ) nowait
#endif
							for(
								size_t k = 0;
								k < internal::getCoordinates( mask ).nonzeroes();
								++k
							) {
								const size_t j = internal::getCoordinates( mask ).index( k );
								vxm_inner_kernel_gather<
									descr, masked, input_masked, left_handed, One
								>(
									rc,
#ifdef _H_GRB_REFERENCE_OMP_BLAS2
									local_update, asyncAssigns,
#endif
									u, y[ j ], j, v, x,
									nrows( A ), internal::getCCS( A ), nnz( A ),
									mask, z, v_mask, vm,
									add, mul, row_l2g, row_g2l, col_l2g
								);
#ifdef _H_GRB_REFERENCE_OMP_BLAS2
								if( asyncAssigns == maxAsyncAssigns ) {
									// warning: return code ignored for brevity;
									//         may not be the best thing to do
									(void)internal::getCoordinates( u ).joinUpdate( local_update );
									asyncAssigns = 0;
								}
#endif
							}
						}
						// end u=vA using CCS
					}
					// end computing u=vA
				}

#ifdef _H_GRB_REFERENCE_OMP_BLAS2
				while( !internal::getCoordinates( u ).joinUpdate( local_update ) ) {}
#endif
				if( rc != SUCCESS ) {
					global_rc = rc;
				}
#ifdef _H_GRB_REFERENCE_OMP_BLAS2
			} // end pragma omp parallel
#endif

			assert( internal::getCoordinates( u ).nonzeroes() <= m );

#ifdef _DEBUG
			std::cout << s << ": exiting SpMV / SpMSpV. Output vector contains "
				<< nnz( u ) << " nonzeroes.\n" << std::flush;
#endif

			// done!
			return global_rc;
		}

	} // namespace internal

	/** \internal Delegates to fully masked variant */
	template<
		Descriptor descr = descriptors::no_operation, class Ring,
		typename IOType, typename InputType1,
		typename InputType2, typename InputType3,
		typename RIT, typename CIT, typename NIT,
		typename Coords
	>
	RC vxm(
		Vector< IOType, reference, Coords > &u,
		const Vector< InputType3, reference, Coords > &mask,
		const Vector< InputType1, reference, Coords > &v,
		const Matrix< InputType2, reference, RIT, CIT, NIT > &A,
		const Ring &ring = Ring(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			grb::is_semiring< Ring >::value, void
		>::type * const = nullptr
	) {
		const Vector< bool, reference, Coords > empty_mask( 0 );
		return vxm< descr, true, false >( u, mask, v, empty_mask, A, ring, phase );
	}

	/** \internal Delegates to fully masked variant */
	template<
		Descriptor descr = descriptors::no_operation,
		class AdditiveMonoid, class MultiplicativeOperator,
		typename IOType, typename InputType1,
		typename InputType2, typename InputType3,
		typename RIT, typename CIT, typename NIT,
		typename Coords
	>
	RC vxm(
		Vector< IOType, reference, Coords > &u,
		const Vector< InputType3, reference, Coords > &mask,
		const Vector< InputType1, reference, Coords > &v,
		const Matrix< InputType2, reference, RIT, CIT, NIT > &A,
		const AdditiveMonoid &add = AdditiveMonoid(),
		const MultiplicativeOperator &mul = MultiplicativeOperator(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if< grb::is_monoid< AdditiveMonoid >::value &&
			grb::is_operator< MultiplicativeOperator >::value &&
			!grb::is_object< IOType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			!grb::is_object< InputType3 >::value &&
			!std::is_same< InputType2, void >::value,
			void
		>::type * const = nullptr
	) {
		const grb::Vector< bool, reference, Coords > empty_mask( 0 );
		return vxm< descr, true, false >( u, mask, v, empty_mask, A, add, mul,
			phase );
	}

	/** \internal Delegates to vxm_generic. */
	template< Descriptor descr = descriptors::no_operation,
		bool output_may_be_masked = true,
		bool input_may_be_masked = true,
		class Ring,
		typename IOType,
		typename InputType1, typename InputType2,
		typename InputType3, typename InputType4,
		typename RIT, typename CIT, typename NIT,
		typename Coords
	>
	RC vxm(
		Vector< IOType, reference, Coords > &u,
		const Vector< InputType3, reference, Coords > &mask,
		const Vector< InputType1, reference, Coords > &v,
		const Vector< InputType4, reference, Coords > &v_mask,
		const Matrix< InputType2, reference, RIT, CIT, NIT > &A,
		const Ring &ring = Ring(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			grb::is_semiring< Ring >::value, void
		>::type * const = nullptr
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
				} );
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
				} );
		} else if( output_may_be_masked &&
			input_may_be_masked &&
			size( mask ) > 0 &&
			size( v_mask ) > 0
		) {
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
				} );
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
				} );
		}
	}

	/** \internal Delegates to fully masked version */
	template<
		Descriptor descr = descriptors::no_operation,
		class Ring,
		typename Coords, typename RIT, typename CIT, typename NIT,
		typename IOType = typename Ring::D4,
		typename InputType1 = typename Ring::D1,
		typename InputType2 = typename Ring::D2
	>
	RC vxm(
		Vector< IOType, reference, Coords > &u,
		const Vector< InputType1, reference, Coords > &v,
		const Matrix< InputType2, reference, RIT, CIT, NIT > &A,
		const Ring &ring = Ring(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			grb::is_semiring< Ring >::value, void
		>::type * const = nullptr
	) {
		const Vector< bool, reference, Coords > empty_mask( 0 );
		return vxm< descr, false, false >( u, empty_mask, v, empty_mask, A, ring,
			phase );
	}

	/** \internal Delegates to fully masked version */
	template<
		Descriptor descr = descriptors::no_operation,
		class AdditiveMonoid, class MultiplicativeOperator, typename IOType,
		typename InputType1, typename InputType2,
		typename RIT, typename CIT, typename NIT,
		typename Coords
	>
	RC vxm(
		Vector< IOType, reference, Coords > &u,
		const Vector< InputType1, reference, Coords > &v,
		const Matrix< InputType2, reference, RIT, CIT, NIT > &A,
		const AdditiveMonoid &add = AdditiveMonoid(),
		const MultiplicativeOperator &mul = MultiplicativeOperator(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			grb::is_monoid< AdditiveMonoid >::value &&
			grb::is_operator< MultiplicativeOperator >::value &&
			!grb::is_object< IOType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			!std::is_same< InputType2, void >::value, void
		>::type * const = nullptr
	) {
		const grb::Vector< bool, reference, Coords > empty_mask( 0 );
		return vxm< descr, false, false >( u, empty_mask, v, empty_mask, A, add, mul,
			phase );
	}

	/** \internal Delegates to fully masked version */
	template<
		Descriptor descr = descriptors::no_operation, class Ring,
		typename Coords, typename RIT, typename CIT, typename NIT,
		typename IOType = typename Ring::D4,
		typename InputType1 = typename Ring::D1,
		typename InputType2 = typename Ring::D2,
		typename InputType3 = bool
	>
	RC mxv(
		Vector< IOType, reference, Coords > &u,
		const Vector< InputType3, reference, Coords > &mask,
		const Matrix< InputType2, reference, RIT, CIT, NIT > &A,
		const Vector< InputType1, reference, Coords > &v,
		const Ring &ring,
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			grb::is_semiring< Ring >::value, void
		>::type * const = nullptr
	) {
		const Vector< bool, reference, Coords > empty_mask( 0 );
		return mxv< descr, true, false >( u, mask, A, v, empty_mask, ring, phase );
	}

	/**
	 * \parblock
	 * Performance semantics vary depending on whether a mask was provided, and on
	 * whether the input vector is sparse or dense. If the input vector \f$ v \f$
	 * is sparse, let \f$ J \f$ be its set of assigned indices. If a non-trivial
	 * mask \f$ \mathit{mask} \f$ is given, let \f$ I \f$ be the set of indices for
	 * which the corresponding \f$ \mathit{mask}_i \f$ evaluate <tt>true</tt>. Then:
	 *   -# For the performance guarantee on the amount of work this function
	 *      entails the following table applies:<br>
	 *      \f$ \begin{tabular}{cccc}
	 *           Masked & Dense input  & Sparse input \\
	 *           \noalign{\smallskip}
	 *           no  & $\Theta(2\mathit{nnz}(A))$      & $\Theta(2\mathit{nnz}(A_{:,J}))$ \\
	 *           yes & $\Theta(2\mathit{nnz}(A_{I,:})$ & $\Theta(\min\{2\mathit{nnz}(A_{I,:}),2\mathit{nnz}(A_{:,J})\})$
	 *          \end{tabular}. \f$
	 *   -# For the amount of data movements, the following table applies:<br>
	 *      \f$ \begin{tabular}{cccc}
	 *           Masked & Dense input  & Sparse input \\
	 *           \noalign{\smallskip}
	 *           no  & $\Theta(\mathit{nnz}(A)+\min\{m,n\}+m+n)$                         & $\Theta(\mathit{nnz}(A_{:,J}+\min\{m,2|J|\}+|J|)+\mathcal{O}(2m)$ \\
	 *           yes & $\Theta(\mathit{nnz}(A_{I,:})+\min\{|I|,n\}+2|I|)+\mathcal{O}(n)$ &
	 * $\Theta(\min\{\Theta(\mathit{nnz}(A_{I,:})+\min\{|I|,n\}+2|I|)+\mathcal{O}(n),\mathit{nnz}(A_{:,J}+\min\{m,|J|\}+2|J|)+\mathcal{O}(2m))$ \end{tabular}. \f$
	 *   -# A call to this function under no circumstance will allocate nor free
	 *      dynamic memory.
	 *   -# A call to this function under no circumstance will make system calls.
	 * The above performance bounds may be changed by the following desciptors:
	 *   * #descriptors::invert_mask: replaces \f$ \Theta(|I|) \f$ data movement
	 *     costs with a \f$ \mathcal{O}(2m) \f$ cost instead, or a
	 *     \f$ \mathcal{O}(m) \f$ cost if #descriptors::structural was defined as
	 *     well (see below). In other words, implementations are not required to
	 *     implement inverted operations efficiently (\f$ 2\Theta(m-|I|) \f$ data
	 *     movements would be optimal but costs another \f$ \Theta(m) \f$ memory
	 *     to maintain).
	 *   * #descriptors::structural: removes \f$ \Theta(|I|) \f$ data movement
	 *     costs as the mask values need no longer be touched.
	 *   * #descriptors::add_identity: adds, at most, the costs of grb::foldl
	 *     (on vectors) to all performance metrics.
	 *   * #descriptors::use_index: removes \f$ \Theta(n) \f$ or
	 *     \f$ \Theta(|J|) \f$ data movement costs as the input vector values need
	 *     no longer be touched.
	 *   * #descriptors::in_place (see also above): turns \f$ \mathcal{O}(2m) \f$
	 *     data movements into \f$ \mathcal{O}(m) \f$ instead; i.e., it halves the
	 *     amount of data movements for writing the output.
	 *   * #descriptors::dense: the input, output, and mask vectors are assumed to
	 *     be dense. This allows the implementation to skip checks or other code
	 *     blocks related to handling of sparse vectors. This may result in use of
	 *     unitialised memory if any of the provided vectors were, in fact,
	 *     sparse.
	 * \endparblock
	 *
	 * \internal Delegates to vxm_generic
	 */
	template<
		Descriptor descr = descriptors::no_operation,
		bool output_may_be_masked = true,
		bool input_may_be_masked = true,
		class Ring,
		typename IOType,
		typename InputType1, typename InputType2,
		typename InputType3, typename InputType4,
		typename RIT, typename CIT, typename NIT,
		typename Coords
	>
	RC mxv(
		Vector< IOType, reference, Coords > &u,
		const Vector< InputType3, reference, Coords > &mask,
		const Matrix< InputType2, reference, RIT, CIT, NIT > &A,
		const Vector< InputType1, reference, Coords > &v,
		const Vector< InputType4, reference, Coords > &v_mask,
		const Ring &ring,
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			grb::is_semiring< Ring >::value, void
		>::type * const = nullptr
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
				} );
		} else if( input_may_be_masked && (
			size( mask ) == 0 && size( v_mask ) > 0 )
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
				} );
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
				} );
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
				} );
		}
	}

	/**
	 * \internal Delegates to fully masked variant.
	 */
	template<
		Descriptor descr = descriptors::no_operation,
		class Ring,
		typename Coords, typename RIT, typename CIT, typename NIT,
		typename IOType = typename Ring::D4,
		typename InputType1 = typename Ring::D1,
		typename InputType2 = typename Ring::D2
	>
	RC mxv(
		Vector< IOType, reference, Coords > &u,
		const Matrix< InputType2, reference, RIT, CIT, NIT > &A,
		const Vector< InputType1, reference, Coords > &v,
		const Ring &ring,
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			grb::is_semiring< Ring >::value, void
		>::type * const = nullptr
	) {
		const Vector< bool, reference, Coords > empty_mask( 0 );
		return mxv< descr, false, false >( u, empty_mask, A, v, empty_mask, ring,
			phase );
	}

	/** \internal Delegates to fully masked version */
	template<
		Descriptor descr = descriptors::no_operation,
		class AdditiveMonoid, class MultiplicativeOperator,
		typename IOType, typename InputType1, typename InputType2,
		typename RIT, typename CIT, typename NIT,
		typename Coords
	>
	RC mxv(
		Vector< IOType, reference, Coords > &u,
		const Matrix< InputType2, reference, RIT, CIT, NIT > &A,
		const Vector< InputType1, reference, Coords > &v,
		const AdditiveMonoid &add = AdditiveMonoid(),
		const MultiplicativeOperator &mul = MultiplicativeOperator(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			grb::is_monoid< AdditiveMonoid >::value &&
			grb::is_operator< MultiplicativeOperator >::value &&
			!grb::is_object< IOType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			!std::is_same< InputType2, void >::value, void
		>::type * const = nullptr
	) {
		const grb::Vector< bool, reference, Coords > empty_mask( 0 );
		return mxv< descr, false, false >( u, empty_mask, A, v, empty_mask, add, mul,
			phase );
	}

	/**
	 * \internal Delegates to vxm_generic
	 */
	template<
		Descriptor descr = descriptors::no_operation,
		bool output_may_be_masked = true,
		bool input_may_be_masked = true,
		class AdditiveMonoid, class MultiplicativeOperator,
		typename IOType,
		typename InputType1, typename InputType2,
		typename InputType3, typename InputType4,
		typename RIT, typename CIT, typename NIT,
		typename Coords
	>
	RC vxm(
		Vector< IOType, reference, Coords > &u,
		const Vector< InputType3, reference, Coords > &mask,
		const Vector< InputType1, reference, Coords > &v,
		const Vector< InputType4, reference, Coords > &v_mask,
		const Matrix< InputType2, reference, RIT, CIT, NIT > &A,
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
			!std::is_same< InputType2, void >::value, void
		>::type * const = nullptr
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
				} );
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
				} );
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
				} );
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
				} );
		}
	}

	/**
	 * \internal Delegates to vxm_generic.
	 */
	template<
		Descriptor descr = descriptors::no_operation,
		bool output_may_be_masked = true,
		bool input_may_be_masked = true,
		class AdditiveMonoid, class MultiplicativeOperator,
		typename IOType,
		typename InputType1, typename InputType2,
		typename InputType3, typename InputType4,
		typename RIT, typename CIT, typename NIT,
		typename Coords
	>
	RC mxv(
		Vector< IOType, reference, Coords > &u,
		const Vector< InputType3, reference, Coords > &mask,
		const Matrix< InputType2, reference, RIT, CIT, NIT > &A,
		const Vector< InputType1, reference, Coords > &v,
		const Vector< InputType4, reference, Coords > &v_mask,
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
			!std::is_same< InputType2, void >::value, void
		>::type * const = nullptr
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
				} );
		} else if( input_may_be_masked && size( mask ) == 0 && size( v_mask ) > 0 ) {
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
				} );
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
				} );
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
				} );
		}
	}

	namespace internal {

#ifndef _H_GRB_REFERENCE_OMP_BLAS2
		/**
		 * A nonzero wrapper for use with grb::eWiseLambda over matricies.
		 *
		 * \internal In the general case, stores a pointer to values. Row and column
		 *           indices are kept as a copy since doing so is in virtually all
		 *           foreseeable cases more efficient than pointer indirection, for
		 *           two reasons:
		 *             1) sizeof(Index) <= sizeof(void*)
		 *             2) pointer chasing
		 *
		 * \internal A vector of instances of this type will be used as input to
		 *           std::sort during the grb::eWiseLambda over matrices. This
		 *           necessitates copy constructors, move constructors, as well as
		 *           assignment.
		 */
		template< typename VType, typename = void >
		class eWiseLambdaNonzero {

			private:

				typedef typename grb::config::RowIndexType RType;
				typedef typename grb::config::ColIndexType CType;
				RType _i;
				CType _j;
				const VType * _v;
				void swap( eWiseLambdaNonzero< VType > &other ) {
					_i = other._i;
					_j = other._j;
					_v = other._v;
					other._i = std::numeric_limits< RType >::max();
					other._j = std::numeric_limits< CType >::max();
					other._v = nullptr;
				}

			public:

				eWiseLambdaNonzero( const RType i, const CType j, const VType &v ) :
					_i( i ), _j( j ), _v( &v )
				{}

				eWiseLambdaNonzero( const eWiseLambdaNonzero< VType > &other ) :
					_i( other._i ), _j( other._j ), _v( other._v )
				{}

				eWiseLambdaNonzero( eWiseLambdaNonzero< VType > &&other ) {
					swap( other );
				}

				eWiseLambdaNonzero< VType >& operator=(
					const eWiseLambdaNonzero< VType >& other
				) {
					eWiseLambdaNonzero< VType > tmp( other );
					swap( tmp );
					return *this;
				}

				RType i() const { return _i; }

				CType j() const { return _j; }

				const VType& v() const { return *_v; }

		};

		/**
		 * \internal This is a specialisation for where the value types are thus small
		 *           that pointer indirection is not worth it also for value types.
		 *           (See also the related consideration regarding row and column
		 *           index type in the base class.)
		 *
		 * \internal Note that by design, VType will never be void, so the below
		 *           sizeofs are `safe'
		 */
		template< typename VType >
		class eWiseLambdaNonzero<
			VType, typename std::enable_if< sizeof(VType) <= 2 * sizeof(size_t) >::type
		> {

			private:

				typedef typename grb::config::RowIndexType RType;
				typedef typename grb::config::ColIndexType CType;
				RType _i;
				CType _j;
				VType _v;

				void swap( eWiseLambdaNonzero< VType > &other ) {
					_i = other._i;
					_j = other._j;
					_v = other._v;
					other._i = std::numeric_limits< RType >::max();
					other._j = std::numeric_limits< CType >::max();
				}


			public:

				eWiseLambdaNonzero( const RType i, const CType j, const VType &v ) :
					_i( i ), _j( j ), _v( v )
				{}

				eWiseLambdaNonzero( const eWiseLambdaNonzero< VType > &other ) :
					_i( other._i ), _j( other._j ), _v( other._v )
				{}

				eWiseLambdaNonzero( eWiseLambdaNonzero< VType > &&other ) {
					swap( other );
				}

				eWiseLambdaNonzero< VType >& operator=(
					const eWiseLambdaNonzero< VType >& other
				) {
					eWiseLambdaNonzero< VType > tmp( other );
					swap( tmp );
					return *this;
				}

				RType i() const { return _i; }

				CType j() const { return _j; }

				const VType& v() const { return _v; }

		};
#endif

		/**
		 * This is a helper function that takes a collection of eWiseLambdaNonzero
		 * instances and adds those into a given matrix' CRS.
		 *
		 * \internal Multiple batches of nonzeroes may be added through multiple
		 *           successive calls to this function.
		 * \internal This function assumes that a counting-sort has been executed
		 *           on the row_start array of CRS before the first call to this
		 *           function.
		 *           Thus for adding a nonzero on row i, this function simply
		 *           decrements row_start[i] and places the nonzero on position
		 *           row_start[i] (its value after decrementing).
		 * \internal It is currently only used in the eWiseLambda for matrices.
		 *
		 * @tparam DataType The nonzero type of the matrix \a A.
		 * @tparam fwd_iterator The type of the forward iterator to the nonzero
		 *                      collection. See \a start and \a end.
		 *
		 * @param[in,out] A Which matrix to update the CRS of.
		 * @param[in] start The start iterator to the collection of nonzeroes.
		 * @param[in]  end  The end iterator to the collection of nonzeroes.
		 */
		template<
			typename DataType, typename RIT, typename CIT, typename NIT,
			typename fwd_iterator
		>
		void addToCRS(
			const Matrix< DataType, reference, RIT, CIT, NIT > &A,
			const fwd_iterator start, const fwd_iterator end
		) {
			auto &CRS = internal::getCRS( A );
#ifdef _DEBUG
			std::cout << "Pre-sorting: \n";
			for( fwd_iterator k = start; k != end; ++k ) {
				typename internal::eWiseLambdaNonzero< DataType > &nonzero = *k;
				std::cout << "\t( " << nonzero.i() << ", " << nonzero.j() << ", "
					<< nonzero.v() << " )\n";
			}
#endif
			std::sort( start, end, [](
					const internal::eWiseLambdaNonzero< DataType > &left,
					const internal::eWiseLambdaNonzero< DataType > &right
				) {
					return (left.i()) < (right.i());
				} );
#ifdef _DEBUG
			std::cout << "Post-sort: \n";
			for( fwd_iterator k = start; k != end; ++k ) {
				typename internal::eWiseLambdaNonzero< DataType > &nonzero = *k;
				std::cout << "\t( " << nonzero.i() << ", " << nonzero.j() << ", "
					<< nonzero.v() << " )\n";
			}
#endif
#ifdef _H_GRB_REFERENCE_OMP_BLAS2
			// Rationale, because the critical section here may *seem* like a bad idea.
			//
			// First some facts:
			//   1. chunks are load balanced, processing each chunk costs roughly the
			//      same
			//   2. there are many calls to this function, one for each cache-sized
			//      chunk
			//   3. typically, the number of chunks will be much larger than the number
			//      of cores
			//   4. while the below loop is Theta( nz ), what precedes costs
			//      Theta( nz log(nz) )
			// here, nz is the number of nonzeroes per chunk (and thus also per call to
			// this function).
			//
			// Following from these four points, the below will naturally lead to a
			// skewed pipelined execution. Parallel resources will only not be fully
			// utilised only during the initial ramp-up and final wind-down.
			//
			// Ramp-up:
			//   initially and ideally, all T threads will arrive at the below critical
			//   section simultaneously. Thus T-1 threads will have to wait on one thread
			//   processing the below, followed by T-2 thread waiting, then by T-3
			//   threads waiting, and so on.
			//
			// Steady-state:
			//   Due to fact #4, we expect the contentions on the below critical section
			//   to all but disappear after processing the first T chunks.
			//
			// Wind-down:
			//   Parallel resources will not be fully utilised when processing the last
			//   T chunks.
			//
			// Trade-off:
			//   To keep the inefficiency from the ramp-up and wind-down stages low, we
			//   want many more chunks then the number of active threads. To keep the
			//   inefficiency arising from the locking mechanism low, we want the number
			//   of nonzeroes nz per chunk large enough.
			//
			// Current policy:
			//   Select nonzeroes nz per chunk to fit a private cache but utilise the
			//   full number of threads that GraphBLAS has been given.
			#pragma omp critical
#endif
			{
				for( fwd_iterator k = start; k != end; ++k ) {
					typename internal::eWiseLambdaNonzero< DataType > &nonzero = *k;
					CRS.row_index[ --(CRS.col_start[ nonzero.i() ]) ] = nonzero.j();
					CRS.values[ CRS.col_start[ nonzero.i() ] ] = nonzero.v();
				}
			}
		}
	}

	/**
	 * Straightforward implementation using the column-major layout.
	 *
	 * @see grb::eWiseLambda for the user-level specification.
	 */
	template<
		class ActiveDistribution, typename Func,
		typename DataType, typename RIT, typename CIT, typename NIT
	>
	RC eWiseLambda(
		const Func f,
		const Matrix< DataType, reference, RIT, CIT, NIT > &A,
		const size_t s, const size_t P
	) {
#ifdef _DEBUG
		std::cout << "entering grb::eWiseLambda (matrices, reference ). A is "
			<< grb::nrows( A ) << " by " << grb::ncols( A ) << " and holds "
			<< grb::nnz( A ) << " nonzeroes.\n";
#endif
		// check for trivial call
		if( grb::nrows( A ) == 0 || grb::ncols( A ) == 0 || grb::nnz( A ) == 0 ) {
			return SUCCESS;
		}

#ifdef _H_GRB_REFERENCE_OMP_BLAS2
		#pragma omp parallel
#endif
		{
			// prep CRS for overwrite
			{
#ifdef _DEBUG
				std::cout << "\t\t original CRS row start = { ";
				for( size_t i = 0; i <= A.m; ++i ) {
					std::cout << A.CRS.col_start[ i ] << " ";
				}
				std::cout << "}\n";
#endif
				size_t m_start = 0, m_end = A.m;
#ifdef _H_GRB_REFERENCE_OMP_BLAS2
				config::OMP::localRange( m_start, m_end, 0, A.m );
#endif
				const size_t tmp = A.CRS.col_start[ m_start + 1 ];
				for( size_t i = m_start + 1; i < m_end; ++i ) {
					A.CRS.col_start[ i ] = A.CRS.col_start[ i + 1 ];
				}
#ifdef _H_GRB_REFERENCE_OMP_BLAS2
				#pragma omp barrier
#endif
				if( m_start < m_end ) {
					A.CRS.col_start[ m_start ] = tmp;
				}
#ifdef _H_GRB_REFERENCE_OMP_BLAS2
				#pragma omp barrier
#endif
#ifdef _DEBUG
 #ifdef _H_GRB_REFERENCE_OMP_BLAS2
				#pragma omp single
 #endif
				{
					std::cout << "\t\t shifted CRS row start = { ";
					for( size_t i = 0; i <= A.m; ++i ) {
						std::cout << A.CRS.col_start[ i ] << " ";
					}
					std::cout << "}\n";
				}
#endif
			}

			// loop over all nonzeroes using CCS
			size_t start, end;
#ifndef _H_GRB_REFERENCE_OMP_BLAS2
			start = 0;
			end = A.CCS.col_start[ A.n ];
#else
			config::OMP::localRange( start, end, 0, A.CCS.col_start[ A.n ] );
#endif

			// while we guarantee a lower bound through the constructors of matrix given
			// as an argument, we dynamically request the maximum chunk size for
			// ingesting into CRS to exploit the possibility that larger buffers were
			// requested by other matrices' constructors.
			size_t maxChunkSize = internal::reference_bufsize /
				sizeof( internal::eWiseLambdaNonzero< DataType > );
			assert( maxChunkSize > 0 );
#ifndef _H_GRB_REFERENCE_OMP_BLAS2
			const size_t maxLocalChunkSize = maxChunkSize;
			typename internal::eWiseLambdaNonzero< DataType > * nonzeroes =
				internal::template getReferenceBuffer<
					typename internal::eWiseLambdaNonzero< DataType >
				>( maxChunkSize );
#else
			typename internal::eWiseLambdaNonzero< DataType > * nonzeroes = nullptr;
			size_t maxLocalChunkSize = 0;
			{
				typename internal::eWiseLambdaNonzero< DataType > * nonzero_buffer =
					internal::template getReferenceBuffer<
						typename internal::eWiseLambdaNonzero< DataType >
					>( maxChunkSize );
				size_t my_buffer_start = 0, my_buffer_end = maxChunkSize;
				config::OMP::localRange( my_buffer_start, my_buffer_end, 0, maxChunkSize );
				maxLocalChunkSize = my_buffer_end - my_buffer_start;
				assert( maxLocalChunkSize > 0 );
				nonzeroes = nonzero_buffer + my_buffer_start;
			}
#endif

#ifdef _DEBUG
 #ifdef _H_GRB_REFERENCE_OMP_BLAS2
			#pragma omp critical
 #endif
			std::cout << "\t processing range " << start << "--" << end << ".\n";
			std::cout << "\t COO buffer for updating CRS (we loop over nonzeroes in "
				<< "CCS) has a maximum size of " << maxChunkSize << "\n";
#endif

			size_t j_start, j_end;
			if( start < end ) {
				// find my starting column
				size_t j_left_range = 0;
				size_t j_right_range = A.n;
				j_start = A.n / 2;
				assert( A.n > 0 );
				while( j_start < A.n && !(
					A.CCS.col_start[ j_start ] <=
						start && start < A.CCS.col_start[ j_start + 1 ]
				) ) {
#ifdef _DEBUG
 #ifdef _H_GRB_REFERENCE_OMP_BLAS2
					#pragma omp critical
 #endif
					std::cout << "\t binary search for " << start << " in [ " << j_left_range
						<< ", " << j_right_range << " ) = [ " << A.CCS.col_start[ j_left_range ]
						<< ", " << A.CCS.col_start[ j_right_range ] << " ). "
						<< "Currently tried and failed at " << j_start << "\n";
#endif
					if( j_right_range == j_left_range ) {
						assert( false );
						break;
					} else if( A.CCS.col_start[ j_start ] > start ) {
						j_right_range = j_start;
					} else {
						j_left_range = j_start + 1;
					}
					assert( j_right_range >= j_left_range );
					j_start = j_right_range - j_left_range;
					j_start /= 2;
					j_start += j_left_range;
				}
#ifdef _DEBUG
 #ifdef _H_GRB_REFERENCE_OMP_BLAS2
				#pragma omp critical
				std::cout << "\t selected j_start = " << j_start << "\n";
 #endif
#endif
				// find my end column
				j_left_range = 0;
				j_right_range = A.n;
				j_end = A.n / 2;
				if( j_end < A.CCS.col_start[ A.n ] ) {
					while( j_end < A.n && !(
						A.CCS.col_start[ j_end ] <= end &&
						end < A.CCS.col_start[ j_end + 1 ]
					) ) {
#ifdef _DEBUG
 #ifdef _H_GRB_REFERENCE_OMP_BLAS2
						#pragma omp critical
 #endif
						std::cout << "\t binary search for " << end << " in [ " << j_left_range
							<< ", " << j_right_range << " ) = [ " << A.CCS.col_start[ j_left_range ]
							<< ", " << A.CCS.col_start[ j_right_range ] << " ). "
							<< "Currently tried and failed at " << j_end << "\n";
#endif
						if( j_right_range == j_left_range ) {
							assert( false );
							break;
						} else if( A.CCS.col_start[ j_end ] > end ) {
							j_right_range = j_end;
						} else {
							j_left_range = j_end + 1;
						}
						assert( j_right_range >= j_left_range );
						j_end = j_right_range - j_left_range;
						j_end /= 2;
						j_end += j_left_range;
					}
				}
				if( j_start > j_end ) {
					j_start = j_end;
				}
#ifdef _DEBUG
 #ifdef _H_GRB_REFERENCE_OMP_BLAS2
				#pragma omp critical
				std::cout << "\t selected j_end = " << j_end << "\n";
 #endif
#endif
#ifndef NDEBUG
				assert( j_end <= A.n );
				assert( start >= A.CCS.col_start[ j_start ] );
				if( j_start < A.n ) {
					assert( start <= A.CCS.col_start[ j_start + 1 ] );
				}
				assert( end >= A.CCS.col_start[ j_end ] );
				if( j_end < A.n ) {
					assert( end <= A.CCS.col_start[ j_end + 1 ] );
				}
#endif

				// prepare fields for in-place CRS update
				size_t pos = 0;
				constexpr size_t chunkSize_c = grb::config::MEMORY::l1_cache_size() /
					sizeof( internal::eWiseLambdaNonzero< DataType > );
				constexpr size_t minChunkSize = chunkSize_c == 0 ? 1 : chunkSize_c;
				const size_t chunkSize = minChunkSize > maxLocalChunkSize ?
					maxLocalChunkSize :
					minChunkSize;

#ifdef _DEBUG
 #ifdef _H_GRB_REFERENCE_OMP_BLAS2
				#pragma omp critical
 #endif
				{
					std::cout << "\t elected chunk size for updating the CRS structure is "
						<< chunkSize << "\n";
				}
#endif

				// preamble
				for(
					size_t k = start;
					k < std::min(
						static_cast< size_t >( A.CCS.col_start[ j_start + 1 ] ), end
					);
					++k
				) {
					// get row index
					const size_t i = A.CCS.row_index[ k ];
#ifdef _DEBUG
					std::cout << "Processing nonzero at ( " << i << ", " << j_start << " )\n";
#endif
					// execute lambda on nonzero
					const size_t col_pid = ActiveDistribution::offset_to_pid(
						j_start, A.n, P
					);
					const size_t col_off = ActiveDistribution::local_offset(
						A.n, col_pid, P
					);
					const size_t global_i = ActiveDistribution::local_index_to_global(
						i, A.m, s, P
					);
					const size_t global_j = ActiveDistribution::local_index_to_global(
						j_start - col_off, A.n, col_pid, P
					);
					assert( k < A.CCS.col_start[ A.n ] );
					f( global_i, global_j, A.CCS.values[ k ] );

					// update CRS
					nonzeroes[ pos++ ] = internal::eWiseLambdaNonzero< DataType >(
						A.CCS.row_index[ k ], j_start, A.CCS.values[ k ]
					);
					if( pos  == chunkSize ) {
						internal::addToCRS( A, nonzeroes, nonzeroes + chunkSize );
						pos = 0;
					}
				}
				// main loop
				if( j_start != j_end ) {
					for( size_t j = j_start + 1; j < j_end; ++j ) {
						for(
							size_t k = A.CCS.col_start[ j ];
							k < static_cast< size_t >( A.CCS.col_start[ j + 1 ] );
							++k
						) {
							// get row index
							const size_t i = A.CCS.row_index[ k ];
#ifdef _DEBUG
							std::cout << "Processing nonzero at ( " << i << ", " << j << " )\n";
#endif
							// execute lambda on nonzero
							const size_t col_pid = ActiveDistribution::offset_to_pid( j, A.n, P );
							const size_t col_off = ActiveDistribution::local_offset(
								A.n, col_pid, P
							);
							const size_t global_i = ActiveDistribution::local_index_to_global(
								i, A.m, s, P
							);
							const size_t global_j = ActiveDistribution::local_index_to_global(
								j - col_off, A.n, col_pid, P
							);
							assert( k < A.CCS.col_start[ A.n ] );
							f( global_i, global_j, A.CCS.values[ k ] );

							// update CRS
							nonzeroes[ pos++ ] = internal::eWiseLambdaNonzero< DataType >(
								A.CCS.row_index[ k ], j, A.CCS.values[ k ]
							);
							if( pos == chunkSize ) {
								internal::addToCRS( A, nonzeroes, nonzeroes + chunkSize );
								pos = 0;
							}
						}
					}
				}
				// postamble
				assert( j_end <= A.n );
				for( size_t k = A.CCS.col_start[ j_end ]; k < end; ++k ) {
					// get row index
					const size_t i = A.CCS.row_index[ k ];
#ifdef _DEBUG
					std::cout << "Processing nonzero at ( " << i << ", " << j_end << " )\n";
#endif
					// execute lambda on nonzero
					const size_t col_pid = ActiveDistribution::offset_to_pid( j_end, A.n, P );
					const size_t col_off = ActiveDistribution::local_offset( A.n, col_pid, P );
					const size_t global_i = ActiveDistribution::local_index_to_global(
						i, A.m, s, P
					);
					const size_t global_j = ActiveDistribution::local_index_to_global(
						j_end - col_off, A.n, col_pid, P
					);
					assert( k < A.CCS.col_start[ A.n ] );
					f( global_i, global_j, A.CCS.values[ k ] );

					// update CRS
					nonzeroes[ pos++ ] = internal::eWiseLambdaNonzero< DataType >(
						A.CCS.row_index[ k ], j_end, A.CCS.values[ k ]
					);
					if( pos == chunkSize ) {
						internal::addToCRS( A, nonzeroes, nonzeroes + chunkSize );
						pos = 0;
					}
				}
				// update CRS
				if( pos > 0 ) {
					internal::addToCRS( A, nonzeroes, nonzeroes + pos );
					pos = 0;
				}
			}
		} // end pragma omp parallel

#ifdef _DEBUG
		std::cout << "\t exiting grb::eWiseLambda (matrices, reference). Contents:\n";
		std::cout << "\t\t CRS row start = { ";
		for( size_t i = 0; i <= A.m; ++i ) {
			std::cout << A.CRS.col_start[ i ] << " ";
		}
		std::cout << "}\n";
		for( size_t i = 0; i < A.m; ++i ) {
			for( size_t k = A.CRS.col_start[ i ]; k < A.CRS.col_start[ i + 1 ]; ++k ) {
				std::cout << "\t\t ( " << i << ", " << A.CRS.row_index[ k ] << " ) = "
					<< A.CRS.values[ k ] << "\n";
			}
		}
		std::cout << "\t\t CCS col start = { ";
		for( size_t j = 0; j <= A.n; ++j ) {
			std::cout << A.CCS.col_start[ j ] << " ";
		}
		std::cout << "}\n";
		for( size_t j = 0; j < A.n; ++j ) {
			for( size_t k = A.CCS.col_start[ j ]; k < A.CCS.col_start[ j + 1 ]; ++k ) {
				std::cout << "\t\t ( " << A.CCS.row_index[ k ] << ", " << j << " ) = "
					<< A.CCS.values[ k ] << "\n";
			}
		}
#endif
		return SUCCESS;
	}

	/**
	 * This function provides dimension checking and will defer to the below
	 * function for the actual implementation.
	 *
	 * @see grb::eWiseLambda for the user-level specification.
	 */
	template<
		typename Func,
		typename DataType1, typename RIT, typename CIT, typename NIT,
		typename DataType2,
		typename Coords, typename... Args
	>
	RC eWiseLambda(
		const Func f,
		const Matrix< DataType1, reference, RIT, CIT, NIT > &A,
		const Vector< DataType2, reference, Coords > &x,
		Args... args
	) {
		// do size checking
		if( !( size( x ) == nrows( A ) || size( x ) == ncols( A ) ) ) {
			std::cerr << "Mismatching dimensions: given vector of size " << size( x )
				<< " has nothing to do with either matrix dimension (" << nrows( A )
				<< " nor " << ncols( A ) << ").\n";
			return MISMATCH;
		}
		// no need for synchronisation, everything is local in reference implementation
		return eWiseLambda( f, A, args... );
	}

	/** @} */

} // namespace grb

// parse this unit again for OpenMP support
#ifdef _GRB_WITH_OMP
 #ifndef _H_GRB_REFERENCE_OMP_BLAS2
  #define _H_GRB_REFERENCE_OMP_BLAS2
  #define reference reference_omp
  #include "graphblas/reference/blas2.hpp"
  #undef reference
  #undef _H_GRB_REFERENCE_OMP_BLAS2
 #endif
#endif

#undef NO_CAST_ASSERT

#endif // end _H_GRB_REFERENCE_BLAS2

