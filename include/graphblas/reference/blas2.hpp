
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

#if ! defined _H_GRB_REFERENCE_BLAS2 || defined _H_GRB_REFERENCE_OMP_BLAS2
#define _H_GRB_REFERENCE_BLAS2

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
		template< Descriptor descr,
			bool masked, // TODO issue #69
			bool input_masked,
			bool left_handed,
			template< typename >
			class One,
			class AdditiveMonoid,
			class Multiplication,
			typename IOType,
			typename InputType1,
			typename InputType2,
			typename InputType3,
			typename InputType4,
			typename Coords,
			typename RowColType,
			typename NonzeroType >
		inline void vxm_inner_kernel_gather( RC & rc,
#ifdef _H_GRB_REFERENCE_OMP_BLAS2
			internal::Coordinates< reference >::Update & local_update,
			size_t & asyncAssigns,
#endif
			Vector< IOType, reference, Coords > & destination_vector,
			IOType & destination_element,
			const size_t & destination_index,
			const Vector< InputType1, reference, Coords > & source_vector,
			const InputType1 * __restrict__ const & source,
			const size_t & source_range,
			const internal::Compressed_Storage< InputType2, RowColType, NonzeroType > & matrix,
			const Vector< InputType3, reference, Coords > & mask_vector,
			const InputType3 * __restrict__ const & mask,
			const Vector< InputType4, reference, Coords > & source_mask_vector,
			const InputType4 * __restrict__ const & source_mask,
			const AdditiveMonoid & add,
			const Multiplication & mul,
			const std::function< size_t( size_t ) > & src_local_to_global,
			const std::function< size_t( size_t ) > & src_global_to_local,
			const std::function< size_t( size_t ) > & dst_local_to_global ) {
			constexpr bool add_identity = descr & descriptors::add_identity;
			constexpr bool dense_hint = descr & descriptors::dense;
			constexpr bool explicit_zero = descr & descriptors::explicit_zero;
#ifdef _DEBUG
			constexpr bool use_index = descr & descriptors::use_index;
#endif
			assert( rc == SUCCESS );

			// check whether we should compute output here
			if( masked ) {
				if( ! internal::getCoordinates( mask_vector ).template mask< descr >( destination_index, mask ) ) {
#ifdef _DEBUG
					std::cout << "Masks says to skip processing destination "
								 "index "
							  << destination_index << "\n";
#endif
					return;
				}
			}

			// take shortcut, if possible
			if( grb::has_immutable_nonzeroes< AdditiveMonoid >::value && internal::getCoordinates( destination_vector ).assigned( destination_index ) &&
				destination_element != add.template getIdentity< IOType >() ) {
				return;
			}

			// start output
			typename AdditiveMonoid::D3 output = add.template getIdentity< typename AdditiveMonoid::D3 >();
			bool set = false;

			// if we need to add identity, do so first:
			if( add_identity ) {
				const size_t id_location = src_global_to_local( dst_local_to_global( destination_index ) );
				if( ( ! input_masked || internal::getCoordinates( source_mask_vector ).template mask< descr >( id_location, source_mask ) ) && id_location < source_range ) {
					if( dense_hint || internal::getCoordinates( source_vector ).assigned( id_location ) ) {
						typename AdditiveMonoid::D1 temp;
						internal::CopyOrApplyWithIdentity< ! left_handed, typename AdditiveMonoid::D1, InputType1, One >::set( temp, source_vector[ id_location ], mul );
						internal::CopyOrApplyWithIdentity< false, typename AdditiveMonoid::D3, typename AdditiveMonoid::D1, AdditiveMonoid::template Identity >::set( output, temp, add );
						set = true;
					}
				}
			}

			// handle row or column at destination_index
			// NOTE: This /em could be parallelised, but will probably only slow things down
#ifdef _DEBUG
			std::cout << "vmx_gather: processing destination index " << destination_index << " / " << internal::getCoordinates( destination_vector ).size() << ". Input matrix has "
					  << ( matrix.col_start[ destination_index + 1 ] - matrix.col_start[ destination_index ] ) << " nonzeroes.\n";
#endif
			for( size_t k = matrix.col_start[ destination_index ]; rc == SUCCESS && k < static_cast< size_t >( matrix.col_start[ destination_index + 1 ] ); ++k ) {
				// declare multiplication output field
				typename Multiplication::D3 result = add.template getIdentity< typename AdditiveMonoid::D3 >();
				// get source index
				const size_t source_index = matrix.row_index[ k ];
				// check mask
				if( input_masked && ! internal::getCoordinates( source_mask_vector ).template mask< descr >( source_index, source_mask ) ) {
#ifdef _DEBUG
					std::cout << "\t vmx_gather: skipping source index " << source_index << " due to input mask\n";
#endif
					continue;
				}
				// check for sparsity at source
				if( ! dense_hint ) {
					if( ! internal::getCoordinates( source_vector ).assigned( source_index ) ) {
#ifdef _DEBUG
						std::cout << "\t vmx_gather: Skipping out of "
									 "computation with source index "
								  << source_index << " since it does not contain a nonzero\n";
#endif
						continue;
					}
				}
				// get nonzero
				const auto nonzero = left_handed ? matrix.template getValue( k, One< typename Multiplication::D2 >::value() ) :
                                                   matrix.template getValue( k, One< typename Multiplication::D1 >::value() );
#ifdef _DEBUG
				std::cout << "\t vmx_gather: interpreted nonzero is " << nonzero << ", which is the " << k << "-th nonzero and has source index " << source_index << "\n";
#endif
				// check if we use source element or whether we use its index value instead
				const auto apply_source = left_handed ? internal::ValueOrIndex< descr, typename Multiplication::D1, InputType1 >::get( source, src_local_to_global, source_index ) :
                                                        internal::ValueOrIndex< descr, typename Multiplication::D2, InputType1 >::get( source, src_local_to_global, source_index );
#ifdef _DEBUG
				if( use_index ) {
					std::cout << "\t vmx_gather (use_index descriptor): apply( "
								 "output, matrix nonzero, vector nonzero, * ) "
								 "= apply( ";
				} else {
					std::cout << "\t vmx_gather: apply( output, matrix "
								 "nonzero, vector nonzero, * ) = apply( ";
				}
#endif
				if( ! left_handed ) {
#ifdef _DEBUG
					std::cout << result << ", " << nonzero << ", " << apply_source << ", * );\n";
#endif
					rc = apply( result, nonzero, apply_source, mul );
				} else {
#ifdef _DEBUG
					std::cout << result << ", " << nonzero << ", " << apply_source << ", * );\n";
#endif
					rc = apply( result, apply_source, nonzero, mul );
				}
#ifdef _DEBUG
				std::cout << "\t vmx_gather: result = " << result << "\n";
#endif
				// sanity check (but apply cannot fail)
				assert( rc == SUCCESS );
				// accumulate result
#ifdef _DEBUG
				std::cout << "\t vmx_gather: foldr( " << result << ", " << output << ", + );\n";
#endif
				rc = foldr( result, output, add.getOperator() );
#ifdef _DEBUG
				std::cout << "\t vmx_gather: output = " << output << "\n";
#endif
				set = true;
				// sanity check (but apply cannot fail)
				assert( rc == SUCCESS );
			}

#ifdef _DEBUG
			if( set ) {
				std::cout << "\t vmx_gather: local contribution to this output "
							 "element at index "
						  << destination_index << " will be " << output
						  << " and this corresponds to an explicitly set "
							 "nonzero.\n";
			} else {
				std::cout << "\t vmx_gather: local contribution to this output "
							 "element at index "
						  << destination_index << " will be " << output << " and this is an unset value.\n";
				if( internal::getCoordinates( destination_vector ).assigned( destination_index ) ) {
					std::cout << "\t(old value " << destination_element << " will remain unmodified.)\n";
				} else {
					std::cout << "\t(no old value existed so the output vector "
								 "will remain unset at this index.)\n";
				}
			}
#endif
			// finally, accumulate in output
			if( explicit_zero || set ) {
#ifdef _DEBUG
				std::cout << "\taccumulating " << output << " into output vector...\n";
#endif
#ifdef _H_GRB_REFERENCE_OMP_BLAS2
				const bool was_already_assigned = internal::getCoordinates( destination_vector ).asyncAssign( destination_index, local_update );
				if( ! was_already_assigned ) {
					(void)asyncAssigns++;
				}
#else
				const bool was_already_assigned = internal::getCoordinates( destination_vector ).assign( destination_index );
#endif
				if( dense_hint || was_already_assigned ) {
#ifdef _DEBUG
					std::cout << "\tfoldl( " << destination_element << ", " << output << ", add.getOperator() );, destination_element = ";
#endif
					rc = foldl( destination_element, output, add.getOperator() );
#ifdef _DEBUG
					std::cout << destination_element << "\n";
#endif
				} else {
#ifdef _DEBUG
					std::cout << "\toutput vector element was previously not "
								 "set. Old (possibly uninitialised value) "
							  << destination_element << " will now be set to " << output << ", result (after, possibly, casting): ";
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
		template< Descriptor descr,
			bool input_dense,
			bool output_dense,
			bool masked,
			bool left_handed,
			template< typename >
			class One,
			typename IOType,
			class AdditiveMonoid,
			class Multiplication,
			typename InputType1,
			typename InputType2,
			typename InputType3,
			typename RowColType,
			typename NonzeroType,
			typename Coords >
		inline void vxm_inner_kernel_scatter( RC & rc,
			Vector< IOType, reference, Coords > & destination_vector,
			IOType * __restrict__ const & destination,
			const size_t & destination_range,
			const Vector< InputType1, reference, Coords > & source_vector,
			const InputType1 * __restrict__ const & source,
			const size_t & source_index,
			const internal::Compressed_Storage< InputType2, RowColType, NonzeroType > & matrix,
			const Vector< InputType3, reference, Coords > & mask_vector,
			const InputType3 * __restrict__ const & mask,
			const AdditiveMonoid & add,
			const Multiplication & mul,
			const std::function< size_t( size_t ) > & src_local_to_global,
			const std::function< size_t( size_t ) > & dst_global_to_local ) {
			constexpr bool add_identity = descr & descriptors::add_identity;
			assert( rc == SUCCESS );

			// check if the source vector has a meaningful element at this index
			if( ! input_dense ) {
				if( ! internal::getCoordinates( source_vector ).assigned( source_index ) ) {
					return;
				}
			}

			// mask did not fall through, so get current element
			const auto input_element = left_handed ? internal::ValueOrIndex< descr, typename Multiplication::D1, InputType1 >::get( source, src_local_to_global, source_index ) :
                                                     internal::ValueOrIndex< descr, typename Multiplication::D2, InputType1 >::get( source, src_local_to_global, source_index );
			// if we need to add identity, do so first:
			if( add_identity ) {
				const size_t global_location = src_local_to_global( source_index );
				const size_t id_location = dst_global_to_local( global_location );
#ifdef _DEBUG
				std::cout << "\t add_identity descriptor: input location == " << source_index << " -> " << global_location << " -> " << id_location << " == output location ?<? " << destination_range
						  << "\n";
#endif
				if( id_location < destination_range ) {
					typename Multiplication::D3 temp;
					internal::CopyOrApplyWithIdentity< ! left_handed, typename Multiplication::D3, InputType1, One >::set( temp, input_element, mul );
					if( output_dense || internal::getCoordinates( destination_vector ).assign( id_location ) ) {
						rc = foldl( destination[ id_location ], temp, add.getOperator() );
					} else {
						internal::CopyOrApplyWithIdentity< false, IOType, typename Multiplication::D3, AdditiveMonoid::template Identity >::set( destination[ id_location ], temp, add );
					}
				}
			}

#ifdef _DEBUG
			std::cout << "vxm_scatter, source index " << source_index << " has " << ( matrix.col_start[ source_index + 1 ] - matrix.col_start[ source_index ] ) << " nonzeroes.\n";
#endif
			// handle row or column at source_index
			for( size_t k = matrix.col_start[ source_index ]; rc == SUCCESS && k < static_cast< size_t >( matrix.col_start[ source_index + 1 ] ); ++k ) {
				// get output index
				const size_t destination_index = matrix.row_index[ k ];
				// check mask
				if( masked ) {
					if( ! internal::getCoordinates( mask_vector ).template mask< descr >( destination_index, mask ) ) {
#ifdef _DEBUG
						std::cout << "\t output to index " << destination_index << " ignored due to output masking\n";
#endif
						continue;
					}
				}
				// get nonzero
				const auto nonzero = left_handed ? matrix.template getValue( k, One< typename Multiplication::D2 >::value() ) :
                                                   matrix.template getValue( k, One< typename Multiplication::D1 >::value() );

				typename Multiplication::D3 result;
				// do multiply
#ifdef _DEBUG
				std::cout << "\t multiplying " << input_element << " with " << nonzero << "\n";
#endif
				if( left_handed ) {
					rc = apply( result, input_element, nonzero, mul );
				} else {
					rc = apply( result, nonzero, input_element, mul );
				}
				// do add
#ifdef _DEBUG
				std::cout << "\t adding " << result << " to " << destination_vector[ destination_index ] << " at index " << destination_index << "\n";
#endif
				if( rc == SUCCESS && internal::getCoordinates( destination_vector ).assign( destination_index ) ) {
					rc = foldl( destination_vector[ destination_index ], result, add.getOperator() );
				} else {
					destination_vector[ destination_index ] = static_cast< typename AdditiveMonoid::D3 >( result );
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
		 * \par Performance guarantees
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
		 *       above performance guarantees cannot be met.
		 */
		template< Descriptor descr,
			bool masked,
			bool input_masked,
			bool left_handed,
			template< typename >
			class One,
			class AdditiveMonoid,
			class Multiplication,
			typename IOType,
			typename InputType1,
			typename InputType2,
			typename InputType3,
			typename InputType4,
			typename Coords >
		RC vxm_generic( Vector< IOType, reference, Coords > & u,
			const Vector< InputType3, reference, Coords > & mask,
			const Vector< InputType1, reference, Coords > & v,
			const Vector< InputType4, reference, Coords > & v_mask,
			const Matrix< InputType2, reference > & A,
			const AdditiveMonoid & add,
			const Multiplication & mul,
			const std::function< size_t( size_t ) > & row_l2g,
			const std::function< size_t( size_t ) > & row_g2l,
			const std::function< size_t( size_t ) > & col_l2g,
			const std::function< size_t( size_t ) > & col_g2l ) {
			// type sanity checking
			NO_CAST_ASSERT(
				( descr > internal::MAX_DESCRIPTOR_VALUE || ! ( descr & descriptors::no_casting ) || std::is_same< InputType3, bool >::value ), "vxm (any variant)", "Mask type is not boolean" );
			NO_CAST_ASSERT( ( descr > internal::MAX_DESCRIPTOR_VALUE || ! ( descr & descriptors::no_casting ) || ! left_handed || std::is_same< InputType1, typename Multiplication::D1 >::value ),
				"vxm (any variant)",
				"Input vector type does not match multiplicative operator first "
				"input domain" );
			NO_CAST_ASSERT( ( descr > internal::MAX_DESCRIPTOR_VALUE || ! ( descr & descriptors::no_casting ) || left_handed || std::is_same< InputType2, typename Multiplication::D1 >::value ),
				"vxm (any variant)",
				"Input vector type does not match multiplicative operator second "
				"input domain" );
			NO_CAST_ASSERT( ( descr > internal::MAX_DESCRIPTOR_VALUE || ! ( descr & descriptors::no_casting ) || ! left_handed || std::is_same< InputType2, typename Multiplication::D2 >::value ),
				"vxm (any variant)",
				"Input matrix type does not match multiplicative operator second "
				"input domain" );
			NO_CAST_ASSERT( ( descr > internal::MAX_DESCRIPTOR_VALUE || ! ( descr & descriptors::no_casting ) || left_handed || std::is_same< InputType1, typename Multiplication::D2 >::value ),
				"vxm (any variant)",
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

			// check for dimension mismatch
			if( ( transposed && ( n != ncols( A ) || m != nrows( A ) ) ) || ( ! transposed && ( n != nrows( A ) || m != ncols( A ) ) ) ) {
#ifdef _DEBUG
				(void)printf( "Mismatch of columns (%zd vs. %zd) or rows (%zd "
							  "vs. %zd) with transposed value %d\n",
					n, ncols( A ), m, nrows( A ), (int)transposed );
#endif
				return MISMATCH;
			}

			// check mask
			if( masked ) {
				if( ( transposed && internal::getCoordinates( mask ).size() != nrows( A ) ) || ( ! transposed && internal::getCoordinates( mask ).size() != ncols( A ) ) ) {
#ifdef _DEBUG
					(void)printf( "Mismatch of mask size (%zd) versus matrix "
								  "rows or columns (%zd or %zd) with "
								  "transposed value %d\n",
						internal::getCoordinates( mask ).size(), nrows( A ), ncols( A ), (int)transposed );
#endif
					return MISMATCH;
				}
			}

			// get raw pointers
			const InputType1 * __restrict__ const x = internal::getRaw( v );
			const InputType3 * __restrict__ const z = internal::getRaw( mask );
			const InputType4 * __restrict__ const vm = internal::getRaw( v_mask );
			IOType * __restrict__ const y = internal::getRaw( u );

			// first handle trivial cases
			if( internal::getCoordinates( v ).nonzeroes() == 0 || ncols( A ) == 0 || nrows( A ) == 0 || nnz( A ) == 0 ||
				( masked && internal::getCoordinates( mask ).nonzeroes() == 0 && ! ( descr & descriptors::invert_mask ) ) ||
				( input_masked && internal::getCoordinates( v_mask ).nonzeroes() == 0 && ! ( descr & descriptors::invert_mask ) ) ) {
				// then the output must be empty
				for( size_t i = 0; i < m; ++i ) {
					if( internal::getCoordinates( u ).assigned( i ) ) {
						if( foldl( y[ i ], add.template getIdentity< IOType >(), add.getOperator() ) != SUCCESS ) {
							return PANIC;
						}
					} else if( descr & descriptors::explicit_zero ) {
						if( setElement( u, add.template getIdentity< IOType >(), i ) != SUCCESS ) {
							return PANIC;
						}
					}
				}
#ifdef _DEBUG
				std::cout << s
						  << ": trivial operation requested; exiting without "
							 "any ops. Input nonzeroes: "
						  << internal::getCoordinates( v ).nonzeroes() << ", matrix size " << nrows( A ) << " by " << ncols( A ) << " with " << nnz( A ) << " nonzeroes.\n";
#endif
				// done
				return SUCCESS;
			}

			// check for illegal arguments
			if( ! ( descr & descriptors::safe_overlap ) && reinterpret_cast< const void * >( y ) == reinterpret_cast< const void * >( x ) ) {
				std::cerr << "Warning: grb::internal::vxm_generic called with "
							 "overlapping input and output vectors.\n";
				return OVERLAP;
			}
			/*if( masked && (reinterpret_cast<const void*>(y) == reinterpret_cast<const void*>(z)) ) {
			    std::cerr << "Warning: grb::internal::vxm_generic called with overlapping mask and output vectors.\n";
			    return OVERLAP;
			}*/

#ifdef _DEBUG
			std::cout << s << ": performing SpMV / SpMSpV using an " << nrows( A ) << " by " << ncols( A ) << " matrix holding " << nnz( A ) << " nonzeroes. The input vector holds "
					  << internal::getCoordinates( v ).nonzeroes() << " nonzeroes.\n";
#endif

			// whether the input mask should be the container used for
			// iterating over input nonzeroes, or whether the input
			// vector itself should be used. This depends on which
			// would lead to a smaller loop size.
			// Abbreviations:
			// - emiim: effective mask is input mask
			// -   eim: effective input mask
			const bool emiim = input_masked ? ( ( descr & descriptors::invert_mask ) || nnz( v ) < nnz( v_mask ) ? false : true ) : false;
			const auto & eim = emiim ? internal::getCoordinates( v_mask ) : internal::getCoordinates( v );
#ifdef _DEBUG
			if( emiim ) {
				std::cout << s << ": effective mask is input mask\n";
			}
#endif

			// global return code. This will be updated by each thread from within a critical section.
			RC global_rc = SUCCESS;

#ifdef _H_GRB_REFERENCE_OMP_BLAS2
			#pragma omp parallel
			{
				internal::Coordinates< reference >::Update local_update = internal::getCoordinates( u ).EMPTY_UPDATE();
				const size_t maxAsyncAssigns = internal::getCoordinates( u ).maxAsyncAssigns();
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
					const size_t CRS_loop_size = masked ? std::min( nrows( A ), 2 * nnz( mask ) ) : nrows( A );
					const size_t CCS_seq_loop_size = ! dense_hint ?
                        std::min( ncols( A ), ( input_masked && ! ( descr & descriptors::invert_mask ) ? 2 * std::min( nnz( v_mask ), nnz( v ) ) : 2 * nnz( v ) ) ) :
                        ncols( A );
#ifdef _H_GRB_REFERENCE_OMP_BLAS2
					// This variant plays it safe and always revert to a parallel mechanism, even if we could mask on input
					// const size_t CCS_loop_size = CRS_loop_size + 1;
					// This variant modifies the sequential loop size to be P times more expensive
					const size_t CCS_loop_size = omp_get_num_threads() * CCS_seq_loop_size;
#else
				const size_t CCS_loop_size = CCS_seq_loop_size;
#endif
					// choose best-performing variant.
					if( CCS_loop_size < CRS_loop_size ) {
#ifdef _H_GRB_REFERENCE_OMP_BLAS2
						#pragma omp single
						{
#endif
							if( ! input_masked && ( dense_hint || nnz( v ) == ncols( A ) ) ) {
								// start u=vA^T using CCS
#ifdef _DEBUG
								std::cout << s << ": in full CCS variant (scatter)\n";
#endif
								// even though transposed, use CCS representation.
								// To avoid write conflicts, we keep things sequential.
								for( size_t j = 0; rc == SUCCESS && j < ncols( A ); ++j ) {
									if( input_masked && ! internal::getCoordinates( v_mask ).template mask< descr >( j, vm ) ) {
										continue;
									}
									vxm_inner_kernel_scatter< descr, dense_hint, dense_hint, masked, left_handed, One >(
										rc, u, y, nrows( A ), v, x, j, internal::getCCS( A ), mask, z, add, mul, col_l2g, row_g2l );
								}
							} else {
#ifdef _DEBUG
								std::cout << s
										  << ": in input-masked CCS variant "
											 "(scatter)\n";
#endif
								// we know the exact sparsity pattern of the input vector
								// use it to call the inner kernel on those columns of A only
								for( size_t k = 0; k < eim.nonzeroes(); ++k ) {
									const size_t j = eim.index( k );
									if( input_masked ) {
										if( ! internal::getCoordinates( v_mask ).template mask< descr >( j, vm ) ) {
#ifdef _DEBUG
											std::cout << s << "\t: input index " << j
													  << " will not be processed due to "
														 "being unmasked.\n";
#endif
											continue;
										}
										if( emiim && ! internal::getCoordinates( v ).assigned( j ) ) {
#ifdef _DEBUG
											std::cout << s << "\t: input index " << j
													  << " will not be processed due to "
														 "having no corresponding input "
														 "vector element.\n";
#endif
											continue;
										}
									}
#ifdef _DEBUG
									std::cout << s << ": processing input vector element " << j << "\n";
#endif
									vxm_inner_kernel_scatter< descr, false, dense_hint, masked, left_handed, One >(
										rc, u, y, nrows( A ), v, x, j, internal::getCCS( A ), mask, z, add, mul, col_l2g, row_g2l );
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
						if( ! masked || ( descr & descriptors::invert_mask ) ) {
							// loop over all columns of the input matrix (can be done in parallel):
#ifdef _DEBUG
							std::cout << s << ": in full CRS variant (gather)\n";
#endif
#ifdef _H_GRB_REFERENCE_OMP_BLAS2
							#pragma omp for schedule( static, config::CACHE_LINE_SIZE::value() ) \
		nowait
#endif
							for( size_t i = 0; i < nrows( A ); ++i ) {
								vxm_inner_kernel_gather< descr, masked, input_masked, left_handed, One >( rc,
#ifdef _H_GRB_REFERENCE_OMP_BLAS2
									local_update, asyncAssigns,
#endif
									u, y[ i ], i, v, x, nrows( A ), internal::getCRS( A ), mask, z, v_mask, vm, add, mul, row_l2g, col_l2g, col_g2l );
#ifdef _H_GRB_REFERENCE_OMP_BLAS2
								if( asyncAssigns == maxAsyncAssigns ) {
									// warning: return code ignored for brevity;
									//         may not be the best thing to do
									(void)internal::getCoordinates( u ).joinUpdate( local_update );
									asyncAssigns = 0;
								}
#endif
							}
						} else {
#ifdef _DEBUG
							std::cout << s << ": in masked CRS variant (gather). Mask has " << internal::getCoordinates( mask ).nonzeroes() << " nonzeroes and size "
									  << internal::getCoordinates( mask ).size() << ":\n";
							for( size_t k = 0; k < internal::getCoordinates( mask ).nonzeroes(); ++k ) {
								std::cout << " " << internal::getCoordinates( mask ).index( k );
							}
							std::cout << "\n";
#endif
							// loop only over the nonzero masks (can still be done in parallel!)
#ifdef _H_GRB_REFERENCE_OMP_BLAS2
							#pragma omp for schedule( dynamic, config::CACHE_LINE_SIZE::value() ) \
		nowait
#endif
							for( size_t k = 0; k < internal::getCoordinates( mask ).nonzeroes(); ++k ) {
								const size_t i = internal::getCoordinates( mask ).index( k );
								assert( i < nrows( A ) );
								vxm_inner_kernel_gather< descr, false, input_masked, left_handed, One >( rc,
#ifdef _H_GRB_REFERENCE_OMP_BLAS2
									local_update, asyncAssigns,
#endif
									u, y[ i ], i, v, x, nrows( A ), internal::getCRS( A ), mask, z, v_mask, vm, add, mul, row_l2g, col_l2g, col_g2l );
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
					const size_t CCS_loop_size = masked ? std::min( ncols( A ), 2 * nnz( mask ) ) : ncols( A );
					const size_t CRS_seq_loop_size = ! dense_hint ?
                        std::min( nrows( A ), ( input_masked && ! ( descr & descriptors::invert_mask ) ? 2 * std::min( nnz( v_mask ), nnz( v ) ) : 2 * nnz( v ) ) ) :
                        nrows( A );
#ifdef _H_GRB_REFERENCE_OMP_BLAS2
					// This variant ensures always choosing the parallel variant
					// const size_t CRS_loop_size = CCS_loop_size + 1;
					// This variant estimates this non-parallel variant's cost at a factor P more
					const size_t CRS_loop_size = omp_get_num_threads() * CRS_seq_loop_size;
#else
				const size_t CRS_loop_size = CRS_seq_loop_size;
#endif

					if( CRS_loop_size < CCS_loop_size ) {
#ifdef _H_GRB_REFERENCE_OMP_BLAS2
						#pragma omp single
						{
#endif
							// start u=vA using CRS, sequential implementation only
							if( ! dense_hint && nnz( v ) < nrows( A ) ) {
								// loop over nonzeroes of v only
								for( size_t k = 0; rc == SUCCESS && k < eim.nonzeroes(); ++k ) {
									const size_t i = eim.index( k );
									if( input_masked ) {
										if( ! eim.template mask< descr >( i, vm ) ) {
											continue;
										}
										if( emiim && ! internal::getCoordinates( v ).assigned( i ) ) {
											continue;
										}
									}
									vxm_inner_kernel_scatter< descr, false, dense_hint, masked, left_handed, One >(
										rc, u, y, ncols( A ), v, x, i, internal::getCRS( A ), mask, z, add, mul, row_l2g, col_g2l );
								}
							} else {
								// use straight for-loop over rows of A
								for( size_t i = 0; rc == SUCCESS && i < nrows( A ); ++i ) {
									if( input_masked ) {
										if( ! internal::getCoordinates( v_mask ).template mask< descr >( i, vm ) ) {
											continue;
										}
									}
									vxm_inner_kernel_scatter< descr, dense_hint, dense_hint, masked, left_handed, One >(
										rc, u, y, ncols( A ), v, x, i, internal::getCRS( A ), mask, z, add, mul, row_l2g, col_g2l );
								}
							}
#ifdef _H_GRB_REFERENCE_OMP_BLAS2
						}
#endif
						// end u=vA using CRS
					} else {
						// start u=vA using CCS
#ifdef _DEBUG
						std::cout << s
								  << ": in column-major vector times matrix "
									 "variant (u=vA)\n";
#endif

						// if not transposed, then CCS is the data structure to go:
						// TODO internal issue #193
						if( ! masked || ( descr & descriptors::invert_mask ) ) {
#ifdef _DEBUG
							std::cout << s << ": loop over all input matrix columns\n";
#endif
#ifdef _H_GRB_REFERENCE_OMP_BLAS2
							#pragma omp for schedule( static, config::CACHE_LINE_SIZE::value() ) \
		nowait
#endif
							for( size_t j = 0; j < ncols( A ); ++j ) {
								vxm_inner_kernel_gather< descr, masked, input_masked, left_handed, One >( rc,
#ifdef _H_GRB_REFERENCE_OMP_BLAS2
									local_update, asyncAssigns,
#endif
									u, y[ j ], j, v, x, nrows( A ), internal::getCCS( A ), mask, z, v_mask, vm, add, mul, row_l2g, row_g2l, col_l2g );
#ifdef _H_GRB_REFERENCE_OMP_BLAS2
								if( asyncAssigns == maxAsyncAssigns ) {
									// warning: return code ignored for brevity;
									//         may not be the best thing to do
									(void)internal::getCoordinates( u ).joinUpdate( local_update );
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
							#pragma omp for schedule( dynamic, config::CACHE_LINE_SIZE::value() ) \
		nowait
#endif
							for( size_t k = 0; k < internal::getCoordinates( mask ).nonzeroes(); ++k ) {
								const size_t j = internal::getCoordinates( mask ).index( k );
								vxm_inner_kernel_gather< descr, masked, input_masked, left_handed, One >( rc,
#ifdef _H_GRB_REFERENCE_OMP_BLAS2
									local_update, asyncAssigns,
#endif
									u, y[ j ], j, v, x, nrows( A ), internal::getCCS( A ), mask, z, v_mask, vm, add, mul, row_l2g, row_g2l, col_l2g );
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
				while( ! internal::getCoordinates( u ).joinUpdate( local_update ) ) {}
#endif
				if( rc != SUCCESS ) {
					global_rc = rc;
				}
#ifdef _H_GRB_REFERENCE_OMP_BLAS2
			} // end pragma omp parallel
#endif

			assert( internal::getCoordinates( u ).nonzeroes() <= m );

#ifdef _DEBUG
			std::cout << s << ": exiting SpMV / SpMSpV. Output vector contains " << nnz( u ) << " nonzeroes.\n";
			fflush( stdout );
#endif

			// done!
			return global_rc;
		}
	} // namespace internal

	/**
	 * Retrieve the row dimension size of this matrix.
	 *
	 * @returns The number of rows the current matrix contains.
	 *
	 * \parblock
	 * \par Performance guarantees.
	 *        -# This function consitutes \f$ \Theta(1) \f$ work.
	 *        -# This function allocates no additional dynamic memory.
	 *        -# This function uses \f$ \mathcal{O}(1) \f$ memory
	 *           beyond that which was already used at function entry.
	 *        -# This function will move
	 *             \f$ \mathit{sizeof}( size\_t ) \f$
	 *           bytes of memory.
	 * \endparblock
	 */
	template< typename InputType >
	size_t nrows( const Matrix< InputType, reference > & A ) noexcept {
		return A.m;
	}

	/**
	 * Retrieve the column dimension size of this matrix.
	 *
	 * @returns The number of columns the current matrix contains.
	 *
	 * \parblock
	 * \par Performance guarantees.
	 *        -# This function consitutes \f$ \Theta(1) \f$ work.
	 *        -# This function allocates no additional dynamic memory.
	 *        -# This function uses \f$ \mathcal{O}(1) \f$ memory
	 *           beyond that which was already used at function entry.
	 *        -# This function will move
	 *             \f$ \mathit{sizeof}( size\_t ) \f$
	 *           bytes of memory.
	 * \endparblock
	 */
	template< typename InputType >
	size_t ncols( const Matrix< InputType, reference > & A ) noexcept {
		return A.n;
	}

	/**
	 * Retrieve the number of nonzeroes contained in this matrix.
	 *
	 * @returns The number of nonzeroes the current matrix contains.
	 *
	 * \parblock
	 * \par Performance guarantees.
	 *        -# This function consitutes \f$ \Theta(1) \f$ work.
	 *        -# This function allocates no additional dynamic memory.
	 *        -# This function uses \f$ \mathcal{O}(1) \f$ memory
	 *           beyond that which was already used at function entry.
	 *        -# This function will move
	 *             \f$ \mathit{sizeof}( size\_t ) \f$
	 *           bytes of memory.
	 * \endparblock
	 */
	template< typename InputType >
	size_t nnz( const Matrix< InputType, reference > & A ) noexcept {
		return A.nz;
	}

	/**
	 * Resizes the nonzero capacity of this matrix. Any current contents of the
	 * matrix are \em not retained.
	 *
	 * The dimension of this matrix is fixed. Only the number of nonzeroes that
	 * may be stored can change. If the matrix has row or column dimension size
	 * zero, all calls to this function are ignored. A request for less capacity
	 * than currently already may be allocated, may be ignored by the
	 * implementation.
	 *
	 * @param[in] nonzeroes The number of nonzeroes this matrix is to contain.
	 *
	 * @return OUTOFMEM When no memory could be allocated to store this matrix.
	 * @return PANIC    When allocation fails for any other reason.
	 * @return SUCCESS  When a valid GraphBLAS matrix has been constructed.
	 *
	 * \parblock
	 * \par Performance guarantees.
	 *        -$ This function consitutes \f$ \mathcal{O}(\mathit{nz} \f$ work.
	 *        -# This function allocates \f$ \mathcal{O}(\mathit{nz}+m+n+1) \f$
	 *           bytes of dynamic memory.
	 *        -# This function will likely make system calls.
	 * \endparblock
	 *
	 * \warning This is an expensive function. Use sparingly and only when
	 *          absolutely necessary
	 */
	template< typename InputType >
	RC resize( Matrix< InputType, reference > & A, const size_t new_nz ) noexcept {
		// delegate
		return A.resize( new_nz );
	}

	/** \internal Delegates to fully masked variant */
	template< Descriptor descr = descriptors::no_operation, class Ring, typename IOType, typename InputType1, typename InputType2, typename InputType3, typename Coords >
	RC vxm( Vector< IOType, reference, Coords > & u,
		const Vector< InputType3, reference, Coords > & mask,
		const Vector< InputType1, reference, Coords > & v,
		const Matrix< InputType2, reference > & A,
		const Ring & ring = Ring(),
		const typename std::enable_if< grb::is_semiring< Ring >::value, void >::type * const = NULL ) {
		const Vector< bool, reference, Coords > empty_mask( 0 );
		return vxm< descr, true, false >( u, mask, v, empty_mask, A, ring );
	}

	/** \internal Delegates to fully masked variant */
	template< Descriptor descr = descriptors::no_operation,
		class AdditiveMonoid,
		class MultiplicativeOperator,
		typename IOType,
		typename InputType1,
		typename InputType2,
		typename InputType3,
		typename Coords >
	RC vxm( Vector< IOType, reference, Coords > & u,
		const Vector< InputType3, reference, Coords > & mask,
		const Vector< InputType1, reference, Coords > & v,
		const Matrix< InputType2, reference > & A,
		const AdditiveMonoid & add = AdditiveMonoid(),
		const MultiplicativeOperator & mul = MultiplicativeOperator(),
		const typename std::enable_if< grb::is_monoid< AdditiveMonoid >::value && grb::is_operator< MultiplicativeOperator >::value && ! grb::is_object< IOType >::value &&
				! grb::is_object< InputType1 >::value && ! grb::is_object< InputType2 >::value && ! grb::is_object< InputType3 >::value && ! std::is_same< InputType2, void >::value,
			void >::type * const = NULL ) {
		const grb::Vector< bool, reference, Coords > empty_mask( 0 );
		return vxm< descr, true, false >( u, mask, v, empty_mask, A, add, mul );
	}

	/** \internal Delegates to vxm_generic. */
	template< Descriptor descr = descriptors::no_operation,
		bool output_may_be_masked = true,
		bool input_may_be_masked = true,
		class Ring,
		typename IOType,
		typename InputType1,
		typename InputType2,
		typename InputType3,
		typename InputType4,
		typename Coords >
	RC vxm( Vector< IOType, reference, Coords > & u,
		const Vector< InputType3, reference, Coords > & mask,
		const Vector< InputType1, reference, Coords > & v,
		const Vector< InputType4, reference, Coords > & v_mask,
		const Matrix< InputType2, reference > & A,
		const Ring & ring = Ring(),
		const typename std::enable_if< grb::is_semiring< Ring >::value, void >::type * const = NULL ) {
		if( output_may_be_masked && size( v_mask ) == 0 && size( mask ) > 0 ) {
			return internal::vxm_generic< descr, true, false, true, Ring::template One >(
				u, mask, v, v_mask, A, ring.getAdditiveMonoid(), ring.getMultiplicativeOperator(),
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
			return internal::vxm_generic< descr, false, true, true, Ring::template One >(
				u, mask, v, v_mask, A, ring.getAdditiveMonoid(), ring.getMultiplicativeOperator(),
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
		} else if( output_may_be_masked && input_may_be_masked && size( mask ) > 0 && size( v_mask ) > 0 ) {
			return internal::vxm_generic< descr, true, true, true, Ring::template One >(
				u, mask, v, v_mask, A, ring.getAdditiveMonoid(), ring.getMultiplicativeOperator(),
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
			return internal::vxm_generic< descr, false, false, true, Ring::template One >(
				u, mask, v, v_mask, A, ring.getAdditiveMonoid(), ring.getMultiplicativeOperator(),
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
	template< Descriptor descr = descriptors::no_operation,
		class Ring,
		typename IOType = typename Ring::D4,
		typename InputType1 = typename Ring::D1,
		typename InputType2 = typename Ring::D2,
		typename Coords >
	RC vxm( Vector< IOType, reference, Coords > & u,
		const Vector< InputType1, reference, Coords > & v,
		const Matrix< InputType2, reference > & A,
		const Ring & ring = Ring(),
		const typename std::enable_if< grb::is_semiring< Ring >::value, void >::type * const = NULL ) {
		const Vector< bool, reference, Coords > empty_mask( 0 );
		return vxm< descr, false, false >( u, empty_mask, v, empty_mask, A, ring );
	}

	/** \internal Delegates to fully masked version */
	template< Descriptor descr = descriptors::no_operation, class AdditiveMonoid, class MultiplicativeOperator, typename IOType, typename InputType1, typename InputType2, typename Coords >
	RC vxm( Vector< IOType, reference, Coords > & u,
		const Vector< InputType1, reference, Coords > & v,
		const Matrix< InputType2, reference > & A,
		const AdditiveMonoid & add = AdditiveMonoid(),
		const MultiplicativeOperator & mul = MultiplicativeOperator(),
		const typename std::enable_if< grb::is_monoid< AdditiveMonoid >::value && grb::is_operator< MultiplicativeOperator >::value && ! grb::is_object< IOType >::value &&
				! grb::is_object< InputType1 >::value && ! grb::is_object< InputType2 >::value && ! std::is_same< InputType2, void >::value,
			void >::type * const = NULL ) {
		const grb::Vector< bool, reference, Coords > empty_mask( 0 );
		return vxm< descr, false, false >( u, empty_mask, v, empty_mask, A, add, mul );
	}

	/** \internal Delegates to fully masked version */
	template< Descriptor descr = descriptors::no_operation,
		class Ring,
		typename IOType = typename Ring::D4,
		typename InputType1 = typename Ring::D1,
		typename InputType2 = typename Ring::D2,
		typename InputType3 = bool,
		typename Coords >
	RC mxv( Vector< IOType, reference, Coords > & u,
		const Vector< InputType3, reference, Coords > & mask,
		const Matrix< InputType2, reference > & A,
		const Vector< InputType1, reference, Coords > & v,
		const Ring & ring,
		const typename std::enable_if< grb::is_semiring< Ring >::value, void >::type * const = NULL ) {
		const Vector< bool, reference, Coords > empty_mask( 0 );
		return mxv< descr, true, false >( u, mask, A, v, empty_mask, ring );
	}

	/** \internal Delegates to vxm_generic */
	template< Descriptor descr = descriptors::no_operation,
		bool output_may_be_masked = true,
		bool input_may_be_masked = true,
		class Ring,
		typename IOType,
		typename InputType1,
		typename InputType2,
		typename InputType3,
		typename InputType4,
		typename Coords >
	RC mxv( Vector< IOType, reference, Coords > & u,
		const Vector< InputType3, reference, Coords > & mask,
		const Matrix< InputType2, reference > & A,
		const Vector< InputType1, reference, Coords > & v,
		const Vector< InputType4, reference, Coords > & v_mask,
		const Ring & ring,
		const typename std::enable_if< grb::is_semiring< Ring >::value, void >::type * const = NULL ) {
		if( descr & descriptors::transpose_matrix ) {
			constexpr const Descriptor new_descr = descr & ( ~descriptors::transpose_matrix );
			if( output_may_be_masked && ( size( v_mask ) == 0 && size( mask ) > 0 ) ) {
				return internal::vxm_generic< new_descr, true, false, true, Ring::template One >(
					u, mask, v, v_mask, A, ring.getAdditiveMonoid(), ring.getMultiplicativeOperator(),
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
			} else if( input_may_be_masked && ( size( mask ) == 0 && size( v_mask ) > 0 ) ) {
				return internal::vxm_generic< new_descr, false, true, true, Ring::template One >(
					u, mask, v, v_mask, A, ring.getAdditiveMonoid(), ring.getMultiplicativeOperator(),
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
			} else if( output_may_be_masked && input_may_be_masked && size( mask ) > 0 && size( v_mask ) > 0 ) {
				return internal::vxm_generic< new_descr, true, true, true, Ring::template One >(
					u, mask, v, v_mask, A, ring.getAdditiveMonoid(), ring.getMultiplicativeOperator(),
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
				return internal::vxm_generic< new_descr, false, false, true, Ring::template One >(
					u, mask, v, v_mask, A, ring.getAdditiveMonoid(), ring.getMultiplicativeOperator(),
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
		} else {
			constexpr const Descriptor new_descr = descr | descriptors::transpose_matrix;
			if( output_may_be_masked && size( v_mask ) == 0 && size( mask ) > 0 ) {
				return internal::vxm_generic< new_descr, true, false, true, Ring::template One >(
					u, mask, v, v_mask, A, ring.getAdditiveMonoid(), ring.getMultiplicativeOperator(),
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
				return internal::vxm_generic< new_descr, false, true, true, Ring::template One >(
					u, mask, v, v_mask, A, ring.getAdditiveMonoid(), ring.getMultiplicativeOperator(),
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
			} else if( output_may_be_masked && input_may_be_masked && size( mask ) > 0 && size( v_mask ) > 0 ) {
				return internal::vxm_generic< new_descr, true, true, true, Ring::template One >(
					u, mask, v, v_mask, A, ring.getAdditiveMonoid(), ring.getMultiplicativeOperator(),
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
				return internal::vxm_generic< new_descr, false, false, true, Ring::template One >(
					u, mask, v, v_mask, A, ring.getAdditiveMonoid(), ring.getMultiplicativeOperator(),
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
	}

	/**
	 * \internal Delegates to fully masked variant.
	 */
	template< Descriptor descr = descriptors::no_operation,
		class Ring,
		typename IOType = typename Ring::D4,
		typename InputType1 = typename Ring::D1,
		typename InputType2 = typename Ring::D2,
		typename Coords >
	RC mxv( Vector< IOType, reference, Coords > & u,
		const Matrix< InputType2, reference > & A,
		const Vector< InputType1, reference, Coords > & v,
		const Ring & ring,
		const typename std::enable_if< grb::is_semiring< Ring >::value, void >::type * const = NULL ) {
		const Vector< bool, reference, Coords > empty_mask( 0 );
		return mxv< descr, false, false >( u, empty_mask, A, v, empty_mask, ring );
	}

	/** \internal Delegates to fully masked version */
	template< Descriptor descr = descriptors::no_operation, class AdditiveMonoid, class MultiplicativeOperator, typename IOType, typename InputType1, typename InputType2, typename Coords >
	RC mxv( Vector< IOType, reference, Coords > & u,
		const Matrix< InputType2, reference > & A,
		const Vector< InputType1, reference, Coords > & v,
		const AdditiveMonoid & add = AdditiveMonoid(),
		const MultiplicativeOperator & mul = MultiplicativeOperator(),
		const typename std::enable_if< grb::is_monoid< AdditiveMonoid >::value && grb::is_operator< MultiplicativeOperator >::value && ! grb::is_object< IOType >::value &&
				! grb::is_object< InputType1 >::value && ! grb::is_object< InputType2 >::value && ! std::is_same< InputType2, void >::value,
			void >::type * const = NULL ) {
		const grb::Vector< bool, reference, Coords > empty_mask( 0 );
		return mxv< descr, false, false >( u, empty_mask, A, v, empty_mask, add, mul );
	}

	/**
	 * \internal Delegates to vxm_generic
	 */
	template< Descriptor descr = descriptors::no_operation,
		bool output_may_be_masked = true,
		bool input_may_be_masked = true,
		class AdditiveMonoid,
		class MultiplicativeOperator,
		typename IOType,
		typename InputType1,
		typename InputType2,
		typename InputType3,
		typename InputType4,
		typename Coords >
	RC vxm( Vector< IOType, reference, Coords > & u,
		const Vector< InputType3, reference, Coords > & mask,
		const Vector< InputType1, reference, Coords > & v,
		const Vector< InputType4, reference, Coords > & v_mask,
		const Matrix< InputType2, reference > & A,
		const AdditiveMonoid & add = AdditiveMonoid(),
		const MultiplicativeOperator & mul = MultiplicativeOperator(),
		const typename std::enable_if< grb::is_monoid< AdditiveMonoid >::value && grb::is_operator< MultiplicativeOperator >::value && ! grb::is_object< IOType >::value &&
				! grb::is_object< InputType1 >::value && ! grb::is_object< InputType2 >::value && ! grb::is_object< InputType3 >::value && ! grb::is_object< InputType4 >::value &&
				! std::is_same< InputType2, void >::value,
			void >::type * const = NULL ) {
		if( output_may_be_masked && size( v_mask ) == 0 && size( mask ) > 0 ) {
			return internal::vxm_generic< descr, true, false, true, AdditiveMonoid::template Identity >(
				u, mask, v, v_mask, A, add, mul,
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
			return internal::vxm_generic< descr, false, true, true, AdditiveMonoid::template Identity >(
				u, mask, v, v_mask, A, add, mul,
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
		} else if( output_may_be_masked && input_may_be_masked && size( mask ) > 0 && size( v_mask ) > 0 ) {
			return internal::vxm_generic< descr, true, true, true, AdditiveMonoid::template Identity >(
				u, mask, v, v_mask, A, add, mul,
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
			return internal::vxm_generic< descr, false, false, true, AdditiveMonoid::template Identity >(
				u, mask, v, v_mask, A, add, mul,
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
	template< Descriptor descr = descriptors::no_operation,
		bool output_may_be_masked = true,
		bool input_may_be_masked = true,
		class AdditiveMonoid,
		class MultiplicativeOperator,
		typename IOType,
		typename InputType1,
		typename InputType2,
		typename InputType3,
		typename InputType4,
		typename Coords >
	RC mxv( Vector< IOType, reference, Coords > & u,
		const Vector< InputType3, reference, Coords > & mask,
		const Matrix< InputType2, reference > & A,
		const Vector< InputType1, reference, Coords > & v,
		const Vector< InputType4, reference, Coords > & v_mask,
		const AdditiveMonoid & add = AdditiveMonoid(),
		const MultiplicativeOperator & mul = MultiplicativeOperator(),
		const typename std::enable_if< grb::is_monoid< AdditiveMonoid >::value && grb::is_operator< MultiplicativeOperator >::value && ! grb::is_object< IOType >::value &&
				! grb::is_object< InputType1 >::value && ! grb::is_object< InputType2 >::value && ! grb::is_object< InputType3 >::value && ! grb::is_object< InputType4 >::value &&
				! std::is_same< InputType2, void >::value,
			void >::type * const = NULL ) {
		if( descr & descriptors::transpose_matrix ) {
			if( output_may_be_masked && size( v_mask ) == 0 && size( mask ) > 0 ) {
				return internal::vxm_generic< descr &( ~descriptors::transpose_matrix ), true, false, true, AdditiveMonoid::template Identity >(
					u, mask, v, v_mask, A, add, mul,
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
				return internal::vxm_generic< descr &( ~descriptors::transpose_matrix ), false, true, true, AdditiveMonoid::template Identity >(
					u, mask, v, v_mask, A, add, mul,
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
			} else if( output_may_be_masked && input_may_be_masked && size( mask ) > 0 && size( v_mask ) > 0 ) {
				return internal::vxm_generic< descr &( ~descriptors::transpose_matrix ), true, true, true, AdditiveMonoid::template Identity >(
					u, mask, v, v_mask, A, add, mul,
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
				return internal::vxm_generic< descr &( ~descriptors::transpose_matrix ), false, false, true, AdditiveMonoid::template Identity >(
					u, mask, v, v_mask, A, add, mul,
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
		} else {
			if( output_may_be_masked && size( v_mask ) == 0 && size( mask ) > 0 ) {
				return internal::vxm_generic< descr | descriptors::transpose_matrix, true, false, true, AdditiveMonoid::template Identity >(
					u, mask, v, v_mask, A, add, mul,
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
				return internal::vxm_generic< descr | descriptors::transpose_matrix, false, true, true, AdditiveMonoid::template Identity >(
					u, mask, v, v_mask, A, add, mul,
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
			} else if( output_may_be_masked && input_may_be_masked && size( mask ) > 0 && size( v_mask ) > 0 ) {
				return internal::vxm_generic< descr | descriptors::transpose_matrix, true, true, true, AdditiveMonoid::template Identity >(
					u, mask, v, v_mask, A, add, mul,
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
				return internal::vxm_generic< descr | descriptors::transpose_matrix, false, false, true, AdditiveMonoid::template Identity >(
					u, mask, v, v_mask, A, add, mul,
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
	}

	/**
	 * Straightforward implementation using the column-major layout.
	 *
	 * @see grb::eWiseLambda for the user-level specification.
	 */
	template< class ActiveDistribution, typename Func, typename DataType >
	RC eWiseLambda( const Func f, const Matrix< DataType, reference > & A, const size_t s, const size_t P ) {
#ifdef _DEBUG
		std::cout << "entering grb::eWiseLambda (matrices, reference ). A is " << grb::nrows( A ) << " by " << grb::ncols( A ) << " and holds " << grb::nnz( A ) << " nonzeroes.\n";
#endif
		// check for trivial call
		if( grb::nrows( A ) == 0 || grb::ncols( A ) == 0 || grb::nnz( A ) == 0 ) {
			return SUCCESS;
		}

#ifdef _H_GRB_REFERENCE_OMP_BLAS2
		#pragma omp parallel
#endif
		{
			// shift CRS start array to the left
			size_t start, end;
#ifndef _H_GRB_REFERENCE_OMP_BLAS2
			start = 1;
			end = A.m;
#else
			config::OMP::localRange( start, end, 1, A.m );
#ifdef _DEBUG
			#pragma omp critical
			std::cout << "Handling shift for " << start << "--" << end << "\n";
#endif
#endif
			const size_t cached = A.CRS.col_start[ start ];
			for( size_t i = start + 1; i <= end; ++i ) {
				A.CRS.col_start[ i - 1 ] = A.CRS.col_start[ i ];
			}
#ifdef _H_GRB_REFERENCE_OMP_BLAS2
			// make sure our write in a neighbouring thread will not
			// cause a data race on read of A.CRS.col_start[ start-1 ]
			#pragma omp barrier
#endif
			A.CRS.col_start[ start - 1 ] = cached;

			// loop over all nonzeroes in all columns using CCS
#ifndef _H_GRB_REFERENCE_OMP_BLAS2
			start = 0;
			end = A.CCS.col_start[ A.n ];
#else
			// make sure no one tries to atomically modify T entries in
			// A.CRS.col_start that are modified 8 lines above here
			#pragma omp barrier
			config::OMP::localRange( start, end, 0, A.CCS.col_start[ A.n ] );
#endif
#ifdef _DEBUG
#ifdef _H_GRB_REFERENCE_OMP_BLAS2
			#pragma omp critical
#endif
			std::cout << "\t processing range " << start << "--" << end << ".\n";
#endif

			if( start < end ) {

				// find my starting column
				size_t j_left_range = 0;
				size_t j_right_range = A.n;
				size_t j_start = A.n / 2;
				assert( A.n > 0 );
				while( j_start < A.n && ! ( A.CCS.col_start[ j_start ] <= start && start < A.CCS.col_start[ j_start + 1 ] ) ) {
#ifdef _DEBUG
#ifdef _H_GRB_REFERENCE_OMP_BLAS2
					#pragma omp critical
#endif
					std::cout << "\t binary search for " << start << " in [ " << j_left_range << ", " << j_right_range << " ) = [ " << A.CCS.col_start[ j_left_range ] << ", "
							  << A.CCS.col_start[ j_right_range ] << " ). Currently tried and failed at " << j_start << "\n";
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
				size_t j_end = A.n / 2;
				if( j_end < A.CCS.col_start[ A.n ] ) {
					while( j_end < A.n && ! ( A.CCS.col_start[ j_end ] <= end && end < A.CCS.col_start[ j_end + 1 ] ) ) {
#ifdef _DEBUG
#ifdef _H_GRB_REFERENCE_OMP_BLAS2
						#pragma omp critical
#endif
						std::cout << "\t binary search for " << end << " in [ " << j_left_range << ", " << j_right_range << " ) = [ " << A.CCS.col_start[ j_left_range ] << ", "
								  << A.CCS.col_start[ j_right_range ] << " ). Currently tried and failed at " << j_end << "\n";
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
				// preamble
				for( size_t k = start; k < std::min( static_cast< size_t >( A.CCS.col_start[ j_start + 1 ] ), end ); ++k ) {
					// get row index
					const size_t i = A.CCS.row_index[ k ];
#ifdef _DEBUG
					std::cout << "Processing nonzero at ( " << i << ", " << j_start << " )\n";
#endif
					// execute lambda on nonzero
					const size_t col_pid = ActiveDistribution::offset_to_pid( j_start, A.n, P );
					const size_t col_off = ActiveDistribution::local_offset( A.n, col_pid, P );
					const size_t global_i = ActiveDistribution::local_index_to_global( i, A.m, s, P );
					const size_t global_j = ActiveDistribution::local_index_to_global( j_start - col_off, A.n, col_pid, P );
					assert( k < A.CCS.col_start[ A.n ] );
					f( global_i, global_j, A.CCS.values[ k ] );
					// #ifndef _H_GRB_REFERENCE_OMP_BLAS2 (issue #22)
					// update CRS structure as well
					size_t k2;
					#pragma omp atomic capture //(issue #22)
					k2 = --( A.CRS.col_start[ i ] );
					assert( k2 < A.CRS.col_start[ A.m ] );
					A.CRS.values[ k2 ] = A.CCS.values[ k ];
					A.CRS.row_index[ k2 ] = j_start;
#ifdef _DEBUG
					std::cout << "CRS position: ( " << i << ", " << j_start << " ) = " << A.CRS.values[ k2 ] << " at position " << k2 << ". New start position for row " << i << " is "
							  << A.CRS.col_start[ i ] << "\n";
#endif
					// #endif (issue #22)
				}
				// main loop
				if( j_start != j_end ) {
					for( size_t j = j_start + 1; j < j_end - 1; ++j ) {
						for( size_t k = A.CCS.col_start[ j ]; k < static_cast< size_t >( A.CCS.col_start[ j + 1 ] ); ++k ) {
							// get row index
							const size_t i = A.CCS.row_index[ k ];
#ifdef _DEBUG
							std::cout << "Processing nonzero at ( " << i << ", " << j << " )\n";
#endif
							// execute lambda on nonzero
							const size_t col_pid = ActiveDistribution::offset_to_pid( j, A.n, P );
							const size_t col_off = ActiveDistribution::local_offset( A.n, col_pid, P );
							const size_t global_i = ActiveDistribution::local_index_to_global( i, A.m, s, P );
							const size_t global_j = ActiveDistribution::local_index_to_global( j - col_off, A.n, col_pid, P );
							assert( k < A.CCS.col_start[ A.n ] );
							f( global_i, global_j, A.CCS.values[ k ] );
							// #ifndef _H_GRB_REFERENCE_OMP_BLAS2 (issue #22)
							// update CRS structure as well
							assert( A.CRS.col_start[ i ] > 0 );
							size_t k2;
							#pragma omp atomic capture // (issue #22)
							k2 = --( A.CRS.col_start[ i ] );
							assert( k2 < A.CRS.col_start[ A.m ] );
							A.CRS.values[ k2 ] = A.CCS.values[ k ];
							A.CRS.row_index[ k2 ] = j;
#ifdef _DEBUG
							std::cout << "CRS position: ( " << i << ", " << j << " ) = " << A.CRS.values[ k2 ] << " at position " << k2 << ". New start position for row " << i << " is "
									  << A.CRS.col_start[ i ] << "\n";
#endif
							// #endif (issue #22)
						}
					}
					// postamble
					assert( j_end <= A.n );
					for( size_t k = A.CCS.col_start[ j_end - 1 ]; k < end; ++k ) {
						// get row index
						const size_t i = A.CCS.row_index[ k ];
#ifdef _DEBUG
						std::cout << "Processing nonzero at ( " << i << ", " << ( j_end - 1 ) << " )\n";
#endif
						// execute lambda on nonzero
						const size_t col_pid = ActiveDistribution::offset_to_pid( j_end - 1, A.n, P );
						const size_t col_off = ActiveDistribution::local_offset( A.n, col_pid, P );
						const size_t global_i = ActiveDistribution::local_index_to_global( i, A.m, s, P );
						const size_t global_j = ActiveDistribution::local_index_to_global( j_end - 1 - col_off, A.n, col_pid, P );
						assert( k < A.CCS.col_start[ A.n ] );
						f( global_i, global_j, A.CCS.values[ k ] );
						//#ifndef _H_GRB_REFERENCE_OMP_BLAS2 (issue #22)
						// update CRS structure as well
						size_t k2;
						#pragma omp atomic capture // (issue #22)
						k2 = --( A.CRS.col_start[ i ] );
						assert( k2 < A.CRS.col_start[ A.m ] );
						A.CRS.values[ k2 ] = A.CCS.values[ k ];
						A.CRS.row_index[ k2 ] = j_end - 1;
#ifdef _DEBUG
						std::cout << "CRS position: ( " << i << ", " << ( j_end - 1 ) << " ) = " << A.CRS.values[ k2 ] << " at position " << k2 << ". New start position for row " << i << " is "
								  << A.CRS.col_start[ i ] << "\n";
#endif
						//#endif (issue #22)
					}
				}
			}
		}
//#ifdef _H_GRB_REFERENCE_OMP_BLAS2 (issue #22)
//		A.CRS.copyTranspose( A.CCS );
//#endif
#ifdef _DEBUG
		std::cout << "\t exiting grb::eWiseLambda (matrices, reference). "
					 "Contents:\n";
		std::cout << "\t\t CRS row start = { ";
		for( size_t i = 0; i <= A.m; ++i ) {
			std::cout << A.CRS.col_start[ i ] << " ";
		}
		std::cout << "}\n";
		for( size_t i = 0; i < A.m; ++i ) {
			for( size_t k = A.CRS.col_start[ i ]; k < A.CRS.col_start[ i + 1 ]; ++k ) {
				std::cout << "\t\t ( " << i << ", " << A.CRS.row_index[ k ] << " ) = " << A.CRS.values[ k ] << "\n";
			}
		}
		std::cout << "\t\t CCS col start = { ";
		for( size_t j = 0; j <= A.n; ++j ) {
			std::cout << A.CCS.col_start[ j ] << " ";
		}
		std::cout << "}\n";
		for( size_t j = 0; j < A.n; ++j ) {
			for( size_t k = A.CCS.col_start[ j ]; k < A.CCS.col_start[ j + 1 ]; ++k ) {
				std::cout << "\t\t ( " << A.CCS.row_index[ k ] << ", " << j << " ) = " << A.CCS.values[ k ] << "\n";
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
	template< typename Func, typename DataType1, typename DataType2, typename Coords, typename... Args >
	RC eWiseLambda( const Func f, const Matrix< DataType1, reference > & A, const Vector< DataType2, reference, Coords > x, Args... args ) {
		// do size checking
		if( ! ( size( x ) == nrows( A ) || size( x ) == ncols( A ) ) ) {
			std::cerr << "Mismatching dimensions: given vector of size " << size( x ) << " has nothing to do with either matrix dimension (" << nrows( A ) << " nor " << ncols( A ) << ").\n";
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
