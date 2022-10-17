
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
 * Defines the GraphBLAS various descriptors.
 *
 * @author A. N. Yzelman
 * @date 15 March, 2016
 */

#ifndef _H_GRB_DESCRIPTOR
#define _H_GRB_DESCRIPTOR

#include <string>


namespace grb {

	/**
	 * Descriptors indicate pre- or post-processing for some or all of the
	 * arguments to an ALP/GraphBLAS call. An example is to transpose the input
	 * matrix during a sparse matrix--vector multiplication:
	 *    <tt>grb::mxv< grb::descriptors::transpose_matrix >( y, A, x, ring );</tt>
	 * the above thus computes \f$ y \to y + A^Tx \f$ and not \f$ y \to y + Ax \f$.
	 *
	 * Such pre-processing often happens on-the-fly, without significant overhead
	 * to the primitive costings in any of its cost dimensions -- work, intra- and
	 * inter-process data movement, synchronisations, and memory usage.
	 *
	 * \note If the application of a descriptor is \em not without significant
	 *       overhead, a backend \em must clearly indicate so.
	 *
	 * Descriptors may be combined using bit-wise operators. For instance, to both
	 * indicate the matrix needs be transposed and the mask needs be inverted, the
	 * following descriptor can be passed:
	 *    <tt> transpose_matrix | invert_mask </tt>
	 */
	typedef unsigned int Descriptor;

	/** Collection of standard descriptors. */
	namespace descriptors {

		/**
		 * Indicates no additional pre- or post-processing on any of
		 * the GraphBLAS function arguments.
		 */
		static constexpr Descriptor no_operation = 0;

		/** Inverts the mask prior to applying it. */
		static constexpr Descriptor invert_mask = 1;

		/**
		 * Transposes the input matrix prior to applying it.
		 */
		static constexpr Descriptor transpose_matrix = 2;

		/**
		 * For data ingestion methods, such as grb::buildVector or grb::buildMatrix,
		 * this descriptor indicates that the input shall not contain any duplicate
		 * entries.
		 *
		 * Use of this descriptor will speed up the corresponding function call
		 * significantly.
		 *
		 * A call to buildMatrix with this descriptor set will pass its arguments to
		 * buildMatrixUnique.
		 *
		 * \warning Use of this descriptor while the data to be ingested actually
		 *          \em does contain duplicates will lead to undefined behaviour.
		 *
		 * Currently, the reference implementation only supports ingesting data
		 * using this descriptor. Support for duplicate input is not yet
		 * implemented everywhere.
		 */
		static constexpr Descriptor no_duplicates = 4;

		/**
		 * Uses the structure of a mask vector only.
		 *
		 * This ignores the actual values of the mask argument. The i-th element of
		 * the mask now evaluates true if the mask has \em any value assigned to its
		 * i-th index, regardless of how that value evaluates. It evaluates false
		 * if there was no value assigned.
		 *
		 * @see structural_complement
		 */
		static constexpr Descriptor structural = 8;

		/**
		 * Uses the structural complement of a mask vector.
		 *
		 * This is a convenience short-hand for:
		 * \code
		 * constexpr Descriptor structural_complement = structural | invert_mask;
		 * \endcode
		 *
		 * This ignores the actual values of the mask argument. The i-th element of
		 * the mask now evaluates true if the mask has \em no value assigned to its
		 * i-th index, and evaluates false otherwise.
		 */
		static constexpr Descriptor structural_complement = structural | invert_mask;

		/**
		 * Indicates that all input and output vectors to an ALP/GraphBLAS primitive
		 * are structurally dense.
		 *
		 * If a user passes this descriptor but one or more vectors to the call are
		 * \em not structurally dense, then #ILLEGAL shall be returned.
		 *
		 * \warning <em>All vectors</em> includes any vectors that operate as masks.
		 *          Thus if the primitive is to operate with structurally sparse masks
		 *          but with otherwise dense vectors, then the dense descriptor may
		 *          \em not be defined.
		 *
		 * \warning For in-place operations with vector outputs --which are all
		 *          ALP/GraphBLAS primitives with vector outputs except grb::set and
		 *          grb::eWiseApply-- the output vector is also an input vector. Thus
		 *          passing this descriptor to such primitive indicates that also the
		 *          output vector is structurally dense.
		 *
		 * \warning For out-of-place operations with vector output(s), passing this
		 *          descriptor also demands that the output vectors are already
		 *          dense.
		 *
		 * \warning Vectors with explicit zeroes (under the semiring passed to the
		 *          related primitive) will be computed with explicitly.
		 *
		 * The benefits of using this descriptor whenever possible are two-fold:
		 *   1) less run-time overhead as code handling sparsity is disabled;
		 *   2) smaller binary sizes as code handling structurally sparse vectors is
		 *      not emitted (unless required elsewhere).
		 *
		 * The consistent use of this descriptor is hence strongly encouraged.
		 */
		static constexpr Descriptor dense = 16;

		/**
		 * For any call to a matrix computation, the input matrix \a A is instead
		 * interpreted as \f$ A+I \f$, with \a I the identity matrix of dimension
		 * matching \a A. If \a A is not square, padding zero columns or rows will
		 * be added to \a I in the largest dimension.
		 */
		static constexpr Descriptor add_identity = 32;

		/**
		 * Instead of using input vector elements, use the index of those elements.
		 *
		 * Indices are cast from their internal data type (<tt>size_t</tt>, e.g.)
		 * to the relevant domain of the operator used.
		 */
		static constexpr Descriptor use_index = 64;

		/**
		 * Disallows the standard casting of input parameters to a compatible domain
		 * in case they did not match exactly.
		 *
		 * Setting this descriptor will yield compile-time errors whenever casting
		 * would have been necessary to successfully compile the requested graphBLAS
		 * operation.
		 *
		 * \warning It is illegal to perform conditional toggling on this descriptor.
		 *
		 * \note With conditional toggling, if <tt>descr</tt> is a descriptor, we
		 *       mean <code>if( descr & descriptors::no_casting ) {
		 *                      new_descr = desc - descriptors::no_casting
		 *                      //followed by any use of this new descriptor
		 *                  }
		 *            </code>
		 *       The reason we cannot allow for this type of toggling is because this
		 *       descriptor makes use of the <tt>static_assert</tt> C++11 function,
		 *       which is checked regardless of the result of the if-statement. Thus
		 *       the above code actually always throws compile errors on mismatching
		 *       domains, no matter the original value in <tt>descr</tt>.
		 *
		 * \internal Simply making this descriptor the one with the largest integral
		 *           value amongst the various descriptors is enough to guarantee
		 *           nothing bad will happen. A notable exception are underflows,
		 *           which are caught by using internal::MAX_DESCRIPTOR_VALUE.
		 */
		static constexpr Descriptor no_casting = 256;

		/**
		 * Computation shall proceed with zeros (according to the current semiring)
		 * propagating throughout the requested computation.
		 *
		 * \warning This may lead to unexpected results if the same output container
		 * is interpreted under a different semiring-- what is zero for the current
		 * semiring may not be zero for another. In other words: the concept of
		 * sparsity will no longer generalise to other semirings.
		 */
		static constexpr Descriptor explicit_zero = 512;

		/**
		 * Indicates overlapping input and output vectors is intentional and safe, due
		 * to, for example, the use of masks.
		 */
		static constexpr Descriptor safe_overlap = 1024;

		/**
		 * For operations involving two matrices, transposes the left-hand side input
		 * matrix prior to applying it.
		 */
		static constexpr Descriptor transpose_left = 2048;

		/**
		 * For operations involving two matrices, transposes the right-hand side input
		 * matrix prior to applying it.
		 */
		static constexpr Descriptor transpose_right = 4096;

		// Put internal, backend-specific descriptors last


		/**
		 * \internal For the reference backend specifically, indicates for that the
		 * row-major storage must be used; the column-major storage shall be ignored
		 * completely. Additionally, the row-major storage is considered of static
		 * size and managed outside of ALP.
		 * This descriptor is for internal use only, and presently only supported for
		 * the grb::mxv and the grb::mxm. For the latter, only the non-transposed
		 * cases are supported.
		 */
		static constexpr Descriptor force_row_major = 8192;

		/**
		 * Translates a descriptor into a string.
		 *
		 * @param[in] descr The input descriptor.
		 *
		 * @returns A detailed English description.
		 */
		std::string toString( const Descriptor descr );

	} // namespace descriptors

	namespace internal {

		/** A descriptor cannot have a higher value than the below. */
		static constexpr Descriptor MAX_DESCRIPTOR_VALUE = 16383;

	} // namespace internal

} // namespace grb

#endif

