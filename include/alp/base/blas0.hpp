
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

#ifndef _H_ALP_BLAS0_BASE
#define _H_ALP_BLAS0_BASE

#include <type_traits> //enable_if

#include <alp/rc.hpp>
#include <alp/type_traits.hpp>

#include "config.hpp"
#include "scalar.hpp"

namespace alp {

	/**
	 * \defgroup BLAS0 The Level-0 Basic Linear Algebra Subroutines (BLAS)
	 *
	 * A collection of functions that let GraphBLAS operators work on
	 * zero-dimensional containers, i.e., on scalars.
	 *
	 * The GraphBLAS uses opaque data types and defines several standard functions
	 * to operate on these data types. Examples types are alp::Vector and
	 * alp::Matrix, example functions are alp::dot and alp::vxm.
	 *
	 * To input data into an opaque GraphBLAS type, each opaque type defines a
	 * member function \a build: alp::Vector::build() and alp::Matrix::build().
	 *
	 * To extract data from opaque GraphBLAS types, each opaque type provides
	 * \em iterators that may be obtained via the STL standard \a begin and \a end
	 * functions:
	 *   - alp::Vector::begin or alp::Vector::cbegin
	 *   - alp::Vector::end or alp::Vector::cend
	 *   - alp::Matrix::begin or alp::Matrix::cbegin
	 *   - alp::Matrix::end or alp::Matrix::cend
	 *
	 * Some GraphBLAS functions, however, reduce all elements in a GraphBLAS
	 * container into a single element of a given type. So for instance, alp::dot
	 * on two vectors of type alp::Vector<double> using the regular real semiring
	 * alp::Semiring<double> will store its output in a variable of type \a double.
	 *
	 * When parametrising GraphBLAS functions in terms of arbitrary Semirings,
	 * Monoids, Operators, and object types, it is useful to have a way to apply
	 * the same operators on whatever type they make functions like alp::dot
	 * produce-- that is, we require functions that enable the application of
	 * GraphBLAS operators on single elements.
	 *
	 * This group of BLAS level 0 functions provides this functionality.
	 *
	 * @{
	 */

	/**
	 * Out-of-place application of the operator \a OP on two data elements.
	 *
	 * The output data will be output to an existing memory location, overwriting
	 * any existing data.
	 *
	 * @tparam descr      The descriptor passed to this operator.
	 * @tparam OP         The type of the oparator to apply.
	 * @tparam InputType1 The left-hand side input argument type.
	 * @tparam InputType2 The right-hand side input argument type.
	 * @tparam OutputType The output argument type.
	 *
	 * \parblock
	 * \par Valid descriptors
	 *   -# alp::descriptors::no_operation for default behaviour.
	 *   -# alp::descriptors::no_casting when a call to this function should *not*
	 *      automatically cast input arguments to operator input domain, and *not*
	 *      automatically cast operator output to the output argument domain.
	 * \endparblock
	 *
	 * If \a InputType1 does not match the left-hand side input domain of \a OP,
	 * or if \a InputType2 does not match the right-hand side input domain of
	 * \a OP, or if \a OutputType does not match the output domain of \a OP while
	 * alp::descriptors::no_casting was set, then the code shall not compile.
	 *
	 * @param[in]  x   The left-hand side input data.
	 * @param[in]  y   The right-hand side input data.
	 * @param[out] out Where to store the result of the operator.
	 * @param[in]  op  The operator to apply (optional).
	 *
	 * \note \a op is optional when the operator type \a OP is explicitly given.
	 *       Thus there are two ways of calling this function:
	 *        -# <code>
	 *             Scalar< double > a, b, c;
	 *             alp::apply< alp::operators::add<double> >( a, b, c );
	 *           </code>, or
	 *        -# <code>
	 *             Scalar< double > a, b, c;
	 *             alp::operators::add< double > addition_over_doubles;
	 *             alp::apply( a, b, c, addition_over_doubles);
	 *           </code>
	 *
	 * \note There should be no performance difference between the two ways of
	 *       calling this function. For compatibility with other ALP
	 *       implementations, the latter type of call is preferred.
	 *
	 * @return alp::SUCCESS A call to this function never fails.
	 *
	 * \parblock
	 * \par Performance semantics.
	 *      -# This call comprises \f$ \Theta(1) \f$ work. The constant factor
	 *         depends on the cost of evaluating the operator.
	 *      -# This call takes \f$ \mathcal{O}(1) \f$ memory beyond the memory
	 *         already used by the application when a call to this function is
	 *         made.
	 *      -# This call incurs at most \f$ \Theta(1) \f$ memory where the
	 *         constant factor depends on the storage requirements of the
	 *         arguments and the temporary storage required for evaluation of
	 *         this operator.
	 * \endparblock
	 *
	 * \warning The use of stateful operators, or even thus use of stateless
	 *          operators that are not included in alp::operators, may cause this
	 *          function to incur performance penalties beyond the worst case
	 *          sketched above.
	 *
	 * @see foldr for applying an operator in-place (if allowed).
	 * @see foldl for applying an operator in-place (if allowed).
	 * @see alp::operators::internal::Operator for a discussion on when foldr and
	 *      foldl successfully generate in-place code.
	 */
	template< 
		class OP,
		typename InputType1, typename InputStructure1,
		typename InputType2, typename InputStructure2,
		typename OutputType, typename OutputStructure,
		enum Backend implementation = config::default_backend 
	>
	RC apply( 
		Scalar< OutputType, OutputStructure, implementation > &out,
		const Scalar< InputType1, InputStructure1, implementation > &x,
		const Scalar< InputType2, InputStructure2, implementation > &y,
		const OP &op = OP(),
		const std::enable_if_t<
			alp::is_operator< OP >::value &&
			!alp::is_object< InputType1 >::value &&
			!alp::is_object< InputType2 >::value &&
			!alp::is_object< OutputType >::value
		> * = nullptr
	) {
#ifdef _DEBUG
		std::cerr << "Selected backend does not implement alp::apply (scalar)\n";
#endif
#ifndef NDEBUG
		const bool backend_does_not_support_scalar_apply = false;
		assert( backend_does_not_support_scalar_apply );
#endif

		(void) out;
		(void) x;
		(void) y;
		(void) op;

		return UNSUPPORTED;
	}

	/**
	 * Application of the operator \a OP on two data elements. The output data
	 * will overwrite the right-hand side input element.
	 *
	 * In mathematical notation, this function calculates \f$ x \odot y \f$ and
	 * copies the result into \a y.
	 *
	 * @tparam OP        The type of the operator to apply.
	 * @tparam InputType The type of the left-hand side input element. This
	 *                   element will be accessed read-only.
	 * @tparam IOType    The type of the right-hand side input element, which will
	 *                   be overwritten.
	 *
	 * \parblock
	 * \par Valid descriptors
	 *   -# alp::descriptors::no_operation for default behaviour.
	 *   -# alp::descriptors::no_casting when a call to this function should *not*
	 *      automatically cast input arguments to operator input domain, and *not*
	 *      automatically cast operator output to the output argument domain.
	 * \endparblock
	 *
	 * If \a InputType does not match the left-hand side input domain
	 * (see alp::operators::internal::Operator::D1) corresponding to \a OP, then
	 * \a x will be temporarily cached and cast into \a D1.
	 * If \a IOType does not match the right-hand side input domain corresponding
	 * to \a OP, then \a y will be temporarily cached and cast into \a D2.
	 * If \a IOType does not match the output domain corresponding to \a OP, then
	 * the result of \f$ x \odot y \f$ will be temporarily cached before cast to
	 * \a IOType and written to \a y.
	 *
	 * @param[in]     x The left-hand side input parameter.
	 * @param[in,out] y On function entry: the right-hand side input parameter.
	 *                  On function exit: the output of the operator.
	 * @param[in]    op The operator to apply (optional).
	 *
	 * @return alp::SUCCESS A call to this function never fails.
	 *
	 * \parblock
	 * \par Performance semantics.
	 *      -# This call comprises \f$ \Theta(1) \f$ work. The constant factor
	 *         depends on the cost of evaluating the operator.
	 *      -# This call will not allocate any new dynamic memory.
	 *      -# This call requires at most \f$ \mathit{sizeof}(D_1+D_2+D_3) \f$
	 *         bytes of temporary storage, plus any temporary requirements for
	 *         evaluating \a op.
	 *      -# This call incurs at most \f$ \mathit{sizeof}(D_1+D_2+D_3) +
	 *         \mathit{sizeof}(\mathit{InputType}+2\mathit{IOType}) \f$ bytes of
	 *         data movement, plus any data movement requirements for evaluating
	 *         \a op.
	 * \endparblock
	 *
	 * \warning The use of stateful operators, or even thus use of stateless
	 *          operators that are not included in alp::operators, may cause this
	 *          function to incur performance penalties beyond the worst case
	 *          sketched above.
	 *
	 * \note For the standard stateless operators in alp::operators, there are
	 *       no additional temporary storage requirements nor any additional data
	 *       movement requirements than the ones mentioned above.
	 *
	 * \note If \a OP is fold-right capable, the temporary storage and data
	 *       movement requirements are less than reported above.
	 *
	 * @see foldl for a left-hand in-place version.
	 * @see apply for an example of how to call this function without explicitly
	 *            passing \a op.
	 * @see alp::operators::internal Operator for a discussion on fold-right
	 *      capable operators and on stateful operators.
	 */
	template< 
		class OP, 
		typename InputType, typename InputStructure, 
		typename IOType, typename IOStructure,
		enum Backend implementation = config::default_backend
	>
	RC foldr(
		const Scalar< InputType, InputStructure, implementation > &x,
		Scalar< IOType, IOStructure, implementation > &y,
		const OP & op = OP(),
		const std::enable_if_t<
			alp::is_operator< OP >::value &&
			! alp::is_object< InputType >::value &&
			! alp::is_object< IOType >::value
		> * = nullptr
	) {

#ifdef _DEBUG
		std::cerr << "Selected backend does not implement alp::foldr (scalar)\n";
#endif
#ifndef NDEBUG
		const bool backend_does_not_support_scalar_foldr = false;
		assert( backend_does_not_support_scalar_foldr );
#endif
		
		(void) x;
		(void) y;
		(void) op;

		return UNSUPPORTED;
	}

	/**
	 * Application of the operator \a OP on two data elements. The output data
	 * will overwrite the left-hand side input element.
	 *
	 * In mathematical notation, this function calculates \f$ x \odot y \f$ and
	 * copies the result into \a x.
	 *
	 * @tparam OP        The type of the operator to apply.
	 * @tparam IOType    The type of the left-hand side input element, which will
	 *                   be overwritten.
	 * @tparam InputType The type of the right-hand side input element. This
	 *                   element will be accessed read-only.
	 *
	 * \parblock
	 * \par Valid descriptors
	 *   -# alp::descriptors::no_operation for default behaviour.
	 *   -# alp::descriptors::no_casting when a call to this function should *not*
	 *      automatically cast input arguments to operator input domain, and *not*
	 *      automatically cast operator output to the output argument domain.
	 * \endparblock
	 *
	 * If \a InputType does not match the right-hand side input domain
	 * (see alp::operators::internal::Operator::D2) corresponding to \a OP, then
	 * \a x will be temporarily cached and cast into \a D2.
	 * If \a IOType does not match the left-hand side input domain corresponding
	 * to \a OP, then \a y will be temporarily cached and cast into \a D1.
	 * If \a IOType does not match the output domain corresponding to \a OP, then
	 * the result of \f$ x \odot y \f$ will be temporarily cached before cast to
	 * \a IOType and written to \a y.
	 *
	 * @param[in,out] x On function entry: the left-hand side input parameter.
	 *                  On function exit: the output of the operator.
	 * @param[in]     y The right-hand side input parameter.
	 * @param[in]    op The operator to apply (optional).
	 *
	 * @return alp::SUCCESS A call to this function never fails.
	 *
	 * \parblock
	 * \par Performance semantics.
	 *      -# This call comprises \f$ \Theta(1) \f$ work. The constant factor
	 *         depends on the cost of evaluating the operator.
	 *      -# This call will not allocate any new dynamic memory.
	 *      -# This call requires at most \f$ \mathit{sizeof}(D_1+D_2+D_3) \f$
	 *         bytes of temporary storage, plus any temporary requirements for
	 *         evaluating \a op.
	 *      -# This call incurs at most \f$ \mathit{sizeof}(D_1+D_2+D_3) +
	 *         \mathit{sizeof}(\mathit{InputType}+2\mathit{IOType}) \f$ bytes of
	 *         data movement, plus any data movement requirements for evaluating
	 *         \a op.
	 * \endparblock
	 *
	 * \warning The use of stateful operators, or even thus use of stateless
	 *          operators that are not included in alp::operators, may cause this
	 *          function to incur performance penalties beyond the worst case
	 *          sketched above.
	 *
	 * \note For the standard stateless operators in alp::operators, there are
	 *       no additional temporary storage requirements nor any additional data
	 *       movement requirements than the ones mentioned above.
	 *
	 * \note If \a OP is fold-left capable, the temporary storage and data
	 *       movement requirements are less than reported above.
	 *
	 * @see foldr for a right-hand in-place version.
	 * @see apply for an example of how to call this function without explicitly
	 *            passing \a op.
	 * @see alp::operators::internal Operator for a discussion on fold-right
	 *      capable operators and on stateful operators.
	 */
	template< 
		class OP, 
		typename InputType, typename InputStructure, 
		typename IOType, typename IOStructure,
		enum Backend implementation = config::default_backend
	>
	RC foldl(
		Scalar< IOType, IOStructure, implementation > &x,
		const Scalar< InputType, InputStructure, implementation > &y,
		const OP & op = OP(),
		const std::enable_if_t<
			alp::is_operator< OP >::value &&
			! alp::is_object< InputType >::value &&
			! alp::is_object< IOType >::value
		> * = nullptr
	) {

#ifdef _DEBUG
		std::cerr << "Selected backend does not implement alp::foldl (scalar)\n";
#endif
#ifndef NDEBUG
		const bool backend_does_not_support_scalar_foldl = false;
		assert( backend_does_not_support_scalar_foldl );
#endif

		(void) x;
		(void) y;
		(void) op;

		return UNSUPPORTED;
	}

	/** @} */

	namespace internal {

		/**
		 * Helper class that, depending on a given descriptor, either returns a
		 * nonzero value from a vector, or its corresponding coordinate.
		 *
		 * This class hence makes the use of the following descriptor(s) transparent:
		 *   -# #alp::descriptors::use_index
		 *
		 * @tparam descr The descriptor under which to write back either the value or
		 *               the index.
		 * @tparam OutputType The type of the output to return.
		 * @tparam D          The type of the input.
		 * @tparam Enabled    Controls, through SFINAE, whether the use of the
		 *                    #use_index descriptor is allowed at all.
		 */
		template< alp::Descriptor descr, typename OutputType, typename D, typename Enabled = void >
		class ValueOrIndex;

		/* Version where use_index is allowed. */
		template< alp::Descriptor descr, typename OutputType, typename D >
		class ValueOrIndex< 
			descr, OutputType, D,
			std::enable_if_t< 
				std::is_arithmetic< OutputType >::value
				&& ! std::is_same< D, void >::value 
			>
		> {
		private:
			static constexpr const bool use_index = descr & alp::descriptors::use_index;
			static_assert(
				use_index
				|| std::is_convertible< D, OutputType >::value, "Cannot convert to the requested output type"
			);

		public:

			static OutputType getFromScalar( const D &x, const size_t index ) noexcept {
				if( use_index ) {
					return static_cast< OutputType >( index );
				} else {
					return static_cast< OutputType >( x );
				}
			}

		};

		/* Version where use_index is not allowed. */
		template< alp::Descriptor descr, typename OutputType, typename D >
		class ValueOrIndex<
			descr, OutputType, D,
			std::enable_if_t< 
				! std::is_arithmetic< OutputType >::value
				&& ! std::is_same< OutputType, void >::value 
			>
		> {
			static_assert(
				!( descr & descriptors::use_index ),
				"use_index descriptor given while output type is not numeric"
			);
			static_assert(
				std::is_convertible< D, OutputType >::value,
				"Cannot convert input to the given output type"
			);

		public:

			static OutputType getFromScalar( const D &x, const size_t ) noexcept {
				return static_cast< OutputType >( x );
			}
		};

	} // namespace internal

} // namespace alp

#undef NO_CAST_ASSERT

#endif // end ``_H_ALP_BLAS0_BASE''
