
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

#ifndef _H_GRB_BLAS0
#define _H_GRB_BLAS0

#include <functional>
#include <stdexcept>
#include <type_traits> //enable_if

#include "graphblas/descriptors.hpp"
#include "graphblas/rc.hpp"
#include "graphblas/type_traits.hpp"

#define NO_CAST_ASSERT( x, y, z )                                                  \
	static_assert( x,                                                              \
		"\n\n"                                                                     \
		"************************************************************************" \
		"************************************************************************" \
		"**********************\n"                                                 \
		"*     ERROR      | " y " " z ".\n"                                        \
		"************************************************************************" \
		"************************************************************************" \
		"**********************\n"                                                 \
		"* Possible fix 1 | Remove no_casting from the template parameters in "    \
		"this call to " y ".\n"                                                    \
		"* Possible fix 2 | Provide a left-hand side input value of the same "     \
		"type as the first domain of the given operator.\n"                        \
		"* Possible fix 3 | Provide a right-hand side input value of the same "    \
		"type as the second domain of the given operator.\n"                       \
		"* Possible fix 4 | Provide an output value of the same type as the "      \
		"third domain of the given operator.\n"                                    \
		"* Note that in case of in-place operators the left-hand side input or "   \
		"right-hand side input also play the role of the output value.\n"          \
		"************************************************************************" \
		"************************************************************************" \
		"**********************\n" );


namespace grb {

	/**
	 * \defgroup BLAS0 Level-0 Primitives
	 * \ingroup GraphBLAS
	 *
	 * A collection of functions that let GraphBLAS operators work on
	 * zero-dimensional containers, i.e., on scalars.
	 *
	 * The GraphBLAS uses opaque data types and defines several standard functions
	 * to operate on these data types. Examples types are grb::Vector and
	 * grb::Matrix, example functions are grb::dot and grb::vxm.
	 *
	 * To input data into an opaque GraphBLAS type, each opaque type defines a
	 * member function \a build: grb::Vector::build() and grb::Matrix::build().
	 *
	 * To extract data from opaque GraphBLAS types, each opaque type provides
	 * \em iterators that may be obtained via the STL standard \a begin and \a end
	 * functions:
	 *   - grb::Vector::begin or grb::Vector::cbegin
	 *   - grb::Vector::end or grb::Vector::cend
	 *   - grb::Matrix::begin or grb::Matrix::cbegin
	 *   - grb::Matrix::end or grb::Matrix::cend
	 *
	 * Some GraphBLAS functions, however, reduce all elements in a GraphBLAS
	 * container into a single element of a given type. So for instance, grb::dot
	 * on two vectors of type grb::Vector<double> using the regular real semiring
	 * grb::Semiring<double> will store its output in a variable of type \a double.
	 *
	 * When parametrising GraphBLAS functions in terms of arbitrary Semirings,
	 * Monoids, Operators, and object types, it is useful to have a way to apply
	 * the same operators on whatever type they make functions like grb::dot
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
	 *   -# grb::descriptors::no_operation for default behaviour.
	 *   -# grb::descriptors::no_casting when a call to this function should *not*
	 *      automatically cast input arguments to operator input domain, and *not*
	 *      automatically cast operator output to the output argument domain.
	 * \endparblock
	 *
	 * If \a InputType1 does not match the left-hand side input domain of \a OP,
	 * or if \a InputType2 does not match the right-hand side input domain of
	 * \a OP, or if \a OutputType does not match the output domain of \a OP while
	 * grb::descriptors::no_casting was set, then the code shall not compile.
	 *
	 * @param[in]  x   The left-hand side input data.
	 * @param[in]  y   The right-hand side input data.
	 * @param[out] out Where to store the result of the operator.
	 * @param[in]  op  The operator to apply (optional).
	 *
	 * \note \a op is optional when the operator type \a OP is explicitly given.
	 *       Thus there are two ways of calling this function:
	 *        -# <code>
	 *             double a, b, c;
	 *             grb::apply< grb::operators::add<double> >( a, b, c );
	 *           </code>, or
	 *        -# <code>
	 *             double a, b, c;
	 *             grb::operators::add< double > addition_over_doubles;
	 *             grb::apply( a, b, c, addition_over_doubles);
	 *           </code>
	 *
	 * \note There should be no performance difference between the two ways of
	 *       calling this function. For compatibility with other GraphBLAS
	 *       implementations, the latter type of call is preferred.
	 *
	 * @return grb::SUCCESS A call to this function never fails.
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
	 *          operators that are not included in grb::operators, may cause this
	 *          function to incur performance penalties beyond the worst case
	 *          sketched above.
	 *
	 * @see foldr for applying an operator in-place (if allowed).
	 * @see foldl for applying an operator in-place (if allowed).
	 * @see grb::operators::internal::Operator for a discussion on when foldr and
	 *      foldl successfully generate in-place code.
	 */
	template< Descriptor descr = descriptors::no_operation,
		class OP,
		typename InputType1, typename InputType2, typename OutputType
	>
	static enum RC apply( OutputType &out,
		const InputType1 &x,
		const InputType2 &y,
		const OP &op = OP(),
		const typename std::enable_if<
			grb::is_operator< OP >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			!grb::is_object< OutputType >::value,
		void >::type * = NULL
	) {
		// static sanity check
		NO_CAST_ASSERT( ( !( descr & descriptors::no_casting ) || (
				std::is_same< InputType1, typename OP::D1 >::value &&
				std::is_same< InputType2, typename OP::D2 >::value &&
				std::is_same< OutputType, typename OP::D3 >::value
			) ),
			"grb::apply (BLAS level 0)",
			"Argument value types do not match operator domains while no_casting "
			"descriptor was set"
		);

		// call apply
		const typename OP::D1 left = static_cast< typename OP::D1 >( x );
		const typename OP::D2 right = static_cast< typename OP::D2 >( y );
		typename OP::D3 output = static_cast< typename OP::D3 >( out );
		op.apply( left, right, output );
		out = static_cast< OutputType >( output );

		// done
		return SUCCESS;
	}

	/**
	 * Application of the operator \a OP on two data elements. The output data
	 * will overwrite the right-hand side input element.
	 *
	 * In mathematical notation, this function calculates \f$ x \odot y \f$ and
	 * copies the result into \a y.
	 *
	 * @tparam descr     The descriptor passed to this operator.
	 * @tparam OP        The type of the operator to apply.
	 * @tparam InputType The type of the left-hand side input element. This
	 *                   element will be accessed read-only.
	 * @tparam IOType    The type of the right-hand side input element, which will
	 *                   be overwritten.
	 *
	 * \parblock
	 * \par Valid descriptors
	 *   -# grb::descriptors::no_operation for default behaviour.
	 *   -# grb::descriptors::no_casting when a call to this function should *not*
	 *      automatically cast input arguments to operator input domain, and *not*
	 *      automatically cast operator output to the output argument domain.
	 * \endparblock
	 *
	 * If \a InputType does not match the left-hand side input domain
	 * (see grb::operators::internal::Operator::D1) corresponding to \a OP, then
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
	 * @return grb::SUCCESS A call to this function never fails.
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
	 *          operators that are not included in grb::operators, may cause this
	 *          function to incur performance penalties beyond the worst case
	 *          sketched above.
	 *
	 * \note For the standard stateless operators in grb::operators, there are
	 *       no additional temporary storage requirements nor any additional data
	 *       movement requirements than the ones mentioned above.
	 *
	 * \note If \a OP is fold-right capable, the temporary storage and data
	 *       movement requirements are less than reported above.
	 *
	 * @see foldl for a left-hand in-place version.
	 * @see apply for an example of how to call this function without explicitly
	 *            passing \a op.
	 * @see grb::operators::internal Operator for a discussion on fold-right
	 *      capable operators and on stateful operators.
	 */
	template< Descriptor descr = descriptors::no_operation, class OP, typename InputType, typename IOType >
	static RC foldr( const InputType & x,
		IOType & y,
		const OP & op = OP(),
		const typename std::enable_if< grb::is_operator< OP >::value && ! grb::is_object< InputType >::value && ! grb::is_object< IOType >::value, void >::type * = NULL ) {
		// static sanity check
		NO_CAST_ASSERT( ( ! ( descr & descriptors::no_casting ) ||
							( std::is_same< InputType, typename OP::D1 >::value && std::is_same< IOType, typename OP::D2 >::value && std::is_same< IOType, typename OP::D3 >::value ) ),
			"grb::foldr (BLAS level 0)",
			"Argument value types do not match operator domains while no_casting "
			"descriptor was set" );

		// call foldr
		const typename OP::D1 left = static_cast< typename OP::D1 >( x );
		typename OP::D3 right = static_cast< typename OP::D3 >( y );
		op.foldr( left, right );
		y = static_cast< IOType >( right );

		// done
		return SUCCESS;
	}

	/**
	 * Application of the operator \a OP on two data elements. The output data
	 * will overwrite the left-hand side input element.
	 *
	 * In mathematical notation, this function calculates \f$ x \odot y \f$ and
	 * copies the result into \a x.
	 *
	 * @tparam descr     The descriptor passed to this operator.
	 * @tparam OP        The type of the operator to apply.
	 * @tparam IOType    The type of the left-hand side input element, which will
	 *                   be overwritten.
	 * @tparam InputType The type of the right-hand side input element. This
	 *                   element will be accessed read-only.
	 *
	 * \parblock
	 * \par Valid descriptors
	 *   -# grb::descriptors::no_operation for default behaviour.
	 *   -# grb::descriptors::no_casting when a call to this function should *not*
	 *      automatically cast input arguments to operator input domain, and *not*
	 *      automatically cast operator output to the output argument domain.
	 * \endparblock
	 *
	 * If \a InputType does not match the right-hand side input domain
	 * (see grb::operators::internal::Operator::D2) corresponding to \a OP, then
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
	 * @return grb::SUCCESS A call to this function never fails.
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
	 *          operators that are not included in grb::operators, may cause this
	 *          function to incur performance penalties beyond the worst case
	 *          sketched above.
	 *
	 * \note For the standard stateless operators in grb::operators, there are
	 *       no additional temporary storage requirements nor any additional data
	 *       movement requirements than the ones mentioned above.
	 *
	 * \note If \a OP is fold-left capable, the temporary storage and data
	 *       movement requirements are less than reported above.
	 *
	 * @see foldr for a right-hand in-place version.
	 * @see apply for an example of how to call this function without explicitly
	 *            passing \a op.
	 * @see grb::operators::internal Operator for a discussion on fold-right
	 *      capable operators and on stateful operators.
	 */
	template< Descriptor descr = descriptors::no_operation, class OP, typename InputType, typename IOType >
	static RC foldl( IOType &x,
		const InputType &y,
		const OP &op = OP(),
		const typename std::enable_if< grb::is_operator< OP >::value &&
			!grb::is_object< InputType >::value &&
			!grb::is_object< IOType >::value, void
		>::type * = nullptr
	) {
		// static sanity check
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) || (
				std::is_same< IOType, typename OP::D1 >::value &&
				std::is_same< InputType, typename OP::D2 >::value &&
				std::is_same< IOType, typename OP::D3 >::value
			) ), "grb::foldl (BLAS level 0)",
			"Argument value types do not match operator domains while no_casting "
			"descriptor was set" );

		// call foldl
		typename OP::D1 left = static_cast< typename OP::D1 >( x );
		const typename OP::D3 right = static_cast< typename OP::D3 >( y );
		op.foldl( left, right );
		x = static_cast< IOType >( left );

		// done
		return SUCCESS;
	}

	/** @} */

	namespace internal {

		/**
		 * Helper class that, depending on a given descriptor, either returns a
		 * nonzero value from a vector, or its corresponding coordinate.
		 *
		 * This class hence makes the use of the following descriptor(s) transparent:
		 *   -# #grb::descriptors::use_index
		 *
		 * @tparam descr The descriptor under which to write back either the value or
		 *               the index.
		 * @tparam OutputType The type of the output to return.
		 * @tparam D          The type of the input.
		 * @tparam Enabled    Controls, through SFINAE, whether the use of the
		 *                    #use_index descriptor is allowed at all.
		 */
		template< grb::Descriptor descr, typename OutputType, typename D, typename Enabled = void >
		class ValueOrIndex;

		/* Version where use_index is allowed. */
		template< grb::Descriptor descr, typename OutputType, typename D >
		class ValueOrIndex< descr, OutputType, D, typename std::enable_if< std::is_arithmetic< OutputType >::value && ! std::is_same< D, void >::value >::type > {
		private:
			static constexpr const bool use_index = descr & grb::descriptors::use_index;
			static_assert( use_index || std::is_convertible< D, OutputType >::value, "Cannot convert to the requested output type" );

		public:
			static OutputType getFromArray( const D * __restrict__ const x, const std::function< size_t( size_t ) > & src_local_to_global, const size_t index ) noexcept {
				if( use_index ) {
					return static_cast< OutputType >( src_local_to_global( index ) );
				} else {
					return static_cast< OutputType >( x[ index ] );
				}
			}
			static OutputType getFromScalar( const D &x, const size_t index ) noexcept {
				if( use_index ) {
					return static_cast< OutputType >( index );
				} else {
					return static_cast< OutputType >( x );
				}
			}
		};

		/* Version where use_index is not allowed. */
		template< grb::Descriptor descr, typename OutputType, typename D >
		class ValueOrIndex< descr, OutputType, D, typename std::enable_if< ! std::is_arithmetic< OutputType >::value && ! std::is_same< OutputType, void >::value >::type > {
			static_assert( ! ( descr & descriptors::use_index ), "use_index descriptor given while output type is not numeric" );
			static_assert( std::is_convertible< D, OutputType >::value, "Cannot convert input to the given output type" );

		public:
			static OutputType getFromArray( const D * __restrict__ const x, const std::function< size_t( size_t ) > &, const size_t index ) noexcept {
				return static_cast< OutputType >( x[ index ] );
			}
			static OutputType getFromScalar( const D &x, const size_t ) noexcept {
				return static_cast< OutputType >( x );
			}
		};

		/**
		 * Helper class that, depending on the type, sets an output value to a given
		 * input value, either by cast-and-assign (if that is possible), or by
		 * applying a given operator with a given left- or right-identity to generate
		 * a matching requested output value.
		 *
		 * This transparently `lifts' input arguments to different domains whenever
		 * required, and allows the use of highly generic semirings.
		 *
		 * @tparam identity_left If an identity is applied, whether the left-identity
		 *                       must be used. If false, the right-identity will be
		 *                       used (if indeed an identity is to be applied).
		 * @tparam OutputType The type of the output to return.
		 * @tparam InputType  The type of the input.
		 * @tparam Identity   The class that can generate both left- and right-
		 *                    identities.
		 * @tparam Enabled    Controls, through SFINAE, whether cast-and-assign or the
		 *                    operator version is used instead.
		 */

		template< bool identity_left, typename OutputType, typename InputType, template< typename > class Identity, typename Enabled = void >
		class CopyOrApplyWithIdentity;

		/* The cast-and-assign version */
		template< bool identity_left, typename OutputType, typename InputType, template< typename > class Identity >
		class CopyOrApplyWithIdentity< identity_left, OutputType, InputType, Identity, typename std::enable_if< std::is_convertible< InputType, OutputType >::value >::type > {
		public:
			template< typename Operator >
			static void set( OutputType & out, const InputType & in, const Operator & ) {
				out = static_cast< OutputType >( in );
			}
		};

		/* The operator with identity version */
		template< bool identity_left, typename OutputType, typename InputType, template< typename > class Identity >
		class CopyOrApplyWithIdentity< identity_left, OutputType, InputType, Identity, typename std::enable_if< ! std::is_convertible< InputType, OutputType >::value >::type > {
		public:
			template< typename Operator >
			static void set( OutputType & out, const InputType & in, const Operator & op ) {
				const auto identity = identity_left ? Identity< typename Operator::D1 >::value() : Identity< typename Operator::D2 >::value();
				if( identity_left ) {
					(void)grb::apply( out, identity, in, op );
				} else {
					(void)grb::apply( out, in, identity, op );
				}
			}
		};

	} // namespace internal

} // namespace grb

#undef NO_CAST_ASSERT

#endif // end ``_H_GRB_BLAS0''
