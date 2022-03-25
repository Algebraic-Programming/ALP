
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
 * @date 29th of March 2017
 */

#ifndef _H_GRB_BLAS1
#define _H_GRB_BLAS1

#include <graphblas/backends.hpp>
#include <graphblas/config.hpp>
#include <graphblas/rc.hpp>
#include <graphblas/type_traits.hpp>

#include "base/vector.hpp"

#ifdef _GRB_WITH_REFERENCE
 #include <graphblas/reference/blas1.hpp>
#endif
#ifdef _GRB_WITH_BANSHEE
 #include <graphblas/banshee/blas1.hpp>
#endif
#ifdef _GRB_WITH_LPF
 #include <graphblas/bsp1d/blas1.hpp>
#endif

// the remainder implements several backend-agnostic short-cuts

#define NO_CAST_RING_ASSERT( x, y, z )                                             \
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
		"* Possible fix 2 | For all mismatches in the domains of input "           \
		"parameters and the semiring domains, as specified in the documentation "  \
		"of the function " y ", supply an input argument of the expected type "    \
		"instead.\n"                                                               \
		"* Possible fix 3 | Provide a compatible semiring where all domains "      \
		"match those of the input parameters, as specified in the documentation "  \
		"of the function " y ".\n"                                                 \
		"************************************************************************" \
		"************************************************************************" \
		"**********************\n" );

namespace grb {

	/**
	 * A standard vector to use for mask parameters. Indicates no mask shall be
	 * used.
	 */
	#define NO_MASK Vector< bool >( 0 )

	/**
	 * Executes an arbitrary element-wise user-defined function \a f using any
	 * number of vectors of equal length, following the nonzero pattern of the
	 * given vector \a x.
	 *
	 * The user-defined function is passed as a lambda which can capture, at
	 * the very least, other instances of type grb::Vector. Use of this function
	 * is preferable whenever multiple element-wise operations are requested that
	 * use one or more identical input vectors. Performing the computation one
	 * after the other in blocking mode would require the same vector to be
	 * streamed multiple times, while with this function the operations can be
	 * fused explicitly instead.
	 *
	 * It shall always be legal to capture non-GraphBLAS objects for read access
	 * only. It shall \em not be legal to capture instances of type grb::Matrix
	 * for read and/or write access.
	 *
	 * If grb::Properties::writableCaptured evaluates true then captured
	 * non-GraphBLAS objects can also be written to, not just read from. The
	 * captured variable is, however, completely local to the calling user process
	 * only-- it will not be synchronised between user processes.
	 * As a rule of thumb, data-centric GraphBLAS implementations \em cannot
	 * support this and will thus have grb::Properties::writableCaptured evaluate
	 * to false. A portable GraphBLAS algorithm should provide a different code
	 * path to handle this case.
	 * When it is legal to write to captured scalar, this function can, e.g., be
	 * used to perform reduction-like operations on any number of equally sized
	 * input vectors.  This would be preferable to a chained number of calls to
	 * grb::dot in case where some vectors are shared between subsequent calls,
	 * for example; the shared vectors are streamed only once using this lambda-
	 * enabled function.
	 *
	 * \warning The lambda shall only be executed on the data local to the user
	 *          process calling this function! This is different from the various
	 *          fold functions, or grb::dot, in that the semantics of those
	 *          functions always end with a globally synchronised result. To
	 *          achieve the same effect with user-defined lambdas, the users
	 *          should manually prescribe how to combine the local results into
	 *          global ones, for instance, by a subsequent call to
	 *          grb::collectives<>::allreduce.
	 *
	 * \note This is an addition to the GraphBLAS. It is alike user-defined
	 *       operators, monoids, and semirings, except it allows execution on
	 *       arbitrarily many inputs and arbitrarily many outputs.
	 *
	 * @tparam Func the user-defined lambda function type.
	 * @tparam DataType the type of the user-supplied vector example.
	 * @tparam backend  the backend type of the user-supplied vector example.
	 *
	 * @param[in] f The user-supplied lambda. This lambda should only capture
	 *              and reference vectors of the same length as \a x. The lambda
	 *              function should prescribe the operations required to execute
	 *              at a given index \a i. Captured GraphBLAS vectors can access
	 *              that element via the operator[]. It is illegal to access any
	 *              element not at position \a i. The lambda takes only the single
	 *              parameter \a i of type <code>const size_t</code>. Captured
	 *              scalars will not be globally updated-- the user must program
	 *              this explicitly. Scalars and other non-GraphBLAS containers
	 *              are always local to their user process.
	 * @param[in] x The vector the lambda will be executed on. This argument
	 *              determines which indices \a i will be accessed during the
	 *              elementwise operation-- elements with indices \a i that
	 *              do not appear in \a x will be skipped during evaluation of
	 *              \a f.
	 * @param[in] args All vectors the lambda is to access elements of. Must be of
	 *                 the same length as \a x. If this constraint is violated,
	 *                 grb::MISMATCH shall be returned. <em>This is a variadic
	 *                 argument and can contain any number of containers of type
	 *                 grb::Vector, passed as though they were separate
	 *                 arguments.</em>
	 *
	 * \note In future GraphBLAS implementations, \a args, apart from doing
	 *       dimension checking, should also facilitate any data distribution
	 *       necessary to successfully execute the element-wise operation. Current
	 *       implementations do not require this since they use the same static
	 *       distribution for all containers.
	 *
	 * \warning Using a grb::Vector inside a lambda passed to this function while
	 *          not passing that same vector into \a args, will result in undefined
	 *          behaviour.
	 *
	 * \note It would be natural to have \a x equal to one of the captured
	 *       GraphBLAS vectors in \a f.
	 *
	 * \warning Due to the constraints on \a f described above, it is illegal to
	 *          capture some vector \a y and have the following line in the body
	 *          of \a f: <code>x[i] += x[i+1]</code>. Vectors can only be
	 *          dereferenced at position \a i and \a i alone.
	 *
	 * @return grb::SUCCESS  When the lambda is successfully executed.
	 * @return grb::MISMATCH When two or more vectors passed to \a args are not of
	 *                       equal length.
	 *
	 * \parblock
	 * \par Example.
	 *
	 * An example valid use:
	 *
	 * \code
	 * void f(
	 *      double &alpha,
	 *      grb::Vector< double > &y,
	 *      const double beta,
	 *      const grb::Vector< double > &x,
	 *      const grb::Semiring< double > ring
	 * ) {
	 *      assert( grb::size(x) == grb::size(y) );
	 *      assert( grb::nnz(x) == grb::size(x) );
	 *      assert( grb::nnz(y) == grb::size(y) );
	 *      alpha = ring.getZero();
	 *      grb::eWiseLambda(
	 *          [&alpha,beta,&x,&y,ring]( const size_t i ) {
	 *              double mul;
	 *              const auto mul_op = ring.getMultiplicativeOperator();
	 *              const auto add_op = ring.getAdditiveOperator();
	 *              grb::apply( y[i], beta, x[i], mul_op );
	 *              grb::apply( mul, x[i], y[i], mul_op );
	 *              grb::foldl( alpha, mul, add_op );
	 *      }, x, y );
	 *      grb::collectives::allreduce( alpha, add_op );
	 * }
	 * \endcode
	 *
	 * This code takes a value \a beta, a vector \a x, and a semiring \a ring and
	 * computes:
	 *   1) \a y as the element-wise multiplication (under \a ring) of \a beta and
	 *      \a x; and
	 *   2) \a alpha as the dot product (under \a ring) of \a x and \a y.
	 * This function can easily be made agnostic to whatever exact semiring is used
	 * by templating the type of \a ring. As it is, this code is functionally
	 * equivalent to:
	 *
	 * \code
	 * grb::eWiseMul( y, beta, x, ring );
	 * grb::dot( alpha, x, y, ring );
	 * \endcode
	 *
	 * The version using the lambdas, however, is expected to execute
	 * faster as both \a x and \a y are streamed only once, while the
	 * latter code may stream both vectors twice.
	 * \endparblock
	 *
	 * \warning The following code is invalid:
	 *          \code
	 *              template< class Operator >
	 *              void f(
	 *                   grb::Vector< double > &x,
	 *                   const Operator op
	 *              ) {
	 *                   grb::eWiseLambda(
	 *                       [&x,&op]( const size_t i ) {
	 *                           grb::apply( x[i], x[i], x[i+1], op );
	 *                   }, x );
	 *              }
	 *          \endcode
	 *          Only a Vector::lambda_reference to position exactly equal to \a i
	 *          may be used within this function.
	 *
	 * \warning There is no similar concept in the official GraphBLAS specs.
	 *
	 * \warning Captured scalars will be local to the user process executing the
	 *          lambda. To retrieve the global dot product, an allreduce must
	 *          explicitly be called.
	 *
	 * @see Vector::operator[]()
	 * @see Vector::lambda_reference
	 */
	template<
		typename Func,
		typename DataType,
		Backend backend,
		typename Coords,
		typename... Args
	>
	RC eWiseLambda(
		const Func f,
		const Vector< DataType, backend, Coords > & x, Args...
	) {
		(void)f;
		(void)x;
		return PANIC;
	}

	/**
	 * Alias for a simple reduce call.
	 *
	 * Will use no mask and will set the accumulator to the given Monoid's
	 * operator.
	 */
	template<
		Descriptor descr = descriptors::no_operation,
		class Monoid,
		typename IOType, typename InputType,
		Backend backend,
		typename Coords
	>
	RC foldl( IOType &x,
		const Vector< InputType, backend, Coords > &y,
		const Monoid &monoid = Monoid(),
		const typename std::enable_if< !grb::is_object< IOType >::value &&
			grb::is_monoid< Monoid >::value,
		void >::type * const = NULL
	) {
		// create empty mask
		Vector< bool, backend, Coords > mask( 0 );
		// call regular reduce function
		return foldl< descr >( x, y, mask, monoid );
	}

	/**
	 * Alias for a simple reduce call.
	 *
	 * Will use no mask and will set the accumulator to the given Monoid's
	 * operator.
	 */
	template<
		Descriptor descr = descriptors::no_operation,
		class OP,
		typename IOType, typename InputType,
		Backend backend, typename Coords
	>
	RC foldl( IOType &x,
		const Vector< InputType, backend, Coords > &y,
		const OP &op = OP(),
		const typename std::enable_if< !grb::is_object< IOType >::value &&
			grb::is_operator< OP >::value,
		void >::type * const = NULL
	) {
		// create empty mask
		Vector< bool, backend, Coords > mask( 0 );
		// call regular reduce function
		return foldl< descr >( x, y, mask, op );
	}

	/**
	 * Provides a generic implementation of the dot computation on semirings by
	 * translating it into a dot computation on an additive commutative monoid
	 * with any multiplicative operator.
	 *
	 * For return codes, exception behaviour, performance semantics, template
	 * and non-template arguments, @see grb::dot.
	 */
	template<
		Descriptor descr = descriptors::no_operation, class Ring,
		typename IOType, typename InputType1, typename InputType2,
		Backend backend, typename Coords
	>
	RC dot( IOType &x,
		const Vector< InputType1, backend, Coords > &left,
		const Vector< InputType2, backend, Coords > &right,
		const Ring &ring = Ring(),
		const typename std::enable_if<
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			!grb::is_object< IOType >::value &&
			grb::is_semiring< Ring >::value,
		void >::type * const = NULL
	) {
		return grb::dot< descr >( x,
			left, right,
			ring.getAdditiveMonoid(),
			ring.getMultiplicativeOperator()
		);
	}

	/**
	 * Provides a generic implementation of the 2-norm computation.
	 *
	 * Proceeds by computing a dot-product on itself and then taking the square
	 * root of the result.
	 *
	 * This function is only available when the output type is floating point.
	 *
	 * For return codes, exception behaviour, performance semantics, template
	 * and non-template arguments, @see grb::dot.
	 *
	 * @param[out] x The 2-norm of \a y. The input value of \a x will be ignored.
	 * @param[in]  y The vector to compute the norm of.
	 * @param[in] ring The Semiring under which the 2-norm is to be computed.
	 *
	 * \warning This function computes \a x out-of-place. This is contrary to
	 *          standard ALP/GraphBLAS functions that are always in-place.
	 *
	 * \warning A \a ring is not sufficient for computing a two-norm. This
	 *          implementation assumes the standard <tt>sqrt</tt> function
	 *          must be applied on the result of a dot-product of \a y with
	 *          itself under the supplied semiring.
	 */
	// template<
	// 	Descriptor descr = descriptors::no_operation, class Ring,
	// 	typename InputType, typename OutputType, typename OutputStructure,
	// 	Backend backend, typename Coords
	// >
	// RC norm2( Scalar< OutputType, OutputStructure, backend > &x,
	// 	const Vector< InputType, backend, Coords > &y,
	// 	const Ring &ring = Ring(),
	// 	const typename std::enable_if<
	// 		std::is_floating_point< OutputType >::value,
	// 	void >::type * const = NULL
	// ) {
	// 	RC ret = grb::dot< descr >( x, y, y, ring );
	// 	if( ret == SUCCESS ) {
	// 		x = sqrt( x );
	// 	}
	// 	return ret;
	// }

	/** Specialization for C++ scalars */
	template<
		Descriptor descr = descriptors::no_operation, class Ring,
		typename InputType, typename OutputType,
		Backend backend, typename Coords
	>
	RC norm2( OutputType &x,
		const Vector< InputType, backend, Coords > &y,
		const Ring &ring = Ring(),
		const typename std::enable_if<
			std::is_floating_point< OutputType >::value,
		void >::type * const = NULL
	) {
		RC ret = grb::dot< descr >( x, y, y, ring );
		if( ret == SUCCESS ) {
			x = sqrt( x );
		}
		return ret;
	}


} // namespace grb

#undef NO_CAST_RING_ASSERT

#endif // end ``_H_GRB_BLAS1''

