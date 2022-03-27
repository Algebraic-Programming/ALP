
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

#ifndef _H_GRB_BASE_BLAS1
#define _H_GRB_BASE_BLAS1

#include <graphblas/rc.hpp>
#include <graphblas/ops.hpp>
#include <graphblas/phase.hpp>
#include <graphblas/monoid.hpp>
#include <graphblas/backends.hpp>
#include <graphblas/semiring.hpp>
#include <graphblas/descriptors.hpp>
#include <graphblas/internalops.hpp>

#include <assert.h>


namespace grb {

	/**
	 * \defgroup BLAS1 The Level-1 ALP/GraphBLAS routines
	 *
	 * A collection of functions that allow ALP/GraphBLAS operators, monoids, and
	 * semirings work on a mix of zero-dimensional and one-dimensional containers;
	 * i.e., allows various linear algebra operations on scalars and objects of
	 * type #grb::Vector.
	 *
	 * All functions return an error code of the enum-type #grb::RC.
	 *
	 * Primitives which produce vector output:
	 *   -# #grb::set (three variants);
	 *   -# #grb::foldr (in-place reduction to the right, scalar-to-vector and
	 *      vector-to-vector);
	 *   -# #grb::foldl (in-place reduction to the left, scalar-to-vector and
	 *      vector-to-vector);
	 *   -# #grb::eWiseApply (out-of-place application of a binary function);
	 *   -# #grb::eWiseAdd (in-place addition of two vectors, a vector and a
	 *      scalar, into a vector); and
	 *   -# #grb::eWiseMul (in-place multiplication of two vectors, a vector and a
	 *      scalar, into a vector).
	 *
	 * \note When #grb::eWiseAdd or #grb::eWiseMul using two input scalars is
	 *       required, consider forming first the resulting scalar using level-0
	 *       primitives, and then using #grb::set, #grb::foldl, or #grb::foldr, as
	 *       appropriate.
	 *
	 * Primitives that produce scalar output:
	 *   -# #grb::foldr (reduction to the right, vector-to-scalar);
	 *   -# #grb::foldl (reduction to the left, vector-to-scalar).
	 *
	 * Primitives that do not require an operator, monoid, or semiring:
	 *   -# #grb::set (three variants).
	 *
	 * Primitives that could take an operator (see #grb::operators):
	 *   -# #grb::foldr, #grb::foldl, and #grb::eWiseApply.
	 * Such operators typically can only be applied on \em dense vectors, i.e.,
	 * vectors with #grb::nnz equal to its #grb::size. Operations on sparse
	 * vectors require an intepretation of missing vector elements, which monoids
	 * or semirings provide.
	 *
	 * Therefore, all aforementioned functions are also defined for monoids instead
	 * of operators.
	 *
	 * The following functions are defined for monoids and semirings, but not for
	 * operators alone:
	 *   -# #grb::eWiseAdd (in-place addition).
	 *
	 * The following functions require a semiring, and are not defined for
	 * operators or monoids alone:
	 *   -# #grb::dot (in-place reduction of two vectors into a scalar); and
	 *   -# #grb::eWiseMul (in-place multiplication).
	 *
	 * Sometimes, operations that are defined for semirings we would sometimes also
	 * like enabled on \em improper semirings. ALP/GraphBLAS statically checks most
	 * properties required for composing proper semirings, and as such, attempts to
	 * compose improper ones will result in a compilation error. In such cases, we
	 * allow to pass an additive monoid and a multiplicative operator instead of a
	 * semiring. The following functions allow this:
	 *   -# #grb::dot, #grb::eWiseAdd, #grb::eWiseMul.
	 * The given multiplicative operator can be any binary operator, and in
	 * particular does not need to be associative.
	 *
	 * The algebraic structures lost with improper semirings typically correspond to
	 * distributivity, zero being an annihilator to multiplication, as well as the
	 * concept of \em one. Due to the latter lost structure, the above functions on
	 * impure semirings are \em not defined for pattern inputs.
	 *
	 * \warning I.e., any attempt to use containers of the form
	 *          \code
	 *              grb::Vector<void>
	 *              grb::Matrix<void>
	 *          \endcode
	 *          with an improper semiring will result in a compile-time error.
	 *
	 * \note Pattern containers are perfectly fine to use with proper semirings.
	 *
	 * \warning If an improper semiring does not have the property that the zero
	 *          identity acts as an annihilator over the multiplicative operator,
	 *          then the result of #grb::eWiseMul may be unintuitive. Please take
	 *          great care in the use of improper semrings.
	 *
	 * For fusing multiple BLAS-1 style operations on any number of inputs and
	 * outputs, users can pass their own operator function to be executed for
	 * every index \a i.
	 *   -# grb::eWiseLambda.
	 * This requires manual application of operators, monoids, and/or semirings
	 * via level-0 interface -- see #grb::apply, #grb::foldl, and #grb::foldr.
	 *
	 * For all of these functions, the element types of input and output types
	 * do not have to match the domains of the given operator, monoid, or
	 * semiring unless the #grb::descriptors::no_casting descriptor was passed.
	 *
	 * An implementation, whether blocking or non-blocking, should have clear
	 * performance semantics for every sequence of graphBLAS calls, no matter
	 * whether those are made from sequential or parallel contexts. Backends
	 * may define different performance semantics depending on which #grb::Phase
	 * primitives execute in.
	 *
	 * @{
	 */

	/**
	 * A standard vector to use for mask parameters.
	 *
	 * Indicates that no mask shall be used.
	 *
	 * \internal Do not use this symbol within backend implementations.
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
	 *
	 * \todo Revise specification regarding recent changes on phases, performance
	 *       semantics, and capacities.
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
#ifndef NDEBUG
		const bool should_not_call_base_vector_ewiselambda = false;
		assert( should_not_call_base_vector_ewiselambda );
#endif
		(void)f;
		(void)x;
		return UNSUPPORTED;
	}

	/**
	 * Foldl from a vector into a scalar.
	 *
	 * Unmasked monoid variant.
	 *
	 * \todo Write specification.
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
		const Phase &phase = EXECUTE,
		const typename std::enable_if< !grb::is_object< IOType >::value &&
			grb::is_monoid< Monoid >::value,
		void >::type * const = nullptr
	) {
#ifndef NDEBUG
		const bool should_not_call_base_scalar_foldl_monoid = false;
		assert( should_not_call_base_scalar_foldl_monoid );
#endif
		(void) y;
		(void) x;
		(void) monoid;
		(void) phase;
		return UNSUPPORTED;
	}

	/**
	 * Foldl from vector into scalar.
	 *
	 * Unmasked operator variant.
	 *
	 * \todo Write specification.
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
		const Phase &phase = EXECUTE,
		const typename std::enable_if< !grb::is_object< IOType >::value &&
			grb::is_operator< OP >::value,
		void >::type * const = nullptr
	) {
#ifndef NDEBUG
		const bool should_not_call_base_scalar_foldl_op = false;
		assert( should_not_call_base_scalar_foldl_op );
#endif
		(void) x;
		(void) y;
		(void) op;
		(void) phase;
		return UNSUPPORTED;
	}

	/**
	 * Dot product over a given semiring.
	 *
	 * \todo Write specification.
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
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			!grb::is_object< IOType >::value &&
			grb::is_semiring< Ring >::value,
		void >::type * const = nullptr
	) {
#ifdef _DEBUG
		std::cout << "Should not call base grb::dot (semiring version)\n";
#endif
		const bool should_not_call_base_dot_semiring = false;
		assert( should_not_call_base_dot_semiring );
		(void) x;
		(void) left;
		(void) right;
		(void) ring;
		(void) phase;
		return UNSUPPORTED;
	}

	/** @} */

} // end namespace grb

#endif // end _H_GRB_BASE_BLAS1

