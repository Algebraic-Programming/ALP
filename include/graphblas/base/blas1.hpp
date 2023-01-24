
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
 * Defines the ALP/GraphBLAS level-1 API
 *
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
	 * \defgroup BLAS1 Level-1 Primitives
	 * \ingroup GraphBLAS
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
	 * Computes \f$ z = \alpha \odot \beta \f$, out of place, operator version.
	 *
	 * @tparam descr      The descriptor to be used. Equal to
	 *                    descriptors::no_operation if left unspecified.
	 * @tparam OP         The operator to use.
	 * @tparam InputType1 The value type of the left-hand vector.
	 * @tparam InputType2 The value type of the right-hand scalar.
	 * @tparam OutputType The value type of the ouput vector.
	 *
	 * @param[out]  z   The output vector.
	 * @param[in] alpha The left-hand input scalar.
	 * @param[in]  beta The right-hand input scalar.
	 * @param[in]   op  The operator \f$ \odot \f$.
	 * @param[in] phase  The #grb::Phase the call should execute. Optional; the
	 *                   default parameter is #grb::EXECUTE.
	 *
	 * Specialisation scalar inputs, operator version. A call to this function
	 * (with #grb::EXECUTE for \a phase) is equivalent to the following code:
	 *
	 * \code
	 * typename OP::D3 tmp;
	 * grb::apply( tmp, x, y, op );
	 * grb::set( z, tmp );
	 * \endcode
	 *
	 * @return #grb::SUCCESS  On successful completion of this call.
	 * @return #grb::FAILED   If \a phase is #grb::EXECUTE, indicates that the
	 *                        capacity of \a z was insufficient. The output vector
	 *                        \a z is cleared, and the call to this function has no
	 *                        further effects.
	 * @return #grb::OUTOFMEM If \a phase is #grb::RESIZE, indicates an
	 *                        out-of-memory exception. The call to this function
	 *                        shall have no other effects beyond returning this
	 *                        error code; the previous state of \a z is retained.
	 * @return #grb::PANIC    A general unmitigable error has been encountered. If
	 *                        returned, ALP enters an undefined state and the user
	 *                        program is encouraged to exit as quickly as possible.
	 *
	 * \par Performance semantics
	 * Each backend must define performance semantics for this primitive.
	 *
	 * @see perfSemantics
	 */
	template<
		Descriptor descr = descriptors::no_operation,
		class OP, enum Backend backend,
		typename OutputType, typename InputType1, typename InputType2,
		typename Coords
	>
	RC eWiseApply(
		Vector< OutputType, backend, Coords > &z,
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
		std::cout << "In eWiseApply ([T1]<-T2<-T3), operator, base\n";
#endif
#ifndef NDEBUG
		const bool should_not_call_eWiseApplyOpASS_base = false;
		assert( should_not_call_eWiseApplyOpASS_base );
#endif
		(void) z;
		(void) alpha;
		(void) beta;
		(void) op;
		(void) phase;
		return UNSUPPORTED;
	}

	/**
	 * Computes \f$ z = \alpha \odot \beta \f$, out of place, monoid version.
	 *
	 * @tparam descr      The descriptor to be used. Equal to
	 *                    descriptors::no_operation if left unspecified.
	 * @tparam Monoid     The monoid to use.
	 * @tparam InputType1 The value type of the left-hand vector.
	 * @tparam InputType2 The value type of the right-hand scalar.
	 * @tparam OutputType The value type of the ouput vector.
	 *
	 * @param[out]  z    The output vector.
	 * @param[in]  alpha The left-hand input scalar.
	 * @param[in]   beta The right-hand input scalar.
	 * @param[in] monoid The monoid with underlying operator \f$ \odot \f$.
	 * @param[in] phase  The #grb::Phase the call should execute. Optional; the
	 *                   default parameter is #grb::EXECUTE.
	 *
	 * Specialisation scalar inputs, monoid version. A call to this function
	 * (with #grb::EXECUTE for \a phase) is equivalent to the following code:
	 *
	 * \code
	 * typename OP::D3 tmp;
	 * grb::apply( tmp, x, y, monoid.getOperator() );
	 * grb::set( z, tmp );
	 * \endcode
	 *
	 * @return #grb::SUCCESS  On successful completion of this call.
	 * @return #grb::FAILED   If \a phase is #grb::EXECUTE, indicates that the
	 *                        capacity of \a z was insufficient. The output vector
	 *                        \a z is cleared, and the call to this function has no
	 *                        further effects.
	 * @return #grb::OUTOFMEM If \a phase is #grb::RESIZE, indicates an
	 *                        out-of-memory exception. The call to this function
	 *                        shall have no other effects beyond returning this
	 *                        error code; the previous state of \a z is retained.
	 * @return #grb::PANIC    A general unmitigable error has been encountered. If
	 *                        returned, ALP enters an undefined state and the user
	 *                        program is encouraged to exit as quickly as possible.
	 *
	 * \par Performance semantics
	 * Each backend must define performance semantics for this primitive.
	 *
	 * @see perfSemantics
	 */
	template<
		Descriptor descr = descriptors::no_operation,
		class Monoid, enum Backend backend,
		typename OutputType, typename InputType1, typename InputType2,
		typename Coords
	>
	RC eWiseApply(
		Vector< OutputType, backend, Coords > &z,
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
		std::cout << "In eWiseApply ([T1]<-T2<-T3), monoid, base\n";
#endif
#ifndef NDEBUG
		const bool should_not_call_eWiseApplyMonASS_base = false;
		assert( should_not_call_eWiseApplyMonASS_base );
#endif
		(void) z;
		(void) alpha;
		(void) beta;
		(void) monoid;
		(void) phase;
		return UNSUPPORTED;
	}

	/**
	 * Computes \f$ z = \alpha \odot y \f$, out of place, operator version.
	 *
	 * Calculates the element-wise operation on one scalar to elements of one
	 * vector, \f$ z = \alpha \odot y \f$, using the given operator. The input and
	 * output vectors must be of equal length.
	 *
	 * For all indices \a i of \a z, its element \f$ z_i \f$ after the call to this
	 * function completes equals \f$ \alpha \odot y_i \f$. Any old entries of \a z
	 * are removed. Entries \a i for which \a y has no nonzero will be skipped.
	 *
	 * After a successful call to this primitive, the sparsity structure of \a z
	 * shall match that of \a y.
	 *
	 * \note When applying element-wise operators on sparse vectors using
	 *       semirings, there is a difference between interpreting missing values
	 *       as an annihilating identity or as a neutral identity-- intuitively,
	 *       such identities are known as `zero' or `one', respectively. As a
	 *       consequence, there are two different variants for element-wise
	 *       operations whose names correspond to their intuitive meanings:
	 *        - #grb::eWiseAdd (neutral), and
	 *        - #grb::eWiseMul (annihilating).
	 *       The above two primitives require a semiring. The same functionality is
	 *       provided by #grb::eWiseApply depending on whether a monoid or operator
	 *       is provided:
	 *        - #grb::eWiseApply using monoids (neutral),
	 *        - #grb::eWiseApply using operators (annihilating).
	 *
	 * \note However, #grb::eWiseAdd and #grb::eWiseMul provide in-place semantics,
	 *       while #grb::eWiseApply does not.
	 *
	 * \note An #grb::eWiseAdd with some semiring and a #grb::eWiseApply using its
	 *       additive monoid thus are equivalent if operating when operating on
	 *       empty outputs.
	 *
	 * \note An #grb::eWiseMul with some semiring and a #grb::eWiseApply using its
	 *       multiplicative operator thus are equivalent when operating on empty
	 *       outputs.
	 *
	 * @tparam descr      The descriptor to be used. Equal to
	 *                    descriptors::no_operation if left unspecified.
	 * @tparam OP         The operator to use.
	 * @tparam InputType1 The value type of the left-hand vector.
	 * @tparam InputType2 The value type of the right-hand scalar.
	 * @tparam OutputType The value type of the ouput vector.
	 *
	 * @param[out]  z   The output vector.
	 * @param[in] alpha The left-hand input scalar.
	 * @param[in]   y   The right-hand input vector.
	 * @param[in]  op   The operator \f$ \odot \f$.
	 * @param[in] phase The #grb::Phase the call should execute. Optional; the
	 *                  default parameter is #grb::EXECUTE.
	 *
	 * @return #grb::SUCCESS  On successful completion of this call.
	 * @return #grb::MISMATCH Whenever the dimensions of \a y and \a z do not
	 *                        match. All input data containers are left untouched
	 *                        if this exit code is returned; it will be as though
	 *                        this call was never made.
	 * @return #grb::FAILED   If \a phase is #grb::EXECUTE, indicates that the
	 *                        capacity of \a z was insufficient. The output vector
	 *                        \a z is cleared, and the call to this function has no
	 *                        further effects.
	 * @return #grb::OUTOFMEM If \a phase is #grb::RESIZE, indicates an
	 *                        out-of-memory exception. The call to this function
	 *                        shall have no other effects beyond returning this
	 *                        error code; the previous state of \a z is retained.
	 * @return #grb::PANIC    A general unmitigable error has been encountered. If
	 *                        returned, ALP enters an undefined state and the user
	 *                        program is encouraged to exit as quickly as possible.
	 *
	 * \par Performance semantics
	 * Each backend must define performance semantics for this primitive.
	 *
	 * @see perfSemantics
	 */
	template<
		Descriptor descr = descriptors::no_operation,
		class OP, enum Backend backend,
		typename OutputType, typename InputType1, typename InputType2,
		typename Coords
	>
	RC eWiseApply(
		Vector< OutputType, backend, Coords > &z,
		const InputType1 alpha,
		const Vector< InputType2, backend, Coords > &y,
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
		std::cout << "In eWiseApply ([T1]<-T2<-[T3]), operator, base\n";
#endif
#ifndef NDEBUG
		const bool should_not_call_eWiseApplyOpASA_base = false;
		assert( should_not_call_eWiseApplyOpASA_base );
#endif
		(void) z;
		(void) alpha;
		(void) y;
		(void) op;
		(void) phase;
		return UNSUPPORTED;
	}

	/**
	 * Computes \f$ z = \alpha \odot y \f$, out of place, masked operator version.
	 *
	 * Calculates the element-wise operation on one scalar to elements of one
	 * vector, \f$ z = \alpha \odot y \f$, using the given operator. The input and
	 * output vectors must be of equal length.
	 *
	 * For all indices \a i of \a z, its element \f$ z_i \f$ after the call to this
	 * function completes equals \f$ \alpha \odot y_i \f$. Any old entries of \a z
	 * are removed. Entries \a i for which \a y has no nonzero will be skipped, as
	 * will entries \a i for which \a mask evaluates <tt>false</tt>.
	 *
	 * \note When applying element-wise operators on sparse vectors using
	 *       semirings, there is a difference between interpreting missing values
	 *       as an annihilating identity or as a neutral identity-- intuitively,
	 *       such identities are known as `zero' or `one', respectively. As a
	 *       consequence, there are two different variants for element-wise
	 *       operations whose names correspond to their intuitive meanings:
	 *        - #grb::eWiseAdd (neutral), and
	 *        - #grb::eWiseMul (annihilating).
	 *       The above two primitives require a semiring. The same functionality is
	 *       provided by #grb::eWiseApply depending on whether a monoid or operator
	 *       is provided:
	 *        - #grb::eWiseApply using monoids (neutral),
	 *        - #grb::eWiseApply using operators (annihilating).
	 *
	 * \note However, #grb::eWiseAdd and #grb::eWiseMul provide in-place semantics,
	 *       while #grb::eWiseApply does not.
	 *
	 * \note An #grb::eWiseAdd with some semiring and a #grb::eWiseApply using its
	 *       additive monoid thus are equivalent if operating when operating on
	 *       empty outputs.
	 *
	 * \note An #grb::eWiseMul with some semiring and a #grb::eWiseApply using its
	 *       multiplicative operator thus are equivalent when operating on empty
	 *       outputs.
	 *
	 * @tparam descr      The descriptor to be used. Equal to
	 *                    descriptors::no_operation if left unspecified.
	 * @tparam OP         The operator to use.
	 * @tparam InputType1 The value type of the left-hand vector.
	 * @tparam InputType2 The value type of the right-hand scalar.
	 * @tparam OutputType The value type of the ouput vector.
	 * @tparam MaskType   The value type of the mask vector.
	 *
	 * @param[out]  z   The output vector.
	 * @param[in]  mask The output mask.
	 * @param[in] alpha The left-hand input scalar.
	 * @param[in]   y   The right-hand input vector.
	 * @param[in]  op   The operator \f$ \odot \f$.
	 * @param[in] phase The #grb::Phase the call should execute. Optional; the
	 *                  default parameter is #grb::EXECUTE.
	 *
	 * @return #grb::SUCCESS  On successful completion of this call.
	 * @return #grb::MISMATCH Whenever the dimensions of \a y and \a z do not
	 *                        match. All input data containers are left untouched
	 *                        if this exit code is returned; it will be as though
	 *                        this call was never made.
	 * @return #grb::FAILED   If \a phase is #grb::EXECUTE, indicates that the
	 *                        capacity of \a z was insufficient. The output vector
	 *                        \a z is cleared, and the call to this function has no
	 *                        further effects.
	 * @return #grb::OUTOFMEM If \a phase is #grb::RESIZE, indicates an
	 *                        out-of-memory exception. The call to this function
	 *                        shall have no other effects beyond returning this
	 *                        error code; the previous state of \a z is retained.
	 * @return #grb::PANIC    A general unmitigable error has been encountered. If
	 *                        returned, ALP enters an undefined state and the user
	 *                        program is encouraged to exit as quickly as possible.
	 *
	 * \par Performance semantics
	 * Each backend must define performance semantics for this primitive.
	 *
	 * @see perfSemantics
	 */
	template<
		Descriptor descr = descriptors::no_operation,
		class OP, enum Backend backend,
		typename OutputType, typename MaskType,
		typename InputType1, typename InputType2,
		typename Coords
	>
	RC eWiseApply(
		Vector< OutputType, backend, Coords > &z,
		const Vector< MaskType, backend, Coords > &mask,
		const InputType1 alpha,
		const Vector< InputType2, backend, Coords > &y,
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
		std::cout << "In masked eWiseApply ([T1]<-T2<-[T3], operator, base)\n";
#endif
#ifndef NDEBUG
		const bool should_not_call_eWiseApplyOpAMSA_base = false;
		assert( should_not_call_eWiseApplyOpAMSA_base );
#endif
		(void) z;
		(void) mask;
		(void) alpha;
		(void) y;
		(void) op;
		(void) phase;
		return UNSUPPORTED;
	}

	/**
	 * Computes \f$ z = \alpha \odot y \f$, out of place, monoid version.
	 *
	 * Calculates the element-wise operation on one scalar to elements of one
	 * vector, \f$ z = \alpha \odot y \f$, using the given operator. The input and
	 * output vectors must be of equal length.
	 *
	 * For all indices \a i of \a z, its element \f$ z_i \f$ after the call to this
	 * function completes equals \f$ \alpha \odot y_i \f$. Any old entries of \a z
	 * are removed.
	 *
	 * After a successful call to this primitive, \a z shall be dense.
	 *
	 * \note When applying element-wise operators on sparse vectors using
	 *       semirings, there is a difference between interpreting missing values
	 *       as an annihilating identity or as a neutral identity-- intuitively,
	 *       such identities are known as `zero' or `one', respectively. As a
	 *       consequence, there are two different variants for element-wise
	 *       operations whose names correspond to their intuitive meanings:
	 *        - #grb::eWiseAdd (neutral), and
	 *        - #grb::eWiseMul (annihilating).
	 *       The above two primitives require a semiring. The same functionality is
	 *       provided by #grb::eWiseApply depending on whether a monoid or operator
	 *       is provided:
	 *        - #grb::eWiseApply using monoids (neutral),
	 *        - #grb::eWiseApply using operators (annihilating).
	 *
	 * \note However, #grb::eWiseAdd and #grb::eWiseMul provide in-place semantics,
	 *       while #grb::eWiseApply does not.
	 *
	 * \note An #grb::eWiseAdd with some semiring and a #grb::eWiseApply using its
	 *       additive monoid thus are equivalent if operating when operating on
	 *       empty outputs.
	 *
	 * \note An #grb::eWiseMul with some semiring and a #grb::eWiseApply using its
	 *       multiplicative operator thus are equivalent when operating on empty
	 *       outputs.
	 *
	 * @tparam descr      The descriptor to be used. Equal to
	 *                    descriptors::no_operation if left unspecified.
	 * @tparam Monoid     The monoid to use.
	 * @tparam InputType1 The value type of the left-hand vector.
	 * @tparam InputType2 The value type of the right-hand scalar.
	 * @tparam OutputType The value type of the ouput vector.
	 *
	 * @param[out]  z    The output vector.
	 * @param[in] alpha  The left-hand input scalar.
	 * @param[in]   y    The right-hand input vector.
	 * @param[in] monoid The monoid that provides the operator \f$ \odot \f$.
	 * @param[in] phase  The #grb::Phase the call should execute. Optional; the
	 *                   default parameter is #grb::EXECUTE.
	 *
	 * @return #grb::SUCCESS  On successful completion of this call.
	 * @return #grb::MISMATCH Whenever the dimensions of \a y and \a z do not
	 *                        match. All input data containers are left untouched
	 *                        if this exit code is returned; it will be as though
	 *                        this call was never made.
	 * @return #grb::FAILED   If \a phase is #grb::EXECUTE, indicates that the
	 *                        capacity of \a z was insufficient. The output vector
	 *                        \a z is cleared, and the call to this function has no
	 *                        further effects.
	 * @return #grb::OUTOFMEM If \a phase is #grb::RESIZE, indicates an
	 *                        out-of-memory exception. The call to this function
	 *                        shall have no other effects beyond returning this
	 *                        error code; the previous state of \a z is retained.
	 * @return #grb::PANIC    A general unmitigable error has been encountered. If
	 *                        returned, ALP enters an undefined state and the user
	 *                        program is encouraged to exit as quickly as possible.
	 *
	 * \par Performance semantics
	 * Each backend must define performance semantics for this primitive.
	 *
	 * @see perfSemantics
	 */
	template<
		Descriptor descr = descriptors::no_operation,
		class Monoid, enum Backend backend,
		typename OutputType, typename InputType1, typename InputType2,
		typename Coords
	>
	RC eWiseApply(
		Vector< OutputType, backend, Coords > &z,
		const InputType1 alpha,
		const Vector< InputType2, backend, Coords > &y,
		const Monoid &monoid = Monoid(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if< !grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			grb::is_monoid< Monoid >::value, void
		>::type * const = nullptr
	) {
#ifdef _DEBUG
		std::cout << "In unmasked eWiseApply ([T1]<-T2<-[T3], monoid, base)\n";
#endif
#ifndef NDEBUG
		const bool should_not_call_eWiseApplyMonoidASA_base = false;
		assert( should_not_call_eWiseApplyMonoidASA_base );
#endif
		(void) z;
		(void) alpha;
		(void) y;
		(void) monoid;
		(void) phase;
		return UNSUPPORTED;
	}

	/**
	 * Computes \f$ z = \alpha \odot y \f$, out of place, masked monoid variant.
	 *
	 * Calculates the element-wise operation on one scalar to elements of one
	 * vector, \f$ z = \alpha \odot y \f$, using the given operator. The input and
	 * output vectors must be of equal length.
	 *
	 * For all indices \a i of \a z, its element \f$ z_i \f$ after the call to this
	 * function completes equals \f$ \alpha \odot y_i \f$. Any old entries of \a z
	 * are removed. Entries \a i for which \a mask evaluates <tt>false</tt> will be
	 * skipped.
	 *
	 * After a successful call to this primitive, the sparsity structure of \a z
	 * shall match that of \a mask (after interpretation).
	 *
	 * \note When applying element-wise operators on sparse vectors using
	 *       semirings, there is a difference between interpreting missing values
	 *       as an annihilating identity or as a neutral identity-- intuitively,
	 *       such identities are known as `zero' or `one', respectively. As a
	 *       consequence, there are two different variants for element-wise
	 *       operations whose names correspond to their intuitive meanings:
	 *        - #grb::eWiseAdd (neutral), and
	 *        - #grb::eWiseMul (annihilating).
	 *       The above two primitives require a semiring. The same functionality is
	 *       provided by #grb::eWiseApply depending on whether a monoid or operator
	 *       is provided:
	 *        - #grb::eWiseApply using monoids (neutral),
	 *        - #grb::eWiseApply using operators (annihilating).
	 *
	 * \note However, #grb::eWiseAdd and #grb::eWiseMul provide in-place semantics,
	 *       while #grb::eWiseApply does not.
	 *
	 * \note An #grb::eWiseAdd with some semiring and a #grb::eWiseApply using its
	 *       additive monoid thus are equivalent if operating when operating on
	 *       empty outputs.
	 *
	 * \note An #grb::eWiseMul with some semiring and a #grb::eWiseApply using its
	 *       multiplicative operator thus are equivalent when operating on empty
	 *       outputs.
	 *
	 * @tparam descr      The descriptor to be used. Equal to
	 *                    descriptors::no_operation if left unspecified.
	 * @tparam Monoid     The monoid to use.
	 * @tparam InputType1 The value type of the left-hand vector.
	 * @tparam InputType2 The value type of the right-hand scalar.
	 * @tparam OutputType The value type of the ouput vector.
	 * @tparam MaskType   The value type of the output mask vector.
	 *
	 * @param[out]  z    The output vector.
	 * @param[out] mask  The output mask.
	 * @param[in] alpha  The left-hand input scalar.
	 * @param[in]   y    The right-hand input vector.
	 * @param[in] monoid The monoid that provides the operator \f$ \odot \f$.
	 * @param[in] phase  The #grb::Phase the call should execute. Optional; the
	 *                   default parameter is #grb::EXECUTE.
	 *
	 * @return #grb::SUCCESS  On successful completion of this call.
	 * @return #grb::MISMATCH Whenever the dimensions of \a mask, a y and \a z do
	 *                        not match. All input data containers are left
	 *                        untouched if this exit code is returned; it will be
	 *                        as though this call was never made.
	 * @return #grb::FAILED   If \a phase is #grb::EXECUTE, indicates that the
	 *                        capacity of \a z was insufficient. The output vector
	 *                        \a z is cleared, and the call to this function has no
	 *                        further effects.
	 * @return #grb::OUTOFMEM If \a phase is #grb::RESIZE, indicates an
	 *                        out-of-memory exception. The call to this function
	 *                        shall have no other effects beyond returning this
	 *                        error code; the previous state of \a z is retained.
	 * @return #grb::PANIC    A general unmitigable error has been encountered. If
	 *                        returned, ALP enters an undefined state and the user
	 *                        program is encouraged to exit as quickly as possible.
	 *
	 * \par Performance semantics
	 * Each backend must define performance semantics for this primitive.
	 *
	 * @see perfSemantics
	 */
	template<
		Descriptor descr = descriptors::no_operation,
		class Monoid, enum Backend backend,
		typename OutputType, typename MaskType,
		typename InputType1, typename InputType2,
		typename Coords
	>
	RC eWiseApply(
		Vector< OutputType, backend, Coords > &z,
		const Vector< MaskType, backend, Coords > &mask,
		const InputType1 alpha,
		const Vector< InputType2, backend, Coords > &y,
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
#ifndef NDEBUG
		const bool should_not_call_eWiseApplyMonoidAMSA_base = false;
		assert( should_not_call_eWiseApplyMonoidAMSA_base );
#endif
		(void) z;
		(void) mask;
		(void) alpha;
		(void) y;
		(void) monoid;
		(void) phase;
		return UNSUPPORTED;
	}

	/**
	 * Computes \f$ z = x \odot \beta \f$, out of place, operator variant.
	 *
	 * Calculates the element-wise operation on one scalar to elements of one
	 * vector, \f$ z = x .* \beta \f$, using the given operator. The input and
	 * output vectors must be of equal length.
	 *
	 * For all valid indices \a i of \a z, its element \f$ z_i \f$ after the call
	 * to this function completes equals \f$ x_i \odot \beta \f$. Any old entries
	 * of \a z are removed.
	 *
	 * Entries \a i for which no nonzero exists in \a x are skipped. Therefore,
	 * after a successful call to this primitive, the nonzero structure of \a z
	 * will match that of \a x.
	 *
	 * \note When applying element-wise operators on sparse vectors using
	 *       semirings, there is a difference between interpreting missing values
	 *       as an annihilating identity or as a neutral identity-- intuitively,
	 *       such identities are known as `zero' or `one', respectively. As a
	 *       consequence, there are two different variants for element-wise
	 *       operations whose names correspond to their intuitive meanings:
	 *        - #grb::eWiseAdd (neutral), and
	 *        - #grb::eWiseMul (annihilating).
	 *       The above two primitives require a semiring. The same functionality is
	 *       provided by #grb::eWiseApply depending on whether a monoid or operator
	 *       is provided:
	 *        - #grb::eWiseApply using monoids (neutral),
	 *        - #grb::eWiseApply using operators (annihilating).
	 *
	 * \note However, #grb::eWiseAdd and #grb::eWiseMul provide in-place semantics,
	 *       while #grb::eWiseApply does not.
	 *
	 * \note An #grb::eWiseAdd with some semiring and a #grb::eWiseApply using its
	 *       additive monoid thus are equivalent if operating when operating on
	 *       empty outputs.
	 *
	 * \note An #grb::eWiseMul with some semiring and a #grb::eWiseApply using its
	 *       multiplicative operator thus are equivalent when operating on empty
	 *       outputs.
	 *
	 * @tparam descr      The descriptor to be used. Equal to
	 *                    descriptors::no_operation if left unspecified.
	 * @tparam OP         The operator to use.
	 * @tparam InputType1 The value type of the left-hand vector.
	 * @tparam InputType2 The value type of the right-hand scalar.
	 * @tparam OutputType The value type of the ouput vector.
	 *
	 * @param[out]  z   The output vector.
	 * @param[in]   x   The left-hand input vector.
	 * @param[in]  beta The right-hand input scalar.
	 * @param[in]   op  The operator \f$ \odot \f$.
	 * @param[in] phase The #grb::Phase the call should execute. Optional; the
	 *                  default parameter is #grb::EXECUTE.
	 *
	 * @return #grb::SUCCESS  On successful completion of this call.
	 * @return #grb::MISMATCH Whenever the dimensions of \a x and \a z do not
	 *                        match. All input data containers are left untouched
	 *                        if this exit code is returned; it will be as though
	 *                        this call was never made.
	 * @return #grb::FAILED   If \a phase is #grb::EXECUTE, indicates that the
	 *                        capacity of \a z was insufficient. The output vector
	 *                        \a z is cleared, and the call to this function has no
	 *                        further effects.
	 * @return #grb::OUTOFMEM If \a phase is #grb::RESIZE, indicates an
	 *                        out-of-memory exception. The call to this function
	 *                        shall have no other effects beyond returning this
	 *                        error code; the previous state of \a z is retained.
	 * @return #grb::PANIC    A general unmitigable error has been encountered. If
	 *                        returned, ALP enters an undefined state and the user
	 *                        program is encouraged to exit as quickly as possible.
	 *
	 * \par Performance semantics
	 * Each backend must define performance semantics for this primitive.
	 *
	 * @see perfSemantics
	 */
	template<
		Descriptor descr = descriptors::no_operation,
		class OP, enum Backend backend,
		typename OutputType, typename InputType1, typename InputType2,
		typename Coords
	>
	RC eWiseApply(
		Vector< OutputType, backend, Coords > &z,
		const Vector< InputType1, backend, Coords > &x,
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
		std::cout << "In eWiseApply ([T1]<-[T2]<-T3), operator, base\n";
#endif
#ifndef NDEBUG
		const bool should_not_call_eWiseApplyOpAAS_base = false;
		assert( should_not_call_eWiseApplyOpAAS_base );
#endif
		(void) z;
		(void) x;
		(void) beta;
		(void) op;
		(void) phase;
		return UNSUPPORTED;
	}

	/**
	 * Computes \f$ z = x \odot \beta \f$, out of place, masked operator variant.
	 *
	 * Calculates the element-wise operation on one scalar to elements of one
	 * vector, \f$ z = x .* \beta \f$, using the given operator. The input and
	 * output vectors must be of equal length.
	 *
	 * For all valid indices \a i of \a z, its element \f$ z_i \f$ after the call
	 * to this function completes equals \f$ x_i \odot \beta \f$. Any old entries
	 * of \a z are removed.
	 *
	 * Entries \a i for which no nonzero exists in \a x are skipped. Entries \a i
	 * for which the mask evaluates <tt>false</tt> are skipped as well.
	 *
	 * \note When applying element-wise operators on sparse vectors using
	 *       semirings, there is a difference between interpreting missing values
	 *       as an annihilating identity or as a neutral identity-- intuitively,
	 *       such identities are known as `zero' or `one', respectively. As a
	 *       consequence, there are two different variants for element-wise
	 *       operations whose names correspond to their intuitive meanings:
	 *        - #grb::eWiseAdd (neutral), and
	 *        - #grb::eWiseMul (annihilating).
	 *       The above two primitives require a semiring. The same functionality is
	 *       provided by #grb::eWiseApply depending on whether a monoid or operator
	 *       is provided:
	 *        - #grb::eWiseApply using monoids (neutral),
	 *        - #grb::eWiseApply using operators (annihilating).
	 *
	 * \note However, #grb::eWiseAdd and #grb::eWiseMul provide in-place semantics,
	 *       while #grb::eWiseApply does not.
	 *
	 * \note An #grb::eWiseAdd with some semiring and a #grb::eWiseApply using its
	 *       additive monoid thus are equivalent if operating when operating on
	 *       empty outputs.
	 *
	 * \note An #grb::eWiseMul with some semiring and a #grb::eWiseApply using its
	 *       multiplicative operator thus are equivalent when operating on empty
	 *       outputs.
	 *
	 * @tparam descr      The descriptor to be used. Equal to
	 *                    descriptors::no_operation if left unspecified.
	 * @tparam OP         The operator to use.
	 * @tparam InputType1 The value type of the left-hand vector.
	 * @tparam InputType2 The value type of the right-hand scalar.
	 * @tparam OutputType The value type of the output vector.
	 * @tparam MaskType   The value type of the output mask vector.
	 *
	 * @param[out]  z   The output vector.
	 * @param[in]  mask The output mask.
	 * @param[in]   x   The left-hand input vector.
	 * @param[in]  beta The right-hand input scalar.
	 * @param[in]   op  The operator \f$ \odot \f$.
	 * @param[in] phase The #grb::Phase the call should execute. Optional; the
	 *                  default parameter is #grb::EXECUTE.
	 *
	 * @return #grb::SUCCESS  On successful completion of this call.
	 * @return #grb::MISMATCH Whenever the dimensions of \a mask, \a x, and \a z do
	 *                        not match. All input data containers are left
	 *                        untouched if this exit code is returned; it will be
	 *                        as though this call was never made.
	 * @return #grb::FAILED   If \a phase is #grb::EXECUTE, indicates that the
	 *                        capacity of \a z was insufficient. The output vector
	 *                        \a z is cleared, and the call to this function has no
	 *                        further effects.
	 * @return #grb::OUTOFMEM If \a phase is #grb::RESIZE, indicates an
	 *                        out-of-memory exception. The call to this function
	 *                        shall have no other effects beyond returning this
	 *                        error code; the previous state of \a z is retained.
	 * @return #grb::PANIC    A general unmitigable error has been encountered. If
	 *                        returned, ALP enters an undefined state and the user
	 *                        program is encouraged to exit as quickly as possible.
	 *
	 * \par Performance semantics
	 * Each backend must define performance semantics for this primitive.
	 *
	 * @see perfSemantics
	 */
	template<
		Descriptor descr = descriptors::no_operation,
		class OP, enum Backend backend,
		typename OutputType, typename MaskType,
		typename InputType1, typename InputType2,
		typename Coords
	>
	RC eWiseApply(
		Vector< OutputType, backend, Coords > &z,
		const Vector< MaskType, backend, Coords > &mask,
		const Vector< InputType1, backend, Coords > &x,
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
		std::cout << "In masked eWiseApply ([T1]<-[T2]<-T3, operator, base)\n";
#endif
#ifndef NDEBUG
		const bool should_not_call_eWiseApplyOpAMAS_base = false;
		assert( should_not_call_eWiseApplyOpAMAS_base );
#endif
		(void) z;
		(void) mask;
		(void) x;
		(void) beta;
		(void) op;
		(void) phase;
		return UNSUPPORTED;
	}

	/**
	 * Computes \f$ z = x \odot \beta \f$, out of place, monoid variant.
	 *
	 * Calculates the element-wise operation on one scalar to elements of one
	 * vector, \f$ z = x \odot \beta \f$, using the given operator. The input and
	 * output vectors must be of equal length.
	 *
	 * For all indices \a i of \a z, its element \f$ z_i \f$ after the call to this
	 * function completes equals \f$ x_i \odot \beta \f$. Any old entries of \a z
	 * are removed.
	 *
	 * After a successful call to this primitive, \a z shall be dense.
	 *
	 * \note When applying element-wise operators on sparse vectors using
	 *       semirings, there is a difference between interpreting missing values
	 *       as an annihilating identity or as a neutral identity-- intuitively,
	 *       such identities are known as `zero' or `one', respectively. As a
	 *       consequence, there are two different variants for element-wise
	 *       operations whose names correspond to their intuitive meanings:
	 *        - #grb::eWiseAdd (neutral), and
	 *        - #grb::eWiseMul (annihilating).
	 *       The above two primitives require a semiring. The same functionality is
	 *       provided by #grb::eWiseApply depending on whether a monoid or operator
	 *       is provided:
	 *        - #grb::eWiseApply using monoids (neutral),
	 *        - #grb::eWiseApply using operators (annihilating).
	 *
	 * \note However, #grb::eWiseAdd and #grb::eWiseMul provide in-place semantics,
	 *       while #grb::eWiseApply does not.
	 *
	 * \note An #grb::eWiseAdd with some semiring and a #grb::eWiseApply using its
	 *       additive monoid thus are equivalent if operating when operating on
	 *       empty outputs.
	 *
	 * \note An #grb::eWiseMul with some semiring and a #grb::eWiseApply using its
	 *       multiplicative operator thus are equivalent when operating on empty
	 *       outputs.
	 *
	 * @tparam descr      The descriptor to be used. Equal to
	 *                    descriptors::no_operation if left unspecified.
	 * @tparam Monoid     The monoid to use.
	 * @tparam InputType1 The value type of the left-hand vector.
	 * @tparam InputType2 The value type of the right-hand scalar.
	 * @tparam OutputType The value type of the ouput vector.
	 *
	 * @param[out]  z    The output vector.
	 * @param[in]   x    The left-hand input vector.
	 * @param[in]  beta  The right-hand input scalar.
	 * @param[in] monoid The monoid that provides the operator \f$ \odot \f$.
	 * @param[in] phase  The #grb::Phase the call should execute. Optional; the
	 *                   default parameter is #grb::EXECUTE.
	 *
	 * @return #grb::SUCCESS  On successful completion of this call.
	 * @return #grb::MISMATCH Whenever the dimensions of \a x and \a z do not
	 *                        match. All input data containers are left untouched
	 *                        if this exit code is returned; it will be as though
	 *                        this call was never made.
	 * @return #grb::FAILED   If \a phase is #grb::EXECUTE, indicates that the
	 *                        capacity of \a z was insufficient. The output vector
	 *                        \a z is cleared, and the call to this function has no
	 *                        further effects.
	 * @return #grb::OUTOFMEM If \a phase is #grb::RESIZE, indicates an
	 *                        out-of-memory exception. The call to this function
	 *                        shall have no other effects beyond returning this
	 *                        error code; the previous state of \a z is retained.
	 * @return #grb::PANIC    A general unmitigable error has been encountered. If
	 *                        returned, ALP enters an undefined state and the user
	 *                        program is encouraged to exit as quickly as possible.
	 *
	 * \par Performance semantics
	 * Each backend must define performance semantics for this primitive.
	 *
	 * @see perfSemantics
	 */
	template<
		Descriptor descr = descriptors::no_operation,
		class Monoid, enum Backend backend,
		typename OutputType, typename InputType1, typename InputType2,
		typename Coords
	>
	RC eWiseApply(
		Vector< OutputType, backend, Coords > &z,
		const Vector< InputType1, backend, Coords > &x,
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
		std::cout << "In unmasked eWiseApply ([T1]<-[T2]<-T3, monoid, base)\n";
#endif
#ifndef NDEBUG
		const bool should_not_call_eWiseApplyMonoidAAS_base = false;
		assert( should_not_call_eWiseApplyMonoidAAS_base );
#endif
		(void) z;
		(void) x;
		(void) beta;
		(void) monoid;
		(void) phase;
		return UNSUPPORTED;
	}

	/**
	 * Computes \f$ z = x \odot \beta \f$, out of place, masked monoid variant.
	 *
	 * Calculates the element-wise operation on one scalar to elements of one
	 * vector, \f$ z = x \odot \beta \f$, using the given operator. The input and
	 * output vectors must be of equal length.
	 *
	 * For all indices \a i of \a z, its element \f$ z_i \f$ after the call to this
	 * function completes equals \f$ x_i \odot \beta \f$. Any old entries of \a z
	 * are removed. Entries \a i for which \a mask evaluates <tt>false</tt> will be
	 * skipped.
	 *
	 * After a successful call to this primitive, the sparsity structure of \a z
	 * matches that of \a mask (after interpretation).
	 *
	 * \note When applying element-wise operators on sparse vectors using
	 *       semirings, there is a difference between interpreting missing values
	 *       as an annihilating identity or as a neutral identity-- intuitively,
	 *       such identities are known as `zero' or `one', respectively. As a
	 *       consequence, there are two different variants for element-wise
	 *       operations whose names correspond to their intuitive meanings:
	 *        - #grb::eWiseAdd (neutral), and
	 *        - #grb::eWiseMul (annihilating).
	 *       The above two primitives require a semiring. The same functionality is
	 *       provided by #grb::eWiseApply depending on whether a monoid or operator
	 *       is provided:
	 *        - #grb::eWiseApply using monoids (neutral),
	 *        - #grb::eWiseApply using operators (annihilating).
	 *
	 * \note However, #grb::eWiseAdd and #grb::eWiseMul provide in-place semantics,
	 *       while #grb::eWiseApply does not.
	 *
	 * \note An #grb::eWiseAdd with some semiring and a #grb::eWiseApply using its
	 *       additive monoid thus are equivalent if operating when operating on
	 *       empty outputs.
	 *
	 * \note An #grb::eWiseMul with some semiring and a #grb::eWiseApply using its
	 *       multiplicative operator thus are equivalent when operating on empty
	 *       outputs.
	 *
	 * @tparam descr      The descriptor to be used. Equal to
	 *                    descriptors::no_operation if left unspecified.
	 * @tparam Monoid     The monoid to use.
	 * @tparam InputType1 The value type of the left-hand vector.
	 * @tparam InputType2 The value type of the right-hand scalar.
	 * @tparam OutputType The value type of the ouput vector.
	 * @tparam MaskType   The value type of the mask vector.
	 *
	 * @param[out]  z    The output vector.
	 * @param[out] mask  The output mask.
	 * @param[in]   x    The left-hand input vector.
	 * @param[in]  beta  The right-hand input scalar.
	 * @param[in] monoid The monoid that provides the operator \f$ \odot \f$.
	 * @param[in] phase  The #grb::Phase the call should execute. Optional; the
	 *                   default parameter is #grb::EXECUTE.
	 *
	 * @return #grb::SUCCESS  On successful completion of this call.
	 * @return #grb::MISMATCH Whenever the dimensions of \a mask, \a x and \a z do
	 *                        not match. All input data containers are left
	 *                        untouched if this exit code is returned; it will be
	 *                        as though this call was never made.
	 * @return #grb::FAILED   If \a phase is #grb::EXECUTE, indicates that the
	 *                        capacity of \a z was insufficient. The output vector
	 *                        \a z is cleared, and the call to this function has no
	 *                        further effects.
	 * @return #grb::OUTOFMEM If \a phase is #grb::RESIZE, indicates an
	 *                        out-of-memory exception. The call to this function
	 *                        shall have no other effects beyond returning this
	 *                        error code; the previous state of \a z is retained.
	 * @return #grb::PANIC    A general unmitigable error has been encountered. If
	 *                        returned, ALP enters an undefined state and the user
	 *                        program is encouraged to exit as quickly as possible.
	 *
	 * \par Performance semantics
	 * Each backend must define performance semantics for this primitive.
	 *
	 * @see perfSemantics
	 */
	template<
		Descriptor descr = descriptors::no_operation,
		class Monoid, enum Backend backend,
		typename OutputType, typename MaskType,
		typename InputType1, typename InputType2,
		typename Coords
	>
	RC eWiseApply(
		Vector< OutputType, backend, Coords > &z,
		const Vector< MaskType, backend, Coords > &mask,
		const Vector< InputType1, backend, Coords > &x,
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
		std::cout << "In masked eWiseApply ([T1]<-[T2]<-T3, monoid, base)\n";
#endif
#ifndef NDEBUG
		const bool should_not_call_eWiseApplyMonoidAMAS_base = false;
		assert( should_not_call_eWiseApplyMonoidAMAS_base );
#endif
		(void) z;
		(void) mask;
		(void) x;
		(void) beta;
		(void) monoid;
		(void) phase;
		return UNSUPPORTED;
	}

	/**
	 * Computes \f$ z = x \odot y \f$, out of place, operator variant.
	 *
	 * Calculates the element-wise operation on one scalar to elements of one
	 * vector, \f$ z = x \odot y \f$, using the given operator. The input and
	 * output vectors must be of equal length.
	 *
	 * For all valid indices \a i of \a z, its element \f$ z_i \f$ after the call
	 * to this function completes equals \f$ x_i \odot y_i \f$. Any old entries
	 * of \a z are removed. Entries \a i which have no nonzero in either \a x or
	 * \a y are skipped.
	 *
	 * After a successful call to this primitive, the nonzero structure of \a z
	 * will match that of the intersection of \a x and \a y.
	 *
	 * \note When applying element-wise operators on sparse vectors using
	 *       semirings, there is a difference between interpreting missing values
	 *       as an annihilating identity or as a neutral identity-- intuitively,
	 *       such identities are known as `zero' or `one', respectively. As a
	 *       consequence, there are two different variants for element-wise
	 *       operations whose names correspond to their intuitive meanings:
	 *        - #grb::eWiseAdd (neutral), and
	 *        - #grb::eWiseMul (annihilating).
	 *       The above two primitives require a semiring. The same functionality is
	 *       provided by #grb::eWiseApply depending on whether a monoid or operator
	 *       is provided:
	 *        - #grb::eWiseApply using monoids (neutral),
	 *        - #grb::eWiseApply using operators (annihilating).
	 *
	 * \note However, #grb::eWiseAdd and #grb::eWiseMul provide in-place semantics,
	 *       while #grb::eWiseApply does not.
	 *
	 * \note An #grb::eWiseAdd with some semiring and a #grb::eWiseApply using its
	 *       additive monoid thus are equivalent if operating when operating on
	 *       empty outputs.
	 *
	 * \note An #grb::eWiseMul with some semiring and a #grb::eWiseApply using its
	 *       multiplicative operator thus are equivalent when operating on empty
	 *       outputs.
	 *
	 * @tparam descr      The descriptor to be used. Optional; the default is
	 *                    #grb::descriptors::no_operation.
	 * @tparam OP         The operator to use.
	 * @tparam InputType1 The value type of the left-hand vector.
	 * @tparam InputType2 The value type of the right-hand scalar.
	 * @tparam OutputType The value type of the ouput vector.
	 *
	 * @param[out]  z    The output vector.
	 * @param[in]   x    The left-hand input vector.
	 * @param[in]   y    The right-hand input vector.
	 * @param[in]  op    The operator \f$ \odot \f$.
	 * @param[in] phase  The #grb::Phase the call should execute. Optional; the
	 *                   default parameter is #grb::EXECUTE.
	 *
	 * @return #grb::SUCCESS  On successful completion of this call.
	 * @return #grb::MISMATCH Whenever the dimensions of \a x, \a y and \a z do not
	 *                        match. All input data containers are left untouched
	 *                        if this exit code is returned; it will be as though
	 *                        this call was never made.
	 * @return #grb::FAILED   If \a phase is #grb::EXECUTE, indicates that the
	 *                        capacity of \a z was insufficient. The output vector
	 *                        \a z is cleared, and the call to this function has no
	 *                        further effects.
	 * @return #grb::OUTOFMEM If \a phase is #grb::RESIZE, indicates an
	 *                        out-of-memory exception. The call to this function
	 *                        shall have no other effects beyond returning this
	 *                        error code; the previous state of \a z is retained.
	 * @return #grb::PANIC    A general unmitigable error has been encountered. If
	 *                        returned, ALP enters an undefined state and the user
	 *                        program is encouraged to exit as quickly as possible.
	 *
	 * \par Performance semantics
	 * Each backend must define performance semantics for this primitive.
	 *
	 * @see perfSemantics
	 */
	template<
		Descriptor descr = descriptors::no_operation,
		class OP, enum Backend backend,
		typename OutputType, typename InputType1, typename InputType2,
		typename Coords
	>
	RC eWiseApply(
		Vector< OutputType, backend, Coords > &z,
		const Vector< InputType1, backend, Coords > &x,
		const Vector< InputType2, backend, Coords > &y,
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
#ifndef NDEBUG
		const bool should_not_call_eWiseApplyOpAAA_base = false;
		assert( should_not_call_eWiseApplyOpAAA_base );
#endif
		(void) z;
		(void) x;
		(void) y;
		(void) op;
		(void) phase;
		return UNSUPPORTED;
	}

	/**
	 * Computes \f$ z = x \odot y \f$, out of place, masked operator variant.
	 *
	 * Calculates the element-wise operation on one scalar to elements of one
	 * vector, \f$ z = x \odot y \f$, using the given operator. The input and
	 * output vectors must be of equal length.
	 *
	 * For all valid indices \a i of \a z, its element \f$ z_i \f$ after the call
	 * to this function completes equals \f$ x_i \odot y_i \f$. Any old entries
	 * of \a z are removed. Entries \a i which have no nonzero in either \a x or
	 * \a y are skipped, as will entries \a i for which \a mask evaluates
	 * <tt>false</tt>.
	 *
	 * After a successful call to this primitive, the nonzero structure of \a z
	 * will match that of the intersection of \a x and \a y.
	 *
	 * \note When applying element-wise operators on sparse vectors using
	 *       semirings, there is a difference between interpreting missing values
	 *       as an annihilating identity or as a neutral identity-- intuitively,
	 *       such identities are known as `zero' or `one', respectively. As a
	 *       consequence, there are two different variants for element-wise
	 *       operations whose names correspond to their intuitive meanings:
	 *        - #grb::eWiseAdd (neutral), and
	 *        - #grb::eWiseMul (annihilating).
	 *       The above two primitives require a semiring. The same functionality is
	 *       provided by #grb::eWiseApply depending on whether a monoid or operator
	 *       is provided:
	 *        - #grb::eWiseApply using monoids (neutral),
	 *        - #grb::eWiseApply using operators (annihilating).
	 *
	 * \note However, #grb::eWiseAdd and #grb::eWiseMul provide in-place semantics,
	 *       while #grb::eWiseApply does not.
	 *
	 * \note An #grb::eWiseAdd with some semiring and a #grb::eWiseApply using its
	 *       additive monoid thus are equivalent if operating when operating on
	 *       empty outputs.
	 *
	 * \note An #grb::eWiseMul with some semiring and a #grb::eWiseApply using its
	 *       multiplicative operator thus are equivalent when operating on empty
	 *       outputs.
	 *
	 * @tparam descr      The descriptor to be used. Optional; the default is
	 *                    #grb::descriptors::no_operation.
	 * @tparam OP         The operator to use.
	 * @tparam InputType1 The value type of the left-hand vector.
	 * @tparam InputType2 The value type of the right-hand scalar.
	 * @tparam OutputType The value type of the ouput vector.
	 * @tparam MaskType   The value type of the output mask vector.
	 *
	 * @param[out]  z    The output vector.
	 * @param[in]  mask  The output mask.
	 * @param[in]   x    The left-hand input vector.
	 * @param[in]   y    The right-hand input vector.
	 * @param[in]  op    The operator \f$ \odot \f$.
	 * @param[in] phase  The #grb::Phase the call should execute. Optional; the
	 *                   default parameter is #grb::EXECUTE.
	 *
	 * @return #grb::SUCCESS  On successful completion of this call.
	 * @return #grb::MISMATCH Whenever the dimensions of \a mask, \a x, \a y, and
	 *                        \a z do not match. All input data containers are left
	 *                        untouched if this exit code is returned; it will be
	 *                        as though this call was never made.
	 * @return #grb::FAILED   If \a phase is #grb::EXECUTE, indicates that the
	 *                        capacity of \a z was insufficient. The output vector
	 *                        \a z is cleared, and the call to this function has no
	 *                        further effects.
	 * @return #grb::OUTOFMEM If \a phase is #grb::RESIZE, indicates an
	 *                        out-of-memory exception. The call to this function
	 *                        shall have no other effects beyond returning this
	 *                        error code; the previous state of \a z is retained.
	 * @return #grb::PANIC    A general unmitigable error has been encountered. If
	 *                        returned, ALP enters an undefined state and the user
	 *                        program is encouraged to exit as quickly as possible.
	 *
	 * \par Performance semantics
	 * Each backend must define performance semantics for this primitive.
	 *
	 * @see perfSemantics
	 */
	template<
		Descriptor descr = descriptors::no_operation,
		class OP, enum Backend backend,
		typename OutputType, typename MaskType,
		typename InputType1, typename InputType2,
		typename Coords
	>
	RC eWiseApply(
		Vector< OutputType, backend, Coords > &z,
		const Vector< MaskType, backend, Coords > &mask,
		const Vector< InputType1, backend, Coords > &x,
		const Vector< InputType2, backend, Coords > &y,
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
		std::cout << "In masked eWiseApply ([T1]<-[T2]<-[T3], operator, base)\n";
#endif
#ifndef NDEBUG
		const bool should_not_call_eWiseApplyOpAMAA_base = false;
		assert( should_not_call_eWiseApplyOpAMAA_base );
#endif
		(void) z;
		(void) mask;
		(void) x;
		(void) y;
		(void) op;
		(void) phase;
		return UNSUPPORTED;
	}

	/**
	 * Computes \f$ z = x \odot y \f$, out of place, monoid variant.
	 *
	 * Calculates the element-wise operation on one scalar to elements of one
	 * vector, \f$ z = x \odot y \f$, using the given operator. The input and
	 * output vectors must be of equal length.
	 *
	 * For all valid indices \a i of \a z, its element \f$ z_i \f$ after the call
	 * to this function completes equals \f$ x_i \odot y_i \f$. Any old entries
	 * of \a z are removed.
	 *
	 * After a successful call to this primitive, the nonzero structure of \a z
	 * will match that of the union of \a x and \a y. An implementing backend may
	 * skip processing indices \a i that are not in the union of the nonzero
	 * structure of \a x and \a y.
	 *
	 * \note When applying element-wise operators on sparse vectors using
	 *       semirings, there is a difference between interpreting missing values
	 *       as an annihilating identity or as a neutral identity-- intuitively,
	 *       such identities are known as `zero' or `one', respectively. As a
	 *       consequence, there are two different variants for element-wise
	 *       operations whose names correspond to their intuitive meanings:
	 *        - #grb::eWiseAdd (neutral), and
	 *        - #grb::eWiseMul (annihilating).
	 *       The above two primitives require a semiring. The same functionality is
	 *       provided by #grb::eWiseApply depending on whether a monoid or operator
	 *       is provided:
	 *        - #grb::eWiseApply using monoids (neutral),
	 *        - #grb::eWiseApply using operators (annihilating).
	 *
	 * \note However, #grb::eWiseAdd and #grb::eWiseMul provide in-place semantics,
	 *       while #grb::eWiseApply does not.
	 *
	 * \note An #grb::eWiseAdd with some semiring and a #grb::eWiseApply using its
	 *       additive monoid thus are equivalent if operating when operating on
	 *       empty outputs.
	 *
	 * \note An #grb::eWiseMul with some semiring and a #grb::eWiseApply using its
	 *       multiplicative operator thus are equivalent when operating on empty
	 *       outputs.
	 *
	 * @tparam descr      The descriptor to be used. Optional; the default is
	 *                    #grb::descriptors::no_operation.
	 * @tparam Monoid     The monoid to use.
	 * @tparam InputType1 The value type of the left-hand vector.
	 * @tparam InputType2 The value type of the right-hand scalar.
	 * @tparam OutputType The value type of the ouput vector.
	 *
	 * @param[out]  z    The output vector.
	 * @param[in]   x    The left-hand input vector.
	 * @param[in]   y    The right-hand input vector.
	 * @param[in] monoid The monoid structure that \f$ \odot \f$ corresponds to.
	 * @param[in] phase  The #grb::Phase the call should execute. Optional; the
	 *                   default parameter is #grb::EXECUTE.
	 *
	 * @return #grb::SUCCESS  On successful completion of this call.
	 * @return #grb::MISMATCH Whenever the dimensions of \a x, \a y and \a z do not
	 *                        match. All input data containers are left untouched
	 *                        if this exit code is returned; it will be as though
	 *                        this call was never made.
	 * @return #grb::FAILED   If \a phase is #grb::EXECUTE, indicates that the
	 *                        capacity of \a z was insufficient. The output vector
	 *                        \a z is cleared, and the call to this function has no
	 *                        further effects.
	 * @return #grb::OUTOFMEM If \a phase is #grb::RESIZE, indicates an
	 *                        out-of-memory exception. The call to this function
	 *                        shall have no other effects beyond returning this
	 *                        error code; the previous state of \a z is retained.
	 * @return #grb::PANIC    A general unmitigable error has been encountered. If
	 *                        returned, ALP enters an undefined state and the user
	 *                        program is encouraged to exit as quickly as possible.
	 *
	 * \par Performance semantics
	 * Each backend must define performance semantics for this primitive.
	 *
	 * @see perfSemantics
	 */
	template<
		Descriptor descr = descriptors::no_operation,
		class Monoid, enum Backend backend,
		typename OutputType, typename InputType1, typename InputType2,
		typename Coords
	>
	RC eWiseApply(
		Vector< OutputType, backend, Coords > &z,
		const Vector< InputType1, backend, Coords > &x,
		const Vector< InputType2, backend, Coords > &y,
		const Monoid &monoid = Monoid(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if< !grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			grb::is_monoid< Monoid >::value, void
		>::type * const = nullptr
	) {
#ifdef _DEBUG
		std::cout << "In unmasked eWiseApply ([T1]<-[T2]<-[T3], monoid, base)\n";
#endif
#ifndef NDEBUG
		const bool should_not_call_eWiseApplyOpAMAA_base = false;
		assert( should_not_call_eWiseApplyOpAMAA_base );
#endif
		(void) z;
		(void) x;
		(void) y;
		(void) monoid;
		(void) phase;
		return UNSUPPORTED;
	}

	/**
	 * Computes \f$ z = x \odot y \f$, out of place, masked monoid variant.
	 *
	 * Calculates the element-wise operation on one scalar to elements of one
	 * vector, \f$ z = x \odot y \f$, using the given operator. The input and
	 * output vectors must be of equal length.
	 *
	 * For all valid indices \a i of \a z, its element \f$ z_i \f$ after the call
	 * to this function completes equals \f$ x_i \odot y_i \f$. Any old entries
	 * of \a z are removed. Entries \a i for which \a mask evaluates <tt>false</tt>
	 * will be skipped.
	 *
	 * \note When applying element-wise operators on sparse vectors using
	 *       semirings, there is a difference between interpreting missing values
	 *       as an annihilating identity or as a neutral identity-- intuitively,
	 *       such identities are known as `zero' or `one', respectively. As a
	 *       consequence, there are two different variants for element-wise
	 *       operations whose names correspond to their intuitive meanings:
	 *        - #grb::eWiseAdd (neutral), and
	 *        - #grb::eWiseMul (annihilating).
	 *       The above two primitives require a semiring. The same functionality is
	 *       provided by #grb::eWiseApply depending on whether a monoid or operator
	 *       is provided:
	 *        - #grb::eWiseApply using monoids (neutral),
	 *        - #grb::eWiseApply using operators (annihilating).
	 *
	 * \note However, #grb::eWiseAdd and #grb::eWiseMul provide in-place semantics,
	 *       while #grb::eWiseApply does not.
	 *
	 * \note An #grb::eWiseAdd with some semiring and a #grb::eWiseApply using its
	 *       additive monoid thus are equivalent if operating when operating on
	 *       empty outputs.
	 *
	 * \note An #grb::eWiseMul with some semiring and a #grb::eWiseApply using its
	 *       multiplicative operator thus are equivalent when operating on empty
	 *       outputs.
	 *
	 * @tparam descr      The descriptor to be used. Optional; the default is
	 *                    #grb::descriptors::no_operation.
	 * @tparam Monoid     The monoid to use.
	 * @tparam InputType1 The value type of the left-hand vector.
	 * @tparam InputType2 The value type of the right-hand scalar.
	 * @tparam OutputType The value type of the ouput vector.
	 * @tparam MaskType   The value type of the mask vector.
	 *
	 * @param[out]  z    The output vector.
	 * @param[in]  mask  The output mask.
	 * @param[in]   x    The left-hand input vector.
	 * @param[in]   y    The right-hand input vector.
	 * @param[in] monoid The monoid structure that \f$ \odot \f$ corresponds to.
	 * @param[in] phase  The #grb::Phase the call should execute. Optional; the
	 *                   default parameter is #grb::EXECUTE.
	 *
	 * @return #grb::SUCCESS  On successful completion of this call.
	 * @return #grb::MISMATCH Whenever the dimensions of \a x, \a y and \a z do not
	 *                        match. All input data containers are left untouched
	 *                        if this exit code is returned; it will be as though
	 *                        this call was never made.
	 * @return #grb::FAILED   If \a phase is #grb::EXECUTE, indicates that the
	 *                        capacity of \a z was insufficient. The output vector
	 *                        \a z is cleared, and the call to this function has no
	 *                        further effects.
	 * @return #grb::OUTOFMEM If \a phase is #grb::RESIZE, indicates an
	 *                        out-of-memory exception. The call to this function
	 *                        shall have no other effects beyond returning this
	 *                        error code; the previous state of \a z is retained.
	 * @return #grb::PANIC    A general unmitigable error has been encountered. If
	 *                        returned, ALP enters an undefined state and the user
	 *                        program is encouraged to exit as quickly as possible.
	 *
	 * \par Performance semantics
	 * Each backend must define performance semantics for this primitive.
	 *
	 * @see perfSemantics
	 */
	template<
		Descriptor descr = descriptors::no_operation,
		class Monoid, enum Backend backend,
		typename OutputType, typename MaskType,
		typename InputType1, typename InputType2,
		typename Coords
	>
	RC eWiseApply(
		Vector< OutputType, backend, Coords > &z,
		const Vector< MaskType, backend, Coords > &mask,
		const Vector< InputType1, backend, Coords > &x,
		const Vector< InputType2, backend, Coords > &y,
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
		std::cout << "In masked eWiseApply ([T1]<-[T2]<-[T3], monoid, base)\n";
#endif
#ifndef NDEBUG
		const bool should_not_call_eWiseApplyMonoidAMAA_base = false;
		assert( should_not_call_eWiseApplyMonoidAMAA_base );
#endif
		(void) z;
		(void) mask;
		(void) x;
		(void) y;
		(void) monoid;
		(void) phase;
		return UNSUPPORTED;
	}

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
	 * As a rule of thumb, data-centric ALP/GraphBLAS implementations \em cannot
	 * support this and will thus have grb::Properties::writableCaptured evaluate
	 * to false. A portable ALP/GraphBLAS algorithm should provide a different code
	 * path to handle this case (or not rely on #grb::eWiseLambda).
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
	 * \note This is an addition to the GraphBLAS C specification. It is alike
	 *       user-defined operators, monoids, and semirings, except it allows
	 *       execution on arbitrarily many inputs and arbitrarily many outputs.
	 *       It is intended for programmers to take control over what is fused
	 *       when and how. The #grb::nonblocking backend attempts to automate the
	 *       application of such fusion opportunities without the user's explicit
	 *       involvement.
	 *
	 * @tparam Func the user-defined lambda function type.
	 * @tparam DataType the type of the user-supplied vector example.
	 * @tparam backend  the backend type of the user-supplied vector example.
	 *
	 * @param[in] f The user-supplied lambda. This lambda should only capture
	 *              and reference vectors of the same length as \a x. The lambda
	 *              function should prescribe the operations required to execute
	 *              at a given index \a i. Captured ALP/GraphBLAS vectors can
	 *              access that element via the operator[]. It is illegal to access
	 *              any element not at position \a i. The lambda takes only the
	 *              single parameter \a i of type <code>const size_t</code>.
	 *              Captured scalars will not be globally updated-- the user must
	 *              program this explicitly. Scalars and other non-GraphBLAS
	 *              containers are always local to their user process.
	 * @param[in] x The vector the lambda will be executed on. This argument
	 *              determines which indices \a i will be accessed during the
	 *              elementwise operation-- elements with indices \a i that
	 *              do not appear in \a x will be skipped during evaluation of
	 *              \a f.
	 *
	 * The remaining arguments must collect all vectors the lambda is to access
	 * elements of. Such vectors must be of the same length as \a x. If this
	 * constraint is violated, #grb::MISMATCH shall be returned.
	 *
	 * \note These are passed using variadic arguments and so can contain any
	 *       number of containers of type #grb::Vector.
	 *
	 * \note In future ALP/GraphBLAS implementations, apart from performing
	 *       dimension checking, may also require data redistribution in case
	 *       different vectors may be distributed differently.
	 *
	 * \warning Using a #grb::Vector inside a lambda passed to this function while
	 *          not passing that same vector into its variadic argument list, will
	 *          result in undefined behaviour.
	 *
	 * \warning Due to the constraints on \a f described above, it is illegal to
	 *          capture some vector \a y and have the following line in the body
	 *          of \a f: <code>x[i] += x[i+1]</code>. Vectors can only be
	 *          dereferenced at position \a i and \a i alone.
	 *
	 * @return #grb::SUCCESS  When the lambda is successfully executed.
	 * @return #grb::MISMATCH When two or more vectors passed to \a args are not of
	 *                        equal length.
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
		(void) f;
		(void) x;
		return UNSUPPORTED;
	}

	/**
	 * Reduces, or \em folds, a vector into a scalar.
	 *
	 * Reduction takes place according a monoid \f$ (\oplus,1) \f$, where
	 * \f$ \oplus:\ D_1 \times D_2 \to D_3 \f$ with associated identities
	 * \f$ 1_k in D_k \f$. Usually, \f$ D_k \subseteq D_3, 1 \leq k < 3 \f$,
	 * though other more exotic structures may be envisioned (and used).
	 *
	 * Let \f$ x_0 = 1 \f$ and let
	 * \f$ x_{i+1} = \begin{cases}
	 *   x_i \oplus y_i\text{ if }y_i\text{ is nonzero and }m_i\text{ evaluates true}
	 *   x_i\text{ otherwise}
	 * \end{cases},\f$
	 * for all \f$ i \in \{ 0, 1, \ldots, n-1 \} \f$.
	 *
	 * \note Per this definition, the folding happens in a left-to-right direction.
	 *       If another direction is wanted, which may have use in cases where
	 *       \f$ D_1 \f$ differs from \f$ D_2 \f$, then either a monoid with those
	 *       operator domains switched may be supplied, or #grb::foldr may be used
	 *       instead.
	 *
	 * After a successfull call, \a x will be equal to \f$ x_n \f$.
	 *
	 * Note that the operator \f$ \oplus \f$ must be associative since it is part
	 * of a monoid. This algebraic property is exploited when parallelising the
	 * requested operation. The identity is required when parallelising over
	 * multiple user processes.
	 *
	 * \warning In so doing, the order of the evaluation of the reduction operation
	 *          should not be expected to be a serial, left-to-right, evaluation of
	 *          the computation chain.
	 *
	 * @tparam descr     The descriptor to be used (descriptors::no_operation if
	 *                   left unspecified).
	 * @tparam Monoid    The monoid to use for reduction.
	 * @tparam InputType The type of the elements in the supplied ALP/GraphBLAS
	 *                   vector \a y.
	 * @tparam IOType    The type of the output scalar \a x.
	 * @tparam MaskType  The type of the elements in the supplied ALP/GraphBLAS
	 *                   vector \a mask.
	 *
	 * @param[out]   x   The result of the reduction.
	 * @param[in]    y   Any ALP/GraphBLAS vector. This vector may be sparse.
	 * @param[in]  mask  Any ALP/GraphBLAS vector. This vector may be sparse.
	 * @param[in] monoid The monoid under which to perform this reduction.
	 *
	 * @return grb::SUCCESS  When the call completed successfully.
	 * @return grb::MISMATCH If a \a mask was not empty and does not have size
	 *                       equal to \a y.
	 * @return grb::ILLEGAL  If the provided input vector \a y was not dense, while
	 *                       #grb::descriptors::dense was given.
	 *
	 * \parblock
	 * \par Valid descriptors
	 * grb::descriptors::no_operation, grb::descriptors::no_casting,
	 * grb::descriptors::dense, grb::descriptors::invert_mask,
	 * grb::descriptors::structural, grb::descriptors::structural_complement
	 *
	 * \note Invalid descriptors will be ignored.
	 *
	 * If grb::descriptors::no_casting is given, then 1) the first domain of
	 * \a monoid must match \a InputType, 2) the second domain of \a op must match
	 * \a IOType, 3) the third domain must match \a IOType, and 4) the element type
	 * of \a mask must be <tt>bool</tt>. If one of these is not true, the code
	 * shall not compile.
	 * \endparblock
	 *
	 * \parblock
	 * \par Performance semantics
	 * Backends must specify performance semantics in the amount of work, intra-
	 * process data movement, inter-process data movement, and the number of
	 * user process synchronisations required. They should also specify whether
	 * any system calls may be made, in particularly those related to dynamic
	 * memory management. If new memory may be allocated, they must specify how
	 * much.
	 * \endparblock
	 *
	 * @see grb::foldr provides similar in-place functionality.
	 * @see grb::eWiseApply provides out-of-place semantics.
	 */
	template<
		Descriptor descr = descriptors::no_operation,
		class Monoid,
		typename InputType, typename IOType, typename MaskType,
		Backend backend, typename Coords
	>
	RC foldl(
		IOType &x,
		const Vector< InputType, backend, Coords > &y,
		const Vector< MaskType, backend, Coords > &mask,
		const Monoid &monoid = Monoid(),
		const typename std::enable_if< !grb::is_object< IOType >::value &&
			!grb::is_object< InputType >::value &&
			!grb::is_object< MaskType >::value &&
			grb::is_monoid< Monoid >::value, void
		>::type * const = nullptr
	) {
#ifndef NDEBUG
		const bool should_not_call_base_scalar_foldl = false;
		assert( should_not_call_base_scalar_foldl );
#endif
		(void) y;
		(void) x;
		(void) mask;
		(void) monoid;
		return UNSUPPORTED;
	}

	/**
	 * Folds a vector into a scalar, left-to-right.
	 *
	 * Unmasked monoid variant. See masked variant for the full documentation.
	 */
	template<
		Descriptor descr = descriptors::no_operation,
		class Monoid,
		typename IOType, typename InputType,
		Backend backend,
		typename Coords
	>
	RC foldl(
		IOType &x,
		const Vector< InputType, backend, Coords > &y,
		const Monoid &monoid = Monoid(),
		const typename std::enable_if<
			!grb::is_object< IOType >::value &&
			grb::is_monoid< Monoid >::value,
		void >::type * const = nullptr
	) {
#ifndef NDEBUG
		const bool should_not_call_base_scalar_foldl_nomask = false;
		assert( should_not_call_base_scalar_foldl_nomask );
#endif
		(void) y;
		(void) x;
		(void) monoid;
		return UNSUPPORTED;
	}

	/**
	 * Folds a vector into a scalar, left-to-right.
	 *
	 * Unmasked operator variant.
	 *
	 * \deprecated This signature is deprecated. It was implemented for reference
	 *             (and reference_omp), but could not be implemented for BSP1D and
	 *             other distributed-memory backends. This signature may be removed
	 *             with any release beyond 0.6.
	 */
	template<
		Descriptor descr = descriptors::no_operation,
		class OP,
		typename IOType, typename InputType, typename MaskType,
		Backend backend, typename Coords
	>
	RC foldl(
		IOType &x,
		const Vector< InputType, backend, Coords > &y,
		const Vector< MaskType, backend, Coords > &mask,
		const OP &op = OP(),
		const typename std::enable_if<
			!grb::is_object< IOType >::value &&
			!grb::is_object< MaskType >::value &&
			grb::is_operator< OP >::value,
		void >::type * const = nullptr
	) {
#ifndef NDEBUG
		const bool should_not_call_base_scalar_foldl_op = false;
		assert( should_not_call_base_scalar_foldl_op );
#endif
		(void) x;
		(void) y;
		(void) mask;
		(void) op;
		return UNSUPPORTED;
	}

	/**
	 * Folds a vector into a scalar, right-to-left.
	 *
	 * Masked variant. See the masked, left-to-right variant for the full
	 * documentation.
	 */
	template<
		Descriptor descr = descriptors::no_operation,
		class Monoid,
		typename InputType, typename IOType, typename MaskType,
		Backend backend, typename Coords
	>
	RC foldr(
		const Vector< InputType, backend, Coords > &x,
		const Vector< MaskType, backend, Coords > &mask,
		IOType &y,
		const Monoid &monoid = Monoid(),
		const typename std::enable_if< !grb::is_object< IOType >::value &&
			!grb::is_object< InputType >::value &&
			!grb::is_object< MaskType >::value &&
			grb::is_monoid< Monoid >::value, void
		>::type * const = nullptr
	) {
#ifndef NDEBUG
		const bool should_not_call_base_scalar_foldr = false;
		assert( should_not_call_base_scalar_foldr );
#endif
		(void) y;
		(void) x;
		(void) mask;
		(void) monoid;
		return UNSUPPORTED;
	}

	/**
	 * Folds a vector into a scalar, right-to-left.
	 *
	 * Unmasked variant. See the masked, left-to-right variant for the full
	 * documentation.
	 */
	template<
		Descriptor descr = descriptors::no_operation,
		class Monoid,
		typename IOType, typename InputType,
		Backend backend, typename Coords
	>
	RC foldr(
		const Vector< InputType, backend, Coords > &y,
		IOType &x,
		const Monoid &monoid = Monoid(),
		const typename std::enable_if<
			!grb::is_object< IOType >::value &&
			grb::is_monoid< Monoid >::value,
		void >::type * const = nullptr
	) {
#ifndef NDEBUG
		const bool should_not_call_base_scalar_foldr_nomask = false;
		assert( should_not_call_base_scalar_foldr_nomask );
#endif
		(void) y;
		(void) x;
		(void) monoid;
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
#ifndef NDEBUG
		const bool should_not_call_base_dot_semiring = false;
		assert( should_not_call_base_dot_semiring );
#endif
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

