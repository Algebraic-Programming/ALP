
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
	 * Computes \f$ z = \alpha \odot \beta \f$, out of place, operator and masked
	 * version.
	 *
	 * @tparam descr      The descriptor to be used. Equal to
	 *                    descriptors::no_operation if left unspecified.
	 * @tparam OP         The operator to use.
	 * @tparam InputType1 The value type of the left-hand vector.
	 * @tparam InputType2 The value type of the right-hand scalar.
	 * @tparam OutputType The value type of the ouput vector.
	 * @tparam MaskType   The value type of the output mask vector.
	 *
	 * @param[out]  z   The output vector.
	 * @param[in]  mask The ouptut mask.
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
		typename OutputType, typename MaskType,
		typename InputType1, typename InputType2,
		typename Coords
	>
	RC eWiseApply(
		Vector< OutputType, backend, Coords > &z,
		const Vector< MaskType, backend, Coords > &mask,
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
		std::cout << "In masked eWiseApply ([T1]<-T2<-T3), operator, base\n";
#endif
#ifndef NDEBUG
		const bool should_not_call_eWiseApplyOpAMSS_base = false;
		assert( should_not_call_eWiseApplyOpAMSS_base );
#endif
		(void) z;
		(void) mask;
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
	 * Computes \f$ z = \alpha \odot \beta \f$, out of place, masked monoid
	 * version.
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
	 * @param[in]  mask  The output mask.
	 * @param[in]  alpha The left-hand input scalar.
	 * @param[in]  beta  The right-hand input scalar.
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
		typename OutputType, typename MaskType,
		typename InputType1, typename InputType2,
		typename Coords
	>
	RC eWiseApply(
		Vector< OutputType, backend, Coords > &z,
		const Vector< MaskType, backend, Coords > &mask,
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
		std::cout << "In masked eWiseApply ([T1]<-T2<-T3), monoid, base\n";
#endif
#ifndef NDEBUG
		const bool should_not_call_eWiseApplyMonAMSS_base = false;
		assert( should_not_call_eWiseApplyMonAMSS_base );
#endif
		(void) z;
		(void) mask;
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
	 * Calculates the element-wise addition of two vectors, \f$ z += x .+ y \f$,
	 * under a given semiring.
	 *
	 * \note This is an in-place operation.
	 *
	 * \deprecated This function has been deprecated since v0.5. It may be removed
	 *             at latest at v1.0 of ALP/GraphBLAS-- or any time earlier.
	 *
	 * \note A call to this function is equivalent to two in-place fold operations
	 *       using the additive monoid of the given semiring. Please update any
	 *       code that calls #grb::eWiseAdd with such a sequence as soon as
	 *       possible.
	 *
	 * \note We may consider providing this function as an algorithm in the
	 *       #grb::algorithms namespace, similar to #grb::algorithms::mpv. Please
	 *       let the maintainers know if you would prefer such a solution over
	 *       outright removal and replacement with two folds.
	 *
	 * @tparam descr      The descriptor to be used. Optional; default is
	 *                    #grb::descriptors::no_operation.
	 * @tparam Ring       The semiring type to perform the element-wise addition
	 *                    on.
	 * @tparam InputType1 The left-hand side input type to the additive operator
	 *                    of the \a ring.
	 * @tparam InputType2 The right-hand side input type to the additive operator
	 *                    of the \a ring.
	 * @tparam OutputType The result type of the additive operator of the
	 *                    \a ring.
	 *
	 * @param[out]  z   The output vector of type \a OutputType. This may be a
	 *                  sparse vector.
	 * @param[in]   x   The left-hand input vector of type \a InputType1. This may
	 *                  be a sparse vector.
	 * @param[in]   y   The right-hand input vector of type \a InputType2. This may
	 *                  be a sparse vector.
	 * @param[in] ring  The generalized semiring under which to perform this
	 *                  element-wise multiplication.
	 * @param[in] phase The #grb::Phase the call should execute. Optional; the
	 *                  default parameter is #grb::EXECUTE.
	 *
	 * \note There is also a masked variant of #grb::eWiseAdd, as well as variants
	 *       where \a x and/or \a y are scalars.
	 *
	 * @return #grb::SUCCESS  On successful completion of this call.
	 * @return #grb::MISMATCH Whenever the dimensions of \a x, \a y, and \a z do
	 *                        not match. All input data containers are left
	 *                        untouched; it will be as though this call was never
	 *                        made.
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
	 * \parblock
	 * \par Valid descriptors
	 * grb::descriptors::no_operation, grb::descriptors::no_casting,
	 * grb::descriptors::dense.
	 *
	 * \note Invalid descriptors will be ignored.
	 *
	 * If #grb::descriptors::no_casting is specified, then 1) the third domain of
	 * \a ring must match \a InputType1, 2) the fourth domain of \a ring must match
	 * \a InputType2, 3) the fourth domain of \a ring must match \a OutputType. If
	 * one of these is not true, the code shall not compile.
	 * \endparblock
	 *
	 * \par Performance semantics
	 * Each backend must define performance semantics for this primitive.
	 *
	 * @see perfSemantics
	 */
	template<
		Descriptor descr = descriptors::no_operation,
		class Ring, enum Backend backend,
		typename OutputType, typename InputType1, typename InputType2,
		typename Coords
	>
	RC eWiseAdd(
		Vector< OutputType, backend, Coords > &z,
		const Vector< InputType1, backend, Coords > &x,
		const Vector< InputType2, backend, Coords > &y,
		const Ring &ring = Ring(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if< !grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			grb::is_semiring< Ring >::value, void
		>::type * const = nullptr
	) {
#ifdef _DEBUG
		std::cout << "in eWiseAdd ([T1] <- [T2] + [T3]), unmasked, base";
#endif
#ifndef NDEBUG
		const bool should_not_call_eWiseAddAAA_base = false;
		assert( should_not_call_eWiseAddAAA_base );
#endif
		(void) z;
		(void) x;
		(void) y;
		(void) ring;
		(void) phase;
		return UNSUPPORTED;
	}

	/**
	 * Calculates the element-wise addition, \f$ z += \alpha .+ y \f$, under a
	 * given semiring.
	 *
	 * \note This is an in-place operation.
	 *
	 * \deprecated This function has been deprecated since v0.5. It may be removed
	 *             at latest at v1.0 of ALP/GraphBLAS-- or any time earlier.
	 *
	 * \note A call to this function is equivalent to two in-place fold operations
	 *       using the additive monoid of the given semiring. Please update any
	 *       code that calls #grb::eWiseAdd with such a sequence as soon as
	 *       possible.
	 *
	 * \note We may consider providing this function as an algorithm in the
	 *       #grb::algorithms namespace, similar to #grb::algorithms::mpv. Please
	 *       let the maintainers know if you would prefer such a solution over
	 *       outright removal and replacement with two folds.
	 *
	 * @tparam descr      The descriptor to be used. Optional; default is
	 *                    #grb::descriptors::no_operation.
	 * @tparam Ring       The semiring type to perform the element-wise addition
	 *                    on.
	 * @tparam InputType1 The left-hand side input type to the additive operator
	 *                    of the \a ring.
	 * @tparam InputType2 The right-hand side input type to the additive operator
	 *                    of the \a ring.
	 * @tparam OutputType The result type of the additive operator of the
	 *                    \a ring.
	 *
	 * @param[out]  z   The output vector of type \a OutputType. This may be a
	 *                  sparse vector.
	 * @param[in] alpha The left-hand input scalar of type \a InputType1.
	 * @param[in]   y   The right-hand input vector of type \a InputType2. This may
	 *                  be a sparse vector.
	 * @param[in] ring  The generalized semiring under which to perform this
	 *                  element-wise multiplication.
	 * @param[in] phase The #grb::Phase the call should execute. Optional; the
	 *                  default parameter is #grb::EXECUTE.
	 *
	 * @return #grb::SUCCESS  On successful completion of this call.
	 * @return #grb::MISMATCH Whenever the dimensions of \a y and \a z do not
	 *                        match. All input data containers are left untouched;
	 *                        it will be as though this call was never made.
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
	 * \parblock
	 * \par Valid descriptors
	 * grb::descriptors::no_operation, grb::descriptors::no_casting,
	 * grb::descriptors::dense.
	 *
	 * \note Invalid descriptors will be ignored.
	 *
	 * If #grb::descriptors::no_casting is specified, then 1) the third domain of
	 * \a ring must match \a InputType1, 2) the fourth domain of \a ring must match
	 * \a InputType2, 3) the fourth domain of \a ring must match \a OutputType. If
	 * one of these is not true, the code shall not compile.
	 * \endparblock
	 *
	 * \par Performance semantics
	 * Each backend must define performance semantics for this primitive.
	 *
	 * @see perfSemantics
	 */
	template<
		Descriptor descr = descriptors::no_operation,
		class Ring, enum Backend backend,
		typename InputType1, typename InputType2, typename OutputType,
		typename Coords
	>
	RC eWiseAdd(
		Vector< OutputType, backend, Coords > &z,
		const InputType1 alpha,
		const Vector< InputType2, backend, Coords > &y,
		const Ring &ring = Ring(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if< !grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			grb::is_semiring< Ring >::value, void
		>::type * const = nullptr
	) {
#ifdef _DEBUG
		std::cout << "in eWiseAdd ([T1] <- T2 + [T3]), unmasked, base";
#endif
#ifndef NDEBUG
		const bool should_not_call_eWiseAddASA_base = false;
		assert( should_not_call_eWiseAddASA_base );
#endif
		(void) z;
		(void) alpha;
		(void) y;
		(void) ring;
		(void) phase;
		return UNSUPPORTED;
	}

	/**
	 * Calculates the element-wise addition, \f$ z += x .+ \beta \f$, under a
	 * given semiring.
	 *
	 * \note This is an in-place operation.
	 *
	 * \deprecated This function has been deprecated since v0.5. It may be removed
	 *             at latest at v1.0 of ALP/GraphBLAS-- or any time earlier.
	 *
	 * \note A call to this function is equivalent to two in-place fold operations
	 *       using the additive monoid of the given semiring. Please update any
	 *       code that calls #grb::eWiseAdd with such a sequence as soon as
	 *       possible.
	 *
	 * \note We may consider providing this function as an algorithm in the
	 *       #grb::algorithms namespace, similar to #grb::algorithms::mpv. Please
	 *       let the maintainers know if you would prefer such a solution over
	 *       outright removal and replacement with two folds.
	 *
	 * @tparam descr      The descriptor to be used. Optional; default is
	 *                    #grb::descriptors::no_operation.
	 * @tparam Ring       The semiring type to perform the element-wise addition
	 *                    on.
	 * @tparam InputType1 The left-hand side input type to the additive operator
	 *                    of the \a ring.
	 * @tparam InputType2 The right-hand side input type to the additive operator
	 *                    of the \a ring.
	 * @tparam OutputType The result type of the additive operator of the
	 *                    \a ring.
	 *
	 * @param[out]  z   The output vector of type \a OutputType. This may be a
	 *                  sparse vector.
	 * @param[in]   x   The left-hand input vector of type \a InputType1. This may
	 *                  be a sparse vector.
	 * @param[in] beta  The right-hand input scalar of type \a InputType2.
	 * @param[in] ring  The generalized semiring under which to perform this
	 *                  element-wise multiplication.
	 * @param[in] phase The #grb::Phase the call should execute. Optional; the
	 *                  default parameter is #grb::EXECUTE.
	 *
	 * @return #grb::SUCCESS  On successful completion of this call.
	 * @return #grb::MISMATCH Whenever the dimensions of \a x and \a z do not
	 *                        match. All input data containers are left untouched;
	 *                        it will be as though this call was never made.
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
	 * \parblock
	 * \par Valid descriptors
	 * grb::descriptors::no_operation, grb::descriptors::no_casting,
	 * grb::descriptors::dense.
	 *
	 * \note Invalid descriptors will be ignored.
	 *
	 * If #grb::descriptors::no_casting is specified, then 1) the third domain of
	 * \a ring must match \a InputType1, 2) the fourth domain of \a ring must match
	 * \a InputType2, 3) the fourth domain of \a ring must match \a OutputType. If
	 * one of these is not true, the code shall not compile.
	 * \endparblock
	 *
	 * \par Performance semantics
	 * Each backend must define performance semantics for this primitive.
	 *
	 * @see perfSemantics
	 */
	template<
		Descriptor descr = descriptors::no_operation,
		class Ring, enum Backend backend,
		typename InputType1, typename InputType2, typename OutputType,
		typename Coords
	>
	RC eWiseAdd(
		Vector< OutputType, backend, Coords > &z,
		const Vector< InputType1, backend, Coords > &x,
		const InputType2 beta,
		const Ring &ring = Ring(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if< !grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			grb::is_semiring< Ring >::value, void
		>::type * const = nullptr
	) {
#ifdef _DEBUG
		std::cout << "in eWiseAdd ([T1] <- [T2] + T3), unmasked, base";
#endif
#ifndef NDEBUG
		const bool should_not_call_eWiseAddAAS_base = false;
		assert( should_not_call_eWiseAddAAS_base );
#endif
		(void) z;
		(void) x;
		(void) beta;
		(void) ring;
		(void) phase;
		return UNSUPPORTED;
	}

	/**
	 * Calculates the element-wise addition, \f$ z += \alpha .+ \beta \f$, under a
	 * given semiring.
	 *
	 * \note This is an in-place operation.
	 *
	 * \deprecated This function has been deprecated since v0.5. It may be removed
	 *             at latest at v1.0 of ALP/GraphBLAS-- or any time earlier.
	 *
	 * \note A call to this function is equivalent to two in-place fold operations
	 *       using the additive monoid of the given semiring. Please update any
	 *       code that calls #grb::eWiseAdd with such a sequence as soon as
	 *       possible.
	 *
	 * \note We may consider providing this function as an algorithm in the
	 *       #grb::algorithms namespace, similar to #grb::algorithms::mpv. Please
	 *       let the maintainers know if you would prefer such a solution over
	 *       outright removal and replacement with two folds.
	 *
	 * @tparam descr      The descriptor to be used. Optional; default is
	 *                    #grb::descriptors::no_operation.
	 * @tparam Ring       The semiring type to perform the element-wise addition
	 *                    on.
	 * @tparam InputType1 The left-hand side input type to the additive operator
	 *                    of the \a ring.
	 * @tparam InputType2 The right-hand side input type to the additive operator
	 *                    of the \a ring.
	 * @tparam OutputType The result type of the additive operator of the
	 *                    \a ring.
	 *
	 * @param[out]  z   The output vector of type \a OutputType. This may be a
	 *                  sparse vector.
	 * @param[in] alpha The left-hand input scalar of type \a InputType1.
	 * @param[in] beta  The right-hand input scalar of type \a InputType2.
	 * @param[in] ring  The generalized semiring under which to perform this
	 *                  element-wise multiplication.
	 * @param[in] phase The #grb::Phase the call should execute. Optional; the
	 *                  default parameter is #grb::EXECUTE.
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
	 * \parblock
	 * \par Valid descriptors
	 * grb::descriptors::no_operation, grb::descriptors::no_casting,
	 * grb::descriptors::dense.
	 *
	 * \note Invalid descriptors will be ignored.
	 *
	 * If #grb::descriptors::no_casting is specified, then 1) the third domain of
	 * \a ring must match \a InputType1, 2) the fourth domain of \a ring must match
	 * \a InputType2, 3) the fourth domain of \a ring must match \a OutputType. If
	 * one of these is not true, the code shall not compile.
	 * \endparblock
	 *
	 * \par Performance semantics
	 * Each backend must define performance semantics for this primitive.
	 *
	 * @see perfSemantics
	 */
	template<
		Descriptor descr = descriptors::no_operation,
		class Ring, enum Backend backend,
		typename InputType1, typename InputType2, typename OutputType,
		typename Coords
	>
	RC eWiseAdd(
		Vector< OutputType, backend, Coords > &z,
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
#ifdef _DEBUG
		std::cout << "in eWiseAdd ([T1] <- T2 + T3), unmasked, base";
#endif
#ifndef NDEBUG
		const bool should_not_call_eWiseAddASS_base = false;
		assert( should_not_call_eWiseAddASS_base );
#endif
		(void) z;
		(void) alpha;
		(void) beta;
		(void) ring;
		(void) phase;
		return UNSUPPORTED;
	}

	/**
	 * Calculates the element-wise addition of two vectors, \f$ z += x .+ y \f$,
	 * under a given semiring, masked variant.
	 *
	 * \note This is an in-place operation.
	 *
	 * \deprecated This function has been deprecated since v0.5. It may be removed
	 *             at latest at v1.0 of ALP/GraphBLAS-- or any time earlier.
	 *
	 * \note A call to this function is equivalent to two in-place fold operations
	 *       using the additive monoid of the given semiring. Please update any
	 *       code that calls #grb::eWiseAdd with such a sequence as soon as
	 *       possible.
	 *
	 * \note We may consider providing this function as an algorithm in the
	 *       #grb::algorithms namespace, similar to #grb::algorithms::mpv. Please
	 *       let the maintainers know if you would prefer such a solution over
	 *       outright removal and replacement with two folds.
	 *
	 * @tparam descr      The descriptor to be used. Optional; default is
	 *                    #grb::descriptors::no_operation.
	 * @tparam Ring       The semiring type to perform the element-wise addition
	 *                    on.
	 * @tparam InputType1 The left-hand side input type to the additive operator
	 *                    of the \a ring.
	 * @tparam InputType2 The right-hand side input type to the additive operator
	 *                    of the \a ring.
	 * @tparam OutputType The result type of the additive operator of the
	 *                    \a ring.
	 * @tparam MaskType   The nonzero type of the output mask vector.
	 *
	 * @param[out]  z   The output vector of type \a OutputType. This may be a
	 *                  sparse vector.
	 * @param[in]  mask The output mask vector of type \a MaskType.
	 * @param[in]   x   The left-hand input vector of type \a InputType1. This may
	 *                  be a sparse vector.
	 * @param[in]   y   The right-hand input vector of type \a InputType2. This may
	 *                  be a sparse vector.
	 * @param[in] ring  The generalized semiring under which to perform this
	 *                  element-wise multiplication.
	 * @param[in] phase The #grb::Phase the call should execute. Optional; the
	 *                  default parameter is #grb::EXECUTE.
	 *
	 * \note There are also variants where \a x and/or \a y are scalars, as well
	 *       as unmasked variants.
	 *
	 * @return #grb::SUCCESS  On successful completion of this call.
	 * @return #grb::MISMATCH Whenever the dimensions of \a mask, \a x, \a y, and
	 *                        \a z do not match. All input data containers are left
	 *                        untouched; it will be as though this call was never
	 *                        made.
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
	 * \parblock
	 * \par Valid descriptors
	 *  - #grb::descriptors::no_operation,
	 *  - #grb::descriptors::no_casting,
	 *  - #grb::descriptors::dense,
	 *  - #grb::descriptors::invert_mask,
	 *  - #grb::descriptors::structural, and
	 *  - #grb::descriptors::structural_complement.
	 *
	 * \note Invalid descriptors will be ignored.
	 *
	 * If #grb::descriptors::no_casting is specified, then 1) the third domain of
	 * \a ring must match \a InputType1, 2) the fourth domain of \a ring must match
	 * \a InputType2, 3) the fourth domain of \a ring must match \a OutputType. If
	 * one of these is not true, the code shall not compile.
	 * \endparblock
	 *
	 * \par Performance semantics
	 * Each backend must define performance semantics for this primitive.
	 *
	 * @see perfSemantics
	 */
	template<
		Descriptor descr = descriptors::no_operation,
		class Ring, enum Backend backend,
		typename OutputType, typename MaskType,
		typename InputType1, typename InputType2,
		typename Coords
	>
	RC eWiseAdd(
		Vector< OutputType, backend, Coords > &z,
		const Vector< MaskType, backend, Coords > &mask,
		const Vector< InputType1, backend, Coords > &x,
		const Vector< InputType2, backend, Coords > &y,
		const Ring &ring = Ring(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if< !grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			grb::is_semiring< Ring >::value, void
		>::type * const = nullptr
	) {
#ifdef _DEBUG
		std::cout << "in eWiseAdd ([T1] <- [T2] + [T3]), masked, base";
#endif
#ifndef NDEBUG
		const bool should_not_call_eWiseAddAMAA_base = false;
		assert( should_not_call_eWiseAddAMAA_base );
#endif
		(void) z;
		(void) mask;
		(void) x;
		(void) y;
		(void) ring;
		(void) phase;
		return UNSUPPORTED;
	}

	/**
	 * Calculates the element-wise addition, \f$ z += \alpha .+ y \f$, under a
	 * given semiring, masked variant.
	 *
	 * \note This is an in-place operation.
	 *
	 * \deprecated This function has been deprecated since v0.5. It may be removed
	 *             at latest at v1.0 of ALP/GraphBLAS-- or any time earlier.
	 *
	 * \note A call to this function is equivalent to two in-place fold operations
	 *       using the additive monoid of the given semiring. Please update any
	 *       code that calls #grb::eWiseAdd with such a sequence as soon as
	 *       possible.
	 *
	 * \note We may consider providing this function as an algorithm in the
	 *       #grb::algorithms namespace, similar to #grb::algorithms::mpv. Please
	 *       let the maintainers know if you would prefer such a solution over
	 *       outright removal and replacement with two folds.
	 *
	 * @tparam descr      The descriptor to be used. Optional; default is
	 *                    #grb::descriptors::no_operation.
	 * @tparam Ring       The semiring type to perform the element-wise addition
	 *                    on.
	 * @tparam InputType1 The left-hand side input type to the additive operator
	 *                    of the \a ring.
	 * @tparam InputType2 The right-hand side input type to the additive operator
	 *                    of the \a ring.
	 * @tparam OutputType The result type of the additive operator of the
	 *                    \a ring.
	 * @tparam MaskType   The nonzero type of the output mask vector.
	 *
	 * @param[out]  z   The output vector of type \a OutputType. This may be a
	 *                  sparse vector.
	 * @param[in]  mask The output mask.
	 * @param[in] alpha The left-hand input scalar of type \a InputType1.
	 * @param[in]   y   The right-hand input vector of type \a InputType2. This may
	 *                  be a sparse vector.
	 * @param[in] ring  The generalized semiring under which to perform this
	 *                  element-wise multiplication.
	 * @param[in] phase The #grb::Phase the call should execute. Optional; the
	 *                  default parameter is #grb::EXECUTE.
	 *
	 * @return #grb::SUCCESS  On successful completion of this call.
	 * @return #grb::MISMATCH Whenever the dimensions of \a mask, \a y, and \a z do
	 *                        not match. All input data containers are left
	 *                        untouched; it will be as though this call was never
	 *                        made.
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
	 * \parblock
	 * \par Valid descriptors
	 *  - #grb::descriptors::no_operation,
	 *  - #grb::descriptors::no_casting,
	 *  - #grb::descriptors::dense,
	 *  - #grb::descriptors::invert_mask,
	 *  - #grb::descriptors::structural, and
	 *  - #grb::descriptors::structural_complement.
	 *
	 * \note Invalid descriptors will be ignored.
	 *
	 * If #grb::descriptors::no_casting is specified, then 1) the third domain of
	 * \a ring must match \a InputType1, 2) the fourth domain of \a ring must match
	 * \a InputType2, 3) the fourth domain of \a ring must match \a OutputType. If
	 * one of these is not true, the code shall not compile.
	 * \endparblock
	 *
	 * \par Performance semantics
	 * Each backend must define performance semantics for this primitive.
	 *
	 * @see perfSemantics
	 */
	template<
		Descriptor descr = descriptors::no_operation,
		class Ring, enum Backend backend,
		typename InputType1, typename InputType2,
		typename OutputType, typename MaskType,
		typename Coords
	>
	RC eWiseAdd(
		Vector< OutputType, backend, Coords > &z,
		const Vector< MaskType, backend, Coords > &mask,
		const InputType1 alpha,
		const Vector< InputType2, backend, Coords > &y,
		const Ring &ring = Ring(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if< !grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			grb::is_semiring< Ring >::value, void
		>::type * const = nullptr
	) {
#ifdef _DEBUG
		std::cout << "in eWiseAdd ([T1] <- T2 + [T3]), masked, base";
#endif
#ifndef NDEBUG
		const bool should_not_call_eWiseAddAMSA_base = false;
		assert( should_not_call_eWiseAddAMSA_base );
#endif
		(void) z;
		(void) mask;
		(void) alpha;
		(void) y;
		(void) ring;
		(void) phase;
		return UNSUPPORTED;
	}

	/**
	 * Calculates the element-wise addition, \f$ z += x .+ \beta \f$, under a
	 * given semiring, masked variant.
	 *
	 * \note This is an in-place operation.
	 *
	 * \deprecated This function has been deprecated since v0.5. It may be removed
	 *             at latest at v1.0 of ALP/GraphBLAS-- or any time earlier.
	 *
	 * \note A call to this function is equivalent to two in-place fold operations
	 *       using the additive monoid of the given semiring. Please update any
	 *       code that calls #grb::eWiseAdd with such a sequence as soon as
	 *       possible.
	 *
	 * \note We may consider providing this function as an algorithm in the
	 *       #grb::algorithms namespace, similar to #grb::algorithms::mpv. Please
	 *       let the maintainers know if you would prefer such a solution over
	 *       outright removal and replacement with two folds.
	 *
	 * @tparam descr      The descriptor to be used. Optional; default is
	 *                    #grb::descriptors::no_operation.
	 * @tparam Ring       The semiring type to perform the element-wise addition
	 *                    on.
	 * @tparam InputType1 The left-hand side input type to the additive operator
	 *                    of the \a ring.
	 * @tparam InputType2 The right-hand side input type to the additive operator
	 *                    of the \a ring.
	 * @tparam OutputType The result type of the additive operator of the
	 *                    \a ring.
	 * @tparam MaskType   The nonzero type of the output mask vector.
	 *
	 * @param[out]  z   The output vector of type \a OutputType. This may be a
	 *                  sparse vector.
	 * @param[in]  mask The output mask.
	 * @param[in]   x   The left-hand input vector of type \a InputType1. This may
	 *                  be a sparse vector.
	 * @param[in] beta  The right-hand input scalar of type \a InputType2.
	 * @param[in] ring  The generalized semiring under which to perform this
	 *                  element-wise multiplication.
	 * @param[in] phase The #grb::Phase the call should execute. Optional; the
	 *                  default parameter is #grb::EXECUTE.
	 *
	 * @return #grb::SUCCESS  On successful completion of this call.
	 * @return #grb::MISMATCH Whenever the dimensions of \a mask, \a x, and \a z do
	 *                        not match. All input data containers are left
	 *                        untouched; it will be as though this call was never
	 *                        made.
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
	 * \parblock
	 * \par Valid descriptors
	 *  - #grb::descriptors::no_operation,
	 *  - #grb::descriptors::no_casting,
	 *  - #grb::descriptors::dense,
	 *  - #grb::descriptors::invert_mask,
	 *  - #grb::descriptors::structural, and
	 *  - #grb::descriptors::structural_complement.
	 *
	 * \note Invalid descriptors will be ignored.
	 *
	 * If #grb::descriptors::no_casting is specified, then 1) the third domain of
	 * \a ring must match \a InputType1, 2) the fourth domain of \a ring must match
	 * \a InputType2, 3) the fourth domain of \a ring must match \a OutputType. If
	 * one of these is not true, the code shall not compile.
	 * \endparblock
	 *
	 * \par Performance semantics
	 * Each backend must define performance semantics for this primitive.
	 *
	 * @see perfSemantics
	 */
	template<
		Descriptor descr = descriptors::no_operation,
		class Ring, enum Backend backend,
		typename InputType1, typename InputType2,
		typename OutputType, typename MaskType,
		typename Coords
	>
	RC eWiseAdd(
		Vector< OutputType, backend, Coords > &z,
		const Vector< MaskType, backend, Coords > &mask,
		const Vector< InputType1, backend, Coords > &x,
		const InputType2 beta,
		const Ring &ring = Ring(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if< !grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			grb::is_semiring< Ring >::value, void
		>::type * const = nullptr
	) {
#ifdef _DEBUG
		std::cout << "in eWiseAdd ([T1] <- [T2] + T3), masked, base";
#endif
#ifndef NDEBUG
		const bool should_not_call_eWiseAddAMAS_base = false;
		assert( should_not_call_eWiseAddAMAS_base );
#endif
		(void) z;
		(void) mask;
		(void) x;
		(void) beta;
		(void) ring;
		(void) phase;
		return UNSUPPORTED;
	}

	/**
	 * Calculates the element-wise addition, \f$ z += \alpha .+ \beta \f$, under a
	 * given semiring, masked variant.
	 *
	 * \note This is an in-place operation.
	 *
	 * \deprecated This function has been deprecated since v0.5. It may be removed
	 *             at latest at v1.0 of ALP/GraphBLAS-- or any time earlier.
	 *
	 * \note A call to this function is equivalent to two in-place fold operations
	 *       using the additive monoid of the given semiring. Please update any
	 *       code that calls #grb::eWiseAdd with such a sequence as soon as
	 *       possible.
	 *
	 * \note We may consider providing this function as an algorithm in the
	 *       #grb::algorithms namespace, similar to #grb::algorithms::mpv. Please
	 *       let the maintainers know if you would prefer such a solution over
	 *       outright removal and replacement with two folds.
	 *
	 * @tparam descr      The descriptor to be used. Optional; default is
	 *                    #grb::descriptors::no_operation.
	 * @tparam Ring       The semiring type to perform the element-wise addition
	 *                    on.
	 * @tparam InputType1 The left-hand side input type to the additive operator
	 *                    of the \a ring.
	 * @tparam InputType2 The right-hand side input type to the additive operator
	 *                    of the \a ring.
	 * @tparam OutputType The result type of the additive operator of the
	 *                    \a ring.
	 * @tparam MaskType   The nonzero type of the output mask vector.
	 *
	 * @param[out]  z   The output vector of type \a OutputType. This may be a
	 *                  sparse vector.
	 * @param[in]  mask The output mask.
	 * @param[in] alpha The left-hand input scalar of type \a InputType1.
	 * @param[in] beta  The right-hand input scalar of type \a InputType2.
	 * @param[in] ring  The generalized semiring under which to perform this
	 *                  element-wise multiplication.
	 * @param[in] phase The #grb::Phase the call should execute. Optional; the
	 *                  default parameter is #grb::EXECUTE.
	 *
	 * @return #grb::SUCCESS  On successful completion of this call.
	 * @return #grb::MISMATCH If \a mask and \a z do not have the same size.
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
	 * \parblock
	 * \par Valid descriptors
	 *  - #grb::descriptors::no_operation,
	 *  - #grb::descriptors::no_casting,
	 *  - #grb::descriptors::dense,
	 *  - #grb::descriptors::invert_mask,
	 *  - #grb::descriptors::structural, and
	 *  - #grb::descriptors::structural_complement.
	 *
	 * \note Invalid descriptors will be ignored.
	 *
	 * If #grb::descriptors::no_casting is specified, then 1) the third domain of
	 * \a ring must match \a InputType1, 2) the fourth domain of \a ring must match
	 * \a InputType2, 3) the fourth domain of \a ring must match \a OutputType. If
	 * one of these is not true, the code shall not compile.
	 * \endparblock
	 *
	 * \par Performance semantics
	 * Each backend must define performance semantics for this primitive.
	 *
	 * @see perfSemantics
	 */
	template<
		Descriptor descr = descriptors::no_operation,
		class Ring, enum Backend backend,
		typename InputType1, typename InputType2,
		typename OutputType, typename MaskType,
		typename Coords
	>
	RC eWiseAdd(
		Vector< OutputType, backend, Coords > &z,
		const Vector< MaskType, backend, Coords > &mask,
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
#ifdef _DEBUG
		std::cout << "in eWiseAdd ([T1] <- T2 + T3), masked, base";
#endif
#ifndef NDEBUG
		const bool should_not_call_eWiseAddAMSS_base = false;
		assert( should_not_call_eWiseAddAMSS_base );
#endif
		(void) z;
		(void) mask;
		(void) alpha;
		(void) beta;
		(void) ring;
		(void) phase;
		return UNSUPPORTED;
	}

	/**
	 * In-place element-wise multiplication of two vectors, \f$ z += x .* y \f$,
	 * under a given semiring.
	 *
	 * @tparam descr      The descriptor to be used. Optional; the default is
	 *                    #grb::descriptors::no_operation.
	 * @tparam Ring       The semiring type to perform the element-wise multiply
	 *                    with.
	 * @tparam InputType1 The left-hand side input type.
	 * @tparam InputType2 The right-hand side input type.
	 * @tparam OutputType The output type.
	 *
	 * @param[out]  z   The output vector of type \a OutputType.
	 * @param[in]   x   The left-hand input vector of type \a InputType1.
	 * @param[in]   y   The right-hand input vector of type \a InputType2.
	 * @param[in] ring  The generalized semiring under which to perform this
	 *                  element-wise multiplication.
	 * @param[in] phase The #grb::Phase the call should execute. Optional; the
	 *                  default parameter is #grb::EXECUTE.
	 *
	 * @return #grb::SUCCESS  On successful completion of this call.
	 * @return #grb::MISMATCH Whenever the dimensions of \a x, \a y, and \a z do
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
	 * \warning Unlike #grb::eWiseApply using monoids, given sparse vectors,
	 *          missing elements in sparse input vectors are now interpreted as a
	 *          the zero identity, therefore annihilating instead of acting as a
	 *          monoid identity. Therefore even when \a z is empty on input, the
	 *          #grb::eWiseApply with monoids does not incur the same behaviour as
	 *          this function. The #grb::eWiseApply with operators \em is similar,
	 *          except that this function is in-place and #grb::eWiseApply is not.
	 *
	 * \parblock
	 * \par Valid descriptors
	 *  - #grb::descriptors::no_operation,
	 *  - #grb::descriptors::no_casting, and
	 *  - #grb::descriptors::dense.
	 *
	 * \note Invalid descriptors will be ignored.
	 *
	 * If #grb::descriptors::no_casting is specified, then 1) the first domain of
	 * \a ring must match \a InputType1, 2) the second domain of \a ring must match
	 * \a InputType2, 3) the third domain of \a ring must match \a OutputType. If
	 * one of these is not true, the code shall not compile.
	 * \endparblock
	 *
	 * \par Performance semantics
	 * Each backend must define performance semantics for this primitive.
	 *
	 * @see perfSemantics
	 */
	template<
		Descriptor descr = descriptors::no_operation,
		class Ring, enum Backend backend,
		typename InputType1, typename InputType2, typename OutputType,
		typename Coords
	>
	RC eWiseMul(
		Vector< OutputType, backend, Coords > &z,
		const Vector< InputType1, backend, Coords > &x,
		const Vector< InputType2, backend, Coords > &y,
		const Ring &ring = Ring(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if< !grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			grb::is_semiring< Ring >::value, void
		>::type * const = nullptr
	) {
#ifdef _DEBUG
		std::cout << "in eWiseMul ([T1] <- [T2] * [T3]), unmasked, base";
#endif
#ifndef NDEBUG
		const bool should_not_call_eWiseMulAAA_base = false;
		assert( should_not_call_eWiseMulAAA_base );
#endif
		(void) z;
		(void) x;
		(void) y;
		(void) ring;
		(void) phase;
		return UNSUPPORTED;
	}

	/**
	 * In-place element-wise multiplication of a scalar and vector,
	 * \f$ z += \alpha .* y \f$, under a given semiring.
	 *
	 * @tparam descr      The descriptor to be used. Optional; the default is
	 *                    #grb::descriptors::no_operation.
	 * @tparam Ring       The semiring type to perform the element-wise multiply
	 *                    with.
	 * @tparam InputType1 The left-hand side input type.
	 * @tparam InputType2 The right-hand side input type.
	 * @tparam OutputType The output type.
	 *
	 * @param[out]  z   The output vector of type \a OutputType.
	 * @param[in] alpha The left-hand input scalar of type \a InputType1.
	 * @param[in]   y   The right-hand input vector of type \a InputType2.
	 * @param[in] ring  The generalized semiring under which to perform this
	 *                  element-wise multiplication.
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
	 * \warning Unlike #grb::eWiseApply using monoids, given sparse vectors,
	 *          missing elements in sparse input vectors are now interpreted as a
	 *          the zero identity, therefore annihilating instead of acting as a
	 *          monoid identity. Therefore even when \a z is empty on input, the
	 *          #grb::eWiseApply with monoids does not incur the same behaviour as
	 *          this function. The #grb::eWiseApply with operators \em is similar,
	 *          except that this function is in-place and #grb::eWiseApply is not.
	 *
	 * \parblock
	 * \par Valid descriptors
	 *  - #grb::descriptors::no_operation,
	 *  - #grb::descriptors::no_casting, and
	 *  - #grb::descriptors::dense.
	 *
	 * \note Invalid descriptors will be ignored.
	 *
	 * If #grb::descriptors::no_casting is specified, then 1) the first domain of
	 * \a ring must match \a InputType1, 2) the second domain of \a ring must match
	 * \a InputType2, 3) the third domain of \a ring must match \a OutputType. If
	 * one of these is not true, the code shall not compile.
	 * \endparblock
	 *
	 * \par Performance semantics
	 * Each backend must define performance semantics for this primitive.
	 *
	 * @see perfSemantics
	 */
	template<
		Descriptor descr = descriptors::no_operation,
		class Ring, enum Backend backend,
		typename InputType1, typename InputType2, typename OutputType,
		typename Coords
	>
	RC eWiseMul(
		Vector< OutputType, backend, Coords > &z,
		const InputType1 alpha,
		const Vector< InputType2, backend, Coords > &y,
		const Ring &ring = Ring(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if< !grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			grb::is_semiring< Ring >::value, void
		>::type * const = nullptr
	) {
#ifdef _DEBUG
		std::cout << "in eWiseMul ([T1] <- T2 * [T3]), unmasked, base";
#endif
#ifndef NDEBUG
		const bool should_not_call_eWiseMulASA_base = false;
		assert( should_not_call_eWiseMulASA_base );
#endif
		(void) z;
		(void) alpha;
		(void) y;
		(void) ring;
		(void) phase;
		return UNSUPPORTED;
	}

	/**
	 * In-place element-wise multiplication of a vector and scalar,
	 * \f$ z += x .* \beta \f$, under a given semiring.
	 *
	 * @tparam descr      The descriptor to be used. Optional; the default is
	 *                    #grb::descriptors::no_operation.
	 * @tparam Ring       The semiring type to perform the element-wise multiply
	 *                    with.
	 * @tparam InputType1 The left-hand side input type.
	 * @tparam InputType2 The right-hand side input type.
	 * @tparam OutputType The output type.
	 *
	 * @param[out]  z   The output vector of type \a OutputType.
	 * @param[in]   x   The left-hand input vector of type \a InputType1.
	 * @param[in] beta  The right-hand input scalar of type \a InputType2.
	 * @param[in] ring  The generalized semiring under which to perform this
	 *                  element-wise multiplication.
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
	 * \warning Unlike #grb::eWiseApply using monoids, given sparse vectors,
	 *          missing elements in sparse input vectors are now interpreted as a
	 *          the zero identity, therefore annihilating instead of acting as a
	 *          monoid identity. Therefore even when \a z is empty on input, the
	 *          #grb::eWiseApply with monoids does not incur the same behaviour as
	 *          this function. The #grb::eWiseApply with operators \em is similar,
	 *          except that this function is in-place and #grb::eWiseApply is not.
	 *
	 * \parblock
	 * \par Valid descriptors
	 *  - #grb::descriptors::no_operation,
	 *  - #grb::descriptors::no_casting, and
	 *  - #grb::descriptors::dense.
	 *
	 * \note Invalid descriptors will be ignored.
	 *
	 * If #grb::descriptors::no_casting is specified, then 1) the first domain of
	 * \a ring must match \a InputType1, 2) the second domain of \a ring must match
	 * \a InputType2, 3) the third domain of \a ring must match \a OutputType. If
	 * one of these is not true, the code shall not compile.
	 * \endparblock
	 *
	 * \par Performance semantics
	 * Each backend must define performance semantics for this primitive.
	 *
	 * @see perfSemantics
	 */
	template<
		Descriptor descr = descriptors::no_operation,
		class Ring, enum Backend backend,
		typename InputType1, typename InputType2, typename OutputType,
		typename Coords
	>
	RC eWiseMul(
		Vector< OutputType, backend, Coords > &z,
		const Vector< InputType1, backend, Coords > &x,
		const InputType2 beta,
		const Ring &ring = Ring(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if< !grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			grb::is_semiring< Ring >::value, void
		>::type * const = nullptr
	) {
#ifdef _DEBUG
		std::cout << "in eWiseMul ([T1] <- [T2] * T3), unmasked, base";
#endif
#ifndef NDEBUG
		const bool should_not_call_eWiseMulAAS_base = false;
		assert( should_not_call_eWiseMulAAS_base );
#endif
		(void) z;
		(void) x;
		(void) beta;
		(void) ring;
		(void) phase;
		return UNSUPPORTED;
	}

	/**
	 * In-place element-wise multiplication of two scalars,
	 * \f$ z += \alpha .* \beta \f$, under a given semiring.
	 *
	 * @tparam descr      The descriptor to be used. Optional; the default is
	 *                    #grb::descriptors::no_operation.
	 * @tparam Ring       The semiring type to perform the element-wise multiply
	 *                    with.
	 * @tparam InputType1 The left-hand side input type.
	 * @tparam InputType2 The right-hand side input type.
	 * @tparam OutputType The output type.
	 *
	 * @param[out]  z   The output vector of type \a OutputType.
	 * @param[in] alpha The left-hand input scalar of type \a InputType1.
	 * @param[in] beta  The right-hand input scalar of type \a InputType2.
	 * @param[in] ring  The generalized semiring under which to perform this
	 *                  element-wise multiplication.
	 * @param[in] phase The #grb::Phase the call should execute. Optional; the
	 *                  default parameter is #grb::EXECUTE.
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
	 * \warning Unlike #grb::eWiseApply using monoids, given sparse vectors,
	 *          missing elements in sparse input vectors are now interpreted as a
	 *          the zero identity, therefore annihilating instead of acting as a
	 *          monoid identity. Therefore even when \a z is empty on input, the
	 *          #grb::eWiseApply with monoids does not incur the same behaviour as
	 *          this function. The #grb::eWiseApply with operators \em is similar,
	 *          except that this function is in-place and #grb::eWiseApply is not.
	 *
	 * \parblock
	 * \par Valid descriptors
	 *  - #grb::descriptors::no_operation,
	 *  - #grb::descriptors::no_casting, and
	 *  - #grb::descriptors::dense.
	 *
	 * \note Invalid descriptors will be ignored.
	 *
	 * If #grb::descriptors::no_casting is specified, then 1) the first domain of
	 * \a ring must match \a InputType1, 2) the second domain of \a ring must match
	 * \a InputType2, 3) the third domain of \a ring must match \a OutputType. If
	 * one of these is not true, the code shall not compile.
	 * \endparblock
	 *
	 * \par Performance semantics
	 * Each backend must define performance semantics for this primitive.
	 *
	 * @see perfSemantics
	 */
	template<
		Descriptor descr = descriptors::no_operation,
		class Ring, enum Backend backend,
		typename InputType1, typename InputType2, typename OutputType,
		typename Coords
	>
	RC eWiseMul(
		Vector< OutputType, backend, Coords > &z,
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
#ifdef _DEBUG
		std::cout << "in eWiseMul ([T1] <- T2 * T3), unmasked, base";
#endif
#ifndef NDEBUG
		const bool should_not_call_eWiseMulASS_base = false;
		assert( should_not_call_eWiseMulASS_base );
#endif
		(void) z;
		(void) alpha;
		(void) beta;
		(void) ring;
		(void) phase;
		return UNSUPPORTED;
	}

	/**
	 * In-place element-wise multiplication of two vectors, \f$ z += x .* y \f$,
	 * under a given semiring, masked variant.
	 *
	 * @tparam descr      The descriptor to be used. Optional; the default is
	 *                    #grb::descriptors::no_operation.
	 * @tparam Ring       The semiring type to perform the element-wise multiply
	 *                    with.
	 * @tparam InputType1 The left-hand side input type.
	 * @tparam InputType2 The right-hand side input type.
	 * @tparam OutputType The output vector type.
	 * @tparam MaskType   The output mask type.
	 *
	 * @param[in,out] z The output vector of type \a OutputType.
	 * @param[in]  mask The ouput mask of type \a MaskType.
	 * @param[in]   x   The left-hand input vector of type \a InputType1.
	 * @param[in]   y   The right-hand input vector of type \a InputType2.
	 * @param[in] ring  The generalized semiring under which to perform this
	 *                  element-wise multiplication.
	 * @param[in] phase The #grb::Phase the call should execute. Optional; the
	 *                  default parameter is #grb::EXECUTE.
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
	 * \warning Unlike #grb::eWiseApply using monoids, given sparse vectors,
	 *          missing elements in sparse input vectors are now interpreted as a
	 *          the zero identity, therefore annihilating instead of acting as a
	 *          monoid identity. Therefore even when \a z is empty on input, the
	 *          #grb::eWiseApply with monoids does not incur the same behaviour as
	 *          this function. The #grb::eWiseApply with operators \em is similar,
	 *          except that this function is in-place and #grb::eWiseApply is not.
	 *
	 * \parblock
	 * \par Valid descriptors
	 *  - #grb::descriptors::no_operation,
	 *  - #grb::descriptors::no_casting,
	 *  - #grb::descriptors::dense,
	 *  - #grb::descriptors::invert_mask,
	 *  - #grb::descriptors::structural, and
	 *  - #grb::descriptors::structural_complement.
	 *
	 * \note Invalid descriptors will be ignored.
	 *
	 * If #grb::descriptors::no_casting is specified, then 1) the first domain of
	 * \a ring must match \a InputType1, 2) the second domain of \a ring must match
	 * \a InputType2, 3) the third domain of \a ring must match \a OutputType. If
	 * one of these is not true, the code shall not compile.
	 * \endparblock
	 *
	 * \par Performance semantics
	 * Each backend must define performance semantics for this primitive.
	 *
	 * @see perfSemantics
	 */
	template<
		Descriptor descr = descriptors::no_operation,
		class Ring, enum Backend backend,
		typename InputType1, typename InputType2,
		typename OutputType, typename MaskType,
		typename Coords
	>
	RC eWiseMul(
		Vector< OutputType, backend, Coords > &z,
		const Vector< MaskType, backend, Coords > &mask,
		const Vector< InputType1, backend, Coords > &x,
		const Vector< InputType2, backend, Coords > &y,
		const Ring &ring = Ring(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if< !grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			grb::is_semiring< Ring >::value, void
		>::type * const = nullptr
	) {
#ifdef _DEBUG
		std::cout << "in eWiseMul ([T1] <- [T2] * [T3]), masked, base";
#endif
#ifndef NDEBUG
		const bool should_not_call_eWiseMulAMAA_base = false;
		assert( should_not_call_eWiseMulAMAA_base );
#endif
		(void) z;
		(void) mask;
		(void) x;
		(void) y;
		(void) ring;
		(void) phase;
		return UNSUPPORTED;
	}

	/**
	 * In-place element-wise multiplication of a scalar and vector,
	 * \f$ z += \alpha .* y \f$, under a given semiring, masked variant.
	 *
	 * @tparam descr      The descriptor to be used. Optional; the default is
	 *                    #grb::descriptors::no_operation.
	 * @tparam Ring       The semiring type to perform the element-wise multiply
	 *                    with.
	 * @tparam InputType1 The left-hand side input type.
	 * @tparam InputType2 The right-hand side input type.
	 * @tparam OutputType The output vector type.
	 * @tparam MaskType   The output mask type.
	 *
	 * @param[in,out] z The output vector of type \a OutputType.
	 * @param[in]  mask The ouput mask of type \a MaskType.
	 * @param[in] alpha The left-hand input scalar of type \a InputType1.
	 * @param[in]   y   The right-hand input vector of type \a InputType2.
	 * @param[in] ring  The generalized semiring under which to perform this
	 *                  element-wise multiplication.
	 * @param[in] phase The #grb::Phase the call should execute. Optional; the
	 *                  default parameter is #grb::EXECUTE.
	 *
	 * @return #grb::SUCCESS  On successful completion of this call.
	 * @return #grb::MISMATCH Whenever the dimensions of \a mask, \a y, and \a z do
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
	 * \warning Unlike #grb::eWiseApply using monoids, given sparse vectors,
	 *          missing elements in sparse input vectors are now interpreted as a
	 *          the zero identity, therefore annihilating instead of acting as a
	 *          monoid identity. Therefore even when \a z is empty on input, the
	 *          #grb::eWiseApply with monoids does not incur the same behaviour as
	 *          this function. The #grb::eWiseApply with operators \em is similar,
	 *          except that this function is in-place and #grb::eWiseApply is not.
	 *
	 * \parblock
	 * \par Valid descriptors
	 *  - #grb::descriptors::no_operation,
	 *  - #grb::descriptors::no_casting,
	 *  - #grb::descriptors::dense,
	 *  - #grb::descriptors::invert_mask,
	 *  - #grb::descriptors::structural, and
	 *  - #grb::descriptors::structural_complement.
	 *
	 * \note Invalid descriptors will be ignored.
	 *
	 * If #grb::descriptors::no_casting is specified, then 1) the first domain of
	 * \a ring must match \a InputType1, 2) the second domain of \a ring must match
	 * \a InputType2, 3) the third domain of \a ring must match \a OutputType. If
	 * one of these is not true, the code shall not compile.
	 * \endparblock
	 *
	 * \par Performance semantics
	 * Each backend must define performance semantics for this primitive.
	 *
	 * @see perfSemantics
	 */
	template<
		Descriptor descr = descriptors::no_operation,
		class Ring, enum Backend backend,
		typename InputType1, typename InputType2,
		typename OutputType, typename MaskType,
		typename Coords
	>
	RC eWiseMul(
		Vector< OutputType, backend, Coords > &z,
		const Vector< MaskType, backend, Coords > &mask,
		const InputType1 alpha,
		const Vector< InputType2, backend, Coords > &y,
		const Ring &ring = Ring(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if< !grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			grb::is_semiring< Ring >::value, void
		>::type * const = nullptr
	) {
#ifdef _DEBUG
		std::cout << "in eWiseMul ([T1] <- T2 * [T3]), masked, base";
#endif
#ifndef NDEBUG
		const bool should_not_call_eWiseMulAMSA_base = false;
		assert( should_not_call_eWiseMulAMSA_base );
#endif
		(void) z;
		(void) mask;
		(void) alpha;
		(void) y;
		(void) ring;
		(void) phase;
		return UNSUPPORTED;
	}

	/**
	 * In-place element-wise multiplication of a vector and scalar,
	 * \f$ z += x .* \beta \f$, under a given semiring, masked variant.
	 *
	 * @tparam descr      The descriptor to be used. Optional; the default is
	 *                    #grb::descriptors::no_operation.
	 * @tparam Ring       The semiring type to perform the element-wise multiply
	 *                    with.
	 * @tparam InputType1 The left-hand side input type.
	 * @tparam InputType2 The right-hand side input type.
	 * @tparam OutputType The output vector type.
	 * @tparam MaskType   The output mask type.
	 *
	 * @param[in,out] z The output vector of type \a OutputType.
	 * @param[in]  mask The output mask of type \a MaskType.
	 * @param[in]   x   The left-hand input vector of type \a InputType1.
	 * @param[in] beta  The right-hand input scalar of type \a InputType2.
	 * @param[in] ring  The generalized semiring under which to perform this
	 *                  element-wise multiplication.
	 * @param[in] phase The #grb::Phase the call should execute. Optional; the
	 *                  default parameter is #grb::EXECUTE.
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
	 * \warning Unlike #grb::eWiseApply using monoids, given sparse vectors,
	 *          missing elements in sparse input vectors are now interpreted as a
	 *          the zero identity, therefore annihilating instead of acting as a
	 *          monoid identity. Therefore even when \a z is empty on input, the
	 *          #grb::eWiseApply with monoids does not incur the same behaviour as
	 *          this function. The #grb::eWiseApply with operators \em is similar,
	 *          except that this function is in-place and #grb::eWiseApply is not.
	 *
	 * \parblock
	 * \par Valid descriptors
	 *  - #grb::descriptors::no_operation,
	 *  - #grb::descriptors::no_casting,
	 *  - #grb::descriptors::dense,
	 *  - #grb::descriptors::invert_mask,
	 *  - #grb::descriptors::structural, and
	 *  - #grb::descriptors::structural_complement.
	 *
	 * \note Invalid descriptors will be ignored.
	 *
	 * If #grb::descriptors::no_casting is specified, then 1) the first domain of
	 * \a ring must match \a InputType1, 2) the second domain of \a ring must match
	 * \a InputType2, 3) the third domain of \a ring must match \a OutputType. If
	 * one of these is not true, the code shall not compile.
	 * \endparblock
	 *
	 * \par Performance semantics
	 * Each backend must define performance semantics for this primitive.
	 *
	 * @see perfSemantics
	 */
	template<
		Descriptor descr = descriptors::no_operation,
		class Ring, enum Backend backend,
		typename InputType1, typename InputType2,
		typename OutputType, typename MaskType,
		typename Coords
	>
	RC eWiseMul(
		Vector< OutputType, backend, Coords > &z,
		const Vector< MaskType, backend, Coords > &mask,
		const Vector< InputType1, backend, Coords > &x,
		const InputType2 beta,
		const Ring &ring = Ring(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if< !grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			grb::is_semiring< Ring >::value, void
		>::type * const = nullptr
	) {
#ifdef _DEBUG
		std::cout << "in eWiseMul ([T1] <- [T2] * T3), masked, base";
#endif
#ifndef NDEBUG
		const bool should_not_call_eWiseMulAMAS_base = false;
		assert( should_not_call_eWiseMulAMAS_base );
#endif
		(void) z;
		(void) mask;
		(void) x;
		(void) beta;
		(void) ring;
		(void) phase;
		return UNSUPPORTED;
	}

	/**
	 * In-place element-wise multiplication of two scalars,
	 * \f$ z += \alpha .* \beta \f$, under a given semiring, masked variant.
	 *
	 * @tparam descr      The descriptor to be used. Optional; the default is
	 *                    #grb::descriptors::no_operation.
	 * @tparam Ring       The semiring type to perform the element-wise multiply
	 *                    with.
	 * @tparam InputType1 The left-hand side input type.
	 * @tparam InputType2 The right-hand side input type.
	 * @tparam OutputType The output vector type.
	 * @tparam MaskType   The output mask type.
	 *
	 * @param[in,out] z The output vector of type \a OutputType.
	 * @param[in]  mask The ouput mask of type \a MaskType.
	 * @param[in] alpha The left-hand input scalar of type \a InputType1.
	 * @param[in] beta  The right-hand input scalar of type \a InputType2.
	 * @param[in] ring  The generalized semiring under which to perform this
	 *                  element-wise multiplication.
	 * @param[in] phase The #grb::Phase the call should execute. Optional; the
	 *                  default parameter is #grb::EXECUTE.
	 *
	 * @return #grb::SUCCESS  On successful completion of this call.
	 * @return #grb::MISMATCH If \a mask and \a z have different size.
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
	 * \warning Unlike #grb::eWiseApply using monoids, given sparse vectors,
	 *          missing elements in sparse input vectors are now interpreted as a
	 *          the zero identity, therefore annihilating instead of acting as a
	 *          monoid identity. Therefore even when \a z is empty on input, the
	 *          #grb::eWiseApply with monoids does not incur the same behaviour as
	 *          this function. The #grb::eWiseApply with operators \em is similar,
	 *          except that this function is in-place and #grb::eWiseApply is not.
	 *
	 * \parblock
	 * \par Valid descriptors
	 *  - #grb::descriptors::no_operation,
	 *  - #grb::descriptors::no_casting,
	 *  - #grb::descriptors::dense,
	 *  - #grb::descriptors::invert_mask,
	 *  - #grb::descriptors::structural, and
	 *  - #grb::descriptors::structural_complement.
	 *
	 * \note Invalid descriptors will be ignored.
	 *
	 * If #grb::descriptors::no_casting is specified, then 1) the first domain of
	 * \a ring must match \a InputType1, 2) the second domain of \a ring must match
	 * \a InputType2, 3) the third domain of \a ring must match \a OutputType. If
	 * one of these is not true, the code shall not compile.
	 * \endparblock
	 *
	 * \par Performance semantics
	 * Each backend must define performance semantics for this primitive.
	 *
	 * @see perfSemantics
	 */
	template<
		Descriptor descr = descriptors::no_operation,
		class Ring, enum Backend backend,
		typename InputType1, typename InputType2,
		typename OutputType, typename MaskType,
		typename Coords
	>
	RC eWiseMul(
		Vector< OutputType, backend, Coords > &z,
		const Vector< MaskType, backend, Coords > &mask,
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
#ifdef _DEBUG
		std::cout << "in eWiseMul ([T1] <- T2 * T3), masked, base";
#endif
#ifndef NDEBUG
		const bool should_not_call_eWiseMulAMSS_base = false;
		assert( should_not_call_eWiseMulAMSS_base );
#endif
		(void) z;
		(void) mask;
		(void) alpha;
		(void) beta;
		(void) ring;
		(void) phase;
		return UNSUPPORTED;
	}

	/**
	 * Executes an arbitrary element-wise user-defined function \a f on any number
	 * of vectors of equal length.
	 *
	 * \warning This is a relatively advanced function. It is recommended to read
	 *          this specifications and its warnings before using it, or to instead
	 *          exclusively only use the other primitives in \ref BLAS1.
	 *
	 * The vectors touched by \a f can be accessed in a read-only or a read/write
	 * fashion. The function \a f must be parametrised in a global index \em i, and
	 * \a f is only allowed to access elements of the captured vectors <em>on that
	 * specific index</em>.
	 *
	 * \warning Any attempt to access a vector element at a position differing
	 *          from \em i will result in undefined behaviour.
	 *
	 * All vectors captured by \a f must furthermore all be given as additional
	 * (variadic) arguments to this primitive. Captured vectors can only be used
	 * for dereferencing elements at a given position \em i; any other use invokes
	 * undefined behaviour.
	 *
	 * \warning In particular, captured vectors may not be passed to other
	 *          ALP/GraphBLAS primitives \em within \a f.
	 *
	 * This primitive will execute \a f on all indices where the first given such
	 * vector argument has nonzeroes. All other indices \em i will be ignored.
	 *
	 * \warning Therefore, for containers of which \a f references the \em i-th
	 *          element, must indeed have a nonzero at position \em i or otherwise
	 *          undefined behaviour is invoked.
	 *
	 * This primitive hence allows a user to implement any level-1 like BLAS
	 * functionality over any number of input/output vectors, and also allows to
	 * compute multiple level-1 (like) BLAS functionalities as a single pass over
	 * the involved containers.
	 *
	 * \note Since the introduction of the nonblocking backend, rewriting \a f in
	 *       terms of native ALP/GraphBLAS primitives no longer implies performance
	 *       penalties (when compiling for the nonblocking backend)-- rather, the
	 *       nonblocking backend is likely to do better than manually fusing
	 *       multiple level-1 like operations using this primitive, especially when
	 *       the captured vectors are small relative to the private caches on the
	 *       target architecture.
	 *
	 * The function \a f may also capture scalars for read-only access.
	 *
	 * \note As a convention, consider always passing scalars by value, since
	 *       otherwise the compilation of your code with a non-blocking backend
	 *       may (likely) result in data races.
	 *
	 * If #grb::Properties::writableCaptured evaluates <tt>true</tt> then captured
	 * scalars may also safely be written to, instead of requiring to be read-only.
	 *
	 * \note This is useful for fusing reductions within other level-1 like
	 *       operations.
	 *
	 * \warning If updating scalars using this primitive, be aware that the
	 *          updates are local to the current user process only.
	 *
	 * \note If, after execution of this primitive, an updated scalar is expected
	 *       to be synchronised across all user processes, see #grb::collectives.
	 *
	 * \note As a rule of thumb, parallel GraphBLAS implementations, due to being
	 *       data-centric, \em cannot support writeable scalar captures and will
	 *       have #grb::Properties::writableCaptured evaluate to <tt>false</tt>.
	 *
	 * \note A portable ALP/GraphBLAS algorithm should therefore either not rely on
	 *       read/write captured scalars passed to this primitive, \em or provide
	 *       different code paths to handle the two cases of the
	 *       #grb::Properties::writableCaptured backend property.
	 *
	 * \note If the above sounds too tedious, consider rewriting \a f in terms of
	 *       native ALP/GraphBLAS functions, with the scalar reductions performed by
	 *       the scalar variants of #grb::foldl and #grb::foldr, e.g.
	 *
	 * \warning When compiling with a blocking backend, rewriting \a f in terms of
	 *          native GraphBLAS primitives typically results in a slowdown due to
	 *          this primitive naturally fusing potentially multiple operations
	 *          together (which was the original motivation of Yzelman et al., 2020
	 *          for introducing this primitive. Rewriting \a f into a (sequence of)
	 *          native GraphBLAS primtives does \em not carry a performance when
	 *          compiling with a nonblocking backend, however.
	 *
	 * \note This is an addition to the GraphBLAS C specification. It is alike
	 *       user-defined operators, monoids, and semirings, except that this
	 *       primitive allows execution on arbitrarily many inputs and arbitrarily
	 *       many outputs.
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
	 * \note Distributed-memory ALP/GraphBLAS backends, apart from performing
	 *       dimension checking, may also require data redistribution in case that
	 *       different vectors are distributed differently.
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
	 * @return #grb::PANIC    When ALP/GraphBLAS has encountered an unrecoverable
	 *                        error. The state of ALP becomes undefined after
	 *                        having returned this error code, and users can only
	 *                        attempt to exit the application gracefully.
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
	 * If the latter code block is compiled using a blocking ALP/GraphBLAS backend,
	 * the version using the lambdas is expected to execute faster as both \a x and
	 * \a y are streamed only once, while the latter code may stream both vectors
	 * twice. This performance difference disappears when compiling the latter code
	 * block using a nonblocking backend instead.
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
	 * @see Vector::operator[]()
	 * @see Vector::lambda_reference
	 *
	 * \par Performance semantics
	 * Each backend must define performance semantics for this primitive. It is
	 * expected that the defined performance semantics depend on the given lambda
	 * function \a f, the size of the containers passed into this primitive, as
	 * well as how many containers are passed into this primitive.
	 *
	 * @see perfSemantics
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
	 * @see grb::foldr provides similar in-place functionality.
	 * @see grb::eWiseApply provides out-of-place semantics.
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
	 * \par Performance semantics
	 * Each backend must define performance semantics for this primitive.
	 *
	 * @see perfSemantics
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
	 * Unmasked operator variant. See masked variant for the full documentation.
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
	 * Calculates the dot product, \f$ z += (x,y) \f$, under a given additive
	 * monoid and multiplicative operator.
	 *
	 * @tparam descr      The descriptor to be used. Optional; the default
	 *                    descriptors is #grb::descriptors::no_operation.
	 * @tparam AddMonoid  The monoid used for addition during the computation of
	 *                    \f$ (x,y) \f$. The same monoid is used for accumulating
	 *                    the result into a given scalar.
	 * @tparam AnyOp      A binary operator that acts as the multiplication during
	 *                    \f$ (x,y) \f$.
	 * @tparam OutputType The output type.
	 * @tparam InputType1 The input element type of the left-hand input vector.
	 * @tparam InputType2 The input element type of the right-hand input vector.
	 *
	 * @param[in,out]  z    Where to fold \f$ (x,y) \f$ into.
	 * @param[in]      x    The left-hand input vector.
	 * @param[in]      y    The right-hand input vector.
	 * @param[in] addMonoid The additive monoid under which the reduction of the
	 *                      results of element-wise multiplications of \a x and
	 *                      \a y are performed.
	 * @param[in]   anyOp   The multiplicative operator using which element-wise
	 *                      multiplications of \a x and \a y are performed. This
	 *                      may be any binary operator.
	 * @param[in]   phase   The #grb::Phase the call should execute. Optional; the
	 *                      default parameter is #grb::EXECUTE.
	 *
	 * \note By this primitive by which a dot-product operates under any additive
	 *       monoid and any binary operator, it follows that a dot product under
	 *       any semiring can be reduced to a call to this primitive instead.
	 *
	 * @return #grb::MISMATCH When the dimensions of \a x and \a y do not match.
	 *                        All input data containers are left untouched if this
	 *                        exit code is returned; it will be as though this call
	 *                        was never made.
	 * @return #grb::SUCCESS  On successful completion of this call.
	 *
	 * \parblock
	 * \par Valid descriptors
	 *   -# grb::descriptors::no_operation
	 *   -# grb::descriptors::no_casting
	 *   -# grb::descriptors::dense
	 *
	 * If the dense descriptor is set, this implementation returns grb::ILLEGAL if
	 * it was detected that either \a x or \a y was sparse. In this case, it shall
	 * otherwise be as though the call to this function had not occurred (no side
	 * effects).
	 * \endparblock
	 *
	 * \par Performance semantics
	 * Each backend must define performance semantics for this primitive.
	 *
	 * @see perfSemantics
	 */
	template<
		Descriptor descr = descriptors::no_operation,
		class AddMonoid, class AnyOp,
		typename OutputType, typename InputType1, typename InputType2,
		enum Backend backend, typename Coords
	>
	RC dot(
		OutputType &z,
		const Vector< InputType1, backend, Coords > &x,
		const Vector< InputType2, backend, Coords > &y,
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
#ifdef _DEBUG
		std::cout << "Should not call base grb::dot (monoid-operator version)\n";
#endif
#ifndef NDEBUG
		const bool should_not_call_base_dot_monOp = false;
		assert( should_not_call_base_dot_monOp );
#endif
		(void) z;
		(void) x;
		(void) y;
		(void) addMonoid;
		(void) anyOp;
		(void) phase;
		return UNSUPPORTED;
	}

	/**
	 * Calculates the dot product, \f$ z += (x,y) \f$, under a given semiring.
	 *
	 * @tparam descr      The descriptor to be used. Optional; default descriptor
	 *                    is #grb::descriptors::no_operation.
	 * @tparam Ring       The semiring type to use.
	 * @tparam OutputType The output type.
	 * @tparam InputType1 The input element type of the left-hand input vector.
	 * @tparam InputType2 The input element type of the right-hand input vector.
	 *
	 * @param[in,out] z The output element \f$ z += (x,y) \f$.
	 * @param[in]     x The left-hand input vector \a x.
	 * @param[in]     y The right-hand input vector \a y.
	 * @param[in]  ring The semiring under which to compute the dot product
	 *                  \f$ (x,y) \f$. The additive monoid is used to accumulate
	 *                  the dot product result into \a z.
	 * @param[in] phase The #grb::Phase the call should execute. Optional; the
	 *                  default parameter is #grb::EXECUTE.
	 *
	 * @return #grb::SUCCESS  On successful completion of this call.
	 * @return #grb::MISMATCH If the dimensions of \a x and \a y do not match. All
	 *                        input data containers are left untouched if this exit
	 *                        code is returned; it will be as though this call was
	 *                        never made.
	 *
	 * \parblock
	 * \par Valid descriptors
	 *   - grb::descriptors::no_operation
	 *   - grb::descriptors::no_casting
	 *   - grb::descriptors::dense
	 *
	 * If the dense descriptor is set, this implementation returns #grb::ILLEGAL if
	 * it was detected that either \a x or \a y was sparse. In this case, it shall
	 * otherwise be as though the call to this function had not occurred (no side
	 * effects).
	 * \endparblock
	 *
	 * \par Performance semantics
	 * Each backend must define performance semantics for this primitive.
	 *
	 * @see perfSemantics
	 */
	template<
		Descriptor descr = descriptors::no_operation,
		class Ring,
		typename IOType, typename InputType1, typename InputType2,
		Backend backend, typename Coords
	>
	RC dot(
		IOType &z,
		const Vector< InputType1, backend, Coords > &x,
		const Vector< InputType2, backend, Coords > &y,
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
		(void) z;
		(void) x;
		(void) y;
		(void) ring;
		(void) phase;
		return UNSUPPORTED;
	}

	/** @} */

} // end namespace grb

#endif // end _H_GRB_BASE_BLAS1

