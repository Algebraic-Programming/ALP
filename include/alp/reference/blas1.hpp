
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
 * @date 14th of January 2022
 */

#ifndef _H_ALP_REFERENCE_BLAS1
#define _H_ALP_REFERENCE_BLAS1

#include <functional>
#include <alp/backends.hpp>
#include <alp/config.hpp>
#include <alp/rc.hpp>
#include <alp/density.hpp>
#include "scalar.hpp"
#include "matrix.hpp"
#include "vector.hpp"
#include "blas0.hpp"
#include "blas2.hpp"

#ifndef NO_CAST_ASSERT
#define NO_CAST_ASSERT( x, y, z )                                              \
	static_assert( x,                                                          \
		"\n\n"                                                                 \
		"********************************************************************" \
		"********************************************************************" \
		"******************************\n"                                     \
		"*     ERROR      | " y " " z ".\n"                                    \
		"********************************************************************" \
		"********************************************************************" \
		"******************************\n"                                     \
		"* Possible fix 1 | Remove no_casting from the template parameters "   \
		"in this call to " y ".\n"                                             \
		"* Possible fix 2 | Provide a value that matches the expected type.\n" \
		"********************************************************************" \
		"********************************************************************" \
		"******************************\n" );
#endif

#define NO_CAST_OP_ASSERT( x, y, z )                                           \
	static_assert( x,                                                          \
		"\n\n"                                                                 \
		"********************************************************************" \
		"********************************************************************" \
		"******************************\n"                                     \
		"*     ERROR      | " y " " z ".\n"                                    \
		"********************************************************************" \
		"********************************************************************" \
		"******************************\n"                                     \
		"* Possible fix 1 | Remove no_casting from the template parameters "   \
		"in this call to " y ".\n"                                             \
		"* Possible fix 2 | For all mismatches in the domains of input "       \
		"parameters and the operator domains, as specified in the "            \
		"documentation of the function " y ", supply an input argument of "    \
		"the expected type instead.\n"                                         \
		"* Possible fix 3 | Provide a compatible operator where all domains "  \
		"match those of the input parameters, as specified in the "            \
		"documentation of the function " y ".\n"                               \
		"********************************************************************" \
		"********************************************************************" \
		"******************************\n" );


namespace alp {

	/**
	 * \defgroup BLAS1 The Level-1 Basic Linear Algebra Subroutines (BLAS)
	 *
	 * A collection of functions that allow ALP/GraphBLAS operators, monoids, and
	 * semirings work on a mix of zero-dimensional and one-dimensional containers;
	 * i.e., allows various linear algebra operations on scalars (both built-in
	 * C++ scalars and objects of type alp::Scalar) and objects of type
	 * alp::Vector.
	 *
	 * C++ built-in scalars are all scalar types that can be
	 * passed to BLAS functions. This includes simple types (e.g. double) and
	 * more complex types (e.g. std::pair as complex number representation).
	 * Such types are referred to as C++ scalars or built-in scalars.
	 *
	 * Operations producing scalars are specialized to both C++ built-in scalars
	 * and alp::Scalars. Functions taking scalars as inputs are specialized only
	 * to alp::Scalars. Depending on backend's Scalar implementation, the
	 * conversion from C++ scalar to alp::Scalar can be implicit or explicit.
	 *
	 * All functions except for alp::size and alp::nnz return an error code of
	 * the enum-type alp::RC. The two functions for retrieving the size and the
	 * nonzeroes of two vectors are excluded from this because they are never
	 * allowed to fail.
	 *
	 * Operations which require a single input vector only and produce scalar
	 * output:
	 *   -# alp::size,
	 *   -# alp::nnz, and
	 *   -# alp::set (three variants).
	 * These do not require an operator, monoid, nor semiring. The following
	 * require an operator:
	 *   -# alp::foldr (reduction to the right),
	 *   -# alp::foldl (reduction to the left).
	 * Operators can only be applied on \em dense vectors. Operations on sparse
	 * vectors require a well-defined way to handle missing vector elements. The
	 * following functions require a monoid instead of an operator and are able
	 * to handle sparse vectors by interpreting missing items as an identity
	 * value:
	 *   -# alp::reducer (reduction to the right),
	 *   -# alp::reducel (reduction to the left).
	 *
	 * Operations which require two input vectors and produce scalar output:
	 *   -# alp::dot   (dot product-- requires a semiring).
	//  * Sparse vectors under a semiring have their missing values interpreted as a
	//  * zero element under the given semiring; i.e., the identity of the additive
	//  * operator.
	 *
	 * Operations which require one input vector and one input/output vector for
	 * full and efficient in-place operations:
	 *   -# alp::foldr (reduction to the right-- requires an operator),
	 *   -# alp::foldl (reduction to the left-- requires an operator).
	 * For alp::foldr, the left-hand side input vector may be replaced by an
	 * input scalar. For alp::foldl, the right-hand side input vector may be
	 * replaced by an input scalar. In either of those cases, the reduction
	 * is equivalent to an in-place vector scaling.
	 *
	 * Operations which require two input vectors and one output vector for
	 * out-of-place operations:
	 *   -# alp::eWiseApply (requires an operator),
	 *   -# alp::eWiseMul   (requires a semiring),
	 *   -# alp::eWiseAdd   (requires a semiring).
	 * Note that multiplication will consider any zero elements as an annihilator
	 * to the multiplicative operator. Therefore, the operator will only be
	 * applied at vector indices where both input vectors have nonzeroes. This is
	 * different from eWiseAdd. This difference only manifests itself when dealing
	 * with semirings, and reflects the intuitively expected behaviour. Any of the
	 * two input vectors (or both) may be replaced with an input scalar instead.
	 *
	 * Operations which require three input vectors and one output vector for
	 * out-of-place operations:
	 *   -# alp::eWiseMulAdd (requires a semiring).
	 * This function can be emulated by first successive calls to alp::eWiseMul
	 * and alp::eWiseAdd. This specialised function, however, has better
	 * performance semantics. This function is closest to the standard axpy
	 * BLAS1 call, with out-of-place semantics. The first input vector may be
	 * replaced by a scalar.
	 *
	 * Again, each of alp::eWiseMul, alp::eWiseAdd, alp::eWiseMulAdd accept sparse
	 * vectors as input and output (since they operate on semirings), while
	 * alp::eWiseApply.
	 *
	 * For fusing multiple BLAS-1 style operations on any number of inputs and
	 * outputs, users can pass their own operator function to be executed for
	 * every index \a i.
	 *   -# alp::eWiseLambda.
	 * This requires manual application of operators, monoids, and/or semirings
	 * via the BLAS-0 interface (see alp::apply, alp::foldl, and alp::foldr).
	 *
	 * For all of these functions, the element types of input and output types
	 * do not have to match the domains of the given operator, monoid, or
	 * semiring unless the alp::descriptors::no_casting descriptor was passed.
	 *
	 * An implementation, whether blocking or non-blocking, should have clear
	 * performance semantics for every sequence of graphBLAS calls, no matter
	 * whether those are made from sequential or parallel contexts.
	 *
	 * @{
	 */

	/**
	 * Clears all elements from the given vector \a x.
	 *
	 * At the end of this operation, the number of nonzero elements in this vector
	 * will be zero. The size of the vector remains unchanged.
	 *
	 * @return alp::SUCCESS When the vector is successfully cleared.
	 *
	 * \note This function cannot fail.
	 *
	//  * \parblock
	//  * \par Performance semantics
	//  *      This function
	//  *        -# contains \f$ \mathcal{O}(n) \f$ work,
	//  *        -# will not allocate new dynamic memory,
	//  *        -# will take at most \f$ \Theta(1) \f$ memory beyond the memory
	//  *           already used by the application before the call to this
	//  *           function.
	//  *        -# will move at most \f$ \mathit{sizeof}(\mathit{bool}) +
	//  *           \mathit{sizeof}(\mathit{size\_t}) \f$ bytes of data.
	//  * \endparblock
	 */
	template<
		typename DataType, typename DataStructure, typename View, typename ImfR, typename ImfC
	>
	RC clear(
		Vector< DataType, DataStructure, Density::Dense, View, ImfR, ImfC, reference > & x
	) noexcept {
		throw std::runtime_error( "Needs an implementation" );
		return SUCCESS;
	}

	/**
	 * Request the size (dimension) of a given Vector.
	 *
	 * The dimension is set at construction of the given Vector and cannot
	 * be changed. A call to this function shall always succeed.
	 *
	 * @tparam DataType      The type of elements contained in the vector \a x.
	 * @tparam DataStructure The structure of the vector \a x.
	 * @tparam View          The view type applied to the vector \a x.
	 *
	 * @param[in] x The Vector of which to retrieve the size.
	 *
	 * @return The size of the Vector \a x.
	 *
	//  * \parblock
	//  * \par Performance semantics
	//  * A call to this function
	//  *  -# consists of \f$ \Theta(1) \f$ work;
	//  *  -# moves \f$ \Theta(1) \f$ bytes of memory;
	//  *  -# does not allocate any dynamic memory;
	//  *  -# shall not make any system calls.
	//  * \endparblock
	 */
	template< typename DataType, typename DataStructure, typename View, typename ImfR, typename ImfC >
	size_t size( const Vector< DataType, DataStructure, Density::Dense, View, ImfR, ImfC, reference > & x ) noexcept {
		return getLength( x );
	}

	/**
	 * Request the number of nonzeroes in a given Vector.
	 *
	 * A call to this function always succeeds.
	 *
	 * @tparam DataType      The type of elements contained in the vector \a x.
	 * @tparam DataStructure The structure of the vector \a x.
	 * @tparam View          The view type applied to the vector \a x.
	 *
	 * @param[in] x The Vector of which to retrieve the number of nonzeroes.
	 *
	 * @return The number of nonzeroes in \a x.
	 *
	//  * \parblock
	//  * \par Performance semantics
	//  * A call to this function
	//  *   -# consists of \f$ \Theta(1) \f$ work;
	//  *   -# moves \f$ \Theta(1) \f$ bytes of memory;
	//  *   -# does not allocate nor free any dynamic memory;
	//  *   -# shall not make any system calls.
	//  * \endparblock
	 */
	template< typename DataType, typename DataStructure, typename View, typename ImfR, typename ImfC >
	size_t nnz( const Vector< DataType, DataStructure, Density::Dense, View, ImfR, ImfC, reference > & x ) noexcept {
		throw std::runtime_error( "Needs an implementation." );
		return 0;
	}

	/** Resizes the vector to have at least the given number of nonzeroes.
	 * The contents of the vector are not retained.
	 *
	 * Resizing of dense containers is not allowed as the capacity is determined
	 * by the container dimensions and the storage scheme. Therefore, this
	 * function will not change the capacity of the vector.
	 *
	 * Even though the capacity remains unchanged, the contents of the vector
	 * are not retained to maintain compatibility with the general specification.
	 * However, the actual memory will not be reallocated. Rather, the vector
	 * will be marked as uninitialized.
	 *
	 * @param[in] x      The Vector to be resized.
	 * @param[in] new_nz The number of nonzeroes this vector is to contain.
	 *
	 * @return SUCCESS   If \a new_nz is not larger than the current capacity
	 *                   of the vector.
	 *         ILLEGAL   If \a new_nz is larger than the current capacity of
	 *                   the vector.
	 *
	 * \parblock
	 * \par Performance semantics.
	 *        -$ This function consitutes \f$ \Theta(1) \f$ work.
	 *        -# This function allocates \f$ \Theta(0) \f$
	 *           bytes of dynamic memory.
	 *        -# This function does not make system calls.
	 * \endparblock
	 * \todo add documentation. In particular, think about the meaning with \a P > 1.
	 */
	template< typename InputType, typename InputStructure, typename View, typename ImfR, typename ImfC, typename length_type >
	RC resize( Vector< InputType, InputStructure, Density::Dense, View, ImfR, ImfC, reference > &x, const length_type new_nz ) {
		(void)x;
		(void)new_nz;
		// TODO implement
		// setInitialized( x, false );
		return PANIC;
	}

	/**
	 * Sets all elements of a Vector to the given value. Can be masked.
	 *
	 * This function is functionally equivalent to
	 * \code
	 * alp::operators::right_assign< DataType > op;
	 * return foldl< descr >( x, val, op );
	 * \endcode,
	 * \code
	 * alp::operators::left_assign< DataType > op;
	 * return foldr< descr >( val, x, op );
	 * \endcode, and the following pseudocode
	 * \code
	 * for( size_t i = 0; i < size(x); ++i ) {
	 *     if( mask(i) ) { setElement( x, i, val ); }
	 * \endcode.
	 *
	 * @tparam descr         The descriptor used for this operation.
	 * @tparam DataType      The type of each element in the vector \a x.
	 * @tparam DataStructure The structure of the vector \a x.
	 * @tparam View          The view type applied to the vector \a x.
	 * @tparam T             The type of the given value.
	 *
	 * \parblock
	 * \par Accepted descriptors
	 *   -# alp::descriptors::no_operation
	 *   -# alp::descriptors::no_casting
	 * \endparblock
	 *
	 * @param[in,out] x The Vector of which every element is to be set to equal
	 *                  \a val.
	 * @param[in]   val The value to set each element of \a x equal to.
	 *
	 * @returns SUCCESS       When the call completes successfully.
	 *
	 * When \a descr includes alp::descriptors::no_casting and if \a T does not
	 * match \a DataType, the code shall not compile.
	 *
	//  * \parblock
	//  * \par Performance semantics
	//  * A call to this function
	//  *   -# consists of \f$ \Theta(n) \f$ work;
	//  *   -# moves \f$ \Theta(n) \f$ bytes of memory;
	//  *   -# does not allocate nor free any dynamic memory;
	//  *   -# shall not make any system calls.
	//  * \endparblock
	 *
	 * @see alp::foldl.
	 * @see alp::foldr.
	 * @see alp::operators::left_assign.
	 * @see alp::operators::right_assign.
	 * @see alp::setElement.
	 */
	template<
		Descriptor descr = descriptors::no_operation,
		typename DataType, typename DataStructure, typename View,
		typename ImfR, typename ImfC,
		typename T, typename ValStructure
	>
	RC set(
		Vector< DataType, DataStructure, Density::Dense, View, ImfR, ImfC, reference > & x,
		const Scalar< T, ValStructure, reference > val,
		const typename std::enable_if<
			!alp::is_object< DataType >::value &&
			!alp::is_object< T >::value,
		void >::type * const = NULL
	) {
		// static sanity checks
		NO_CAST_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< DataType, T >::value ), "alp::set (Vector, unmasked)",
			"called with a value type that does not match that of the given "
			"vector" );

		if( ! internal::getInitialized( val ) ) {
			internal::setInitialized( x, false );
			return SUCCESS;
		}

		// foldl requires left-hand side to be initialized prior to the call
		internal::setInitialized( x, true );
		return foldl( x, val, alp::operators::right_assign< DataType >() );
	}

	/**
	 * Sets the element of a given Vector at a given position to a given value.
	 *
	 * If the input Vector \a x already has an element \f$ x_i \f$, that element
	 * is overwritten to the given value \a val. If no such element existed, it
	 * is added and set equal to \a val. The number of nonzeroes in \a x may thus
	 * be increased by one due to a call to this function.
	 *
	 * The parameter \a i may not be greater or equal than the size of \a x.
	 *
	 * @tparam descr         The descriptor to be used during evaluation of this
	 *                       function.
	 * @tparam DataType      The type of the elements of \a x.
	 * @tparam DataStructure The structure of the vector \a x.
	 * @tparam View          The view type applied to the vector \a x.
	 * @tparam T             The type of the value to be set.
	 *
	 * @param[in,out] x The vector to be modified.
	 * @param[in]   val The value \f$ x_i \f$ should read after function exit.
	 * @param[in]     i The index of the element of \a x to set.
	 *
	 * @return alp::SUCCESS   Upon successful execution of this operation.
	 * @return alp::MISMATCH  If \a i is greater or equal than the dimension of
	 *                        \a x.
	 *
	 * \parblock
	 * \par Accepted descriptors
	 *   -# alp::descriptors::no_operation
	 *   -# alp::descriptors::no_casting
	 * \endparblock
	 *
	 * When \a descr includes alp::descriptors::no_casting and if \a T does not
	 * match \a DataType, the code shall not compile.
	 *
	//  * \parblock
	//  * \par Performance semantics
	//  * A call to this function
	//  *   -# consists of \f$ \Theta(1) \f$ work;
	//  *   -# moves \f$ \Theta(1) \f$ bytes of memory;
	//  *   -# does not allocate nor free any dynamic memory;
	//  *   -# shall not make any system calls.
	//  * \endparblock
	 */
	template< Descriptor descr = descriptors::no_operation,
		typename DataType, typename DataStructure, typename View, typename ImfR, typename ImfC, typename ValStructure,
		typename T
	>
	RC setElement( Vector< DataType, DataStructure, Density::Dense, View, ImfR, ImfC, reference > & x,
		const Scalar< T, ValStructure, reference > val,
		const size_t i,
		const typename std::enable_if< ! alp::is_object< DataType >::value && ! alp::is_object< T >::value, void >::type * const = NULL ) {
		// static sanity checks
		NO_CAST_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< DataType, T >::value ), "alp::set (Vector, at index)",
			"called with a value type that does not match that of the given "
			"Vector" );

		throw std::runtime_error( "Needs an implementation." );

		// done
		return SUCCESS;
	}

	/** C++ scalar variant */
	template< Descriptor descr = descriptors::no_operation,
		typename DataType, typename DataStructure, typename View, typename ImfR, typename ImfC,
		typename T
	>
	RC setElement( Vector< DataType, DataStructure, Density::Dense, View, ImfR, ImfC, reference > & x,
		const T val,
		const size_t i,
		const typename std::enable_if< ! alp::is_object< DataType >::value && ! alp::is_object< T >::value, void >::type * const = NULL ) {
		// static sanity checks
		NO_CAST_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< DataType, T >::value ), "alp::set (Vector, at index)",
			"called with a value type that does not match that of the given "
			"Vector" );

		// delegate
		return setElement( x, Scalar< T >( val ), i );
	}

	/**
	 * Sets the content of a given vector \a x to be equal to that of
	 * another given vector \a y. Can be masked.
	 *
	 * This operation is functionally equivalent to
	 * \code
	 * alp::operators::right_assign< T > op;
	 * alp::foldl( x, y, op );
	 * \endcode,
	 * \code
	 * alp::operators::left_assign < T > op;
	 * alp::foldr( y, x, op );
	 * \endcode, as well as the following pseudocode
	 * \code
	 * for( each nonzero in y ) {
	 *    setElement( x, nonzero.index, nonzero.value );
	 * }
	 * \endcode.
	 *
	 * The vector \a x may not equal \a y.
	 *
	 * \parblock
	 * \par Accepted descriptors
	 *   -# alp::descriptors::no_operation
	 *   -# alp::descriptors::no_casting
	 * \endparblock
	 *
	 * @tparam descr           The descriptor of the operation.
	 * @tparam OutputType      The type of each element in the output vector.
	 * @tparam InputType       The type of each element in the input vector.
	 * @tparam OutputStructure The structure of the ouput vector.
	 * @tparam InputStructure  The structure of the input vector.
	 * @tparam OuputView       The view applied to the output vector.
	 * @tparam InputView       The view applied to the input vector.
	 *
	 * @param[in,out] x The vector to be set.
	 * @param[in]     y The source vector.
	 *
	 * When \a descr includes alp::descriptors::no_casting and if \a InputType
	 * does not match \a OutputType, the code shall not compile.
	 *
	//  * \parblock
	//  * \par Performance semantics
	//  * A call to this function
	//  *   -# consists of \f$ \Theta(n) \f$ work;
	//  *   -# moves \f$ \Theta(n) \f$ bytes of memory;
	//  *   -# does not allocate nor free any dynamic memory;
	//  *   -# shall not make any system calls.
	//  * \endparblock
	 *
	 * @see alp::foldl.
	 * @see alp::foldr.
	 * @see alp::operators::left_assign.
	 * @see alp::operators::right_assign.
	 * @see alp::setElement.
	 */
	template< Descriptor descr = descriptors::no_operation,
		typename OutputType, typename OutputStructure, typename OutputView, typename OutputImfR, typename OutputImfC,
		typename InputType, typename InputStructure, typename InputView, typename InputImfR, typename InputImfC
	>
	RC set(
		Vector< OutputType, OutputStructure, Density::Dense, OutputView, OutputImfR, OutputImfC, reference > & x,
		const Vector< InputType, InputStructure, Density::Dense, InputView, InputImfR, InputImfC, reference > & y
	) {
		// static sanity checks
		NO_CAST_ASSERT(
			( ! ( descr & descriptors::no_casting ) || std::is_same< OutputType, InputType >::value ), "alp::copy (Vector)", "called with vector parameters whose element data types do not match" );
		constexpr bool out_is_void = std::is_void< OutputType >::value;
		constexpr bool in_is_void = std::is_void< OutputType >::value;
		static_assert( ! in_is_void || out_is_void,
			"alp::set (reference, Vector <- Vector, masked): "
			"if input is void, then the output must be also" );
		static_assert( ! ( descr & descriptors::use_index ) || ! out_is_void,
			"alp::set (reference, Vector <- Vector, masked): "
			"use_index descriptor cannot be set if output vector is void" );

		// check contract
		if( reinterpret_cast< void * >( &x ) == reinterpret_cast< const void * >( &y ) ) {
			return ILLEGAL;
		}

		if( getLength( x ) != getLength( y ) ) {
			return MISMATCH;
		}

		if( !internal::getInitialized( y ) ) {
			setInitialized( x, false );
			return SUCCESS;
		}

		internal::setInitialized( x, true );
		return foldl( x, y, alp::operators::right_assign< OutputType >() );
	}

	/**
	 * Folds all elements in a ALP Vector \a x into a single value \a beta.
	 *
	 * The original value of \a beta is used as the right-hand side input of the
	 * operator \a op. A left-hand side input for \a op is retrieved from the
	 * input Vector \a x. The result of the operation is stored in \a beta.
	 * This process is repeated for every element in \a x.
	 *
	 * At function exit, \a beta will equal
	 * \f$ \beta \odot x_0 \odot x_1 \odot \ldots x_{n-1} \f$.
	 *
	 * @tparam descr     The descriptor used for evaluating this function. By
	 *                   default, this is alp::descriptors::no_operation.
	 * @tparam OP        The type of the operator to be applied.
	 *                   The operator must be associative.
	 * @tparam InputType The type of the elements of \a x.
	 * @tparam IOType    The type of the value \a y.
	 * @tparam InputStructure The structure of the vector \a x.
	 * @tparam InputView      The view type applied to the vector \a x.
	 *
	 * @param[in]     x    The input Vector \a x that will not be modified.
	 *                     This input Vector must be dense.
	 * @param[in,out] beta On function entry: the initial value to be applied to
	 *                     \a op from the right-hand side.
	 *                     On function exit: the result of repeated applications
	 *                     from the left-hand side of elements of \a x.
	 * @param[in]    op    The monoid under which to perform this right-folding.
	 *
	 * \note We only define fold under monoids, not under plain operators.
	 *
	 * @returns alp::SUCCESS This function always succeeds.
	 * @returns alp::ILLEGAL When a sparse Vector is passed. In this case, the
	 *                       call to this function will have no other effects.
	 *
	 * \warning Since this function folds from left-to-right using binary
	 *          operators, this function \em cannot take sparse vectors as input--
	 *          a monoid is required to give meaning to missing vector entries.
	 *          See alp::reducer for use with sparse vectors instead.
	 *
	 * \parblock
	 * \par Valid descriptors
	 * alp::descriptors::no_operation, alp::descriptors::no_casting.
	 *
	 * \note Invalid descriptors will be ignored.
	 *
	 * If alp::descriptors::no_casting is specified, then 1) the first domain of
	 * \a op must match \a IOType, 2) the second domain of \a op must match
	 * \a InputType, and 3) the third domain must match \a IOType. If one of these
	 * is not true, the code shall not compile.
	 *
	 * \endparblock
	 *
	 * \parblock
	 * \par Valid operator types
	 * The given operator \a op is required to be:
	 *   -# associative.
	 * \endparblock
	 *
	//  * \parblock
	//  * \par Performance semantics
	//  *      -# This call comprises \f$ \Theta(n) \f$ work, where \f$ n \f$ equals
	//  *         the size of the vector \a x. The constant factor depends on the
	//  *         cost of evaluating the underlying binary operator. A good
	//  *         implementation uses vectorised instructions whenever the input
	//  *         domains, the output domain, and the operator used allow for this.
	//  *
	//  *      -# This call will not result in additional dynamic memory allocations.
	//  *
	//  *      -# This call takes \f$ \mathcal{O}(1) \f$ memory beyond the memory
	//  *         used by the application at the point of a call to this function.
	//  *
	//  *      -# This call incurs at most
	//  *         \f$ n \cdot\mathit{sizeof}(\mathit{InputType}) + \mathcal{O}(1) \f$
	//  *         bytes of data movement. A good implementation will rely on in-place
	//  *         operators.
	//  * \endparblock
	 *
	 * @see alp::operators::internal::Operator for a discussion on when in-place
	 *      and/or vectorised operations are used.
	 */
	template< Descriptor descr = descriptors::no_operation,
		typename InputType, typename InputStructure, typename InputView, typename InputImfR, typename InputImfC,
		typename IOType, typename IOStructure,
		class Monoid
	>
	RC foldr(
		const Vector< InputType, InputStructure, Density::Dense, InputView, InputImfR, InputImfC, reference > &x,
		Scalar< IOType, IOStructure, reference > &beta,
		const Monoid &monoid = Monoid(),
		const std::enable_if_t<
			!alp::is_object< InputType >::value && ! alp::is_object< IOType >::value && alp::is_monoid< Monoid >::value
		> * const = nullptr
	) {

#ifdef _DEBUG
		std::cout << "foldr(Vector,Scalar,Monoid) called. Vector has size " << getLength( x ) << " .\n";
#endif

		internal::setInitialized(
			beta,
			internal::getInitialized( beta ) && internal::getInitialized( x )
		);

		if( !internal::getInitialized( beta ) ) {
			return SUCCESS;
		}

		const size_t n = getLength( x );
		for ( size_t i = 0; i < n; ++i ) {
			(void) internal::foldr( x[ i ], *beta, monoid.getOperator() );
		}
		return SUCCESS;
	}

	/** C++ scalar variant */
	template< Descriptor descr = descriptors::no_operation,
		typename InputType, typename InputStructure, typename InputView, typename InputImfR, typename InputImfC,
		typename IOType,
		class Monoid
	>
	RC foldr(
		const Vector< InputType, InputStructure, Density::Dense, InputView, InputImfR, InputImfC, reference > &x,
		IOType &beta,
		const Monoid & monoid = Monoid(),
		const std::enable_if_t<
			!alp::is_object< InputType >::value && ! alp::is_object< IOType >::value && alp::is_monoid< Monoid >::value
		> * const = nullptr
	) {
		return foldr( x, Scalar< IOType >( beta ), monoid );
	}

	/**
	 * For all elements in a ALP Vector \a y, fold the value \f$ \alpha \f$
	 * into each element.
	 *
	 * The original value of \f$ \alpha \f$ is used as the left-hand side input
	 * of the operator \a op. The right-hand side inputs for \a op are retrieved
	 * from the input vector \a y. The result of the operation is stored in \a y,
	 * thus overwriting its previous values.
	 *
	 * The value of \f$ y_i \f$ after a call to thus function thus equals
	 * \f$ \alpha \odot y_i \f$, for all \f$ i \in \{ 0, 1, \dots, n - 1 \} \f$.
	 *
	 * @tparam descr         The descriptor used for evaluating this function.
	 *                       By default, this is alp::descriptors::no_operation.
	 * @tparam OP            The type of the operator to be applied.
	 * @tparam InputType     The type of \a alpha.
	 * @tparam IOType        The type of the elements in \a y.
	 * @tparam IOStructure   The structure of the vector \a y.
	 * @tparam IOView        The view applied to the vector \a y.
	 *
	 * @param[in]     alpha The input value to apply as the left-hand side input
	 *                      to \a op.
	 * @param[in,out] y     On function entry: the initial values to be applied as
	 *                      the right-hand side input to \a op.
	 *                      On function exit: the output data.
	 * @param[in]     op    The monoid under which to perform this left-folding.
	 *
	 * @returns alp::SUCCESS This function always succeeds.
	 *
	 * \note We only define fold under monoids, not under plain operators.
	 *
	 * \parblock
	 * \par Valid descriptors
	 * alp::descriptors::no_operation, alp::descriptors::no_casting.
	 *
	 * \note Invalid descriptors will be ignored.
	 *
	 * If alp::descriptors::no_casting is specified, then 1) the first domain of
	 * \a op must match \a IOType, 2) the second domain of \a op must match
	 * \a InputType, and 3) the third domain must match \a IOType. If one of these
	 * is not true, the code shall not compile.
	 *
	 * \endparblock
	 *
	 * \parblock
	 * \par Valid operator types
	 * The given operator \a op is required to be:
	 *   -# (no requirements).
	 * \endparblock
	 *
	//  * \parblock
	//  * \par Performance semantics
	//  *      -# This call comprises \f$ \Theta(n) \f$ work, where \f$ n \f$ equals
	//  *         the size of the vector \a x. The constant factor depends on the
	//  *         cost of evaluating the underlying binary operator. A good
	//  *         implementation uses vectorised instructions whenever the input
	//  *         domains, the output domain, and the operator used allow for this.
	//  *
	//  *      -# This call will not result in additional dynamic memory allocations.
	//  *
	//  *      -# This call takes \f$ \mathcal{O}(1) \f$ memory beyond the memory
	//  *         used by the application at the point of a call to this function.
	//  *
	//  *      -# This call incurs at most
	//  *         \f$ 2n \cdot \mathit{sizeof}(\mathit{IOType}) + \mathcal{O}(1) \f$
	//  *         bytes of data movement.
	//  * \endparblock
	 *
	 * @see alp::operators::internal::Operator for a discussion on when in-place
	 *      and/or vectorised operations are used.
	 */
	template< Descriptor descr = descriptors::no_operation,
		typename InputType, typename InputStructure,
		typename IOType, typename IOStructure, typename IOView, typename IOImfR, typename IOImfC,
		class Monoid
	>
	RC foldr(
		const Scalar< InputType, InputStructure, reference > &alpha,
		Vector< IOType, IOStructure, Density::Dense, IOView, IOImfR, IOImfC, reference > &y,
		const Monoid & monoid = Monoid(),
		const std::enable_if_t<
			!alp::is_object< InputType >::value && ! alp::is_object< IOType >::value && alp::is_monoid< Monoid >::value
		> * const = nullptr
	) {
		// static sanity checks
		NO_CAST_OP_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename Monoid::D1, IOType >::value ), "alp::foldl",
			"called with a vector y of a type that does not match the first domain "
			"of the given operator" );
		NO_CAST_OP_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename Monoid::D2, InputType >::value ), "alp::foldl",
			"called on a vector y of a type that does not match the second domain "
			"of the given operator" );
		NO_CAST_OP_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename Monoid::D3, IOType >::value ), "alp::foldl",
			"called on a vector y of a type that does not match the third domain "
			"of the given operator" );

#ifdef _DEBUG
		std::cout << "foldr(Scalar,Vector,Monoid) called. Vector has size " << getLength( y ) << " .\n";
#endif
		internal::setInitialized(
			y,
			internal::getInitialized( alpha ) && internal::getInitialized( y )
		);

		if( !internal::getInitialized( y ) ) {
			return SUCCESS;
		}

		const size_t n = getLength( y );
		for ( size_t i = 0; i < n; ++i ) {
			(void) internal::foldr( *alpha, y[ i ], monoid.getOperator() );
		}
		return SUCCESS;
	}

	/**
	 * Computes y = x + y, operator variant.
	 *
	 * Specialisation for scalar \a x.
	 */
	template< Descriptor descr = descriptors::no_operation,
		typename InputType, typename InputStructure,
		typename IOType, typename IOStructure, typename IOView, typename IOImfR, typename IOImfC,
		class OP
	>
	RC foldr(
		const Scalar< InputType, InputStructure, reference > &alpha,
		Vector< IOType, IOStructure, Density::Dense, IOView, IOImfR, IOImfC, reference > &y,
		const OP & op = OP(),
		const std::enable_if_t<
			!alp::is_object< InputType >::value && ! alp::is_object< IOType >::value && alp::is_operator< OP >::value
		> * const = nullptr
	) {
		// static sanity checks
		NO_CAST_OP_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename OP::D1, IOType >::value ), "alp::foldr",
			"called with a vector y of a type that does not match the first domain "
			"of the given operator" );
		NO_CAST_OP_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename OP::D2, InputType >::value ), "alp::foldr",
			"called on a vector y of a type that does not match the second domain "
			"of the given operator" );
		NO_CAST_OP_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename OP::D3, IOType >::value ), "alp::foldr",
			"called on a vector y of a type that does not match the third domain "
			"of the given operator" );

#ifdef _DEBUG
		std::cout << "foldr(Scalar,Vector,OP) called. Vector has size " << getLength( y ) << " .\n";
#endif

		internal::setInitialized(
			y,
			internal::getInitialized( alpha ) && internal::getInitialized( y )
		);

		if( !internal::getInitialized( y ) ) {
			return SUCCESS;
		}

		const size_t n = getLength( y );
		for ( size_t i = 0; i < n; ++i ) {
			(void) internal::foldr( *alpha, y[ i ], op );
		}
		return SUCCESS;

	}

	/**
	 * Folds all elements in a ALP Vector \a x into the corresponding
	 * elements from an input/output vector \a y. The vectors must be of equal
	 * size \f$ n \f$. For all \f$ i \in \{0,1,\ldots,n-1\} \f$, the new value
	 * of at the i-th index of \a y after a call to this function thus equals
	 * \f$ x_i \odot y_i \f$.
	 *
	 * @tparam descr     The descriptor used for evaluating this function. By
	 *                   default, this is alp::descriptors::no_operation.
	 * @tparam OP        The type of the operator to be applied.
	 * @tparam IOType         The type of the elements of \a y.
	 * @tparam InputType      The type of the elements of \a x.
	 * @tparam IOStructure    The structure of the vector \a y.
	 * @tparam InputStructure The structure of the vector \a x.
	 * @tparam IOView         The View applied on the vector \a y.
	 * @tparam InputView      The View applied on the vector \a x.
	 *
	 * @param[in]     x  The input vector \a y that will not be modified.
	 * @param[in,out] y  On function entry: the initial value to be applied to
	 *                   \a op as the right-hand side input.
	 *                   On function exit: the result of repeated applications
	 *                   from the right-hand side using elements from \a y.
	 * @param[in]     op The operator under which to perform this right-folding.
	 *
	 * @returns alp::SUCCESS This function always succeeds.
	 *
	 * \note The element-wise fold is also defined for monoids.
	 *
	 * \parblock
	 * \par Valid descriptors
	 * alp::descriptors::no_operation, alp::descriptors::no_casting.
	 *
	 * \note Invalid descriptors will be ignored.
	 *
	 * If alp::descriptors::no_casting is specified, then 1) the first domain of
	 * \a op must match \a InputType, 2) the second domain of \a op must match
	 * \a IOType, and 3) the third domain must match \a IOType. If one of these
	 * is not true, the code shall not compile.
	 *
	 * \endparblock
	 *
	 * \parblock
	 * \par Valid operator types
	 * The given operator \a op is required to be:
	 *   -# (no requirements).
	 * \endparblock
	 *
	//  * \parblock
	//  * \par Performance semantics
	//  *      -# This call comprises \f$ \Theta(n) \f$ work, where \f$ n \f$ equals
	//  *         the size of the vector \a x. The constant factor depends on the
	//  *         cost of evaluating the underlying binary operator. A good
	//  *         implementation uses vectorised instructions whenever the input
	//  *         domains, the output domain, and the operator used allow for this.
	//  *
	//  *      -# This call will not result in additional dynamic memory allocations.
	//  *
	//  *      -# This call takes \f$ \mathcal{O}(1) \f$ memory beyond the memory
	//  *         used by the application at the point of a call to this function.
	//  *
	//  *      -# This call incurs at most
	//  *         \f$ n \cdot (
	//  *                       \mathit{sizeof}(InputType) + 2\mathit{sizeof}(IOType)
	//  *                     ) + \mathcal{O}(1)
	//  *         \f$
	//  *         bytes of data movement. A good implementation will rely on in-place
	//  *         operators whenever allowed.
	//  * \endparblock
	 *
	 * @see alp::operators::internal::Operator for a discussion on when in-place
	 *      and/or vectorised operations are used.
	 */
	template< Descriptor descr = descriptors::no_operation,
		typename InputType, typename InputStructure, typename InputView, typename InputImfR, typename InputImfC,
		typename IOType, typename IOStructure, typename IOView, typename IOImfR, typename IOImfC,
		class OP
	>
	RC foldr(
		const Vector< InputType, InputStructure, Density::Dense, InputView, InputImfR, InputImfC, reference > &x,
		Vector< IOType, IOStructure, Density::Dense, IOView, IOImfR, IOImfC, reference > &y,
		const OP & op = OP(),
		const std::enable_if_t<
			alp::is_operator< OP >::value && ! alp::is_object< InputType >::value && ! alp::is_object< IOType >::value
		> * = nullptr
	) {
		// static sanity checks
		NO_CAST_OP_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename OP::D1, InputType >::value ), "alp::eWiseFoldr",
			"called with a vector x of a type that does not match the first domain "
			"of the given operator" );
		NO_CAST_OP_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename OP::D2, IOType >::value ), "alp::eWiseFoldr",
			"called on a vector y of a type that does not match the second domain "
			"of the given operator" );
		NO_CAST_OP_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename OP::D3, IOType >::value ), "alp::eWiseFoldr",
			"called on a vector y of a type that does not match the third domain "
			"of the given operator" );

#ifdef _DEBUG
		std::cout << "foldr(Vector,Vector,OP) called. ";
		std::cout << "Vector 1 has size " << getLength( x ) << " . ";
		std::cout << "Vector 2 has size " << getLength( y ) << " .\n";
#endif
		internal::setInitialized(
			y,
			internal::getInitialized( x ) && internal::getInitialized( y )
		);

		if( !internal::getInitialized( y ) ) {
			return SUCCESS;
		}

		const size_t n = getLength( y );

		if( getLength( x ) != n ) {
			return MISMATCH;
		}

		for ( size_t i = 0; i < n; ++i ) {
			(void) internal::foldr( x[ i ], y[ i ], op );
		}
		return SUCCESS;
	}

	/**
	 * Folds all elements in a ALP Vector \a x into the corresponding
	 * elements from an input/output vector \a y. The vectors must be of equal
	 * size \f$ n \f$. For all \f$ i \in \{0,1,\ldots,n-1\} \f$, the new value
	 * of at the i-th index of \a y after a call to this function thus equals
	 * \f$ x_i \odot y_i \f$.
	 *
	 * @tparam descr     The descriptor used for evaluating this function. By
	 *                   default, this is alp::descriptors::no_operation.
	 * @tparam Monoid    The type of the monoid to be applied.
	 * @tparam IOType         The type of the elements of \a y.
	 * @tparam InputType      The type of the elements of \a x.
	 * @tparam IOStructure    The structure of the vector \a y.
	 * @tparam InputStructure The structure of the vector \a x.
	 * @tparam IOView         The view type applied to the vector \a y.
	 * @tparam InputView      The view type applied to the vector \a x.
	 *
	 * @param[in]       x    The input vector \a y that will not be modified.
	 * @param[in,out]   y    On function entry: the initial value to be applied
	 *                       to \a op as the right-hand side input.
	 *                       On function exit: the result of repeated applications
	 *                       from the right-hand side using elements from \a y.
	 * @param[in]     monoid The monoid under which to perform this right-folding.
	 *
	 * @returns alp::SUCCESS This function always succeeds.
	 *
	 * \note The element-wise fold is also defined for operators.
	 *
	 * \parblock
	 * \par Valid descriptors
	 * alp::descriptors::no_operation, alp::descriptors::no_casting.
	 *
	 * \note Invalid descriptors will be ignored.
	 *
	 * If alp::descriptors::no_casting is specified, then 1) the first domain of
	 * \a op must match \a InputType, 2) the second domain of \a op must match
	 * \a IOType, and 3) the third domain must match \a IOType. If one of these
	 * is not true, the code shall not compile.
	 *
	 * \endparblock
	 *
	 * \parblock
	 * \par Valid monoid types
	 * The given operator \a op is required to be:
	 *   -# (no requirements).
	 * \endparblock
	 *
	//  * \parblock
	//  * \par Performance semantics
	//  *      -# This call comprises \f$ \Theta(n) \f$ work, where \f$ n \f$ equals
	//  *         the size of the vector \a x. The constant factor depends on the
	//  *         cost of evaluating the underlying binary operator. A good
	//  *         implementation uses vectorised instructions whenever the input
	//  *         domains, the output domain, and the operator used allow for this.
	//  *
	//  *      -# This call will not result in additional dynamic memory allocations.
	//  *
	//  *      -# This call takes \f$ \mathcal{O}(1) \f$ memory beyond the memory
	//  *         used by the application at the point of a call to this function.
	//  *
	//  *      -# This call incurs at most
	//  *         \f$ n \cdot (
	//  *                       \mathit{sizeof}(InputType) + 2\mathit{sizeof}(IOType)
	//  *                     ) + \mathcal{O}(1)
	//  *         \f$
	//  *         bytes of data movement. A good implementation will rely on in-place
	//  *         operators whenever allowed.
	//  * \endparblock
	 *
	 * @see alp::operators::internal::Operator for a discussion on when in-place
	 *      and/or vectorised operations are used.
	 */
	template< Descriptor descr = descriptors::no_operation,
		typename InputType, typename InputStructure, typename InputView, typename InputImfR, typename InputImfC,
		typename IOType, typename IOStructure, typename IOView, typename IOImfR, typename IOImfC,
		class Monoid
	>
	RC foldr(
		const Vector< InputType, InputStructure, Density::Dense, InputView, InputImfR, InputImfC, reference > &x,
		Vector< IOType, IOStructure, Density::Dense, IOView, IOImfR, IOImfC, reference > &y,
		const Monoid & monoid = Monoid(),
		const std::enable_if_t<
			alp::is_monoid< Monoid >::value && ! alp::is_object< InputType >::value && ! alp::is_object< IOType >::value
		> * = nullptr
	) {
		// static sanity checks
		NO_CAST_OP_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename Monoid::D1, InputType >::value ), "alp::eWiseFoldr",
			"called with a vector x of a type that does not match the first domain "
			"of the given monoid" );
		NO_CAST_OP_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename Monoid::D2, IOType >::value ), "alp::eWiseFoldr",
			"called on a vector y of a type that does not match the second domain "
			"of the given monoid" );
		NO_CAST_OP_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename Monoid::D3, IOType >::value ), "alp::eWiseFoldr",
			"called on a vector y of a type that does not match the third domain "
			"of the given monoid" );

		return foldr( x, y, monoid.getOperator() );
	}

	/**
	 * For all elements in a ALP Vector \a x, fold the value \f$ \beta \f$
	 * into each element.
	 *
	 * The original value of \f$ \beta \f$ is used as the right-hand side input
	 * of the operator \a op. The left-hand side inputs for \a op are retrieved
	 * from the input vector \a x. The result of the operation is stored in
	 * \f$ \beta \f$, thus overwriting its previous value. This process is
	 * repeated for every element in \a y.
	 *
	 * The value of \f$ x_i \f$ after a call to thus function thus equals
	 * \f$ x_i \odot \beta \f$, for all \f$ i \in \{ 0, 1, \dots, n - 1 \} \f$.
	 *
	 * @tparam descr       The descriptor used for evaluating this function. By
	 *                     default, this is alp::descriptors::no_operation.
	 * @tparam OP          The type of the operator to be applied.
	 * @tparam IOType      The type of the value \a beta.
	 * @tparam InputType   The type of the elements of \a x.
	 * @tparam IOStructure The structure of the vector \a x.
	 * @tparam IOView      The view type applied to the vector \a x.
	 *
	 * @param[in,out] x    On function entry: the initial values to be applied as
	 *                     the left-hand side input to \a op. The input vector must
	 *                     be dense.
	 *                     On function exit: the output data.
	 * @param[in]     beta The input value to apply as the right-hand side input
	 *                     to \a op.
	 * @param[in]     op   The operator under which to perform this left-folding.
	 *
	 * @returns alp::SUCCESS This function always succeeds.
	 *
	 * \note This function is also defined for monoids.
	 *
	 * \warning If \a x is sparse and this operation is requested, a monoid instead
	 *          of an operator is required!
	 *
	 * \parblock
	 * \par Valid descriptors
	 * alp::descriptors::no_operation, alp::descriptors::no_casting.
	 *
	 * \note Invalid descriptors will be ignored.
	 *
	 * If alp::descriptors::no_casting is specified, then 1) the first domain of
	 * \a op must match \a IOType, 2) the second domain of \a op must match
	 * \a InputType, and 3) the third domain must match \a IOType. If one of these
	 * is not true, the code shall not compile.
	 *
	 * \endparblock
	 *
	 * \parblock
	 * \par Valid operator types
	 * The given operator \a op is required to be:
	 *   -# (no requirement).
	 * \endparblock
	 *
	//  * \parblock
	//  * \par Performance semantics
	//  *      -# This call comprises \f$ \Theta(n) \f$ work, where \f$ n \f$ equals
	//  *         the size of the vector \a x. The constant factor depends on the
	//  *         cost of evaluating the underlying binary operator. A good
	//  *         implementation uses vectorised instructions whenever the input
	//  *         domains, the output domain, and the operator used allow for this.
	//  *
	//  *      -# This call will not result in additional dynamic memory allocations.
	//  *
	//  *      -# This call takes \f$ \mathcal{O}(1) \f$ memory beyond the memory
	//  *         used by the application at the point of a call to this function.
	//  *
	//  *      -# This call incurs at most
	//  *         \f$ 2n \cdot \mathit{sizeof}(\mathit{IOType}) + \mathcal{O}(1) \f$
	//  *         bytes of data movement.
	//  * \endparblock
	 *
	 * @see alp::operators::internal::Operator for a discussion on when in-place
	 *      and/or vectorised operations are used.
	 */
	template< Descriptor descr = descriptors::no_operation,
		typename IOType, typename IOStructure, typename IOView, typename IOImfR, typename IOImfC,
		typename InputType, typename InputStructure,
		class Op
	>
	RC foldl(
		Vector< IOType, IOStructure, Density::Dense, IOView, IOImfR, IOImfC, reference > &x,
		const Scalar< InputType, InputStructure, reference > beta,
		const Op &op = Op(),
		const std::enable_if_t<
			! alp::is_object< IOType >::value && ! alp::is_object< InputType >::value && alp::is_operator< Op >::value
		> * = nullptr
	) {
		// static sanity checks
		NO_CAST_OP_ASSERT(
			( ! ( descr & descriptors::no_casting )	|| std::is_same< typename Op::D1, IOType >::value ),
			"alp::foldl",
			"called with a vector x of a type that does not match the first domain "
			"of the given operator"
		);
		NO_CAST_OP_ASSERT(
			( ! ( descr & descriptors::no_casting )	|| std::is_same< typename Op::D2, InputType >::value ),
			"alp::foldl",
			"called on a vector y of a type that does not match the second domain "
			"of the given operator"
		);
		NO_CAST_OP_ASSERT(
			( ! ( descr & descriptors::no_casting )	|| std::is_same< typename Op::D3, IOType >::value ),
			"alp::foldl",
			"called on a vector x of a type that does not match the third domain "
			"of the given operator"
		);

#ifdef _DEBUG
		std::cout << "foldl(Vector,Scalar,Op) called. Vector has size " << getLength( x ) << " .\n";
#endif

		internal::setInitialized(
			x,
			internal::getInitialized( x ) && internal::getInitialized( beta )
		);

		if( !internal::getInitialized( x ) ) {
			return SUCCESS;
		}

		const size_t n = getLength( x );
		for ( size_t i = 0; i < n; ++i ) {
			(void) internal::foldl( x[ i ], *beta, op );
		}
		return SUCCESS;
	}

	/**
	 * For all elements in a ALP Vector \a x, fold the value \f$ \beta \f$
	 * into each element.
	 *
	 * Masked operator variant.
	 */
	template< Descriptor descr = descriptors::no_operation,
		typename IOType, typename IOStructure, typename IOView, typename IOImfR, typename IOImfC,
		typename MaskType, typename MaskStructure, typename MaskView, typename MaskImfR, typename MaskImfC,
		typename InputType, typename InputStructure,
		class Op
	>
	RC foldl(
		Vector< IOType, IOStructure, Density::Dense, IOView, IOImfR, IOImfC, reference > & x,
		const Vector< MaskType, MaskStructure, Density::Dense, MaskView, MaskImfR, MaskImfC, reference > & m,
		const Scalar< InputType, InputStructure, reference > &beta,
		const Op & op = Op(),
		const std::enable_if_t<
			!alp::is_object< IOType >::value && ! alp::is_object< InputType >::value && alp::is_operator< Op >::value
		> * = nullptr
	) {
		// static sanity checks
		NO_CAST_OP_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename Op::D1, IOType >::value ), "alp::foldl",
			"called with a vector x of a type that does not match the first domain "
			"of the given operator" );
		NO_CAST_OP_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename Op::D2, InputType >::value ), "alp::foldl",
			"called on a vector y of a type that does not match the second domain "
			"of the given operator" );
		NO_CAST_OP_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename Op::D3, IOType >::value ), "alp::foldl",
			"called on a vector x of a type that does not match the third domain "
			"of the given operator" );
		NO_CAST_OP_ASSERT(
			( ! ( descr & descriptors::no_casting ) || std::is_same< bool, MaskType >::value ), "alp::foldl (reference, vector <- scalar, masked)", "provided mask does not have boolean entries" );
		if( size( m ) == 0 ) {
			return foldl< descr >( x, beta, op );
		}
		throw std::runtime_error( "Needs an implementation." );
		return SUCCESS;
	}

	/**
	 * For all elements in a ALP Vector \a x, fold the value \f$ \beta \f$
	 * into each element.
	 *
	 * The original value of \f$ \beta \f$ is used as the right-hand side input
	 * of the operator \a op. The left-hand side inputs for \a op are retrieved
	 * from the input vector \a x. The result of the operation is stored in
	 * \f$ \beta \f$, thus overwriting its previous value. This process is
	 * repeated for every element in \a y.
	 *
	 * The value of \f$ x_i \f$ after a call to thus function thus equals
	 * \f$ x_i \odot \beta \f$, for all \f$ i \in \{ 0, 1, \dots, n - 1 \} \f$.
	 *
	 * @tparam descr       The descriptor used for evaluating this function. By
	 *                     default, this is alp::descriptors::no_operation.
	 * @tparam Monoid      The type of the monoid to be applied.
	 * @tparam IOType      The type of the elements of \a x.
	 * @tparam InputType   The type of the value \a beta.
	 * @tparam IOStructure The structure of the vector \a x.
	 * @tparam IOView      The view type applied to the vector \a x.
	 *
	 * @param[in,out] x    On function entry: the initial values to be applied as
	 *                     the left-hand side input to \a op.
	 *                     On function exit: the output data.
	 * @param[in]     beta The input value to apply as the right-hand side input
	 *                     to \a op.
	 * @param[in]   monoid The monoid under which to perform this left-folding.
	 *
	 * @returns alp::SUCCESS This function always succeeds.
	 *
	 * \note This function is also defined for operators.
	 *
	 * \parblock
	 * \par Valid descriptors
	 * alp::descriptors::no_operation, alp::descriptors::no_casting.
	 *
	 * \note Invalid descriptors will be ignored.
	 *
	 * If alp::descriptors::no_casting is specified, then 1) the first domain of
	 * \a op must match \a IOType, 2) the second domain of \a op must match
	 * \a InputType, and 3) the third domain must match \a IOType. If one of these
	 * is not true, the code shall not compile.
	 *
	 * \endparblock
	 *
	 * \parblock
	 * \par Valid operator types
	 * The given operator \a op is required to be:
	 *   -# (no requirement).
	 * \endparblock
	 *
	//  * \parblock
	//  * \par Performance semantics
	//  *      -# This call comprises \f$ \Theta(n) \f$ work, where \f$ n \f$ equals
	//  *         the size of the vector \a x. The constant factor depends on the
	//  *         cost of evaluating the underlying binary operator. A good
	//  *         implementation uses vectorised instructions whenever the input
	//  *         domains, the output domain, and the operator used allow for this.
	//  *
	//  *      -# This call will not result in additional dynamic memory allocations.
	//  *
	//  *      -# This call takes \f$ \mathcal{O}(1) \f$ memory beyond the memory
	//  *         used by the application at the point of a call to this function.
	//  *
	//  *      -# This call incurs at most
	//  *         \f$ 2n \cdot \mathit{sizeof}(\mathit{IOType}) + \mathcal{O}(1) \f$
	//  *         bytes of data movement.
	//  * \endparblock
	 *
	 * @see alp::operators::internal::Operator for a discussion on when in-place
	 *      and/or vectorised operations are used.
	 */
	template< Descriptor descr = descriptors::no_operation,
		typename IOType, typename IOStructure, typename IOView, typename IOImfR, typename IOImfC,
		typename InputType,
		class Monoid
	>
	RC foldl(
		Vector< IOType, IOStructure, Density::Dense, IOView, IOImfR, IOImfC, reference > & x,
		const InputType beta,
		const Monoid & monoid = Monoid(),
		const std::enable_if_t<
			!alp::is_object< IOType >::value && ! alp::is_object< InputType >::value && alp::is_monoid< Monoid >::value
		> * = nullptr
	) {
		// static sanity checks
		NO_CAST_OP_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename Monoid::D1, IOType >::value ), "alp::foldl",
			"called with a vector x of a type that does not match the first domain "
			"of the given monoid" );
		NO_CAST_OP_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename Monoid::D2, InputType >::value ), "alp::foldl",
			"called on a vector y of a type that does not match the second domain "
			"of the given monoid" );
		NO_CAST_OP_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename Monoid::D3, IOType >::value ), "alp::foldl",
			"called on a vector x of a type that does not match the third domain "
			"of the given monoid" );

		throw std::runtime_error( "Needs an implementation." );
		return SUCCESS;
	}

	/**
	 * For all elements in a ALP Vector \a x, fold the value \f$ \beta \f$
	 * into each element.
	 *
	 * Masked monoid variant.
	 */
	template< Descriptor descr = descriptors::no_operation,
		typename IOType, typename IOStructure, typename IOView, typename IOImfR, typename IOImfC,
		typename MaskType, typename MaskStructure, typename MaskView, typename MaskImfR, typename MaskImfC,
		typename InputType,
		class Monoid
	>
	RC foldl( Vector< IOType, IOStructure, Density::Dense, IOView, IOImfR, IOImfC, reference > & x,
		const Vector< MaskType, MaskStructure, Density::Dense, MaskView, MaskImfR, MaskImfC, reference > & m,
		const InputType & beta,
		const Monoid & monoid = Monoid(),
		const std::enable_if_t<
			!alp::is_object< IOType >::value && ! alp::is_object< MaskType >::value && ! alp::is_object< InputType >::value && alp::is_monoid< Monoid >::value
		> * = nullptr
		) {
		// static sanity checks
		NO_CAST_OP_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename Monoid::D1, IOType >::value ), "alp::foldl",
			"called with a vector x of a type that does not match the first domain "
			"of the given monoid" );
		NO_CAST_OP_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename Monoid::D2, InputType >::value ), "alp::foldl",
			"called on a vector y of a type that does not match the second domain "
			"of the given monoid" );
		NO_CAST_OP_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename Monoid::D3, IOType >::value ), "alp::foldl",
			"called on a vector x of a type that does not match the third domain "
			"of the given monoid" );
		NO_CAST_OP_ASSERT(
			( ! ( descr & descriptors::no_casting ) || std::is_same< bool, MaskType >::value ), "alp::foldl (reference, vector <- scalar, masked, monoid)", "provided mask does not have boolean entries" );
		if( size( m ) == 0 ) {
			return foldl< descr >( x, beta, monoid );
		}

		throw std::runtime_error( "Needs an implementation." );
		return SUCCESS;
	}

	/**
	 * Folds all elements in a ALP Vector \a y into the corresponding
	 * elements from an input/output vector \a x. The vectors must be of equal
	 * size \f$ n \f$. For all \f$ i \in \{0,1,\ldots,n-1\} \f$, the new value
	 * of at the i-th index of \a x after a call to this function thus equals
	 * \f$ x_i \odot y_i \f$.
	 *
	 * @tparam descr          The descriptor used for evaluating this function. By
	 *                        default, this is alp::descriptors::no_operation.
	 * @tparam OP             The type of the operator to be applied.
	 * @tparam IOType         The type of the value \a x.
	 * @tparam InputType      The type of the elements of \a y.
	 * @tparam IOStructure    The structure of the vector \a x.
	 * @tparam InputStructure The structure of the vector \a y.
	 * @tparam IOView         The view type applied to the vector \a x.
	 * @tparam InputView      The view type applied to the vector \a y.
	 *
	 * @param[in,out] x On function entry: the vector whose elements are to be
	 *                  applied to \a op as the left-hand side input.
	 *                  On function exit: the vector containing the result of
	 *                  the requested computation.
	 * @param[in]    y  The input vector \a y whose elements are to be applied
	 *                  to \a op as right-hand side input.
	 * @param[in]    op The operator under which to perform this left-folding.
	 *
	 * @returns alp::SUCCESS This function always succeeds.
	 *
	 * \note This function is also defined for monoids.
	 *
	 * \parblock
	 * \par Valid descriptors
	 * alp::descriptors::no_operation, alp::descriptors::no_casting.
	 *
	 * \note Invalid descriptors will be ignored.
	 *
	 * If alp::descriptors::no_casting is specified, then 1) the first domain of
	 * \a op must match \a IOType, 2) the second domain of \a op must match
	 * \a InputType, and 3) the third domain must match \a IOType. If one of these
	 * is not true, the code shall not compile.
	 *
	 * \endparblock
	 *
	 * \parblock
	 * \par Valid operator types
	 * The given operator \a op is required to be:
	 *   -# (no requirements).
	 * \endparblock
	 *
	//  * \parblock
	//  * \par Performance semantics
	//  *      -# This call comprises \f$ \Theta(n) \f$ work, where \f$ n \f$ equals
	//  *         the size of the vector \a x. The constant factor depends on the
	//  *         cost of evaluating the underlying binary operator. A good
	//  *         implementation uses vectorised instructions whenever the input
	//  *         domains, the output domain, and the operator used allow for this.
	//  *
	//  *      -# This call will not result in additional dynamic memory allocations.
	//  *
	//  *      -# This call takes \f$ \mathcal{O}(1) \f$ memory beyond the memory
	//  *         used by the application at the point of a call to this function.
	//  *
	//  *      -# This call incurs at most
	//  *         \f$ n \cdot (
	//  *                \mathit{sizeof}(\mathit{IOType}) +
	//  *                \mathit{sizeof}(\mathit{InputType})
	//  *             ) + \mathcal{O}(1) \f$
	//  *         bytes of data movement. A good implementation will apply in-place
	//  *         vectorised instructions whenever the input domains, the output
	//  *         domain, and the operator used allow for this.
	//  * \endparblock
	 *
	 * @see alp::operators::internal::Operator for a discussion on when in-place
	 *      and/or vectorised operations are used.
	 */
	template<
		Descriptor descr = descriptors::no_operation,
		typename IOType, typename IOStructure, typename IOView, typename IOImfR, typename IOImfC,
		typename InputType, typename InputStructure, typename InputView, typename InputImfR, typename InputImfC,
		class OP
	>
	RC foldl(
		Vector< IOType, IOStructure, Density::Dense, IOView, IOImfR, IOImfC, reference > &x,
		const Vector< InputType, InputStructure, Density::Dense, InputView, InputImfR, InputImfC, reference > &y,
		const OP &op = OP(),
		const std::enable_if_t<
			alp::is_operator< OP >::value && !alp::is_object< IOType >::value && !alp::is_object< InputType >::value
		> * = nullptr
	) {
		// static sanity checks
		NO_CAST_OP_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename OP::D1, IOType >::value ), "alp::foldl",
			"called with a vector x of a type that does not match the first domain "
			"of the given operator" );
		NO_CAST_OP_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename OP::D2, InputType >::value ), "alp::foldl",
			"called on a vector y of a type that does not match the second domain "
			"of the given operator" );
		NO_CAST_OP_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename OP::D3, IOType >::value ), "alp::foldl",
			"called on a vector x of a type that does not match the third domain "
			"of the given operator" );

		// dynamic sanity checks
		const size_t n = size( x );
		if( n != size( y ) ) {
			return MISMATCH;
		}

		if( !internal::getInitialized( x ) ) {
			return SUCCESS;
		}

		if( !internal::getInitialized( y ) ) {
			internal::setInitialized( x, false );
			return SUCCESS;
		}

		for( size_t i = 0; i < n; ++i ) {
			/** \internal \todo Implement RC check. Also applies to other locations. */
			(void) internal::foldl( x[ i ], y[ i ], op );
		}

		return SUCCESS;
	}

	/**
	 * Folds all elements in a ALP Vector \a y into the corresponding
	 * elements from an input/output vector \a x. The vectors must be of equal
	 * size \f$ n \f$. For all \f$ i \in \{0,1,\ldots,n-1\} \f$, the new value
	 * of at the i-th index of \a x after a call to this function thus equals
	 * \f$ x_i \odot y_i \f$.
	 *
	 * @tparam descr          The descriptor used for evaluating this function. By
	 *                        default, this is alp::descriptors::no_operation.
	 * @tparam Monoid         The type of the monoid to be applied.
	 * @tparam IOType         The type of the value \a x.
	 * @tparam InputType      The type of the elements of \a y.
	 * @tparam IOStructure    The structure of the vector \a x.
	 * @tparam InputStructure The structure of the vector \a y.
	 * @tparam IOView         The view type applied to the vector \a x.
	 * @tparam InputView      The view type applied to the vector \a y.
	 *
	 * @param[in,out]  x    On function entry: the vector whose elements are to be
	 *                      applied to \a op as the left-hand side input.
	 *                      On function exit: the vector containing the result of
	 *                      the requested computation.
	 * @param[in]      y    The input vector \a y whose elements are to be applied
	 *                      to \a op as right-hand side input.
	 * @param[in]    monoid The operator under which to perform this left-folding.
	 *
	 * @returns alp::SUCCESS This function always succeeds.
	 *
	 * \note This function is also defined for operators.
	 *
	 * \parblock
	 * \par Valid descriptors
	 * alp::descriptors::no_operation, alp::descriptors::no_casting.
	 *
	 * \note Invalid descriptors will be ignored.
	 *
	 * If alp::descriptors::no_casting is specified, then 1) the first domain of
	 * \a op must match \a IOType, 2) the second domain of \a op must match
	 * \a InputType, and 3) the third domain must match \a IOType. If one of these
	 * is not true, the code shall not compile.
	 *
	 * \endparblock
	 *
	 * \parblock
	 * \par Valid operator types
	 * The given operator \a op is required to be:
	 *   -# (no requirements).
	 * \endparblock
	 *
	//  * \parblock
	//  * \par Performance semantics
	//  *      -# This call comprises \f$ \Theta(n) \f$ work, where \f$ n \f$ equals
	//  *         the size of the vector \a x. The constant factor depends on the
	//  *         cost of evaluating the underlying binary operator. A good
	//  *         implementation uses vectorised instructions whenever the input
	//  *         domains, the output domain, and the operator used allow for this.
	//  *
	//  *      -# This call will not result in additional dynamic memory allocations.
	//  *
	//  *      -# This call takes \f$ \mathcal{O}(1) \f$ memory beyond the memory
	//  *         used by the application at the point of a call to this function.
	//  *
	//  *      -# This call incurs at most
	//  *         \f$ n \cdot (
	//  *                \mathit{sizeof}(\mathit{IOType}) +
	//  *                \mathit{sizeof}(\mathit{InputType})
	//  *             ) + \mathcal{O}(1) \f$
	//  *         bytes of data movement. A good implementation will apply in-place
	//  *         vectorised instructions whenever the input domains, the output
	//  *         domain, and the operator used allow for this.
	//  * \endparblock
	 *
	 * @see alp::operators::internal::Operator for a discussion on when in-place
	 *      and/or vectorised operations are used.
	 */
	template< Descriptor descr = descriptors::no_operation,
		typename IOType, typename IOStructure, typename IOView, typename IOImfR, typename IOImfC,
		typename InputType, typename InputStructure, typename InputView, typename InputImfR, typename InputImfC,
		class Monoid
	>
	RC foldl( Vector< IOType, IOStructure, Density::Dense, IOView, IOImfR, IOImfC, reference > & x,
		const Vector< InputType, InputStructure, Density::Dense, InputView, InputImfR, InputImfC, reference > & y,
		const Monoid & monoid = Monoid(),
		const std::enable_if_t<
			alp::is_monoid< Monoid >::value && ! alp::is_object< IOType >::value && ! alp::is_object< InputType >::value
		  > * = nullptr
		) {
		// static sanity checks
		NO_CAST_OP_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename Monoid::D1, IOType >::value ), "alp::foldl",
			"called with a vector x of a type that does not match the first domain "
			"of the given operator" );
		NO_CAST_OP_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename Monoid::D2, InputType >::value ), "alp::foldl",
			"called on a vector y of a type that does not match the second domain "
			"of the given operator" );
		NO_CAST_OP_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename Monoid::D3, IOType >::value ), "alp::foldl",
			"called on a vector x of a type that does not match the third domain "
			"of the given operator" );

		// dynamic sanity checks
		const size_t n = size( x );
		if( n != size( y ) ) {
			return MISMATCH;
		}

		throw std::runtime_error( "Needs an implementation." );
		return SUCCESS;
	}

	/**
	 * Calculates the element-wise operation on one scalar to elements of one
	 * vector, \f$ z = x .* \beta \f$, using the given operator. The input and
	 * output vectors must be of equal length.
	 *
	 * The vectors \a x or \a y may not be sparse.
	 *
	 * For all valid indices \a i of \a z, its element \f$ z_i \f$ after
	 * the call to this function completes equals \f$ x_i \odot \beta \f$.
	 *
	 * \warning Use of sparse vectors is only supported in full generality
	 *          when applied via a monoid or semiring; otherwise, there is
	 *          no concept for correctly interpreting any missing vector
	 *          elements during the requested computation.
	 * \note    When applying element-wise operators on sparse vectors
	 *          using semirings, there is a difference between interpreting missing
	 *          values as an annihilating identity or as a neutral identity--
	 *          intuitively, such identities are known as `zero' or `one',
	 *          respectively. As a consequence, there are three different variants
	 *          for element-wise operations whose names correspond to their
	 *          intuitive meanings w.r.t. those identities:
	 *            -# eWiseAdd (neutral),
	 *            -# eWiseMul (annihilating), and
	 *            -# eWiseApply using monoids (neutral).
	 *          An eWiseAdd with some semiring and an eWiseApply using its additive
	 *          monoid are totally equivalent.
	 *
	 * @tparam descr            The descriptor to be used. Equal to
	 *                          descriptors::no_operation if left unspecified.
	 * @tparam OP               The operator to use.
	 * @tparam InputType1       The value type of the left-hand vector.
	 * @tparam InputType2       The value type of the right-hand scalar.
	 * @tparam OutputType       The value type of the ouput vector.
	 * @tparam InputStructure1  The structure of the left-hand vector.
	 * @tparam OutputStructure1 The structure of the output vector.
	 * @tparam InputView1       The view type applied to the left-hand vector.
	 * @tparam OutputView1      The view type applied to the output vector.
	 *
	 * @param[in]   x   The left-hand input vector.
	 * @param[in]  beta The right-hand input scalar.
	 * @param[out]  z   The pre-allocated output vector.
	 * @param[in]   op  The operator to use.
	 *
	 * @return alp::MISMATCH Whenever the dimensions of \a x and \a z do not
	 *                       match. All input data containers are left untouched
	 *                       if this exit code is returned; it will be as though
	 *                       this call was never made.
	 * @return alp::SUCCESS  On successful completion of this call.
	 *
	//  * \parblock
	//  * \par Performance semantics
	//  *      -# This call comprises \f$ \Theta(n) \f$ work, where \f$ n \f$ equals
	//  *         the size of the vectors \a x and \a z. The constant factor depends
	//  *         on the cost of evaluating the operator. A good implementation uses
	//  *         vectorised instructions whenever the input domains, the output
	//  *         domain, and the operator used allow for this.
	//  *
	//  *      -# This call takes \f$ \mathcal{O}(1) \f$ memory beyond the memory
	//  *         used by the application at the point of a call to this function.
	//  *
	//  *      -# This call incurs at most
	//  *         \f$ n(
	//  *               \mathit{sizeof}(\mathit{D1}) + \mathit{sizeof}(\mathit{D3})
	//  *             ) +
	//  *         \mathcal{O}(1) \f$
	//  *         bytes of data movement. A good implementation will stream \a y
	//  *         into \a z to apply the multiplication operator in-place, whenever
	//  *         the input domains, the output domain, and the operator allow for
	//  *         this.
	//  * \endparblock
	 */
	template< Descriptor descr = descriptors::no_operation,
		typename OutputType, typename OutputStructure, typename OutputView, typename OutputImfR, typename OutputImfC,
		typename InputType1, typename InputStructure1, typename InputView1, typename InputImfR, typename InputImfC,
		typename InputType2, typename InputStructure2,
		class OP
	>
	RC eWiseApply( Vector< OutputType, OutputStructure, Density::Dense, OutputView, OutputImfR, OutputImfC, reference > & z,
		const Vector< InputType1, InputStructure1, Density::Dense, InputView1, InputImfR, InputImfC, reference > & x,
		const Scalar< InputType2, InputStructure2, reference > &beta,
		const OP & op = OP(),
		const typename std::enable_if< ! alp::is_object< OutputType >::value && ! alp::is_object< InputType1 >::value && ! alp::is_object< InputType2 >::value && alp::is_operator< OP >::value,
			void >::type * const = NULL ) {
	#ifdef _DEBUG
		std::cout << "In eWiseApply ([T1]<-[T2]<-T3), operator variant\n";
	#endif
		throw std::runtime_error( "Needs an implementation." );
		return SUCCESS;
	}

	/**
	 * Computes \f$ z = x \odot y \f$, out of place.
	 *
	 * Specialisation for \a x and \a y scalar, operator version.
	 */
	template< Descriptor descr = descriptors::no_operation,
		typename OutputType, typename OutputStructure, typename OutputView, typename OutputImfR, typename OutputImfC,
		typename InputType1, typename InputStructure1,
		typename InputType2, typename InputStructure2,
		class OP
	>
	RC eWiseApply( Vector< OutputType, OutputStructure, Density::Dense, OutputView, OutputImfR, OutputImfC, reference > & z,
		const Scalar< InputType1, InputStructure1, reference> &alpha,
		const Scalar< InputType2, InputStructure2, reference> &beta,
		const OP & op = OP(),
		const typename std::enable_if< ! alp::is_object< OutputType >::value && ! alp::is_object< InputType1 >::value && ! alp::is_object< InputType2 >::value && alp::is_operator< OP >::value,
			void >::type * const = NULL ) {
	#ifdef _DEBUG
		std::cout << "In eWiseApply ([T1]<-T2<-T3), operator variant\n";
	#endif
		typename OP::D3 val;
		RC ret = apply< descr >( val, alpha, beta, op );
		ret = ret ? ret : set< descr >( z, val );
		return ret;
	}

	/**
	 * Computes \f$ z = x \odot y \f$, out of place.
	 *
	 * Specialisation for \a x and \a y scalar, monoid version.
	 */
	template< Descriptor descr = descriptors::no_operation,
		typename OutputType, typename OutputStructure, typename OutputView, typename OutputImfR, typename OutputImfC,
		typename InputType1, typename InputStructure1,
		typename InputType2, typename InputStructure2,
		class Monoid
	>
	RC eWiseApply( Vector< OutputType, OutputStructure, Density::Dense, OutputView, OutputImfR, OutputImfC, reference > & z,
		const Scalar< InputType1, InputStructure1, reference> &alpha,
		const Scalar< InputType2, InputStructure2, reference> &beta,
		const Monoid & monoid = Monoid(),
		const typename std::enable_if< ! alp::is_object< OutputType >::value && ! alp::is_object< InputType1 >::value && ! alp::is_object< InputType2 >::value && alp::is_monoid< Monoid >::value,
			void >::type * const = NULL ) {
	#ifdef _DEBUG
		std::cout << "In eWiseApply ([T1]<-T2<-T3), monoid variant\n";
	#endif
		// simply delegate to operator variant
		return eWiseApply< descr >( z, alpha, beta, monoid.getOperator() );
	}

	/**
	 * Computes \f$ z = x \odot y \f$, out of place.
	 *
	 * Specialisation for scalar \a y, masked operator version.
	 */
	template< Descriptor descr = descriptors::no_operation,
		typename OutputType, typename OutputStructure, typename OutputView, typename OutputImfR, typename OutputImfC,
		typename MaskType, typename MaskStructure, typename MaskView, typename MaskImfR, typename MaskImfC,
		typename InputType1, typename InputStructure1, typename InputView1, typename InputImfR1, typename InputImfC1,
		typename InputType2, typename InputStructure2,
		class OP
	>
	RC eWiseApply( Vector< OutputType, OutputStructure, Density::Dense, OutputView, OutputImfR, OutputImfC, reference > & z,
		const Vector< MaskType, MaskStructure, Density::Dense, MaskView, MaskImfR, MaskImfC, reference > & mask,
		const Vector< InputType1, InputStructure1, Density::Dense, InputView1, InputImfR1, InputImfC1, reference > & x,
		const Scalar< InputType2, InputStructure2, reference > &beta,
		const OP & op = OP(),
		const typename std::enable_if< ! alp::is_object< OutputType >::value && ! alp::is_object< MaskType >::value && ! alp::is_object< InputType1 >::value && ! alp::is_object< InputType2 >::value &&
				alp::is_operator< OP >::value,
			void >::type * const = NULL ) {
	#ifdef _DEBUG
		std::cout << "In masked eWiseApply ([T1]<-[T2]<-T3, using operator)\n";
	#endif
		// check for empty mask
		if( size( mask ) == 0 ) {
			return eWiseApply< descr >( z, x, beta, op );
		}
		throw std::runtime_error( "Needs an implementation." );
		return SUCCESS;
	}

	/**
	 * Computes \f$ z = x \odot y \f$, out of place.
	 *
	 * Monoid version.
	 */
	template< Descriptor descr = descriptors::no_operation,
		typename OutputType, typename OutputStructure, typename OutputView, typename OutputImfR, typename OutputImfC,
		typename InputType1, typename InputStructure1, typename InputView1, typename InputImfR1, typename InputImfC1,
		typename InputType2, typename InputStructure2, typename InputView2, typename InputImfR2, typename InputImfC2,
		class Monoid
	>
	RC eWiseApply( Vector< OutputType, OutputStructure, Density::Dense, OutputView, OutputImfR, OutputImfC, reference > & z,
		const Vector< InputType1, InputStructure1, Density::Dense, InputView1, InputImfR1, InputImfC1, reference > & x,
		const Vector< InputType2, InputStructure2, Density::Dense, InputView2, InputImfR2, InputImfC2, reference > & y,
		const Monoid & monoid = Monoid(),
		const typename std::enable_if< ! alp::is_object< OutputType >::value && ! alp::is_object< InputType1 >::value && ! alp::is_object< InputType2 >::value && alp::is_monoid< Monoid >::value,
			void >::type * const = NULL ) {
	#ifdef _DEBUG
		std::cout << "In unmasked eWiseApply ([T1]<-[T2]<-[T3], using monoid)\n";
	#endif
		throw std::runtime_error( "Needs an implementation." );
		return SUCCESS;
	}

	/**
	 * Computes \f$ z = x \odot y \f$, out of place.
	 *
	 * Specialisation for scalar \a x. Monoid version.
	 */
	template< Descriptor descr = descriptors::no_operation,
		typename OutputType, typename OutputStructure, typename OutputView, typename OutputImfR, typename OutputImfC,
		typename InputType1, typename InputStructure1,
		typename InputType2, typename InputStructure2, typename InputView2, typename InputImfR2, typename InputImfC2,
		class Monoid
	>
	RC eWiseApply( Vector< OutputType, OutputStructure, Density::Dense, OutputView, OutputImfR, OutputImfC, reference > & z,
		const Scalar< InputType1, InputStructure1, reference> &alpha,
		const Vector< InputType2, InputStructure2, Density::Dense, InputView2, InputImfR2, InputImfC2, reference > & y,
		const Monoid & monoid = Monoid(),
		const typename std::enable_if< ! alp::is_object< OutputType >::value && ! alp::is_object< InputType1 >::value && ! alp::is_object< InputType2 >::value && alp::is_monoid< Monoid >::value,
			void >::type * const = NULL ) {
	#ifdef _DEBUG
		std::cout << "In unmasked eWiseApply ([T1]<-T2<-[T3], using monoid)\n";
	#endif
		throw std::runtime_error( "Needs an implementation." );
		return SUCCESS;
	}

	/**
	 * Computes \f$ z = x \odot y \f$, out of place.
	 *
	 * Specialisation for scalar \a y. Monoid version.
	 */
	template< Descriptor descr = descriptors::no_operation,
		typename OutputType, typename OutputStructure, typename OutputView, typename OutputImfR, typename OutputImfC,
		typename InputType1, typename InputStructure1, typename InputView1, typename InputImfR1, typename InputImfC1,
		typename InputType2, typename InputStructure2,
		class Monoid
	>
	RC eWiseApply( Vector< OutputType, OutputStructure, Density::Dense, OutputView, OutputImfR, OutputImfC, reference > & z,
		const Vector< InputType1, InputStructure1, Density::Dense, InputView1, InputImfR1, InputImfC1, reference > & x,
		const Scalar< InputType2, InputStructure2, reference > &beta,
		const Monoid & monoid = Monoid(),
		const typename std::enable_if< ! alp::is_object< OutputType >::value && ! alp::is_object< InputType1 >::value && ! alp::is_object< InputType2 >::value && alp::is_monoid< Monoid >::value,
			void >::type * const = NULL ) {
	#ifdef _DEBUG
		std::cout << "In unmasked eWiseApply ([T1]<-T2<-[T3], using monoid)\n";
	#endif
		throw std::runtime_error( "Needs an implementation." );
		return SUCCESS;
	}

	/**
	 * Computes \f$ z = x \odot y \f$, out of place.
	 *
	 * Masked monoid version.
	 */
	template< Descriptor descr = descriptors::no_operation,
		typename OutputType, typename OutputStructure, typename OutputView, typename OutputImfR, typename OutputImfC,
		typename MaskType, typename MaskStructure, typename MaskView, typename MaskImfR, typename MaskImfC,
		typename InputType1, typename InputStructure1, typename InputView1, typename InputImfR1, typename InputImfC1,
		typename InputType2, typename InputStructure2, typename InputView2, typename InputImfR2, typename InputImfC2,
		class Monoid
	>
	RC eWiseApply( Vector< OutputType, OutputStructure, Density::Dense, OutputView, OutputImfR, OutputImfC, reference > & z,
		const Vector< MaskType, MaskStructure, Density::Dense, MaskView, MaskImfR, MaskImfC, reference > & mask,
		const Vector< InputType1, InputStructure1, Density::Dense, InputView1, InputImfR1, InputImfC1, reference > & x,
		const Vector< InputType2, InputStructure2, Density::Dense, InputView2, InputImfR2, InputImfC2, reference > & y,
		const Monoid & monoid = Monoid(),
		const typename std::enable_if< ! alp::is_object< OutputType >::value && ! alp::is_object< MaskType >::value && ! alp::is_object< InputType1 >::value && ! alp::is_object< InputType2 >::value &&
				alp::is_monoid< Monoid >::value,
			void >::type * const = NULL ) {
	#ifdef _DEBUG
		std::cout << "In masked eWiseApply ([T1]<-[T2]<-[T3], using monoid)\n";
	#endif
		throw std::runtime_error( "Needs an implementation." );
		return SUCCESS;
	}

	/**
	 * Computes \f$ z = x \odot y \f$, out of place.
	 *
	 * Specialisation for scalar \a x. Masked monoid version.
	 */
	template< Descriptor descr = descriptors::no_operation,
		typename OutputType, typename OutputStructure, typename OutputView, typename OutputImfR, typename OutputImfC,
		typename MaskType, typename MaskStructure, typename MaskView, typename MaskImfR, typename MaskImfC,
		typename InputType1, typename InputStructure1,
		typename InputType2, typename InputStructure2, typename InputView2, typename InputImfR2, typename InputImfC2,
		class Monoid
	>
	RC eWiseApply( Vector< OutputType, OutputStructure, Density::Dense, OutputView, OutputImfR, OutputImfC, reference > & z,
		const Vector< MaskType, MaskStructure, Density::Dense, MaskView, MaskImfR, MaskImfC, reference > & mask,
		const Scalar< InputType1, InputStructure1, reference> &alpha,
		const Vector< InputType2, InputStructure2, Density::Dense, InputView2, InputImfR2, InputImfC2, reference > & y,
		const Monoid & monoid = Monoid(),
		const typename std::enable_if< ! alp::is_object< OutputType >::value && ! alp::is_object< MaskType >::value && ! alp::is_object< InputType1 >::value && ! alp::is_object< InputType2 >::value &&
				alp::is_monoid< Monoid >::value,
			void >::type * const = NULL ) {
	#ifdef _DEBUG
		std::cout << "In masked eWiseApply ([T1]<-T2<-[T3], using monoid)\n";
	#endif
		throw std::runtime_error( "Needs an implementation." );
		return SUCCESS;
	}

	/**
	 * Computes \f$ z = x \odot y \f$, out of place.
	 *
	 * Specialisation for scalar \a y. Masked monoid version.
	 */
	template< Descriptor descr = descriptors::no_operation,
		typename OutputType, typename OutputStructure, typename OutputView, typename OutputImfR, typename OutputImfC,
		typename MaskType, typename MaskStructure, typename MaskView, typename MaskImfR, typename MaskImfC,
		typename InputType1, typename InputStructure1, typename InputView1, typename InputImfR1, typename InputImfC1,
		typename InputType2, typename InputStructure2,
		class Monoid
	>
	RC eWiseApply( Vector< OutputType, OutputStructure, Density::Dense, OutputView, OutputImfR, OutputImfC, reference > & z,
		const Vector< MaskType, MaskStructure, Density::Dense, MaskView, MaskImfR, MaskImfC, reference > & mask,
		const Vector< InputType1, InputStructure1, Density::Dense, InputView1, InputImfR1, InputImfC1, reference > & x,
		const Scalar< InputType2, InputStructure2, reference > &beta,
		const Monoid & monoid = Monoid(),
		const typename std::enable_if< ! alp::is_object< OutputType >::value && ! alp::is_object< MaskType >::value && ! alp::is_object< InputType1 >::value && ! alp::is_object< InputType2 >::value &&
				alp::is_monoid< Monoid >::value,
			void >::type * const = NULL ) {
	#ifdef _DEBUG
		std::cout << "In masked eWiseApply ([T1]<-[T2]<-T3, using monoid)\n";
	#endif
		throw std::runtime_error( "Needs an implementation." );
		return SUCCESS;
	}

	/**
	 * Calculates the element-wise operation on one scalar to elements of one
	 * vector, \f$ z = \alpha .* y \f$, using the given operator. The input and
	 * output vectors must be of equal length.
	 *
	 * The vectors \a x or \a y may not be sparse.
	 *
	 * For all valid indices \a i of \a z, its element \f$ z_i \f$ after
	 * the call to this function completes equals \f$ \alpha \odot y_i \f$.
	 *
	 * \warning Use of sparse vectors is only supported in full generality
	 *          when applied via a monoid or semiring; otherwise, there is
	 *          no concept for correctly interpreting any missing vector
	 *          elements during the requested computation.
	 * \note    When applying element-wise operators on sparse vectors
	 *          using semirings, there is a difference between interpreting missing
	 *          values as an annihilating identity or as a neutral identity--
	 *          intuitively, identities are known as `zero' or `one',
	 *          respectively. As a consequence, there are three different variants
	 *          for element-wise operations whose names correspond to their
	 *          intuitive meanings w.r.t. those identities:
	 *            -# eWiseAdd,
	 *            -# eWiseMul, and
	 *            -# eWiseMulAdd.
	 *
	 * @tparam descr The descriptor to be used. Equal to descriptors::no_operation
	 *               if left unspecified.
	 * @tparam OP    The operator to use.
	 * @tparam InputType1      The value type of the left-hand scalar.
	 * @tparam InputType2      The value type of the right-hand side vector.
	 * @tparam OutputStructure The value Structure of the ouput vector.
	 * @tparam InputStructure2 The value Structure of the right-hand side vector.
	 * @tparam OutputView      The view type of the ouput vector.
	 * @tparam InputView2      The view type of the right-hand side vector.
	 *
	 * @param[in]  alpha The left-hand scalar.
	 * @param[in]   y    The right-hand input vector.
	 * @param[out]  z    The pre-allocated output vector.
	 * @param[in]   op   The operator to use.
	 *
	 * @return alp::MISMATCH Whenever the dimensions of \a y and \a z do not
	 *                       match. All input data containers are left untouched
	 *                       if this exit code is returned; it will be as though
	 *                       this call was never made.
	 * @return alp::SUCCESS  On successful completion of this call.
	 *
	//  * \parblock
	//  * \par Performance semantics
	//  *      -# This call comprises \f$ \Theta(n) \f$ work, where \f$ n \f$ equals
	//  *         the size of the vectors \a y and \a z. The constant factor depends
	//  *         on the cost of evaluating the operator. A good implementation uses
	//  *         vectorised instructions whenever the input domains, the output
	//  *         domain, and the operator used allow for this.
	//  *
	//  *      -# This call takes \f$ \mathcal{O}(1) \f$ memory beyond the memory
	//  *         used by the application at the point of a call to this function.
	//  *
	//  *      -# This call incurs at most
	//  *         \f$ n(
	//  *               \mathit{sizeof}(\mathit{D2}) + \mathit{sizeof}(\mathit{D3})
	//  *             ) +
	//  *         \mathcal{O}(1) \f$
	//  *         bytes of data movement. A good implementation will stream \a y
	//  *         into \a z to apply the multiplication operator in-place, whenever
	//  *         the input domains, the output domain, and the operator allow for
	//  *         this.
	//  * \endparblock
	 */
	template< Descriptor descr = descriptors::no_operation,
		typename OutputType, typename OutputStructure, typename OutputView, typename OutputImfR, typename OutputImfC,
		typename InputType1, typename InputStructure1,
		typename InputType2, typename InputStructure2, typename InputView2, typename InputImfR2, typename InputImfC2,
		class OP
	>
	RC eWiseApply( Vector< OutputType, OutputStructure, Density::Dense, OutputView, OutputImfR, OutputImfC, reference > & z,
		const Scalar< InputType1, InputStructure1, reference > &alpha,
		const Vector< InputType2, InputStructure2, Density::Dense, InputView2, InputImfR2, InputImfC2, reference > & y,
		const OP & op = OP(),
		const typename std::enable_if< ! alp::is_object< OutputType >::value && ! alp::is_object< InputType1 >::value && ! alp::is_object< InputType2 >::value && alp::is_operator< OP >::value,
			void >::type * const = NULL ) {
	#ifdef _DEBUG
		std::cout << "In eWiseApply ([T1]<-T2<-[T3]), operator variant\n";
	#endif
		throw std::runtime_error( "Needs an implementation." );
		return SUCCESS;
	}

	/**
	 * Computes \f$ z = x \odot y \f$, out of place.
	 *
	 * Specialisation for scalar \a x. Masked operator version.
	 */
	template< Descriptor descr = descriptors::no_operation,
		typename OutputType, typename OutputStructure, typename OutputView, typename OutputImfR, typename OutputImfC,
		typename MaskType, typename MaskStructure, typename MaskView, typename MaskImfR, typename MaskImfC,
		typename InputType1, typename InputStructure1,
		typename InputType2, typename InputStructure2, typename InputView2, typename InputImfR2, typename InputImfC2,
		class OP
	>
	RC eWiseApply( Vector< OutputType, OutputStructure, Density::Dense, OutputView, OutputImfR, OutputImfC, reference > & z,
		const Vector< MaskType, MaskStructure, Density::Dense, MaskView, MaskImfR, MaskImfC, reference > & mask,
		const Scalar< InputType1, InputStructure1, reference> &alpha,
		const Vector< InputType2, InputStructure2, Density::Dense, InputView2, InputImfR2, InputImfC2, reference > & y,
		const OP & op = OP(),
		const typename std::enable_if< ! alp::is_object< OutputType >::value && ! alp::is_object< MaskType >::value && ! alp::is_object< InputType1 >::value && ! alp::is_object< InputType2 >::value &&
				alp::is_operator< OP >::value,
			void >::type * const = NULL ) {
	#ifdef _DEBUG
		std::cout << "In masked eWiseApply ([T1]<-T2<-[T3], operator variant)\n";
	#endif
		throw std::runtime_error( "Needs an implementation." );
		return SUCCESS;
	}

	/**
	 * Calculates the element-wise operation on elements of two vectors,
	 * \f$ z = x .* y \f$, using the given operator. The vectors must be
	 * of equal length.
	 *
	 * The vectors \a x or \a y may not be sparse.
	 *
	 * For all valid indices \a i of \a z, its element \f$ z_i \f$ after
	 * the call to this function completes equals \f$ x_i \odot y_i \f$.
	 *
	 * \warning Use of sparse vectors is only supported in full generality
	 *          when applied via a monoid or semiring; otherwise, there is
	 *          no concept for correctly interpreting any missing vector
	 *          elements during the requested computation.
	 * \note    When applying element-wise operators on sparse vectors
	 *          using semirings, there is a difference between interpreting missing
	 *          values as an annihilating identity or as a neutral identity--
	 *          intuitively, identities are known as `zero' or `one',
	 *          respectively. As a consequence, there are three different variants
	 *          for element-wise operations whose names correspond to their
	 *          intuitive meanings w.r.t. those identities:
	 *            -# eWiseAdd,
	 *            -# eWiseMul, and
	 *            -# eWiseMulAdd.
	 *
	 * @tparam descr The descriptor to be used (descriptors::no_operation if left
	 *               unspecified).
	 * @tparam OP    The operator to use.
	 * @tparam InputType1      The value type of the left-hand side vector.
	 * @tparam InputType2      The value type of the right-hand side vector.
	 * @tparam OutputType      The value type of the ouput vector.
	 * @tparam InputStructure1 The structure of the left-hand side vector.
	 * @tparam InputStructure2 The structure of the right-hand side vector.
	 * @tparam OutputStructure The structure of the ouput vector.
	 * @tparam InputView1      The value View of the left-hand side vector.
	 * @tparam InputView2      The value View of the right-hand side vector.
	 * @tparam OutputView      The value View of the ouput vector.
	 *
	 * @param[in]  x  The left-hand input vector. May not equal \a y.
	 * @param[in]  y  The right-hand input vector. May not equal \a x.
	 * @param[out] z  The pre-allocated output vector.
	 * @param[in]  op The operator to use.
	 *
	 * @return alp::ILLEGAL  When \a x equals \a y.
	 * @return alp::MISMATCH Whenever the dimensions of \a x, \a y, and \a z
	 *                       do not match. All input data containers are left
	 *                       untouched if this exit code is returned; it will
	 *                       be as though this call was never made.
	 * @return alp::SUCCESS  On successful completion of this call.
	 *
	//  * \parblock
	//  * \par Performance semantics
	//  *      -# This call comprises \f$ \Theta(n) \f$ work, where \f$ n \f$ equals
	//  *         the size of the vectors \a x, \a y, and \a z. The constant factor
	//  *         depends on the cost of evaluating the operator. A good
	//  *         implementation uses vectorised instructions whenever the input
	//  *         domains, the output domain, and the operator used allow for this.
	//  *
	//  *      -# This call takes \f$ \mathcal{O}(1) \f$ memory beyond the memory
	//  *         used by the application at the point of a call to this function.
	//  *
	//  *      -# This call incurs at most
	//  *         \f$ n(
	//  *               \mathit{sizeof}(\mathit{OutputType}) +
	//  *               \mathit{sizeof}(\mathit{InputType1}) +
	//  *               \mathit{sizeof}(\mathit{InputType2})
	//  *             ) +
	//  *         \mathcal{O}(1) \f$
	//  *         bytes of data movement. A good implementation will stream \a x or
	//  *         \a y into \a z to apply the multiplication operator in-place,
	//  *         whenever the input domains, the output domain, and the operator
	//  *         used allow for this.
	//  * \endparblock
	 */
	template< Descriptor descr = descriptors::no_operation,
		typename OutputType, typename OutputStructure, typename OutputView, typename OutputImfR, typename OutputImfC,
		typename InputType1, typename InputStructure1, typename InputView1, typename InputImfR1, typename InputImfC1,
		typename InputType2, typename InputStructure2, typename InputView2, typename InputImfR2, typename InputImfC2,
		class OP
	>
	RC eWiseApply( Vector< OutputType, OutputStructure, Density::Dense, OutputView, OutputImfR, OutputImfC, reference > & z,
		const Vector< InputType1, InputStructure1, Density::Dense, InputView1, InputImfR1, InputImfC1, reference > & x,
		const Vector< InputType2, InputStructure2, Density::Dense, InputView2, InputImfR2, InputImfC2, reference > & y,
		const OP & op = OP(),
		const typename std::enable_if< ! alp::is_object< OutputType >::value && ! alp::is_object< InputType1 >::value && ! alp::is_object< InputType2 >::value && alp::is_operator< OP >::value,
			void >::type * const = NULL ) {
	#ifdef _DEBUG
		std::cout << "In eWiseApply ([T1]<-[T2]<-[T3]), operator variant\n";
	#endif
		throw std::runtime_error( "Needs an implementation." );
		return SUCCESS;
	}

	/**
	 * Computes \f$ z = x \odot y \f$, out of place.
	 *
	 * Masked operator version.
	 */
	template< Descriptor descr = descriptors::no_operation,
		typename OutputType, typename OutputStructure, typename OutputView, typename OutputImfR, typename OutputImfC,
		typename MaskType, typename MaskStructure, typename MaskView, typename MaskImfR, typename MaskImfC,
		typename InputType1, typename InputStructure1, typename InputView1, typename InputImfR1, typename InputImfC1,
		typename InputType2, typename InputStructure2, typename InputView2, typename InputImfR2, typename InputImfC2,
		class OP
	>
	RC eWiseApply( Vector< OutputType, OutputStructure, Density::Dense, OutputView, OutputImfR, OutputImfC, reference > & z,
		const Vector< MaskType, MaskStructure, Density::Dense, MaskView, MaskImfR, MaskImfC, reference > & mask,
		const Vector< InputType1, InputStructure1, Density::Dense, InputView1, InputImfR1, InputImfC1, reference > & x,
		const Vector< InputType2, InputStructure2, Density::Dense, InputView2, InputImfR2, InputImfC2, reference > & y,
		const OP & op = OP(),
		const typename std::enable_if< ! alp::is_object< OutputType >::value && ! alp::is_object< MaskType >::value && ! alp::is_object< InputType1 >::value && ! alp::is_object< InputType2 >::value &&
				alp::is_operator< OP >::value,
			void >::type * const = NULL ) {
	#ifdef _DEBUG
		std::cout << "In masked eWiseApply ([T1]<-[T2]<-[T3], using operator)\n";
	#endif
		throw std::runtime_error( "Needs an implementation." );
		return SUCCESS;
	}

	/**
	 * Calculates the element-wise multiplication of two vectors,
	 *     \f$ z = z + x .* y \f$,
	 * under a given semiring.
	 *
	 * @tparam descr      The descriptor to be used (descriptors::no_operation
	 *                    if left unspecified).
	 * @tparam Ring       The semiring type to perform the element-wise multiply
	 *                    on.
	 * @tparam InputType1 The left-hand side input type to the multiplicative
	 *                    operator of the \a ring.
	 * @tparam InputType2 The right-hand side input type to the multiplicative
	 *                    operator of the \a ring.
	 * @tparam OutputType The the result type of the multiplicative operator of
	 *                    the \a ring.
	 * @tparam InputStructure1  The structure of the left-hand side input to
	 *                          the multiplicative operator of the \a ring.
	 * @tparam InputStructure2  The structure of the right-hand side input
	 *                          to the multiplicative operator of the \a ring.
	 * @tparam OutputStructure1 The structure of the output to the
	 *                          multiplicative operator of the \a ring.
	 * @tparam InputView1       The view type applied to the left-hand side
	 *                          input to the multiplicative operator
	 *                          of the \a ring.
	 * @tparam InputView2       The view type applied to the right-hand side
	 *                          input to the multiplicative operator
	 *                          of the \a ring.
	 * @tparam OutputView1      The view type applied to the output to the
	 *                          multiplicative operator of the \a ring.
	 *
	 * @param[out]  z  The output vector of type \a OutputType.
	 * @param[in]   x  The left-hand input vector of type \a InputType1.
	 * @param[in]   y  The right-hand input vector of type \a InputType2.
	 * @param[in] ring The generalized semiring under which to perform this
	 *                 element-wise multiplication.
	 *
	 * @return alp::MISMATCH Whenever the dimensions of \a x, \a y, and \a z do
	 *                       not match. All input data containers are left
	 *                       untouched if this exit code is returned; it will be
	 *                       as though this call was never made.
	 * @return alp::SUCCESS  On successful completion of this call.
	 *
	 * \parblock
	 * \par Valid descriptors
	 * alp::descriptors::no_operation, alp::descriptors::no_casting.
	 *
	 * \note Invalid descriptors will be ignored.
	 *
	 * If alp::descriptors::no_casting is specified, then 1) the first domain of
	 * \a ring must match \a InputType1, 2) the second domain of \a ring must match
	 * \a InputType2, 3) the third domain of \a ring must match \a OutputType. If
	 * one of these is not true, the code shall not compile.
	 *
	 * \endparblock
	 *
	//  * \parblock
	//  * \par Performance semantics
	//  *      -# This call takes \f$ \Theta(n) \f$ work, where \f$ n \f$ equals the
	//  *         size of the vectors \a x, \a y, and \a z. The constant factor
	//  *         depends on the cost of evaluating the multiplication operator. A
	//  *         good implementation uses vectorised instructions whenever the input
	//  *         domains, the output domain, and the multiplicative operator used
	//  *         allow for this.
	//  *
	//  *      -# This call will not result in additional dynamic memory allocations.
	//  *
	//  *      -# This call takes \f$ \mathcal{O}(1) \f$ memory beyond the memory
	//  *         used by the application at the point of a call to this function.
	//  *
	//  *      -# This call incurs at most \f$ n( \mathit{sizeof}(\mathit{D1}) +
	//  *         \mathit{sizeof}(\mathit{D2}) + \mathit{sizeof}(\mathit{D3})) +
	//  *         \mathcal{O}(1) \f$ bytes of data movement. A good implementation
	//  *         will stream \a x or \a y into \a z to apply the multiplication
	//  *         operator in-place, whenever the input domains, the output domain,
	//  *         and the operator used allow for this.
	//  * \endparblock
	 *
	 * \warning When given sparse vectors, the zero now annihilates instead of
	 *       acting as an identity. Thus the eWiseMul cannot simply map to an
	 *       eWiseApply of the multiplicative operator.
	 *
	 * @see This is a specialised form of eWiseMulAdd.
	 */
	template< Descriptor descr = descriptors::no_operation,
		typename OutputType, typename OutputStructure, typename OutputView, typename OutputImfR, typename OutputImfC,
		typename InputType1, typename InputStructure1, typename InputView1, typename InputImfR1, typename InputImfC1,
		typename InputType2, typename InputStructure2, typename InputView2, typename InputImfR2, typename InputImfC2,
		class Ring
	>
	RC eWiseMul( Vector< OutputType, OutputStructure, Density::Dense, OutputView, OutputImfR, OutputImfC, reference > & z,
		const Vector< InputType1, InputStructure1, Density::Dense, InputView1, InputImfR1, InputImfC1, reference > & x,
		const Vector< InputType2, InputStructure2, Density::Dense, InputView2, InputImfR2, InputImfC2, reference > & y,
		const Ring & ring = Ring(),
		const typename std::enable_if< ! alp::is_object< OutputType >::value && ! alp::is_object< InputType1 >::value && ! alp::is_object< InputType2 >::value && alp::is_semiring< Ring >::value,
			void >::type * const = NULL ) {
		// static sanity checks
		NO_CAST_OP_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename Ring::D1, InputType1 >::value ), "alp::eWiseMul",
			"called with a left-hand side input vector with element type that does not "
			"match the first domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename Ring::D2, InputType2 >::value ), "alp::eWiseMul",
			"called with a right-hand side input vector with element type that does "
			"not match the second domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename Ring::D3, OutputType >::value ), "alp::eWiseMul",
			"called with an output vector with element type that does not match the "
			"third domain of the given semiring" );
	#ifdef _DEBUG
		std::cout << "eWiseMul (reference, vector <- vector x vector) dispatches to eWiseMulAdd (vector <- vector x vector + 0)\n";
	#endif
		// return eWiseMulAdd< descr >( z, x, y, ring.template getZero< Ring::D4 >(), ring );
		return PANIC;
	}

	/**
	 * Computes \f$ z = z + x * y \f$.
	 *
	 * Specialisation for scalar \a x.
	 */
	template< Descriptor descr = descriptors::no_operation,
		typename OutputType, typename OutputStructure, typename OutputView, typename OutputImfR, typename OutputImfC,
		typename InputType1, typename InputStructure1,
		typename InputType2, typename InputStructure2, typename InputView2, typename InputImfR2, typename InputImfC2,
		class Ring
	>
	RC eWiseMul( Vector< OutputType, OutputStructure, Density::Dense, OutputView, OutputImfR, OutputImfC, reference > & z,
		const Scalar< InputType1, InputStructure1, reference > &alpha,
		const Vector< InputType2, InputStructure2, Density::Dense, InputView2, InputImfR2, InputImfC2, reference > & y,
		const Ring & ring = Ring(),
		const typename std::enable_if< ! alp::is_object< OutputType >::value && ! alp::is_object< InputType1 >::value && ! alp::is_object< InputType2 >::value && alp::is_semiring< Ring >::value,
			void >::type * const = NULL ) {
		// static sanity checks
		NO_CAST_OP_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename Ring::D1, InputType1 >::value ), "alp::eWiseMul",
			"called with a left-hand side input vector with element type that does not "
			"match the first domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename Ring::D2, InputType2 >::value ), "alp::eWiseMul",
			"called with a right-hand side input vector with element type that does "
			"not match the second domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename Ring::D3, OutputType >::value ), "alp::eWiseMul",
			"called with an output vector with element type that does not match the "
			"third domain of the given semiring" );
	#ifdef _DEBUG
		std::cout << "eWiseMul (reference, vector <- scalar x vector) dispatches to eWiseMulAdd (vector <- scalar x vector + 0)\n";
	#endif
		// return eWiseMulAdd< descr >( z, alpha, y, ring.template getZero< typename Ring::D4 >(), ring );
		return PANIC;
	}

	/**
	 * Computes \f$ z = z + x * y \f$.
	 *
	 * Specialisation for scalar \a y.
	 */
	template< Descriptor descr = descriptors::no_operation,
		typename OutputType, typename OutputStructure, typename OutputView, typename OutputImfR, typename OutputImfC,
		typename InputType1, typename InputStructure1, typename InputView1, typename InputImfR1, typename InputImfC1,
		typename InputType2, typename InputStructure2,
		class Ring
	>
	RC eWiseMul( Vector< OutputType, OutputStructure, Density::Dense, OutputView, OutputImfR, OutputImfC, reference > & z,
		const Vector< InputType1, InputStructure1, Density::Dense, InputView1, InputImfR1, InputImfC1, reference > & x,
		const Scalar< InputType2, InputStructure2, reference > &beta,
		const Ring & ring = Ring(),
		const typename std::enable_if< ! alp::is_object< OutputType >::value && ! alp::is_object< InputType1 >::value && ! alp::is_object< InputType2 >::value && alp::is_semiring< Ring >::value,
			void >::type * const = NULL ) {
		// static sanity checks
		NO_CAST_OP_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename Ring::D1, InputType1 >::value ), "alp::eWiseMul",
			"called with a left-hand side input vector with element type that does not "
			"match the first domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename Ring::D2, InputType2 >::value ), "alp::eWiseMul",
			"called with a right-hand side input vector with element type that does "
			"not match the second domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename Ring::D3, OutputType >::value ), "alp::eWiseMul",
			"called with an output vector with element type that does not match the "
			"third domain of the given semiring" );
	#ifdef _DEBUG
		std::cout << "eWiseMul (reference) dispatches to eWiseMulAdd with 0.0 as additive scalar\n";
	#endif
		// return eWiseMulAdd< descr >( z, x, beta, ring.template getZero< typename Ring::D4 >(), ring.getMultiplicativeOperator() );
		return PANIC;
	}

	// internal namespace for implementation of alp::dot
	namespace internal {

		/** @see alp::dot */
		template<
			Descriptor descr = descriptors::no_operation,
			typename OutputType, typename OutputStructure,
			typename InputType1, typename InputStructure1, typename InputView1, typename InputImfR1, typename InputImfC1,
			typename InputType2, typename InputStructure2, typename InputView2, typename InputImfR2, typename InputImfC2,
			class AddMonoid, class AnyOp
		>
		RC dot_generic( Scalar< OutputType, OutputStructure, reference > &z,
			const alp::Vector< InputType1, InputStructure1, Density::Dense, InputView1, InputImfR1, InputImfC1, reference > &x,
			const alp::Vector< InputType2, InputStructure2, Density::Dense, InputView2, InputImfR2, InputImfC2, reference > &y,
			const AddMonoid &addMonoid = AddMonoid(),
			const AnyOp &anyOp = AnyOp()
		) {
			throw std::runtime_error( "Needs an implementation." );
			return SUCCESS;
		}

	} // namespace internal

	/**
	 * Calculates the dot product, \f$ \alpha = (x,y) \f$, under a given additive
	 * monoid and multiplicative operator.
	 *
	 * @tparam descr      The descriptor to be used (descriptors::no_operation
	 *                    if left unspecified).
	 * @tparam Ring       The semiring type to use.
	 * @tparam OutputType The output type.
	 * @tparam InputType1 The input element type of the left-hand input vector.
	 * @tparam InputType2 The input element type of the right-hand input vector.
	 * @tparam InputStructure1  The structure of the left-hand input vector.
	 * @tparam InputStructure2  The structure of the right-hand input vector.
	 * @tparam InputView1       The view type applied to the left-hand input vector.
	 * @tparam InputView2       The view type applied to the right-hand input vector.
	 *
	 * @param[in,out]  z    The output element \f$ z + \alpha \f$.
	 * @param[in]      x    The left-hand input vector.
	 * @param[in]      y    The right-hand input vector.
	 * @param[in] addMonoid The additive monoid under which the reduction of the
	 *                      results of element-wise multiplications of \a x and
	 *                      \a y are performed.
	 * @param[in]   anyop   The multiplicative operator under which element-wise
	 *                      multiplications of \a x and \a y are performed. This can
	 *                      be any binary operator.
	 *
	 * By the definition that a dot-product operates under any additive monoid and
	 * any binary operator, it follows that a dot-product under any semiring can be
	 * trivially reduced to a call to this version instead.
	 *
	 * @return alp::MISMATCH When the dimensions of \a x and \a y do not match. All
	 *                       input data containers are left untouched if this exit
	 *                       code is returned; it will be as though this call was
	 *                       never made.
	 * @return alp::SUCCESS  On successful completion of this call.
	 *
	//  * \parblock
	//  * \par Performance semantics
	//  *      -# This call takes \f$ \Theta(n/p) \f$ work at each user process, where
	//  *         \f$ n \f$ equals the size of the vectors \a x and \a y, and
	//  *         \f$ p \f$ is the number of user processes. The constant factor
	//  *         depends on the cost of evaluating the addition and multiplication
	//  *         operators. A good implementation uses vectorised instructions
	//  *         whenever the input domains, output domain, and the operators used
	//  *         allow for this.
	//  *
	//  *      -# This call takes \f$ \mathcal{O}(1) \f$ memory beyond the memory used
	//  *         by the application at the point of a call to this function.
	//  *
	//  *      -# This call incurs at most
	//  *         \f$ n( \mathit{sizeof}(\mathit{D1}) + \mathit{sizeof}(\mathit{D2}) ) + \mathcal{O}(p) \f$
	//  *         bytes of data movement.
	//  *
	//  *      -# This call incurs at most \f$ \Theta(\log p) \f$ synchronisations
	//  *         between two or more user processes.
	//  *
	//  *      -# A call to this function does result in any system calls.
	//  * \endparblock
	 *
	 * \note This requires an implementation to pre-allocate \f$ \Theta(p) \f$
	 *       memory for inter-process reduction, if the underlying communication
	 *       layer indeed requires such a buffer. This buffer may not be allocated
	 *       (nor freed) during a call to this function.
	 *
	 * \parblock
	 * \par Valid descriptors
	 *   -# alp::descriptors::no_operation
	 *   -# alp::descriptors::no_casting
	 *   -# alp::descriptors::dense
	 * \endparblock
	 *
	 * If the dense descriptor is set, this implementation returns alp::ILLEGAL if
	 * it was detected that either \a x or \a y was sparse. In this case, it shall
	 * otherwise be as though the call to this function had not occurred (no side
	 * effects).
	 *
	 * \note The standard, in contrast, only specifies undefined behaviour would
	 *       occur. This implementation goes beyond the standard by actually
	 *       specifying what will happen.
	 */
	template<
		Descriptor descr = descriptors::no_operation,
		typename OutputType, typename OutputStructure,
		typename InputType1, typename InputStructure1, typename InputView1, typename InputImfR1, typename InputImfC1,
		typename InputType2, typename InputStructure2, typename InputView2, typename InputImfR2, typename InputImfC2,
		class AddMonoid, class AnyOp
	>
	RC dot(
		Scalar< OutputType, OutputStructure, reference > &z,
		const Vector< InputType1, InputStructure1, Density::Dense, InputView1, InputImfR1, InputImfC1, reference > &x,
		const Vector< InputType2, InputStructure2, Density::Dense, InputView2, InputImfR2, InputImfC2, reference > &y,
		const AddMonoid &addMonoid = AddMonoid(),
		const AnyOp &anyOp = AnyOp(),
		const typename std::enable_if_t< !alp::is_object< OutputType >::value &&
			!alp::is_object< InputType1 >::value &&
			!alp::is_object< InputType2 >::value &&
			alp::is_monoid< AddMonoid >::value &&
			alp::is_operator< AnyOp >::value
		> * const = nullptr
	) {
		// static sanity checks
		NO_CAST_ASSERT( ( !( descr & descriptors::no_casting ) || std::is_same< InputType1, typename AnyOp::D1 >::value ), "alp::dot",
			"called with a left-hand vector value type that does not match the first "
			"domain of the given multiplicative operator" );
		NO_CAST_ASSERT( ( !( descr & descriptors::no_casting ) || std::is_same< InputType2, typename AnyOp::D2 >::value ), "alp::dot",
			"called with a right-hand vector value type that does not match the second "
			"domain of the given multiplicative operator" );
		NO_CAST_ASSERT( ( !( descr & descriptors::no_casting ) || std::is_same< typename AddMonoid::D3, typename AnyOp::D1 >::value ), "alp::dot",
			"called with a multiplicative operator output domain that does not match "
			"the first domain of the given additive operator" );
		NO_CAST_ASSERT( ( !( descr & descriptors::no_casting ) || std::is_same< OutputType, typename AddMonoid::D2 >::value ), "alp::dot",
			"called with an output vector value type that does not match the second "
			"domain of the given additive operator" );
		NO_CAST_ASSERT( ( !( descr & descriptors::no_casting ) || std::is_same< typename AddMonoid::D3, typename AddMonoid::D2 >::value ), "alp::dot",
			"called with an additive operator whose output domain does not match its "
			"second input domain" );
		NO_CAST_ASSERT( ( !( descr & descriptors::no_casting ) || std::is_same< OutputType, typename AddMonoid::D3 >::value ), "alp::dot",
			"called with an output vector value type that does not match the third "
			"domain of the given additive operator" );
		(void)z;
		if( size( x ) != size( y ) ) {
			return MISMATCH;
		}

		if( !( internal::getInitialized( z ) && internal::getInitialized( x ) && internal::getInitialized( y ) ) ) {
#ifdef _DEBUG
			std::cout << "dot(): one of input vectors or scalar are not initialized: do noting!\n";
#endif
			return SUCCESS;
		}

		std::function< void( typename AddMonoid::D3 &, const size_t, const size_t ) > data_lambda =
			[ &x, &y, &anyOp ]( typename AddMonoid::D3 &result, const size_t i, const size_t j ) {
				(void) j;
				internal::apply( result, x[ i ], y[ i ], anyOp );
			};

		std::function< bool() > init_lambda =
			[ &x ]() -> bool {
				return internal::getInitialized( x );
			};

		Vector<
			typename AddMonoid::D3,
			structures::General,
			Density::Dense,
			view::Functor< std::function< void( typename AddMonoid::D3 &, const size_t, const size_t ) > >,
			imf::Id, imf::Id,
			reference
		> temp(
			init_lambda,
			getLength( x ),
			data_lambda
		);
		RC rc = foldl( z, temp, addMonoid );
		return rc;
	}

	/** C++ scalar specialization */
	template<
		Descriptor descr = descriptors::no_operation,
		typename OutputType,
		typename InputType1, typename InputStructure1, typename InputView1, typename InputImfR1, typename InputImfC1,
		typename InputType2, typename InputStructure2, typename InputView2, typename InputImfR2, typename InputImfC2,
		class AddMonoid, class AnyOp
	>
	RC dot( OutputType &z,
		const Vector< InputType1, InputStructure1, Density::Dense, InputView1, InputImfR1, InputImfC1, reference > &x,
		const Vector< InputType2, InputStructure2, Density::Dense, InputView2, InputImfR2, InputImfC2, reference > &y,
		const AddMonoid &addMonoid = AddMonoid(),
		const AnyOp &anyOp = AnyOp(),
		const typename std::enable_if< !alp::is_object< OutputType >::value &&
			!alp::is_object< InputType1 >::value &&
			!alp::is_object< InputType2 >::value &&
			alp::is_monoid< AddMonoid >::value &&
			alp::is_operator< AnyOp >::value,
		void >::type * const = NULL
	) {
		// static sanity checks
		NO_CAST_ASSERT( ( !( descr & descriptors::no_casting ) || std::is_same< InputType1, typename AnyOp::D1 >::value ), "alp::dot",
			"called with a left-hand vector value type that does not match the first "
			"domain of the given multiplicative operator" );
		NO_CAST_ASSERT( ( !( descr & descriptors::no_casting ) || std::is_same< InputType2, typename AnyOp::D2 >::value ), "alp::dot",
			"called with a right-hand vector value type that does not match the second "
			"domain of the given multiplicative operator" );
		NO_CAST_ASSERT( ( !( descr & descriptors::no_casting ) || std::is_same< typename AddMonoid::D3, typename AnyOp::D1 >::value ), "alp::dot",
			"called with a multiplicative operator output domain that does not match "
			"the first domain of the given additive operator" );
		NO_CAST_ASSERT( ( !( descr & descriptors::no_casting ) || std::is_same< OutputType, typename AddMonoid::D2 >::value ), "alp::dot",
			"called with an output vector value type that does not match the second "
			"domain of the given additive operator" );
		NO_CAST_ASSERT( ( !( descr & descriptors::no_casting ) || std::is_same< typename AddMonoid::D3, typename AddMonoid::D2 >::value ), "alp::dot",
			"called with an additive operator whose output domain does not match its "
			"second input domain" );
		NO_CAST_ASSERT( ( !( descr & descriptors::no_casting ) || std::is_same< OutputType, typename AddMonoid::D3 >::value ), "alp::dot",
			"called with an output vector value type that does not match the third "
			"domain of the given additive operator" );
		Scalar< OutputType, structures::General, reference > res( z );
		RC rc = dot( res, x, y, addMonoid, anyOp );
		if( rc != SUCCESS ) {
			return rc;
		}
		/** \internal \todo: extract res.value into z */
		return SUCCESS;
	}

	/**
	 * Provides a generic implementation of the dot computation on semirings by
	 * translating it into a dot computation on an additive commutative monoid
	 * with any multiplicative operator.
	 *
	 * For return codes, exception behaviour, performance semantics, template
	 * and non-template arguments, @see alp::dot.
	 */
	template<
		Descriptor descr = descriptors::no_operation,
		typename IOType, typename IOStructure,
		typename InputType1, typename InputStructure1, typename InputView1, typename InputImfR1, typename InputImfC1,
		typename InputType2, typename InputStructure2, typename InputView2, typename InputImfR2, typename InputImfC2,
		class Ring,
		Backend backend >
	RC dot( Scalar< IOType, IOStructure, backend > &x,
		const Vector< InputType1, InputStructure1, Density::Dense, InputView1, InputImfR1, InputImfC1, backend > &left,
		const Vector< InputType2, InputStructure2, Density::Dense, InputView2, InputImfR2, InputImfC2, backend > &right,
		const Ring &ring = Ring(),
		const typename std::enable_if<
			!alp::is_object< InputType1 >::value &&
			!alp::is_object< InputType2 >::value &&
			!alp::is_object< IOType >::value &&
			alp::is_semiring< Ring >::value,
		void >::type * const = NULL
	) {
		return alp::dot< descr >( x,
		// return alp::dot( x,
			left, right,
			ring.getAdditiveMonoid(),
			ring.getMultiplicativeOperator()
		);
	}

	/** C++ scalar specialization. */
	template<
		Descriptor descr = descriptors::no_operation, class Ring,
		typename IOType,
		typename InputType1, typename InputStructure1, typename InputView1, typename InputImfR1, typename InputImfC1,
		typename InputType2, typename InputStructure2, typename InputView2, typename InputImfR2, typename InputImfC2,
		Backend backend
	>
	RC dot( IOType &x,
		const Vector< InputType1, InputStructure1, Density::Dense, InputView1, InputImfR1, InputImfC1, backend > &left,
		const Vector< InputType2, InputStructure2, Density::Dense, InputView2, InputImfR2, InputImfC2, backend > &right,
		const Ring &ring = Ring(),
		const typename std::enable_if<
			!alp::is_object< InputType1 >::value &&
			!alp::is_object< InputType2 >::value &&
			!alp::is_object< IOType >::value &&
			alp::is_semiring< Ring >::value,
		void >::type * const = NULL
	) {
		Scalar< IOType, structures::General, backend > res( x );
		RC rc = alp::dot< descr >( x,
			left, right,
			ring.getAdditiveMonoid(),
			ring.getMultiplicativeOperator()
		);
		if( rc != SUCCESS ) {
			return rc;
		}
		/** \internal \todo extract res.value into x */
		return SUCCESS;
	}

	/** No implementation notes. */
	template< typename Func,
		typename DataType, typename DataStructure, typename DataView, typename DataImfR, typename DataImfC
	>
	RC eWiseMap( const Func f, Vector< DataType, DataStructure, Density::Dense, DataView, DataImfR, DataImfC, reference > & x ) {
		throw std::runtime_error( "Needs an implementation." );
		return SUCCESS;
	}

	/**
	 * This is the eWiseLambda that performs length checking by recursion.
	 *
	 * in the reference implementation all vectors are distributed equally, so no
	 * need to synchronise any data structures. We do need to do error checking
	 * though, to see when to return alp::MISMATCH. That's this function.
	 *
	 * @see Vector::operator[]()
	 * @see Vector::lambda_reference
	 */
	template<
		typename Func,
		typename DataType1, typename DataStructure1, typename DataView1, typename InputImfR1, typename InputImfC1,
		typename DataType2, typename DataStructure2, typename DataView2, typename InputImfR2, typename InputImfC2,
		typename... Args
	>
	RC eWiseLambda(
		const Func f,
		Vector< DataType1, DataStructure1, Density::Dense, DataView1, InputImfR1, InputImfC1, reference > &x,
		const Vector< DataType2, DataStructure2, Density::Dense, DataView2, InputImfR2, InputImfC2, reference > &y,
		Args const &... args
	) {
		// catch mismatch
		if( getLength( x ) != getLength( y ) ) {
			return MISMATCH;
		}
		// continue
		return eWiseLambda( f, x, args... );
	}

	/**
	 * No implementation notes. This is the `real' implementation on reference
	 * vectors.
	 *
	 * @see Vector::operator[]()
	 * @see Vector::lambda_reference
	 */
	template<
		typename Func,
		typename DataType, typename DataStructure, typename DataView, typename DataImfR, typename DataImfC
	>
	RC eWiseLambda( const Func f, Vector< DataType, DataStructure, Density::Dense, DataView, DataImfR, DataImfC, reference > &x ) {
#ifdef _DEBUG
		std::cout << "Info: entering eWiseLambda function on vectors.\n";
#endif
		auto x_as_matrix = get_view< view::matrix >( x );
		return eWiseLambda(
			[ &f ]( const size_t i, const size_t j, DataType &val ) {
				(void)j;
				f( i, val );
			},
			x_as_matrix
		);
	}

	/**
	 * Reduces a vector into a scalar. Reduction takes place according a monoid
	 * \f$ (\oplus,1) \f$, where \f$ \oplus:\ D_1 \times D_2 \to D_3 \f$ with an
	 * associated identity \f$ 1 \in \{D_1,D_2,D_3\} \f$. Elements from the given
	 * vector \f$ y \in \{D_1,D_2\} \f$ will be applied at the left-hand or right-
	 * hand side of \f$ \oplus \f$; which, exactly, is implementation-dependent
	 * but should not matter since \f$ \oplus \f$ should be associative.
	 *
	 * Let \f$ x_0 = 1 \f$ and let
	 * \f$ x_{i+1} = \begin{cases}
	 *   x_i \oplus y_i\text{ if }y_i\text{ is nonzero}
	 *   x_i\text{ otherwise}
	 * \end{cases},\f$
	 * for all \f$ i \in \{ 0, 1, \ldots, n-1 \} \f$. On function exit \a x will be
	 * set to \f$ x_n \f$.
	 *
	 * This function assumes that \f$ \odot \f$ under the given domains consitutes
	 * a valid monoid, which for standard associative operators it usually means
	 * that \f$ D_3 \subseteq D_2 \subseteq D_1 \f$. If not, or if the operator is
	 * non-standard, the monoid axioms are to be enforced in some other way-- the
	 * user is responsible for checking this is indeed the case or undefined
	 * behaviour will occur.
	 *
	 * \note While the monoid identity may be used to easily provide parallel
	 *       implementations of this function, having a notion of an identity is
	 *       mandatory to be able to interpret sparse vectors; this is why we do
	 *       not allow a plain operator to be passed to this function.
	 *
	 * @tparam descr     The descriptor to be used (descriptors::no_operation if
	 *                   left unspecified).
	 * @tparam Monoid    The monoid to use for reduction. A monoid is required
	 *                   because the output value \a y needs to be initialised
	 *                   with an identity first.
	 * @tparam InputType The type of the elements in the supplied GraphBLAS
	 *                   vector \a y.
	 * @tparam IOType    The type of the output value \a x.
	 *
	 * @param[out]   x   The result of reduction.
	 * @param[in]    y   A valid GraphBLAS vector. This vector may be sparse.
	 * @param[in] monoid The monoid under which to perform this reduction.
	 *
	 * @return alp::SUCCESS When the call completed successfully.
	 * @return alp::ILLEGAL If the provided input vector \a y was not dense.
	 * @return alp::ILLEGAL If the provided input vector \a y was empty.
	 *
	 * \parblock
	 * \par Valid descriptors
	 * alp::descriptors::no_operation, alp::descriptors::no_casting,
	 * alp::descriptors::dense
	 *
	 * \note Invalid descriptors will be ignored.
	 *
	 * If alp::descriptors::no_casting is specified, then 1) the first domain of
	 * \a monoid must match \a InputType, 2) the second domain of \a op must match
	 * \a IOType, and 3) the third domain must match \a IOType. If one of
	 * these is not true, the code shall not compile.
	 * \endparblock
	 *
	//  * \parblock
	//  * \par Performance semantics
	//  *      -# This call comprises \f$ \Theta(n) \f$ work, where \f$ n \f$ equals
	//  *         the size of the vector \a x. The constant factor depends on the
	//  *         cost of evaluating the underlying binary operator. A good
	//  *         implementation uses vectorised instructions whenever the input
	//  *         domains, the output domain, and the operator used allow for this.
	//  *
	//  *      -# This call will not result in additional dynamic memory allocations.
	//  *         No system calls will be made.
	//  *
	//  *      -# This call takes \f$ \mathcal{O}(1) \f$ memory beyond the memory
	//  *         used by the application at the point of a call to this function.
	//  *
	//  *      -# This call incurs at most
	//  *         \f$ n \mathit{sizeof}(\mathit{InputType}) + \mathcal{O}(1) \f$
	//  *         bytes of data movement. If \a y is sparse, a call to this function
	//  *         incurs at most \f$ n \mathit{sizeof}( \mathit{bool} ) \f$ extra
	//  *         bytes of data movement.
	//  * \endparblock
	 *
	 * @see alp::foldl provides similar functionality.
	 */
	template< Descriptor descr = descriptors::no_operation,
		typename IOType, typename IOStructure,
		typename InputType, typename InputStructure, typename InputView, typename InputImfR, typename InputImfC,
		class Monoid
	>
	RC foldl(
		Scalar< IOType, IOStructure, reference > &alpha,
		const Vector< InputType, InputStructure, Density::Dense, InputView, InputImfR, InputImfC, reference > &y,
		const Monoid &monoid = Monoid(),
		const typename std::enable_if_t<
			! alp::is_object< IOType >::value && ! alp::is_object< InputType >::value && alp::is_monoid< Monoid >::value
		> * const = nullptr
	) {

		// static sanity checks
		NO_CAST_ASSERT(
			( ! ( descr & descriptors::no_casting ) || std::is_same< IOType, InputType >::value ),
			"alp::reduce",
			"called with a scalar IO type that does not match the input vector type"
		);
		NO_CAST_OP_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< InputType, typename Monoid::D1 >::value ), "alp::reduce",
			"called with an input vector value type that does not match the first "
			"domain of the given monoid" );
		NO_CAST_OP_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< InputType, typename Monoid::D2 >::value ), "alp::reduce",
			"called with an input vector type that does not match the second domain of "
			"the given monoid" );
		NO_CAST_OP_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< InputType, typename Monoid::D3 >::value ), "alp::reduce",
			"called with an input vector type that does not match the third domain of "
			"the given monoid" );

#ifdef _DEBUG
		std::cout << "foldl(Scalar,Vector,Monoid) called. Vector has size " << getLength( y ) << " .\n";
#endif

		internal::setInitialized(
			alpha,
			internal::getInitialized( alpha ) && internal::getInitialized( y )
		);

		if( !internal::getInitialized( alpha ) ) {
			return SUCCESS;
		}

		const size_t n = getLength( y );
		for ( size_t i = 0; i < n; ++i ) {
			(void) internal::foldl( *alpha, y[ i ], monoid.getOperator() );
		}
		return SUCCESS;
	}

	/**
	 * Sort vectors, function available to user, e.g. to sort eigenvectors
	 *
	 * @param[in] toSort vector of indices to sort, should not be modified
	 * @param[in] cmp function with strict weak ordering relation between indices, eg bool cmp(const Type1 &a, const Type2 &b)
	 *            cmp must not modify the objects passed to it
	 *
	 * @param[out] permutation iterator over index permutations which sort toSort vector
	 *
	 * Complexity should be lower than O(n*log(n)), and space complexity should be lower than \Theta(n+T+P)
	 */
	template<
		typename IndexType, typename IndexStructure, typename IndexView, typename IndexImfR, typename IndexImfC,
		typename ValueType, typename ValueStructure, typename ValueView, typename ValueImfR, typename ValueImfC,
		typename Compare
	>
	RC sort(
		Vector< IndexType, IndexStructure, Density::Dense, IndexView, IndexImfR, IndexImfC, reference > &permutation,
		const Vector< ValueType, ValueStructure, Density::Dense, ValueView, ValueImfR, ValueImfC, reference > &toSort,
		Compare cmp
		//PHASE &phase = EXECUTE
	) noexcept {
		return SUCCESS;
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
	 * and non-template arguments, @see alp::dot.
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
	template<
		Descriptor descr = descriptors::no_operation,
		typename OutputType, typename OutputStructure,
		typename InputType, typename InputStructure, typename InputView, typename InputImfR, typename InputImfC,
		class Ring,
		Backend backend
	>
	RC norm2( Scalar< OutputType, OutputStructure, backend > &x,
		const Vector< InputType, InputStructure, Density::Dense, InputView, InputImfR, InputImfC, backend > &y,
		const Ring &ring = Ring(),
		const typename std::enable_if<
			std::is_floating_point< OutputType >::value,
		void >::type * const = NULL
	) {
		RC ret = alp::dot< descr >( x, y, y, ring );
		if( ret == SUCCESS ) {
			x = sqrt( x );
		}
		return ret;
	}

	/** C++ scalar version */
	template<
		Descriptor descr = descriptors::no_operation,
		typename OutputType,
		typename InputType, typename InputStructure, typename InputView, typename InputImfR, typename InputImfC,
		class Ring,
		Backend backend
	>
	RC norm2(
		OutputType &x,
		const Vector< InputType, InputStructure, Density::Dense, InputView, InputImfR, InputImfC, backend > &y,
		const Ring &ring = Ring(),
		const typename std::enable_if<
			std::is_floating_point< OutputType >::value,
		void >::type * const = nullptr
	) {
		Scalar< OutputType, structures::General, reference > res( x );
		RC rc = norm2( res, y, ring );
		if( rc != SUCCESS ) {
			return rc;
		}
		/** \internal \todo extract res.value into x */
		return SUCCESS;
	}

} // end namespace ``alp''

#undef NO_CAST_ASSERT

#endif // end ``_H_ALP_REFERENCE_BLAS1''

