
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

#if ! defined _H_GRB_BANSHEE_BLAS1 || defined _H_GRB_BANSHEE_OMP_BLAS1
#define _H_GRB_BANSHEE_BLAS1

#include <type_traits> //for std::enable_if

#include <graphblas/backends.hpp>
#include <graphblas/blas0.hpp>
#include <graphblas/descriptors.hpp>
#include <graphblas/internalops.hpp>
#include <graphblas/ops.hpp>
#include <graphblas/rc.hpp>
#include <graphblas/semiring.hpp>

#include "coordinates.hpp"
#include "vector.hpp"

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

namespace grb {

	namespace internal {

		template< Descriptor descr, typename OutputType, typename IndexType, typename ValueType >
		OutputType
		setIndexOrValue( const IndexType & index, const ValueType & value, const typename std::enable_if< std::is_convertible< IndexType, OutputType >::value, void >::type * const = NULL ) {
			if( descr & grb::descriptors::use_index ) {
				return static_cast< OutputType >( index );
			} else {
				return static_cast< OutputType >( value );
			}
		}

		template< Descriptor descr, typename OutputType, typename IndexType, typename ValueType >
		OutputType
		setIndexOrValue( const IndexType & index, const ValueType & value, const typename std::enable_if< ! std::is_convertible< IndexType, OutputType >::value, void >::type * const = NULL ) {
			(void)index;
			static_assert( ! ( descr & grb::descriptors::use_index ),
				"use_index descriptor passed while the index type cannot be cast "
				"to the output type" );
			return static_cast< OutputType >( value );
		}

	} // namespace internal

	/**
	 * \defgroup BLAS1 The Level-1 Basic Linear Algebra Subroutines (BLAS)
	 *
	 * A collection of functions that allow GraphBLAS operators, monoids, and
	 * semirings work on a mix of zero-dimensional and one-dimensional containers;
	 * i.e., allows various linear algebra operations on scalars and objects of
	 * type grb::Vector.
	 *
	 * All functions except for grb::size and grb::nnz return an error code of
	 * the enum-type grb::RC. The two functions for retrieving the size and the
	 * nonzeroes of two vectors are excluded from this because they are never
	 * allowed to fail.
	 *
	 * Operations which require a single input vector only and produce scalar
	 * output:
	 *   -# grb::size,
	 *   -# grb::nnz, and
	 *   -# grb::set (three variants).
	 * These do not require an operator, monoid, nor semiring. The following
	 * require an operator:
	 *   -# grb::foldr (reduction to the right),
	 *   -# grb::foldl (reduction to the left).
	 * Operators can only be applied on \em dense vectors. Operations on sparse
	 * vectors require a well-defined way to handle missing vector elements. The
	 * following functions require a monoid instead of an operator and are able
	 * to handle sparse vectors by interpreting missing items as an identity
	 * value:
	 *   -# grb::reducer (reduction to the right),
	 *   -# grb::reducel (reduction to the left).
	 *
	 * Operations which require two input vectors and produce scalar output:
	 *   -# grb::dot   (dot product-- requires a semiring).
	 * Sparse vectors under a semiring have their missing values interpreted as a
	 * zero element under the given semiring; i.e., the identity of the additive
	 * operator.
	 *
	 * Operations which require one input vector and one input/output vector for
	 * full and efficient in-place operations:
	 *   -# grb::foldr (reduction to the right-- requires an operator),
	 *   -# grb::foldl (reduction to the left-- requires an operator).
	 * For grb::foldr, the left-hand side input vector may be replaced by an
	 * input scalar. For grb::foldl, the right-hand side input vector may be
	 * replaced by an input scalar. In either of those cases, the reduction
	 * is equivalent to an in-place vector scaling.
	 *
	 * Operations which require two input vectors and one output vector for
	 * out-of-place operations:
	 *   -# grb::eWiseApply (requires an operator),
	 *   -# grb::eWiseMul   (requires a semiring),
	 *   -# grb::eWiseAdd   (requires a semiring).
	 * Note that multiplication will consider any zero elements as an annihilator
	 * to the multiplicative operator. Therefore, the operator will only be
	 * applied at vector indices where both input vectors have nonzeroes. This is
	 * different from eWiseAdd. This difference only manifests itself when dealing
	 * with semirings, and reflects the intuitively expected behaviour. Any of the
	 * two input vectors (or both) may be replaced with an input scalar instead.
	 *
	 * Operations which require three input vectors and one output vector for
	 * out-of-place operations:
	 *   -# grb::eWiseMulAdd (requires a semiring).
	 * This function can be emulated by first successive calls to grb::eWiseMul
	 * and grb::eWiseAdd. This specialised function, however, has better
	 * performance semantics. This function is closest to the standard axpy
	 * BLAS1 call, with out-of-place semantics. The first input vector may be
	 * replaced by a scalar.
	 *
	 * Again, each of grb::eWiseMul, grb::eWiseAdd, grb::eWiseMulAdd accept sparse
	 * vectors as input and output (since they operate on semirings), while
	 * grb::eWiseApply.
	 *
	 * For fusing multiple BLAS-1 style operations on any number of inputs and
	 * outputs, users can pass their own operator function to be executed for
	 * every index \a i.
	 *   -# grb::eWiseLambda.
	 * This requires manual application of operators, monoids, and/or semirings
	 * via the BLAS-0 interface (see grb::apply, grb::foldl, and grb::foldr).
	 *
	 * For all of these functions, the element types of input and output types
	 * do not have to match the domains of the given operator, monoid, or
	 * semiring unless the grb::descriptors::no_casting descriptor was passed.
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
	 * @return grb::SUCCESS When the vector is successfully cleared.
	 *
	 * \note This function cannot fail.
	 *
	 * \parblock
	 * \par Performance semantics
	 *      This function
	 *        -# contains \f$ \mathcal{O}(n) \f$ work,
	 *        -# will not allocate new dynamic memory,
	 *        -# will take at most \f$ \Theta(1) \f$ memory beyond the memory
	 *           already used by the application before the call to this
	 *           function.
	 *        -# will move at most \f$ \mathit{sizeof}(\mathit{bool}) +
	 *           \mathit{sizeof}(\mathit{size\_t}) \f$ bytes of data.
	 * \endparblock
	 */
	template< typename DataType, typename Coords >
	RC clear( Vector< DataType, banshee, Coords > & x ) noexcept {
		internal::getCoordinates( x ).clear();
		return SUCCESS;
	}

	/**
	 * Request the size (dimension) of a given vector.
	 *
	 * The dimension is set at construction of the given vector and cannot be
	 * changed. A call to this function shall always succeed.
	 *
	 * @tparam DataType The type of elements contained in the vector \a x.
	 *
	 * @param[in] x The vector of which to retrieve the size.
	 *
	 * @return The size of the vector \a x.
	 *
	 * \parblock
	 * \par Performance semantics
	 * A call to this function
	 *  -# consists of \f$ \Theta(1) \f$ work;
	 *  -# moves \f$ \Theta(1) \f$ bytes of memory;
	 *  -# does not allocate any dynamic memory;
	 *  -# shall not make any system calls.
	 * \endparblock
	 */
	template< typename DataType, typename Coords >
	size_t size( const Vector< DataType, banshee, Coords > & x ) noexcept {
		return internal::getCoordinates( x ).size();
	}

	/**
	 * Request the number of nonzeroes in a given vector.
	 *
	 * A call to this function always succeeds.
	 *
	 * @tparam DataType The type of elements contained in this vector.
	 *
	 * @param[in] x The vector of which to retrieve the number of nonzeroes.
	 *
	 * @return The number of nonzeroes in \a x.
	 *
	 * \parblock
	 * \par Performance semantics
	 * A call to this function
	 *   -# consists of \f$ \Theta(1) \f$ work;
	 *   -# moves \f$ \Theta(1) \f$ bytes of memory;
	 *   -# does not allocate nor free any dynamic memory;
	 *   -# shall not make any system calls.
	 * \endparblock
	 */
	template< typename DataType, typename Coords >
	size_t nnz( const Vector< DataType, banshee, Coords > & x ) noexcept {
		return internal::getCoordinates( x ).nonzeroes();
	}

	/** \todo add documentation. In particular, think about the meaning with \a P > 1. */
	template< typename InputType, typename Coords >
	RC resize( Vector< InputType, banshee, Coords > & x, const size_t new_nz ) {
		// check if we have a mismatch
		if( new_nz > grb::size( x ) ) {
			return MISMATCH;
		}
		// in the banshee implementation, vectors are of static size
		// so this function immediately succeeds
		return SUCCESS;
	}

	/**
	 * Sets all elements of a vector to the given value. This makes the given
	 * vector completely dense.
	 *
	 * This code is functionally equivalent to both
	 * \code
	 * grb::operators::right_assign< DataType > op;
	 * return foldl< descr >( x, val, op );
	 * \endcode
	 * and
	 * \code
	 * grb::operators::left_assign< DataType > op;
	 * return foldr< descr >( val, x, op );
	 * \endcode
	 *
	 * Their performance semantics also match.
	 *
	 * @tparam descr    The descriptor used for this operation.
	 * @tparam DataType The type of each element in the given vector.
	 * @tparam T        The type of the given value.
	 *
	 * @param[in,out] x The vector of which every element is to be set to equal
	 *                  \a val.
	 * @param[in]   val The value to set each element of \a x equal to.
	 *
	 * @returns SUCCESS       When the call completes successfully.
	 * @returns OUT_OF_MEMORY If the capacity of \a x was insufficient to hold a
	 *                        completely dense vector and not enough memory could
	 *                        be allocated to remedy this. When this error code
	 *                        is returned, the state of the program shall be as
	 *                        though the call to this function had never occurred.
	 *
	 * \parblock
	 * \par Accepted descriptors
	 *   -# grb::descriptors::no_operation
	 *   -# grb::descriptors::no_casting
	 * \endparblock
	 *
	 * When \a descr includes grb::descriptors::no_casting and if \a T does not
	 * match \a DataType, the code shall not compile.
	 *
	 * \parblock
	 * \par Performance semantics
	 * A call to this function
	 *   -# consists of \f$ \Theta(n) \f$ work;
	 *   -# moves \f$ \Theta(n) \f$ bytes of memory;
	 *   -# does not allocate nor free any dynamic memory;
	 *   -# shall not make any system calls.
	 * \endparblock
	 *
	 * \warning If the capacity of \a x was insufficient to store a dense vector
	 *          then a call to this function may make the appropriate system calls
	 *          to allocate \f$ \Theta( n \mathit{sizeof}(DataType) ) \f$ bytes of
	 *          memory.
	 *
	 * @see grb::foldl.
	 * @see grb::foldr.
	 * @see grb::operators::left_assign.
	 * @see grb::operators::right_assign.
	 */
	template< Descriptor descr = descriptors::no_operation, typename DataType, typename Coords, typename T >
	RC set(
		Vector< DataType, banshee, Coords > & x, const T val,
		const typename std::enable_if<
			! grb::is_object< DataType >::value && ! grb::is_object< T >::value,
		void >::type * const = NULL
	) {
		// static sanity checks
		NO_CAST_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< DataType, T >::value ), "grb::set (Vector, unmasked)",
			"called with a value type that does not match that of the given "
			"vector" );

		// pre-cast value to be copied
		const DataType toCopy = static_cast< DataType >( val );

		// make vector dense if it was not already
		internal::getCoordinates( x ).assignAll();
		DataType * const raw = internal::getRaw( x );
		const size_t n = internal::getCoordinates( x ).size();
		if( descr & descriptors::use_index ) {
			for( size_t i = 0; i < n; ++i ) {
				raw[ i ] = static_cast< DataType >( i );
			}
		} else {
			for( size_t i = 0; i < n; ++i ) {
				raw[ i ] = toCopy;
			}
		}
		// sanity check
		assert( internal::getCoordinates( x ).nonzeroes() == internal::getCoordinates( x ).size() );

		// done
		return SUCCESS;
	}

	/**
	 * Sets the element of a given vector at a given position to a given value.
	 *
	 * If the input vector \a x already has an element \f$ x_i \f$, that element
	 * is overwritten to the given value \a val. If no such element existed, it
	 * is added and set equal to \a val. The number of nonzeroes in \a x may thus
	 * be increased by one due to a call to this function.
	 *
	 * The parameter \a i may not be greater or equal than the size of \a x.
	 *
	 * @tparam descr    The descriptor to be used during evaluation of this
	 *                  function.
	 * @tparam DataType The type of the elements of \a x.
	 * @tparam T        The type of the value to be set.
	 *
	 * @param[in,out] x The vector to be modified.
	 * @param[in]   val The value \f$ x_i \f$ should read after function exit.
	 * @param[in]     i The index of the element of \a x to set.
	 *
	 * @return grb::SUCCESS   Upon successful execution of this operation.
	 * @return grb::MISMATCH  If \a i is greater or equal than the dimension of
	 *                        \a x.
	 * @returns OUT_OF_MEMORY If the capacity of \a x was insufficient to add the
	 *                        new value \a val at index \a i, \em and not enough
	 *                        memory could be allocated to remedy this. When this
	 *                        error code is returned, the state of the program
	 *                        shall be as though the call to this function had
	 *                        never occurred.
	 *
	 * \parblock
	 * \par Accepted descriptors
	 *   -# grb::descriptors::no_operation
	 *   -# grb::descriptors::no_casting
	 * \endparblock
	 *
	 * When \a descr includes grb::descriptors::no_casting and if \a T does not
	 * match \a DataType, the code shall not compile.
	 *
	 * \parblock
	 * \par Performance semantics
	 * A call to this function
	 *   -# consists of \f$ \Theta(1) \f$ work;
	 *   -# moves \f$ \Theta(1) \f$ bytes of memory;
	 *   -# does not allocate nor free any dynamic memory;
	 *   -# shall not make any system calls.
	 * \endparblock
	 *
	 * \warning If the capacity of \a x was insufficient to store a dense vector
	 *          then a call to this function may make the appropriate system calls
	 *          to allocate \f$ \Theta( n \mathit{sizeof}(DataType) ) \f$ bytes of
	 *          memory, where \a n is the new size of the vector \a x. This will
	 *          cause additional memory movement and work complexity as well.
	 */
	template< Descriptor descr = descriptors::no_operation, typename DataType, typename Coords, typename T >
	RC setElement( Vector< DataType, banshee, Coords > & x,
		const T val,
		const size_t i,
		const typename std::enable_if< ! grb::is_object< DataType >::value && ! grb::is_object< T >::value, void >::type * const = NULL ) {
		// static sanity checks
		NO_CAST_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< DataType, T >::value ), "grb::set (Vector, at index)",
			"called with a value type that does not match that of the given "
			"vector" );

		// dynamic sanity checks
		if( i >= internal::getCoordinates( x ).size() ) {
			return MISMATCH;
		}

		// do set
		(void)internal::getCoordinates( x ).assign( i );
		internal::getRaw( x )[ i ] = static_cast< DataType >( val );

		// done
		return SUCCESS;
	}

	/**
	 * Sets the content of a given vector \a x to be equal to that of
	 * another given vector \a y.
	 *
	 * The vector \a x may not equal \a y or undefined behaviour will occur.
	 *
	 * \parblock
	 * \par Accepted descriptors
	 *   -# grb::descriptors::no_operation
	 *   -# grb::descriptors::no_casting
	 * \endparblock
	 *
	 * When \a descr includes grb::descriptors::no_casting and if \a InputType
	 * does not match \a OutputType, the code shall not compile.
	 *
	 * \parblock
	 * \par Performance semantics
	 * A call to this function
	 *   -# consists of \f$ \mathcal{O}(n) \f$ work;
	 *   -# moves \f$ \mathcal{O}(n) \f$ bytes of memory;
	 *   -# does not allocate nor free any dynamic memory;
	 *   -# shall not make any system calls.
	 *
	 * \note The use of big-Oh instead of big-Theta is intentional.
	 *       Implementations that chose to emulate sparse vectors using dense
	 *       storage are allowed, but clearly better performance can be attained.
	 * \endparblock
	 *
	 * \warning If the capacity of \a x was insufficient to store a dense vector
	 *          then a call to this function may make the appropriate system calls
	 *          to allocate \f$ \Theta( n \mathit{sizeof}(DataType) ) \f$ bytes of
	 *          memory.
	 *
	 * \todo This documentation is to be extended.
	 */
	template< Descriptor descr = descriptors::no_operation, typename OutputType, typename Coords, typename InputType >
	RC set( Vector< OutputType, banshee, Coords > & x, const Vector< InputType, banshee, Coords > & y ) {
		// static sanity checks
		NO_CAST_ASSERT(
			( ! ( descr & descriptors::no_casting ) || std::is_same< OutputType, InputType >::value ), "grb::copy (Vector)", "called with vector parameters whose element data types do not match" );

		// check contract
		if( reinterpret_cast< void * >( &x ) == reinterpret_cast< const void * >( &y ) ) {
			return ILLEGAL;
		}

		// get relevant descriptors
		constexpr const bool use_index = descr & descriptors::use_index;

		// get length
		const size_t n = internal::getCoordinates( y ).size();

		// get raw value arrays
		OutputType * __restrict__ const dst = internal::getRaw( x );
		const InputType * __restrict__ const src = internal::getRaw( y );

		// dynamic sanity checks
		if( n != internal::getCoordinates( x ).size() ) {
			return MISMATCH;
		}

		// catch boundary case
		if( n == 0 ) {
			return SUCCESS;
		}

		// get #nonzeroes
		const size_t nz = internal::getCoordinates( y ).nonzeroes();
#ifdef _DEBUG
		printf( "grb::set called with source vector containing %d nonzeroes.\n", (int)nz );
#endif

		// first copy contents
		if( src == NULL && dst == NULL ) {
			// then source is a pattern vector, just copy its pattern
			for( size_t i = 0; i < nz; ++i ) {
				(void)internal::getCoordinates( x ).asyncCopy( internal::getCoordinates( y ), i );
			}
		} else if( ! use_index && src == NULL && dst != NULL ) {
			// then we have to cast a pattern vector into a non-pattern one
			for( size_t i = 0; i < nz; ++i ) {
				const auto index = internal::getCoordinates( x ).asyncCopy( internal::getCoordinates( y ), i );
				dst[ index ] = OutputType();
			}
		} else {
			// finally, the regular copy variant:
			for( size_t i = 0; i < nz; ++i ) {
				const auto index = internal::getCoordinates( x ).asyncCopy( internal::getCoordinates( y ), i );
				dst[ index ] = internal::setIndexOrValue< descr, OutputType >( index, src[ index ] );
			}
		}

		// set number of nonzeroes
		internal::getCoordinates( x ).joinCopy( internal::getCoordinates( y ) );

		// done
		return SUCCESS;
	}

	/**
	 * Masked variant of grb::set (vector copy).
	 *
	 * \todo extend documentation.
	 *
	 * @see grb::set
	 */
	template< Descriptor descr = descriptors::no_operation, typename OutputType, typename MaskType, typename InputType, typename Coords >
	RC set( Vector< OutputType, banshee, Coords > & x,
		const Vector< MaskType, banshee, Coords > & mask,
		const Vector< InputType, banshee, Coords > & y,
		const typename std::enable_if< ! grb::is_object< OutputType >::value && ! grb::is_object< MaskType >::value && ! grb::is_object< InputType >::value, void >::type * const = NULL ) {
		// static sanity checks
		NO_CAST_ASSERT(
			( ! ( descr & descriptors::no_casting ) || std::is_same< OutputType, InputType >::value ), "grb::set (Vector)", "called with vector parameters whose element data types do not match" );
		NO_CAST_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< MaskType, bool >::value ), "grb::set (Vector)", "called with non-bool mask element types" );
		static_assert( ! std::is_void< InputType >::value,
			"grb::set (Vector, masked) with void input vector currently not "
			"supported!" );
		static_assert( ! std::is_void< OutputType >::value,
			"grb::set (Vector, masked) with void output vector currently not "
			"supported!" );

		// delegate if possible
		if( internal::getCoordinates( mask ).size() == 0 ) {
			return set( x, y );
		}

		// catch contract violations
		if( reinterpret_cast< void * >( &x ) == reinterpret_cast< const void * >( &y ) ) {
			return ILLEGAL;
		}

		// get relevant descriptors
		constexpr const bool use_index = descr & descriptors::use_index;

		// get length
		const size_t n = internal::getCoordinates( y ).size();

		// dynamic sanity checks
		if( n != internal::getCoordinates( x ).size() ) {
			return MISMATCH;
		}
		if( internal::getCoordinates( mask ).size() != n ) {
			return MISMATCH;
		}

		// catch trivial case
		if( n == 0 ) {
			return SUCCESS;
		}

		// return code
		RC ret = SUCCESS;

		// handle non-trivial, fully masked vector copy

		for( size_t i = 0; ret == SUCCESS && i < internal::getCoordinates( y ).size(); ++i ) {
			if( ! utils::interpretMask< descr >( internal::getCoordinates( mask ).assigned( i ), internal::getRaw( mask ) + i ) ) {
				continue;
			}
			if( internal::getCoordinates( y ).assigned( i ) ) {
				// get value
				const InputType value = use_index ? static_cast< InputType >( i ) : internal::getRaw( y )[ i ];
				(void)internal::getCoordinates( x ).assign( i );
				internal::getRaw( x )[ i ] = value;
			}
		}
		// done
		return ret;
	}

	namespace internal {

		template< Descriptor descr = descriptors::no_operation,
			bool masked,
			bool left, // if this is false, the right-looking fold is assumed
			class OP,
			typename IOType,
			typename InputType,
			typename MaskType,
			typename Coords >
		RC fold_from_vector_to_scalar_generic( IOType & fold_into, const Vector< InputType, banshee, Coords > & to_fold, const Vector< MaskType, banshee, Coords > & mask, const OP & op = OP() ) {
			// static sanity checks
			static_assert( grb::is_associative< OP >::value,
				"grb::foldl can only be called on associate operators. This "
				"function should not have been called-- please submit a "
				"bugreport." );

			// fold is only defined on dense vectors
			if( nnz( to_fold ) < size( to_fold ) ) {
				return ILLEGAL;
			}

			// mask must be of equal size as input vector
			if( masked && size( to_fold ) != size( mask ) ) {
				return MISMATCH;
			}

			// handle trivial case
			if( masked && nnz( mask ) == 0 ) {
				return SUCCESS;
			}

			// some globals used during the folding
			RC ret = SUCCESS;         // final return code
			IOType global = IOType(); // global variable in which to fold
			size_t root = 0;          // which process is the root of the fold (in case we have multiple processes)
			// handle trivial sequential cases
			if( ! masked ) {
				// this op is only defined on dense vectors, check this is the case
				assert( internal::getCoordinates( to_fold ).nonzeroes() == internal::getCoordinates( to_fold ).size() );
				// no mask, vectors are dense, sequential execution-- so rely on underlying operator
				if( left ) {
					global = internal::getRaw( to_fold )[ 0 ];
					op.foldlArray( global, internal::getRaw( to_fold ) + 1, internal::getCoordinates( to_fold ).size() - 1 );
				} else {
					global = internal::getRaw( to_fold )[ internal::getCoordinates( to_fold ).size() - 1 ];
					op.foldrArray( internal::getRaw( to_fold ), global, internal::getCoordinates( to_fold ).size() - 1 );
				}
			} else {
				// masked sequential case
				const size_t n = internal::getCoordinates( to_fold ).size();
				constexpr size_t s = 0;
				constexpr size_t P = 1;
				size_t i = 0;
				const size_t end = internal::getCoordinates( to_fold ).size();
				// some sanity checks
				assert( i <= end );
				assert( end <= n );
#ifdef NDEBUG
				(void)n;
#endif
				// assume current i needs to be processed
				bool process_current_i = true;
				// i is at relative position -1. We keep forwarding until we find an index we should process
				//(or until we hit the end of our block)
				if( masked && i < end ) {

					// check if we need to process current i
					process_current_i = utils::interpretMask< descr >( internal::getCoordinates( mask ).assigned( i ), internal::getRaw( mask ) + i );
					// if not
					while( ! process_current_i ) {
						// forward to next element
						++i;
						// check that we are within bounds
						if( i == end ) {
							break;
						}
						// evaluate whether we should process this i-th element
						process_current_i = utils::interpretMask< descr >( internal::getCoordinates( mask ).assigned( i ), internal::getRaw( mask ) + i );
					}
				}

				// whether we have any nonzeroes assigned at all
				const bool empty = i >= end;

				// in the sequential case, the empty case should have been handled earlier
				assert( ! empty );
				{
					// check if we have a root already
					if( ! empty && root == P ) {
						// no, so take it
						root = s;
					}
				}
				// declare thread-local variable and set our variable to the first value in our block
				IOType local = i < end ? static_cast< IOType >( internal::getRaw( to_fold )[ i ] ) : static_cast< IOType >( internal::getRaw( to_fold )[ 0 ] );
				// if we have a value to fold
				if( ! empty ) {
					// loop over all remaining values, if any
					while( true ) {
						// forward to next variable
						++i;
						// forward more (possibly) if in the masked case
						if( masked ) {
							process_current_i = utils::interpretMask< descr >( internal::getCoordinates( mask ).assigned( i ), internal::getRaw( mask ) + i );
							while( ! process_current_i && i + 1 < end ) {
								++i;
								process_current_i = utils::interpretMask< descr >( internal::getCoordinates( mask ).assigned( i ), internal::getRaw( mask ) + i );
							}
						}
						// stop if past end
						if( i >= end || ! process_current_i ) {
							break;
						}
						// store result of fold in local variable
						RC rc;

						if( left ) {
							rc = foldl< descr >( local, internal::getRaw( to_fold )[ i ], op );
						} else {
							rc = foldr< descr >( internal::getRaw( to_fold )[ i ], local, op );
						}
						// sanity check
						assert( rc == SUCCESS );
						// error propagation
						if( rc != SUCCESS ) {
							ret = rc;
							break;
						}
					}
				}
			}
#ifdef _DEBUG
			printf( "Accumulating %d into %d using fold\n", (int)global, (int)fold_into );
#endif
			// accumulate
			if( ret == SUCCESS ) {
				ret = foldl< descr >( fold_into, global, op );
			}

			// done
			return ret;
		}

		template< Descriptor descr,
			bool left, // if this is false, the right-looking fold is assumed
			bool sparse,
			class OP,
			typename IOType,
			typename InputType,
			typename Coords >
		RC fold_from_scalar_to_vector_generic( Vector< IOType, banshee, Coords > & vector, const InputType & scalar, const OP & op ) {
			const auto & coor = internal::getCoordinates( vector );
			if( sparse ) {
				// assume we were called with a monoid
				for( size_t i = 0; i < coor.size(); ++i ) {
					if( coor.assigned( i ) ) {
						if( left ) {
							(void)foldl< descr >( vector[ i ], scalar, op );
						} else {
							(void)foldr< descr >( scalar, vector[ i ], op );
						}
					} else {
						vector[ i ] = scalar;
					}
				}
			} else {
				// delegate sequential case to operator
				if( left ) {
					op.eWiseFoldlAS( internal::getRaw( vector ), scalar, internal::getCoordinates( vector ).size() );
				} else {
					op.eWiseFoldrSA( scalar, internal::getRaw( vector ), internal::getCoordinates( vector ).size() );
				}
			}
			return SUCCESS;
		}

		/**
		 * Generic fold implementation on two vectors.
		 *
		 * @tparam descr  The descriptor under which the operation takes place.
		 * @tparam left   Whether we are folding left (or right, otherwise).
		 * @tparam sparse Whether one of \a fold_into or \a to_fold is sparse.
		 * @tparam OP     The operator to use while folding.
		 * @tparam IType  The input data type (of \a to_fold).
		 * @tparam IOType The input/output data type (of \a fold_into).
		 *
		 * \note Sparseness is passed explicitly since it is illegal when not
		 *       called using a monoid. This function, however, has no way to
		 *       check for this user input.
		 *
		 * @param[in,out] fold_into The vector whose elements to fold into.
		 * @param[in]     to_fold   The vector whose elements to fold.
		 * @param[in]     op        The operator to use while folding.
		 *
		 * The sizes of \a fold_into and \a to_fold must match; this is an elementwise
		 * fold.
		 *
		 * @returns #ILLEGAL  If \a sparse is <tt>false</tt> while one of \a fold_into
		 *                    or \a to_fold is sparse.
		 * @returns #MISMATCH If the sizes of \a fold_into and \a to_fold do not
		 *                    match.
		 * @returns #SUCCESS  On successful completion of this function call.
		 */
		template< Descriptor descr,
			bool left, // if this is false, the right-looking fold is assumed
			bool sparse,
			class OP,
			typename IOType,
			typename IType,
			typename Coords >
		RC fold_from_vector_to_vector_generic( Vector< IOType, banshee, Coords > & fold_into, const Vector< IType, banshee, Coords > & to_fold, const OP & op ) {
			// take at least a number of elements so that no two threads operate on the same cache line
			const size_t n = size( fold_into );
			if( n != size( to_fold ) ) {
				return MISMATCH;
			}
			if( ! sparse && nnz( fold_into ) < n ) {
				return ILLEGAL;
			}
			if( ! sparse && nnz( to_fold ) < n ) {
				return ILLEGAL;
			}
			if( ! sparse ) {
#ifdef _DEBUG
				printf( "fold_from_vector_to_vector_generic: in dense "
						"variant\n" );
#endif
#ifdef _DEBUG
				printf( "fold_from_vector_to_vector_generic: in sequential "
						"variant\n" );
#endif
				if( left ) {
					op.eWiseFoldlAA( internal::getRaw( fold_into ), internal::getRaw( to_fold ), n );
				} else {
					op.eWiseFoldrAA( internal::getRaw( to_fold ), internal::getRaw( fold_into ), n );
				}

			} else {
#ifdef _DEBUG
				printf( "fold_from_vector_to_vector_generic: in sparse "
						"variant\n" );
				printf( "\tfolding vector of %d nonzeroes into a vector of %d "
						"nonzeroes...\n",
					(int)nnz( to_fold ), (int)nnz( fold_into ) );
#endif
				if( nnz( fold_into ) == n ) {
					// use sparsity structure of to_fold for this eWiseFold
					if( left ) {
#ifdef _DEBUG
						printf( "fold_from_vector_to_vector_generic: using "
								"eWiseLambda, foldl, using to_fold's "
								"sparsity\n" );
#endif
						return eWiseLambda(
							[ &fold_into, &to_fold, &op ]( const size_t i ) {
#ifdef _DEBUG
								printf( "Left-folding %d into %d", (int)to_fold[ i ], (int)fold_into[ i ] );
#endif
								(void)foldl< descr >( fold_into[ i ], to_fold[ i ], op );
#ifdef _DEBUG
								printf( " resulting into %d\n", (int)fold_into[ i ] );
#endif
							},
							to_fold, fold_into );
					} else {
#ifdef _DEBUG
						printf( "fold_from_vector_to_vector_generic: using "
								"eWiseLambda, foldl, using to_fold's "
								"sparsity\n" );
#endif
						return eWiseLambda(
							[ &fold_into, &to_fold, &op ]( const size_t i ) {
#ifdef _DEBUG
								printf( "Right-folding %d into %d", (int)to_fold[ i ], (int)fold_into[ i ] );
#endif
								(void)foldr< descr >( to_fold[ i ], fold_into[ i ], op );
#ifdef _DEBUG
								printf( " resulting into %d\n", (int)fold_into[ i ] );
#endif
							},
							to_fold, fold_into );
					}
				} else if( nnz( to_fold ) == n ) {
					// use sparsity structure of fold_into for this eWiseFold
					if( left ) {
#ifdef _DEBUG
						printf( "fold_from_vector_to_vector_generic: using "
								"eWiseLambda, foldl, using fold_into's "
								"sparsity\n" );
#endif
						return eWiseLambda(
							[ &fold_into, &to_fold, &op ]( const size_t i ) {
#ifdef _DEBUG
								printf( "Left-folding %d into %d", (int)to_fold[ i ], (int)fold_into[ i ] );
#endif
								(void)foldl< descr >( fold_into[ i ], to_fold[ i ], op );
#ifdef _DEBUG
								printf( " resulting into %d\n", (int)fold_into[ i ] );
#endif
							},
							fold_into, to_fold );
					} else {
#ifdef _DEBUG
						printf( "fold_from_vector_to_vector_generic: using "
								"eWiseLambda, foldr, using fold_into's "
								"sparsity\n" );
#endif
						return eWiseLambda(
							[ &fold_into, &to_fold, &op ]( const size_t i ) {
#ifdef _DEBUG
								printf( "Right-folding %d into %d", (int)to_fold[ i ], (int)fold_into[ i ] );
#endif
								(void)foldr< descr >( to_fold[ i ], fold_into[ i ], op );
#ifdef _DEBUG
								printf( " resulting into %d\n", (int)fold_into[ i ] );
#endif
							},
							fold_into, to_fold );
					}
				} else {
#ifdef _DEBUG
					printf( "fold_from_vector_to_vector_generic: using "
							"specialised code to merge two sparse vectors\n" );
#endif
					// both sparse, cannot rely on #eWiseLambda
					const IType * __restrict__ const tf_raw = internal::getRaw( to_fold );
					const auto & tf = internal::getCoordinates( to_fold );
					IOType * __restrict__ const fi_raw = internal::getRaw( fold_into );
					auto & fi = internal::getCoordinates( fold_into );
#ifdef _DEBUG
					printf( "\tin sequential version...\n" );
#endif
					for( size_t k = 0; k < tf.nonzeroes(); ++k ) {
						const size_t i = tf.index( k );
						assert( i < n );
						if( fi.assigned( i ) ) {
							if( left ) {
#ifdef _DEBUG
								printf( "\tfoldl< descr >( fi_raw[ i ], "
										"tf_raw[ i ], op ), i = %d: %d goes "
										"into %d",
									(int)i, (int)tf_raw[ i ], (int)fi_raw[ i ] );
#endif
								(void)foldl< descr >( fi_raw[ i ], tf_raw[ i ], op );
#ifdef _DEBUG
								printf( " which results in %d\n", (int)fi_raw[ i ] );
#endif
							} else {
#ifdef _DEBUG
								printf( "\tfoldr< descr >( tf_raw[ i ], "
										"fi_raw[ i ], op ), i = %d : %d goes "
										"into %d",
									(int)i, (int)tf_raw[ i ], (int)fi_raw[ i ] );
#endif
								(void)foldr< descr >( tf_raw[ i ], fi_raw[ i ], op );
#ifdef _DEBUG
								printf( " which results in %d\n", (int)fi_raw[ i ] );
#endif
							}
						} else {
#ifdef _DEBUG
							printf( "\tindex %d is unset. Old value %d will be "
									"overwritten with %d\n",
								(int)i, (int)fi_raw[ i ], (int)tf_raw[ i ] );
#endif
							fi_raw[ i ] = tf_raw[ i ];
							(void)fi.assign( i );
						}
					}
				}
			}

#ifdef _DEBUG
			printf( "\tCall to fold_from_vector_to_vector_generic done. Output "
					"now contains %d / %d nonzeroes\n",
				(int)nnz( fold_into ), (int)size( fold_into ) );
#endif
			// done
			return SUCCESS;
		}

	} // namespace internal

	/**
	 * Folds all elements in a GraphBLAS vector \a x into a single value \a beta.
	 *
	 * The original value of \a beta is used as the right-hand side input of the
	 * operator \a op. A left-hand side input for \a op is retrieved from the
	 * input vector \a x. The result of the operation is stored in \a beta. This
	 * process is repeated for every element in \a x.
	 *
	 * At function exit, \a beta will equal
	 * \f$ \beta \odot x_0 \odot x_1 \odot \ldots x_{n-1} \f$.
	 *
	 * @tparam descr     The descriptor used for evaluating this function. By
	 *                   default, this is grb::descriptors::no_operation.
	 * @tparam OP        The type of the operator to be applied.
	 *                   The operator must be associative.
	 * @tparam InputType The type of the elements of \a x.
	 * @tparam IOType    The type of the value \a y.
	 *
	 * @param[in]     x    The input vector \a x that will not be modified. This
	 *                     input vector must be dense.
	 * @param[in,out] beta On function entry: the initial value to be applied to
	 *                     \a op from the right-hand side.
	 *                     On function exit: the result of repeated applications
	 *                     from the left-hand side of elements of \a x.
	 * @param[in]    op    The monoid under which to perform this right-folding.
	 *
	 * \note We only define fold under monoids, not under plain operators.
	 *
	 * @returns grb::SUCCESS This function always succeeds.
	 * @returns grb::ILLEGAL When a sparse vector is passed. In this case, the call
	 *                       to this function will have no other effects.
	 *
	 * \warning Since this function folds from left-to-right using binary
	 *          operators, this function \em cannot take sparse vectors as input--
	 *          a monoid is required to give meaning to missing vector entries.
	 *          See grb::reducer for use with sparse vectors instead.
	 *
	 * \parblock
	 * \par Valid descriptors
	 * grb::descriptors::no_operation, grb::descriptors::no_casting.
	 *
	 * \note Invalid descriptors will be ignored.
	 *
	 * If grb::descriptors::no_casting is specified, then 1) the first domain of
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
	 * \parblock
	 * \par Performance semantics
	 *      -# This call comprises \f$ \Theta(n) \f$ work, where \f$ n \f$ equals
	 *         the size of the vector \a x. The constant factor depends on the
	 *         cost of evaluating the underlying binary operator. A good
	 *         implementation uses vectorised instructions whenever the input
	 *         domains, the output domain, and the operator used allow for this.
	 *
	 *      -# This call will not result in additional dynamic memory allocations.
	 *
	 *      -# This call takes \f$ \mathcal{O}(1) \f$ memory beyond the memory
	 *         used by the application at the point of a call to this function.
	 *
	 *      -# This call incurs at most
	 *         \f$ n \cdot\mathit{sizeof}(\mathit{InputType}) + \mathcal{O}(1) \f$
	 *         bytes of data movement. A good implementation will rely on in-place
	 *         operators.
	 * \endparblock
	 *
	 * @see grb::operators::internal::Operator for a discussion on when in-place
	 *      and/or vectorised operations are used.
	 */
	template< Descriptor descr = descriptors::no_operation, class Monoid, typename InputType, typename Coords, typename IOType >
	RC foldr( const Vector< InputType, banshee, Coords > & x,
		IOType & beta,
		const Monoid & monoid = Monoid(),
		const typename std::enable_if< ! grb::is_object< InputType >::value && ! grb::is_object< IOType >::value && grb::is_monoid< Monoid >::value, void >::type * const = NULL ) {
		grb::Vector< bool, banshee, Coords > mask( 0 );
		return internal::fold_from_vector_to_scalar_generic< descr, false, false >( beta, x, mask, monoid.getOperator() );
	}

	/**
	 * For all elements in a GraphBLAS vector \a y, fold the value \f$ \alpha \f$
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
	 * @tparam descr     The descriptor used for evaluating this function. By
	 *                   default, this is grb::descriptors::no_operation.
	 * @tparam OP        The type of the operator to be applied.
	 * @tparam InputType The type of \a alpha.
	 * @tparam IOType    The type of the elements in \a y.
	 *
	 * @param[in]     alpha The input value to apply as the left-hand side input
	 *                      to \a op.
	 * @param[in,out] y     On function entry: the initial values to be applied as
	 *                      the right-hand side input to \a op.
	 *                      On function exit: the output data.
	 * @param[in]     op    The monoid under which to perform this left-folding.
	 *
	 * @returns grb::SUCCESS This function always succeeds.
	 *
	 * \note We only define fold under monoids, not under plain operators.
	 *
	 * \parblock
	 * \par Valid descriptors
	 * grb::descriptors::no_operation, grb::descriptors::no_casting.
	 *
	 * \note Invalid descriptors will be ignored.
	 *
	 * If grb::descriptors::no_casting is specified, then 1) the first domain of
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
	 * \parblock
	 * \par Performance semantics
	 *      -# This call comprises \f$ \Theta(n) \f$ work, where \f$ n \f$ equals
	 *         the size of the vector \a x. The constant factor depends on the
	 *         cost of evaluating the underlying binary operator. A good
	 *         implementation uses vectorised instructions whenever the input
	 *         domains, the output domain, and the operator used allow for this.
	 *
	 *      -# This call will not result in additional dynamic memory allocations.
	 *
	 *      -# This call takes \f$ \mathcal{O}(1) \f$ memory beyond the memory
	 *         used by the application at the point of a call to this function.
	 *
	 *      -# This call incurs at most
	 *         \f$ 2n \cdot \mathit{sizeof}(\mathit{IOType}) + \mathcal{O}(1) \f$
	 *         bytes of data movement.
	 * \endparblock
	 *
	 * @see grb::operators::internal::Operator for a discussion on when in-place
	 *      and/or vectorised operations are used.
	 */
	template< Descriptor descr = descriptors::no_operation, class Monoid, typename IOType, typename Coords, typename InputType >
	RC foldr( const InputType & alpha,
		Vector< IOType, banshee, Coords > & y,
		const Monoid & monoid = Monoid(),
		const typename std::enable_if< ! grb::is_object< InputType >::value && ! grb::is_object< IOType >::value && grb::is_monoid< Monoid >::value, void >::type * const = NULL ) {
		monoid.getOperator().eWiseFoldrSA( alpha, internal::getRaw( y ), internal::getCoordinates( y ).size() );
		return SUCCESS;
	}

	/**
	 * Folds all elements in a GraphBLAS vector \a x into the corresponding
	 * elements from an input/output vector \a y. The vectors must be of equal
	 * size \f$ n \f$. For all \f$ i \in \{0,1,\ldots,n-1\} \f$, the new value
	 * of at the i-th index of \a y after a call to this function thus equals
	 * \f$ x_i \odot y_i \f$.
	 *
	 * @tparam descr     The descriptor used for evaluating this function. By
	 *                   default, this is grb::descriptors::no_operation.
	 * @tparam OP        The type of the operator to be applied.
	 * @tparam IOType    The type of the elements of \a x.
	 * @tparam InputType The type of the elements of \a y.
	 *
	 * @param[in]     x  The input vector \a y that will not be modified.
	 * @param[in,out] y  On function entry: the initial value to be applied to
	 *                   \a op as the right-hand side input.
	 *                   On function exit: the result of repeated applications
	 *                   from the right-hand side using elements from \a y.
	 * @param[in]     op The operator under which to perform this right-folding.
	 *
	 * @returns grb::SUCCESS This function always succeeds.
	 *
	 * \note The element-wise fold is also defined for monoids.
	 *
	 * \parblock
	 * \par Valid descriptors
	 * grb::descriptors::no_operation, grb::descriptors::no_casting.
	 *
	 * \note Invalid descriptors will be ignored.
	 *
	 * If grb::descriptors::no_casting is specified, then 1) the first domain of
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
	 * \parblock
	 * \par Performance semantics
	 *      -# This call comprises \f$ \Theta(n) \f$ work, where \f$ n \f$ equals
	 *         the size of the vector \a x. The constant factor depends on the
	 *         cost of evaluating the underlying binary operator. A good
	 *         implementation uses vectorised instructions whenever the input
	 *         domains, the output domain, and the operator used allow for this.
	 *
	 *      -# This call will not result in additional dynamic memory allocations.
	 *
	 *      -# This call takes \f$ \mathcal{O}(1) \f$ memory beyond the memory
	 *         used by the application at the point of a call to this function.
	 *
	 *      -# This call incurs at most
	 *         \f$ n \cdot (
	 *                       \mathit{sizeof}(InputType) + 2\mathit{sizeof}(IOType)
	 *                     ) + \mathcal{O}(1)
	 *         \f$
	 *         bytes of data movement. A good implementation will rely on in-place
	 *         operators whenever allowed.
	 * \endparblock
	 *
	 * @see grb::operators::internal::Operator for a discussion on when in-place
	 *      and/or vectorised operations are used.
	 */
	template< Descriptor descr = descriptors::no_operation, class OP, typename IOType, typename InputType, typename Coords >
	RC foldr( const Vector< InputType, banshee, Coords > & x,
		Vector< IOType, banshee, Coords > & y,
		const OP & op = OP(),
		const typename std::enable_if< grb::is_operator< OP >::value && ! grb::is_object< InputType >::value && ! grb::is_object< IOType >::value, void >::type * = NULL ) {
		// static sanity checks
		NO_CAST_OP_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename OP::D1, InputType >::value ), "grb::eWiseFoldr",
			"called with a vector x of a type that does not match the first domain "
			"of the given operator" );
		NO_CAST_OP_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename OP::D2, IOType >::value ), "grb::eWiseFoldr",
			"called on a vector y of a type that does not match the second domain "
			"of the given operator" );
		NO_CAST_OP_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename OP::D3, IOType >::value ), "grb::eWiseFoldr",
			"called on a vector y of a type that does not match the third domain "
			"of the given operator" );
		if( size( x ) != size( y ) ) {
			return MISMATCH;
		}

#ifdef _DEBUG
		printf( "In foldr ([T]<-[T])\n" );
#endif

		if( nnz( x ) < size( x ) || nnz( y ) < size( y ) ) {
			return internal::fold_from_vector_to_vector_generic< descr, false, true >( y, x, op );
		} else {
			return internal::fold_from_vector_to_vector_generic< descr, false, false >( y, x, op );
		}
	}

	/**
	 * Folds all elements in a GraphBLAS vector \a x into the corresponding
	 * elements from an input/output vector \a y. The vectors must be of equal
	 * size \f$ n \f$. For all \f$ i \in \{0,1,\ldots,n-1\} \f$, the new value
	 * of at the i-th index of \a y after a call to this function thus equals
	 * \f$ x_i \odot y_i \f$.
	 *
	 * @tparam descr     The descriptor used for evaluating this function. By
	 *                   default, this is grb::descriptors::no_operation.
	 * @tparam Monoid    The type of the monoid to be applied.
	 * @tparam IOType    The type of the elements of \a x.
	 * @tparam InputType The type of the elements of \a y.
	 *
	 * @param[in]       x    The input vector \a y that will not be modified.
	 * @param[in,out]   y    On function entry: the initial value to be applied
	 *                       to \a op as the right-hand side input.
	 *                       On function exit: the result of repeated applications
	 *                       from the right-hand side using elements from \a y.
	 * @param[in]     monoid The monoid under which to perform this right-folding.
	 *
	 * @returns grb::SUCCESS This function always succeeds.
	 *
	 * \note The element-wise fold is also defined for operators.
	 *
	 * \parblock
	 * \par Valid descriptors
	 * grb::descriptors::no_operation, grb::descriptors::no_casting.
	 *
	 * \note Invalid descriptors will be ignored.
	 *
	 * If grb::descriptors::no_casting is specified, then 1) the first domain of
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
	 * \parblock
	 * \par Performance semantics
	 *      -# This call comprises \f$ \Theta(n) \f$ work, where \f$ n \f$ equals
	 *         the size of the vector \a x. The constant factor depends on the
	 *         cost of evaluating the underlying binary operator. A good
	 *         implementation uses vectorised instructions whenever the input
	 *         domains, the output domain, and the operator used allow for this.
	 *
	 *      -# This call will not result in additional dynamic memory allocations.
	 *
	 *      -# This call takes \f$ \mathcal{O}(1) \f$ memory beyond the memory
	 *         used by the application at the point of a call to this function.
	 *
	 *      -# This call incurs at most
	 *         \f$ n \cdot (
	 *                       \mathit{sizeof}(InputType) + 2\mathit{sizeof}(IOType)
	 *                     ) + \mathcal{O}(1)
	 *         \f$
	 *         bytes of data movement. A good implementation will rely on in-place
	 *         operators whenever allowed.
	 * \endparblock
	 *
	 * @see grb::operators::internal::Operator for a discussion on when in-place
	 *      and/or vectorised operations are used.
	 */
	template< Descriptor descr = descriptors::no_operation, class Monoid, typename IOType, typename InputType, typename Coords >
	RC foldr( const Vector< InputType, banshee, Coords > & x,
		Vector< IOType, banshee, Coords > & y,
		const Monoid & monoid = Monoid(),
		const typename std::enable_if< grb::is_monoid< Monoid >::value && ! grb::is_object< InputType >::value && ! grb::is_object< IOType >::value, void >::type * = NULL ) {
		// static sanity checks
		NO_CAST_OP_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename Monoid::D1, InputType >::value ), "grb::eWiseFoldr",
			"called with a vector x of a type that does not match the first domain "
			"of the given monoid" );
		NO_CAST_OP_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename Monoid::D2, IOType >::value ), "grb::eWiseFoldr",
			"called on a vector y of a type that does not match the second domain "
			"of the given monoid" );
		NO_CAST_OP_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename Monoid::D3, IOType >::value ), "grb::eWiseFoldr",
			"called on a vector y of a type that does not match the third domain "
			"of the given monoid" );

		// dynamic sanity checks
		if( size( x ) != size( y ) ) {
			return MISMATCH;
		}

		// delegate
		return foldr( x, y, monoid.getOperator() );
	}

	/**
	 * For all elements in a GraphBLAS vector \a x, fold the value \f$ \beta \f$
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
	 * @tparam descr     The descriptor used for evaluating this function. By
	 *                   default, this is grb::descriptors::no_operation.
	 * @tparam OP        The type of the operator to be applied.
	 * @tparam IOType    The type of the value \a beta.
	 * @tparam InputType The type of the elements of \a x.
	 *
	 * @param[in,out] x    On function entry: the initial values to be applied as
	 *                     the left-hand side input to \a op. The input vector must
	 *                     be dense.
	 *                     On function exit: the output data.
	 * @param[in]     beta The input value to apply as the right-hand side input
	 *                     to \a op.
	 * @param[in]     op   The operator under which to perform this left-folding.
	 *
	 * @returns grb::SUCCESS This function always succeeds.
	 *
	 * \note This function is also defined for monoids.
	 *
	 * \warning If \a x is sparse and this operation is requested, a monoid instead
	 *          of an operator is required!
	 *
	 * \parblock
	 * \par Valid descriptors
	 * grb::descriptors::no_operation, grb::descriptors::no_casting.
	 *
	 * \note Invalid descriptors will be ignored.
	 *
	 * If grb::descriptors::no_casting is specified, then 1) the first domain of
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
	 * \parblock
	 * \par Performance semantics
	 *      -# This call comprises \f$ \Theta(n) \f$ work, where \f$ n \f$ equals
	 *         the size of the vector \a x. The constant factor depends on the
	 *         cost of evaluating the underlying binary operator. A good
	 *         implementation uses vectorised instructions whenever the input
	 *         domains, the output domain, and the operator used allow for this.
	 *
	 *      -# This call will not result in additional dynamic memory allocations.
	 *
	 *      -# This call takes \f$ \mathcal{O}(1) \f$ memory beyond the memory
	 *         used by the application at the point of a call to this function.
	 *
	 *      -# This call incurs at most
	 *         \f$ 2n \cdot \mathit{sizeof}(\mathit{IOType}) + \mathcal{O}(1) \f$
	 *         bytes of data movement.
	 * \endparblock
	 *
	 * @see grb::operators::internal::Operator for a discussion on when in-place
	 *      and/or vectorised operations are used.
	 */
	template< Descriptor descr = descriptors::no_operation, class Op, typename IOType, typename Coords, typename InputType >
	RC foldl( Vector< IOType, banshee, Coords > & x,
		const InputType & beta,
		const Op & op = Op(),
		const typename std::enable_if< ! grb::is_object< IOType >::value && ! grb::is_object< InputType >::value && grb::is_operator< Op >::value, void >::type * = NULL ) {
		// static sanity checks
		NO_CAST_OP_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename Op::D1, IOType >::value ), "grb::foldl",
			"called with a vector x of a type that does not match the first domain "
			"of the given operator" );
		NO_CAST_OP_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename Op::D2, InputType >::value ), "grb::foldl",
			"called on a vector y of a type that does not match the second domain "
			"of the given operator" );
		NO_CAST_OP_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename Op::D3, IOType >::value ), "grb::foldl",
			"called on a vector x of a type that does not match the third domain "
			"of the given operator" );

		// if no monoid was given, then we can only handle dense vectors
		if( nnz( x ) < size( x ) ) {
			return ILLEGAL;
		} else {
			return internal::fold_from_scalar_to_vector_generic< descr, true, false >( x, beta, op );
		}
	}

	/**
	 * For all elements in a GraphBLAS vector \a x, fold the value \f$ \beta \f$
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
	 * @tparam descr     The descriptor used for evaluating this function. By
	 *                   default, this is grb::descriptors::no_operation.
	 * @tparam Monoid    The type of the monoid to be applied.
	 * @tparam IOType    The type of the value \a beta.
	 * @tparam InputType The type of the elements of \a x.
	 *
	 * @param[in,out] x    On function entry: the initial values to be applied as
	 *                     the left-hand side input to \a op.
	 *                     On function exit: the output data.
	 * @param[in]     beta The input value to apply as the right-hand side input
	 *                     to \a op.
	 * @param[in]   monoid The monoid under which to perform this left-folding.
	 *
	 * @returns grb::SUCCESS This function always succeeds.
	 *
	 * \note This function is also defined for operators.
	 *
	 * \parblock
	 * \par Valid descriptors
	 * grb::descriptors::no_operation, grb::descriptors::no_casting.
	 *
	 * \note Invalid descriptors will be ignored.
	 *
	 * If grb::descriptors::no_casting is specified, then 1) the first domain of
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
	 * \parblock
	 * \par Performance semantics
	 *      -# This call comprises \f$ \Theta(n) \f$ work, where \f$ n \f$ equals
	 *         the size of the vector \a x. The constant factor depends on the
	 *         cost of evaluating the underlying binary operator. A good
	 *         implementation uses vectorised instructions whenever the input
	 *         domains, the output domain, and the operator used allow for this.
	 *
	 *      -# This call will not result in additional dynamic memory allocations.
	 *
	 *      -# This call takes \f$ \mathcal{O}(1) \f$ memory beyond the memory
	 *         used by the application at the point of a call to this function.
	 *
	 *      -# This call incurs at most
	 *         \f$ 2n \cdot \mathit{sizeof}(\mathit{IOType}) + \mathcal{O}(1) \f$
	 *         bytes of data movement.
	 * \endparblock
	 *
	 * @see grb::operators::internal::Operator for a discussion on when in-place
	 *      and/or vectorised operations are used.
	 */
	template< Descriptor descr = descriptors::no_operation, class Monoid, typename IOType, typename Coords, typename InputType >
	RC foldl( Vector< IOType, banshee, Coords > & x,
		const InputType & beta,
		const Monoid & monoid = Monoid(),
		const typename std::enable_if< ! grb::is_object< IOType >::value && ! grb::is_object< InputType >::value && grb::is_monoid< Monoid >::value, void >::type * = NULL ) {
		// static sanity checks
		NO_CAST_OP_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename Monoid::D1, IOType >::value ), "grb::foldl",
			"called with a vector x of a type that does not match the first domain "
			"of the given monoid" );
		NO_CAST_OP_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename Monoid::D2, InputType >::value ), "grb::foldl",
			"called on a vector y of a type that does not match the second domain "
			"of the given monoid" );
		NO_CAST_OP_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename Monoid::D3, IOType >::value ), "grb::foldl",
			"called on a vector x of a type that does not match the third domain "
			"of the given monoid" );

		// delegate to generic case
		if( ( descr & descriptors::dense ) || internal::getCoordinates( x ).isDense() ) {
			return internal::fold_from_scalar_to_vector_generic< descr, true, false >( x, beta, monoid.getOperator() );
		} else {
			return internal::fold_from_scalar_to_vector_generic< descr, true, true >( x, beta, monoid.getOperator() );
		}
	}

	/**
	 * Folds all elements in a GraphBLAS vector \a y into the corresponding
	 * elements from an input/output vector \a x. The vectors must be of equal
	 * size \f$ n \f$. For all \f$ i \in \{0,1,\ldots,n-1\} \f$, the new value
	 * of at the i-th index of \a x after a call to this function thus equals
	 * \f$ x_i \odot y_i \f$.
	 *
	 * @tparam descr     The descriptor used for evaluating this function. By
	 *                   default, this is grb::descriptors::no_operation.
	 * @tparam OP        The type of the operator to be applied.
	 * @tparam IOType    The type of the value \a x.
	 * @tparam InputType The type of the elements of \a y.
	 *
	 * @param[in,out] x On function entry: the vector whose elements are to be
	 *                  applied to \a op as the left-hand side input.
	 *                  On function exit: the vector containing the result of
	 *                  the requested computation.
	 * @param[in]    y  The input vector \a y whose elements are to be applied
	 *                  to \a op as right-hand side input.
	 * @param[in]    op The operator under which to perform this left-folding.
	 *
	 * @returns grb::SUCCESS This function always succeeds.
	 *
	 * \note This function is also defined for monoids.
	 *
	 * \parblock
	 * \par Valid descriptors
	 * grb::descriptors::no_operation, grb::descriptors::no_casting.
	 *
	 * \note Invalid descriptors will be ignored.
	 *
	 * If grb::descriptors::no_casting is specified, then 1) the first domain of
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
	 * \parblock
	 * \par Performance semantics
	 *      -# This call comprises \f$ \Theta(n) \f$ work, where \f$ n \f$ equals
	 *         the size of the vector \a x. The constant factor depends on the
	 *         cost of evaluating the underlying binary operator. A good
	 *         implementation uses vectorised instructions whenever the input
	 *         domains, the output domain, and the operator used allow for this.
	 *
	 *      -# This call will not result in additional dynamic memory allocations.
	 *
	 *      -# This call takes \f$ \mathcal{O}(1) \f$ memory beyond the memory
	 *         used by the application at the point of a call to this function.
	 *
	 *      -# This call incurs at most
	 *         \f$ n \cdot (
	 *                \mathit{sizeof}(\mathit{IOType}) +
	 *                \mathit{sizeof}(\mathit{InputType})
	 *             ) + \mathcal{O}(1) \f$
	 *         bytes of data movement. A good implementation will apply in-place
	 *         vectorised instructions whenever the input domains, the output
	 *         domain, and the operator used allow for this.
	 * \endparblock
	 *
	 * @see grb::operators::internal::Operator for a discussion on when in-place
	 *      and/or vectorised operations are used.
	 */
	template< Descriptor descr = descriptors::no_operation, class OP, typename IOType, typename InputType, typename Coords >
	RC foldl( Vector< IOType, banshee, Coords > & x,
		const Vector< InputType, banshee, Coords > & y,
		const OP & op = OP(),
		const typename std::enable_if< grb::is_operator< OP >::value && ! grb::is_object< IOType >::value && ! grb::is_object< InputType >::value, void >::type * = NULL ) {
		// static sanity checks
		NO_CAST_OP_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename OP::D1, IOType >::value ), "grb::foldl",
			"called with a vector x of a type that does not match the first domain "
			"of the given operator" );
		NO_CAST_OP_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename OP::D2, InputType >::value ), "grb::foldl",
			"called on a vector y of a type that does not match the second domain "
			"of the given operator" );
		NO_CAST_OP_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename OP::D3, IOType >::value ), "grb::foldl",
			"called on a vector x of a type that does not match the third domain "
			"of the given operator" );
		// dynamic sanity checks
		const size_t n = size( x );
		if( n != size( y ) ) {
			return MISMATCH;
		}
		// all OK, execute
		RC ret = SUCCESS;
		if( nnz( x ) < n || nnz( y ) < n ) {
			ret = internal::fold_from_vector_to_vector_generic< descr, true, true >( x, y, op );
		} else {
			assert( nnz( x ) == n );
			assert( nnz( y ) == n );
			ret = internal::fold_from_vector_to_vector_generic< descr, true, false >( x, y, op );
		}
		return ret;
	}

	/**
	 * Folds all elements in a GraphBLAS vector \a y into the corresponding
	 * elements from an input/output vector \a x. The vectors must be of equal
	 * size \f$ n \f$. For all \f$ i \in \{0,1,\ldots,n-1\} \f$, the new value
	 * of at the i-th index of \a x after a call to this function thus equals
	 * \f$ x_i \odot y_i \f$.
	 *
	 * @tparam descr     The descriptor used for evaluating this function. By
	 *                   default, this is grb::descriptors::no_operation.
	 * @tparam Monoid    The type of the monoid to be applied.
	 * @tparam IOType    The type of the value \a x.
	 * @tparam InputType The type of the elements of \a y.
	 *
	 * @param[in,out]  x    On function entry: the vector whose elements are to be
	 *                      applied to \a op as the left-hand side input.
	 *                      On function exit: the vector containing the result of
	 *                      the requested computation.
	 * @param[in]      y    The input vector \a y whose elements are to be applied
	 *                      to \a op as right-hand side input.
	 * @param[in]    monoid The operator under which to perform this left-folding.
	 *
	 * @returns grb::SUCCESS This function always succeeds.
	 *
	 * \note This function is also defined for operators.
	 *
	 * \parblock
	 * \par Valid descriptors
	 * grb::descriptors::no_operation, grb::descriptors::no_casting.
	 *
	 * \note Invalid descriptors will be ignored.
	 *
	 * If grb::descriptors::no_casting is specified, then 1) the first domain of
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
	 * \parblock
	 * \par Performance semantics
	 *      -# This call comprises \f$ \Theta(n) \f$ work, where \f$ n \f$ equals
	 *         the size of the vector \a x. The constant factor depends on the
	 *         cost of evaluating the underlying binary operator. A good
	 *         implementation uses vectorised instructions whenever the input
	 *         domains, the output domain, and the operator used allow for this.
	 *
	 *      -# This call will not result in additional dynamic memory allocations.
	 *
	 *      -# This call takes \f$ \mathcal{O}(1) \f$ memory beyond the memory
	 *         used by the application at the point of a call to this function.
	 *
	 *      -# This call incurs at most
	 *         \f$ n \cdot (
	 *                \mathit{sizeof}(\mathit{IOType}) +
	 *                \mathit{sizeof}(\mathit{InputType})
	 *             ) + \mathcal{O}(1) \f$
	 *         bytes of data movement. A good implementation will apply in-place
	 *         vectorised instructions whenever the input domains, the output
	 *         domain, and the operator used allow for this.
	 * \endparblock
	 *
	 * @see grb::operators::internal::Operator for a discussion on when in-place
	 *      and/or vectorised operations are used.
	 */
	template< Descriptor descr = descriptors::no_operation, class Monoid, typename IOType, typename InputType, typename Coords >
	RC foldl( Vector< IOType, banshee, Coords > & x,
		const Vector< InputType, banshee, Coords > & y,
		const Monoid & monoid = Monoid(),
		const typename std::enable_if< grb::is_monoid< Monoid >::value && ! grb::is_object< IOType >::value && ! grb::is_object< InputType >::value, void >::type * = NULL ) {
		// static sanity checks
		NO_CAST_OP_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename Monoid::D1, IOType >::value ), "grb::eWiseFoldl",
			"called with a vector x of a type that does not match the first domain "
			"of the given operator" );
		NO_CAST_OP_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename Monoid::D2, InputType >::value ), "grb::eWiseFoldl",
			"called on a vector y of a type that does not match the second domain "
			"of the given operator" );
		NO_CAST_OP_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename Monoid::D3, IOType >::value ), "grb::eWiseFoldl",
			"called on a vector x of a type that does not match the third domain "
			"of the given operator" );

		return foldl( x, y, monoid.getOperator() );
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
	 * @tparam InputType1 The value type of the left-hand vector.
	 * @tparam InputType2 The value type of the right-hand scalar.
	 * @tparam OutputType The value type of the ouput vector.
	 *
	 * @param[in]   x   The left-hand input vector.
	 * @param[in]  beta The right-hand input scalar.
	 * @param[out]  z   The pre-allocated output vector.
	 * @param[in]   op  The operator to use.
	 *
	 * @return grb::MISMATCH Whenever the dimensions of \a x and \a z do not
	 *                       match. All input data containers are left untouched
	 *                       if this exit code is returned; it will be as though
	 *                       this call was never made.
	 * @return grb::SUCCESS  On successful completion of this call.
	 *
	 * \parblock
	 * \par Performance semantics
	 *      -# This call comprises \f$ \Theta(n) \f$ work, where \f$ n \f$ equals
	 *         the size of the vectors \a x and \a z. The constant factor depends
	 *         on the cost of evaluating the operator. A good implementation uses
	 *         vectorised instructions whenever the input domains, the output
	 *         domain, and the operator used allow for this.
	 *
	 *      -# This call takes \f$ \mathcal{O}(1) \f$ memory beyond the memory
	 *         used by the application at the point of a call to this function.
	 *
	 *      -# This call incurs at most
	 *         \f$ n(
	 *               \mathit{sizeof}(\mathit{D1}) + \mathit{sizeof}(\mathit{D3})
	 *             ) +
	 *         \mathcal{O}(1) \f$
	 *         bytes of data movement. A good implementation will stream \a y
	 *         into \a z to apply the multiplication operator in-place, whenever
	 *         the input domains, the output domain, and the operator allow for
	 *         this.
	 * \endparblock
	 */
	template< Descriptor descr = descriptors::no_operation, class OP, typename OutputType, typename InputType1, typename Coords, typename InputType2 >
	RC eWiseApply( Vector< OutputType, banshee, Coords > & z,
		const Vector< InputType1, banshee, Coords > & x,
		const InputType2 beta,
		const OP & op = OP(),
		const typename std::enable_if< ! grb::is_object< OutputType >::value && ! grb::is_object< InputType1 >::value && ! grb::is_object< InputType2 >::value && grb::is_operator< OP >::value,
			void >::type * const = NULL ) {
#ifdef _DEBUG
		printf( "In eWiseApply ([T1]<-[T2]<-T3), operator variant\n" );
#endif
		// sanity check
		const size_t n = internal::getCoordinates( z ).size();
		if( internal::getCoordinates( x ).size() != n ) {
			return MISMATCH;
		}
		if( internal::getCoordinates( x ).nonzeroes() < n ) {
			return ILLEGAL;
		}

		// the result of this operation will be dense
		if( internal::getCoordinates( z ).nonzeroes() < n ) {
			internal::getCoordinates( z ).assignAll();
		}

		// rely on lambda for easy implementation
		return eWiseLambda(
			[ &beta, &x, &z, &op ]( const size_t i ) {
				apply( z[ i ], x[ i ], beta, op );
			},
			z, x );
	}

	namespace internal {

		template< bool masked, Descriptor descr, class OP, typename OutputType, typename MaskType, typename InputType1, typename InputType2 >
		RC sparse_apply_generic( OutputType * const z_p,
			Coordinates< banshee > & z_coors,
			MaskType * const mask_p,
			const InputType1 * const x_p,
			const Coordinates< banshee > & left_coors,
			const InputType2 * const y_p,
			const Coordinates< banshee > & right_coors,
			const OP & op,
			const size_t n,
			const Coordinates< banshee > * const mask_coors = NULL ) {
#ifdef NDEBUG
			(void)n;
#endif
			// assertions
			assert( ! masked || mask_coors != NULL );
			assert( ! masked || mask_coors->size() == n );
			assert( left_coors.nonzeroes() <= n );
			assert( right_coors.nonzeroes() <= n );

#ifdef _DEBUG
			printf( "\tinternal::sparse_apply_generic called\n" );
#endif
			if( left_coors.nonzeroes() < right_coors.nonzeroes() ) {
				// expensive pass #1
				for( size_t k = 0; k < left_coors.nonzeroes(); ++k ) {
					const size_t index = left_coors.index( k );
					assert( index < n );
					if( masked && ! utils::interpretMask< descr >( mask_coors->assigned( index ), mask_p + index ) ) {
						continue;
					}
					if( right_coors.assigned( index ) ) {
						(void)z_coors.assign( index );
						grb::apply( z_p[ index ], x_p[ index ], y_p[ index ], op );
					} else {
						(void)z_coors.assign( index );
						z_p[ index ] = x_p[ index ];
					}
				}
				// cheaper pass #2
				for( size_t k = 0; k < right_coors.nonzeroes(); ++k ) {
					const size_t index = right_coors.index( k );
					assert( index < n );
					if( left_coors.assigned( index ) ) {
						continue;
					}
					if( masked && ! utils::interpretMask< descr >( mask_coors->assigned( index ), mask_p + index ) ) {
						continue;
					}
					(void)z_coors.assign( index );
					z_p[ index ] = y_p[ index ];
				}
			} else {
				// expensive pass #1
				for( size_t k = 0; k < right_coors.nonzeroes(); ++k ) {
					const size_t index = right_coors.index( k );
					assert( index < n );
					if( masked && ! utils::interpretMask< descr >( mask_coors->assigned( index ), mask_p + index ) ) {
						continue;
					}
					if( left_coors.assigned( index ) ) {
						(void)z_coors.assign( index );
						grb::apply( z_p[ index ], x_p[ index ], y_p[ index ], op );
					} else {
						(void)z_coors.assign( index );
						z_p[ index ] = y_p[ index ];
					}
				}
				// cheaper pass #2
				for( size_t k = 0; k < left_coors.nonzeroes(); ++k ) {
					const size_t index = left_coors.index( k );
					assert( index < n );
					if( right_coors.assigned( index ) ) {
						continue;
					}
					if( masked && ! utils::interpretMask< descr >( mask_coors->assigned( index ), mask_p + index ) ) {
						continue;
					}
					(void)z_coors.assign( index );
					z_p[ index ] = x_p[ index ];
				}
			}
			return SUCCESS;
		}

		template< bool left_scalar, bool right_scalar, bool left_sparse, bool right_sparse, Descriptor descr, class OP, typename OutputType, typename MaskType, typename InputType1, typename InputType2 >
		RC masked_apply_generic( OutputType * const z_p,
			Coordinates< banshee > & z_coors,
			const MaskType * const mask_p,
			const Coordinates< banshee > & mask_coors,
			const InputType1 * const x_p,
			const InputType2 * const y_p,
			const OP & op,
			const size_t n,
			const Coordinates< banshee > * const left_coors = NULL,
			const InputType1 * const left_identity = NULL,
			const Coordinates< banshee > * const right_coors = NULL,
			const InputType2 * const right_identity = NULL ) {
			// assertions
			static_assert( ! ( left_scalar && left_sparse ), "left_scalar and left_sparse cannot both be set!" );
			static_assert( ! ( right_scalar && right_sparse ), "right_scalar and right_sparse cannot both be set!" );
			assert( ! left_sparse || left_coors != NULL );
			assert( ! left_sparse || left_identity != NULL );
			assert( ! right_sparse || right_coors != NULL );
			assert( ! right_sparse || right_identity != NULL );

#ifdef _DEBUG
			printf( "\tinternal::masked_apply_generic called with nnz(mask)=%d "
					"and descriptor %d\n",
				(int)mask_coors.nonzeroes(), (int)descr );
			if( mask_coors.nonzeroes() > 0 ) {
				printf( "\t\tNonzero mask indices: %d", (int)mask_coors.index( 0 ) );
				assert( mask_coors.assigned( mask_coors.index( 0 ) ) );
				for( size_t k = 1; k < mask_coors.nonzeroes(); ++k ) {
					printf( ", %d", (int)mask_coors.index( k ) );
					assert( mask_coors.assigned( mask_coors.index( k ) ) );
				}
				printf( "\n" );
			}
			size_t unset = 0;
			for( size_t i = 0; i < mask_coors.size(); ++i ) {
				if( ! mask_coors.assigned( i ) ) {
					(void)++unset;
				}
			}
			assert( unset == mask_coors.size() - mask_coors.nonzeroes() );
#endif
			// whether to use a Theta(n) or a Theta(nnz(mask)) loop
			const bool bigLoop = mask_coors.nonzeroes() == n || ( descr & descriptors::invert_mask );

			// get block size
			constexpr size_t size_t_block_size = config::SIMD_SIZE::value() / sizeof( size_t );
			constexpr size_t op_block_size = OP::blocksize;
			constexpr size_t min_block_size = op_block_size > size_t_block_size ? size_t_block_size : op_block_size;
			const size_t block_size = bigLoop ? op_block_size : ( size_t_block_size > 0 ? min_block_size : op_block_size );

			// whether we have a dense hint
			constexpr bool dense = descr & descriptors::dense;

			// declare buffers that fit in a single SIMD register
			bool mask_b[ block_size ];
			OutputType z_b[ block_size ];
			InputType1 x_b[ block_size ];
			InputType2 y_b[ block_size ];

			for( size_t k = 0; k < block_size; ++k ) {
				if( left_scalar ) {
					x_b[ k ] = *x_p;
				}
				if( right_scalar ) {
					y_b[ k ] = *y_p;
				}
			}

			if( bigLoop ) {
#ifdef _DEBUG
				printf( "\t in bigLoop variant\n" );
#endif
				const size_t num_blocks = n / block_size;
				const size_t start = 0;
				const size_t end = num_blocks;
				size_t i = start * block_size;
				// vectorised code
				for( size_t b = start; b < end; ++b ) {
					for( size_t k = 0; k < block_size; ++k ) {
						const size_t index = i + k;
						assert( index < n );
						mask_b[ k ] = mask_coors.template mask< descr >( index, mask_p + index );
					}
					// check for no output
					if( left_sparse && right_sparse ) {
						for( size_t k = 0; k < block_size; ++k ) {
							const size_t index = i + k;
							assert( index < n );
							if( mask_b[ k ] ) {
								if( ! left_coors->assigned( index ) && ! right_coors->assigned( index ) ) {
									mask_b[ k ] = false;
								}
							}
						}
					}
					for( size_t k = 0; k < block_size; ++k ) {
						const size_t index = i + k;
						assert( index < n );
						if( mask_b[ k ] ) {
							if( ! left_scalar ) {
								if( left_sparse && ! left_coors->assigned( index ) ) {
									x_b[ k ] = *left_identity;
								} else {
									x_b[ k ] = *( x_p + index );
								}
							}
							if( ! right_scalar ) {
								if( right_sparse && ! right_coors->assigned( i + k ) ) {
									y_b[ k ] = *right_identity;
								} else {
									y_b[ k ] = *( y_p + index );
								}
							}
						}
					}
					for( size_t k = 0; k < block_size; ++k ) {
						if( mask_b[ k ] ) {
							apply( z_b[ k ], x_b[ k ], y_b[ k ], op );
						}
					}
					for( size_t k = 0; k < block_size; ++k ) {
						const size_t index = i + k;
						assert( index < n );
						if( mask_b[ k ] ) {
							if( ! dense ) {
								(void)z_coors.assign( index );
							}
							*( z_p + index ) = z_b[ k ];
						}
					}
					i += block_size;
				}
				// scalar coda
				for( size_t i = end * block_size; i < n; ++i ) {
					if( mask_coors.template mask< descr >( i, mask_p + i ) ) {
						if( ! dense ) {
							(void)z_coors.assign( i );
						}
						const InputType1 * const x_e = left_scalar ? x_p : ( ( ! left_sparse || left_coors->assigned( i ) ) ? x_p + i : left_identity );
						const InputType2 * const y_e = right_scalar ? y_p : ( ( ! right_sparse || right_coors->assigned( i ) ) ? y_p + i : right_identity );
						OutputType * const z_e = z_p + i;
						apply( *z_e, *x_e, *y_e, op );
					}
				}
			} else {
#ifdef _DEBUG
				printf( "\t in smallLoop variant\n" );
#endif
				// require additional index buffer
				size_t indices[ block_size ];
				// loop over mask pattern
				const size_t mask_nnz = mask_coors.nonzeroes();
				const size_t num_blocks = mask_nnz / block_size;
				const size_t start = 0;
				const size_t end = num_blocks;
				size_t k = start * block_size;
				// vectorised code
				for( size_t b = start; b < end; ++b ) {
					for( size_t t = 0; t < block_size; ++t ) {
						indices[ t ] = mask_coors.index( k + t );
						mask_b[ t ] = mask_coors.template mask< descr >( indices[ t ], mask_p + indices[ t ] );
						if( mask_b[ t ] ) {
							if( ! left_scalar ) {
								if( left_sparse && ! left_coors->assigned( indices[ t ] ) ) {
									x_b[ t ] = *left_identity;
								} else {
									x_b[ t ] = *( x_p + indices[ t ] );
								}
							}
							if( ! right_scalar ) {
								if( right_sparse && ! right_coors->assigned( indices[ t ] ) ) {
									y_b[ t ] = *right_identity;
								} else {
									y_b[ t ] = *( y_p + indices[ t ] );
								}
							}
						}
					}
					// check for no output
					if( left_sparse && right_sparse ) {
						for( size_t t = 0; t < block_size; ++t ) {
							const size_t index = indices[ t ];
							assert( index < n );
							if( mask_b[ t ] ) {
								if( ! left_coors->assigned( index ) && ! right_coors->assigned( index ) ) {
									mask_b[ t ] = false;
								}
							}
						}
					}
					for( size_t t = 0; t < block_size; ++t ) {
						if( mask_b[ t ] ) {
							apply( z_b[ t ], x_b[ t ], y_b[ t ], op );
						}
					}
					for( size_t t = 0; t < block_size; ++t ) {
						if( mask_b[ t ] ) {
							if( ! dense ) {
								(void)z_coors.assign( indices[ t ] );
							}
							*( z_p + indices[ t ] ) = z_b[ t ];
						}
					}
					k += block_size;
				}
				// scalar coda
				for( size_t k = end * block_size; k < mask_nnz; ++k ) {
					const size_t i = mask_coors.index( k );
					if( mask_coors.template mask< descr >( i, mask_p + i ) ) {
						if( left_sparse && right_sparse ) {
							if( ! left_coors->assigned( i ) && ! right_coors->assigned( i ) ) {
								continue;
							}
						}
						if( ! dense ) {
							(void)z_coors.assign( i );
						}
						const InputType1 * const x_e = left_scalar ? x_p : ( ( ! left_sparse || left_coors->assigned( i ) ) ? x_p + i : left_identity );
						const InputType2 * const y_e = right_scalar ? y_p : ( ( ! right_sparse || right_coors->assigned( i ) ) ? y_p + i : right_identity );
						OutputType * const z_e = z_p + i;
						apply( *z_e, *x_e, *y_e, op );
					}
				}
			}
			return SUCCESS;
		}
	} // namespace internal

	template< Descriptor descr = descriptors::no_operation, class OP, typename OutputType, typename MaskType, typename InputType1, typename Coords, typename InputType2 >
	RC eWiseApply( Vector< OutputType, banshee, Coords > & z,
		const Vector< MaskType, banshee, Coords > & mask,
		const Vector< InputType1, banshee, Coords > & x,
		const InputType2 beta,
		const OP & op = OP(),
		const typename std::enable_if< ! grb::is_object< OutputType >::value && ! grb::is_object< MaskType >::value && ! grb::is_object< InputType1 >::value && ! grb::is_object< InputType2 >::value &&
				grb::is_operator< OP >::value,
			void >::type * const = NULL ) {
#ifdef _DEBUG
		printf( "In masked eWiseApply ([T1]<-[T2]<-T3, using operator)\n)" );
#endif
		// check for empty mask
		if( size( mask ) == 0 ) {
			return eWiseApply< descr >( z, x, beta, op );
		}

		// other run-time checks
		const size_t n = internal::getCoordinates( z ).size();
		if( internal::getCoordinates( x ).size() != n ) {
			return MISMATCH;
		}
		if( internal::getCoordinates( mask ).size() != n ) {
			return MISMATCH;
		}
		if( internal::getCoordinates( x ).nonzeroes() < n ) {
			return ILLEGAL;
		}

		auto & z_coors = internal::getCoordinates( z );
		const auto & mask_coors = internal::getCoordinates( mask );
		OutputType * const z_p = internal::getRaw( z );
		const MaskType * const mask_p = internal::getRaw( mask );
		const InputType1 * const x_p = internal::getRaw( x );

		// the output sparsity structure is implied by mask and descr
		z_coors.clear();

		return internal::masked_apply_generic< false, true, false, false, descr >( z_p, z_coors, mask_p, mask_coors, x_p, &beta, op, n );
	}

	template< Descriptor descr = descriptors::no_operation, class Monoid, typename OutputType, typename InputType1, typename InputType2, typename Coords >
	RC eWiseApply( Vector< OutputType, banshee, Coords > & z,
		const Vector< InputType1, banshee, Coords > & x,
		const Vector< InputType2, banshee, Coords > & y,
		const Monoid & monoid = Monoid(),
		const typename std::enable_if< ! grb::is_object< OutputType >::value && ! grb::is_object< InputType1 >::value && ! grb::is_object< InputType2 >::value && grb::is_monoid< Monoid >::value,
			void >::type * const = NULL ) {
#ifdef _DEBUG
		printf( "In unmasked eWiseApply ([T1]<-[T2]<-[T3], using monoid)\n" );
#endif
		// other run-time checks
		const size_t n = internal::getCoordinates( z ).size();
		if( internal::getCoordinates( x ).size() != n ) {
			return MISMATCH;
		}
		if( internal::getCoordinates( y ).size() != n ) {
			return MISMATCH;
		}

		// check if we can dispatch to dense variant
		if( grb::nnz( x ) == n && grb::nnz( y ) == n ) {
			return eWiseApply< descr >( z, x, y, monoid.getOperator() );
		}

		// we are in the unmasked sparse variant
		auto & z_coors = internal::getCoordinates( z );
		OutputType * const z_p = internal::getRaw( z );
		const InputType1 * const x_p = internal::getRaw( x );
		const InputType2 * const y_p = internal::getRaw( y );
		const auto & x_coors = internal::getCoordinates( x );
		const auto & y_coors = internal::getCoordinates( y );
		const auto op = monoid.getOperator();

		// z will have an a-priori unknown sparsity structure
		z_coors.clear();

		return internal::sparse_apply_generic< false, descr, typename Monoid::Operator, OutputType, bool, InputType1, InputType2 >( z_p, z_coors, NULL, x_p, x_coors, y_p, y_coors, op, n );
	}

	template< Descriptor descr = descriptors::no_operation, class Monoid, typename OutputType, typename InputType1, typename InputType2, typename Coords >
	RC eWiseApply( Vector< OutputType, banshee, Coords > & z,
		const InputType1 alpha,
		const Vector< InputType2, banshee, Coords > & y,
		const Monoid & monoid = Monoid(),
		const typename std::enable_if< ! grb::is_object< OutputType >::value && ! grb::is_object< InputType1 >::value && ! grb::is_object< InputType2 >::value && grb::is_monoid< Monoid >::value,
			void >::type * const = NULL ) {
#ifdef _DEBUG
		printf( "In unmasked eWiseApply ([T1]<-T2<-[T3], using monoid)\n" );
#endif

		// other run-time checks
		const size_t n = internal::getCoordinates( z ).size();
		if( internal::getCoordinates( y ).size() != n ) {
			return MISMATCH;
		}

		// check if we can dispatch to dense variant
		if( grb::nnz( y ) == n ) {
			return eWiseApply< descr >( z, alpha, y, monoid.getOperator() );
		}

		// we are in the unmasked sparse variant
		auto & z_coors = internal::getCoordinates( z );
		OutputType * const z_p = internal::getRaw( z );
		const InputType2 * const y_p = internal::getRaw( y );
		const auto & y_coors = internal::getCoordinates( y );
		const auto op = monoid.getOperator();

		// the result will always be dense
		z_coors.assignAll();

		for( size_t i = 0; i < n; ++i ) {
			if( y_coors.assigned( i ) ) {
				grb::apply( z_p[ i ], alpha, y_p[ i ], op );
			} else {
				z_p[ i ] = static_cast< OutputType >( alpha );
			}
		}

		return SUCCESS;
	}

	template< Descriptor descr = descriptors::no_operation, class Monoid, typename OutputType, typename InputType1, typename Coords, typename InputType2 >
	RC eWiseApply( Vector< OutputType, banshee, Coords > & z,
		const Vector< InputType1, banshee, Coords > & x,
		const InputType2 beta,
		const Monoid & monoid = Monoid(),
		const typename std::enable_if< ! grb::is_object< OutputType >::value && ! grb::is_object< InputType1 >::value && ! grb::is_object< InputType2 >::value && grb::is_monoid< Monoid >::value,
			void >::type * const = NULL ) {
#ifdef _DEBUG
		printf( "In unmasked eWiseApply ([T1]<-[T2]<-T3, using monoid)\n" );
#endif
		// other run-time checks
		const size_t n = internal::getCoordinates( z ).size();
		if( internal::getCoordinates( x ).size() != n ) {
			return MISMATCH;
		}

		// check if we can dispatch to dense variant
		if( grb::nnz( x ) == n ) {
			return eWiseApply< descr >( z, x, beta, monoid.getOperator() );
		}

		// we are in the unmasked sparse variant
		auto & z_coors = internal::getCoordinates( z );
		OutputType * const z_p = internal::getRaw( z );
		const InputType1 * const x_p = internal::getRaw( x );
		const auto & x_coors = internal::getCoordinates( x );
		const auto op = monoid.getOperator();

		// the result will always be dense
		z_coors.assignAll();

		for( size_t i = 0; i < n; ++i ) {
			if( x_coors.assigned( i ) ) {
				grb::apply( z_p[ i ], x_p[ i ], beta, op );
			} else {
				z_p[ i ] = static_cast< OutputType >( beta );
			}
		}

		return SUCCESS;
	}

	template< Descriptor descr = descriptors::no_operation, class Monoid, typename OutputType, typename MaskType, typename InputType1, typename InputType2, typename Coords >
	RC eWiseApply( Vector< OutputType, banshee, Coords > & z,
		const Vector< MaskType, banshee, Coords > & mask,
		const Vector< InputType1, banshee, Coords > & x,
		const Vector< InputType2, banshee, Coords > & y,
		const Monoid & monoid = Monoid(),
		const typename std::enable_if< ! grb::is_object< OutputType >::value && ! grb::is_object< MaskType >::value && ! grb::is_object< InputType1 >::value && ! grb::is_object< InputType2 >::value &&
				grb::is_monoid< Monoid >::value,
			void >::type * const = NULL ) {
#ifdef _DEBUG
		printf( "In masked eWiseApply ([T1]<-[T2]<-[T3], using monoid)\n" );
#endif
		if( size( mask ) == 0 ) {
			return eWiseApply< descr >( z, x, y, monoid );
		}

		// other run-time checks
		const size_t n = internal::getCoordinates( z ).size();
		if( internal::getCoordinates( x ).size() != n ) {
			return MISMATCH;
		}
		if( internal::getCoordinates( y ).size() != n ) {
			return MISMATCH;
		}
		if( internal::getCoordinates( mask ).size() != n ) {
			return MISMATCH;
		}

		// check if we can dispatch to dense variant
		if( grb::nnz( x ) == n && grb::nnz( y ) == n ) {
			return eWiseApply< descr >( z, mask, x, y, monoid.getOperator() );
		}

		// we are in the masked sparse variant
		auto & z_coors = internal::getCoordinates( z );
		const auto & mask_coors = internal::getCoordinates( mask );
		OutputType * const z_p = internal::getRaw( z );
		const MaskType * const mask_p = internal::getRaw( mask );
		const InputType1 * const x_p = internal::getRaw( x );
		const InputType2 * const y_p = internal::getRaw( y );
		const auto & x_coors = internal::getCoordinates( x );
		const auto & y_coors = internal::getCoordinates( y );
		const InputType1 left_identity = monoid.template getIdentity< InputType1 >();
		const InputType2 right_identity = monoid.template getIdentity< InputType2 >();
		const auto op = monoid.getOperator();

		// z will have an a priori unknown sparsity structure
		z_coors.clear();

		if( grb::nnz( x ) < n && grb::nnz( y ) < n && grb::nnz( x ) + grb::nnz( y ) < grb::nnz( mask ) ) {
			return internal::sparse_apply_generic< true, descr >( z_p, z_coors, mask_p, x_p, x_coors, y_p, y_coors, op, n, &mask_coors );
		} else if( grb::nnz( x ) < n && grb::nnz( y ) == n ) {
			return internal::masked_apply_generic< false, false, true, false, descr, typename Monoid::Operator, OutputType, MaskType, InputType1, InputType2 >(
				z_p, z_coors, mask_p, mask_coors, x_p, y_p, op, n, &x_coors, &left_identity );
		} else if( grb::nnz( y ) < n && grb::nnz( x ) == n ) {
			return internal::masked_apply_generic< false, false, false, true, descr, typename Monoid::Operator, OutputType, MaskType, InputType1, InputType2 >(
				z_p, z_coors, mask_p, mask_coors, x_p, y_p, op, n, NULL, NULL, &y_coors, &right_identity );
		} else {
			return internal::masked_apply_generic< false, false, true, true, descr, typename Monoid::Operator, OutputType, MaskType, InputType1, InputType2 >(
				z_p, z_coors, mask_p, mask_coors, x_p, y_p, op, n, &x_coors, &left_identity, &y_coors, &right_identity );
		}
	}

	template< Descriptor descr = descriptors::no_operation, class Monoid, typename OutputType, typename MaskType, typename InputType1, typename InputType2, typename Coords >
	RC eWiseApply( Vector< OutputType, banshee, Coords > & z,
		const Vector< MaskType, banshee, Coords > & mask,
		const InputType1 alpha,
		const Vector< InputType2, banshee, Coords > & y,
		const Monoid & monoid = Monoid(),
		const typename std::enable_if< ! grb::is_object< OutputType >::value && ! grb::is_object< MaskType >::value && ! grb::is_object< InputType1 >::value && ! grb::is_object< InputType2 >::value &&
				grb::is_monoid< Monoid >::value,
			void >::type * const = NULL ) {
#ifdef _DEBUG
		printf( "In masked eWiseApply ([T1]<-T2<-[T3], using monoid)\n" );
#endif
		if( size( mask ) == 0 ) {
			return eWiseApply< descr >( z, alpha, y, monoid );
		}

		// other run-time checks
		const size_t n = internal::getCoordinates( z ).size();
		if( internal::getCoordinates( y ).size() != n ) {
			return MISMATCH;
		}
		if( internal::getCoordinates( mask ).size() != n ) {
			return MISMATCH;
		}

		// check if we can dispatch to dense variant
		if( grb::nnz( y ) == n ) {
			return eWiseApply< descr >( z, mask, alpha, y, monoid.getOperator() );
		}

		// we are in the masked sparse variant
		auto & z_coors = internal::getCoordinates( z );
		const auto & mask_coors = internal::getCoordinates( mask );
		OutputType * const z_p = internal::getRaw( z );
		const MaskType * const mask_p = internal::getRaw( mask );
		const InputType2 * const y_p = internal::getRaw( y );
		const auto & y_coors = internal::getCoordinates( y );
		const InputType2 right_identity = monoid.template getIdentity< InputType2 >();
		const auto op = monoid.getOperator();

		// the sparsity structure of z will be a result of the given mask and descr
		z_coors.clear();

		return internal::masked_apply_generic< true, false, false, true, descr, typename Monoid::Operator, OutputType, MaskType, InputType1, InputType2 >(
			z_p, z_coors, mask_p, mask_coors, &alpha, y_p, op, n, NULL, NULL, &y_coors, &right_identity );
	}

	template< Descriptor descr = descriptors::no_operation, class Monoid, typename OutputType, typename MaskType, typename InputType1, typename InputType2, typename Coords >
	RC eWiseApply( Vector< OutputType, banshee, Coords > & z,
		const Vector< MaskType, banshee, Coords > & mask,
		const Vector< InputType1, banshee, Coords > & x,
		const InputType2 beta,
		const Monoid & monoid = Monoid(),
		const typename std::enable_if< ! grb::is_object< OutputType >::value && ! grb::is_object< MaskType >::value && ! grb::is_object< InputType1 >::value && ! grb::is_object< InputType2 >::value &&
				grb::is_monoid< Monoid >::value,
			void >::type * const = NULL ) {
#ifdef _DEBUG
		printf( "In masked eWiseApply ([T1]<-[T2]<-T3, using monoid)\n" );
#endif
		if( size( mask ) == 0 ) {
			return eWiseApply< descr >( z, x, beta, monoid );
		}

		// other run-time checks
		const size_t n = internal::getCoordinates( z ).size();
		if( internal::getCoordinates( x ).size() != n ) {
			return MISMATCH;
		}
		if( internal::getCoordinates( mask ).size() != n ) {
			return MISMATCH;
		}

		// check if we can dispatch to dense variant
		if( grb::nnz( x ) == n ) {
			return eWiseApply< descr >( z, mask, x, beta, monoid.getOperator() );
		}

		// we are in the masked sparse variant
		auto & z_coors = internal::getCoordinates( z );
		const auto & mask_coors = internal::getCoordinates( mask );
		OutputType * const z_p = internal::getRaw( z );
		const MaskType * const mask_p = internal::getRaw( mask );
		const InputType1 * const x_p = internal::getRaw( x );
		const auto & x_coors = internal::getCoordinates( x );
		const InputType1 left_identity = monoid.template getIdentity< InputType1 >();
		const auto op = monoid.getOperator();

		// the sparsity structure of z will be the result of the given mask and descr
		z_coors.clear();

		return internal::masked_apply_generic< false, true, true, false, descr >( z_p, z_coors, mask_p, mask_coors, x_p, &beta, op, n, &x_coors, &left_identity );
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
	 * @tparam InputType1 The value type of the left-hand scalar.
	 * @tparam InputType2 The value type of the right-hand side vector.
	 * @tparam OutputType The value type of the ouput vector.
	 *
	 * @param[in]  alpha The left-hand scalar.
	 * @param[in]   y    The right-hand input vector.
	 * @param[out]  z    The pre-allocated output vector.
	 * @param[in]   op   The operator to use.
	 *
	 * @return grb::MISMATCH Whenever the dimensions of \a y and \a z do not
	 *                       match. All input data containers are left untouched
	 *                       if this exit code is returned; it will be as though
	 *                       this call was never made.
	 * @return grb::SUCCESS  On successful completion of this call.
	 *
	 * \parblock
	 * \par Performance semantics
	 *      -# This call comprises \f$ \Theta(n) \f$ work, where \f$ n \f$ equals
	 *         the size of the vectors \a y and \a z. The constant factor depends
	 *         on the cost of evaluating the operator. A good implementation uses
	 *         vectorised instructions whenever the input domains, the output
	 *         domain, and the operator used allow for this.
	 *
	 *      -# This call takes \f$ \mathcal{O}(1) \f$ memory beyond the memory
	 *         used by the application at the point of a call to this function.
	 *
	 *      -# This call incurs at most
	 *         \f$ n(
	 *               \mathit{sizeof}(\mathit{D2}) + \mathit{sizeof}(\mathit{D3})
	 *             ) +
	 *         \mathcal{O}(1) \f$
	 *         bytes of data movement. A good implementation will stream \a y
	 *         into \a z to apply the multiplication operator in-place, whenever
	 *         the input domains, the output domain, and the operator allow for
	 *         this.
	 * \endparblock
	 */
	template< Descriptor descr = descriptors::no_operation, class OP, typename OutputType, typename InputType1, typename InputType2, typename Coords >
	RC eWiseApply( Vector< OutputType, banshee, Coords > & z,
		const InputType1 alpha,
		const Vector< InputType2, banshee, Coords > & y,
		const OP & op = OP(),
		const typename std::enable_if< ! grb::is_object< OutputType >::value && ! grb::is_object< InputType1 >::value && ! grb::is_object< InputType2 >::value && grb::is_operator< OP >::value,
			void >::type * const = NULL ) {
#ifdef _DEBUG
		printf( "In eWiseApply ([T1]<-T2<-[T3]), operator variant\n" );
#endif
		// sanity check
		const size_t n = internal::getCoordinates( z ).nonzeroes();
		if( internal::getCoordinates( y ).nonzeroes() != n ) {
			return MISMATCH;
		}
		if( internal::getCoordinates( y ).nonzeroes() < n ) {
			return ILLEGAL;
		}

		if( internal::getCoordinates( z ).nonzeroes() < n ) {
			internal::getCoordinates( z ).assignAll();
		}

		return eWiseLambda(
			[ &alpha, &y, &z, &op ]( const size_t i ) {
				apply( z[ i ], alpha, y[ i ], op );
			},
			z );
	}

	template< Descriptor descr = descriptors::no_operation, class OP, typename OutputType, typename MaskType, typename InputType1, typename InputType2, typename Coords >
	RC eWiseApply( Vector< OutputType, banshee, Coords > & z,
		const Vector< MaskType, banshee, Coords > & mask,
		const InputType1 alpha,
		const Vector< InputType2, banshee, Coords > & y,
		const OP & op = OP(),
		const typename std::enable_if< ! grb::is_object< OutputType >::value && ! grb::is_object< MaskType >::value && ! grb::is_object< InputType1 >::value && ! grb::is_object< InputType2 >::value &&
				grb::is_operator< OP >::value,
			void >::type * const = NULL ) {
#ifdef _DEBUG
		printf( "In masked eWiseApply ([T1]<-T2<-[T3], operator variant)\n" );
#endif
		// check for empty mask
		if( size( mask ) == 0 ) {
			return eWiseApply< descr >( z, alpha, y, op );
		}

		// sanity check
		const size_t n = internal::getCoordinates( z ).size();
		if( internal::getCoordinates( y ).size() != n ) {
			return MISMATCH;
		}
		if( internal::getCoordinates( mask ).size() != n ) {
			return MISMATCH;
		}
		if( internal::getCoordinates( y ).nonzeroes() < n ) {
			return ILLEGAL;
		}

		auto & z_coors = internal::getCoordinates( z );
		const auto & mask_coors = internal::getCoordinates( mask );
		OutputType * const z_p = internal::getRaw( z );
		const MaskType * const mask_p = internal::getRaw( mask );
		const InputType2 * const y_p = internal::getRaw( y );

		// the output sparsity structure is implied by mask and descr
		z_coors.clear();

		return internal::masked_apply_generic< true, false, false, false, descr >( z_p, z_coors, mask_p, mask_coors, &alpha, y_p, op, n );
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
	 * @tparam InputType1 The value type of the left-hand side vector.
	 * @tparam InputType2 The value type of the right-hand side vector.
	 * @tparam OutputType The value type of the ouput vector.
	 *
	 * @param[in]  x  The left-hand input vector. May not equal \a y.
	 * @param[in]  y  The right-hand input vector. May not equal \a x.
	 * @param[out] z  The pre-allocated output vector.
	 * @param[in]  op The operator to use.
	 *
	 * @return grb::ILLEGAL  When \a x equals \a y.
	 * @return grb::MISMATCH Whenever the dimensions of \a x, \a y, and \a z
	 *                       do not match. All input data containers are left
	 *                       untouched if this exit code is returned; it will
	 *                       be as though this call was never made.
	 * @return grb::SUCCESS  On successful completion of this call.
	 *
	 * \parblock
	 * \par Performance semantics
	 *      -# This call comprises \f$ \Theta(n) \f$ work, where \f$ n \f$ equals
	 *         the size of the vectors \a x, \a y, and \a z. The constant factor
	 *         depends on the cost of evaluating the operator. A good
	 *         implementation uses vectorised instructions whenever the input
	 *         domains, the output domain, and the operator used allow for this.
	 *
	 *      -# This call takes \f$ \mathcal{O}(1) \f$ memory beyond the memory
	 *         used by the application at the point of a call to this function.
	 *
	 *      -# This call incurs at most
	 *         \f$ n(
	 *               \mathit{sizeof}(\mathit{OutputType}) +
	 *               \mathit{sizeof}(\mathit{InputType1}) +
	 *               \mathit{sizeof}(\mathit{InputType2})
	 *             ) +
	 *         \mathcal{O}(1) \f$
	 *         bytes of data movement. A good implementation will stream \a x or
	 *         \a y into \a z to apply the multiplication operator in-place,
	 *         whenever the input domains, the output domain, and the operator
	 *         used allow for this.
	 * \endparblock
	 */
	template< Descriptor descr = descriptors::no_operation, class OP, typename OutputType, typename InputType1, typename InputType2, typename Coords >
	RC eWiseApply( Vector< OutputType, banshee, Coords > & z,
		const Vector< InputType1, banshee, Coords > & x,
		const Vector< InputType2, banshee, Coords > & y,
		const OP & op = OP(),
		const typename std::enable_if< ! grb::is_object< OutputType >::value && ! grb::is_object< InputType1 >::value && ! grb::is_object< InputType2 >::value && grb::is_operator< OP >::value,
			void >::type * const = NULL ) {
#ifdef _DEBUG
		printf( "In eWiseApply ([T1]<-[T2]<-[T3]), operator variant\n" );
#endif
		// sanity check
		const size_t n = internal::getCoordinates( z ).size();
		if( internal::getCoordinates( x ).size() != n || internal::getCoordinates( y ).size() != n ) {
#ifdef _DEBUG
			printf( "\tinput vectors mismatch in dimensions!\n" );
#endif
			return MISMATCH;
		}
		if( internal::getCoordinates( x ).nonzeroes() < n ) {
#ifdef _DEBUG
			printf( "\tleft-hand input vector is sparse but I have been given "
					"an operator, not a monoid!\n" );
#endif
			return ILLEGAL;
		}
		if( internal::getCoordinates( y ).nonzeroes() < n ) {
#ifdef _DEBUG
			printf( "\tright-hand input vector is sparse but I have been given "
					"an operator, not a monoid!\n" );
#endif
			return ILLEGAL;
		}

		if( internal::getCoordinates( z ).nonzeroes() < n ) {
			internal::getCoordinates( z ).assignAll();
		}

		const InputType1 * __restrict__ a = internal::getRaw( x );
		const InputType2 * __restrict__ b = internal::getRaw( y );
		OutputType * __restrict__ c = internal::getRaw( z );

		// check for possible shortcuts
		if( static_cast< const void * >( &x ) == static_cast< const void * >( &y ) && is_idempotent< OP >::value ) {
			return set< descr >( z, x );
		}

		// check whether an in-place variant is actually requested
		if( static_cast< const void * >( a ) == static_cast< void * >( c ) ) {
			return foldl< descr >( z, y, op );
		}
		if( static_cast< const void * >( b ) == static_cast< void * >( c ) ) {
			return foldr< descr >( x, z, op );
		}

		// no, so use eWiseApply
		{
			const size_t start = 0;
			const size_t end = n;
			if( end > start ) {
				op.eWiseApply( a + start, b + start, c + start,
					end - start ); // this function is vectorised
			}
		}

		// done
		return SUCCESS;
	}

	template< Descriptor descr = descriptors::no_operation, class OP, typename OutputType, typename MaskType, typename InputType1, typename InputType2, typename Coords >
	RC eWiseApply( Vector< OutputType, banshee, Coords > & z,
		const Vector< MaskType, banshee, Coords > & mask,
		const Vector< InputType1, banshee, Coords > & x,
		const Vector< InputType2, banshee, Coords > & y,
		const OP & op = OP(),
		const typename std::enable_if< ! grb::is_object< OutputType >::value && ! grb::is_object< MaskType >::value && ! grb::is_object< InputType1 >::value && ! grb::is_object< InputType2 >::value &&
				grb::is_operator< OP >::value,
			void >::type * const = NULL ) {
#ifdef _DEBUG
		printf( "In masked eWiseApply ([T1]<-[T2]<-[T3], using operator)\n" );
#endif
		// check for empty mask
		if( size( mask ) == 0 ) {
			return eWiseApply< descr >( z, x, y, op );
		}

		// other run-time checks
		auto & z_coors = internal::getCoordinates( z );
		const auto & mask_coors = internal::getCoordinates( mask );
		const size_t n = z_coors.size();
		if( internal::getCoordinates( x ).size() != n ) {
			return MISMATCH;
		}
		if( internal::getCoordinates( y ).size() != n ) {
			return MISMATCH;
		}
		if( mask_coors.size() != n ) {
			return MISMATCH;
		}
		if( internal::getCoordinates( x ).nonzeroes() < n ) {
			return ILLEGAL;
		}
		if( internal::getCoordinates( y ).nonzeroes() < n ) {
			return ILLEGAL;
		}

		OutputType * const z_p = internal::getRaw( z );
		const MaskType * const mask_p = internal::getRaw( mask );
		const InputType1 * const x_p = internal::getRaw( x );
		const InputType2 * const y_p = internal::getRaw( y );

		// the output sparsity structure is unknown a priori
		z_coors.clear();

		return internal::masked_apply_generic< false, false, false, false, descr >( z_p, z_coors, mask_p, mask_coors, x_p, y_p, op, n );
	}

	/**
	 * Calculates the element-wise addition of two vectors, \f$ z = x .+ y \f$,
	 * under this semiring.
	 *
	 * @tparam descr      The descriptor to be used (descriptors::no_operation
	 *                    if left unspecified).
	 * @tparam Ring       The semiring type to perform the element-wise addition
	 *                    on.
	 * @tparam InputType1 The left-hand side input type to the additive operator
	 *                    of the \a ring.
	 * @tparam InputType2 The right-hand side input type to the additive operator
	 *                    of the \a ring.
	 * @tparam OutputType The the result type of the additive operator of the
	 *                    \a ring.
	 *
	 * @param[out]  z  The output vector of type \a OutputType. This may be a
	 *                 sparse vector.
	 * @param[in]   x  The left-hand input vector of type \a InputType1. This may
	 *                 be a sparse vector.
	 * @param[in]   y  The right-hand input vector of type \a InputType2. This may
	 *                 be a sparse vector.
	 * @param[in] ring The generalized semiring under which to perform this
	 *                 element-wise multiplication.
	 *
	 * @return grb::MISMATCH Whenever the dimensions of \a x, \a y, and \a z do
	 *                       not match. All input data containers are left
	 *                       untouched; it will be as though this call was never
	 *                       made.
	 * @return grb::SUCCESS  On successful completion of this call.
	 *
	 * \parblock
	 * \par Valid descriptors
	 * grb::descriptors::no_operation, grb::descriptors::no_casting,
	 * grb::descriptors::dense.
	 *
	 * \note Invalid descriptors will be ignored.
	 *
	 * If grb::descriptors::no_casting is specified, then 1) the third domain of
	 * \a ring must match \a InputType1, 2) the fourth domain of \a ring must match
	 * \a InputType2, 3) the fourth domain of \a ring must match \a OutputType. If
	 * one of these is not true, the code shall not compile.
	 * \endparblock
	 *
	 * \parblock
	 * \par Performance semantics
	 *      -# This call takes \f$ \Theta(n) \f$ work, where \f$ n \f$ equals the
	 *         size of the vectors \a x, \a y, and \a z. The constant factor
	 *         depends on the cost of evaluating the addition operator. A good
	 *         implementation uses vectorised instructions whenever the input
	 *         domains, the output domain, and the the additive operator used
	 *         allow for this.
	 *
	 *      -# This call will not result in additional dynamic memory allocations.
	 *         No system calls will be made.
	 *
	 *      -# This call takes \f$ \mathcal{O}(1) \f$ memory beyond the memory
	 *         used by the application at the point of a call to this function.
	 *
	 *      -# This call incurs at most
	 *         \f$ n( \mathit{sizeof}(
	 *             \mathit{InputType1} +
	 *             \mathit{InputType2} +
	 *             \mathit{OutputType}
	 *           ) + \mathcal{O}(1) \f$
	 *         bytes of data movement. A good implementation will stream \a x or
	 *         \a y into \a z to apply the additive operator in-place, whenever
	 *         the input domains, the output domain, and the operator used allow
	 *         for this.
	 * \endparblock
	 *
	 * @see This is a specialised form of eWiseMulAdd.
	 */
	template< Descriptor descr = descriptors::no_operation, class Ring, typename OutputType, typename InputType1, typename InputType2, typename Coords >
	RC eWiseAdd( Vector< OutputType, banshee, Coords > & z,
		const Vector< InputType1, banshee, Coords > & x,
		const Vector< InputType2, banshee, Coords > & y,
		const Ring & ring = Ring(),
		const typename std::enable_if< ! grb::is_object< OutputType >::value && ! grb::is_object< InputType1 >::value && ! grb::is_object< InputType2 >::value && grb::is_semiring< Ring >::value,
			void >::type * const = NULL ) {
		// static sanity checks
		NO_CAST_OP_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename Ring::D4, OutputType >::value ), "grb::eWiseAdd",
			"called with an output vector with element type that does not match "
			"the fourth domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename Ring::D3, InputType1 >::value ), "grb::eWiseAdd",
			"called with a left-hand side input vector with element type that does "
			"not match the third domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename Ring::D4, OutputType >::value ), "grb::eWiseAdd",
			"called with a right-hand side input vector with element type that "
			"does not match the fourth domain of the given semiring" );

		// get size
		const size_t n = internal::getCoordinates( z ).size();

		// dynamic sanity checks
		if( internal::getCoordinates( x ).size() != n || internal::getCoordinates( y ).size() != n ) {
			return MISMATCH;
		}

		// dense case
		if( internal::getCoordinates( z ).nonzeroes() == n && internal::getCoordinates( x ).nonzeroes() == n && internal::getCoordinates( y ).nonzeroes() == n ) {
			return eWiseApply< descr >( z, x, y, ring.getAdditiveOperator() );
		}

		// sparse case
		RC ret = SUCCESS;
		for( size_t i = 0; i < n; ++i ) {
			// check for zero
			if( ! internal::getCoordinates( x ).assigned( i ) && ! internal::getCoordinates( y ).assigned( i ) ) {
				if( internal::getCoordinates( z ).assigned( i ) ) {
					internal::getRaw( z )[ i ] = ring.template getZero< OutputType >();
				}
				continue;
			}
			// check for copy
			if( internal::getCoordinates( x ).assigned( i ) ) {
				(void)internal::getCoordinates( z ).assign( i );
				internal::getRaw( z )[ i ] = static_cast< OutputType >( static_cast< typename Ring::D3 >( static_cast< typename Ring::D1 >( internal::getRaw( x )[ i ] ) ) );
				continue;
			}
			// check for copy
			if( internal::getCoordinates( y ).assigned( i ) ) {
				(void)internal::getCoordinates( z ).assign( i );
				internal::getRaw( z )[ i ] = static_cast< OutputType >( static_cast< typename Ring::D3 >( static_cast< typename Ring::D2 >( internal::getRaw( y )[ i ] ) ) );
				continue;
			}
			// apply operator
			(void)internal::getCoordinates( z ).assign( i );
			const RC rc = apply( z[ i ], x[ i ], y[ i ], ring.getAdditiveOperator() );
			if( rc != SUCCESS ) {
				ret = rc;
			}
		}
		// done
		return ret;
	}

	/** \todo Documentation pending. */
	template< Descriptor descr = descriptors::no_operation, class Ring, typename InputType1, typename InputType2, typename OutputType, typename Coords >
	RC eWiseAdd( Vector< OutputType, banshee, Coords > & z,
		const InputType1 alpha,
		const Vector< InputType2, banshee, Coords > & y,
		const Ring & ring = Ring(),
		const typename std::enable_if< ! grb::is_object< OutputType >::value && ! grb::is_object< InputType1 >::value && ! grb::is_object< InputType2 >::value && grb::is_semiring< Ring >::value,
			void >::type * const = NULL ) {
		// static sanity checks
		NO_CAST_OP_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename Ring::D1, InputType1 >::value ), "grb::eWiseAdd",
			"called with a left-hand side input vector with element type that does "
			"not match the first domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename Ring::D2, InputType2 >::value ), "grb::eWiseAdd",
			"called with a right-hand side input vector with element type that "
			"does not match the second domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename Ring::D3, OutputType >::value ), "grb::eWiseAdd",
			"called with an output vector with element type that does not match "
			"the third domain of the given semiring" );

		// get size
		const size_t n = internal::getCoordinates( z ).size();

		// dynamic sanity checks
		if( internal::getCoordinates( y ).size() != n ) {
			return MISMATCH;
		}

		// output is always dense
		internal::getCoordinates( z ).assignAll();

		if( ( descr & descriptors::dense ) || internal::getCoordinates( y ).nonzeroes() == n ) {
			constexpr size_t blocksize = Ring::AdditiveOperator::blocksize;
			OutputType out[ blocksize ];
			InputType1 lhs[ blocksize ];
			InputType2 rhs[ blocksize ];
			OutputType * __restrict__ const z_p = internal::getRaw( z );
			const InputType2 * __restrict__ const y_p = internal::getRaw( y );
			for( size_t k = 0; k < blocksize; ++k ) {
				lhs[ k ] = alpha;
			}
			const size_t start = 0;
			const size_t end = n;
			size_t i = start;
			for( ; i < end; i += blocksize ) {
				for( size_t k = 0; k < blocksize; ++k ) {
					rhs[ k ] = y_p[ i + k ];
				}
				for( size_t k = 0; k < blocksize; ++k ) {
					const RC rc = apply( out[ k ], lhs[ k ], rhs[ k ], ring.getAdditiveOperator() );
#ifdef NDEBUG
					(void)rc;
#else
					assert( rc == SUCCESS );
#endif
				}
				for( size_t k = 0; k < blocksize; ++k ) {
					z_p[ i + k ] = out[ k ];
				}
			}
			for( ; i < n; ++i ) {
				const RC rc = apply( internal::getRaw( z )[ i ], alpha, internal::getRaw( y )[ i ], ring.getAdditiveOperator() );
				assert( rc == SUCCESS );
			}
		}

		// sparse input case
		RC ret = SUCCESS;
		for( size_t i = 0; i < n; ++i ) {
			// check for zero
			if( ! internal::getCoordinates( y ).assigned( i ) ) {
				internal::getRaw( z )[ i ] = alpha;
				continue;
			}
			// overwrite old value
			const RC rc = apply( internal::getRaw( z )[ i ], alpha, internal::getRaw( y )[ i ], ring.getAdditiveOperator() );
			assert( rc == SUCCESS );
		}
		// done
		return ret;
	}

	/**
	 * Calculates the element-wise multiplication of two vectors,
	 * \f$ z = x .* y \f$, under this semiring.
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
	 *
	 * @param[out]  z  The output vector of type \a OutputType.
	 * @param[in]   x  The left-hand input vector of type \a InputType1.
	 * @param[in]   y  The right-hand input vector of type \a InputType2.
	 * @param[in] ring The generalized semiring under which to perform this
	 *                 element-wise multiplication.
	 *
	 * @return grb::MISMATCH Whenever the dimensions of \a x, \a y, and \a z do
	 *                       not match. All input data containers are left
	 *                       untouched if this exit code is returned; it will be
	 *                       as though this call was never made.
	 * @return grb::SUCCESS  On successful completion of this call.
	 *
	 * \parblock
	 * \par Valid descriptors
	 * grb::descriptors::no_operation, grb::descriptors::no_casting.
	 *
	 * \note Invalid descriptors will be ignored.
	 *
	 * If grb::descriptors::no_casting is specified, then 1) the first domain of
	 * \a ring must match \a InputType1, 2) the second domain of \a ring must match
	 * \a InputType2, 3) the third domain of \a ring must match \a OutputType. If
	 * one of these is not true, the code shall not compile.
	 *
	 * \endparblock
	 *
	 * \parblock
	 * \par Performance semantics
	 *      -# This call takes \f$ \Theta(n) \f$ work, where \f$ n \f$ equals the
	 *         size of the vectors \a x, \a y, and \a z. The constant factor
	 *         depends on the cost of evaluating the multiplication operator. A
	 *         good implementation uses vectorised instructions whenever the input
	 *         domains, the output domain, and the multiplicative operator used
	 *         allow for this.
	 *
	 *      -# This call will not result in additional dynamic memory allocations.
	 *
	 *      -# This call takes \f$ \mathcal{O}(1) \f$ memory beyond the memory
	 *         used by the application at the point of a call to this function.
	 *
	 *      -# This call incurs at most \f$ n( \mathit{sizeof}(\mathit{D1}) +
	 *         \mathit{sizeof}(\mathit{D2}) + \mathit{sizeof}(\mathit{D3})) +
	 *         \mathcal{O}(1) \f$ bytes of data movement. A good implementation
	 *         will stream \a x or \a y into \a z to apply the multiplication
	 *         operator in-place, whenever the input domains, the output domain,
	 *         and the operator used allow for this.
	 * \endparblock
	 *
	 * \warning
	 *       When given sparse vectors, the zero now annihilates instead of being
	 *       an identity. Thus the eWiseMul cannot simply map to an eWiseApply of
	 *       the multiplicative operator once we have sparse vectors.
	 *
	 * @see This is a specialised form of eWiseMulAdd.
	 */
	template< Descriptor descr = descriptors::no_operation, class Ring, typename InputType1, typename InputType2, typename OutputType, typename Coords >
	RC eWiseMul( Vector< OutputType, banshee, Coords > & z,
		const Vector< InputType1, banshee, Coords > & x,
		const Vector< InputType2, banshee, Coords > & y,
		const Ring & ring = Ring(),
		const typename std::enable_if< ! grb::is_object< OutputType >::value && ! grb::is_object< InputType1 >::value && ! grb::is_object< InputType2 >::value && grb::is_semiring< Ring >::value,
			void >::type * const = NULL ) {
		// static sanity checks
		NO_CAST_OP_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename Ring::D1, InputType1 >::value ), "grb::eWiseMul",
			"called with a left-hand side input vector with element type that does "
			"not match the first domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename Ring::D2, InputType2 >::value ), "grb::eWiseMul",
			"called with a right-hand side input vector with element type that "
			"does not match the second domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename Ring::D3, OutputType >::value ), "grb::eWiseMul",
			"called with an output vector with element type that does not match "
			"the third domain of the given semiring" );

		// get size
		const size_t n = internal::getCoordinates( z ).size();

		// dynamic sanity checks
		if( internal::getCoordinates( x ).size() != n || internal::getCoordinates( y ).size() != n ) {
			return MISMATCH;
		}

		// check for dense case
		if( n == internal::getCoordinates( x ).nonzeroes() && n == internal::getCoordinates( y ).nonzeroes() && n == internal::getCoordinates( z ).nonzeroes() ) {
			return eWiseApply< descr >( z, x, y, ring.getMultiplicativeOperator() );
		}

		// sparse case
		RC ret = SUCCESS;
		for( size_t i = 0; i < n; ++i ) {
			// check for zero
			if( ! internal::getCoordinates( x ).assigned( i ) || ! internal::getCoordinates( y ).assigned( i ) ) {
				if( internal::getCoordinates( z ).assigned( i ) ) {
					internal::getRaw( z )[ i ] = ring.template getZero< OutputType >();
				}
				continue;
			}
			// apply operator
			(void)internal::getCoordinates( z ).assign( i );
			// overwrite old value
			const RC rc = apply( z[ i ], x[ i ], y[ i ], ring.getMultiplicativeOperator() );
			if( rc != SUCCESS ) {
				ret = rc;
			}
		}
		// done
		return ret;
	}

	/** \todo Documentation pending. */
	template< Descriptor descr = descriptors::no_operation, class Ring, typename InputType1, typename InputType2, typename OutputType, typename Coords >
	RC eWiseMul( Vector< OutputType, banshee, Coords > & z,
		const InputType1 alpha,
		const Vector< InputType2, banshee, Coords > & y,
		const Ring & ring = Ring(),
		const typename std::enable_if< ! grb::is_object< OutputType >::value && ! grb::is_object< InputType1 >::value && ! grb::is_object< InputType2 >::value && grb::is_semiring< Ring >::value,
			void >::type * const = NULL ) {
		// static sanity checks
		NO_CAST_OP_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename Ring::D1, InputType1 >::value ), "grb::eWiseMul",
			"called with a left-hand side input vector with element type that does "
			"not match the first domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename Ring::D2, InputType2 >::value ), "grb::eWiseMul",
			"called with a right-hand side input vector with element type that "
			"does not match the second domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename Ring::D3, OutputType >::value ), "grb::eWiseMul",
			"called with an output vector with element type that does not match "
			"the third domain of the given semiring" );

		// get size
		const size_t n = internal::getCoordinates( z ).size();

		// dynamic sanity checks
		if( internal::getCoordinates( y ).size() != n ) {
			return MISMATCH;
		}

		// cache left-hand side input
		const typename Ring::D1 a = alpha;

		// sparse case
		RC ret = SUCCESS;
		for( size_t i = 0; i < n; ++i ) {
			// check for zero
			if( ! internal::getCoordinates( y ).assigned( i ) ) {
				if( internal::getCoordinates( z ).assigned( i ) ) {
					internal::getRaw( z )[ i ] = ring.template getZero< OutputType >();
				}
				continue;
			}
			// apply operator
			(void)internal::getCoordinates( z ).assign( i );
			// overwrite old value
			const RC rc = apply( internal::getRaw( z )[ i ], a, internal::getRaw( y )[ i ], ring.getMultiplicativeOperator() );
			if( rc != SUCCESS ) {
				ret = rc; // assumes updates to enum RC are atomic!
			}
		}
		// done
		return ret;
	}

	// declare an internal version of eWiseMulAdd containing the full sparse & dense implementations
	namespace internal {

		/** @see grb::eWiseMulAdd */
		template< Descriptor descr = descriptors::no_operation, class Ring, typename InputType1, typename InputType2, typename InputType3, typename OutputType, typename Coords >
		RC eWiseMulAdd( Vector< OutputType, banshee, Coords > & _z,
			const InputType1 alpha,
			const Vector< InputType2, banshee, Coords > & _x,
			const Vector< InputType3, banshee, Coords > & _y,
			const Ring & ring = Ring(),
			const typename std::enable_if< ! grb::is_object< OutputType >::value && ! grb::is_object< InputType1 >::value && ! grb::is_object< InputType2 >::value &&
					! grb::is_object< InputType3 >::value && grb::is_semiring< Ring >::value,
				void >::type * const = NULL ) {

			// check whether we are in the sparse or dense case
			constexpr bool ssparse = ! ( descr & descriptors::dense );
			const size_t n = internal::getCoordinates( _x ).size();
			const bool sparse = ssparse || internal::getCoordinates( _x ).nonzeroes() < n || internal::getCoordinates( _y ).nonzeroes() < n;

			if( ! sparse || internal::getCoordinates( _y ).nonzeroes() == n ) {
				internal::getCoordinates( _z ).assignAll();
			}

			// Could specialise here for sparse x and dense y and vice versa

			if( sparse ) {
				// get raw pointers
				const InputType2 * __restrict__ const x = internal::getRaw( _x );
				const InputType3 * __restrict__ const y = internal::getRaw( _y );
				OutputType * __restrict__ const z = internal::getRaw( _z );
				RC ret = SUCCESS;
				// TODO: issue #42
				for( size_t i = 0; i < n; ++i ) {
					typename Ring::D3 temp;
					if( internal::getCoordinates( _x ).assigned( i ) ) {
						const RC rc = apply( temp, alpha, x[ i ], ring.getMultiplicativeOperator() );
						if( rc != SUCCESS ) {
							ret = rc;
						}
					} else {
						if( ! internal::getCoordinates( _y ).assigned( i ) ) {
							continue;
						} else {
							temp = ring.template getZero< typename Ring::D3 >();
						}
					}
					(void)internal::getCoordinates( _z ).assign( i );
					const RC rc = apply( z[ i ], temp, y[ i ], ring.getAdditiveOperator() );
#ifndef NDEBUG
					(void)rc;
#else
					assert( rc == SUCCESS );
#endif
				}
				return ret;
			}

			// dense case:
			const size_t start = 0;
			const size_t end = n;

			// get raw pointers
			const typename Ring::D1 aa = static_cast< typename Ring::D1 >( alpha );
			const InputType2 * __restrict__ const x = internal::getRaw( _x );
			const InputType3 * __restrict__ const y = internal::getRaw( _y );
			OutputType * __restrict__ const z = internal::getRaw( _z );

			// do vectorised out-of-place operations. Allows for aligned overlap.
			// Non-aligned ovelap is not possible due to GraphBLAS semantics.
			size_t i = start;
			// note: read the tail code (under this while loop) comments first for greater understanding
			while( i + Ring::blocksize <= end ) {
				// vector registers
				typename Ring::D2 xx[ Ring::blocksize ];
				typename Ring::D4 yy[ Ring::blocksize ];
				typename Ring::D3 zz[ Ring::blocksize ];
				bool xmask[ Ring::blocksize ];
				bool ymask[ Ring::blocksize ];

				// read-in
				for( size_t b = 0; b < Ring::blocksize; ++b, ++i ) {
					// read masks
					if( sparse ) { // TODO issue #41, this code should be in the above branch
						xmask[ b ] = internal::getCoordinates( _x ).assigned( i );
						ymask[ b ] = internal::getCoordinates( _y ).assigned( i );
					}

					// if there is no multiplication and no addition, do nothing
					if( sparse && ! xmask[ b ] && ! ymask[ b ] )
						continue;

					// if there is to be multiplied, read right-hand side
					if( ! sparse || xmask[ b ] ) {
						xx[ b ] = static_cast< typename Ring::D2 >( x[ i ] );
					}

					// if there is to be added, read right-hand side
					if( ! sparse || ymask[ b ] ) {
						yy[ b ] = static_cast< typename Ring::D4 >( y[ i ] );
					} else {
						// if there is nothing to be read, set output to zero
						yy[ b ] = ring.template getZero< typename Ring::D4 >();
					}
				}

				// rewind
				i -= Ring::blocksize;

				// operate
				for( size_t b = 0; b < Ring::blocksize; ++b ) {
					// check if we are simply returning zero; if yes, continue
					if( sparse && ! xmask[ b ] && ! ymask[ b ] )
						continue;

					// check if multiplication was necessary
					if( ! sparse || xmask[ b ] ) {
						apply( zz[ b ], aa, xx[ b ], ring.getMultiplicativeOperator() );
						foldr( zz[ b ], yy[ b ], ring.getAdditiveOperator() );
					}

					// in the other case (no multiplication, yes addition),
					// then yy already holds the correct values
				}

				// read-out
				for( size_t b = 0; b < Ring::blocksize; ++b, ++i ) {
					// if we were returning zero
					if( sparse && ! xmask[ b ] && ! ymask[ b ] ) {
						// set zero
						if( internal::getCoordinates( _z ).assigned( i ) ) {
							internal::getRaw( _z )[ i ] = ring.template getZero< typename Ring::D4 >();
						}
					} else {
						// write back result
						if( sparse ) {
							(void)internal::getCoordinates( _z ).assign( i );
						}
						z[ i ] = static_cast< OutputType >( yy[ b ] );
					}
				}
			}

			// perform tail
			for( ; i < end; ++i ) {
				// all zero-- so return zero
				if( sparse && ! internal::getCoordinates( _x ).assigned( i ) && ! internal::getCoordinates( _y ).assigned( i ) ) {
					if( internal::getCoordinates( _z ).assigned( i ) ) {
						internal::getRaw( _z )[ i ] = ring.template getZero< typename Ring::D4 >();
					}
					// done
					continue;
				}

				// x is zero-- just copy the value that is to be added to zero
				if( sparse && ! internal::getCoordinates( _x ).assigned( i ) ) {
					(void)internal::getCoordinates( _z ).assign( i );
					// keep the same casting behaviour
					z[ i ] = static_cast< OutputType >( static_cast< typename Ring::D4 >( y[ i ] ) );
					// done
					continue;
				}

				// do multiply
				const typename Ring::D2 xx = static_cast< typename Ring::D2 >( x[ i ] );
				typename Ring::D3 zz;
				apply( zz, aa, xx, ring.getMultiplicativeOperator() );

				// get value to add to
				typename Ring::D4 yy = internal::getCoordinates( _y ).assigned( i ) ? static_cast< typename Ring::D4 >( y[ i ] ) : ring.template getZero< typename Ring::D4 >();

				// do add
				foldr( zz, yy, ring.getAdditiveOperator() );

				// write out
				if( sparse ) {
					(void)internal::getCoordinates( _z ).assign( i );
				}
				z[ i ] = static_cast< OutputType >( yy );
			}

			// done
			return SUCCESS;
		}

		/** @see grb::eWiseMulAdd */
		template< Descriptor descr = descriptors::no_operation, class Ring, typename InputType1, typename InputType2, typename InputType3, typename OutputType, typename Coords >
		RC eWiseMulAdd( Vector< OutputType, banshee, Coords > & _z,
			const Vector< InputType1, banshee, Coords > & _a,
			const Vector< InputType2, banshee, Coords > & _x,
			const Vector< InputType3, banshee, Coords > & _y,
			const Ring & ring = Ring(),
			const typename std::enable_if< ! grb::is_object< OutputType >::value && ! grb::is_object< InputType1 >::value && ! grb::is_object< InputType2 >::value &&
					! grb::is_object< InputType3 >::value && grb::is_semiring< Ring >::value,
				void >::type * const = NULL ) {
			// check for dense descriptor
			constexpr bool sparse = ! ( descr & descriptors::dense );

			// get size
			const size_t n = internal::getCoordinates( _x ).size();
			if( sparse ) {
				// get raw pointers
				const InputType1 * __restrict__ const a = internal::getRaw( _a );
				const InputType2 * __restrict__ const x = internal::getRaw( _x );
				const InputType3 * __restrict__ const y = internal::getRaw( _y );
				OutputType * __restrict__ const z = internal::getRaw( _z );
				RC ret = SUCCESS;
				for( size_t i = 0; i < n; ++i ) {
					typename Ring::D3 temp;
					// both have to be nonzero for multiplication to be useful
					if( internal::getCoordinates( _a ).assigned( i ) && internal::getCoordinates( _x ).assigned( i ) ) {
						const RC rc = apply( temp, a[ i ], x[ i ], ring.getMultiplicativeOperator() );
						if( rc != SUCCESS ) {
							ret = rc;
						}
					} else {
						// if multiply yields zero and there's nothing to add
						if( ! internal::getCoordinates( _y ).assigned( i ) ) {
							// then do not do anything to output
							continue;
						} else {
							// otherwise set temp to 0
							temp = ring.template getZero< typename Ring::D3 >();
						}
					}
					const RC rc = apply( z[ i ], temp, y[ i ], ring.getAdditiveOperator() );
					if( rc != SUCCESS ) {
						ret = rc;
					} else {
						(void)internal::getCoordinates( _z ).assign( i );
					}
				}
				return ret;
			}

			// dense case:
			const size_t start = 0;
			const size_t end = n;

			// get raw pointers
			const InputType1 * __restrict__ const a = internal::getRaw( _a );
			const InputType2 * __restrict__ const x = internal::getRaw( _x );
			const InputType3 * __restrict__ const y = internal::getRaw( _y );
			OutputType * __restrict__ const z = internal::getRaw( _z );

			// do vectorised out-of-place operations. Allows for aligned overlap.
			// Non-aligned ovelap is not possible due to GraphBLAS semantics.
			size_t i = start;
			while( i + Ring::blocksize <= end ) {
				// vector registers
				typename Ring::D1 aa[ Ring::blocksize ];
				typename Ring::D2 xx[ Ring::blocksize ];
				typename Ring::D4 yy[ Ring::blocksize ];
				typename Ring::D3 zz[ Ring::blocksize ];
				bool amask[ Ring::blocksize ];
				bool xmask[ Ring::blocksize ];
				bool ymask[ Ring::blocksize ];

				// read-in
				for( size_t b = 0; b < Ring::blocksize; ++b, ++i ) {
					// get masks
					if( sparse ) {
						amask[ b ] = internal::getCoordinates( _a ).assigned( i );
						xmask[ b ] = internal::getCoordinates( _x ).assigned( i );
						ymask[ b ] = internal::getCoordinates( _y ).assigned( i );
					}
					// if multiplication is necessary
					if( ! sparse || ( amask[ b ] && xmask[ b ] ) ) {
						// read values to be multiplied
						aa[ b ] = static_cast< typename Ring::D1 >( a[ i ] );
						xx[ b ] = static_cast< typename Ring::D2 >( x[ i ] );
					}
					// if addition is necessary
					if( ! sparse || ymask[ b ] ) {
						// read values to be added to
						yy[ b ] = static_cast< typename Ring::D4 >( y[ i ] );
					} else {
						// set output to zero
						yy[ b ] = ring.template getZero< typename Ring::D4 >();
					}
				}

				// rewind
				i -= Ring::blocksize;

				// operate
				for( size_t b = 0; b < Ring::blocksize; ++b ) {
					// do multiplication, if requested
					if( ! sparse || ( amask[ b ] && xmask[ b ] ) ) {
						apply( zz[ b ], aa[ b ], xx[ b ], ring.getMultiplicativeOperator() );
					}
					// do addition, if requested
					if( ! sparse || ymask[ b ] ) {
						foldr( zz[ b ], yy[ b ], ring.getAdditiveOperator() );
					}
				}

				// read-out
				for( size_t b = 0; b < Ring::blocksize; ++b, ++i ) {
					// if we end up with a zero value
					if( sparse && yy[ b ] == ring.template getZero< typename Ring::D4 >() ) {
						// then subtract it from the set of nonzeroes stored
						if( internal::getCoordinates( _z ).assigned( i ) ) {
							internal::getRaw( _z )[ i ] = ring.template getZero< typename Ring::D4 >();
						}
					} else {
						// check if this is a new nonzero
						if( sparse ) {
							(void)internal::getCoordinates( _z ).assign( i );
						}
						// record new value
						z[ i ] = static_cast< OutputType >( yy[ b ] );
					}
				}
			}

			// perform tail
			for( ; i < end; ++i ) {
				// check if multiplication is necessary at all
				if( sparse && ( ! internal::getCoordinates( _a ).assigned( i ) || ! internal::getCoordinates( _x ).assigned( i ) ) ) {
					// if not, check if addition is necessary
					if( internal::getCoordinates( _y ).assigned( i ) ) {
						// yes, so copy
						(void)internal::getCoordinates( _z ).assign( i );
						// keep the same casting behaviour
						z[ i ] = static_cast< OutputType >( static_cast< typename Ring::D4 >( y[ i ] ) );
					} else {
						// if addition is also zero, then the result is zero
						if( internal::getCoordinates( _z ).assigned( i ) ) {
							internal::getRaw( _z )[ i ] = ring.template getZero< typename Ring::D4 >();
						}
					}
				}

				// do multiply
				const typename Ring::D1 aa = static_cast< typename Ring::D1 >( a[ i ] );
				const typename Ring::D2 xx = static_cast< typename Ring::D2 >( x[ i ] );
				typename Ring::D4 zz;
				(void)apply( zz, aa, xx, ring.getMultiplicativeOperator() );

				// get value to add
				typename Ring::D4 yy;
				if( ! sparse || internal::getCoordinates( _y ).assigned( i ) ) {
					yy = static_cast< typename Ring::D4 >( y[ i ] );
				} else {
					yy = ring.template getZero< typename Ring::D4 >();
				}

				// do add
				foldr( zz, yy, ring.getAdditiveOperator() );

				// write out
				(void)internal::getCoordinates( _z ).assign( i );
				z[ i ] = static_cast< OutputType >( yy );
			}

			// done
			return SUCCESS;
		}

	} // namespace internal

	/**
	 * Calculates the axpy, \f$ z = \alpha * x .+ y \f$, under this semiring.
	 *
	 * @tparam descr      The descriptor to be used (descriptors::no_operation
	 *                    if left unspecified).
	 * @tparam Ring       The semiring type to perform the element-wise
	 *                    multiply-add on.
	 * @tparam InputType1 The left-hand side input type to the multiplicative
	 *                    operator of the \a ring.
	 * @tparam InputType2 The right-hand side input type to the multiplicative
	 *                    operator of the \a ring.
	 * @tparam InputType3 The output type to the multiplicative operator of the
	 *                    \a ring \em and the left-hand side input type to the
	 *                    additive operator of the \a ring.
	 * @tparam OutputType The right-hand side input type to the additive operator
	 *                    of the \a ring \em and the result type of the same
	 *                    operator.
	 *
	 * @param[out] _z    The pre-allocated output vector. Must be a vector from
	 *                   \a _D4.
	 * @param[in]  alpha The scaling factor of x. Must be an element of \a _D1.
	 * @param[in]  _x    The left-hand side input vector. Must be a vector from
	 *                   \a _D2.
	 * @param[in]  _y    The right-hand side input vector. Must be a vector from
	 *                   \a _D4.
	 * @param[in]  ring  The generalized semiring under which to perform this
	 *                   element-wise multiplication.
	 *
	 * @return grb::MISMATCH Whenever the dimensions of \a x, \a y, and \a z do
	 *                       not match. In this case, all input data containers
	 *                       are left untouched and it will simply be as though
	 *                       this call was never made.
	 * @return grb::SUCCESS  On successful completion of this call.
	 *
	 * \parblock
	 * \par Valid descriptors
	 * grb::descriptors::no_operation, grb::descriptors::no_casting.
	 *
	 * \note Invalid descriptors will be ignored.
	 *
	 * If grb::descriptors::no_casting is specified, then 1) the first domain of
	 * \a ring must match \a InputType1, 2) the second domain of \a ring must match
	 * \a InputType2, 3) the third domain of \a ring must match \a InputType3,
	 * 4) the fourth domain of \a ring must match \a OutputType. If one of these is
	 * not true, the code shall not compile.
	 *
	 * \endparblock
	 *
	 * \parblock
	 * \par Performance semantics
	 *      -# This call takes \f$ \Theta(n) \f$ work, where \f$ n \f$ equals the
	 *         size of the vectors \a x, \a y, and \a z. The constant factor
	 *         depends on the cost of evaluating the addition and multiplication
	 *         operators. A good implementation uses vectorised instructions
	 *         whenever the input domains, the output domain, and the operators
	 *         used allow for this.
	 *
	 *      -# This call will not allocate any additional memory.
	 *
	 *      -# This call takes \f$ \mathcal{O}(1) \f$ memory beyond those already
	 *         used by the application when this function is called.
	 *
	 *      -# This call incurs at most \f$ n( \mathit{sizeof}(
	 *           \mathit{InputType2} + \mathit{bool}
	 *           \mathit{InputType3} + \mathit{bool}
	 *           \mathit{OutputType} + \mathit{bool}
	 *         ) + \mathit{sizeof}( \mathit{InputType1} ) + \mathcal{O}(1) \f$
	 *         bytes of data movement. A good implementation will apply the
	 *         additive and multiplicative operator in-place, whenever the input
	 *         domains, the output domain, and the operator used allow for this.
	 * \endparblock
	 */
	template< Descriptor descr = descriptors::no_operation, class Ring, typename InputType1, typename InputType2, typename InputType3, typename OutputType, typename Coords >
	RC eWiseMulAdd( Vector< OutputType, banshee, Coords > & _z,
		const InputType1 alpha,
		const Vector< InputType2, banshee, Coords > & _x,
		const Vector< InputType3, banshee, Coords > & _y,
		const Ring & ring = Ring(),
		const typename std::enable_if< ! grb::is_object< OutputType >::value && ! grb::is_object< InputType1 >::value && ! grb::is_object< InputType2 >::value &&
				! grb::is_object< InputType3 >::value && grb::is_semiring< Ring >::value,
			void >::type * const = NULL ) {
		// static sanity checks
		NO_CAST_OP_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename Ring::D1, InputType1 >::value ), "grb::eWiseMulAdd",
			"called with a left-hand scalar alpha of an element type that does not "
			"match the first domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename Ring::D2, InputType2 >::value ), "grb::eWiseMulAdd",
			"called with a right-hand vector _x with an element type that does not "
			"match the second domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename Ring::D4, InputType3 >::value ), "grb::eWiseMulAdd",
			"called with an additive vector _y with an element type that does not "
			"match the fourth domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename Ring::D4, OutputType >::value ), "grb::eWiseMulAdd",
			"called with a result vector _z with an element type that does not "
			"match the fourth domain of the given semiring" );

		// dynamic sanity checks
		const size_t n = internal::getCoordinates( _z ).size();
		if( internal::getCoordinates( _x ).size() != n || internal::getCoordinates( _y ).size() != n ) {
			return MISMATCH;
		}

		// catch trivial cases
		const InputType1 zeroIT1 = ring.template getZero< InputType1 >();
		if( alpha == zeroIT1 ) {
			return set( _z, _y );
		}
		if( internal::getCoordinates( _x ).nonzeroes() == 0 ) {
			return set( _z, _y );
		}
		if( internal::getCoordinates( _y ).nonzeroes() == 0 ) {
			return eWiseMul< descr >( _z, alpha, _x, ring );
		}

		// check for density
		constexpr bool sparse = ! ( descr & descriptors::dense );
		if( sparse ) {
			// check whether all inputs are actually dense
			if( internal::getCoordinates( _x ).nonzeroes() == n && internal::getCoordinates( _y ).nonzeroes() == n && internal::getCoordinates( _z ).nonzeroes() == n ) {
				// yes, so set dense descriptor; performance loss
				// is minimal with an intercept at this point
				return internal::eWiseMulAdd< descr + descriptors::dense >( _z, alpha, _x, _y, ring );
			}
		}

		return internal::eWiseMulAdd< descr >( _z, alpha, _x, _y, ring );
	}

	/**
	 * Calculates the elementwise multiply-add, \f$ z = a .* x .+ y \f$, under
	 * this semiring.
	 *
	 * @tparam descr      The descriptor to be used (descriptors::no_operation
	 *                    if left unspecified).
	 * @tparam Ring       The semiring type to perform the element-wise
	 *                    multiply-add on.
	 * @tparam InputType1 The left-hand side input type to the multiplicative
	 *                    operator of the \a ring.
	 * @tparam InputType2 The right-hand side input type to the multiplicative
	 *                    operator of the \a ring.
	 * @tparam InputType3 The output type to the multiplicative operator of the
	 *                    \a ring \em and the left-hand side input type to the
	 *                    additive operator of the \a ring.
	 * @tparam OutputType The right-hand side input type to the additive operator
	 *                    of the \a ring \em and the result type of the same
	 *                    operator.
	 *
	 * @param[out] _z  The pre-allocated output vector.
	 * @param[in]  _a  The elements for left-hand side multiplication.
	 * @param[in]  _x  The elements for right-hand side multiplication.
	 * @param[in]  _y  The elements for right-hand size addition.
	 * @param[in] ring The ring to perform the eWiseMulAdd under.
	 *
	 * @return grb::MISMATCH Whenever the dimensions of \a _a, \a _x, \a _y, and
	 *                       \a z do not match. In this case, all input data
	 *                       containers are left untouched and it will simply be
	 *                       as though this call was never made.
	 * @return grb::SUCCESS  On successful completion of this call.
	 *
	 * \warning An implementation is not obligated to detect overlap whenever
	 *          it occurs. If part of \a z overlaps with \a x, \a y, or \a a,
	 *          undefined behaviour will occur \em unless this function returns
	 *          grb::OVERLAP. In other words: an implementation which returns
	 *          erroneous results when vectors overlap and still returns
	 *          grb::SUCCESS thus is also a valid GraphBLAS implementation!
	 *
	 * \parblock
	 * \par Valid descriptors
	 * grb::descriptors::no_operation, grb::descriptors::no_casting.
	 *
	 * \note Invalid descriptors will be ignored.
	 *
	 * If grb::descriptors::no_casting is specified, then 1) the first domain of
	 * \a ring must match \a InputType1, 2) the second domain of \a ring must match
	 * \a InputType2, 3) the third domain of \a ring must match \a InputType3,
	 * 4) the fourth domain of \a ring must match \a OutputType. If one of these is
	 * not true, the code shall not compile.
	 * \endparblock
	 *
	 * \parblock
	 * \par Performance semantics
	 *      -# This call takes \f$ \Theta(n) \f$ work, where \f$ n \f$ equals the
	 *         size of the vectors \a _a, \a _x, \a _y, and \a _z. The constant
	 *         factor depends on the cost of evaluating the addition and
	 *         multiplication operators. A good implementation uses vectorised
	 *         instructions whenever the input domains, the output domain, and
	 *         the operators used allow for this.
	 *
	 *      -# This call takes \f$ \mathcal{O}(1) \f$ memory beyond the memory
	 *         already used by the application when this function is called.
	 *
	 *      -# This call incurs at most \f$ n( \mathit{sizeof}(
	 *           \mathit{InputType1} + \mathit{bool}
	 *           \mathit{InputType2} + \mathit{bool}
	 *           \mathit{InputType3} + \mathit{bool}
	 *           \mathit{OutputType} + \mathit{bool}
	 *         ) + \mathcal{O}(1) \f$
	 *         bytes of data movement. A good implementation will stream \a _a,
	 *         \a _x or \a _y into \a _z to apply the additive and multiplicative
	 *         operators in-place, whenever the input domains, the output domain,
	 *         and the operators used allow for this.
	 * \endparblock
	 */
	template< Descriptor descr = descriptors::no_operation, class Ring, typename InputType1, typename InputType2, typename InputType3, typename OutputType, typename Coords >
	RC eWiseMulAdd( Vector< OutputType, banshee, Coords > & _z,
		const Vector< InputType1, banshee, Coords > & _a,
		const Vector< InputType2, banshee, Coords > & _x,
		const Vector< InputType3, banshee, Coords > & _y,
		const Ring & ring = Ring(),
		const typename std::enable_if< ! grb::is_object< OutputType >::value && ! grb::is_object< InputType1 >::value && ! grb::is_object< InputType2 >::value &&
				! grb::is_object< InputType3 >::value && grb::is_semiring< Ring >::value,
			void >::type * const = NULL ) {
		(void)ring;
		// static sanity checks
		NO_CAST_OP_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename Ring::D1, InputType1 >::value ), "grb::eWiseMulAdd",
			"called with a left-hand vector _a with an element type that does not "
			"match the first domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename Ring::D2, InputType2 >::value ), "grb::eWiseMulAdd",
			"called with a right-hand vector _x with an element type that does not "
			"match the second domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename Ring::D4, InputType3 >::value ), "grb::eWiseMulAdd",
			"called with an additive vector _y with an element type that does not "
			"match the fourth domain of the given semiring" );
		NO_CAST_OP_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename Ring::D4, OutputType >::value ), "grb::eWiseMulAdd",
			"called with a result vector _z with an element type that does not "
			"match the fourth domain of the given semiring" );

		// catch trivial cases
		if( internal::getCoordinates( _a ).nonzeroes() == 0 ) {
			return set( _z, _y );
		}
		if( internal::getCoordinates( _x ).nonzeroes() == 0 ) {
			return set( _z, _y );
		}
		if( internal::getCoordinates( _y ).nonzeroes() == 0 ) {
			return eWiseMul< descr >( _z, _a, _x, ring );
		}

		// dynamic sanity checks
		const size_t n = internal::getCoordinates( _z ).size();
		if( internal::getCoordinates( _x ).size() != n || internal::getCoordinates( _y ).size() != n || internal::getCoordinates( _a ).size() != n ) {
			return MISMATCH;
		}

		// check for dense variant
		if( ( descr & descriptors::dense ) ||
			( internal::getCoordinates( _x ).nonzeroes() == n && internal::getCoordinates( _y ).nonzeroes() == n && internal::getCoordinates( _a ).nonzeroes() == n ) ) {
			// check if it's actually dense...
			if( ( descr & descriptors::dense ) &&
				( internal::getCoordinates( _x ).nonzeroes() < n || internal::getCoordinates( _y ).nonzeroes() < n || internal::getCoordinates( _a ).nonzeroes() < n ) ) {
				return ILLEGAL;
			}
			// dense variant sets output dense
			internal::getCoordinates( _z ).assignAll();
			// passes dense descriptor to internal implementation
			return internal::eWiseMulAdd< descr | descriptors::dense >( _z, _a, _x, _y, ring );
		}

		// pass to implementation as is
		return internal::eWiseMulAdd< descr >( _z, _a, _x, _y, ring );
	}

	// open internal namespace for implementation of grb::dot
	namespace internal {

		/** @see grb::dot */
		template< Descriptor descr = descriptors::no_operation, class AddMonoid, class AnyOp, typename OutputType, typename InputType1, typename InputType2, typename Coords >
		RC dot_generic( OutputType & z,
			const Vector< InputType1, banshee, Coords > & x,
			const Vector< InputType2, banshee, Coords > & y,
			const AddMonoid & addMonoid = AddMonoid(),
			const AnyOp & anyOp = AnyOp() ) {
			const size_t n = internal::getCoordinates( x ).size();
			if( n != internal::getCoordinates( y ).size() ) {
				return MISMATCH;
			}

			// check if dense flag is set correctly
			constexpr bool dense = descr & descriptors::dense;
			const size_t nzx = internal::getCoordinates( x ).nonzeroes();
			const size_t nzy = internal::getCoordinates( y ).nonzeroes();
			if( dense ) {
				if( n != nzx || n != nzy ) {
					return PANIC;
				}
			} else {
				if( n == nzx && n == nzy ) {
					return PANIC;
				}
			}

			size_t loopsize = n;
			auto * coors_r_p = &( internal::getCoordinates( x ) );
			auto * coors_q_p = &( internal::getCoordinates( y ) );
			if( ! dense ) {
				if( nzx < nzy ) {
					loopsize = nzx;
				} else {
					loopsize = nzy;
					std::swap( coors_r_p, coors_q_p );
				}
			}
			auto & coors_r = *coors_r_p;
			auto & coors_q = *coors_q_p;

			const size_t start = 0;
			const size_t end = loopsize;
			if( end > start ) {
				// get raw alias
				const InputType1 * __restrict__ a = internal::getRaw( x );
				const InputType2 * __restrict__ b = internal::getRaw( y );

				// overwrite z with first multiplicant
				typename AddMonoid::D3 reduced;
				if( dense ) {
					apply( reduced, a[ end - 1 ], b[ end - 1 ], anyOp );
				} else {
					const size_t index = coors_r.index( end - 1 );
					if( coors_q.assigned( index ) ) {
						apply( reduced, a[ index ], b[ index ], anyOp );
					} else {
						reduced = addMonoid.template getIdentity< typename AddMonoid::D3 >();
					}
				}

				// enter vectorised loop
				size_t i = start;
				if( dense ) {
					while( i + AnyOp::blocksize < end - 1 ) {
						// declare buffers
						static_assert( AnyOp::blocksize > 0,
							"Configuration error: vectorisation blocksize set to "
							"0!" );
						typename AnyOp::D1 xx[ AnyOp::blocksize ];
						typename AnyOp::D2 yy[ AnyOp::blocksize ];
						typename AnyOp::D3 zz[ AnyOp::blocksize ];

						// prepare registers
						for( size_t k = 0; k < AnyOp::blocksize; ++k ) {
							xx[ k ] = static_cast< typename AnyOp::D1 >( a[ i ] );
							yy[ k ] = static_cast< typename AnyOp::D2 >( b[ i++ ] );
						}

						// perform element-wise multiplication
						for( size_t k = 0; k < AnyOp::blocksize; ++k ) {
							apply( zz[ k ], xx[ k ], yy[ k ], anyOp );
						}

						// perform reduction into output element
						addMonoid.getOperator().foldlArray( reduced, zz, AnyOp::blocksize );
						//^--> note that this foldl operates on raw arrays,
						//     and thus should not be mistaken with a foldl
						//     on a grb::Vector.
					}
				} else {
					while( i + AnyOp::blocksize < end - 1 ) {
						// declare buffers
						static_assert( AnyOp::blocksize > 0,
							"Configuration error: vectorisation blocksize set to "
							"0!" );
						typename AnyOp::D1 xx[ AnyOp::blocksize ];
						typename AnyOp::D2 yy[ AnyOp::blocksize ];
						typename AnyOp::D3 zz[ AnyOp::blocksize ];
						bool mask[ AnyOp::blocksize ];

						// prepare registers
						for( size_t k = 0; k < AnyOp::blocksize; ++k, ++i ) {
							mask[ k ] = coors_q.assigned( coors_r.index( i ) );
						}

						// do masked load
						for( size_t k = 0; k < AnyOp::blocksize; ++k, ++i ) {
							if( mask[ k ] ) {
								xx[ k ] = static_cast< typename AnyOp::D1 >( a[ i ] );
								yy[ k ] = static_cast< typename AnyOp::D2 >( b[ i ] );
							}
						}

						// rewind
						i -= AnyOp::blocksize;

						// perform element-wise multiplication
						for( size_t k = 0; k < AnyOp::blocksize; ++k, ++i ) {
							if( mask[ k ] ) {
								apply( zz[ k ], xx[ k ], yy[ k ], anyOp );
							} else {
								zz[ k ] = addMonoid.template getIdentity< typename AnyOp::D3 >();
							}
						}

						// perform reduction into output element
						addMonoid.getOperator().foldlArray( reduced, zz, AnyOp::blocksize );
						//^--> note that this foldl operates on raw arrays,
						//     and thus should not be mistaken with a foldl
						//     on a grb::Vector.
					}
				}

				// perform element-by-element updates for remainder (if any)
				for( ; i < end - 1; ++i ) {
					OutputType temp;
					const size_t index = coors_r.index( i );
					if( dense || coors_q.assigned( index ) ) {
						apply( temp, a[ index ], b[ index ], anyOp );
						foldr( temp, reduced, addMonoid.getOperator() );
					}
				}

				// write back result
				z = static_cast< OutputType >( reduced );
			}

			// done!
			return SUCCESS;
		}

	} // namespace internal

	/**
	 * Calculates the dot product, \f$ z = (x,y) \f$, under the given semiring.
	 *
	 * @tparam descr      The descriptor to be used (descriptors::no_operation
	 *                    if left unspecified).
	 * @tparam Ring       The semiring type to use.
	 * @tparam OutputType The output type.
	 * @tparam InputType1 The input element type of the left-hand input vector.
	 * @tparam InputType2 The input element type of the right-hand input vector.
	 *
	 * @param[out]  z  The output element \f$ \alpha \f$.
	 * @param[in]   x  The left-hand input vector.
	 * @param[in]   y  The right-hand input vector.
	 * @param[in] ring The semiring to perform the dot-product under. If left
	 *                 undefined, the default constructor of \a Ring will be used.
	 *
	 * @return grb::MISMATCH When the dimensions of \a x and \a y do not match. All
	 *                       input data containers are left untouched if this exit
	 *                       code is returned; it will be as though this call was
	 *                       never made.
	 * @return grb::SUCCESS  On successful completion of this call.
	 *
	 * \parblock
	 * \par Performance semantics
	 *      -# This call takes \f$ \Theta(n/p) \f$ work at each user process, where
	 *         \f$ n \f$ equals the size of the vectors \a x and \a y, and
	 *         \f$ p \f$ is the number of user processes. The constant factor
	 *         depends on the cost of evaluating the addition and multiplication
	 *         operators. A good implementation uses vectorised instructions
	 *         whenever the input domains, output domain, and the operators used
	 *         allow for this.
	 *
	 *      -# This call takes \f$ \mathcal{O}(1) \f$ memory beyond the memory used
	 *         by the application at the point of a call to this function.
	 *
	 *      -# This call incurs at most
	 *         \f$ n( \mathit{sizeof}(\mathit{D1}) + \mathit{sizeof}(\mathit{D2}) ) + \mathcal{O}(p) \f$
	 *         bytes of data movement.
	 *
	 *      -# This call incurs at most \f$ \Theta(\log p) \f$ synchronisations
	 *         between two or more user processes.
	 *
	 *      -# A call to this function does result in any system calls.
	 * \endparblock
	 *
	 * \note This requires an implementation to pre-allocate \f$ \Theta(p) \f$
	 *       memory for inter-process reduction, if the underlying communication
	 *       layer indeed requires such a buffer. This buffer may not be allocated
	 *       (nor freed) during a call to this function.
	 *
	 * \parblock
	 * \par Valid descriptors
	 *   -# grb::descriptors::no_operation
	 *   -# grb::descriptors::no_casting
	 * \endparblock
	 */
	template< Descriptor descr = descriptors::no_operation, class AddMonoid, class AnyOp, typename OutputType, typename InputType1, typename InputType2, typename Coords >
	RC dot( OutputType & z,
		const Vector< InputType1, banshee, Coords > & x,
		const Vector< InputType2, banshee, Coords > & y,
		const AddMonoid & addMonoid = AddMonoid(),
		const AnyOp & anyOp = AnyOp(),
		const typename std::enable_if< ! grb::is_object< OutputType >::value && ! grb::is_object< InputType1 >::value && ! grb::is_object< InputType2 >::value && grb::is_monoid< AddMonoid >::value &&
				grb::is_operator< AnyOp >::value,
			void >::type * const = NULL ) {
		// static sanity checks
		NO_CAST_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< InputType1, typename AnyOp::D1 >::value ), "grb::dot",
			"called with a left-hand vector value type that does not match the "
			"first domain of the given multiplicative operator" );
		NO_CAST_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< InputType2, typename AnyOp::D2 >::value ), "grb::dot",
			"called with a right-hand vector value type that does not match the "
			"second domain of the given multiplicative operator" );
		NO_CAST_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename AddMonoid::D3, typename AnyOp::D1 >::value ), "grb::dot",
			"called with a multiplicative operator output domain that does not "
			"match the first domain of the given additive operator" );
		NO_CAST_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< OutputType, typename AddMonoid::D2 >::value ), "grb::dot",
			"called with an output vector value type that does not match the "
			"second domain of the given additive operator" );
		NO_CAST_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename AddMonoid::D3, typename AddMonoid::D2 >::value ), "grb::dot",
			"called with an additive operator whose output domain does not match "
			"its second input domain" );
		NO_CAST_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< OutputType, typename AddMonoid::D3 >::value ), "grb::dot",
			"called with an output vector value type that does not match the third "
			"domain of the given additive operator" );

		// dynamic sanity check
		const size_t n = internal::getCoordinates( y ).size();
		if( internal::getCoordinates( x ).size() != n ) {
			return MISMATCH;
		}

		// cache nnzs
		const size_t nnzx = internal::getCoordinates( x ).nonzeroes();
		const size_t nnzy = internal::getCoordinates( y ).nonzeroes();

		// catch trivial case
		if( nnzx == 0 && nnzy == 0 ) {
			z = addMonoid.template getIdentity< OutputType >();
			return SUCCESS;
		}

		// if descriptor says nothing about being dense...
		if( ! ( descr & descriptors::dense ) ) {
			// check if inputs are actually dense...
			if( nnzx == n && nnzy == n ) {
				// call implementation with the right descriptor
				return internal::dot_generic< descr | descriptors::dense >( z, x, y, addMonoid, anyOp );
			}
		} else {
			// descriptor says dense, but if any of the vectors are actually sparse...
			if( internal::getCoordinates( x ).nonzeroes() < n || internal::getCoordinates( y ).nonzeroes() < n ) {
				// call implementation with corrected descriptor
				return internal::dot_generic< descr & ~( descriptors::dense ) >( z, x, y, addMonoid, anyOp );
			}
		}

		// all OK, pass to implementation
		return internal::dot_generic< descr >( z, x, y, addMonoid, anyOp );
	}

	/** No implementation notes. */
	template< typename Func, typename DataType, typename Coords >
	RC eWiseMap( const Func f, Vector< DataType, banshee, Coords > & x ) {
		const auto & coors = internal::getCoordinates( x );
		if( coors.isDense() ) {
			// vector is distributed sequentially, so just loop over it
			for( size_t i = 0; i < coors.size(); ++i ) {
				// apply the lambda
				DataType & xval = internal::getRaw( x )[ i ];
				xval = f( xval );
			}
		} else {
			for( size_t k = 0; k < coors.nonzeroes(); ++k ) {
				DataType & xval = internal::getRaw( x )[ coors.index( k ) ];
				xval = f( xval );
			}
		}
		// and done!
		return SUCCESS;
	}

	/**
	 * This is the eWiseLambda that performs length checking by recursion.
	 *
	 * in the banshee implementation all vectors are distributed equally, so no
	 * need to synchronise any data structures. We do need to do error checking
	 * though, to see when to return grb::MISMATCH. That's this function.
	 *
	 * @see Vector::operator[]()
	 * @see Vector::lambda_banshee
	 */
	template< typename Func, typename DataType1, typename DataType2, typename Coords, typename... Args >
	RC eWiseLambda( const Func f, const Vector< DataType1, banshee, Coords > & x, const Vector< DataType2, banshee, Coords > & y, Args const &... args ) {
		// catch mismatch
		if( size( x ) != size( y ) ) {
			return MISMATCH;
		}
		// continue
		return eWiseLambda( f, x, args... );
	}

	/**
	 * No implementation notes. This is the `real' implementation on banshee
	 * vectors.
	 *
	 * @see Vector::operator[]()
	 * @see Vector::lambda_banshee
	 */
	template< typename Func, typename DataType, typename Coords >
	RC eWiseLambda( const Func f, const Vector< DataType, banshee, Coords > & x ) {
#ifdef _DEBUG
		printf( "Info: entering eWiseLambda function on vectors.\n" );
#endif
		const auto & coors = internal::getCoordinates( x );
		if( coors.isDense() ) {
			// vector is distributed sequentially, so just loop over it
			for( size_t i = 0; i < coors.size(); ++i ) {
				// apply the lambda
				f( i );
			}
		} else {
			for( size_t k = 0; k < coors.nonzeroes(); ++k ) {
				const size_t i = coors.index( k );
#ifdef _DEBUG
				printf( "\tprocessing coordinate %d which has index %d\n", (int)k, (int)i );
#endif
				f( i );
			}
		}
		// and done!
		return SUCCESS;
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
	 * @return grb::SUCCESS When the call completed successfully.
	 * @return grb::ILLEGAL If the provided input vector \a y was not dense.
	 * @return grb::ILLEGAL If the provided input vector \a y was empty.
	 *
	 * \parblock
	 * \par Valid descriptors
	 * grb::descriptors::no_operation, grb::descriptors::no_casting,
	 * grb::descriptors::dense
	 *
	 * \note Invalid descriptors will be ignored.
	 *
	 * If grb::descriptors::no_casting is specified, then 1) the first domain of
	 * \a monoid must match \a InputType, 2) the second domain of \a op must match
	 * \a IOType, and 3) the third domain must match \a IOType. If one of
	 * these is not true, the code shall not compile.
	 * \endparblock
	 *
	 * \parblock
	 * \par Performance semantics
	 *      -# This call comprises \f$ \Theta(n) \f$ work, where \f$ n \f$ equals
	 *         the size of the vector \a x. The constant factor depends on the
	 *         cost of evaluating the underlying binary operator. A good
	 *         implementation uses vectorised instructions whenever the input
	 *         domains, the output domain, and the operator used allow for this.
	 *
	 *      -# This call will not result in additional dynamic memory allocations.
	 *         No system calls will be made.
	 *
	 *      -# This call takes \f$ \mathcal{O}(1) \f$ memory beyond the memory
	 *         used by the application at the point of a call to this function.
	 *
	 *      -# This call incurs at most
	 *         \f$ n \mathit{sizeof}(\mathit{InputType}) + \mathcal{O}(1) \f$
	 *         bytes of data movement. If \a y is sparse, a call to this function
	 *         incurs at most \f$ n \mathit{sizeof}( \mathit{bool} ) \f$ extra
	 *         bytes of data movement.
	 * \endparblock
	 *
	 * @see grb::foldl provides similar functionality.
	 */
	template< Descriptor descr = descriptors::no_operation, class Monoid, typename InputType, typename IOType, typename MaskType, typename Coords >
	RC foldl( IOType & x,
		const Vector< InputType, banshee, Coords > & y,
		const Vector< MaskType, banshee, Coords > & mask,
		const Monoid & monoid = Monoid(),
		const typename std::enable_if< ! grb::is_object< IOType >::value && ! grb::is_object< InputType >::value && ! grb::is_object< MaskType >::value && grb::is_monoid< Monoid >::value,
			void >::type * const = NULL ) {
#ifdef _DEBUG
		printf( "foldl: IOType <- [InputType] with a monoid called. Array has "
				"size %d with %d nonzeroes. It has a mask of size %d with %d "
				"nonzeroes.\n ",
			(int)size( y ), (int)nnz( y ), (int)size( mask ), (int)nnz( mask ) );
#endif

		// static sanity checks
		NO_CAST_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< IOType, InputType >::value ), "grb::reduce",
			"called with a scalar IO type that does not match the input vector "
			"type" );
		NO_CAST_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< InputType, typename Monoid::D1 >::value ), "grb::reduce",
			"called with an input vector value type that does not match the first "
			"domain of the given monoid" );
		NO_CAST_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< InputType, typename Monoid::D2 >::value ), "grb::reduce",
			"called with an input vector type that does not match the second "
			"domain of the given monoid" );
		NO_CAST_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< InputType, typename Monoid::D3 >::value ), "grb::reduce",
			"called with an input vector type that does not match the third domain "
			"of the given monoid" );
		NO_CAST_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< bool, MaskType >::value ), "grb::reduce", "called with a vector mask type that is not boolean" );

		// dynamic sanity checks
		if( size( mask ) > 0 && size( mask ) != size( y ) ) {
			return MISMATCH;
		}

		// do minimal work
		RC ret = SUCCESS;
		IOType global_reduced = monoid.template getIdentity< IOType >();

		// check if we have a mask
		const bool masked = internal::getCoordinates( mask ).size() > 0;

		if( masked || internal::getCoordinates( y ).nonzeroes() < internal::getCoordinates( y ).size() ) {
			// sparse case
			for( size_t i = 0; i < internal::getCoordinates( y ).size(); ++i ) {
				// if mask OK and there is a nonzero
				if( ( ! masked || internal::getCoordinates( mask ).template mask< descr >( i, internal::getRaw( mask ) + i ) ) && internal::getCoordinates( y ).assigned( i ) ) {
					// fold it into y
					RC rc = foldl( global_reduced, internal::getRaw( y )[ i ], monoid.getOperator() );
					assert( rc == SUCCESS );
					if( rc != SUCCESS ) {
						ret = rc;
					}
				}
			}
		} else {
			// dense case relies on foldlArray
			monoid.getOperator().foldlArray( global_reduced, internal::getRaw( y ), internal::getCoordinates( y ).nonzeroes() );
		}

		// do accumulation
		if( ret == SUCCESS ) {
#ifdef _DEBUG
			printf( "Accumulating %d into %d using foldl\n", (int)global_reduced, (int)x );
#endif
			ret = foldl( x, global_reduced, monoid.getOperator() );
		}

		// done
		return ret;
	}

	/**
	 * TODO documentation
	 */
	template< Descriptor descr = descriptors::no_operation, typename T, typename U, typename Coords >
	RC zip( Vector< std::pair< T, U >, banshee, Coords > & z,
		const Vector< T, banshee, Coords > & x,
		const Vector< U, banshee, Coords > & y,
		const typename std::enable_if< ! grb::is_object< T >::value && ! grb::is_object< U >::value, void >::type * const = NULL ) {
		const size_t n = size( z );
		if( n != size( x ) ) {
			return MISMATCH;
		}
		if( n != size( y ) ) {
			return MISMATCH;
		}
		if( nnz( x ) < n ) {
			return ILLEGAL;
		}
		if( nnz( y ) < n ) {
			return ILLEGAL;
		}
		auto & z_coors = internal::getCoordinates( z );
		const T * const x_raw = internal::getRaw( x );
		const U * const y_raw = internal::getRaw( y );
		std::pair< T, U > * z_raw = internal::getRaw( z );
		z_coors.assignAll();
		for( size_t i = 0; i < n; ++i ) {
			z_raw[ i ].first = x_raw[ i ];
			z_raw[ i ].second = y_raw[ i ];
		}
		return SUCCESS;
	}

	/**
	 * TODO documentation
	 */
	template< Descriptor descr = descriptors::no_operation, typename T, typename U, typename Coords >
	RC unzip( Vector< T, banshee, Coords > & x,
		Vector< U, banshee, Coords > & y,
		const Vector< std::pair< T, U >, banshee, Coords > & in,
		const typename std::enable_if< ! grb::is_object< T >::value && ! grb::is_object< U >::value, void >::type * const = NULL ) {
		const size_t n = size( in );
		if( n != size( x ) ) {
			return MISMATCH;
		}
		if( n != size( y ) ) {
			return MISMATCH;
		}
		if( nnz( in ) < n ) {
			return ILLEGAL;
		}
		auto & x_coors = internal::getCoordinates( x );
		auto & y_coors = internal::getCoordinates( y );
		T * const x_raw = internal::getRaw( x );
		U * const y_raw = internal::getRaw( y );
		const std::pair< T, U > * in_raw = internal::getRaw( in );
		x_coors.assignAll();
		y_coors.assignAll();
		for( size_t i = 0; i < n; ++i ) {
			x_raw[ i ] = in_raw[ i ].first;
			y_raw[ i ] = in_raw[ i ].second;
		}
		return SUCCESS;
	}

	/** @} */
	//   ^-- ends BLAS-1 module

} // namespace grb

#undef NO_CAST_ASSERT
#undef NO_CAST_OP_ASSERT

#endif // end `_H_GRB_BANSHEE_BLAS1'
