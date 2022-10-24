
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

#ifndef _H_ALP_REFERENCE_IO
#define _H_ALP_REFERENCE_IO

#include <alp/base/io.hpp>
#include "matrix.hpp"

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

namespace alp {

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
		Vector< DataType, DataStructure, Density::Dense, View, ImfR, ImfC, reference > &x
	) noexcept {
		throw std::runtime_error( "Needs an implementation" );
		return SUCCESS;
	}

	/**
	 * Resizes the Scalar to have at least the given number of nonzeroes.
	 * The contents of the scalar are not retained.
	 *
	 * Resizing of dense containers is not allowed as the capacity is determined
	 * by the container dimensions and the storage scheme. Therefore, this
	 * function will not change the capacity of the container.
	 *
	 * The resize function for Scalars exist to maintain compatibility with
	 * other containers (i.e., vector and matrix).
	 *
	 * Even though the capacity remains unchanged, the contents of the scalar
	 * are not retained to maintain compatibility with the general specification.
	 * However, the actual memory will not be reallocated. Rather, the scalar
	 * will be marked as uninitialized.
	 *
	 * @param[in] x      The Scalar to be resized.
	 * @param[in] new_nz The number of nonzeroes this vector is to contain.
	 *
	 * @return SUCCESS   If \a new_nz is not larger than 1.
	 *         ILLEGAL   If \a new_nz is larger than 1.
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
	template< typename InputType, typename InputStructure, typename length_type >
	RC resize( Scalar< InputType, InputStructure, reference > &s, const length_type new_nz ) noexcept {
		if( new_nz <= 1 ) {
			setInitialized( s, false );
			return SUCCESS;
		} else {
			return ILLEGAL;
		}
	}

	/**
	 * Resizes the vector to have at least the given number of nonzeroes.
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
	RC resize(
		Vector< InputType, InputStructure, Density::Dense, View, ImfR, ImfC, reference > &x,
		const length_type new_nz
	) noexcept {
		(void) x;
		(void) new_nz;
		// \todo Add implementation.
		// setInitialized( x, false );
		return PANIC;
	}

	/**
	 * Resizes the matrix to have at least the given number of nonzeroes.
	 * The contents of the matrix are not retained.
	 *
	 * Resizing of dense containers is not allowed as the capacity is determined
	 * by the container dimensions and the storage scheme. Therefore, this
	 * function will not change the capacity of the matrix.
	 *
	 * Even though the capacity remains unchanged, the contents of the matrix
	 * are not retained to maintain compatibility with the general specification.
	 * However, the actual memory will not be reallocated. Rather, the matrix
	 * will be marked as uninitialized.
	 *
	 * @param[in] A         The matrix to be resized.
	 * @param[in] nonzeroes The number of nonzeroes this matrix is to contain.
	 *
	 * @return SUCCESS   If \a new_nz is not larger than the current capacity
	 *                   of the matrix.
	 *         ILLEGAL   If \a new_nz is larger than the current capacity of
	 *                   the matrix.
	 *
	 * \parblock
	 * \par Performance semantics.
	 *        -$ This function consitutes \f$ \Theta(1) \f$ work.
	 *        -# This function allocates \f$ \Theta(0) \f$
	 *           bytes of dynamic memory.
	 *        -# This function does not make system calls.
	 * \endparblock
	 */
	template< typename InputType, typename InputStructure, typename InputView, typename InputImfR, typename InputImfC >
	RC resize(
		Matrix< InputType, InputStructure, Density::Dense, InputView, InputImfR, InputImfC, reference > &A,
		const size_t new_nz
	) noexcept {
		(void) A;
		(void) new_nz;
		// \todo Add implementation.
		// setInitialized( A, false );
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
		Vector< DataType, DataStructure, Density::Dense, View, ImfR, ImfC, reference > &x,
		const Scalar< T, ValStructure, reference > val,
		const typename std::enable_if<
			!alp::is_object< DataType >::value &&
			!alp::is_object< T >::value,
		void >::type * const = NULL
	) {
		// static sanity checks
		NO_CAST_ASSERT( ( !( descr & descriptors::no_casting ) || std::is_same< DataType, T >::value ), "alp::set (Vector, unmasked)",
			"called with a value type that does not match that of the given "
			"vector" );

		if( !internal::getInitialized( val ) ) {
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
	template<
		Descriptor descr = descriptors::no_operation,
		typename DataType, typename DataStructure, typename View, typename ImfR, typename ImfC, typename ValStructure,
		typename T
	>
	RC setElement(
		Vector< DataType, DataStructure, Density::Dense, View, ImfR, ImfC, reference > &x,
		const Scalar< T, ValStructure, reference > val,
		const size_t i,
		const typename std::enable_if< !alp::is_object< DataType >::value && !alp::is_object< T >::value, void >::type * const = NULL
	) {
		// static sanity checks
		NO_CAST_ASSERT( ( !( descr & descriptors::no_casting ) || std::is_same< DataType, T >::value ), "alp::set (Vector, at index)",
			"called with a value type that does not match that of the given "
			"Vector" );

		throw std::runtime_error( "Needs an implementation." );

		// done
		return SUCCESS;
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
		Vector< OutputType, OutputStructure, Density::Dense, OutputView, OutputImfR, OutputImfC, reference > &x,
		const Vector< InputType, InputStructure, Density::Dense, InputView, InputImfR, InputImfC, reference > &y
	) {
		// static sanity checks
		NO_CAST_ASSERT(
			( !( descr & descriptors::no_casting ) || std::is_same< OutputType, InputType >::value ), "alp::copy (Vector)", "called with vector parameters whose element data types do not match" );
		constexpr bool out_is_void = std::is_void< OutputType >::value;
		constexpr bool in_is_void = std::is_void< OutputType >::value;
		static_assert( !in_is_void || out_is_void,
			"alp::set (reference, Vector <- Vector, masked): "
			"if input is void, then the output must be also" );
		static_assert( !( descr & descriptors::use_index ) || !out_is_void,
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
	 * Sets all elements of the output matrix to the values of the input matrix.
	 * C = A
	 *
	 * @tparam descr
	 * @tparam OutputType      Data type of the output matrix C
	 * @tparam OutputStructure Structure of the matrix C
	 * @tparam OutputView      View type applied to the matrix C
	 * @tparam InputType       Data type of the scalar a
	 *
	 * @param C    Matrix whose values are to be set
	 * @param A    The input matrix
	 *
	 * @return RC  SUCCESS on the successful execution of the set
	 */
	template< Descriptor descr = descriptors::no_operation,
		typename OutputType, typename OutputStructure, typename OutputView, typename OutputImfR, typename OutputImfC,
		typename InputType, typename InputStructure, typename InputView, typename InputImfR, typename InputImfC
	>
	RC set(
		Matrix< OutputType, OutputStructure, Density::Dense, OutputView, OutputImfR, OutputImfC, reference > &C,
		const Matrix< InputType, InputStructure, Density::Dense, InputView, InputImfR, InputImfC, reference > &A
	) noexcept {
		static_assert(
			!std::is_same< OutputType, void >::value,
			"alp::set (set to value): cannot have a pattern matrix as output"
		);
#ifdef _DEBUG
		std::cout << "Called alp::set (matrix-to-matrix, reference)" << std::endl;
#endif
		// static checks
		NO_CAST_ASSERT(
			( !( descr & descriptors::no_casting ) || std::is_same< InputType, OutputType >::value ),
			"alp::set", "called with non-matching value types"
		);

		static_assert(
			!internal::is_functor_based<
				Matrix< OutputType, OutputStructure, Density::Dense, OutputView, OutputImfR, OutputImfC, reference >
			>::value,
			"alp::set cannot be called with a functor-based matrix as a destination."
		);

		// TODO: Improve this check to account for non-zero structrue (i.e., bands)
		//       and algebraic properties (e.g., symmetry)
		static_assert(
			std::is_same< OutputStructure, InputStructure >::value,
			"alp::set cannot be called for containers with different structures."
		);

		if( ( nrows( C ) != nrows( A ) ) || ( ncols( C ) != ncols( A ) ) ) {
			return MISMATCH;
		}

		if( !internal::getInitialized( A ) ) {
			internal::setInitialized( C, false );
			return SUCCESS;
		}

		internal::setInitialized( C, true );
		return foldl( C, A, alp::operators::right_assign< OutputType >() );
	}

	/**
	 * Sets all elements of the given matrix to the value of the given scalar.
	 * C = val
	 *
	 * @tparam descr
	 * @tparam OutputType      Data type of the output matrix C
	 * @tparam OutputStructure Structure of the matrix C
	 * @tparam OutputView      View type applied to the matrix C
	 * @tparam InputType       Data type of the scalar a
	 *
	 * @param C    Matrix whose values are to be set
	 * @param val  The value to set the elements of the matrix C
	 *
	 * @return RC  SUCCESS on the successful execution of the set
	 */
	template< Descriptor descr = descriptors::no_operation,
		typename OutputType, typename OutputStructure, typename OutputView, typename OutputImfR, typename OutputImfC,
		typename InputType, typename InputStructure
	>
	RC set(
		Matrix< OutputType, OutputStructure, Density::Dense, OutputView, OutputImfR, OutputImfC, reference > &C,
		const Scalar< InputType, InputStructure, reference > &val
	) noexcept {

		static_assert(
			!std::is_same< OutputType, void >::value,
			"alp::set (set to matrix): cannot have a pattern matrix as output"
		);
#ifdef _DEBUG
		std::cout << "Called alp::set (matrix-to-value, reference)" << std::endl;
#endif
		// static checks
		NO_CAST_ASSERT(
			( !( descr & descriptors::no_casting ) || std::is_same< InputType, OutputType >::value ),
			"alp::set", "called with non-matching value types"
		);

		static_assert(
			!internal::is_functor_based<
				Matrix< OutputType, OutputStructure, Density::Dense, OutputView, OutputImfR, OutputImfC, reference >
			>::value,
			"alp::set cannot be called with a functor-based matrix as a destination."
		);

		if( !internal::getInitialized( val ) ) {
			internal::setInitialized( C, false );
			return SUCCESS;
		}

		internal::setInitialized( C, true );
		return foldl( C, val, alp::operators::right_assign< OutputType >() );
	}

	/**
	 * Assigns elements to a matrix from an iterator.
	 *
	 * @tparam InputType      The matrix's element type.
	 * @tparam fwd_iterator   The source iterator type.
	 *
	 * The iterator \a fwd_iterator must be  STL-compatible, may
	 * support the following three public functions:
	 *  -# <tt>S fwd_iterator.i();</tt> which returns the row index of the current
	 *     nonzero;
	 *  -# <tt>S fwd_iterator.j();</tt> which returns the column index of the
	 *     current nonzero;
	 *  -# <tt>V fwd_iterator.v();</tt> which returns the nonzero value of the
	 *     current nonzero.
	 *
	 * It also may provide the following public typedefs:
	 *  -# <tt>fwd_iterator::row_coordinate_type</tt>
	 *  -# <tt>fwd_iterator::column_coordinate_type</tt>
	 *  -# <tt>fwd_iterator::nonzero_value_type</tt>
	 *
	 * @param[in]  _start Iterator pointing to the first element to be added.
	 * @param[in]  _end   Iterator pointing past the last element to be added.
	 * 
	 * @return alp::MISMATCH -# the dimension of the input and output containers
	 *                          do not match.
	 *                       When this error code is returned the state of this
	 *                       container will be as though this function was never
	 *                       called; however, the given forward iterators may
	 *                       have been copied and the copied iterators may have
	 *                       incurred multiple increments and dereferences.
	 * @return alp::SUCCESS  When the function completes successfully.
	 *
	 * \parblock
	 * \par Performance semantics.
	 *        -# A call to this function will use \f$ \Theta(1) \f$ bytes
	 *           of memory beyond the memory in use at the function call entry.
	 *        -# This function will copy the input forward iterator at most
	 *           \em once.
	 *        -# This function moves
	 *           \f$ \Theta(mn) \f$ bytes of data.
	 *        -# This function will likely make system calls.
	 * \endparblock
	 *
	 * \warning This is an expensive function. Use sparingly and only when
	 *          absolutely necessary.
	 *
	 */
	template< typename InputType, typename fwd_iterator >
	RC buildMatrixUnique( internal::Matrix< InputType, reference > &A, fwd_iterator start, const fwd_iterator end ) {
		return A.template buildMatrixUnique( start, end );
	}

	/**
	 * @brief \a buildMatrix version. The semantics of this function equals the one of
	 *        \a buildMatrixUnique for the \a reference backend.
	 * 
	 * @see alp::buildMatrix
	 */
	template< typename InputType, typename fwd_iterator >
	RC buildMatrix( internal::Matrix< InputType, reference > &A, fwd_iterator start, const fwd_iterator end ) {
		return A.template buildMatrixUnique( start, end );
	}


	/**
	 * Assigns elements to a structured matrix from an iterator.
	 *
	 * @tparam MatrixT The structured matrix type.
	 * @tparam fwd_iterator   The source iterator type.
	 *
	 * The iterator \a fwd_iterator must be  STL-compatible, may
	 * support the following three public functions:
	 *  -# <tt>S fwd_iterator.i();</tt> which returns the row index of the current
	 *     nonzero;
	 *  -# <tt>S fwd_iterator.j();</tt> which returns the column index of the
	 *     current nonzero;
	 *  -# <tt>V fwd_iterator.v();</tt> which returns the nonzero value of the
	 *     current nonzero.
	 *
	 * It also may provide the following public typedefs:
	 *  -# <tt>fwd_iterator::row_coordinate_type</tt>
	 *  -# <tt>fwd_iterator::column_coordinate_type</tt>
	 *  -# <tt>fwd_iterator::nonzero_value_type</tt>
	 *
	 * @param[in]  _start Iterator pointing to the first element to be added.
	 * @param[in]  _end   Iterator pointing past the last element to be added.
	 * 
	 * @return alp::MISMATCH -# the dimension of the input and output containers
	 *                          do not match.
	 *                       When this error code is returned the state of this
	 *                       container will be as though this function was never
	 *                       called; however, the given forward iterators may
	 *                       have been copied and the copied iterators may have
	 *                       incurred multiple increments and dereferences.
	 * @return alp::SUCCESS  When the function completes successfully.
	 *
	 * \parblock
	 * \par Performance semantics.
	 *        -# A call to this function will use \f$ \Theta(1) \f$ bytes
	 *           of memory beyond the memory in use at the function call entry.
	 *        -# This function will copy the input forward iterator at most
	 *           \em once.
	 *        -# This function moves
	 *           \f$ \mathcal{O}(mn) \f$ bytes of data.
	 *        -# This function will likely make system calls.
	 * \endparblock
	 *
	 * \warning This is an expensive function. Use sparingly and only when
	 *          absolutely necessary.
	 *
	 */
	template< typename MatrixT, typename fwd_iterator >
	RC buildMatrixUnique( MatrixT &A, const fwd_iterator &start, const fwd_iterator &end ) noexcept {
		(void)A;
		(void)start;
		(void)end;
		return PANIC;
		// return A.template buildMatrixUnique( start, end );
	}

	/**
	 * @brief \a buildMatrix version. The semantics of this function equals the one of
	 *        \a buildMatrixUnique for the \a reference backend.
	 * 
	 * @see alp::buildMatrix
	 */
	template< typename InputType, typename Structure, typename View, typename ImfR, typename ImfC, typename fwd_iterator >
	RC buildMatrix(
		Matrix< InputType, Structure, Density::Dense, View, ImfR, ImfC, reference > &A,
		const fwd_iterator &start,
		const fwd_iterator &end
	) noexcept {
		(void)A;
		(void)start;
		(void)end;

		// Temporarily assuming 1-1 mapping with user container
		internal::setInitialized(A, true);

		InputType * praw, * p;
		
		size_t len = internal::getLength( internal::getContainer( A ) );
		praw = p = internal::getRaw( internal::getContainer( A ) );

		for( fwd_iterator it = start; p < praw + len && it != end; ++it, ++p ) {
			*p = *it;
		}

		return SUCCESS;
	}

	/**
	 * @brief \a buildVector version.
	 *
	 */
	template< typename InputType, typename Structure, typename View, typename ImfR, typename ImfC, typename fwd_iterator >
	RC buildVector(
		Vector< InputType, Structure, Density::Dense, View, ImfR, ImfC, reference > &v,
		const fwd_iterator &start,
		const fwd_iterator &end
	) noexcept {
		// Temporarily assuming 1-1 mapping with user container
		internal::setInitialized(v, true);

		InputType * praw, * p;
		
		size_t len = internal::getLength( internal::getContainer( v ) );
		praw = p = internal::getRaw( internal::getContainer( v ) );

		for( fwd_iterator it = start; p < praw + len && it != end; ++it, ++p ) {
			*p = *it;
		}

		return SUCCESS;
	}

} // end namespace ``alp''

#undef NO_CAST_ASSERT

#endif // end ``_H_ALP_REFERENCE_IO''

