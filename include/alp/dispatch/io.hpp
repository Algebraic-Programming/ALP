
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

#ifndef _H_ALP_DISPATCH_IO
#define _H_ALP_DISPATCH_IO

#include <alp/base/io.hpp>
#include <alp/vector.hpp>
#include <alp/scalar.hpp>
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
	 * Request the size (dimension) of a given Vector.
	 */
	template<
		typename DataType, typename DataStructure, typename View, typename ImfR, typename ImfC
	>
	size_t size(
		const Vector< DataType, DataStructure, Density::Dense, View, ImfR, ImfC, dispatch > &x
	) noexcept {
		return getLength( x );
	}

	/**
	 * Sets the value of a given scalar \a alpha to be equal to that of
	 * another given scalar \a beta.
	 */
	template<
		Descriptor descr = descriptors::no_operation,
		typename OutputType, typename OutputStructure,
		typename InputType, typename InputStructure
	>
	RC set(
		Scalar< OutputType, OutputStructure, dispatch > &alpha,
		const Scalar< InputType, InputStructure, dispatch > &beta,
		const std::enable_if_t<
			!alp::is_object< InputType >::value &&
			!alp::is_object< OutputType >::value
		> * const = nullptr
	) {
		// static sanity checks
		NO_CAST_ASSERT(
			( !( descr & descriptors::no_casting ) || std::is_same< OutputType, InputType >::value ),
			"alp::set (scalar)",
			"called with a value type that does not match that of the given "
			"scalar"
		);

		if( !internal::getInitialized( beta ) ) {
			internal::setInitialized( alpha, false );
			return SUCCESS;
		}

		// foldl requires left-hand side to be initialized prior to the call
		internal::setInitialized( alpha, true );
		return foldl( alpha, beta, alp::operators::right_assign< OutputType >() );
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
		Matrix< OutputType, OutputStructure, Density::Dense, OutputView, OutputImfR, OutputImfC, dispatch > &C,
		const Matrix< InputType, InputStructure, Density::Dense, InputView, InputImfR, InputImfC, dispatch > &A
	) noexcept {
		static_assert(
			!std::is_same< OutputType, void >::value,
			"alp::set (set to value): cannot have a pattern matrix as output"
		);
#ifdef _DEBUG
		std::cout << "Called alp::set (matrix-to-matrix, dispatch)" << std::endl;
#endif
		// static checks
		NO_CAST_ASSERT(
			( !( descr & descriptors::no_casting ) || std::is_same< InputType, OutputType >::value ),
			"alp::set", "called with non-matching value types"
		);

		static_assert(
			!internal::is_functor_based<
				Matrix< OutputType, OutputStructure, Density::Dense, OutputView, OutputImfR, OutputImfC, dispatch >
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
		Matrix< OutputType, OutputStructure, Density::Dense, OutputView, OutputImfR, OutputImfC, dispatch > &C,
		const Scalar< InputType, InputStructure, dispatch > &val
	) noexcept {

		static_assert(
			!std::is_same< OutputType, void >::value,
			"alp::set (set to matrix): cannot have a pattern matrix as output"
		);
#ifdef _DEBUG
		std::cout << "Called alp::set (matrix-to-value, dispatch)" << std::endl;
#endif
		// static checks
		NO_CAST_ASSERT(
			( !( descr & descriptors::no_casting ) || std::is_same< InputType, OutputType >::value ),
			"alp::set", "called with non-matching value types"
		);

		static_assert(
			!internal::is_functor_based<
				Matrix< OutputType, OutputStructure, Density::Dense, OutputView, OutputImfR, OutputImfC, dispatch >
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
	 * @brief \a buildMatrix version. The semantics of this function equals the one of
	 *        \a buildMatrixUnique for the \a reference backend.
	 * 
	 * @see alp::buildMatrix
	 */
	template< typename InputType, typename Structure, typename View, typename ImfR, typename ImfC, typename fwd_iterator >
	RC buildMatrix(
		Matrix< InputType, Structure, Density::Dense, View, ImfR, ImfC, dispatch > &A,
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
		Vector< InputType, Structure, Density::Dense, View, ImfR, ImfC, dispatch > &v,
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

#endif // end ``_H_ALP_DISPATCH_IO''

