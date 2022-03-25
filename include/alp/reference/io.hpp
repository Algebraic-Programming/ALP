
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

#ifndef _H_GRB_DENSEREF_IO
#define _H_GRB_DENSEREF_IO

#include <graphblas/base/io.hpp>
#include "matrix.hpp"

namespace grb {

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
	 * @return grb::MISMATCH -# the dimension of the input and output containers
	 *                          do not match.
	 *                       When this error code is returned the state of this
	 *                       container will be as though this function was never
	 *                       called; however, the given forward iterators may
	 *                       have been copied and the copied iterators may have
	 *                       incurred multiple increments and dereferences.
	 * @return grb::SUCCESS  When the function completes successfully.
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
	RC buildMatrixUnique( Matrix< InputType, reference_dense > & A, fwd_iterator start, const fwd_iterator end ) {
		return A.template buildMatrixUnique( start, end );
	}

	/**
	 * @brief \a buildMatrix version. The semantics of this function equals the one of
	 *        \a buildMatrixUnique for the \a reference_dense backend.
	 * 
	 * @see grb::buildMatrix
	 */
	template< typename InputType, typename fwd_iterator >
	RC buildMatrix( Matrix< InputType, reference_dense > & A, fwd_iterator start, const fwd_iterator end ) {
		return A.template buildMatrixUnique( start, end );
	}


	/**
	 * Assigns elements to a structured matrix from an iterator.
	 *
	 * @tparam StructuredMatrixT The structured matrix type.
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
	 * @return grb::MISMATCH -# the dimension of the input and output containers
	 *                          do not match.
	 *                       When this error code is returned the state of this
	 *                       container will be as though this function was never
	 *                       called; however, the given forward iterators may
	 *                       have been copied and the copied iterators may have
	 *                       incurred multiple increments and dereferences.
	 * @return grb::SUCCESS  When the function completes successfully.
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
	template< typename StructuredMatrixT, typename fwd_iterator >
	RC buildMatrixUnique( StructuredMatrixT & A, const fwd_iterator & start, const fwd_iterator & end ) noexcept {
		(void)A;
		(void)start;
		(void)end;
		return PANIC;
		// return A.template buildMatrixUnique( start, end );
	}

	/**
	 * @brief \a buildMatrix version. The semantics of this function equals the one of
	 *        \a buildMatrixUnique for the \a reference_dense backend.
	 * 
	 * @see grb::buildMatrix
	 */
	template< typename InputType, typename Structure, typename View, typename fwd_iterator >
	RC buildMatrix( StructuredMatrix< InputType, Structure, Density::Dense, View, reference_dense > & A, const fwd_iterator & start, const fwd_iterator & end ) noexcept {
		(void)A;
		(void)start;
		(void)end;
		return PANIC;
		// return A.template buildMatrixUnique( start, end );
	}

} // end namespace ``grb''

#endif // end ``_H_GRB_DENSEREF_IO''

