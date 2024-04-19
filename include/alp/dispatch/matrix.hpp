
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

#ifndef _H_ALP_DISPATCH_MATRIX
#define _H_ALP_DISPATCH_MATRIX

#include <alp/backends.hpp>
#include <alp/base/matrix.hpp>
#include <alp/amf-based/matrix.hpp>
#include <alp/config.hpp>
#include <alp/storage.hpp>
#include <alp/structures.hpp>
#include <alp/imf.hpp>

namespace alp {

	namespace internal {

		/**
		 * Returns the pointer to the element corresponding to element (0,0)
		 * of the provided matrix.
		 *
		 * \note Gets the raw pointer to the underlying container of the
		 *       original matrix and adds the offset to the first element
		 *       defined by the matrix view, using getStorageIndex function.
		 *
		 * @tparam MatrixType  Type of the given ALP matrix
		 *
		 * @param[in] A        The ALP matrix
		 *
		 * @returns Pointer of type MatrixType::value_type (a.k.a T)
		 *
		 */
		template<
			typename MatrixType,
			std::enable_if_t< alp::is_matrix< MatrixType >::value > * = nullptr
		>
		typename MatrixType::value_type *getRawPointerToFirstElement( MatrixType &A ) {
			return &( internal::access( A, internal::getStorageIndex( A, 0, 0 ) ) );
		}

		/** const variant */
		template<
			typename MatrixType,
			std::enable_if_t< alp::is_matrix< MatrixType >::value > * = nullptr
		>
		const typename MatrixType::value_type *getRawPointerToFirstElement( const MatrixType &A ) {
			return &( internal::access( A, internal::getStorageIndex( A, 0, 0 ) ) );
		}

		/**
		 * Returns the leading dimension corresponding to the underlying
		 * container of the provided matrix.
		 *
		 * @tparam MatrixType  Type of the given ALP matrix
		 *
		 * @param[in] A        The ALP matrix
		 *
		 * @returns Leading dimension.
		 *
		 */
		template<
			typename MatrixType,
			std::enable_if_t< alp::is_matrix< MatrixType >::value > * = nullptr
		>
		size_t getLeadingDimension( const MatrixType &A ) {
			// Get the distance between two elements in two consecutive rows
			size_t row_diff = internal::getStorageIndex( A, 1, 0 ) - internal::getStorageIndex( A, 0, 0 );
			// Get the distance between two elements in two consecutive columns
			size_t col_diff = internal::getStorageIndex( A, 0, 1 ) - internal::getStorageIndex( A, 0, 0 );
			// For row-wise storage, row_diff is LDA and col_diff must be 1.
			// For col-wise storage, col_diff is LDA and row_diff must be 1.
			// In other words, one of row_diff/col_diff must be one and the other is LDA.
			if( ( row_diff > 1 ) && ( col_diff > 1 ) ) {
				std::cout << "getLeadingDimension: it seems that the container uses stride > 1 for minor dimension. "
					<< "This is not supported by BLAS.\n";
			}
			if( row_diff > 1 ) {
				return row_diff;
			} else {
				return col_diff;
			}
		}

	} // namespace internal

} // namespace alp

#endif // end ``_H_ALP_DISPATCH_MATRIX''
