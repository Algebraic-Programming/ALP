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
 *
 * @file 
 * 
 * This file registers matrix storage schemes that are either
 * implemented, under implementation, or were at any point in time
 * conceived and noteworthy enough to be recorded for future consideration.
 * 
 */

#ifndef _H_GRB_STORAGE
#define _H_GRB_STORAGE

#include <tuple>
#include <type_traits>

namespace grb {

	namespace storage {

		/**
		 * WIP.
		 * Collection of possible storage scheme options for dense matrices.
		 *
		 * \note Although the list for now includes classic BLAS/LAPACK storage schemes we may decide to extend or replace 
		 * 		 this list with different schemes. A user should not make any assumptions on which specific scheme is 
		 * 		 selected internally by a backend to to store a given structure.
		 * 
		 * \note This labelled formulation will be replaced by a more flexible description of the underlying mappings 
		 * 		 associated to each scheme in the spirit of the index mapping functions formualation.
		 * 
		 * @see \a imf.hpp
		 */
		enum Dense {

			/**
			 * Conventional storage in a 2D array. The matrix element \f$A(i,j)\f$ is stored in array element \f$a(i,j)\f$.
			 * Although some non-general structured matrices may forbid access to part of the array,
			 * with this storage option a full rectangular array must be allocated.
			 * This option could also be used as default/initial choice when a storage scheme decision has not yet been made.
			 */
			full,

			/**
			 * Compact 2D storage for Band matrices. An \f$m-\times-n\f$ band matrix with \f$kl\f$ subdiagonals
			 * and \f$ku\f$ superdiagonals may be stored compactly in a 2D array with \f$m\f$ rows and \f$kl+ku+1\f$ columns.
			 * Rows of the matrix are stored in corresponding rows of the array, and diagonals of the matrix are stored
			 * in columns of the array.
			 * This storage scheme should be used in practice only if \f$kl, ku \ll \min(m,n)\f$, although it should work correctly
			 * for all values of \f$kl\f$ and \f$ku\f$.
			 */
			band,

			/**
			 * A tridiagonal matrix of order \f$n\f$ is stored in three 1D arrays, one of length \f$n\f$
			 * containing the diagonal elements, and two of length \f$n-1\f$ containing the subdiagonal
			 * and superdiagonal elements.
			 * Symmetric tridiagonal and bidiagonal matrices are stored in two 1D arrays, one of length \f$n\f$
			 * containing the diagonal elements, and one of length \f$n-1\f$ containing the off-diagonal elements.
			 * A diagonal matrix is stored as a 1D array of length \f$n\f$.
			 * Symmetric, Hermitian or triangular matrices store the relevant triangle packed by rows in a 1D array:
			 * \li \c AlignLeft \f$A(i,j)\f$ is stored in \f$a( j + i*(i + 1)/2 )\f$ for \f$i \leq j\f$
			 * \li \c AlignLeft \f$A(i,j)\f$ is stored in \f$a( j + i*(2*n - i - 1)/2 )\f$ for \f$j \leq i\f$
			 */
			array1d
		}; // Dense

		// enum Sparse {
		//     ...
		// };

	} // namespace storage

} // namespace grb

#endif // _H_GRB_STORAGE
