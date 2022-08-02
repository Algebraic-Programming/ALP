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
 * Utilities for handling input iterators.
 *
 * @author Alberto Scolari
 * @date 20/06/2022
 */

#include <cstddef>

#include <graphblas/rc.hpp>
#include <graphblas/type_traits.hpp>

#ifndef _H_GRB_ITERATOR_UTILS
#define _H_GRB_ITERATOR_UTILS


namespace grb {

	namespace utils {

		namespace internal {

			/**
			 * Checks whether the input iterator \a it stores valid row and column
			 * coordinates.
			 *
			 * @tparam IterT the iterator type
			 *
			 * @param[in] it   The given input iterator.
			 * @param[in] rows The number of matrix rows.
			 * @param[in] cols The number of matrix columns.
			 *
			 * @return RC SUCCESS if the iterator's row and column values (i.e.,
			 *                    <tt>it.i()</tt> <tt>it.j()</tt>, respectively) are both
			 *                    within the given matrix boundaries, and MISMATCH
			 *                    otherwise.
			 */
			template< typename IterT >
			inline RC check_input_coordinates(
				const IterT &it,
				const typename IterT::RowIndexType rows,
				const typename IterT::ColumnIndexType cols
			) {
				static_assert( grb::internal::is_input_iterator< void, IterT >::value,
					"IterT is not an input iterator" );
				if( it.i() >= rows ) {
#ifndef NDEBUG
					std::cerr << "Error: " << rows << " x " << cols
					<< " matrix nonzero ingestion encounters row "
					<< "index at " << it.i() << std::endl;
#endif
					return MISMATCH;
				}
				if( it.j() >= cols ) {
#ifndef NDEBUG
					std::cerr << "Error: " << rows << " x " << cols
					<< " matrix nonzero ingestion encounters column "
					<< "input at " << it.j() << std::endl;
#endif
					return MISMATCH;
				}
				return SUCCESS;
			}

		} // end namespace internal

	} // end namespace utils

} // end namespace grb

#endif // end _H_GRB_ITERATOR_UTILS

