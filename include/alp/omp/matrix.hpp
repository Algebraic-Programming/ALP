
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

#ifndef _H_ALP_OMP_MATRIX
#define _H_ALP_OMP_MATRIX

#include <alp/backends.hpp>
#include <alp/base/matrix.hpp>
#include <alp/amf-based/matrix.hpp>
#include "storagebasedmatrix.hpp"

namespace alp {

	// Currently no backend specific implementation

	namespace internal {

		/** Specialization for diagonal view over Square matrix */
		template<
			enum view::Views target_view = view::original,
			typename SourceMatrix,
			std::enable_if_t<
				is_matrix< SourceMatrix >::value
			> * = nullptr
		>
		typename SourceMatrix::template view_type< view::diagonal >::type
		get_view( SourceMatrix &source, const size_t tr, const size_t tc, const size_t rt, const size_t br, const size_t bc ) {
			size_t block_id = getAmf( source ).getBlockId( tr, tc, rt, br, bc );

			return target_t( source );
		}

	} // namespace internal

} // namespace alp

#endif // end ``_H_ALP_OMP_MATRIX''
