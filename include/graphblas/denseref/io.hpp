
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

	template< typename InputType, typename fwd_iterator >
	RC buildMatrix( Matrix< InputType, reference_dense > & A, fwd_iterator start, const fwd_iterator end ) {
		return A.template buildMatrixUnique( start, end );
	}

	template< typename InputType, typename Structure, typename Storage, typename View, typename fwd_iterator >
	RC buildMatrix( StructuredMatrix< InputType, Structure, Storage, View, reference_dense > & A, const fwd_iterator & start, const fwd_iterator & end ) noexcept {
		return A.template buildMatrixUnique( start, end );
	}

} // end namespace ``grb''

#endif // end ``_H_GRB_DENSEREF_IO''

