
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
 * @author A. Karanasiou
 * @date 3rd of March 2022
 */

#include <graphblas/config.hpp>

namespace grb {

	template<
		Descriptor descr = descriptors::no_operation,
		typename InputType, typename fwd_iterator
	>
	RC buildMatrixUnique( Matrix< InputType, hyperdags > &A,
		fwd_iterator start, const fwd_iterator end,
		const IOMode mode
	) {
		return buildMatrixUnique<descr>( internal::getMatrix(A), start, end, mode );
	}

} // namespace grb

