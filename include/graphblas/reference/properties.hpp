
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
 * @date 5th of May 2017
 */

#if ! defined _H_GRB_REFERENCE_PROPERTIES || defined _H_GRB_REFERENCE_OMP_PROPERTIES
#define _H_GRB_REFERENCE_PROPERTIES

#include <graphblas/base/properties.hpp>

namespace grb {

	/** No implementation notes. */
	template<>
	class Properties< reference > {
	public:
#ifdef _H_GRB_REFERENCE_OMP_PROPERTIES
		/** No implementation notes. */
		constexpr static bool writableCaptured = false;
#else
		/** No implementation notes. */
		constexpr static bool writableCaptured = true;
#endif
	};

} // namespace grb

// parse this unit again for OpenMP support
#ifdef _GRB_WITH_OMP
#ifndef _H_GRB_REFERENCE_OMP_PROPERTIES
#define _H_GRB_REFERENCE_OMP_PROPERTIES
#define reference reference_omp
#include "graphblas/reference/properties.hpp"
#undef reference
#undef _H_GRB_REFERENCE_OMP_PROPERTIES
#endif
#endif

#endif // end `_H_GRB_REFERENCE_PROPERTIES
