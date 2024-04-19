
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

#ifndef _H_GRB_DENSEREF_INIT
#define _H_GRB_DENSEREF_INIT

#include <graphblas/base/init.hpp>

namespace grb {

	/** \internal No-op init */
	template<>
	RC init< reference_dense >( const size_t, const size_t, void * const );

	/** \internal No-op init */
	template<>
	RC finalize< reference_dense >();

} // end namespace ``grb''

#endif // end ``_H_GRB_DENSEREF_INIT''
