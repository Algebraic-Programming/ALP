
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
 * @file
 *
 * Provides the initialisation and finalisation routines for the nonblocking
 * backend.
 *
 * @author Aristeidis Mastoras
 * @date 16th of May, 2022
 */

#ifndef _H_GRB_NONBLOCKING_INIT
#define _H_GRB_NONBLOCKING_INIT

#include <graphblas/base/init.hpp>
#include <graphblas/utils/DMapper.hpp>


namespace grb {

	template<>
	RC init< nonblocking >( const size_t, const size_t, void * const );

	template<>
	RC finalize< nonblocking >();

	namespace internal {

		/**
		 * When <tt>true</tt>, calling a fake nonblocking primitive for a first time
		 * will emit a warning to the standard error stream.
		 */
		bool nonblocking_warn_if_not_native;

	}

} // namespace grb

#endif //``end _H_GRB_NONBLOCKING_INIT''

