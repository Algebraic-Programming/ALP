
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
 * Implements the grb::wait for the nonblocking backend.
 *
 * @author Aristeidis Mastoras
 * @date 16th of May, 2022
 */

#include <graphblas/nonblocking/io.hpp>
#include <graphblas/nonblocking/lazy_evaluation.hpp>


namespace grb {

	namespace internal {

		extern LazyEvaluation le;

	}

	/**
	 * \internal This is a nonblocking implementation, and all
	 * pending operations must be completed.
	 */
	template<>
	RC wait< nonblocking >() {

		return internal::le.execution();
	}

}

