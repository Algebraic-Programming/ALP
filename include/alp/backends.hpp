
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
 * @author: A. N. Yzelman
 * @date 21st of December, 2016
 *
 * @file This file contains a register of all backends that are either
 *       implemented, under implementation, or were at any point in time
 *       conceived and noteworthy enough to be recorded for future
 *       consideration to implement. It does so via the alp::Backend
 *       enum.
 */

#ifndef _H_ALP_BACKENDS
#define _H_ALP_BACKENDS

namespace alp {

	/**
	 * This enum collects all implemented backends. Depending on compile flags,
	 * some of these options may be disabled.
	 */
	enum Backend {

		/**
		 * The ALP reference backend.
		 */
		reference,

		/**
		 * The ALP OpenMP backend.
		 */
		omp,

	};

} // namespace alp

#endif

