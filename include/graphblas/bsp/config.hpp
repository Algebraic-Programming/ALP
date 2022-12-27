
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
 * @date 22nd of January, 2021
 * @brief Separates the LPF default initialisation parameters from the
 *        backends based on LPF.
 */

#ifndef _H_GRB_LPF_CONFIG
#define _H_GRB_LPF_CONFIG

#include <cstddef>


namespace grb {

	namespace config {

		/**
		 * Lightweight Parallel Foundations defaults.
		 */
		class LPF {

			public:

				/**
				 * Return the default number of memory registrations used by GraphBLAS.
				 */
				static constexpr size_t regs() {
					return 500;
				}

				/**
				 * Return the default maximum h relation expressed in the number of messages
				 * (instead of bytes) used by GraphBLAS.
				 */
				static constexpr size_t maxh() {
					return 200;
				}

		};

	} // namespace config

} // namespace grb

#endif

