
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
 */

#ifndef _H_GRB_UTILS_CONFIG
#define _H_GRB_UTILS_CONFIG

namespace grb {
	namespace config {
		/** Parser defaults. */
		class PARSER {
		public:
			/** The default buffer size. */
			static constexpr size_t bsize() {
				return ( 1ul << 20 );
			} // 1MB
			/** The read block size. */
			static constexpr size_t read_bsize() {
				return ( 1ul << 17 );
			} // 128kB (SSDs should set this higher(!))
		};
	} // namespace config
} // namespace grb

#endif
