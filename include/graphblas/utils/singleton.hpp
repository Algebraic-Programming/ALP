
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
 * Implements a singleton class that is tuned for use with ALP launchers (but
 * could be useful outside that context too).
 *
 * @author A. N. Yzelman
 * @date 20th of November, 2023
 */

#ifndef _H_GRB_UTILS_SINGLETON
#define _H_GRB_UTILS_SINGLETON

namespace grb {

	namespace utils {

		/**
		 * This class describes a read-only singleton.
		 *
		 * The data corresponding to a singleton may be retrieved via a call to
		 * ::getData. The singleton includes storage.
		 *
		 * If multiple singletons of the same data type \a T are required, then each
		 * such singleton should define a unique \a key.
		 *
		 * @tparam T   The data type of the singleton.
		 * @tparam key The identifier of this singleton.
		 *
		 * The type \a T must be default-constructible.
		 */
		template< typename T, size_t key = 0 >
		class Singleton {

			private:

				/**
				 * \internal Make sure no-one except ::getData can construct an instance.
				 */
				Singleton();


			public:

				/**
				 * Retrieves data from this singleton.
				 *
				 * Throws an error if the singleton was not yet initialised.
				 *
				 * @returns The data with which the singleton was initialised.
				 */
				static T& getData() {
					static T data;
					return data;
				}

		};

	} // end namespace grb::utils

} // end namespace grb

#endif // end _H_GRB_UTILS_SINGLETON

