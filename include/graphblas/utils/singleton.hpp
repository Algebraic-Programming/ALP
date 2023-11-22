
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
		 * This class describes a singleton of a given type \a T.
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
		 *
		 * \warning The use of singletons is almost always discouraged.
		 *
		 * \warning In particular, never use this class within template library
		 *          implementations, including ALP!
		 *
		 * \note The recommendation is have this class used only by final, top-level
		 *       application codes (if indeed they must be used). The rationale for
		 *       this is that singleton classes otherwise may by multiple modules of
		 *       an application, without them being aware of each others' use, with
		 *       all kinds of horrendous effects then becoming possible (if not
		 *       likely).
		 *       
		 * \note Indeed, the only current uses of this class are within the test
		 *       suite, in their top-level .cpp files only.
		 */
		template< typename T, size_t key = 0 >
		class Singleton {

			private:

				/**
				 * \internal Make sure no-one can construct an instance of this class.
				 */
				Singleton();


			public:

				/**
				 * @returns The data corresponding to this singleton.
				 *
				 * \warning The user code must typically distinguish between the first use
				 *          of the singleton (which then initialises the data with something
				 *          meaningful), versus subsequent use that uses the initialised
				 *          data. By default, i.e., on the very first initial access to the
				 *          singleton data, the data corresponds to its default-constructed
				 *          state.
				 *
				 * This function is thread-safe, but the underlying data type \a T may of
				 * course have its own ideas on thread-safety. (I.e., the use of singleton
				 * is only truly thread-safe if the subset the interface of \a T it uses is
				 * thread-safe.)
				 */
				static T& getData() {
					static T data;
					return data;
				}

		};

	} // end namespace grb::utils

} // end namespace grb

#endif // end _H_GRB_UTILS_SINGLETON

