
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

#ifndef _H_GRB_PROPERTIES_BASE
#define _H_GRB_PROPERTIES_BASE

#include <graphblas/backends.hpp>

namespace grb {

	/**
	 * Collection of various properties on the given GraphBLAS backend.
	 *
	 * @tparam implementation The implementation of which to access its properties.
	 *
	 * The properties collected here are meant to be compile-time constants that
	 * provide insight in what features the back-end supports.
	 */
	template< enum Backend implementation >
	class Properties {

	public:
		/**
		 * Whether a non-GraphBLAS object captured by a lambda-function and passed to
		 * grb::eWiseLambda can be written to.
		 *
		 * If the implementation backend is fully Single Program, Multiple Data
		 * (SPMD), then this is expected to be legal and result in user-process local
		 * updates. This function would thus return \a true.
		 *
		 * If the implementaiton backend is parallel but supports only a single user
		 * processes, i.e., for a \em data-centric backend, writing to a shared
		 * object results in race conditions and thus is technically impossible. This
		 * function would thus return \a false.
		 *
		 * @return A boolean \a true if and only if capturing a non-GraphBLAS object
		 *         inside a lambda-function for write access, and passing it to
		 *         grb::eWiseLambda would yield valid user process local results. If
		 *         not, \a false is returned instead.
		 *
		 * @see grb::eWiseLambda()
		 */
		static constexpr bool writableCaptured = true;
	};

} // namespace grb

#endif // end _H_GRB_PROPERTIES_BASE
