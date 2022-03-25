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
 *
 * @file 
 * 
 * This file registers the enum that allows a user to specify the density of a
 * given ALP container.
 * 
 */

#ifndef _H_GRB_DENSITY
#define _H_GRB_DENSITY

#include <tuple>
#include <type_traits>

namespace grb {

	/**
	 * Specifies whether an ALP container is dense or sparse.
	 * 
	 * This is specified by the user and may be used by a backend to drive
	 * a choice of a storage scheme.
	 *
	 */
	enum Density {
		/**
		 * Dense containers do not allow nonzero elements.
		 *
		 * Depending on the container's \a Structure, the backend may decide to
		 * not store all the elements. For example, an upper triangular matrix
		 * can be stored without the all-zero part below the diagonal.
		 * 
		 * @see Structure
		 * 
		 */
		Dense,
		/**
		 * Sparse containers mostly having nonzero elements.
		 *
		 * The backend can decide which specific format to use.
		 *
		 */
		Sparse
	}; // enum Density

} // namespace grb

#endif // _H_GRB_DENSITY
