
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
 * Provides a mechanism for inspecting properties of various backends
 *
 * @author A. N. Yzelman
 * @date 5th of May 2017
 */

#ifndef _H_GRB_PROPERTIES_BASE
#define _H_GRB_PROPERTIES_BASE

#include <graphblas/backends.hpp>


namespace grb {

	/**
	 * Collection of various properties on the given ALP/GraphBLAS \a backend.
	 *
	 * @tparam backend The backend of which to access its properties.
	 *
	 * The properties collected here are meant to be compile-time constants that
	 * provide insight in what features the given \a backend supports. ALP user
	 * code may rely on the properties specified herein. All ALP backends must
	 * define all properties here specified.
	 *
	 * The default template class shall be empty in order to ensure implementing
	 * backends must specialise this class, while also making sure no backend may
	 * accidentally implicitly and erroneously propagate global defaults.
	 *
	 * \ingroup backends
	 */
	template< enum Backend backend >
	class Properties {

#ifdef __DOXYGEN__

		public:

			/**
			 * Whether a scalar, non-ALP/GraphBLAS object, may be captured by and written
			 * to by a lambda function that is passed to #grb::eWiseLambda.
			 *
			 * Typically, if the \a backend is shared-memory parallel, this function
			 * would return <tt>false</tt>. Purely Single Program, Multiple Data (SPMD)
			 * backends over distributed memory, including simple sequential backends,
			 * would have this property return <tt>true</tt>.
			 *
			 * Notably, hybrid SPMD + OpenMP backends (e.g., #grb::hybrid), are not pure
			 * SPMD and as such would return <tt>false</tt>.
			 *
			 * @see grb::eWiseLambda()
			 */
			static constexpr const bool writableCaptured = true;

			/**
			 * Whether the given \a backend supports blocking execution or is, instead,
			 * non-blocking.
			 *
			 * In blocking execution mode, any ALP/GraphBLAS primitive, when it returns,
			 * is guaranteed to have completed the requested computation.
			 *
			 * If a given \a backend has this property <tt>true</tt> then the
			 * #isNonblockingExecution property must read <tt>false</tt>, and vice versa.
			 */
			static constexpr const bool isBlockingExecution = true;

			/**
			 * Whether the given \a backend is non-blocking or is, instead, blocking.
			 *
			 * In non-blocking execution mode, any ALP/GraphBLAS primitive, on return,
			 * \em may in fact \em not have completed the requested computation.
			 *
			 * Non-blocking execution thus allows for the lazy evaluation of an ALP
			 * code, which, in turn, allows for cross-primitive optimisations to be
			 * automatically applied.
			 *
			 * If a given \a backend has this property <tt>true</tt> then the
			 * #isBlockingExecution property must read <tt>false</tt>, and vice versa.
			 */
			static constexpr const bool isNonblockingExecution = !isBlockingExecution;
#endif
		};

} // namespace grb

#endif // end _H_GRB_PROPERTIES_BASE

