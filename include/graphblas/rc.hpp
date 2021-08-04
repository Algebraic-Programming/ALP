
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
 * Defines all possible GraphBLAS error codes.
 *
 * @author A. N. Yzelman
 * @date 9--11 August, 2016
 */

#ifndef _H_GRB_RC
#define _H_GRB_RC

#include <string>

namespace grb {

	/**
	 * Return codes of public functions.
	 */
	enum RC {

		/**
		 * Default success code.
		 *
		 * All GraphBLAS functions may return this error code even if not explicitly
		 * documented. Any non-SUCCESS error code shall have no side effects; if a
		 * call fails, it shall be as though the call was never made. The only
		 * exception is #grb::PANIC.
		 */
		SUCCESS = 0,

		/**
		 * Generic fatal error code.
		 *
		 * Signals an illegal state of all GraphBLAS objects connected to the call
		 * returning this error. Users can only exit gracefully when encoutering
		 * errors of this type-- after a GraphBLAS function returns this error
		 * code, the state of the library becomes undefined.
		 *
		 * An implementation is encouraged to write clear error messages to stderr
		 * prior to returning this error code.
		 *
		 * Rationale: instead of using <tt>assert</tt> within GraphBLAS
		 * implementations which would crash the entire application, implementations
		 * should instead simply return #grb::PANIC and let the GraphBLAS user shut
		 * down his or her application as gracefully as possible.
		 *
		 * All GraphBLAS functions may return this error code even if not explicitly
		 * documented.
		 */
		PANIC,

		/**
		 * Out of memory error code.
		 *
		 * User can mitigate by freeing memory and retrying the call or by reducing
		 * the amount of memory required by this call.
		 *
		 * This error code may only be returned when explicitly documented as such.
		 */
		OUTOFMEM,

		/**
		 * One or more of the GraphBLAS objects corresponding to the call returning
		 * this error have mismatching dimensions.
		 *
		 * User can mitigate by reissuing with correct parameters. It is usually not
		 * possible to mitigate at run-time; usually this signals a logic programming
		 * error.
		 *
		 * This error code may only be returned when explicitly documented as such.
		 */
		MISMATCH,

		/**
		 * One or more of the GraphBLAS objects corresponding to the call returning
		 * this error refer to the same object while this is forbidden.
		 *
		 * User can mitigate by reissuing with correct parameters. It is usually not
		 * possible to mitigate at run-time; usually this signals a logic programming
		 * error. Implementations are not required to return this error code and may
		 * incur undefined behaviour instead.
		 *
		 * This error code may only be returned when explicitly documented as such.
		 */
		OVERLAP,

		/**
		 * One or more output parameters would overflow on this function call.
		 *
		 * Users can mitigate by supplying a larger integral types.
		 *
		 * This error code may only be returned when explicitly documented as such.
		 */
		OVERFLW,

		/**
		 * A bsp::init() assuming multiple user processes while this is not supported
		 * by the chosen implementation backend will reduce this error code.
		 *
		 * @see config::default_backend for a description of how the current backend
		 *                              is selected (if not explicitly).
		 *
		 * This error code may only be returned when explicitly documented as such.
		 */
		UNSUPPORTED,

		/**
		 * A call to a GraphBLAS function with an illegal parameter value might
		 * return this error code. When returned, no undefined behaviour will occur
		 * as a result of having passed the illegal argument.
		 *
		 * This error code may only be returned when explicitly documented as such.
		 */
		ILLEGAL,

		/**
		 * Indicates when one of the grb::algorithms has failed to achieve its
		 * intended result, for instance, when an iterative method failed to
		 * converged within its alloted resources.
		 *
		 * This error code may only be returned when explicitly documented as such.
		 */
		FAILED

	};

	/** @returns A string describing the given error code. */
	std::string toString( const RC code );

} // namespace grb

#endif
