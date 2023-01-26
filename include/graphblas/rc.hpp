
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
 * Defines the ALP error codes.
 *
 * @author A. N. Yzelman
 * @date 9--11 August, 2016
 */

#ifndef _H_GRB_RC
#define _H_GRB_RC

#include <string>


namespace grb {

	/**
	 * Return codes of ALP primitives.
	 *
	 * All primitives that are not \em getters return one of the codes defined
	 * here. All primitives may return #SUCCESS, and all primitives may return
	 * #PANIC. All other error codes are optional-- please see the description of
	 * each primitive which other error codes may be valid.
	 *
	 * For core ALP primitives, any non-SUCCESS and non-PANIC error code shall have
	 * no side effects; if a call fails, it shall be as though the call was never
	 * made.
	 */
	enum RC {

		/**
		 * Indicates the primitive has executed successfully.
		 *
		 * All primitives may return this error code.
		 */
		SUCCESS = 0,

		/**
		 * Generic fatal error code. Signals that ALP has entered an undefined state.
		 *
		 * Users can only do their best to exit their application gracefully once
		 * PANIC has been encountered.
		 *
		 * An implementation (backend) is encouraged to write clear error messages to
		 * stderr prior to returning this error code.
		 *
		 * All primitives may return this error code even if not explicitly
		 * documented.
		 */
		PANIC,

		/**
		 * Signals an out-of-memory error while executing the requested primitive.
		 *
		 * User can mitigate by freeing memory and retrying the call or by reducing
		 * the amount of memory required by this call.
		 *
		 * This error code may only be returned when explicitly documented as such.
		 */
		OUTOFMEM,

		/**
		 * One or more of the ALP/GraphBLAS objects passed to the primitive that
		 * returned this error have mismatching dimensions.
		 *
		 * User can mitigate by reissuing with correct parameters. It is usually not
		 * possible to mitigate at run-time; more often than not, this error signals
		 * a logical programming error.
		 *
		 * This error code may only be returned when explicitly documented as such.
		 */
		MISMATCH,

		/**
		 * One or more of the GraphBLAS objects corresponding to the call returning
		 * this error refer to the same object while this explicitly is forbidden.
		 *
		 * \deprecated This error code will be replaced with #ILLEGAL.
		 *
		 * User can mitigate by reissuing with correct parameters. It is usually not
		 * possible to mitigate at run-time; more often than not, this error signals
		 * a logical programming error.
		 *
		 * This error code may only be returned when explicitly documented as such,
		 * but note the deprecation message-- any uses of #OVERLAP will be replaced
		 * with #ILLEGAL before v1.0 is released.
		 */
		OVERLAP,

		/**
		 * Indicates that execution of the requested primitive with the given
		 * arguments would result in overflow.
		 *
		 * Users can mitigate by modifying the offending call. It is usually not
		 * possible to mitigate at run-time; more often than not, this error signals
		 * the underlying problem is too large to handle with whatever current
		 * resources have been assigned to ALP.
		 *
		 * This error code may only be returned when explicitly documented as such.
		 */
		OVERFLW,

		/**
		 * Indicates that the execution of the requested primitive with the given
		 * arguments is not supported by the selected backend.
		 *
		 * This error code should never be returned by a fully compliant backend.
		 *
		 * If encountered, the end-user may mitigate by selecting a different backend.
		 */
		UNSUPPORTED,

		/**
		 * A call to a primitive has determined that one of its arguments was
		 * illegal as per the specification of the primitive.
		 *
		 * User can mitigate by reissuing with correct parameters. It is usually not
		 * possible to mitigate at run-time; more often than not, this error signals
		 * a logical programming error.
		 *
		 * This error code may only be returned when explicitly documented as such;
		 * in other words, the specification precisely determines which (combinations
		 * of) inputs are illegal.
		 */
		ILLEGAL,

		/**
		 * Indicates when one of the #grb::algorithms has failed to achieve its
		 * intended result, for instance, when an iterative method failed to
		 * converged within its alloted resources.
		 *
		 * This error code may only be returned when explicitly documented as such,
		 * and may never be returned by core ALP primitives-- it is reserved for
		 * use by algorithms only.
		 */
		FAILED

	};

	/** @returns A string describing the given error code. */
	std::string toString( const RC code );

} // namespace grb

#endif

