
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
 * Specifies the #grb::init and #grb::finalize functionalities.
 *
 * @author A. N. Yzelman
 * @date 24th of January, 2017
 */

#ifndef _H_GRB_INIT_BASE
#define _H_GRB_INIT_BASE

#include <graphblas/rc.hpp>

#include "config.hpp"


namespace grb {

	/**
	 * Initialises the calling user process.
	 *
	 * \deprecated Please use #grb::Launcher instead. This primitive will be
	 *             removed from verson 1.0 onwards.
	 *
	 * @tparam backend Which GraphBLAS backend this call to init initialises.
	 *
	 * By default, the backend that is selected by the user at compile-time is
	 * used. If no backend was selected, #grb::reference is assumed.
	 *
	 * @param[in] s The ID of this user process.
	 * @param[in] P The total number of user processes.
	 *
	 * If the backend supports multiple user processes, the user can invoke this
	 * function with \a P equal to one or higher; if the backend supports only a
	 * single user process, then \a P must equal one.
	 *
	 * The value for the user process ID \a s must be larger or equal to zero and
	 * must be strictly smaller than \a P. If \a P > 1, each user process must
	 * call this function collectively, each user process should pass the same
	 * value for \a P, and each user process should pass a unique value for \a s
	 * amongst all \a P collective calls made.
	 *
	 * @param[in] implementation_data Any implementation-defined data structure
	 *                                required for successful completion of this
	 *                                call.
	 *
	 * An implementation may define that additional data is required for a call to
	 * this function to complete successfully. Such data may be passed via the
	 * final argument to this function, \a implementation_data.
	 *
	 * If the implementation does not support multiple user processes, then a
	 * value for \a implementation_data shall not be required. In particular, a
	 * call to this function with an empty parameter list shall then be legal
	 * and infer the following default arguments: zero for \a s, one for \a P,
	 * and \a NULL for \a implementation_data. When such an implementation is
	 * requested to initialise multiple user processes, then #grb::UNSUPPORTED
	 * shall be returned.
	 *
	 * A call to this function must be matched with a call to #grb::finalize.
	 * After a successful call to this function, a new call to #grb::init without
	 * first calling #grb::finalize shall incur undefined behaviour. The
	 * construction of ALP/GraphBLAS containers without a preceding successful call
	 * to #grb::init will result in undefined behaviour. Any valid GraphBLAS
	 * containers will become invalid after a call to #grb::finalize.
	 *
	 * \internal
	 * \note For a pure MPI implementation, for instance, \a implementation_data
	 *       may be a pointer to the MPI communicator corresponding to these user
	 *       processes.
	 *
	 * \note The implementations based on LPF require direct passing of the
	 *       \a lpf_t corresponding to the BSP context of the user processes;
	 *       this is legal since the LPF defines the \a lpf_t type as a void
	 *       pointer.
	 * \endinternal
	 *
	 * @return SUCCESS     If the initialisation was successful.
	 * @return UNSUPPORTED When the implementation does not support multiple
	 *                     user processes while the given \a P was larger than 1.
	 * @return PANIC       If returned, the state of the ALP library becomes
	 *                     undefined.
	 *
	 * After a call to this function that exits with a non-SUCCESS and non-PANIC
	 * error code, the program shall behave as though the call were never made.
	 *
	 * \note There is no argument checking. If \a s is larger or equal to \a P,
	 *       undefined behaviour occurs. If \a implementation_data was invalid
	 *       or corrupted, undefined behaviour occurs.
	 *
	 * \internal
	 * \todo Define #grb::ILLEGAL to be returned if \f$ s \geq P \f$.
	 * \endinternal
	 *
	 * \par Performance semantics
	 *      Implementations and backends must specify the complexity of this
	 *      function in terms of \a P.
	 *
	 * \note Compared to the GraphBLAS C specification, this function lacks a
	 *       choice whether to execute in `blocking' or `non-blocking' mode. With
	 *       ALP, the selected backend controls whether execution proceeds in a
	 *       non-blocking manner or not. Thus selecting a blocking backend for
	 *       compilation results in the application of blocking semantics, while
	 *       selecting a non-blocking backend results in the application of non-
	 *       blocking semantics.
	 * \note Note that in the GraphBLAS C specification, a blocking mode is a
	 *       valid implementation of a non-blocking mode. Therefore, this
	 *       specification will still yield a valid C API implementation when
	 *       properly wrapping around a blocking ALP/GraphBLAS backend.
	 * \note This specification allows for #grb::init to be called multiple times
	 *       from the same process and the same thread. The parameters \a s and
	 *       \a P (and \a implementation_data) may differ each time. Each
	 *       (repeated) call must of course continue to meet all the above
	 *       requirements.
	 * \note The GraphBLAS C API does not have the notion of user processes. We
	 *       believe this notion is necessary to properly integrate into parallel
	 *       frameworks, and also to affect proper and efficient parallel I/O.
	 *
	 * \warning This primitive has been deprecated since version 0.5. Please update
	 *          your code to use the #grb::Launcher instead.
	 *
	 * \internal The implementation will be retained after deprecation has been
	 *           pushed through, as the #grb::Launcher depends on it. However, the
	 *           #grb::init and #grb::finalize must then be moved into the
	 *           #grb::internal namespace.
	 */
	template< enum Backend backend = config::default_backend >
	RC init( const size_t s, const size_t P, void * const implementation_data ) {
		(void) s;
		(void) P;
		(void) implementation_data;
		return PANIC;
	}

	/**
	 * Initialises the calling user process.
	 *
	 * \deprecated Please use #grb::Launcher instead. This primitive will be
	 *             removed from verson 1.0 onwards.
	 *
	 * This variant takes no input arguments. It will assume a single user process
	 * exists; i.e., the call is equivalent to one to #grb::init with \a s zero
	 * and \a P one (and \a implementation_data <tt>NULL</tt>).
	 *
	 * @tparam backend The backend implementation to initialise.
	 *
	 * @return SUCCESS     If the initialisation was successful.
	 * @return PANIC       If returned, the state of the ALP library becomes
	 *                     undefined.
	 *
	 * \warning This primitive has been deprecated since version 0.5. Please update
	 *          your code to use the #grb::Launcher instead.
	 */
	template< enum Backend backend = config::default_backend >
	RC init() {
		return grb::init< backend >( 0, 1, nullptr );
	}

	/**
	 * Finalises an ALP/GraphBLAS context opened by the last call to #grb::init.
	 *
	 * \deprecated Please use #grb::Launcher instead. This primitive will be
	 *             removed from verson 1.0 onwards.
	 *
	 * This function must be called collectively and must follow a call to
	 * #grb::init. After successful execution of this function, a new call to
	 * #grb::init may be made. (This function is re-entrant.)
	 *
	 * After a call to this function, any ALP/GraphBLAS objects that remain in
	 * scope become invalid.
	 *
	 * \warning Invalid ALP/GraphBLAS containers will remain invalid no matter if a
	 *          next call to #grb::init is made.
	 *
	 * @tparam backend Which ALP/GraphBLAS backend to finalise.
	 *
	 * @return SUCCESS If finalisation was successful.
	 * @return PANIC   If this function fails, the state of the ALP/GraphBLAS
	 *                 implementation becomes undefined. This means none of its
	 *                 functions should be called during the remainder program
	 *                 execution; in particular this means a new call to
	 *                 #grb::init will not remedy the situation.
	 *
	 * \par Performance semantics
	 *      None. Implementations are encouraged to specify the complexity of
	 *      their implementation of this function in terms of the parameter
	 *      \a P the matching call to #grb::init was called with.
	 *
	 * \warning This primitive has been deprecated since version 0.5. Please update
	 *          your code to use the #grb::Launcher instead.
	 */
	template< enum Backend backend = config::default_backend >
	RC finalize() {
		return PANIC;
	}

} // namespace grb

#endif // end _H_GRB_INIT_BASE

