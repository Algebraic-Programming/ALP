
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
	 * If the backend supports multiple user processes, the user can invoke this
	 * function with \a P equal to one or higher; if the backend supports only a
	 * single user process, then \a P must equal one.
	 * The value for the user process ID \a s must be larger or equal to zero and
	 * must be strictly smaller than \a P. If \a P > 1, each user process must
	 * call this function collectively, each user process should pass the same
	 * value for \a P, and each user process should pass a unique value for \a s
	 * amongst all \a P collective calls made.
	 *
	 * An implementation may define that additional data is required for a call to
	 * this function to complete successfully. Such data may be passed via the
	 * final argument to this function, \a implementation_data.
	 *
	 * If the implementation does not support multiple user processes, then a
	 * value for \a implementation_data shall not be required. In parcticular, a
	 * call to this function with an empty parameter list shall then be legal
	 * and infer the following default arguments: zero for \a s, one for \a P,
	 * and \a NULL for \a implementation_data. When such an implementation is
	 * requested to initialise multiple user processes, the grb::UNSUPPORTED
	 * error code shall be returned.
	 *
	 * A call to this function must be matched with a call to grb::finalize().
	 * After a successful call to this function, a new call to grb::init() without
	 * first calling grb::finalize() shall incur undefined behaviour. The
	 * construction of GraphBLAS containers without a preceding successful call
	 * to grb::init() will result in invalid GraphBLAS objects. Any valid
	 * GraphBLAS containers will become invalid after a call to grb::finalize().
	 * Any use of GraphBLAS functions on invalid containers will result in
	 * undefined behaviour.
	 *
	 * @tparam backend Which GraphBLAS backend this call to init initialises.
	 *
	 * @param[in] s The ID of this user process.
	 * @param[in] P The total number of user processes.
	 * @param[in] implementation_data Any implementation-defined data structure
	 *                                required for successful completion of this
	 *                                call.
	 *
	 * \note For a pure MPI implementation, for instance, \a implementation_data
	 *       may be a pointer to the MPI communicator corresponding to these user
	 *       processes.
	 *
	 * \note The implementations based on PlatformBSP require direct passing of
	 *       the \a bsp_t corresponding to the BSP context of the user processes;
	 *       this is legal since the PlatformBSP specification defines the
	 *       \a bsp_t type as a void pointer.
	 *
	 * @return SUCCESS     If the initialisation was successful.
	 * @return UNSUPPORTED When the implementation does not support multiple
	 *                     user processes (\a P larger than 1). After a call to
	 *                     this function exits with this error code the library
	 *                     state  shall be as though the call never were made.
	 * @return PANIC       If this function fails, the state of this GraphBLAS
	 *                     implementation becomes undefined.
	 *
	 * \note There is no argument checking. If \a s is larger or equal to \a P,
	 *       undefined behaviour occurs. If \a implementation_data was invalid
	 *       or corrupted, undefined behaviour occurs.
	 *
	 * \par Performance semantics
	 *      None. Implementations are encouraged to specify the complexity of
	 *      their implementation of this function in terms of \a P.
	 *
	 * \note Compared to the GraphBLAS C specification, this function lacks a
	 *       choice whether to execute in `blocking' or `non-blocking' mode.
	 *       All functions in the Huawei GraphBLAS are blocking. A choice
	 *       between blocking and non-blocking execution may be added later.
	 * \note Note that a blocking mode is a valid implementation of a non-
	 *       blocking mode, as specified in the GraphBLAS C API. Therefore
	 *       this specification will still yield a valid implementation of
	 *       the C API when properly wrapped.
	 * \note Non-blocking mode with clear performance semantics are possible via
	 *       carefully designed algorithmic skeletons. This is future work.
	 * \note This specification allows for grb::init() to be called multiple
	 *       times from the same process and the same thread, as long as all the
	 *       above requirements are met at each call. The parameters \a s and
	 *       \a P (and \a implementation_data) may differ each time.
	 * \note This is an extension of the GraphBLAS C API, whom only allow a
	 *       single initialisation and a single matching finalisation.
	 * \note The GraphBLAS C API does not have the notion of user processes. We
	 *       believe this notion is necessary to properly integrate into parallel
	 *       frameworks, and also to affect proper and efficient parallel I/O.
	 */
	template< enum Backend backend = config::default_backend >
	RC init( const size_t s, const size_t P, void * const implementation_data ) {
		(void)s;
		(void)P;
		(void)implementation_data;
		return PANIC;
	}

	/**
	 * Implementations must ensure that initialisation without explicitly given
	 * values regarding user processes etc. should still result in a successful
	 * initialisation in all cases except where it cannot initialise due to
	 * external factors.
	 * A call to this function could, for instance, reduce to a full grb::init()
	 * while using the default parameters 0 for \a s, 1 for \a P, and \a NULL for
	 * \a implementation_data:
	 * \code
	 * return init< backend >( 0, 1, NULL );
	 * \endcode
	 *
	 * @tparam backend The backend implementation to initialise.
	 *
	 * @return SUCCESS     If the initialisation was successful.
	 * @return PANIC       If this function fails, the state of this GraphBLAS
	 *                     implementation becomes undefined.
	 */
	template< enum Backend backend = config::default_backend >
	RC init() {
		return grb::init< backend >( 0, 1, NULL );
	}

	/**
	 * Finalises a graphBLAS context opened by the last call to grb::init().
	 *
	 * This function must be called collectively and must follow a call to
	 * grb::init(). After successful execution of this function, a new call
	 * to grb::init() may be made.
	 * After a call to this function, any graphBLAS objects that remain in scope
	 * are invalid. The only graphBLAS functions on invalid containers which
	 * shall \em not incur undefined behaviour are their destructors.
	 *
	 * \warning Invalid GraphBLAS containers will remain invalid no matter if a
	 *          next call to grb::init() is made.
	 *
	 * @tparam backend Which GraphBLAS backend this call to init initialises.
	 *
	 * @return SUCCESS If the initialisation was successful.
	 * @return PANIC   If this function fails, the state of the GraphBLAS
	 *                 implementation becomes undefined. This means none of its
	 *                 functions should be called during the remainder program
	 *                 execution; in particular this means a new call to
	 *                 grb::init() will not remedy the situaiton.
	 *
	 * \par Performance semantics
	 *      None. Implementations are encouraged to specify the complexity of
	 *      their implementation of this function in terms of the parameter
	 *      \a P the matching call to grb::init() was called with.
	 */
	template< enum Backend backend = config::default_backend >
	RC finalize() {
		return PANIC;
	}

} // namespace grb

#endif // end _H_GRB_INIT_BASE
