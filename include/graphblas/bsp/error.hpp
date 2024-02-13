
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
 * Provides some helper routines that translates LPF errors to GraphBLAS errors.
 *
 * @author A. N. Yzelman
 * @date 13th of February, 2024
 */

#ifndef _H_GRB_BSP_ERROR
#define _H_GRB_BSP_ERROR

namespace grb::internal {

	namespace {
		/**
		 * This function assumes lpf_rc is materialised from calls to lpf_sync,
		 * lpf_register_*, lpf_deregister, LPF collectives, lpf_get, and/or lpf_put.
		 *
		 * @param[in] lpf_rc The resulting LPF error code.
		 *
		 * As such, the only expected error codes for \a lpf_rc are LPF_SUCCESS and
		 * LPF_ERR_FATAL, the latter of which cannot be mitigated and encapsulates
		 * run-time conditions that cannot be normally checked for (e.g., someone in
		 * the server room tripping over a network cable, thus bringing down a
		 * connection).
		 *
		 * @returns SUCCESS If \a lpf_rc was LPF_SUCCESS; and
		 * @returns PANIC   if \a lpf_rc was LPF_ERR_FATAL.
		 *
		 * On any other LPF error code, this function will return PANIC but also log
		 * an error to stderr and, if enabled, trips an assertion.
		 */
		inline RC checkLPFerror( const lpf_err_t lpf_rc ) {
			if( lpf_rc != LPF_SUCCESS ) {
				// failure at this point cannot be mitigated and possibly violates LPF spec
				/* LCOV_EXCL_START */
				if( lpf_rc != LPF_ERR_FATAL ) {
					std::cerr << "Error (level-1 collectives, BSP): LPF returned an "
						<< "unexpected error code. Please submit a bug report.\n";
#ifndef NDEBUG
					const bool lpf_spec_says_this_should_never_happen = false;
					assert( lpf_spec_says_this_should_never_happen );
#endif
				}
				return PANIC;
				/* LCOV_EXCL_STOP */
			}
			return SUCCESS;
		}

	} // end anonymous namespace

} // end namespace grb::internal

#endif // end ``_H_GRB_BSP_ERROR´´

