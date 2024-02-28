
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
 * @date 27th of January, 2017
 */

#ifndef _H_GRB_DISTRIBUTION_BASE
#define _H_GRB_DISTRIBUTION_BASE

#include <graphblas/backends.hpp>

#include "config.hpp"

namespace grb {

	namespace internal {

		/**
		 * \todo documentation.
		 */
		template< enum Backend backend = config::default_backend >
		class Distribution {

		public:
			/** \todo documentation. */
			static constexpr size_t blocksize() {
				return 1;
			}

			/**
			 * \todo Expand documentation.
			 *
			 * Arguments, in order:
			 *
			 * @param[in] The global index
			 * @param[in] The global length
			 * @param[in] The total number of processes
			 *
			 * In the default case (a single user process) all arguments are ignored.
			 */
			static constexpr size_t global_index_to_process_id( const size_t, const size_t, const size_t ) {
				return 0;
			}

			/** \todo documentation. */
			static inline size_t global_index_to_local( const size_t global, const size_t n, const size_t P ) {
				(void)n;
				(void)P;
				return global;
			}

			/**
			 * For a given local index at a given process, calculate the corresponding
			 * global index.
			 *
			 * @param[in] local The local index of the vector or matrix row/column
			 *                  coordinate.
			 * @param[in]   n   The total length of the given vector, or the total
			 *                  number of matrix rows or columns.
			 * @param[in]   s   This process ID.
			 * @param[in]   P   The global number of user processes tied up with this
			 *                  GraphBLAS run.
			 *
			 * @return The global index of the given local \a index.
			 */
			static inline size_t local_index_to_global( const size_t local, const size_t n, const size_t s, const size_t P ) {
				(void)n;
				(void)s;
				(void)P;
				return local;
			}

			/** \todo documentation. */
			static inline size_t local_offset( const size_t global_size, const size_t local_pid, const size_t npid ) {
				(void)global_size;
				(void)local_pid;
				(void)npid;
				return 0;
			}

			/**
			 * Inverse function of #local_offset.
			 *
			 * \todo Expand documentation.
			 */
			static inline size_t offset_to_pid( const size_t offset, const size_t global_size, const size_t npid ) {
				(void)offset;
				(void)global_size;
				(void)npid;
				return 0;
			}
		};

	} // namespace internal

} // namespace grb

#endif // end _H_GRB_DISTRIBUTION_BASE
