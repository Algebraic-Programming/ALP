
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
		 * This class controls the distribution of vector and matrix data over user
		 * processes.
		 *
		 * This base class provides a trivial implementation for backends that support
		 * one user process; therefore, only backends that support more than one user
		 * process need to specialise it.
		 */
		template< enum Backend backend = config::default_backend >
		class Distribution {

			public:

				/**
				 * The block size of the distribution.
				 *
				 * Rather than distributing individual elements within a range, a block of
				 * this size will be distributed instead.
				 */
				static constexpr size_t blocksize() {
					return 1;
				}

				/**
				 * Translates a global index to a process ID that owns that index.
				 *
				 * @param[in] i The global index.
				 * @param[in] n The global length.
				 *
				 * The argument \a i must be strictly smaller than \a n.
				 *
				 * @param[in] P The total number of processes.
				 *
				 * The argument \a P must be strictly larger than one.
				 */
				static constexpr size_t global_index_to_process_id(
// need to employ an ugly hack to make sure all of compilation without warnings
// and doxygen work
#ifdef __DOXYGEN__
					const size_t i, const size_t n, const size_t P
#else
					const size_t, const size_t, const size_t
#endif
				) {
					return 0;
				}

				/**
				 * Translates a global index to a local one.
				 *
				 * @param[in] global The global index.
				 * @param[in] n      The global size.
				 * @param[in] P      The number of user processes.
				 *
				 * The argument \a global must be strictly smaller than \a n.
				 *
				 * The argument \a P must be strictly larger than one.
				 */
				static inline size_t global_index_to_local(
					const size_t global, const size_t n,
					const size_t P
				) {
					(void) n;
					(void) P;
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
				static inline size_t local_index_to_global(
					const size_t local, const size_t n,
					const size_t s, const size_t P
				) {
					(void) n;
					(void) s;
					(void) P;
					return local;
				}

				/**
				 * For a given global length, how many elements or rows are stored at
				 * \em all user processes preceding a given process \a s.
				 *
				 * @param[in] global_size The globally distributed range.
				 * @param[in] s           The process ID.
				 * @param[in] P           The total number of processes.
				 *
				 * @returns The number of elements preceding \a s.
				 */
				static constexpr size_t local_offset(
// need to employ an ugly hack to make sure all of compilation without warnings
// and doxygen work
#ifdef __DOXYGEN__
					const size_t global_size,
					const size_t s, const size_t P
#else
					const size_t, const size_t, const size_t
#endif
				) {
					return 0;
				}

				/**
				 * Inverse function of #local_offset.
				 *
				 * @param[in] offset      The offset to query for.
				 * @param[in] global_size The globally distributed range.
				 * @param[in] P           The total number of processes.
				 *
				 * @returns The process whose #local_offset is the maximum of all those
				 *          smaller or equal to \a offset.
				 */
				static inline size_t offset_to_pid(
					const size_t offset, const size_t global_size,
					const size_t P
				) {
					(void) offset;
					(void) global_size;
					(void) P;
					return 0;
				}

		};

	} // namespace internal

} // namespace grb

#endif // end _H_GRB_DISTRIBUTION_BASE

