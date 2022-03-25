
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
 * @author: A. N. Yzelman
 * @date 21st of December, 2016
 *
 * @file This file contains a register of all backends that are either
 *       implemented, under implementation, or were at any point in time
 *       conceived and noteworthy enough to be recorded for future
 *       consideration to implement. It does so via the grb::Backend
 *       enum.
 */

#ifndef _H_GRB_BACKENDS
#define _H_GRB_BACKENDS

namespace grb {

	/**
	 * This enum collects all implemented backends. Depending on compile flags,
	 * some of these options may be disabled.
	 */
	enum Backend {

		/**
		 * The sequential reference implementation. Supports fast operations with
		 * both sparse and dense vectors.
		 */
		reference,

		/**
		 * The threaded reference implementation. Supports fast operations with both
		 * sparse and dense vectors.
		 */
		reference_omp,

		/**
		 * A shared-memory parallel distribution based on a row-wise 1D data
		 * distribution using shared vector data.
		 */
		shmem1D,

		/**
		 * Like shmem1D, but using interleaved vector allocation. Useful for multi-
		 * socket single-node targets. From experience, this is a good choice for up
		 * to four sockets-- after which BSP2D becomes preferred.
		 */
		NUMA1D,

		/**
		 * A superclass of all BSP-based implementations.
		 */
		GENERIC_BSP,

		/**
		 * A parallel implementation based on a row-wise 1D data distribution,
		 * implemented using PlatformBSP.
		 */
		BSP1D,

		/**
		 * Like BSP1D, but stores each matrix twice. Combined with the normal
		 * reference implementation, this actually stores all matrices four times
		 * This implementation is useful for maximum performance, at the cost of
		 * the additional memory usage.
		 */
		doublyBSP1D,

		/**
		 * A parallel implementation based on a block-cyclic 2D data distribution,
		 * implemented using PlatformBSP. This implementation will likely outperform
		 * BSP1D and doublyBSP1D as the number of nodes involved in the computation
		 * increases with the problem sizes.
		 */
		BSP2D,

		/**
		 * Like BSP2D, but automatically improves the distribution while executing
		 * user code-- while initial computations are slowed down, the user
		 * application will speed up as this GraphBLAS implementation infers more
		 * information about the best data distribution.
		 * When enough statistics are gathered, data is redistributed and all future
		 * operations execute much faster than with BSP2D alone.
		 */
		autoBSP,

		/**
		 * Like autoBSP, except that the best distribution is precomputed whenever a
		 * matrix is read in. This pre-processing step is very expensive. Use autoBSP
		 * when unsure if the costs of a full preprocessing stage is worth it.
		 */
		optBSP,

		/**
		 * A hybrid that uses shmem1D within each socket and BSP1D between sockets.
		 * Recommended for a limited number of sockets and a limited amount of nodes,
		 * i.e., for a small cluster.
		 */
		hybridSmall,

		/**
		 * A hybrid that uses numa1D within each socket and BSP1D between sockets.
		 * Recommended for a limited number of nodes with up to two sockets each.
		 *
		 * This variant is expected to perform better than hybrid1D for middle-sized
		 * clusters.
		 */
		hybridMid,

		/**
		 * A hybrid that uses numa1D within each socket and autoBSP between sockets.
		 * Recommended for a large number of nodes with up to two sockets each.
		 *
		 * This variant is expected to perform better than hybridSmall and hybridMid
		 * for larger clusters.
		 *
		 * If there are many nodes each with many sockets (four or more) each, then
		 * the use of flat (non-hybrid) #BSP2D or #autoBSP is recommended instead.
		 */
		hybridLarge,

		/**
		 * A hybrid variant that is optimised for a minimal memory footprint.
		 */
		minFootprint,

		/**
		 * A variant for RISC-V processors.
		 *
		 * Collaboration with ETH Zurich (ongoing).
		 */
		banshee,

		/**
		 * A variant for RISC-V processors with (I)SSR extensions
		 *
		 * Collaboration with ETH Zurich (ongoing).
		 */
		banshee_ssr

	};

} // namespace grb

#endif
