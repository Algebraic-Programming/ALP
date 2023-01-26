
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
 * This file contains a register of all backends that are either implemented,
 * under implementation, or conceived and recorded for future consideration to
 * implement.
 *
 * @author: A. N. Yzelman
 * @date 21st of December, 2016
 */

#ifndef _H_GRB_BACKENDS
#define _H_GRB_BACKENDS


namespace grb {

	/**
	 * A collection of all backends. Depending on which dependences were
	 * configured during the bootstrapping of this ALP installation, some of these
	 * backends may be disabled.
	 *
	 * \internal
	 * The collection includes backend identifiers that are for internal use only.
	 * \endinternal
	 *
	 * \ingroup backends
	 */
	enum Backend {

		/**
		 * The sequential reference implementation. Supports fast operations with
		 * both sparse and dense vectors, and employs auto-vectorisation.
		 */
		reference,

		/**
		 * The threaded reference implementation. Supports fast operations with both
		 * sparse and dense vectors. Employs OpenMP used with a mixture of fork/join
		 * and SPMD programming styles.
		 */
		reference_omp,

		/**
		 * A backend that automatically extracts hyperDAGs from user computations. It
		 * only captures metadata for recording the hyperDAG, and relies on another
		 * backend to actually execute the requested computations-- by default, this
		 * is the #reference backend.
		 */
		hyperdags,

		/**
		 * \internal
		 * A shared-memory parallel distribution based on a row-wise 1D block-cyclic
		 * data distribution using shared vector data.
		 * \endinternal
		 */
		shmem1D,

		/**
		 * \internal
		 * Like shmem1D, but using interleaved vector allocation. Useful for multi-
		 * socket single-node targets. From experience, this is a good choice for up
		 * to four sockets-- after which BSP2D becomes preferred.
		 * \endinternal
		 */
		NUMA1D,

		/**
		 * \internal
		 * A superclass of all LPF-based implementations. Not a "real" (selectable)
		 * backend.
		 * \endinternal
		 */
		GENERIC_BSP,

		/**
		 * A parallel implementation based on a row-wise 1D data distribution,
		 * implemented using LPF.
		 *
		 * This backend manages multiple user processes, manages data distributions
		 * of containers between those user processes, and decomposes primitives into
		 * local compute phases with intermittent communications. For local compute
		 * phases it composes with a single user process backend, #reference by
		 * default.
		 */
		BSP1D,

		/**
		 * \internal
		 * Like BSP1D, but stores each matrix twice. Combined with the normal
		 * reference implementation, this actually stores all matrices four times
		 * This implementation is useful for maximum performance, at the cost of
		 * the additional memory usage.
		 * \endinternal
		 */
		doublyBSP1D,

		/**
		 * \internal
		 * A parallel implementation based on a block-cyclic 2D data distribution,
		 * implemented using PlatformBSP. This implementation will likely outperform
		 * BSP1D and doublyBSP1D as the number of nodes involved in the computation
		 * increases with the problem sizes.
		 * \endinternal
		 */
		BSP2D,

		/**
		 * \internal
		 * Like BSP2D, but automatically improves the distribution while executing
		 * user code-- while initial computations are slowed down, the user
		 * application will speed up as this GraphBLAS implementation infers more
		 * information about the best data distribution.
		 * When enough statistics are gathered, data is redistributed and all future
		 * operations execute much faster than with BSP2D alone.
		 * \endinternal
		 */
		autoBSP,

		/**
		 * \internal
		 * Like autoBSP, except that the best distribution is precomputed whenever a
		 * matrix is read in. This pre-processing step is very expensive. Use autoBSP
		 * when unsure if the costs of a full preprocessing stage is worth it.
		 * \endinternal
		 */
		optBSP,

		/**
		 * A composed backend that uses #reference_omp within each user process and
		 * #BSP1D between sockets.
		 *
		 * This backend is implemented using the #BSP1D code, with the process-local
		 * backend overridden from #reference to #reference_omp.
		 */
		hybrid,

		/**
		 * \internal
		 * A hybrid that uses #shmem1D within each socket and #BSP1D between sockets.
		 * Recommended for a limited number of sockets and a limited amount of nodes,
		 * i.e., for a small cluster.
		 * \endinternal
		 */
		hybridSmall,

		/**
		 * \internal
		 * A hybrid that uses #numa1D within each socket and #BSP1D between sockets.
		 * Recommended for a limited number of nodes with up to two sockets each.
		 *
		 * This variant is expected to perform better than #hybridSmall for
		 * middle-sized clusters.
		 * \endinternal
		 */
		hybridMid,

		/**
		 * \internal
		 * A hybrid that uses #numa1D within each socket and #autoBSP between sockets.
		 * Recommended for a large number of nodes with up to two sockets each.
		 *
		 * This variant is expected to perform better than #hybridSmall and #hybridMid
		 * for larger clusters.
		 *
		 * If there are many nodes each with many sockets (four or more) each, then
		 * the use of flat (non-hybrid) #BSP2D or #autoBSP is recommended instead.
		 * \endinternal
		 */
		hybridLarge,

		/**
		 * \internal
		 * A hybrid variant that is optimised for a minimal memory footprint.
		 * \endinternal
		 */
		minFootprint,

		/**
		 * A variant for Snitch RISC-V cores. It is based on an older #reference
		 * backend.
		 */
		banshee,

		/**
		 * \internal
		 * A variant for RISC-V processors with (I)SSR extensions.
		 *
		 * \note This backend is used internally by the #banshee backend; it is not
		 *       selectable.
		 * \endinternal
		 */
		banshee_ssr

	};

} // namespace grb

#endif

