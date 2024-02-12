
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
 * @author A. N. Yzelman & J. M. Nash
 * @date 20th of February, 2017
 */

#ifndef _H_GRB_BSP_INTERNAL_COLL
#define _H_GRB_BSP_INTERNAL_COLL

#include <lpf/core.h>

#include <lpf/collectives.h>

#include <graphblas/bsp1d/init.hpp>


namespace grb {

	/**
	 * Collective communications using the GraphBLAS operators for reduce-style
	 * operations.
	 *
	 * The functions defined herein are only available when compiled with LPF.
	 */
	namespace internal {

		// allgather collective from library

		/*
		 * Ensure the destination buffer can hold at least this number of elements.
		 *
		 * It shall only be legal to call collectives that result in output not
		 * exceeding what is initialised here; otherwise, use of said collectives
		 * will result in UB.
		 */
		template< typename T >
		RC initCollectivesBuffer( const size_t num_elements ) {
			auto &data = internal::grb_BSP1D.load();
			return data.ensureBufferSize( num_elements * sizeof( T ) );
		}

		/* Schedules a gather operation from memory slot src (with src_offset)
		 * across each process, to memory slot dst (with offset dst_offset) on
		 * the root process. The size of each message may vary across processes,
		 * however, the accumulated sizes should add up to the provided total.
		 *
		 * @param[in] src        The source local or global memory slot.
		 * @param[in] src_offset The source memory slot offset.
		 * @param[in] dst        The destination local or global memory slot.
		 * @param[in] dst_offset The destination memory slot offset.
		 * @param[in] size       The number of bytes to transfer.
		 * @param[in] total      The total number of bytes transfered by all
		 *                       processes.
		 * @param[in] root       The id of the root process.
		 *
		 * \parblock
		 * \par Performance semantics:
		 * -# Problem size N: \f$ total \f$
		 * -# local work: \f$ 0 \f$ ;
		 * -# transferred bytes: \f$ N \f$ ;
		 * -# BSP cost: \f$ Ng + l \f$;
		 * \endparblock
		 *
		 */
		RC gather(
			const lpf_memslot_t src, const size_t src_offset,
			const lpf_memslot_t dst, const size_t dst_offset,
			const size_t size, const size_t total,
			const lpf_pid_t root
		);

		/* Schedules an allgather operation from memory slot src (with src_offset)
		 * across each process, to memory slot dst (with offset dst_offset).
		 * The size of each message may vary across processes,
		 * however, the accumulated sizes should add up to the provided total.
		 *
		 * @param[in] src          The source local or global memory slot.
		 * @param[in] src_offset   The source memory slot offset.
		 * @param[in] dst          The destination local or global memory slot.
		 * @param[in] dst_offset   The destination memory slot offset.
		 * @param[in] size         The number of bytes to transfer. This can be
		 *                         different on each process.
		 * @param[in] total        The total number of bytes received by each 
		 *                         process.
		 * @param[in] exclude_self If <tt>false</tt>, this process will gather its
		 *                         local source to its local destination.
		 *
		 * \parblock
		 * \par Performance semantics:
		 *  -# Problem size N: \f$ total \f$
		 *  -# local work: \f$ 0 \f$ ;
		 *  -# transferred bytes: \f$ N \f$ ;
		 *  -# BSP cost: \f$ Ng + l \f$;
		 * \endparblock
		 *
		 * @returns #SUCCESS If the requested operation completed successfully.
		 * @returns #ILLEGAL If \a size is larger than \a total
		 * @returns #PANIC   If the underlying communication layer encounters an
		 *                   unrecoverable error.
		 */
		RC allgather(
			const lpf_memslot_t src, const size_t src_offset,
			const lpf_memslot_t dst, const size_t dst_offset,
			const size_t size, const size_t total,
			const bool exclude_self = true
		);

		/**
		 * \internal
		 *
		 * Schedules an alltoall operation from memory slot src and offset src_offset
		 * across each process, to the BSP data buffer.
		 *
		 * @param[in] src           The source global memory slot.
		 * @param[in] src_offset    The source memory slot offset.
		 * @param[in] size          The number of bytes to transfer.
		 * @param[in] buffer_offset The number of bytes in the buffer to offset. This
		 *                          corresponds to the destination memory.
		 * @param[in] exclude_self  Whether or not to copy local elements.
		 *
		 * The supplied \a size and \a buffer_offset can vary across processes.
		 *
		 * The last two elements are optional. The default for \a buffer_offset is
		 * \f$ 0 \f$ while the default for \a exclude_self is <tt>true</tt>.
		 *
		 * \parblock
		 * \par Performance semantics:
		 * -# Problem size N: \f$ P * \max_s \mathit{size} \f$,
		 *    with \f$ s \f$ the process IDs.
		 * -# local work: \f$ 0 \f$
		 * -# intra-process data movement: \a size if not \a exclude_self;
		 *                                 0 otherwise.
		 * -# inter-process data movement: \f$ N \f$
		 * -# latencies: \f$ 1 \f$
		 * -# BSP cost: \f$ Ng + l \f$
		 * \endparblock
		 *
		 * \endinternal
		 */
		RC alltoall(
			const lpf_memslot_t src, const size_t src_offset, const size_t size,
			const size_t buffer_offset = 0,
			const bool exclude_self = true
		);

		/* Schedules an alltoallv operation from memory slot src into the buffer.
		 *
		 * @param[in] src          The source global memory slot.
		 * @param[in] out          An array of size P describing how many bytes this
		 *                         process sends to each process \a k.
		 * @param[in] src_disp     An array of size \a P noting the offset of \a src
		 *                         where the data for each of the processes resides.
		 * @param[in] in           An array of size P describing how many bytes this
		 *                         process receives from each process \a k.
		 * @param[in] dst_disp     An array of size \a P noting the offset for each
		 *                         of the receiving processes.
		 * @param[in] exclude_self Whether or not to copy local elements.
		 *
		 * The last argument is optional. The default for \a exclude_self is
		 * <tt>true</tt>.
		 *
		 * \parblock
		 * \par Performance semantics:
		 * -# Problem size N: \f$ \mathcal{O}(P\mathit{max\_h})\f$
		 * -# local work: \f$ 0 \f$ ;
		 * -# h-relation: \f$ \max\{\sum\mathit{in}_i,\sum\mathit{out}_i\} \f$,
		 *    or less if \a exclude_self is <tt>true</tt>.
		 * -# BSP cost: \f$ \mathcal{O}(hg + l) \f$;
		 * \endparblock
		 */
		RC alltoallv(
			const lpf_memslot_t src,
			const size_t * out,
			const size_t src_offset, const size_t * src_disp,
			const size_t * in,
			const size_t dst_offset, const size_t * dst_disp,
			const bool exclude_self = true
		);

		/**
		 * Specify the maximum number of messages, the maximum buffer size for these
		 * messages, and the allocation of a local or global memory sot - preamble to
		 * communications
		 *
		 * @param[in] data The persistent BSP state.
		 * @param[in] coll The BSP collective comms structure.
		 * @param[in] maxMessages The maximum number of messages being transferred.
		 * @param[in] maxBufSize The maximum number of bytes required for communications.
		 * @param[in] localMemslot The number of local memory slots requested.
		 * @param[in] globalMemslot The number of gobal memory slots requested.
		 */
		RC commsPreamble(
			internal::BSP1D_Data &data,
			lpf_coll_t * coll,
			const size_t maxMessages,
			const size_t maxBufSize = 0,
			const unsigned int localMemslot = 0,
			const unsigned int globalMemslot = 0
		);

		/**
		 * Specify the maximum number of messages, the maximum buffer size for these
		 * messages, and the allocation of a local or global memory sot - postamble to
		 * communications
		 *
		 * @param[in] data          The persistent BSP state.
		 * @param[in] coll          The BSP collective comms structure.
		 * @param[in] maxMessages   The maximum number of messages that were being
		 *                          transferred.
		 * @param[in] maxBufSize    The maximum number of bytes that were required for
		 *                          communications.
		 * @param[in] localMemslot  The number of local memory slots that were
		 *                          requested.
		 * @param[in] globalMemslot The number of gobal memory slots that were
		 *                          requested.
		 */
		RC commsPostamble(
			internal::BSP1D_Data &data,
			lpf_coll_t * coll,
			const size_t maxMessages,
			const size_t maxBufSize = 0,
			const unsigned int localMemslot = 0,
			const unsigned int globalMemslot = 0
		);

	} // namespace internal

} // namespace grb

#endif // end ``_H_GRB_BSP_COLL''
