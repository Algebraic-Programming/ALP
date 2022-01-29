
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
 * @author A. N. Yzelman & J. M. Nash
 * @date 20th of February, 2017
 */

#include <graphblas/bsp/internal-collectives.hpp>

grb::RC
grb::internal::commsPreamble( internal::BSP1D_Data & data, lpf_coll_t * coll, const size_t maxMessages, const size_t maxBufSize, const unsigned int localMemslot, const unsigned int globalMemslot ) {
#ifdef _DEBUG
	std::cout << "internal::commsPreamble, called with coll at " << coll << ", max messages and buffer size " << maxMessages << " - " << maxBufSize << ", local and global memslots " << localMemslot
			  << " - " << globalMemslot << "\n";
#endif
	if( lpf_collectives_init( data.context, data.s, data.P, 0, 0, 0, coll ) != LPF_SUCCESS ) {
#ifdef _DEBUG
		std::cerr << "internal::commsPreamble, could not initialise lpf_coll_t!" << std::endl;
#endif
#ifndef NDEBUG
		const bool could_not_initialize_lpf_collectives = false;
		assert( could_not_initialize_lpf_collectives );
#endif
		return PANIC;
	}
	if( maxBufSize > 0 && data.checkBufferSize( maxBufSize ) != SUCCESS ) {
#ifdef _DEBUG
		std::cerr << "internal::commsPreamble, could not reserve buffer size "
			<< "of " << maxBufSize << "!" << std::endl;
#endif
#ifndef NDEBUG
		const bool insufficient_buffer_capacity_for_requested_pattern = false;
		assert( insufficient_buffer_capacity_for_requested_pattern );
#endif
		return PANIC;
	}
	if( maxMessages > 0 && data.ensureMaxMessages( maxMessages ) != SUCCESS ) {
#ifdef _DEBUG
		std::cerr << "internal::commsPreamble, could not reserve max msg "
			<< "buffer size of " << maxMessages << "!" << std::endl;
#endif
#ifndef NDEBUG
		const bool could_not_resize_lpf_message_buffer = false;
		assert( could_not_resize_lpf_message_buffer );
#endif
		return PANIC;
	}
	if( (localMemslot > 0 || globalMemslot > 0) &&
		data.ensureMemslotAvailable( localMemslot + globalMemslot ) != SUCCESS
	) {
#ifdef _DEBUG
		std::cerr << "internal::commsPreamble, could not reserve " << localMemslot
			<< " local memory slots!" << std::endl;
		std::cerr << "internal::commsPreamble, could not reserve " << globalMemslot
			<< " global memory slots!" << std::endl;
#endif
#ifndef NDEBUG
		const bool could_not_resize_lpf_memory_slot_capacity = false;
		assert( could_not_resize_lpf_memory_slot_capacity );
#endif
		return PANIC;
	}
#ifdef _DEBUG
	std::cout << "internal::commsPreamble, taking requested number of memslots..." << std::endl;
#endif
	if( localMemslot > 0 ) {
		data.signalMemslotTaken( localMemslot );
	}
	if( globalMemslot > 0 ) {
		data.signalMemslotTaken( globalMemslot );
	}
#ifdef _DEBUG
	std::cout << "internal::commsPreamble, success." << std::endl;
#endif
	return SUCCESS;
}

grb::RC grb::internal::commsPostamble( internal::BSP1D_Data &data,
	lpf_coll_t * const coll,
	const size_t maxMessages,
	const size_t maxBufSize,
	const unsigned int localMemslot,
	const unsigned int globalMemslot ) {
	(void)maxMessages;
	(void)maxBufSize;
	if( localMemslot > 0 ) {
		data.signalMemslotReleased( localMemslot );
	}
	if( globalMemslot ) {
		data.signalMemslotReleased( globalMemslot );
	}
	if( lpf_collectives_destroy( *coll ) != LPF_SUCCESS ) {
		assert( false );
		return PANIC;
	}
	return SUCCESS;
}

grb::RC grb::internal::gather( const lpf_memslot_t src, const size_t src_offset, const lpf_memslot_t dst, const size_t dst_offset, const size_t size, const size_t total, const lpf_pid_t root ) {
	// sanity check
	if( src_offset + size > total ) {
		return ILLEGAL;
	}
	if( dst_offset + size > total ) {
		return ILLEGAL;
	}

	// we need access to LPF context
	internal::BSP1D_Data & data = internal::grb_BSP1D.load();

	// ensure we can support comms pattern: total
	lpf_coll_t coll;
	if( commsPreamble( data, &coll, data.P, total ) != SUCCESS ) {
		assert( false );
		return PANIC;
	}

	// schedule gather
	if( data.s != root ) {
		// put local data remotely
		if( lpf_put( data.context, src, src_offset, root, dst, dst_offset, size, LPF_MSG_DEFAULT ) != LPF_SUCCESS ) {
			assert( false );
			return PANIC;
		}
	}

	// finish gather
	if( lpf_sync( data.context, LPF_SYNC_DEFAULT ) != LPF_SUCCESS ) {
		assert( false );
		return PANIC;
	}

	if( commsPostamble( data, &coll, data.P, total ) != SUCCESS ) {
		assert( false );
		return PANIC;
	}

	// done
	return SUCCESS;
}

grb::RC grb::internal::allgather(
	const lpf_memslot_t src, const size_t src_offset,
	const lpf_memslot_t dst, const size_t dst_offset,
	const size_t size, const size_t total,
	const bool exclude_self
) {
	// sanity check
	if( size > total ) {
		return ILLEGAL;
	}

	// we need access to LPF context
	internal::BSP1D_Data &data = internal::grb_BSP1D.load();

	// ensure we can support comms pattern: total
	lpf_coll_t coll;
	if( commsPreamble( data, &coll, data.P, total ) != SUCCESS ) {
#ifndef NDEBUG
		const bool commsPreamble_returned_error = false;
		assert( commsPreamble_returned_error );
#endif
		return PANIC;
	}

	// schedule allgather
	for( lpf_pid_t i = 0; i < data.P; ++i ) {
		// may not want to send to myself
		if( exclude_self && i == data.s ) {
			continue;
		}
		// put local data remotely
		if( lpf_put( data.context,
				src, src_offset,
				i, dst, dst_offset,
				size,
				LPF_MSG_DEFAULT
			) != LPF_SUCCESS
		) {
#ifndef NDEBUG
			const bool lpf_put_returned_error = false;
			assert( lpf_put_returned_error );
#endif
			return PANIC;
		}
	}

	// finish allgather
	if( lpf_sync( data.context, LPF_SYNC_DEFAULT ) != LPF_SUCCESS ) {
#ifndef NDEBUG
		const bool lpf_sync_returned_error = false;
		assert( lpf_sync_returned_error );
#endif
		return PANIC;
	}

	if( commsPostamble( data, &coll, data.P, total ) != SUCCESS ) {
#ifndef NDEBUG
		const bool commsPostamble_returned_error = false;
		assert( commsPostamble_returned_error );
#endif
		return PANIC;
	}

	// done
	return SUCCESS;
}

grb::RC grb::internal::alltoall( const lpf_memslot_t src, const size_t src_offset, const size_t size, const size_t buffer_offset, const bool exclude_self ) {
	// we need access to LPF context
	internal::BSP1D_Data & data = internal::grb_BSP1D.load();
#ifdef _DEBUG
	std::cout << data.s << ", calls alltoall with src slot " << src << ", offset " << src_offset
		<< ", element size " << size << ", buffer (output) offset " << buffer_offset
		<< ", and exclude_self " << exclude_self << "\n";
#endif

	// catch trivial case
	if( data.P == 1 && exclude_self ) {
		return SUCCESS;
	}

	// make sure we can support comms pattern
	const size_t nmsgs = exclude_self ? 2 * data.P - 2 : 2 * data.P;
	RC ret = data.ensureMaxMessages( nmsgs );

	// get remote contributions
	for( lpf_pid_t k = 0; ret == SUCCESS && size > 0 && k < data.P; ++k ) {
		if( exclude_self && static_cast< size_t >( k ) == data.s ) {
			continue;
		}
#ifdef _DEBUG
		std::cout << data.s << ", alltoall calls lpf_get from process " << k << " slot " << src << " offset " << src_offset << " into buffer at offset " << buffer_offset + k * size
				  << ". Copy size is " << size << "." << std::endl;
#endif
		if( lpf_get( data.context, k, src, src_offset, data.slot, buffer_offset + k * size, size, LPF_MSG_DEFAULT ) != LPF_SUCCESS ) {
			assert( false );
			ret = PANIC;
		}
	}

	// finish alltoall
	if( ret == SUCCESS && lpf_sync( data.context, LPF_SYNC_DEFAULT ) != LPF_SUCCESS ) {
		assert( false );
		ret = PANIC;
	}

	// done
	return ret;
}

grb::RC grb::internal::alltoallv( const lpf_memslot_t src,
	const size_t * out,
	const size_t src_offset,
	const size_t * src_disp,
	const size_t * in,
	const size_t dst_offset,
	const size_t * dst_disp,
	const bool exclude_self ) {
	// we need access to LPF context
	internal::BSP1D_Data & data = internal::grb_BSP1D.load();
#ifdef _DEBUG
	std::cout << data.s << ", entering alltoallv. I am sending ( " << out[ 0 ];
	for( size_t i = 1; i < data.P; ++i ) {
		std::cout << ", " << out[ i ];
	}
	std::cout << " ) bytes to each of the " << data.P << " processes. Exclude_self reads " << exclude_self << ".\n";
	std::cout << "\tI will receive ( " << in[ 0 ];
	for( size_t i = 1; i < data.P; ++i ) {
		std::cout << ", " << in[ i ];
	}
	std::cout << " ) bytes from remote processes." << std::endl;
	std::cout << "\tMy local offsets are ( " << src_disp[ 0 ];
	for( size_t i = 1; i < data.P; ++i ) {
		std::cout << ", " << src_disp[ i ];
	}
	std::cout << " ) bytes." << std::endl;
	std::cout << "\tMy destination offsets are ( " << dst_disp[ 0 ];
	for( size_t i = 1; i < data.P; ++i ) {
		std::cout << ", " << dst_disp[ i ];
	}
	std::cout << " ) bytes." << std::endl;
#else
	(void)out;
#endif

	// catch trivial case
	if( data.P == 1 && exclude_self ) {
		return SUCCESS;
	}

	// calculate number of messages required
	size_t nmsgs_in = 0, nmsgs_out = 0;
	for( size_t i = 0; i < data.P; ++i ) {
		if( exclude_self && i == data.s ) {
			continue;
		}
		nmsgs_in += in[ i ] > 0 ? 1 : 0;
		nmsgs_out += out[ i ] > 0 ? 1 : 0;
	}
	const size_t nmsgs = std::max( nmsgs_in, nmsgs_out );

	// make sure we can support comms pattern
	RC ret = data.ensureMaxMessages( nmsgs );

	// check if we can continue
	if( ret != SUCCESS ) {
		return ret;
	}

	if( nmsgs > 0 ) {
		for( size_t k = 0; k < data.P; ++k ) {
			if( exclude_self && k == data.s ) {
				continue;
			}
			if( in[ k ] == 0 ) {
				continue;
			}
#ifdef _DEBUG
			std::cout << data.s << ", alltoallv issues get from process " << k << " slot " << src << " at offset " << src_offset + src_disp[ k ] << " bytes, to local slot " << data.slot
					  << " at offset " << dst_offset + dst_disp[ k ] << ", copying " << in[ k ] << " bytes." << std::endl;
#endif
			if( lpf_get( data.context, k, src, src_offset + src_disp[ k ], data.slot, dst_offset + dst_disp[ k ], in[ k ], LPF_MSG_DEFAULT ) != LPF_SUCCESS ) {
				assert( false );
				return PANIC;
			}
		}
	}
#ifdef _DEBUG
	else {
		std::cout << data.s << ", alltoallv: empty alltoallv at PID " << data.s << "\n";
	}
#endif

	if( lpf_sync( data.context, LPF_SYNC_DEFAULT ) != LPF_SUCCESS ) {
		assert( false );
		return PANIC;
	}

	// done
	return SUCCESS;
}

