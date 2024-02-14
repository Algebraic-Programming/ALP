
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

#include <graphblas/bsp/error.hpp>
#include <graphblas/bsp/internal-collectives.hpp>


grb::RC grb::internal::gather(
	const lpf_memslot_t src, const size_t src_offset,
	const lpf_memslot_t dst, const size_t dst_offset,
	const size_t size, const size_t total,
	const lpf_pid_t root
) {
#ifdef _DEBUG
	std::cout << "In internal::gather (direct)" << std::endl;
#endif
	// sanity check
	if( size > total ) {
		return ILLEGAL;
	}
	if( size > total ) {
		return ILLEGAL;
	}

	// we need access to LPF context
	internal::BSP1D_Data &data = internal::grb_BSP1D.load();

	if( root >= data.P ) {
		return ILLEGAL;
	}

	// ensure we can support comms pattern
	RC ret = data.ensureMaxMessages( data.P - 1 );
	if( ret != SUCCESS ) { return ret; }

	// schedule and execute
	lpf_err_t lpf_rc = LPF_SUCCESS;
	if( data.s != root ) {
		// put local data remotely
		lpf_rc = lpf_put(
			data.context,
			src, src_offset, root,
			dst, dst_offset,
			size,
			LPF_MSG_DEFAULT
		);
	}
	if( lpf_rc == LPF_SUCCESS ) {
		lpf_rc = lpf_sync( data.context, LPF_SYNC_DEFAULT );
	}

	// error handling
	ret = checkLPFerror( lpf_rc, "internal::gather, direct variant" );

#ifdef _DEBUG
	std::cout << "\t internal::gather (direct): exiting" << std::endl;
#endif

	// done
	return ret;
}

grb::RC grb::internal::allgather(
	const lpf_memslot_t src, const size_t src_offset,
	const lpf_memslot_t dst, const size_t dst_offset,
	const size_t size, const size_t total,
	const bool exclude_self
) {
#ifdef _DEBUG
	std::cout << "In internal::allgather (direct)" << std::endl;
#endif
	// sanity check
	if( size > total ) {
		return ILLEGAL;
	}

	// we need access to LPF context
	internal::BSP1D_Data &data = internal::grb_BSP1D.load();

	// check for trivial execution
	if( data.P == 1 ) {
		return SUCCESS;
	}

	RC ret = SUCCESS;
	lpf_err_t lpf_rc = LPF_SUCCESS;

	if( exclude_self ) {
		ret = data.ensureMaxMessages( 2 * data.P - 2 );
	} else {
		ret = data.ensureMaxMessages( 2 * data.P );
	}
	if( ret == SUCCESS ) {
		for( lpf_pid_t i = 0; i < data.P; ++i ) {
			if( exclude_self && i == data.s ) { continue; }
				lpf_rc = lpf_put(
						data.context,
						src, src_offset,
						i, dst, dst_offset,
						size, LPF_MSG_DEFAULT
					);
		}

		// finish comms
		if( lpf_rc == LPF_SUCCESS ) {
			lpf_rc = lpf_sync( data.context, LPF_SYNC_DEFAULT );
		}

		// finish LPF section
		ret = checkLPFerror( lpf_rc, "grb::internal::allgather (direct, BSP)" );
	}

#ifdef _DEBUG
	std::cout << "\t internal::allgather (direct): exiting" << std::endl;
#endif

	// done
	return ret;
}

grb::RC grb::internal::alltoall(
	const lpf_memslot_t src, const size_t src_offset,
	const size_t size,
	const size_t buffer_offset,
	const bool exclude_self
) {
	// we need access to LPF context
	internal::BSP1D_Data &data = internal::grb_BSP1D.load();
#ifdef _DEBUG
	std::cout << "In internal::alltoall (direct)" << std::endl;
	std::cout << data.s << ", calls alltoall with src slot " << src << ", offset "
		<< src_offset << ", element size " << size << ", buffer (output) offset "
		<< buffer_offset << ", and exclude_self " << exclude_self << "\n";
#endif

	// catch trivial case
	if( data.P == 1 && exclude_self ) {
		return SUCCESS;
	}

	// make sure we can support comms pattern
	const size_t nmsgs = exclude_self ? 2 * data.P - 2 : 2 * data.P;
	RC ret = data.ensureMaxMessages( nmsgs );
	if( ret != SUCCESS ) { return ret; }

	// get remote contributions
	lpf_err_t lpf_rc = LPF_SUCCESS;
	for( lpf_pid_t k = 0; lpf_rc == SUCCESS && size > 0 && k < data.P; ++k ) {
		if( exclude_self && static_cast< size_t >( k ) == data.s ) {
			continue;
		}
#ifdef _DEBUG
		std::cout << data.s << ", alltoall calls lpf_get from process " << k
			<< " slot " << src << " offset " << src_offset << " into buffer at offset "
			<< buffer_offset + k * size << ". Copy size is " << size << "." << std::endl;
#endif
		lpf_rc = lpf_get(
			data.context,
			k, src, src_offset,
			data.slot, buffer_offset + k * size,
			size,
			LPF_MSG_DEFAULT
		);
	}

	// finish alltoall
	if( lpf_rc == LPF_SUCCESS ) {
		lpf_rc = lpf_sync( data.context, LPF_SYNC_DEFAULT );
	}

	// end of LPF section
	ret = checkLPFerror( lpf_rc, "internal::alltoall (direct, BSP)" );

#ifdef _DEBUG
	std::cout << "\t internal::alltoall (direct): exiting" << std::endl;
#endif

	// done
	return ret;
}

grb::RC grb::internal::alltoallv(
	const lpf_memslot_t src,
	const size_t * out, const size_t src_offset, const size_t * src_disp,
	const size_t * in, const size_t dst_offset, const size_t * dst_disp,
	const bool exclude_self
) {
	// we need access to LPF context
	internal::BSP1D_Data &data = internal::grb_BSP1D.load();
#ifdef _DEBUG
	std::cout << "In internal::alltoallv (direct)" << std::endl;
	std::cout << data.s << ", entering alltoallv. I am sending ( " << out[ 0 ];
	for( size_t i = 1; i < data.P; ++i ) {
		std::cout << ", " << out[ i ];
	}
	std::cout << " ) bytes to each of the " << data.P << " processes. ";
	std::cout << "Exclude_self reads " << exclude_self << ".\n";
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
	(void) out;
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
	if( ret != SUCCESS ) { return ret; }

	// schedule pattern
	lpf_err_t lpf_rc = LPF_SUCCESS;
	if( nmsgs > 0 ) {
		for( size_t k = 0; lpf_rc == LPF_SUCCESS && k < data.P; ++k ) {
			if( exclude_self && k == data.s ) { continue; }
			if( in[ k ] == 0 ) { continue; }
#ifdef _DEBUG
			std::cout << data.s << ", alltoallv issues get from process " << k
				<< " slot " << src << " at offset " << src_offset + src_disp[ k ]
				<< " bytes, to local slot " << data.slot << " at offset "
				<< dst_offset + dst_disp[ k ] << ", copying " << in[ k ] << " bytes."
				<< std::endl;
#endif
			lpf_rc = lpf_get(
				data.context,
				k, src, src_offset + src_disp[ k ],
				data.slot, dst_offset + dst_disp[ k ],
				in[ k ],
				LPF_MSG_DEFAULT
			);
		}
	}
#ifdef _DEBUG
	else {
		std::cout << data.s << ", alltoallv: empty alltoallv at PID " << data.s
			<< "\n";
	}
#endif

	// execute pattern
	if( lpf_rc == LPF_SUCCESS ) {
		lpf_rc = lpf_sync( data.context, LPF_SYNC_DEFAULT );
	}

	// handle any errors
	ret = checkLPFerror( lpf_rc, "internal::alltoallv (direct)" );

#ifdef _DEBUG
	std::cout << "\t internal::alltoallv (direct): exiting" << std::endl;
#endif

	// done
	return ret;
}

