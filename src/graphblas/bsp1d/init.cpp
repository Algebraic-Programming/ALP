
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

// DBG
//#define _DEBUG
//#undef NDEBUG

//DO NOT REMOVE THIS HEADER!!! needed for 'finalize< _GRB_BSP1D_BACKEND >();'
#include <graphblas/init.hpp>


#include <graphblas/bsp/config.hpp>
#include <graphblas/bsp1d/init.hpp>

grb::utils::ThreadLocalStorage< grb::internal::BSP1D_Data > grb::internal::grb_BSP1D;

template<>
grb::RC grb::init< grb::BSP1D >(
	const size_t s, const size_t P, const lpf_t ctx
) {
	if( s == 0 ) {
		std::cerr << "Info: grb::init (BSP1D) called using " << P << " user processes.\n";
	}
#ifdef _DEBUG
	std::cout << s << ": creating thread-local store..." << std::endl;
#endif
	grb::internal::grb_BSP1D.store();
#ifdef _DEBUG
	std::cout << s << ": retrieving thread-local store..." << std::endl;
#endif
	grb::internal::BSP1D_Data &data = grb::internal::grb_BSP1D.load();
#ifdef _DEBUG
	std::cout << s << ": initializing thread-local store..." << std::endl;
#endif
	return data.initialize( ctx,
		static_cast< lpf_pid_t >(s),
		static_cast< lpf_pid_t >(P),
		grb::config::LPF::regs(),
		grb::config::LPF::maxh()
	);
}

template<>
grb::RC grb::finalize< grb::BSP1D >() {
	// retrieve local data
	grb::internal::BSP1D_Data &data = grb::internal::grb_BSP1D.load();
	// use it to print info
	const size_t s = data.s;
	if( s == 0 ) {
		std::cerr << "Info: grb::finalize (bsp1d) called.\n";
	}
#ifndef NDEBUG
	for( size_t k = 0; k < data.P; ++k ) {
		if( k == s ) {
			std::cerr << "\t process " << s << " is finalising\n";
		}
		if( lpf_sync( data.context, LPF_SYNC_DEFAULT )
			!= LPF_SUCCESS
		) {
			std::cerr << "\t process " << s << " failed to sync\n";
		}
	}
#endif
	// and then destroy it
	const grb::RC ret1 = data.destroy();
	// even if destroying local data failed, still try to destroy backend
	const grb::RC ret2 = finalize< _GRB_BSP1D_BACKEND >();
	// return the first non-success return code
	if( ret1 == SUCCESS ) {
		return ret2;
	}
	return ret2;
}

grb::RC grb::internal::BSP1D_Data::initialize( lpf_t _context,
	const lpf_pid_t _s, const lpf_pid_t _P,
	const size_t _regs, const size_t _maxh
) {
	// required buffer for all-to-all on #messages, followed by a prefix-sum
	// also dubs as the required buffer sufficient for all-reducing any
	// standard raw data type. To guarantee all-reducing user-defined types
	// is possible, a grb::Scalar must be introduced.
	const size_t bufferSize = 3 * _P * sizeof( size_t );
	return initialize( _context, _s, _P, _regs, _maxh, bufferSize );
}

grb::RC grb::internal::BSP1D_Data::initialize(
	lpf_t _context,
	const lpf_pid_t _s, const lpf_pid_t _P,
	const size_t _regs, const size_t _maxh,
	const size_t _bufsize
) {
#ifdef _DEBUG
	std::cout << "grb::internal::BSP1D_Data::initialize called with arguments "
		<< _context << ", " << s << ", " << _P << ", " << _regs << ", "
		<< _maxh << ", " << _bufsize << "\n";
#endif
	// arg checks
	assert( _context != NULL );
	assert( _regs > 0 );

	// initialise fields
	regs_taken = 0;
	s = _s;
	P = _P;
	context = _context;
	lpf_regs = 0;
	lpf_maxh = 0;
	buffer_size = 0;
	destroyed = false;
	payload_size = 0;
	tag_size = 0;
	max_msgs = 0;
	cur_payload = new( std::nothrow ) size_t[ P ];
	if( cur_payload == NULL ) {
#ifdef _DEBUG
		std::cerr << "\t out of memory while initializing (1)\n";
#endif
		return OUTOFMEM;
	}
	cur_msgs = new( std::nothrow ) size_t[ P ];
	if( cur_msgs == NULL ) {
		delete[] cur_payload;
#ifdef _DEBUG
		std::cerr << "\t out of memory while initializing (2)\n";
#endif
		return OUTOFMEM;
	}
	for( size_t i = 0; i < P; ++i ) {
		cur_payload[ i ] = cur_msgs[ i ] = 0;
	}
	queue_status = 0;
	queue = LPF_INVALID_BSMP;

	if( _bufsize > 0 ) {
		// try and allocate the buffer
		const int prc = posix_memalign(
			&buffer,
			grb::config::CACHE_LINE_SIZE::value(),
			_bufsize
		);
		// Note: always uses local allocation (and rightly so)
		if( prc != 0 ) {
			delete[] cur_payload;
			delete[] cur_msgs;
#ifdef _DEBUG
			std::cerr << "\t out of memory while initializing (3)\n";
#endif
			return OUTOFMEM;
		}
	} else {
		// set buffer to empty
		buffer = NULL;
	}

#ifdef _DEBUG
	std::cout << "\t " << s << ": calling lpf_resize_message_queue, requesting a "
		<< "capacity of " << _maxh << ". Context is " << context << std::endl;
#endif

	// resize LPF buffers accordingly
	lpf_err_t lpfrc = lpf_resize_message_queue( context, _maxh );
	if( lpfrc == LPF_SUCCESS ) {
		lpfrc = lpf_probe( context, &lpf_info );
	}

	// interpret mitigable error codes
	if( lpfrc == LPF_ERR_OUT_OF_MEMORY ) {
		delete[] cur_payload;
		delete[] cur_msgs;
#ifdef _DEBUG
		std::cerr << "Out of memory while initializing (4)\n";
#endif
		return OUTOFMEM;
	}
#ifdef _DEBUG
	std::cout << s << ": calling lpf_resize_memory_register, "
		<< "requesting a capacity of " << _regs << ". "
		<< "Context is " << context << std::endl;
#endif
	lpfrc = lpf_resize_memory_register( context, _regs );
	if( lpfrc == LPF_ERR_OUT_OF_MEMORY ) {
		delete[] cur_payload;
		delete[] cur_msgs;
#ifdef _DEBUG
		std::cerr << "Out of memory while initializing (5)\n";
#endif
		return OUTOFMEM;
	} else if( lpfrc == LPF_SUCCESS ) {
		// activate the new buffer sizes
		lpfrc = lpf_sync( context, LPF_SYNC_DEFAULT );
	}
	// if non-mitigable, return panic
	if( lpfrc != LPF_SUCCESS ) {
		delete[] cur_payload;
		delete[] cur_msgs;
#ifdef _DEBUG
		std::cerr << "Communication layer failed to initialize\n";
#endif
		return PANIC;
	}

	// register the buffer for communication and activate it
	lpfrc = lpf_register_global( context, buffer, buffer_size, &slot );
#ifdef _DEBUG
	std::cout << _s << ": address " << buffer << " (size " << buffer_size << ") "
		<< "binds to slot " << slot << "\n";
#endif
	if( lpfrc == LPF_SUCCESS ) {
		lpfrc = lpf_sync( context, LPF_SYNC_DEFAULT );
	}
	if( lpfrc != LPF_SUCCESS ) {
		delete[] cur_payload;
		delete[] cur_msgs;
#ifdef _DEBUG
		std::cerr << "Failed to initialize global buffer\n";
#endif
		return PANIC;
	}

	// all OK
	lpf_regs = _regs;
	lpf_maxh = _maxh;
	buffer_size = _bufsize;
	signalMemslotTaken();
	return grb::init< _GRB_BSP1D_BACKEND >( 0, 1, NULL );
}

grb::RC grb::internal::BSP1D_Data::destroy() {
	// deregister buffer area
	lpf_err_t lpfrc = lpf_deregister( context, slot );

	// invalidate all fields
	regs_taken = 0;
	s = 0;
	P = 0;
	context = LPF_NONE;
	lpf_regs = 0;
	lpf_maxh = 0;
	buffer_size = 0;
	free( buffer );
	buffer = NULL;
	slot = LPF_INVALID_MEMSLOT;
	payload_size = 0;
	tag_size = 0;
	max_msgs = 0;
	if( cur_payload != NULL ) {
		delete[] cur_payload;
		cur_payload = NULL;
	}
	if( cur_msgs != NULL ) {
		delete[] cur_msgs;
		cur_msgs = NULL;
	}
	queue_status = 0;
	if( queue != LPF_INVALID_BSMP ) {
		if( lpf_bsmp_destroy( &queue ) != LPF_SUCCESS ) {
			return PANIC;
		}
		assert( queue == LPF_INVALID_BSMP );
	}
	destroyed = true;

	// check return status
	if( lpfrc != LPF_SUCCESS ) {
		// this should never happen according to the specs
		return PANIC;
	}
	// done
	return SUCCESS;
}

// The following function was declared inline:
// grb::RC grb::internal::BSP1D_Data::checkBufferSize( const size_t size_in );

grb::RC grb::internal::BSP1D_Data::ensureBufferSize( const size_t size_in ) {
	grb::RC ret = SUCCESS;
#ifdef _DEBUG
	std::cout << "\t " << s << ": ensureBufferSize( " << size_in << " )\n";
#endif

	// check if we can exit early
	if( checkBufferSize( size_in ) == SUCCESS ) {
#ifdef _DEBUG
		std::cout << "\t" << s << ": current capacity suffices\n";
#endif
		return ret;
	} else {
#ifdef _DEBUG
		std::cout << "\t" << s
			<< ": current capacity is insufficient, reallocating global "
			<< "buffer...\n";
#endif
	}

	// whenever this function is called, to have global consistency between
	// memslots, always re-register
	lpf_err_t lpfrc = lpf_deregister( context, slot );
	if( lpfrc != LPF_SUCCESS ) {
#ifdef _DEBUG
		std::cout << "\t" << s << ": could not deregister memory slot "
			<< slot << "\n";
#endif
		return PANIC;
	}

#ifdef _DEBUG
	std::cout << "\t" << s << ": reallocating local buffer\n";
#endif
	// calculate new buffer size
	size_t size = 2 * buffer_size;
	if( size < size_in ) {
		size = size_in;
	}

	// make a new allocation
	void * replacement = nullptr;
	const int prc = posix_memalign(
		&replacement,
		grb::config::CACHE_LINE_SIZE::value(),
		size
	);
	// check for success
	if( prc == ENOMEM ) {
#ifdef _DEBUG
		std::cout << "\t" << s << ": address " << buffer << " "
			<< "(size " << buffer_size << ") binds to slot " << slot << "\n";
#endif
		// return appropriate error code
		ret = OUTOFMEM;
	} else if( prc == 0 ) {
		// set new memory
		free( buffer );
		buffer = replacement;
		buffer_size = size;
	} else {
#ifdef _DEBUG
		std::cerr << "\t" << s << ": unexpected error during posix_memalign call\n";
#endif
		assert( false );
		return PANIC;
	}

	// replace memory slot
	lpfrc = lpf_register_global( context, buffer, buffer_size, &slot );
#ifdef _DEBUG
	std::cout << "\t" << s << ": address " << buffer << " "
		<< "(size " << buffer_size << ") binds to slot " << slot << "\n";
#endif
	if( lpfrc == LPF_SUCCESS ) {
		lpfrc = lpf_sync( context, LPF_SYNC_DEFAULT );
	}
	if( lpfrc != LPF_SUCCESS ) {
#ifdef _DEBUG
		std::cout << "\t" << s << ": could not sync on taking into effect new buffer"
			<< std::endl;
#endif
		return PANIC;
	}

	// done
	return ret;
}

grb::RC grb::internal::BSP1D_Data::ensureMemslotAvailable(
	const size_t count
) {
#ifdef _DEBUG
	std::cout << s << ": call to ensureMemslotAvailable( " << count << " ) "
		<< "started. This must be a collective call with matching arguments.\n";
	std::cout << "\t current number of memslots reserved: " << lpf_regs << ". "
		<< "Number of regs taken: " << regs_taken << ". "
		<< "New slots requested: " << count << std::endl;
#endif
	// dynamic sanity check
	assert( lpf_regs > 0 );
	// catch trivial case
	if( count == 0 ) {
		return SUCCESS;
	}
	// see if we need to do anything
	if( regs_taken + count > lpf_regs ) {
		// standard amortised behaviour
		lpf_regs *= 2;
		// shortcut if initial doubling was not sufficient
		if( regs_taken + count > lpf_regs ) {
			lpf_regs = regs_taken + count;
		}
#ifdef _DEBUG
		std::cout << "Resizing to " << lpf_regs << " memory slots\n";
#endif
		// resize to new length
		lpf_err_t rc = lpf_resize_memory_register( context, lpf_regs );
		// panic on error
		if( rc != LPF_SUCCESS ) {
			// alternative behaviour would be to free some temporary buffer and retry
			//(if it was an out-of-memory error)
			return PANIC;
		}
		// let resize take effect
		rc = lpf_sync( context, LPF_SYNC_DEFAULT );
		// check if sync failed
		if( rc != LPF_SUCCESS ) {
			// a failed sync indicates communication layer has broken down
			return PANIC;
		}
	} else {
#ifdef _DEBUG
		std::cout << "I already had enough space for the requested number of memory "
			<< "slots\n";
#endif
	}
	// done
	return SUCCESS;
}

grb::RC grb::internal::BSP1D_Data::ensureMaxMessages( const size_t hmax ) {
	// see if we need to do anything
	if( hmax <= lpf_maxh ) {
		// then no, return
		return SUCCESS;
	}

	// we need to resize, use standard amortised doubling
	lpf_maxh *= 2;

	// shortcut if initial doubling was not sufficient
	if( hmax > lpf_maxh ) {
		lpf_maxh = hmax;
	}

	// resize to new length
	lpf_err_t rc = lpf_resize_message_queue( context, lpf_maxh );
	if( rc != LPF_SUCCESS ) {
		// alternative behaviour would be to free some temporary buffer and retry
		//(if it was an out-of-memory error)
		return PANIC;
	}

	// let resize take effect
	rc = lpf_sync( context, LPF_SYNC_DEFAULT );
	if( rc != LPF_SUCCESS ) {
		return PANIC;
	}

	// done
	return SUCCESS;
}

grb::RC grb::internal::BSP1D_Data::ensureBSMPCapacity(
	const size_t ts, const size_t ps,
	const size_t nm
) {
	// check if we need to do nothing
	if( ts <= tag_size && ps <= payload_size && nm <= max_msgs ) {
		return SUCCESS;
	}

	// delete any old queue
	if( queue != LPF_INVALID_BSMP ) {
		if( lpf_bsmp_destroy( &queue ) != LPF_SUCCESS ) {
			return PANIC;
		}
	} else {
		// if this a new queue, reserve enough BSP buffer fot creating an new one
		if( ensureMemslotAvailable( 4 ) != SUCCESS ) {
			return PANIC;
		}
		// assume lpf_bsmp_create is successful, in which case four registrations are
		// taken successfully. If create is unsuccessful, then PANIC will be returned
		// which means the state of this library becomes undefined anyway.
		regs_taken += 4;
	}

	// update message queue parameters
	if( ts > tag_size ) {
		tag_size = ts;
	}
	if( ps > 0 && payload_size == 0 ) {
		payload_size = ps;
	}
	if( nm > 0 && max_msgs == 0 ) {
		max_msgs = nm;
	}
	while( ps <= payload_size ) {
		payload_size *= 2;
	}
	while( nm <= max_msgs ) {
		max_msgs *= 2;
	}

	// create new buffer
	if( lpf_bsmp_create( context,
			s, P,
			payload_size, tag_size,
			max_msgs, &queue
		) != LPF_SUCCESS
	) {
		return PANIC;
	}

	// done
	return SUCCESS;
}

void grb::internal::BSP1D_Data::signalMemslotTaken( const unsigned int count ) {
	regs_taken += count;
	assert( regs_taken <= lpf_regs );
}

/** Decrements \a regs_taken by one. */
void grb::internal::BSP1D_Data::signalMemslotReleased(
	const unsigned int count
) {
#ifdef _DEBUG
	std::cout << "signalMemslotReleased( " << count << " ) requested "
		<< "while regs_taken is " << regs_taken << " / " << lpf_regs << std::endl;
#endif
	assert( regs_taken <= lpf_regs );
	if( count == 0 ) {
		return;
	}
	assert( regs_taken >= count );
	regs_taken -= count;
}

