
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

//header needed for 'finalize< _GRB_BSP1D_BACKEND >();'
#include <graphblas/init.hpp>

#include <graphblas/bsp/config.hpp>
#include <graphblas/bsp1d/init.hpp>


grb::utils::ThreadLocalStorage< grb::internal::BSP1D_Data >
grb::internal::grb_BSP1D;

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
	return data.initialize(
		ctx,
		static_cast< lpf_pid_t >(s),
		static_cast< lpf_pid_t >(P),
		grb::config::LPF::MEMSLOT_CAPACITY_DEFAULT,
		grb::config::LPF::MAX_H_RELATION_DEFAULT
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
			std::cerr << "\t process " << s << " is finalising" << std::endl;
		}
		if( lpf_sync( data.context, LPF_SYNC_DEFAULT )
			!= LPF_SUCCESS
		) {
			std::cerr << "\t process " << s << " failed to sync" << std::endl;
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

grb::RC grb::internal::BSP1D_Data::initialize(
	lpf_t _context,
	const lpf_pid_t _s, const lpf_pid_t _P,
	const size_t _regs, const size_t _maxh
) {
	// required buffer for an all-to-all on #messages followed by a prefix-sum

	// also dubs as the required buffer sufficient for all-reducing any
	// fundamental raw data type.

	// To guarantee all-reducing user-defined types is possible, a grb::Scalar
	// must be introduced(!)
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
	assert( _context != nullptr );
	assert( _regs > 0 );

	// initialise fields
	regs_taken = 0;
	coll_call_capacity = 0;
	coll_reduction_bsize = 0;
	coll_other_bsize = 0;
	lpf_info = LPF_INVALID_MACHINE;
	s = _s;
	P = _P;
	context = _context;
	lpf_regs = 0;
	lpf_maxh = 0;
	buffer_size = 0;
	buffer = nullptr;
	slot = LPF_INVALID_MEMSLOT;
	payload_size = 0;
	tag_size = 0;
	max_msgs = 0;
	queue_status = 0;
	queue = LPF_INVALID_BSMP;
	coll = LPF_INVALID_COLL;
	destroyed = false;
	// end default initialisers

	// first try allocations and initialisations for BSMP
	try {
		cur_payload.resize( P, 0 );
		cur_msgs.resize( P, 0 );
	} catch( std::bad_alloc &ex ) {
		std::cerr << "\tError during BSP1D::init: "
			<< "out of memory while initialising BSMP meta-data:\n"
			<< "\t\t" << ex.what() << "\n";
		return OUTOFMEM;
	}

	// now make allocation requests of LPF:

	// first, the machine info
	lpf_err_t lpfrc = lpf_probe( context, &lpf_info );
	if( lpfrc == LPF_ERR_OUT_OF_MEMORY ) {
		std::cerr << "\tError during BSP1D::init: "
			"Out of memory while initialising lpf_machine_t\n";
		return OUTOFMEM;
	} else if( lpfrc != LPF_SUCCESS ) {
		/* LCOV_EXCL_START */
#ifndef NDEBUG
		const bool lpf_probe_spec_says_this_should_never_occur = false;
		assert( lpf_probe_spec_says_this_should_never_occur );
#endif
		return PANIC;
		/* LCOV_EXCL_STOP */
	}

	// resize internal LPF buffers:
	//  - the message buffer
	size_t maxh;
	{
		// make sure we allocate at least enough buffer for a broadcast
		const size_t max_msgs_for_bcast = P > 1
			? std::max( P + 1, 2 * P - 3 )
			: P + 1;
		maxh = max_msgs_for_bcast > _maxh
			? max_msgs_for_bcast
			: _maxh;
	}
#ifdef _DEBUG
	std::cout << "\t " << s << ": calling lpf_resize_message_queue, requesting a "
		<< "capacity of " << maxh << ". Context is " << context << std::endl;
#endif
	assert( lpfrc == LPF_SUCCESS );
	lpfrc = lpf_resize_message_queue( context, maxh );
	if( lpfrc == LPF_ERR_OUT_OF_MEMORY ) {
		std::cerr << "\tError during BSP1D::init: "
			<< "Out of memory while initializing LPF message queue\n";
		return OUTOFMEM;
	}

	// - the memory slot buffer
#ifdef _DEBUG
	std::cout << s << ": calling lpf_resize_memory_register, requesting a "
		<< "capacity of " << _regs << ". Context is " << context << std::endl;
#endif
	// note: we ask the requested number of memory slots for a capacity, plus one
	//       in order to initialise the LPF collectives.
	lpfrc = lpf_resize_memory_register( context, _regs + 1 );
	if( lpfrc == LPF_ERR_OUT_OF_MEMORY ) {
		std::cerr << "\tError during BSP1D::init: "
			<< "out of memory while initializing memory slot buffer\n";
		return OUTOFMEM;
	}

	// activation of requested LPF buffers is required before we can initialise
	// LPF libraries and register our own buffer for communication
	if( lpfrc == LPF_SUCCESS ) {
		// activate the new buffer sizes
		lpfrc = lpf_sync( context, LPF_SYNC_DEFAULT );
	}
	if( lpfrc != LPF_SUCCESS ) {
		// we exclude this block from coverage since the only specified possible
		// return type is PANIC, which can't really be tested for
		/* LCOV_EXCL_START */
#ifndef NDEBUG
		if( lpfrc != LPF_ERR_FATAL ) {
			const bool spec_says_this_should_never_occur = false;
			assert( spec_says_this_should_never_occur );
		}
#endif
		// any error at this point cannot be mitigated by ALP; return panic
#ifdef _DEBUG
		std::cerr << "Communication layer failed to initialize\n";
#endif
		return PANIC;
		/* LCOV_EXCL_STOP */
	}

	// initialise the LPF collectives
	assert( lpfrc == LPF_SUCCESS );
	coll_call_capacity = grb::config::LPF::COLL_CALL_CAPACITY_DEFAULT;
	coll_reduction_bsize = grb::config::LPF::COLL_REDUCTION_BSIZE_DEFAULT;
	coll_other_bsize = grb::config::LPF::COLL_OTHER_BSIZE_DEFAULT;
	lpfrc = lpf_collectives_init(
		context,
		s, P,
		coll_call_capacity,
		coll_reduction_bsize,
		coll_other_bsize,
		&coll
	);
	if( lpfrc != LPF_SUCCESS ) {
		if( lpfrc != LPF_ERR_OUT_OF_MEMORY ) {
			/* LCOV_EXCL_START */
			std::cerr << "Unexpected error code from LPF, please submit a bug report\n";
#ifndef NDEBUG
			const bool spec_says_this_should_never_occur = false;
			assert( spec_says_this_should_never_occur );
#endif
			return PANIC;
			/* LCOV_EXCL_STOP */
		}
		std::cerr << "\tError during BSP1D::init: "
			<< "out of memory while initialising LPF collectives\n";
		return OUTOFMEM;
	}

	// initialise the buffer
	if( _bufsize > 0 ) {
		// try and allocate the buffer
		// note: the below always uses local allocation (and rightly so)
		const int prc = posix_memalign(
			&buffer,
			grb::config::CACHE_LINE_SIZE::value(),
			_bufsize
		);
		if( prc != 0 ) {
			std::cerr << "\tError during BSP1D::init: "
				<< "out of memory while initialising global buffer\n";
			return OUTOFMEM;
		}
	} else { assert( buffer == nullptr ); }

	// register the buffer for communication
	assert( lpfrc == LPF_SUCCESS );
	lpfrc = lpf_register_global( context, buffer, _bufsize, &slot );
#ifdef _DEBUG
	std::cout << "\tBSP::init (" << _s << "): address " << buffer << " (size "
		<< _bufsize << ") " << "binds to slot " << slot << "\n";
#endif
	assert( lpfrc == LPF_SUCCESS );
	if( lpfrc != LPF_SUCCESS ) {
		/* LCOV_EXCL_START */
#ifndef NDEBUG
		const bool lpf_reg_global_spec_says_this_should_never_occur = false;
		assert( lpf_reg_global_spec_says_this_should_never_occur );
#endif
		return PANIC;
		/* LCOV_EXCL_STOP */
	}

	// activate our buffer registration as well as the collectives instance
	lpfrc = lpf_sync( context, LPF_SYNC_DEFAULT );
	if( lpfrc != LPF_SUCCESS ) {
		/* LCOV_EXCL_START */
		if( buffer != nullptr ) { free( buffer ); }
		(void) lpf_collectives_destroy( coll );
		(void) lpf_deregister( context, slot );
#ifdef _DEBUG
		std::cerr << "Failed to activate global buffer and LPF collectives\n";
#endif
		return PANIC;
		/* LCOV_EXCL_STOP */
	}

	// all OK
	lpf_regs = _regs;
	lpf_maxh = maxh;
	buffer_size = _bufsize;
	signalMemslotTaken( 2 ); // one for the buffer, another for the collectives
	return grb::init< _GRB_BSP1D_BACKEND >( 0, 1, nullptr );
}

grb::RC grb::internal::BSP1D_Data::destroy() {
	grb::RC ret = SUCCESS;

	// first clean up everything that needs the LPF context
	if( slot != LPF_INVALID_MEMSLOT ) {
		if( lpf_deregister( context, slot ) != LPF_SUCCESS ) {
			// this should never happen according to the LPF specs
			/* LCOV_EXCL_START */
#ifndef NDEBUG
			const bool lpf_specs_violated = false;
			assert( lpf_specs_violated );
#endif
			ret = PANIC;
			/* LCOV_EXCL_STOP */
		}
		slot = LPF_INVALID_MEMSLOT;
	}
	if( queue != LPF_INVALID_BSMP ) {
		if( lpf_bsmp_destroy( &queue ) != LPF_SUCCESS ) {
			std::cerr << "Error while destroying LPF BSMP queue\n";
			ret = PANIC;
		}
		assert( queue == LPF_INVALID_BSMP );
	}
	if( lpf_collectives_destroy( coll ) != LPF_SUCCESS ) {
		std::cerr << "Error while destroying LPF collectives instance\n";
		ret = PANIC;
	}
	coll = LPF_INVALID_COLL;

	// then clean up the rest, in order of declaration in bsp1d/init.hpp:
	regs_taken = 0;
	coll_call_capacity = 0;
	coll_reduction_bsize = 0;
	coll_other_bsize = 0;
	lpf_info = LPF_INVALID_MACHINE;
	s = 0;
	P = 0;
	context = LPF_NONE;
	lpf_regs = 0;
	lpf_maxh = 0;
	buffer_size = 0;
	if( buffer != nullptr ) {
		free( buffer );
		buffer = nullptr;
	}
	payload_size = 0;
	tag_size = 0;
	max_msgs = 0;
	cur_payload.clear();
	cur_msgs.clear();
	queue_status = 0;
	put_requests.clear();
	get_requests.clear();
	destroyed = true;
	mapper.clear();

	// done
	return ret;
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
		// this should never happen according to the LPF spec
		/* LCOV_EXCL_START */
#ifndef NDEBUG
		const bool lpf_spec_violated = false;
		assert( lpf_spec_violated );
#endif
#ifdef _DEBUG
		std::cout << "\t" << s << ": could not deregister memory slot "
			<< slot << "\n";
#endif
		return PANIC;
		/* LCOV_EXCL_STOP */
	}

#ifdef _DEBUG
	std::cout << "\t" << s << ": reallocating local buffer\n";
#endif
	// calculate new buffer size
	const size_t size = std::max( 2 * buffer_size, size_in );

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
		/* LCOV_EXCL_START */
#ifdef _DEBUG
		std::cerr << "\t" << s << ": unexpected error during posix_memalign call\n";
#endif
#ifndef NDEBUG
		const bool unhandled_posix_memalign_error = false;
		assert( unhandled_posix_memalign_error );
#endif
		return PANIC;
		/* LCOV_EXCL_STOP */
	}

	// replace memory slot
	lpfrc = lpf_register_global( context, buffer, buffer_size, &slot );
#ifdef _DEBUG
	std::cout << "\tBSP::ensureBufferSize (" << s << "): address " << buffer << " "
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
	if( regs_taken + count + 1 > lpf_regs ) {
		// standard amortised behaviour
		lpf_regs *= 2;
		// shortcut if initial doubling was not sufficient
		if( regs_taken + count + 1 > lpf_regs ) {
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
		tag_size = std::max( 2 * tag_size, ts );
	}
	if( ps > payload_size ) {
		payload_size = std::max( 2 * payload_size, ps );
	}
	if( nm > max_msgs ) {
		max_msgs = std::max( 2 * max_msgs, nm );
	}

	// create new buffer
	if( lpf_bsmp_create(
			context,
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

grb::RC grb::internal::BSP1D_Data::ensureCollectivesCapacity(
	const size_t requested_call_capacity,
	const size_t requested_reduction_bsize,
	const size_t requested_other_bsize
) {
	const bool need_reinitialisation =
		requested_call_capacity > coll_call_capacity ||
		requested_reduction_bsize > coll_reduction_bsize ||
		requested_other_bsize > coll_other_bsize;
	if( !need_reinitialisation ) { return SUCCESS; }

	lpf_err_t lpf_rc = lpf_collectives_destroy( coll );
	if( lpf_rc == LPF_SUCCESS ) {
		const size_t target_call_capacity =
			requested_call_capacity > coll_call_capacity
				? std::max( 2 * coll_call_capacity, requested_call_capacity )
				: coll_call_capacity;
		const size_t target_reduction_bsize =
			requested_reduction_bsize > coll_reduction_bsize
				? std::max( 2 * coll_reduction_bsize, requested_reduction_bsize )
				: coll_reduction_bsize;
		const size_t target_other_bsize =
			requested_other_bsize > coll_other_bsize
				? std::max( 2 * coll_other_bsize, requested_other_bsize )
				: coll_other_bsize;
		lpf_rc = lpf_collectives_init(
			context,
			s, P,
			target_call_capacity,
			target_reduction_bsize,
			target_other_bsize,
			&coll
		);
		if( lpf_rc == LPF_SUCCESS ) {
			coll_call_capacity = target_call_capacity;
			coll_reduction_bsize = target_reduction_bsize;
			coll_other_bsize = target_other_bsize;
		} else if( lpf_rc == LPF_ERR_OUT_OF_MEMORY ) {
#ifdef _DEBUG
			std::cout << "BSP1D::ensureCollectivesCapacity: requested capacity goes out "
				<< "of memory\n";
#endif
			return OUTOFMEM;
		}
	}

	if( lpf_rc != LPF_SUCCESS ) {
		// This should never happen according to the LPF specs
		/* LCOV_EXCL_START */
#ifndef NDEBUG
		const bool lpf_coll_can_only_succeed_or_oom = false;
		assert( lpf_coll_can_only_succeed_or_oom );
#endif
		std::cerr << "Error during call to BSP1D_Data::ensureCollectivesCapacity\n";
		return PANIC;
		/* LCOV_EXCL_STOP */
	}

	// done
#ifdef _DEBUG
	std::cout << "BSP1D::ensureCollectivesCapacity, current capacities:\n"
		<< "\tnumber of collective calls supported: " << coll_call_capacity << "\n"
		<< "\treduction byte size supported:        " << coll_reduction_bsize << "\n"
		<< "\tother byte size supported:            " << coll_other_bsize << "\n";
#endif
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

