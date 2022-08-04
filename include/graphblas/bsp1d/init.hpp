
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

#ifndef _H_GRB_BSP1D_INIT
#define _H_GRB_BSP1D_INIT

#include <vector> //queue of HP put and get requests

#include "config.hpp"

#include <graphblas/base/init.hpp>
#include <graphblas/rc.hpp>

#include <graphblas/utils/DMapper.hpp>
#include <graphblas/utils/ThreadLocalStorage.hpp>

#include <assert.h> //assertions
#include <lpf/bsmp.h>
#include <lpf/core.h>


namespace grb {

	namespace internal {

		/** All information corresponding to a get request. */
		struct get_request {
			lpf_pid_t src_pid;
			lpf_memslot_t src;
			size_t src_offset;
			void * dst;
			size_t size;
		};

		/** All information corresponding to a put request. */
		struct put_request {
			void * src;
			lpf_pid_t dst_pid;
			lpf_memslot_t dst;
			size_t dst_offset;
			size_t size;
		};

		/**
		 * These are all user process local data elements required to successfully
		 * execute parallel GraphBLAS calls.
		 *
		 * \warning For compatibility with POSIX Threads thread-local storage, this
		 *          class should be a struct in the sense that it should \em not
		 *          rely on constructors setting \a const private values, for
		 *          instance.
		 */
		class BSP1D_Data {

		private:

			/** Number of slots taken */
			size_t regs_taken;

			/** Information on the current environment. */
			lpf_machine_t lpf_info;


		public:

			/** The user process ID. */
			lpf_pid_t s;

			/** The number of user processes. */
			lpf_pid_t P;

			/** The current LPF context. */
			lpf_t context;

			/** Number of memory areas registered by the GraphBLAS. */
			size_t lpf_regs;

			/** Maximum possible h-relation requested by the GraphBLAS. */
			size_t lpf_maxh;

			/** The size, in bytes, of the \a buffer memory area. */
			size_t buffer_size;

			/** Local buffer that may be used by the GraphBLAS. */
			void * buffer;

			/** Memory slot related to the \a buffer. */
			lpf_memslot_t slot;

			/**
			 * The maximum combined payload size (in bytes) of the BSMP buffer.
			 *
			 * @see queue.
			 */
			size_t payload_size;

			/**
			 * The tag size per BSMP message.
			 *
			 * @see queue.
			 */
			size_t tag_size;

			/**
			 * The max number of messages that can be sent or received during a single
			 * BSMP epoch.
			 *
			 * @see queue.
			 */
			size_t max_msgs;

			/**
			 * The current number of bytes being sent to a sibling process.
			 *
			 * This array should be of size \a P after successful initialisation.
			 *
			 * @see queue.
			 */
			size_t * cur_payload;

			/**
			 * The current number of messages sent to a sibling process.
			 *
			 * This array should be of size \a P after successful initialisation.
			 *
			 * @see queue.
			 */
			size_t * cur_msgs;

			/**
			 * The status of the BSMP #queue.
			 *
			 * 0: not allocated.
			 * 1: allocated and in write mode.
			 * 2: allocated and in read mode.
			 * 3: resize requested (first time).
			 * 4: resize requested (after toggle); only used if there is no difference
			 *    between read and write mode, from the user perspective.
			 */
			unsigned int queue_status;

			/** Queue of put requests. */
			std::vector< struct put_request > put_requests;

			/** Queue of get requests. */
			std::vector< struct get_request > get_requests;

			/** A BSMP message queue. */
			lpf_bsmp_t queue;

			/** Whether a finalize has been called. */
			bool destroyed;

			/** Mapper to assign IDs to BSP1D containers .*/
			utils::DMapper< uintptr_t > mapper;

			/**
			 * Initialises all fields.
			 *
			 * @return grb::SUCCESS  On successful initialisation of this structure.
			 * @return grb::OUTOFMEM When \a buffer of size \a _bufsize could not be
			 *                       allocated.
			 * @return grb::PANIC    When an error occurs that leaves this library in
			 *                       an undefined state.
			 */
			RC initialize( lpf_t _context, const lpf_pid_t _s, const lpf_pid_t _P, const size_t _regs, const size_t _maxh, const size_t _bufsize );

			/**
			 * Frees all allocated resources and sets the \a destroyed flag to \a true.
			 *
			 * @return grb::SUCCESS When all resources were freed without error.
			 * @return grb::PANIC   Upon error of the underlying communication layer.
			 */
			RC destroy();

			/**
			 * Initialises all fields.
			 *
			 * Alias to the full \a initialize() function where \a _bufsize is set to
			 * \a _P times the size (in bytes) of a \a double.
			 */
			RC initialize( lpf_t _context, const lpf_pid_t _s, const lpf_pid_t _P, const size_t _regs, const size_t _maxh );

			/**
			 * Ensures the buffer is at least of the given size.
			 *
			 * @param[in] size The requested buffer capacity in bytes.
			 *
			 * At function exit, if grb::SUCCESS is returned, the internal \a buffer is
			 * of <em>at least</em> the requested \a size. The buffer memory area is
			 * registered globally.
			 *
			 * \warning The contents of the buffer are undefined after a call to this
			 *          function, unless grb::OUTOFMEM is returned (in which case the
			 *          buffer contents are left unchanged).
			 *
			 * @return grb::OUTOFMEM When there is not enough memory available to
			 *                       complete this call successfully. On function exit,
			 *                       it will be as though the call to this function
			 *                       never took place.
			 * @return grb::SUCCESS  When the call to this function completes
			 *                       successfully.
			 * @return grb::PANIC    When an error occurs that leaves this library in
			 *                       an undefined state.
			 */
			RC ensureBufferSize( const size_t size );

			/**
			 * Like #ensureBufferSize, but doesn't enlarge the buffer.
			 *
			 * @returns #SUCCESS If the buffer size is large enough.
			 * @returns #PANIC otherwise.
			 */
			inline RC checkBufferSize( const size_t size ) {
#ifdef _DEBUG
				std::cout << "checkBufferSize: current size is " << buffer_size << ", requested size is " << size << " (bytes)" << std::endl;
#endif
				if( buffer_size < size ) {
					return grb::PANIC;
				} else {
					return grb::SUCCESS;
				}
			}

			/**
			 * Ensures the GraphBLAS has reserved enough LPF buffer space to execute
			 * its requested communications. This may involve resizing the LPF buffers.
			 *
			 * If a lower capacity is requested than currently available, this function
			 * may not reduce the already-allocated buffer sizes.
			 *
			 * @param[in] count The number of additional memory slots that are expected
			 *                  to be needed. Default value is 1.
			 *
			 * Passing a zero value for \a count will return grb::SUCCESS immediately.
			 *
			 * @return SUCCESS Enough space is available for another \a lpf_memslot_t
			 *                 to be registered.
			 * @return PANIC   Could not ensure a large enough buffer space. The state
			 *                 of the library has become undefined.
			 */
			RC ensureMemslotAvailable( const size_t count = 1 );

			/**
			 * Ensures the GraphBLAS has reserved enough LPF buffer space to execute
			 * its requested communications. This may involve resizing the LPF buffers.
			 *
			 * @param[in] hmax The maximum number of ingoing and outgoing messages
			 *                 during any superstep from now.
			 *
			 * @return SUCCESS Enoguh space is available for the requested
			 *                 communication pattern.
			 * @return PANIC   Could not ensure a large enough buffer space. The state
			 *                 of the library has become undefined.
			 */
			RC ensureMaxMessages( const size_t hmax );

			/**
			 * Ensures the GraphBLAS has reserved enough BSMP buffer space to execute
			 * its requested one-sided message passing communications. This may involve
			 * resizing the LPF buffers and may involve resizing the BSMP buffers.
			 *
			 * If a lower capacity is requested than currently available, this function
			 * may not reduce the already-allocated buffer sizes.
			 *
			 * \warning Any current contents of the BSMP queue may be deleted.
			 *
			 * @param[in] tag_size     The total number of bytes reserved for the tag
			 *                         of every message sent or received using this
			 *                         queue.
			 * @param[in] payload_size The total combined buffer size (in bytes) for
			 *                         incoming BSMP messages. This is an optional
			 *                         parameter. (The default is 0.)
			 * @param[in] num_messages The total number of times \a lpf_send or
			 *                         \a lpf_move may be called in between calls to
			 *                         \a lpf_bsmp_sync. This is an optional parameter.
			 *                         (The default is 0.)
			 *
			 * @return SUCCESS Enough space is available for another \a lpf_memslot_t
			 *                 to be registered.
			 * @return PANIC   Could not ensure a large enough buffer space. The state
			 *                 of the library has become undefined.
			 */
			RC ensureBSMPCapacity( const size_t tag_size, const size_t payload_size = 0, const size_t num_messages = 0 );

			/**
			 * Increments \a regs_taken.
			 *
			 * @param[in] count (Optional) The number of memslots that should be added
			 *                  to \a regs_taken. Default is 1. Passing zero will turn
			 *                  a call to this function into a no-op.
			 */
			void signalMemslotTaken( const unsigned int count = 1 );

			/**
			 * Decrements \a regs_taken.
			 *
			 * @param[in] count (Optional) The number of memslots that should be
			 *                  substracted from \a regs_taken. Default is 1. Passing
			 *                  zero will turn a call to this function into a no-op.
			 */
			void signalMemslotReleased( const unsigned int count = 1 );

			/**
			 * @tparam T A type the user expects the buffer to have elements of.
			 *
			 * @return A pointer to the internal buffer, interpreted as a restricted
			 *         pointer to the user-given type \a T.
			 */
			template< typename T >
			inline T * getBuffer( const size_t offset = 0 ) {
				return reinterpret_cast< T * __restrict__ >( static_cast< char * >( buffer ) + offset );
			}

			/**
			 * Allows inspection of the message gap of the underlying BSP machine.
			 *
			 * @param[in] message_size The minimum message size assumed active during
			 *                         a full-deplex all-to-all communication.
			 *
			 * @returns The cost of sending a byte, in seconds per byte, during a full-
			 *          duplex all-to-all communication.
			 *
			 * @note In the intended use of this number, it is the ratio between network
			 *       speed, #stream_memspeed and #random_access_memspeed that matters.
			 *       The latter two speeds are currently hardcoded.  While untested, it
			 *       is reasonable to think the ratios do not change too much between
			 *       architectures. Nevertheless, for best results, the hardcoded
			 *       numbers are best benchmarked on the deployment hardware.
			 */
			inline double getMessageGap( const size_t message_size ) const noexcept {
				return lpf_info.g( P, message_size, LPF_SYNC_DEFAULT );
			}

			/**
			 * Allows inspection of the latency of the underlying LPF machine.
			 *
			 * @param[in] message_size The minimum message size assumed active during
			 *                         a full-deplex all-to-all communication.
			 *
			 * @returns The number of seconds it takes to start up a full-duplex
			 *          all-to-all communication.
			 */
			inline double getLatency( const size_t message_size ) const noexcept {
				return lpf_info.l( P, message_size, LPF_SYNC_DEFAULT );
			}
		};

		/**
		 * This global variable stores the thread-local data required by this
		 * GraphBLAS implementation.
		 */
		extern grb::utils::ThreadLocalStorage< grb::internal::BSP1D_Data > grb_BSP1D;

	} // namespace internal

	/**
	 * This implementation expects the \a lpf_t value to be passed as the third
	 * argument to this function. This value should correspond to the parallel
	 * context described by \a s and \a P; the LPF process ID and the user process
	 * ID must match, or undefined behaviour will occur.
	 *
	 * The BSP1D implementation relies on a backend. This backend is assumed to use
	 * a single user process, i.e. meaning that all threading is transparent to the
	 * user. The backend is set at compile time via the \a _GRB_BSP1D_BACKEND flag.
	 *
	 * Implementation notes: casts the parameters \a s and \a P to \a lpf_pid_t.
	 *                       No overflow checking is performed. The complexity of
	 *                       the grb::init depends on the complexity of the
	 *                       corresponding call to the lpf_hook.
	 *
	 * @see grb::init() for the user-level specification.
	 *
	 * \warning This primitive has been deprecated since version 0.5. Please update
	 *          your code to use the grb::Launcher instead.
	 */
	template<>
	grb::RC init< BSP1D >( const size_t s, const size_t P, const lpf_t lpf );

	/**
	 * This implementation employs this function to free and deregister buffers.
	 *
	 * @see grb::finalize() for the user-level specification.
	 *
	 * \warning This primitive has been deprecated since version 0.5. Please update
	 *          your code to use the grb::Launcher instead.
	 */
	template<>
	grb::RC finalize< BSP1D >();

} // namespace grb

#endif

