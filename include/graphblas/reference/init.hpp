
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
 * @date 2nd of February, 2017
 */

#if !defined _H_GRB_REFERENCE_INIT || defined _H_GRB_REFERENCE_OMP_INIT
#define _H_GRB_REFERENCE_INIT

#include <new>
#include <algorithm>

#include <graphblas/base/init.hpp>
#include <graphblas/utils/DMapper.hpp>


namespace grb {

	namespace internal {

#ifndef _H_GRB_REFERENCE_OMP_INIT
		// these are all the global fields for the reference backend

		/** \internal Used for generating deterministic IDs. */
		extern grb::utils::DMapper< uintptr_t > reference_mapper;

		/** \internal Shared buffer */
		extern char * reference_buffer;

		/** \internal Shared buffer size */
		extern size_t reference_bufsize;

		/**
		 * \internal
		 * Helper function to get the size of the current buffer.
		 *
		 * @tparam D      the data type to store in the buffer
		 * @return size_t the number of elements of type #D that can be stored
		 *                in the current buffer
		 *
		 * \warning This function should never be used as a conditional to decide
		 *          when to resize the global buffer: for this, use the
		 *            -# #ensureBufferSize()
		 *          function instead. This function is only intended for deciding
		 *          when a larger buffer exists to use any extra space should that
		 *          indeed be available.
		 */
		template< typename D >
		size_t getCurrentBufferSize() noexcept {
			return reference_bufsize / sizeof( D );
		}

		/**
		 * \internal
		 * Helper function that ensures a given size is available.
		 *
		 * @tparam D The buffer element type desired.
		 *
		 * @param[in] n The desired number of elements of type \a D.
		 *
		 * This implementation uses recursive doubling.
		 *
		 * @returns true  if the requested size is available.
		 * @returns false if allocation for the requested buffers size has failed.
		 * \endinternal
		 */
		template< typename D >
		bool ensureReferenceBufsize( const size_t n ) noexcept {
			const size_t targetSize = n * sizeof( D );
			if( reference_bufsize < targetSize ) {
				size_t newSize = std::max( 2 * reference_bufsize, targetSize );
				if( reference_bufsize > 0 ) {
					delete[] reference_buffer;
				}
				reference_buffer = new( std::nothrow ) char[ newSize ];
				if( reference_buffer == nullptr ) {
					reference_bufsize = 0;
					return false;
				} else {
					reference_bufsize = newSize;
					return true;
				}
			} else {
				return true;
			}
		}

		/**
		 * \internal
		 * Gets a buffer of the requested size iff the requested buffer does not
		 * exceed the available buffer size.
		 *
		 * @tparam D The buffer element type desired.
		 *
		 * @param[in] n The desired number of elements of type \a D.
		 *
		 * @returns An array of type \a D of size \a n.
		 *
		 * @see ensureReferenceBufsize.
		 * \endinternal
		 */
		template< typename D >
		D * getReferenceBuffer( const size_t n ) {
			assert( n * sizeof( D ) <= reference_bufsize );
 #ifdef NDEBUG
			(void)n;
 #endif
			return reinterpret_cast< D * >( reference_buffer );
		}
#else
		/** \internal A small shared buffer for index offset computations. */
		extern size_t * __restrict__ privateSizetOMP;
#endif
	} // namespace internal

	/**
	 * This function completes in \f$ \Theta(1) \f$, moves \f$ \Theta(1) \f$ data,
	 * does not allocate nor free any memory, and does not make any system calls.
	 *
	 * This implementation does not support multiple user processes.
	 *
	 * @see grb::init for the user-level specification.
	 *
	 * \warning This primitive has been deprecated since version 0.5. Please update
	 *          your code to use the grb::Launcher instead.
	 */
	template<>
	RC init< reference >( const size_t, const size_t, void * const );

#ifdef _H_GRB_REFERENCE_OMP_INIT
	/**
	 * Gets a private unsigned integer of type <tt>size_t</tt>.
	 *
	 * This is a thread-safe function.
	 *
	 * @returns A private modifiable <tt>size_t</tt>.
	 *
	 * \warning This value may only be called when using the #reference_omp
	 *          back-end. Any other use will incur UB!
	 */
	inline size_t & getPrivateSizet() {
		return internal::privateSizetOMP[ omp_get_thread_num() * grb::config::CACHE_LINE_SIZE::value() ];
	}

	/**
	 * Allows reading remote integers normally accessed privately using
	 * #getPrivateSizet. This is not a thread-safe function.
	 *
	 * @param[in] i Which thread's private integer to access. This argument must
	 *              not exceed the total number of available threads.
	 *
	 * @returns The requested value.
	 *
	 * \warning This value may only be called when using the #reference_omp
	 *          back-end. Any other use will incur UB!
	 */
	inline const size_t & readRemoteSizet( const size_t & i ) {
		return internal::privateSizetOMP[ i * grb::config::CACHE_LINE_SIZE::value() ];
	}
#endif

	/**
	 * This function completes in \f$ \Theta(1) \f$, moves \f$ \Theta(1) \f$ data,
	 * does not allocate nor free any memory, and does not make any system calls.
	 *
	 * @see grb::finalize() for the user-level specification.
	 *
	 * \warning This primitive has been deprecated since version 0.5. Please update
	 *          your code to use the grb::Launcher instead.
	 */
	template<>
	RC finalize< reference >();

} // namespace grb

#ifdef _GRB_WITH_OMP
 #ifndef _H_GRB_REFERENCE_OMP_INIT
  #define _H_GRB_REFERENCE_OMP_INIT
  #define reference reference_omp
  #include <omp.h>
  #include "init.hpp"
  #undef reference
  #undef _H_GRB_REFERENCE_OMP_INIT
 #endif
#endif

#endif //``end _H_GRB_REFERENCE_INIT''

