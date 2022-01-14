
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

#if ! defined _H_GRB_REFERENCE_INIT || defined _H_GRB_REFERENCE_OMP_INIT
#define _H_GRB_REFERENCE_INIT

#include <graphblas/base/init.hpp>

namespace grb {

	namespace internal {
#ifndef _H_GRB_REFERENCE_OMP_INIT
		extern char * reference_buffer;
		extern size_t reference_bufsize;
		template< typename D >
		bool ensureReferenceBufsize( const size_t n ) {
			const size_t targetSize = n * sizeof( D );
			if( reference_bufsize < targetSize ) {
				size_t newSize = 2 * reference_bufsize;
				if( newSize < targetSize ) {
					newSize = targetSize;
				}
				if( reference_bufsize > 0 ) {
					delete[] reference_buffer;
				}
				reference_buffer = new char[ newSize ];
				if( reference_buffer == NULL ) {
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
		template< typename D >
		D * getReferenceBuffer( const size_t n ) {
			assert( n * sizeof( D ) <= reference_bufsize );
#ifdef NDEBUG
			(void)n;
#endif
			return reinterpret_cast< D * >( reference_buffer );
		}
#else
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

