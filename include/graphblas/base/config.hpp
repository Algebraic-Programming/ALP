
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
 * Defines both configuration parameters effective for all backends, as
 * well as defines structured ways of passing backend-specific parameters.
 *
 * @author A. N. Yzelman
 * @date 8th of August, 2016
 */

#ifndef _H_GRB_CONFIG_BASE
#define _H_GRB_CONFIG_BASE

#include <cstddef> //size_t
#include <string>

#include <assert.h>
#include <unistd.h> //sysconf
#include <cstdint> // for uintptr_t

#include <graphblas/backends.hpp>

#ifndef _GRB_NO_STDIO
 #include <iostream> //std::cout
#endif

// if the user did not define _GRB_BACKEND, set it to the default sequential
// implementation
#ifndef _GRB_BACKEND
 #define _GRB_BACKEND reference
#endif


namespace grb {

	/**
	 * Compile-time configuration constants as well as implementation details that
	 * are derived from such settings.
	 */
	namespace config {

		/**
		 * \defgroup config Configuration
		 *
		 * This module collects all configuration settings.
		 */

		/**
		 * \defgroup commonConfig Common configuration settings
		 * \ingroup config
		 *
		 * Configuration elements contained in this group affect all backends.
		 *
		 * @{
		 */

		/**
		 * \internal
		 * The default backend to be selected for an end user.
		 * \ingroup config
		 * \endinternal
		 */
		static constexpr grb::Backend default_backend = _GRB_BACKEND;

		/**
		 * Contains information about the target architecture cache line size.
		 *
		 * \ingroup config
		 */
		class CACHE_LINE_SIZE {

			private:

				/**
				 * The cache line size in bytes. Update this value at compile time to
				 * reflect the target architecture.
				 */
				static constexpr size_t bytes = 64;

			public:

				/**
				 * \internal
				 * @return The cache line size in bytes.
				 * @see grb::config::CACHE_LINE_SIZE::bytes
				 * \endinternal
				 */
				static constexpr size_t value() {
					return bytes;
				}

		};

		/**
		 * The SIMD size, in bytes.
		 *
		 * \ingroup config
		 */
		class SIMD_SIZE {

			private:

				/**
				 * The SIMD size, in bytes. Update this value at compile time to reflect
				 * the target architecture.
				 */
				static constexpr size_t bytes = 64;

			public:

				/**
				 * \internal
				 * @return The SIMD size in bytes.
				 * @see grb::config::SIMD_SIZE::bytes
				 * \endinternal
				 */
				static constexpr size_t value() {
					return bytes;
				}

		};

		/**
		 * \internal
		 * How many elements of a given data type fit into a SIMD register.
		 * \ingroup config
		 * \endinternal
		 */
		template< typename T >
		class SIMD_BLOCKSIZE {

			public:

				/**
				 * \internal
				 * Calculates the block size this operator should use.
				 *
				 * \warning This rounds down. If instances of T are too large, this could
				 *          result in a zero value. See #value for a correction.
				 * \endinternal
				 */
				static constexpr size_t unsafe_value() {
					return SIMD_SIZE::value() / sizeof( T );
				}

				/**
				 * \internal
				 * The maximum of one and the number of elements that fit into a single
				 * cache line.
				 * \endinternal
				 */
				static constexpr size_t value() {
					return unsafe_value() > 0 ? unsafe_value() : 1;
				}

		};

		/**
		 * \internal
		 * How many hardware threads the operating system exposes.
		 *
		 * \warning On contemporary x86-based hardware, the reported number by
		 *          value() will include that of each hyper-thread. This number
		 *          thus does not necessarily equal the number of cores available.
		 *
		 * \ingroup config
		 * \endinternal
		 */
		class HARDWARE_THREADS {

			public:

				/**
				 * \internal
				 * Returns the number of online hardware threads as reported by the
				 * operating system.
				 *
				 * \warning This is a UNIX system call.
				 *
				 * @returns The number of hardware threads currently online. The return
				 *          type is specified by the UNIX standard.
				 * \endinternal
				 */
				static long value() {
					return sysconf( _SC_NPROCESSORS_ONLN );
				}

		};

		/**
		 * Benchmarking default configuration parameters.
		 *
		 * \ingroup config
		 */
		class BENCHMARKING {

			public:

				/** @returns The default number of inner repetitions. */
				static constexpr size_t inner() {
					return 1;
				}

				/** @returns The default number of outer repetitions. */
				static constexpr size_t outer() {
					return 10;
				}

		};

		/**
		 * Memory configuration parameters.
		 *
		 * \ingroup config
		 */
		class MEMORY {

			public:

				/** @returns the private L1 data cache size, in bytes. */
				static constexpr size_t l1_cache_size() {
					return 32768;
				}

				/**
				 * @returns What is considered a lot of memory, in 2-log of bytes.
				 */
				static constexpr size_t big_memory() {
					return 31;
				} // 2GB

				/**
				 * \internal
				 * The memory speed under random accesses of 8-byte words.
				 *
				 * @returns The requested speed in MiB/s/process.
				 *
				 * @note The default value was measured on a two-socket Ivy Bridge node
				 *       with 128GB quad-channel DDR4 memory at 1600 MHz per socket.
				 *
				 * @note In the intended use of these variables, it is the ratio between
				 *       #stream_memspeed and #random_access_memspeed that matters. While
				 *       untested, it is reasonable to think the ratios do not change too
				 *       much between architectures. Nevertheless, for best results, these
				 *       numbers are best set to benchmarked values on the deployment
				 *       hardware.
				 *
				 * @note Preliminary experiments have not resulted in a decisive gain from
				 *       using this parameter, and hence it is currently not used by any
				 *       backend.
				 * \endinternal
				 */
				static constexpr double random_access_memspeed() {
					return 147.298;
				}

				/**
				 * \internal
				 * The memory speed under a limited number of streams of uncached data.
				 *
				 * @returns The requested speed in MiB/s/process.
				 *
				 * @note The default value was measured on a two-socket Ivy Bridge node
				 *       with 128GB quad-channel DDR4 memory at 1600 MHz per socket.
				 *
				 * @note In the intended use of these variables, it is the ratio between
				 *       #stream_memspeed and #random_access_memspeed that matters. While
				 *       untested, it is reasonable to think the ratios do not change too
				 *       much between architectures. Nevertheless, for best results, these
				 *       numbers are best set to benchmarked values on the deployment
				 *       hardware.
				 *
				 * @note Preliminary experiments have not resulted in a decisive gain from
				 *       using this parameter, and hence it is currently not used by any
				 *       backend.
				 * \endinternal
				 */
				static constexpr double stream_memspeed() {
					return 1931.264;
				}

				/**
				 * \internal
				 * Prints memory usage info to stdout, but only for big memory allocations.
				 *
				 * @returns true if and only if this function printed information to stdout.
				 * \endinternal
				 */
				static bool report(
					const std::string prefix, const std::string action,
					const size_t size, const bool printNewline = true
				) {
#ifdef _GRB_NO_STDIO
					(void) prefix;
					(void) action;
					(void) size;
					(void) printNewline;
					return false;
#else
					constexpr size_t big =
 #ifdef _DEBUG
						true;
 #else
						( 1ul << big_memory() );
 #endif
					if( size >= big ) {
						std::cout << "Info: ";
						std::cout << prefix << " ";
						std::cout << action << " ";
						if( sizeof( size_t ) * 8 > 40 && ( size >> 40 ) > 2 ) {
							std::cout << ( size >> 40 ) << " TB of memory";
						} else if( sizeof( size_t ) * 8 > 30 && ( size >> 30 ) > 2 ) {
							std::cout << ( size >> 30 ) << " GB of memory";
						} else if( sizeof( size_t ) * 8 > 20 && ( size >> 20 ) > 2 ) {
							std::cout << ( size >> 20 ) << " MB of memory";
						} else if( sizeof( size_t ) * 8 > 10 && ( size >> 10 ) > 2 ) {
							std::cout << ( size >> 10 ) << " kB of memory";
						} else {
							std::cout << size << " bytes of memory";
						}
						if( printNewline ) {
							std::cout << ".\n";
						}
						return true;
					}
					return false;
#endif
				}
		};

		/**
		 * Collects a series of implementation choices corresponding to some given
		 * \a backend.
		 *
		 * These implementation choices are useful for \em compositional backends;
		 * i.e., backends that rely on a nested sub-backend for functionality. To
		 * facilitate composability, backends are required to provide the functions
		 * specified herein.
		 *
		 * \note An example are the #grb::BSP1D and #grb::hybrid backends, that both
		 *       share the exact same code, relying on either the #grb::reference or
		 *       the #grb::reference_omp backend, respectively.
		 *
		 * \note The user documentation does not list all required fields; for a
		 *       complete overview, see the developer documentation instead.
		 *
		 * The default class declaration is declared empty to ensure no one backend
		 * implicitly relies on global defaults. Every backend therefore must
		 * specialise this class and implement the specified functions.
		 *
		 * \warning Portable ALP user code does not rely on the implementation details
		 *          gathered in this class.
		 *
		 * \note For properties of a backend that may (also) affect ALP user code,
		 *       see #grb::Properties.
		 *
		 * The user documentation only documents the settings that could be useful to
		 * modify.
		 *
		 * \warning Modifying the documented functions should be done with care.
		 *
		 * \warning Any such modifications typically requires rebuilding the ALP
		 *          library itself.
		 *
		 * \note For viewing all implementation choices, please see the developer
		 *       documentation.
		 *
		 * \ingroup config
		 */
		template< grb::Backend backend = default_backend >
		class IMPLEMENTATION {
#ifdef __DOXYGEN__
			public:

				/**
				 * Defines how private memory regions are allocated.
				 *
				 * @returns how a memory region that will not be accessed by threads other
				 * than the allocating thread, should be allocated.
				 */
				static constexpr ALLOC_MODE defaultAllocMode();

				/**
				 * Defines how shared memory regions are allocated.
				 *
				 * @returns how a memory region that may be accessed by thread other than
				 * the allocating thread, should be allocated.
				 */
				static constexpr ALLOC_MODE sharedAllocMode();

				/**
				 * \internal
				 * @returns whether the selected backend implements vectors as having fixed
				 * capacities. This is \em not a configuration choice for most backends,
				 * but rather a fixed consequence of design choices.
				 *
				 * \note The only legal fixed capacity a functional ALP/GraphBLAS backend
				 *       may provide is one that is equal to its size.
				 *
				 * \note A backend backed by a sparse accumulator (SPA) will typically have
				 *       fixed vector capacities, whereas one based on sets or other types
				 *       of tree structures will typically have dynamic vector capacities.
				 * \endinternal
				 */
				static constexpr bool fixedVectorCapacities();
#endif
		};

		/**
		 * What data type should be used to store row indices.
		 *
		 * Some uses cases may require this to be set to <tt>size_t</tt>-- others may
		 * do with (much) smaller data types instead.
		 *
		 * \note The data type for indices of general arrays is not configurable. This
		 *       set of implementations use <tt>size_t</tt> for those.
		 *
		 * \ingroup config
		 */
		typedef unsigned int RowIndexType;

		/**
		 * What data type should be used to store column indices.
		 *
		 * Some uses cases may require this to be set to <tt>size_t</tt>-- others may
		 * do with (much) smaller data types instead.
		 *
		 * \note The data type for indices of general arrays is not configurable. This
		 *       set of implementations use <tt>size_t</tt> for those.
		 *
		 * \ingroup config
		 */
		typedef unsigned int ColIndexType;

		/**
		 * What data type should be used to refer to an array containing nonzeroes.
		 *
		 * Some uses cases may require this to be set to <tt>size_t</tt>-- others may
		 * do with (much) smaller data types instead.
		 *
		 * \note The data type for indices of general arrays is not configurable. This
		 *       set of implementations use <tt>size_t</tt> for those.
		 *
		 * \ingroup config
		 */
		typedef size_t NonzeroIndexType;

		/**
		 * What data type should be used to store vector indices.
		 *
		 * Some uses cases may require this to be set to <tt>size_t</tt>-- others may
		 * do with (much) smaller data types instead.
		 *
		 * \note The data type for indices of general arrays is not configurable. This
		 *       set of implementations use <tt>size_t</tt> for those.
		 *
		 * \ingroup config
		 */
		typedef unsigned int VectorIndexType;

	} // namespace config

} // namespace grb

#endif // end _H_GRB_CONFIG_BASE

