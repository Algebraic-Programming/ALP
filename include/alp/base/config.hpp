
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
 * @date 8th of August, 2016
 */

#ifndef _H_ALP_CONFIG_BASE
#define _H_ALP_CONFIG_BASE

#include <cstddef> //size_t
#ifndef _ALP_NO_STDIO
 #include <iostream> //std::cout
#endif
#include <string>

#include <assert.h>
#include <unistd.h> //sysconf

#include <alp/backends.hpp>


// if the user did not define _ALP_BACKEND, set it to the default sequential
// implementation
#ifndef _ALP_BACKEND
 #define _ALP_BACKEND reference
#endif

// if the user did not define _ALP_SECONDARY_BACKEND, set it to the default
// sequential implementation. This setting may be used by other backends for
// backend-specific purposes. For example, a parallel backend may use this
// setting to constrol to which sequential backend it dispatches sequential
// work.
#ifndef _ALP_SECONDARY_BACKEND
 #define _ALP_SECONDARY_BACKEND reference
#endif

/**
 * The main GraphBLAS namespace.
 *
 * All GraphBLAS functions and objects are defined within.
 */
namespace alp {

	/** Contains compile-time configuration constants. */
	namespace config {

		/** The default backend to be selected for an end user. */
		static constexpr alp::Backend default_backend = _ALP_BACKEND;

		/** The cache line size, in bytes. */
		class CACHE_LINE_SIZE {

		private:
			/**
			 * The cache line size in bytes. Update this value at compile time to
			 * reflect the target architecture.
			 */
			static constexpr size_t bytes = 64;

		public:
			/**
			 * @return The cache line size in bytes.
			 * @see alp::config::CACHE_LINE_SIZE::bytes
			 */
			static constexpr size_t value() {
				return bytes;
			}
		};

		/** The SIMD size, in bytes. */
		class SIMD_SIZE {

		private:
			/**
			 * The SIMD size, in bytes. Update this value at compile time to reflect
			 * the target architecture.
			 */
			static constexpr size_t bytes = 32;

		public:
			/**
			 * @return The SIMD size in bytes.
			 * @see alp::config::SIMD_SIZE::bytes
			 */
			static constexpr size_t value() {
				return bytes;
			}
		};

		/** How many elements of a given data type fit into a SIMD register. */
		template< typename T >
		class SIMD_BLOCKSIZE {
		public:
			/**
			 * Calculates the block size this operator should use.
			 *
			 * \warning This rounds down. If instances of T are too large, this could
			 *          result in a zero value. See #value for a correction.
			 */
			static constexpr size_t unsafe_value() {
				return SIMD_SIZE::value() / sizeof( T );
			}

			/**
			 * The maximum of one and the number of elements that fit into a single
			 * cache line.
			 */
			static constexpr size_t value() {
				return unsafe_value() > 0 ? unsafe_value() : 1;
			}
		};

		/**
		 * How many hardware threads the operating system exposes.
		 *
		 * \warning On contemporary x86-based hardware, the reported number by
		 *          value() will include that of each hyper-thread. This number
		 *          thus does not necessarily equal the number of cores available.
		 */
		class HARDWARE_THREADS {
		public:
			/**
			 * Returns the number of online hardware threads as reported by the OS.
			 *
			 * \warning This is a UNIX system call.
			 *
			 * @returns The number of hardware threads currently online. The return
			 *          type is specified by the UNIX standard.
			 */
			static long value() {
				return sysconf( _SC_NPROCESSORS_ONLN );
			}
		};

		/** Benchmarking defaults. */
		class BENCHMARKING {
		public:
			/** The default number of inner repititions. */
			static constexpr size_t inner() {
				return 1;
			}

			/** The default number of outer repititions. */
			static constexpr size_t outer() {
				return 10;
			}
		};

		/** Memory defaults. */
		class MEMORY {
		public:

			/** The private L1 data cache size, in bytes. */
			static constexpr size_t l1_cache_size() {
				return 32768;
			}

			/** What is considered a lot of memory, in 2-log of bytes. */
			static constexpr size_t big_memory() {
				return 31;
			} // 2GB

			/**
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
			 */
			static constexpr double random_access_memspeed() {
				return 147.298;
			}

			/**
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
			 */
			static constexpr double stream_memspeed() {
				return 1931.264;
			}

			/**
			 * Prints memory usage info to stdout, but only for big memory allocations.
			 *
			 * @returns true if and only if this function printed information to stdout.
			 */
			static bool report( const std::string prefix, const std::string action, const size_t size, const bool printNewline = true ) {
#ifdef _ALP_NO_STDIO
				(void)prefix;
				(void)action;
				(void)size;
				(void)printNewline;
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
		 * What data type should be used to store row indices.
		 *
		 * Some uses cases may require this to be set to <tt>size_t</tt>-- others may
		 * do with (much) smaller data types instead.
		 *
		 * \note The data type for indices of general arrays is not configurable. This
		 *       set of implementations use <tt>size_t</tt> for those.
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
		 */
		typedef unsigned int VectorIndexType;

	} // namespace config

} // namespace alp

#endif // end _H_ALP_CONFIG_BASE
