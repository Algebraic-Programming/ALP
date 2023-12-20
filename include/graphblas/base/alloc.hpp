
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
 * @date 13th of September, 2017
 */

#ifndef _H_GRB_BASE_ALLOC
#define _H_GRB_BASE_ALLOC

#include <stdlib.h> //posix_memalign
#include <iostream> //std::c{err,out}

#include <utility> //std_forward

#include <assert.h>

#include <graphblas/backends.hpp>
#include <graphblas/rc.hpp>
#include <graphblas/utils/autodeleter.hpp>

#ifndef _GRB_NO_LIBNUMA
 #include <numa.h> //numa_alloc_interleaved
#endif

#ifndef _GRB_NO_STDIO
 #include <iostream> //std::err
#endif

namespace grb {

	namespace utils {

		namespace internal {

			template< enum Backend implementation >
			class AllocatorFunctions {

			private:
				/** Disable instantiation. */
				AllocatorFunctions() {}

			public:

				template< typename T, typename... Targs >
				static RC alloc( size_t &, T * __restrict__ &, const size_t, const bool, utils::AutoDeleter< T > &, Targs &&... ) {
#ifndef _GRB_NO_STDIO
					std::cerr << "Error: backend " << implementation << " did not define an allocation mechanism!" << std::endl;
#endif
					return PANIC;
				}

				static void postAlloc( const RC, const size_t, const std::string, const std::string ) {
#ifndef _GRB_NO_STDIO
					std::cerr << "Error: backend " << implementation << " did not define an allocation mechanism!" << std::endl;
#endif
				}
			};

			template< enum Backend implementation >
			class Allocator {
			private:
				/** Disable instantiation. */
				Allocator() {}

			public:
				/** Forward to the given backend's allocation functions. */
				typedef internal::AllocatorFunctions< implementation > functions;
			};

		} // namespace internal

	}     // namespace utils

} // namespace grb

// define user API:
namespace grb {

	namespace utils {

		class Allocator {
			size_t allocated_memory;
			RC failed_ret;
			unsigned successful_allocations;

			public:

				typedef Allocator SelfType;

				Allocator() : allocated_memory( 0 ), failed_ret( SUCCESS ), successful_allocations( 0 ) {}

				Allocator( const SelfType & ) = delete;

				SelfType & operator=( const SelfType & ) = delete;

				size_t getAllocatedBytes() const noexcept { return allocated_memory; }

				template< typename T, enum Backend implementation >
				SelfType& alloc(
						const size_t size, const bool shared,
						utils::AutoDeleter< T, implementation > &deleter
				) {
					if( failed_ret != SUCCESS ) {
#ifdef _DEBUG
						std::cerr << "allocator is deactivated\n";
#endif
						return *this;
					}
					const RC ret = internal::Allocator< implementation >::functions::alloc(
						allocated_memory, size, shared, deleter );
					if( ret == SUCCESS ) {
							successful_allocations++;
					} else {
						failed_ret = ret;
#ifdef _DEBUG
						std::cerr << "allocation nr. " << successful_allocations << " is unsuccesful, dactivating allocator\n";
#endif
					}
					return *this;
				}

				bool isSuccessful() const noexcept {
					return failed_ret == SUCCESS;
				}

				RC getLastAllocationResult() const noexcept {
					return failed_ret;
				}

				bool printReport( const char* prefix, const char* postfix ) const {
					const bool success = isSuccessful();
					const char *event = success ? "allocated" : "failed to allocate";
					config::MEMORY::report( prefix, event, getAllocatedBytes(), false );
					if( postfix != nullptr ) {
						std::cout << ", reason: " << postfix << ".\n";
					}
					if( !success ) {
						std::cout << "Allocation nr. " << successful_allocations << " failed" << std::endl;
					}
					return success;
				}
		};

	} // namespace utils

} // namespace grb

#endif // _H_GRB_BASE_ALLOC

